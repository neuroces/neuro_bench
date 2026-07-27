"""Ground-truth resolvers for NeuroBench questions.

A *resolver* recomputes a question's answer directly from the full NWB file —
including the label tables (``intervals``) that are **withheld** from the model
at run time. This is what makes every authored answer provably derivable from an
objective property of the data (BioMysteryBench principle #1), and lets us
regenerate ``answers.json`` deterministically rather than hand-typing answers.

Resolvers are pure functions ``(nwb, **params) -> answer`` registered in
``RESOLVERS``. ``resolve(nwb, ground_truth)`` dispatches by method name.

Heavy numerical imports are done lazily so this module imports offline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

# Canonical EEG frequency bands (Hz). Used by spectral resolvers so that
# "dominant band" has a single, reproducible definition.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

_EEG_SERIES = "ElectricalSeriesEEG"


# --- signal helpers ----------------------------------------------------------


def _eeg_series(nwb):
    if _EEG_SERIES not in nwb.acquisition:
        raise KeyError(f"{_EEG_SERIES!r} not present (this recording has no EEG series)")
    return nwb.acquisition[_EEG_SERIES]


def eeg_sampling_rate(nwb) -> float:
    """Effective sampling rate (Hz), derived from timestamps if ``rate`` is unset."""
    es = _eeg_series(nwb)
    if getattr(es, "rate", None):
        return float(es.rate)
    n = es.data.shape[0]
    t0 = float(es.timestamps[0])
    t1 = float(es.timestamps[-1])
    return (n - 1) / (t1 - t0)


def eeg_window(nwb, start_s: float, duration_s: float, channel: int = 0):
    """Stream a single-channel EEG window [start_s, start_s+duration_s)."""
    import numpy as np

    es = _eeg_series(nwb)
    fs = eeg_sampling_rate(nwb)
    t0 = float(es.timestamps[0]) if getattr(es, "rate", None) is None else 0.0
    i0 = int(round((start_s - t0) * fs))
    i1 = int(round((start_s + duration_s - t0) * fs))
    return np.asarray(es.data[i0:i1, channel], dtype=float), fs


def _band_powers(nwb, start_s: float, duration_s: float, channel: int) -> dict[str, float]:
    import numpy as np
    from scipy import signal

    seg, fs = eeg_window(nwb, start_s, duration_s, channel)
    freqs, psd = signal.welch(seg, fs=fs, nperseg=int(fs * 2))
    powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        powers[name] = float(np.trapezoid(psd[mask], freqs[mask]))
    return powers


# --- resolvers ---------------------------------------------------------------


def _epochs_df(nwb):
    if not nwb.intervals or "epochs" not in nwb.intervals:
        raise KeyError("NWB file has no 'epochs' interval table")
    df = nwb.intervals["epochs"].to_dataframe()
    df = df.copy()
    df["tags"] = df["tags"].apply(list)
    return df


def state_at_window(nwb, start_s: float, duration_s: float, **_: Any) -> str:
    """Return 'awake' or 'anesthetized' for the given window.

    The window must lie cleanly inside an ``isoflurane_anesthesia`` epoch
    (anesthetized) or entirely before the earliest ``isoflurane_induction``
    onset (awake); otherwise the window is ambiguous and rejected.
    """
    df = _epochs_df(nwb)
    w0, w1 = start_s, start_s + duration_s

    for _, row in df.iterrows():
        if "isoflurane_anesthesia" in row["tags"]:
            if row["start_time"] <= w0 and w1 <= row["stop_time"]:
                return "anesthetized"

    induction = df[df["tags"].apply(lambda t: "isoflurane_induction" in t)]
    if len(induction):
        first_onset = float(induction["start_time"].min())
        if w1 <= first_onset:
            return "awake"

    raise ValueError(
        f"Window [{w0}, {w1}] is not cleanly awake or anesthetized; pick another window."
    )


def induction_onset_time(nwb, **_: Any) -> float:
    """Recording time (s) at which isoflurane induction begins."""
    df = _epochs_df(nwb)
    induction = df[df["tags"].apply(lambda t: "isoflurane_induction" in t)]
    if not len(induction):
        raise ValueError("No isoflurane_induction epoch found")
    return round(float(induction["start_time"].min()), 2)


def anesthesia_onset_time(nwb, **_: Any) -> float:
    """Recording time (s) at which stable isoflurane anesthesia begins.

    Unlike ``induction_onset_time`` (the moment gas is turned on), this is the
    start of the ``isoflurane_anesthesia`` epoch — the point the EEG has settled
    into its anesthetized signature, which is what a spectral detector can find.
    """
    df = _epochs_df(nwb)
    anesthesia = df[df["tags"].apply(lambda t: "isoflurane_anesthesia" in t)]
    if not len(anesthesia):
        raise ValueError("No isoflurane_anesthesia epoch found")
    return round(float(anesthesia["start_time"].min()), 2)


def eeg_channel_count(nwb, **_: Any) -> int:
    """Number of channels in the EEG electrical series."""
    return int(_eeg_series(nwb).data.shape[1])


def dominant_band(nwb, start_s: float, duration_s: float, channel: int = 0, **_: Any) -> str:
    """Name of the highest-power canonical band in the given EEG window."""
    powers = _band_powers(nwb, start_s, duration_s, channel)
    return max(powers, key=powers.get)


# --- intracellular current-clamp electrophysiology (Category 2) --------------

_CC_RESP_PREFIX = "CurrentClampSeries"
_CC_STIM_PREFIX = "CurrentClampStimulusSeries"
_SPIKE_THRESHOLD_MV = -10.0

# Cre driver markers that identify GABAergic (inhibitory) interneurons.
_INHIBITORY_MARKERS = {
    "Sst",
    "Pvalb",
    "Vip",
    "Htr3a",
    "Ndnf",
    "Lamp5",
    "Chodl",
    "Gad2",
    "Chat",
    "Chrna2",
}


def to_millivolts(data, conversion: float):
    """Normalize a current-clamp response to mV (handles volt- or mV-scaled data)."""
    import numpy as np

    v = np.asarray(data, dtype=float) * conversion
    if np.nanmedian(np.abs(v)) < 1.0:  # stored in volts
        v = v * 1e3
    return v


def to_picoamps(data, conversion: float):
    """Normalize a stimulus current to pA (handles ampere- or pA-scaled data)."""
    import numpy as np

    s = np.asarray(data, dtype=float) * conversion
    if np.nanmax(np.abs(s)) < 1e-3:  # stored in amperes
        s = s * 1e12
    return s


def count_spikes(v_mv) -> int:
    """Count upward crossings of the spike threshold in a mV trace."""
    import numpy as np

    above = np.asarray(v_mv) > _SPIKE_THRESHOLD_MV
    return int(np.sum((~above[:-1]) & (above[1:])))


def icephys_sweeps(nwb) -> list[dict[str, Any]]:
    """Per-sweep injected current (pA), spike count, and pulse duration (s)."""
    import numpy as np

    resp = sorted(k for k in nwb.acquisition if k.startswith(_CC_RESP_PREFIX))
    stim = sorted(k for k in nwb.stimulus if k.startswith(_CC_STIM_PREFIX))
    sweeps = []
    for i, (rk, sk) in enumerate(zip(resp, stim)):
        s_series = nwb.stimulus[sk]
        s = to_picoamps(s_series.data[:], s_series.conversion)
        base = float(np.median(s[: max(1, len(s) // 10)]))
        inj = float(np.median(s[len(s) // 3 : 2 * len(s) // 3]) - base)
        r_series = nwb.acquisition[rk]
        v = to_millivolts(r_series.data[:], r_series.conversion)
        rate = float(getattr(s_series, "rate", 0) or getattr(r_series, "rate", 0) or 0)
        dev = np.abs(s - base) > 5.0
        dur = float(np.sum(dev) / rate) if rate else float("nan")
        sweeps.append(
            {
                "index": i,
                "injected_pA": round(inj, 1),
                "n_spikes": count_spikes(v),
                "pulse_s": round(dur, 4),
            }
        )
    return sweeps


def _cre_marker(nwb) -> str:
    genotype = getattr(getattr(nwb, "subject", None), "genotype", None)
    if not genotype:
        raise ValueError("no subject genotype available")
    return genotype.split("-")[0].split(";")[0].strip()


def cell_subtype(nwb, **_: Any) -> str:
    """Interneuron subtype from the Cre driver line (e.g. 'Sst', 'Pvalb')."""
    return _cre_marker(nwb)


def cell_class(nwb, **_: Any) -> str:
    """'inhibitory' or 'excitatory' from the Cre driver line."""
    return "inhibitory" if _cre_marker(nwb) in _INHIBITORY_MARKERS else "excitatory"


def rheobase(nwb, **_: Any) -> float:
    """Minimum positive current step (pA) that elicits at least one spike."""
    spiking = [
        s["injected_pA"] for s in icephys_sweeps(nwb) if s["injected_pA"] > 0 and s["n_spikes"] > 0
    ]
    if not spiking:
        raise ValueError("no spiking sweep found")
    return round(min(spiking), 1)


def firing_rate_at_current(nwb, current_pA: float, tol_pA: float = 15.0, **_: Any) -> float:
    """Firing rate (Hz) in the sweep whose injected current is closest to current_pA."""
    candidates = [s for s in icephys_sweeps(nwb) if abs(s["injected_pA"] - current_pA) <= tol_pA]
    if not candidates:
        raise ValueError(f"no sweep within {tol_pA} pA of {current_pA} pA")
    sweep = min(candidates, key=lambda s: abs(s["injected_pA"] - current_pA))
    if not sweep["pulse_s"]:
        raise ValueError("cannot determine pulse duration")
    return round(sweep["n_spikes"] / sweep["pulse_s"], 2)


# --- extracellular Neuropixels spike sorting (Category 1) --------------------

# Mean-waveform sampling rate for DANDI 000409 (IBL Brain Wide Map): the
# spike-sorted templates are stored at the Neuropixels AP-band rate.
WAVEFORM_FS_HZ = 30000.0

# Non-neural electrode labels (fiber tracts, ventricles, out-of-brain). A unit
# whose peak channel sits here is not a valid brain-region question.
_NON_BRAIN_LOCATIONS = {"void", "root"}


def coarse_brain_region(location: str) -> str | None:
    """Normalize a fine Allen ``electrodes.location`` label to a gross region.

    Waveform shape and firing statistics carry gross-region signatures (cortex vs
    hippocampus vs thalamus vs striatum vs midbrain) but not fine Allen subnuclei,
    so ground truth is defined at the gross level. Returns ``None`` for labels
    outside the target set (fiber tracts, ventricles, olfactory/amygdala, etc.) so
    the resolver can reject them rather than author an unanswerable question.
    """
    l = location.lower().strip()
    if l.startswith("field ca") or "dentate gyrus" in l or l == "subiculum" or "fasciola" in l:
        return "Hippocampus"
    if "thalamus" in l or "geniculate" in l:
        return "Thalamus"
    if l == "caudoputamen":
        return "Striatum"
    if (
        "midbrain" in l
        or "substantia nigra" in l
        or "ventral tegmental" in l
        or "colliculus" in l
        or "pretectal" in l
        or "periaqueduct" in l
    ):
        return "Midbrain"
    if "area" in l and "layer" in l and "entorhinal" not in l:
        return "Cortex"
    return None


def _units_table(nwb):
    units = getattr(nwb, "units", None)
    if units is None:
        raise KeyError("NWB file has no units table")
    return units


def _unit_row(units, unit_id: int) -> int:
    """Row index of a unit given its units-table id (ids are not 0-based)."""
    ids = [int(i) for i in units.id[:]]
    try:
        return ids.index(int(unit_id))
    except ValueError as exc:
        raise KeyError(f"unit_id {unit_id} not in units table") from exc


def unit_peak_electrode(nwb, unit_id: int) -> int:
    """Electrode id of a unit's peak (max-amplitude) channel."""
    units = _units_table(nwb)
    if "max_electrode" not in units.colnames:
        raise KeyError("units table has no 'max_electrode' column")
    row = _unit_row(units, unit_id)
    return int(units["max_electrode"].data[row])


def _electrode_location(nwb, electrode_id: int) -> str:
    el = getattr(nwb, "electrodes", None)
    if el is None:
        raise KeyError("NWB file has no electrodes table")
    if "location" not in el.colnames:
        raise KeyError("electrodes table has no 'location' column")
    ids = [int(i) for i in el.id[:]]
    try:
        row = ids.index(int(electrode_id))
    except ValueError as exc:
        raise KeyError(f"electrode {electrode_id} not in electrodes table") from exc
    return str(el["location"].data[row])


def unit_spike_times(nwb, unit_id: int):
    """Sorted spike times (s) for one unit."""
    import numpy as np

    units = _units_table(nwb)
    if "spike_times" not in units.colnames:
        raise KeyError("units table has no 'spike_times' column")
    row = _unit_row(units, unit_id)
    return np.sort(np.asarray(units["spike_times"][row], dtype=float))


def unit_waveform_mean(nwb, unit_id: int):
    """Mean waveform for one unit (samples x channels, may contain NaN padding)."""
    import numpy as np

    units = _units_table(nwb)
    if "waveform_mean" not in units.colnames:
        raise KeyError("units table has no 'waveform_mean' column")
    row = _unit_row(units, unit_id)
    return np.asarray(units["waveform_mean"][row], dtype=float)


def peak_channel_waveform(waveform):
    """Reduce a (samples x channels) mean waveform to its peak-channel trace.

    The peak channel is the one with the largest peak-to-peak amplitude; NaN
    padding (channels/samples not stored) is ignored.
    """
    import numpy as np

    w = np.asarray(waveform, dtype=float)
    if w.ndim == 1:
        trace = w
    else:
        valid = ~np.all(np.isnan(w), axis=0)  # drop all-NaN (unstored) channels
        p2p = np.full(w.shape[1], -np.inf)
        p2p[valid] = np.nanmax(w[:, valid], axis=0) - np.nanmin(w[:, valid], axis=0)
        pc = int(np.argmax(p2p))
        trace = w[:, pc]
    return trace[~np.isnan(trace)]


def waveform_peak_trough_width_ms(waveform, fs: float = WAVEFORM_FS_HZ) -> float:
    """Trough-to-peak duration (ms) of a unit's peak-channel mean waveform.

    A classic extracellular feature: narrow (<~0.4 ms) marks fast-spiking
    (putative inhibitory) units, broad (>~0.5 ms) marks regular-spiking units.
    """
    import numpy as np

    trace = peak_channel_waveform(waveform)
    if trace.size < 3:
        raise ValueError("waveform too short to measure peak-trough width")
    trough = int(np.argmin(trace))
    after = trace[trough:]
    peak = trough + int(np.argmax(after))
    return round((peak - trough) / fs * 1000.0, 4)


def extracellular_firing_rate(
    nwb, unit_id: int, start_s: float | None = None, stop_s: float | None = None
) -> float:
    """Mean firing rate (Hz) of a unit over [start_s, stop_s] (default: full span)."""
    import numpy as np

    st = unit_spike_times(nwb, unit_id)
    if st.size == 0:
        return 0.0
    lo = float(st[0]) if start_s is None else float(start_s)
    hi = float(st[-1]) if stop_s is None else float(stop_s)
    if hi <= lo:
        raise ValueError("stop_s must be greater than start_s")
    n = int(np.sum((st >= lo) & (st <= hi)))
    return round(n / (hi - lo), 3)


def unit_isi_stats(nwb, unit_id: int) -> dict[str, Any]:
    """Inter-spike-interval statistics (mean ISI, CV) for one unit."""
    import numpy as np

    st = unit_spike_times(nwb, unit_id)
    isi = np.diff(st)
    return {
        "n_spikes": int(st.size),
        "mean_isi_s": float(isi.mean()) if isi.size else None,
        "cv_isi": float(isi.std() / isi.mean()) if isi.size and isi.mean() else None,
    }


def unit_brain_region(nwb, unit_id: int, **_: Any) -> str:
    """Gross brain region of a unit, from its peak channel's ``electrodes.location``.

    Ground truth for Category 1: resolved from the (withheld) electrodes table so
    the model must infer it from waveform shape + firing statistics instead.
    Raises ``KeyError`` if the required tables/columns are missing and
    ``ValueError`` if the peak channel is not in a target gross region.
    """
    peak = unit_peak_electrode(nwb, unit_id)
    location = _electrode_location(nwb, peak)
    region = coarse_brain_region(location)
    if region is None:
        raise ValueError(
            f"unit {unit_id} peak channel is in {location!r}, not a target brain region"
        )
    return region


RESOLVERS: dict[str, Callable[..., Any]] = {
    "state_at_window": state_at_window,
    "induction_onset_time": induction_onset_time,
    "anesthesia_onset_time": anesthesia_onset_time,
    "eeg_channel_count": eeg_channel_count,
    "dominant_band": dominant_band,
    "cell_subtype": cell_subtype,
    "cell_class": cell_class,
    "rheobase": rheobase,
    "firing_rate_at_current": firing_rate_at_current,
    "unit_brain_region": unit_brain_region,
}


def resolve(nwb, ground_truth: dict[str, Any]) -> Any:
    """Dispatch a ``ground_truth`` spec ({'method':..., 'params':{...}}) to its resolver."""
    method = ground_truth["method"]
    if method not in RESOLVERS:
        raise KeyError(f"Unknown ground-truth method {method!r}; known: {sorted(RESOLVERS)}")
    params = ground_truth.get("params", {})
    return RESOLVERS[method](nwb, **params)


# --- answer hashing ----------------------------------------------------------


def answer_hash(answer: Any) -> str:
    """Stable SHA-256 of a canonical JSON encoding of an answer (for integrity)."""
    payload = json.dumps(answer, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

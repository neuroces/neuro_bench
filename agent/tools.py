"""Agent tool set for NeuroBench.

A :class:`ToolSession` is bound to a single question's asset and its
``redaction`` list. Every tool call goes through the session, which guarantees
the model can never read the withheld label tables (e.g. ``intervals``) — it can
only see signals and derived features, plus a sandboxed ``run_python`` escape
hatch. This enforces, at run time, the non-circularity that the ground-truth
resolvers rely on at authoring time.

The named tools mirror the plan's canonical tool set; ``run_python`` preserves
method-agnosticism for anything the named tools don't cover.
"""

from __future__ import annotations

from typing import Any

from agent.sandbox import run_sandboxed
from benchmark import ground_truth

# What the run_python sandbox exposes a full session at, to bound payload size.
_RUN_PYTHON_DECIMATED_FS = 250.0


class RedactionError(RuntimeError):
    """Raised when a tool would expose a redacted (withheld) part of the file."""


class ToolSession:
    """Controlled access to one recording for one question."""

    def __init__(
        self,
        dandi_id: str | None = None,
        version: str | None = None,
        asset_path: str | None = None,
        redaction: tuple[str, ...] | list[str] = (),
        *,
        nwb: Any = None,
        channel_default: int = 0,
        window: dict[str, float] | None = None,
    ):
        self.dandi_id = dandi_id
        self.version = version
        self.asset_path = asset_path
        self.redaction = set(redaction)
        self.channel_default = channel_default
        self.window = (
            window  # optional {"start_s","duration_s"} for run_python exposure
        )
        self._nwb = nwb
        self._handle = None

    # --- lifecycle -----------------------------------------------------------

    @classmethod
    def from_question(
        cls, question: dict[str, Any], *, nwb: Any = None
    ) -> "ToolSession":
        src = question["source"]
        return cls(
            dandi_id=src["dandi_id"],
            version=src["version"],
            asset_path=src["asset_path"],
            redaction=question.get("redaction", []),
            channel_default=src.get("channel") or 0,
            window=src.get("window"),
            nwb=nwb,
        )

    def nwb(self):
        if self._nwb is None:
            from data.nwb_access import load_nwb

            self._handle = load_nwb(self.dandi_id, self.version, self.asset_path)
            self._nwb = self._handle.nwb
        return self._nwb

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "ToolSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _guard(self, part: str) -> None:
        if part in self.redaction:
            raise RedactionError(
                f"Access to '{part}' is withheld for this question; "
                "infer the answer from the signal instead."
            )

    # --- named tools ---------------------------------------------------------

    def describe_recording(self) -> dict[str, Any]:
        """Metadata summary of the recording (redacted parts omitted)."""
        nwb = self.nwb()
        acquisition = {}
        for name, series in getattr(nwb, "acquisition", {}).items():
            shape = tuple(getattr(getattr(series, "data", None), "shape", ()))
            acquisition[name] = {"shape": shape}

        info: dict[str, Any] = {
            "identifier": getattr(nwb, "identifier", None),
            "session_description": getattr(nwb, "session_description", None),
            "acquisition": acquisition,
        }
        subject = getattr(nwb, "subject", None)
        if subject is not None and "subject" not in self.redaction:
            info["subject"] = {
                "subject_id": getattr(subject, "subject_id", None),
                "species": getattr(subject, "species", None),
                "sex": getattr(subject, "sex", None),
            }
        try:
            info["eeg_sampling_rate_hz"] = round(ground_truth.eeg_sampling_rate(nwb), 3)
        except Exception:  # noqa: BLE001 - metadata is best-effort
            info["eeg_sampling_rate_hz"] = None

        if "intervals" not in self.redaction:
            info["intervals"] = list(getattr(nwb, "intervals", {}) or {})
        if "units" not in self.redaction:
            units = getattr(nwb, "units", None)
            info["n_units"] = 0 if units is None else len(units)
        return info

    def get_eeg_epoch(
        self,
        start_s: float,
        duration_s: float,
        channel: int | None = None,
        max_points: int = 2000,
    ) -> dict[str, Any]:
        """Return an EEG epoch, decimated to at most ``max_points`` samples."""
        channel = self.channel_default if channel is None else channel
        seg, fs = ground_truth.eeg_window(self.nwb(), start_s, duration_s, channel)
        step = max(1, len(seg) // max_points)
        preview = seg[::step]
        return {
            "channel": channel,
            "fs_hz": round(float(fs), 3),
            "n_samples": int(len(seg)),
            "decimation": step,
            "samples": [round(float(x), 4) for x in preview],
        }

    def compute_psd(
        self,
        start_s: float,
        duration_s: float,
        channel: int | None = None,
        fmax: float = 40.0,
    ) -> dict[str, Any]:
        """Welch PSD of an EEG epoch, with band powers and peak frequency."""
        import numpy as np
        from scipy import signal

        channel = self.channel_default if channel is None else channel
        seg, fs = ground_truth.eeg_window(self.nwb(), start_s, duration_s, channel)
        freqs, psd = signal.welch(seg, fs=fs, nperseg=int(fs * 2))
        mask = freqs <= fmax
        freqs, psd = freqs[mask], psd[mask]

        band_powers = {}
        for name, (lo, hi) in ground_truth.BANDS.items():
            m = (freqs >= lo) & (freqs < hi)
            band_powers[name] = float(np.trapezoid(psd[m], freqs[m]))
        dominant = max(band_powers, key=band_powers.get)
        peak = float(freqs[int(np.argmax(psd))])
        return {
            "channel": channel,
            "fs_hz": round(float(fs), 3),
            "freqs_hz": [round(float(f), 3) for f in freqs],
            "psd": [float(p) for p in psd],
            "band_powers": {k: round(v, 6) for k, v in band_powers.items()},
            "dominant_band": dominant,
            "peak_frequency_hz": round(peak, 3),
        }

    def list_units(self) -> dict[str, Any]:
        self._guard("units")
        units = getattr(self.nwb(), "units", None)
        if units is None:
            return {"n_units": 0, "unit_ids": []}
        return {"n_units": len(units), "unit_ids": [int(i) for i in units.id[:]]}

    def get_spike_waveforms(self, unit_ids: list[int]) -> dict[str, Any]:
        self._guard("units")
        units = getattr(self.nwb(), "units", None)
        if units is None:
            raise ValueError("No units table in this recording.")
        df = units.to_dataframe()
        out = {}
        for uid in unit_ids:
            row = df.loc[uid]
            wf = row.get("waveform_mean")
            out[str(uid)] = None if wf is None else [float(x) for x in wf]
        return {"waveforms": out}

    def compute_isi(self, unit_id: int) -> dict[str, Any]:
        import numpy as np

        self._guard("units")
        units = getattr(self.nwb(), "units", None)
        if units is None:
            raise ValueError("No units table in this recording.")
        spikes = np.asarray(
            units.to_dataframe().loc[unit_id]["spike_times"], dtype=float
        )
        isi = np.diff(np.sort(spikes))
        return {
            "unit_id": unit_id,
            "n_spikes": int(spikes.size),
            "mean_isi_s": float(isi.mean()) if isi.size else None,
            "cv_isi": (
                float(isi.std() / isi.mean()) if isi.size and isi.mean() else None
            ),
        }

    def compute_cross_correlogram(
        self, unit_a: int, unit_b: int, bin_s: float = 0.001, window_s: float = 0.05
    ) -> dict[str, Any]:
        import numpy as np

        self._guard("units")
        units = getattr(self.nwb(), "units", None)
        if units is None:
            raise ValueError("No units table in this recording.")
        df = units.to_dataframe()
        a = np.sort(np.asarray(df.loc[unit_a]["spike_times"], dtype=float))
        b = np.sort(np.asarray(df.loc[unit_b]["spike_times"], dtype=float))
        edges = np.arange(-window_s, window_s + bin_s, bin_s)
        counts = np.zeros(len(edges) - 1, dtype=int)
        for t in a:
            d = b - t
            d = d[(d >= -window_s) & (d <= window_s)]
            counts += np.histogram(d, bins=edges)[0]
        centers = (edges[:-1] + edges[1:]) / 2
        return {
            "bin_s": bin_s,
            "lags_s": [round(float(c), 4) for c in centers],
            "counts": [int(c) for c in counts],
        }

    def run_python(self, code: str, timeout_s: int = 20) -> dict[str, Any]:
        """Execute analysis code in a sandbox with EEG data exposed (labels withheld)."""
        return run_sandboxed(code, self._exposure(), timeout_s=timeout_s)

    # --- intracellular current-clamp tools (Category 2) ----------------------

    def list_current_clamp_sweeps(self) -> dict[str, Any]:
        """Per-sweep injected current (pA) and spike count — the F-I curve."""
        sweeps = ground_truth.icephys_sweeps(self.nwb())
        return {"n_sweeps": len(sweeps), "sweeps": sweeps}

    def get_current_clamp_sweep(self, index: int, max_points: int = 2000) -> dict[str, Any]:
        """Return one current-clamp sweep: injected current, spike count, and voltage trace."""
        import numpy as np

        nwb = self.nwb()
        sweeps = ground_truth.icephys_sweeps(nwb)
        if index < 0 or index >= len(sweeps):
            raise IndexError(f"sweep {index} out of range (0..{len(sweeps) - 1})")
        resp = sorted(k for k in nwb.acquisition if k.startswith("CurrentClampSeries"))
        series = nwb.acquisition[resp[index]]
        v = ground_truth.to_millivolts(series.data[:], series.conversion)
        step = max(1, len(v) // max_points)
        meta = sweeps[index]
        return {
            "index": index,
            "injected_pA": meta["injected_pA"],
            "n_spikes": meta["n_spikes"],
            "fs_hz": float(getattr(series, "rate", 0) or 0),
            "decimation": step,
            "voltage_mV": [round(float(x), 3) for x in v[::step]],
        }

    # --- helpers -------------------------------------------------------------

    def _exposure(self) -> dict[str, Any]:
        """Signal data made available to run_python — never label tables."""
        import numpy as np

        nwb = self.nwb()
        fs = ground_truth.eeg_sampling_rate(nwb)
        channel = self.channel_default
        if self.window:
            seg, fs = ground_truth.eeg_window(
                nwb, self.window["start_s"], self.window["duration_s"], channel
            )
            return {"fs": float(fs), "eeg": np.asarray(seg, dtype=float)}
        # No window: expose the full channel, decimated to bound payload size.
        es = nwb.acquisition["ElectricalSeriesEEG"]
        step = max(1, int(round(fs / _RUN_PYTHON_DECIMATED_FS)))
        seg = np.asarray(es.data[::step, channel], dtype=float)
        return {"fs": float(fs / step), "eeg": seg}

    # --- dispatch ------------------------------------------------------------

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Route a tool call (name + JSON args) to its method, catching errors."""
        method = getattr(self, tool_name, None)
        if method is None or tool_name not in TOOL_NAMES:
            return {"error": f"unknown tool {tool_name!r}"}
        try:
            return method(**arguments)
        except RedactionError as exc:
            return {"error": str(exc), "redacted": True}
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return {"error": f"{type(exc).__name__}: {exc}"}


TOOL_NAMES = (
    "describe_recording",
    "get_eeg_epoch",
    "compute_psd",
    "list_units",
    "get_spike_waveforms",
    "compute_isi",
    "compute_cross_correlogram",
    "list_current_clamp_sweeps",
    "get_current_clamp_sweep",
    "run_python",
)

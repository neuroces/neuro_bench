"""Offline tests for Category-1 (Neuropixels extracellular) resolvers and tools.

Uses small stub tables that mimic the pynwb ``DynamicTable`` access pattern the
ground-truth helpers rely on (``table.id[:]``, ``table.colnames``,
``table[col].data[row]`` for scalar columns, ``table[col][row]`` for ragged
columns) so the whole suite runs without streaming from DANDI.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent.tools import ToolSession
from benchmark import ground_truth as gt


# --- stub NWB tables ---------------------------------------------------------


class _Col:
    """A column exposing both ``.data[row]`` and ``col[row]`` access."""

    def __init__(self, values):
        self.data = values

    def __getitem__(self, row):
        return self.data[row]


class _Table:
    def __init__(self, ids, columns):
        self._ids = list(ids)
        self._columns = {name: _Col(vals) for name, vals in columns.items()}
        self.colnames = list(columns)

    @property
    def id(self):
        return self._ids  # list supports [:] slicing

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, name):
        return self._columns[name]


def _spike_waveform(
    n_samples=60,
    n_channels=4,
    peak_ch=2,
    trough=20,
    width=15,
    amp=-100.0,
    rebound=40.0,
    nan_channel=None,
):
    """A synthetic mean waveform (samples x channels) with a clear trough->peak."""
    w = np.random.RandomState(0).normal(0, 1.0, size=(n_samples, n_channels))
    w[:, peak_ch] = 0.0
    w[trough, peak_ch] = amp
    w[trough + width, peak_ch] = rebound
    if nan_channel is not None:
        w[:, nan_channel] = np.nan
    return w


def make_units_nwb():
    """Three units: 10=Hippocampus (CA1), 20=Cortex (S1 L5), 30=void (non-brain)."""
    ids = [10, 20, 30]
    # regular ~10 Hz train for unit 10; bursty-ish for others (only rate/ISI tested)
    st10 = np.arange(0.0, 10.0, 0.1)  # 100 spikes over 9.9 s
    st20 = np.arange(0.0, 20.0, 0.5)  # 40 spikes over 19.5 s
    st30 = np.arange(0.0, 5.0, 0.25)  # 20 spikes over 4.75 s
    waveforms = [
        _spike_waveform(peak_ch=2, trough=20, width=15, nan_channel=0),  # 0.5 ms
        _spike_waveform(peak_ch=1, trough=18, width=30),  # 1.0 ms
        _spike_waveform(peak_ch=0, trough=10, width=6),  # 0.2 ms
    ]
    units = _Table(
        ids,
        {
            "max_electrode": [5, 8, 2],
            "spike_times": [st10, st20, st30],
            "waveform_mean": waveforms,
        },
    )
    electrodes = _Table(
        list(range(10)),
        {
            "location": [
                "root",
                "root",
                "void",
                "root",
                "root",
                "Field CA1",
                "root",
                "root",
                "Primary somatosensory area trunk layer 5",
                "root",
            ],
        },
    )
    return SimpleNamespace(
        units=units,
        electrodes=electrodes,
        intervals={},
        subject=SimpleNamespace(subject_id="m", species="Mus musculus", sex="M"),
        acquisition={},
        identifier="npx-stub",
        session_description="Neuropixels extracellular stub",
    )


# --- coarse region normalization ---------------------------------------------


def test_coarse_brain_region_mapping():
    assert gt.coarse_brain_region("Field CA1") == "Hippocampus"
    assert gt.coarse_brain_region("Dentate gyrus molecular layer") == "Hippocampus"
    assert gt.coarse_brain_region("Posterior complex of the thalamus") == "Thalamus"
    assert gt.coarse_brain_region("Caudoputamen") == "Striatum"
    assert gt.coarse_brain_region("Substantia nigra reticular part") == "Midbrain"
    assert (
        gt.coarse_brain_region("Superior colliculus motor related intermediate gray layer")
        == "Midbrain"
    )
    assert gt.coarse_brain_region("Primary somatosensory area trunk layer 5") == "Cortex"
    assert gt.coarse_brain_region("Secondary motor area layer 2/3") == "Cortex"
    # non-target / non-brain labels normalize to None
    assert gt.coarse_brain_region("void") is None
    assert gt.coarse_brain_region("root") is None
    assert gt.coarse_brain_region("corpus callosum body") is None
    # entorhinal cortex is deliberately excluded from the isocortex bucket
    assert gt.coarse_brain_region("Entorhinal area lateral part layer 5") is None


# --- resolver ----------------------------------------------------------------


def test_unit_brain_region_resolves_via_peak_channel():
    nwb = make_units_nwb()
    assert gt.unit_brain_region(nwb, 10) == "Hippocampus"
    assert gt.unit_brain_region(nwb, 20) == "Cortex"


def test_unit_brain_region_rejects_non_brain_location():
    nwb = make_units_nwb()
    with pytest.raises(ValueError):
        gt.unit_brain_region(nwb, 30)  # peak channel is 'void'


def test_unit_brain_region_missing_unit_raises_keyerror():
    nwb = make_units_nwb()
    with pytest.raises(KeyError):
        gt.unit_brain_region(nwb, 999)


def test_unit_brain_region_no_units_table():
    nwb = SimpleNamespace(units=None, electrodes=None)
    with pytest.raises(KeyError):
        gt.unit_brain_region(nwb, 10)


# --- feature helpers ---------------------------------------------------------


def test_waveform_peak_trough_width_ms():
    wf = _spike_waveform(peak_ch=2, trough=20, width=15, nan_channel=0)
    # 15 samples at 30 kHz = 0.5 ms
    assert gt.waveform_peak_trough_width_ms(wf) == pytest.approx(0.5, abs=1e-6)


def test_peak_channel_waveform_ignores_nan_channels():
    wf = _spike_waveform(peak_ch=2, trough=20, width=15, nan_channel=0)
    trace = gt.peak_channel_waveform(wf)
    assert not np.isnan(trace).any()
    assert int(np.argmin(trace)) == 20


def test_extracellular_firing_rate():
    nwb = make_units_nwb()
    # 100 spikes spanning [0, 9.9] s -> ~10.1 Hz over the default full span
    assert gt.extracellular_firing_rate(nwb, 10) == pytest.approx(10.1, abs=0.2)
    # explicit interval
    assert gt.extracellular_firing_rate(nwb, 10, 0.0, 1.0) == pytest.approx(11.0, abs=1.0)


def test_unit_isi_stats_regular_train():
    nwb = make_units_nwb()
    stats = gt.unit_isi_stats(nwb, 10)
    assert stats["n_spikes"] == 100
    assert stats["mean_isi_s"] == pytest.approx(0.1, abs=1e-6)
    assert stats["cv_isi"] == pytest.approx(0.0, abs=1e-6)  # perfectly regular


# --- tools + integrity -------------------------------------------------------


def test_cat1_tools_compute_features_via_session():
    s = ToolSession(nwb=make_units_nwb(), redaction=["electrodes"])
    assert s.list_units()["unit_ids"] == [10, 20, 30]
    feat = s.compute_waveform_features(10)
    assert feat["peak_trough_width_ms"] == pytest.approx(0.5, abs=1e-6)
    rate = s.compute_firing_rate(10)
    assert rate["firing_rate_hz"] == pytest.approx(10.1, abs=0.2)
    assert rate["n_spikes"] == 100


def test_cat1_tools_never_expose_region():
    """No Cat-1 tool output may contain the withheld region/location strings."""
    s = ToolSession(nwb=make_units_nwb(), redaction=["electrodes"])
    banned = ["Field CA1", "somatosensory", "void", "location", "Hippocampus", "Cortex"]
    outputs = [
        s.describe_recording(),
        s.list_units(),
        s.get_spike_waveforms([10, 20]),
        s.compute_waveform_features(10),
        s.compute_firing_rate(10),
        s.compute_isi(10),
    ]
    blob = repr(outputs).lower()
    for term in banned:
        assert term.lower() not in blob, term


def test_describe_recording_omits_electrodes():
    s = ToolSession(nwb=make_units_nwb(), redaction=["electrodes"])
    info = s.describe_recording()
    assert "electrodes" not in info
    assert info["n_units"] == 3

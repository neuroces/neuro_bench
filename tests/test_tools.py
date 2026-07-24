"""Tests for the agent tool layer (offline stub + one network test)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from agent.tools import RedactionError, ToolSession
from benchmark import verify_ground_truth


def make_stub_nwb(fs=250.0, duration=40.0):
    """Synthetic NWB-like object: ch0 = 2 Hz (delta), ch1 = 20 Hz (beta)."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    data = np.stack([np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 20 * t)], axis=1)
    eeg = SimpleNamespace(data=data, timestamps=t, rate=None)
    return SimpleNamespace(
        identifier="stub-001",
        session_description="synthetic EEG",
        acquisition={"ElectricalSeriesEEG": eeg},
        intervals={"epochs": object(), "trials": object()},
        units=None,
        subject=SimpleNamespace(subject_id="m1", species="Mus musculus", sex="M"),
    )


@pytest.fixture
def session():
    return ToolSession(nwb=make_stub_nwb(), redaction=["intervals"])


# --- signal tools (offline) --------------------------------------------------


def test_get_eeg_epoch_decimates(session):
    out = session.get_eeg_epoch(0.0, 30.0, channel=0, max_points=500)
    assert out["n_samples"] == int(250 * 30)
    assert len(out["samples"]) <= 500
    assert out["fs_hz"] == pytest.approx(250.0, abs=0.5)


def test_compute_psd_identifies_delta_and_beta(session):
    delta = session.compute_psd(0.0, 30.0, channel=0)
    assert delta["dominant_band"] == "delta"
    assert delta["peak_frequency_hz"] == pytest.approx(2.0, abs=0.5)

    beta = session.compute_psd(0.0, 30.0, channel=1)
    assert beta["dominant_band"] == "beta"
    assert beta["peak_frequency_hz"] == pytest.approx(20.0, abs=0.5)


# --- redaction enforcement (offline) ----------------------------------------


def test_describe_recording_omits_redacted_intervals(session):
    info = session.describe_recording()
    assert "intervals" not in info  # redacted
    assert info["acquisition"]["ElectricalSeriesEEG"]["shape"][1] == 2


def test_describe_recording_includes_intervals_when_not_redacted():
    s = ToolSession(nwb=make_stub_nwb(), redaction=[])
    assert set(s.describe_recording()["intervals"]) == {"epochs", "trials"}


def test_units_access_blocked_when_redacted():
    s = ToolSession(nwb=make_stub_nwb(), redaction=["units"])
    with pytest.raises(RedactionError):
        s.list_units()
    # dispatch converts the error into a structured tool result
    result = s.dispatch("list_units", {})
    assert result.get("redacted") is True


def test_exposure_never_includes_labels(session):
    session.window = {"start_s": 0.0, "duration_s": 10.0}
    exposure = session._exposure()
    assert set(exposure) == {"fs", "eeg"}


# --- run_python sandbox (offline) -------------------------------------------


def test_run_python_returns_result(session):
    session.window = {"start_s": 0.0, "duration_s": 10.0}
    out = session.run_python("result = float(np.mean(eeg))")
    assert out["error"] is None
    assert isinstance(out["result"], float)


def test_run_python_peak_frequency(session):
    session.window = {"start_s": 0.0, "duration_s": 20.0}
    code = (
        "f, p = signal.welch(eeg, fs=fs, nperseg=int(fs*2))\n"
        "result = float(f[int(np.argmax(p))])"
    )
    out = session.run_python(code)
    assert out["result"] == pytest.approx(2.0, abs=0.5)


def test_run_python_cannot_see_nwb(session):
    session.window = {"start_s": 0.0, "duration_s": 5.0}
    out = session.run_python("result = 'nwb' in dir()")
    assert out["result"] is False


# --- dispatch routing (offline) ---------------------------------------------


def test_dispatch_unknown_tool(session):
    assert "error" in session.dispatch("nonexistent_tool", {})


def test_dispatch_routes_compute_psd(session):
    out = session.dispatch(
        "compute_psd", {"start_s": 0.0, "duration_s": 30.0, "channel": 0}
    )
    assert out["dominant_band"] == "delta"


# --- pilot asset (network) ---------------------------------------------------


@pytest.mark.network
def test_tools_on_pilot_anesthesia_window():
    questions = {q["id"]: q for q in verify_ground_truth.load_questions()}
    q = questions["C3-Q02"]  # anesthesia window on sub-521885
    with ToolSession.from_question(q) as s:
        info = s.describe_recording()
        assert "intervals" not in info  # redacted at run time
        psd = s.compute_psd(1900.0, 30.0, channel=0)
        assert psd["dominant_band"] == "delta"  # anesthesia -> delta dominant

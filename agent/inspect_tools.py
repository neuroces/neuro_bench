"""Inspect ``@tool`` wrappers over the tested :class:`ToolSession` compute engine.

Each tool reads the sample-scoped :class:`QuestionContext` (populated by the
setup solver) to build a session bound to the right asset and redaction, then
calls the corresponding ``ToolSession`` method. Open NWB handles are cached per
asset so repeated tool calls don't re-stream the file.

Redaction is enforced inside ``ToolSession`` — these wrappers cannot see the
withheld label tables either.
"""

from __future__ import annotations

import json
from typing import Any

from agent.context import QuestionContext
from agent.tools import ToolSession
from inspect_ai.tool import tool
from inspect_ai.util import store_as

_NWB_CACHE: dict[tuple[str, str, str], Any] = {}


def _session() -> ToolSession:
    ctx = store_as(QuestionContext)
    key = (ctx.dandi_id, ctx.version, ctx.asset_path)
    handle = _NWB_CACHE.get(key)
    if handle is None:
        from data.nwb_access import load_nwb

        handle = load_nwb(*key)
        _NWB_CACHE[key] = handle
    return ToolSession(
        redaction=ctx.redaction,
        channel_default=ctx.channel,
        window=ctx.window,
        nwb=handle.nwb,
    )


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=float)


def _safe(method: str, *args, **kwargs) -> str:
    """Call a ToolSession method, returning a JSON error string instead of raising.

    Keeps a wrong-tool-for-this-recording call (e.g. an EEG tool on a patch-clamp
    file) from crashing the sample — the model gets an error and can adapt.
    """
    try:
        return _dumps(getattr(_session(), method)(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
        return _dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool
def describe_recording():
    async def execute() -> str:
        """Return a metadata summary of the recording (withheld tables omitted)."""
        return _safe("describe_recording")

    return execute


@tool
def get_eeg_epoch():
    async def execute(
        start_s: float, duration_s: float, channel: int = 0, max_points: int = 2000
    ) -> str:
        """Return a single-channel EEG epoch, decimated to at most max_points samples.

        Args:
            start_s: Epoch start time in seconds.
            duration_s: Epoch duration in seconds.
            channel: EEG channel index.
            max_points: Maximum number of samples to return.
        """
        return _safe("get_eeg_epoch", start_s, duration_s, channel, max_points)

    return execute


@tool
def compute_psd():
    async def execute(
        start_s: float, duration_s: float, channel: int = 0, fmax: float = 40.0
    ) -> str:
        """Compute the Welch PSD of an EEG epoch (band powers, dominant band, peak freq).

        Args:
            start_s: Epoch start time in seconds.
            duration_s: Epoch duration in seconds.
            channel: EEG channel index.
            fmax: Maximum frequency (Hz) to report.
        """
        return _safe("compute_psd", start_s, duration_s, channel, fmax)

    return execute


@tool
def list_units():
    async def execute() -> str:
        """List sorted-unit ids in the recording, if a units table exists."""
        return _safe("list_units")

    return execute


@tool
def compute_isi():
    async def execute(unit_id: int) -> str:
        """Return inter-spike-interval statistics for one unit.

        Args:
            unit_id: The unit id.
        """
        return _safe("compute_isi", unit_id)

    return execute


@tool
def get_spike_waveforms():
    async def execute(unit_ids: list[int]) -> str:
        """Return each unit's peak-channel mean waveform (µV) and its sampling rate.

        Args:
            unit_ids: List of unit ids to fetch waveforms for.
        """
        return _safe("get_spike_waveforms", unit_ids)

    return execute


@tool
def compute_waveform_features():
    async def execute(unit_id: int) -> str:
        """Return peak-channel waveform features (trough-to-peak width in ms) for one unit.

        Args:
            unit_id: The unit id.
        """
        return _safe("compute_waveform_features", unit_id)

    return execute


@tool
def compute_firing_rate():
    async def execute(
        unit_id: int, start_s: float | None = None, stop_s: float | None = None
    ) -> str:
        """Return a unit's mean firing rate (Hz) over an interval (default: full span).

        Args:
            unit_id: The unit id.
            start_s: Interval start in seconds (optional).
            stop_s: Interval end in seconds (optional).
        """
        return _safe("compute_firing_rate", unit_id, start_s, stop_s)

    return execute


@tool
def run_python():
    async def execute(code: str, timeout_s: int = 20) -> str:
        """Run analysis code in a sandbox with the EEG signal exposed as `eeg` (rate `fs`).

        Assign a JSON-serializable `result`. The raw file and label tables are not
        accessible — infer any withheld labels from the signal.

        Args:
            code: Python source to execute.
            timeout_s: Wall-clock timeout in seconds.
        """
        return _safe("run_python", code, timeout_s=timeout_s)

    return execute


@tool
def list_current_clamp_sweeps():
    async def execute() -> str:
        """List every current-clamp sweep with its injected current (pA) and spike count."""
        return _safe("list_current_clamp_sweeps")

    return execute


@tool
def get_current_clamp_sweep():
    async def execute(index: int, max_points: int = 2000) -> str:
        """Return one current-clamp sweep: injected current, spike count, and voltage trace.

        Args:
            index: Sweep index (see list_current_clamp_sweeps).
            max_points: Maximum voltage samples to return (decimated).
        """
        return _safe("get_current_clamp_sweep", index, max_points)

    return execute


def all_tools() -> list:
    """The full tool set exposed to the agent for the EEG (Category 3) questions."""
    return [
        describe_recording(),
        get_eeg_epoch(),
        compute_psd(),
        list_units(),
        compute_isi(),
        run_python(),
    ]


def cat2_tools() -> list:
    """Tool set for the patch-clamp (Category 2) interneuron questions."""
    return [
        describe_recording(),
        list_current_clamp_sweeps(),
        get_current_clamp_sweep(),
    ]


def cat1_tools() -> list:
    """Tool set for the Neuropixels extracellular (Category 1) region questions."""
    return [
        describe_recording(),
        list_units(),
        get_spike_waveforms(),
        compute_waveform_features(),
        compute_firing_rate(),
        compute_isi(),
    ]


def tools_for(category: int | None) -> list:
    """Select the tool set appropriate to a question category."""
    if category == 1:
        return cat1_tools()
    if category == 2:
        return cat2_tools()
    if category == 3:
        return all_tools()
    # Mixed / all-category run: expose EEG, patch-clamp and extracellular tools.
    return [
        describe_recording(),
        get_eeg_epoch(),
        compute_psd(),
        list_units(),
        get_spike_waveforms(),
        compute_waveform_features(),
        compute_firing_rate(),
        compute_isi(),
        list_current_clamp_sweeps(),
        get_current_clamp_sweep(),
        run_python(),
    ]

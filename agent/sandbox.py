"""Subprocess sandbox for the ``run_python`` tool.

Model-generated code is untrusted, so we execute it in a *separate* Python
process with:

* a curated namespace — only ``np``, ``scipy.signal`` and a pre-extracted
  ``data`` dict (never the raw NWB file, so label tables cannot be re-read);
* CPU-time and address-space limits (POSIX ``resource`` limits);
* a wall-clock timeout enforced by the parent.

**Known limitation:** this is process isolation, not a full security sandbox —
it does not block filesystem or network access. Container-based isolation is a
tracked follow-up (see the implementation plan's open items). It is adequate for
running our own evaluations locally; do not expose it as a public service.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 20
DEFAULT_MEM_MB = 2048

_PREAMBLE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    from scipy import signal

    _npz = np.load({npz_path!r}, allow_pickle=False)
    data = {{k: _npz[k] for k in _npz.files}}
    fs = float(data["fs"]) if "fs" in data else None
    eeg = data.get("eeg")

    def _jsonable(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {{k: _jsonable(v) for k, v in x.items()}}
        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]
        return x

    result = None
    """
).strip()

_EPILOGUE = textwrap.dedent(
    """
    print("___NEUROBENCH_RESULT___")
    try:
        print(json.dumps(_jsonable(result)))
    except (TypeError, ValueError):
        print(json.dumps(str(result)))
    """
).strip()


def _limits(mem_mb: int):
    """Return a preexec_fn that caps CPU time and address space (POSIX only)."""
    try:
        import resource
    except ImportError:
        return None

    def _apply():
        cpu = DEFAULT_TIMEOUT_S
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        mem = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass

    return _apply


def run_sandboxed(
    code: str,
    data: dict[str, Any] | None = None,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    mem_mb: int = DEFAULT_MEM_MB,
) -> dict[str, Any]:
    """Execute ``code`` in a sandboxed subprocess.

    ``data`` arrays are exposed to the code as a ``data`` dict (plus ``fs`` and
    ``eeg`` shortcuts). The code may assign a JSON-serializable ``result``.

    Returns ``{"stdout", "result", "error", "timed_out"}``.
    """
    import numpy as np

    data = data or {}
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "data.npz"
        np.savez(npz_path, **{k: np.asarray(v) for k, v in data.items()})

        script = f"{_PREAMBLE.format(npz_path=str(npz_path))}\n\n{code}\n\n{_EPILOGUE}"
        script_path = Path(tmp) / "user_code.py"
        script_path.write_text(script)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_limits(mem_mb),
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "result": None, "error": "timeout", "timed_out": True}

        stdout = proc.stdout
        result = None
        marker = "___NEUROBENCH_RESULT___"
        if marker in stdout:
            stdout, _, tail = stdout.partition(marker)
            tail = tail.strip()
            if tail:
                try:
                    result = json.loads(tail)
                except json.JSONDecodeError:
                    result = tail

        error = proc.stderr.strip() if proc.returncode != 0 else None
        return {
            "stdout": stdout.strip(),
            "result": result,
            "error": error,
            "timed_out": False,
        }

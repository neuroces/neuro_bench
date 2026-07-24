"""Prefetch/verify assets referenced by the benchmark manifest.

No bulk downloads are required to *run* the benchmark (everything streams via
``data.nwb_access``). This utility exists to (a) verify every manifest row
resolves and opens, and (b) warm the resolved-URL cache for offline demos.

Usage::

    python -m data.download_assets --verify            # check all manifest rows
    python -m data.download_assets --verify --dandi-id 000458
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data.nwb_access import NWBAccessError, open_nwb

MANIFEST_PATH = Path(__file__).parent / "asset_manifest.csv"


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify_row(row: dict[str, str]) -> tuple[bool, str]:
    """Resolve and open one asset; return (ok, message)."""
    try:
        with open_nwb(row["dandi_id"], row["version"], row["asset_path"]) as nwb:
            n_units = len(nwb.units) if getattr(nwb, "units", None) is not None else 0
            return True, f"identifier={nwb.identifier!r} units={n_units}"
    except (NWBAccessError, Exception) as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify/prefetch benchmark assets.")
    parser.add_argument(
        "--verify", action="store_true", help="Resolve and open each asset."
    )
    parser.add_argument("--dandi-id", help="Only process rows for this dandiset.")
    args = parser.parse_args()

    rows = read_manifest()
    if args.dandi_id:
        rows = [r for r in rows if r["dandi_id"] == args.dandi_id]

    if not rows:
        print("No manifest rows to process.")
        return 0

    if not args.verify:
        print(f"{len(rows)} manifest row(s). Pass --verify to resolve and open them.")
        return 0

    failures = 0
    for row in rows:
        ok, msg = verify_row(row)
        status = "OK  " if ok else "FAIL"
        failures += 0 if ok else 1
        print(
            f"[{status}] {row['question_id']:<16} {row['dandi_id']} {row['asset_path']}  {msg}"
        )

    print(f"\n{len(rows) - failures}/{len(rows)} assets verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

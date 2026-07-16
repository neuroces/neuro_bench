"""Lazy NWB streaming from the DANDI Archive.

The whole benchmark reads NWB files *without* downloading them in full: we
resolve a DANDI asset to its S3 content URL and hand that to ``remfile`` so
``h5py``/``pynwb`` only fetch the byte ranges they actually touch.

Heavy third-party imports (dandi, remfile, h5py, pynwb) are done lazily inside
functions so this module imports cleanly in offline/CI environments.

Typical use::

    from data.nwb_access import open_nwb

    with open_nwb("000458", "0.230317.0039", "sub-.../file.nwb") as nwb:
        print(nwb.identifier)
        units = nwb.units.to_dataframe()
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_CACHE_DIR = Path(os.environ.get("NEUROBENCH_CACHE_DIR", ".cache/nwb"))
_URL_CACHE_FILE = "asset_url_cache.json"

# Retry policy for network operations (DANDI API + remote open).
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.5


class NWBAccessError(RuntimeError):
    """Raised when an asset cannot be resolved or opened."""


def _cache_dir() -> Path:
    d = Path(os.environ.get("NEUROBENCH_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _url_cache_path() -> Path:
    return _cache_dir() / _URL_CACHE_FILE


def _load_url_cache() -> dict[str, str]:
    path = _url_cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_url_cache(cache: dict[str, str]) -> None:
    try:
        _url_cache_path().write_text(json.dumps(cache, indent=2, sort_keys=True))
    except OSError:
        # A non-writable cache should never break a run; just skip persisting.
        pass


def _retry(fn, *, what: str):
    """Call ``fn`` with exponential backoff, re-raising as NWBAccessError."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad for network ops
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
    raise NWBAccessError(f"Failed to {what} after {_MAX_RETRIES} attempts") from last_exc


def get_dandiset(dandi_id: str, version: str):
    """Return a pinned DANDI dandiset handle (lazy import)."""
    import dandi.dandiapi as da

    client = da.DandiAPIClient()
    return _retry(
        lambda: client.get_dandiset(dandi_id, version),
        what=f"open dandiset {dandi_id}@{version}",
    )


def list_assets(dandi_id: str, version: str, glob: str = "*.nwb") -> list[str]:
    """List asset paths in a dandiset matching ``glob`` (sorted)."""
    dandiset = get_dandiset(dandi_id, version)
    assets = _retry(
        lambda: list(dandiset.get_assets_by_glob(glob)),
        what=f"list assets in {dandi_id}@{version} matching {glob!r}",
    )
    return sorted(a.path for a in assets)


def resolve_asset_url(
    dandi_id: str,
    version: str,
    asset_path: str,
    *,
    use_cache: bool = True,
) -> str:
    """Resolve a DANDI asset to a stable (query-stripped) S3 content URL.

    Results are cached on disk keyed by ``dandi_id/version/asset_path`` because
    the stripped content URL is stable for a pinned dataset version.
    """
    key = f"{dandi_id}@{version}::{asset_path}"
    cache = _load_url_cache() if use_cache else {}
    if use_cache and key in cache:
        return cache[key]

    dandiset = get_dandiset(dandi_id, version)
    asset = _retry(
        lambda: dandiset.get_asset_by_path(asset_path),
        what=f"resolve asset {asset_path} in {dandi_id}@{version}",
    )
    url = _retry(
        lambda: asset.get_content_url(follow_redirects=1, strip_query=True),
        what=f"get content URL for {asset_path}",
    )

    if use_cache:
        cache[key] = url
        _save_url_cache(cache)
    return url


@dataclass
class NWBHandle:
    """Holds an open NWB object and the underlying IO/file handles.

    Prefer the ``open_nwb`` context manager, which closes these automatically.
    """

    nwb: object
    _io: object
    _h5: object

    def close(self) -> None:
        for handle in (self._io, self._h5):
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    def __enter__(self) -> object:
        return self.nwb

    def __exit__(self, *exc) -> None:
        self.close()


def _open_from_url(url: str) -> NWBHandle:
    import h5py
    import pynwb
    import remfile

    def _open():
        rem = remfile.File(url)
        h5 = h5py.File(rem, "r")
        io = pynwb.NWBHDF5IO(file=h5, load_namespaces=True)
        return NWBHandle(nwb=io.read(), _io=io, _h5=h5)

    return _retry(_open, what=f"open remote NWB at {url}")


@contextmanager
def open_nwb(
    dandi_id: str,
    version: str,
    asset_path: str,
    *,
    use_cache: bool = True,
) -> Iterator[object]:
    """Context manager yielding a streamed NWB object; closes IO on exit."""
    url = resolve_asset_url(dandi_id, version, asset_path, use_cache=use_cache)
    handle = _open_from_url(url)
    try:
        yield handle.nwb
    finally:
        handle.close()


def load_nwb(
    dandi_id: str,
    version: str,
    asset_path: str,
    *,
    use_cache: bool = True,
) -> NWBHandle:
    """Open a streamed NWB file and return its handle.

    The caller is responsible for calling ``.close()`` (or using the returned
    handle as a context manager). Prefer ``open_nwb`` for scoped access.
    """
    url = resolve_asset_url(dandi_id, version, asset_path, use_cache=use_cache)
    return _open_from_url(url)

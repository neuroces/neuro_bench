"""Tests for the data access layer (offline unit tests + one network test)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from data import datasets, nwb_access

MANIFEST = Path(__file__).parent.parent / "data" / "asset_manifest.csv"


# --- dataset registry (offline) ---------------------------------------------


def test_pilot_dataset_is_pinned():
    ds = datasets.get_dataset(datasets.PILOT_DANDI_ID)
    assert ds.version is not None
    assert datasets.require_version(datasets.PILOT_DANDI_ID) == ds.version


def test_require_version_raises_for_unpinned():
    # 000003 / 000552 are intentionally unpinned until their categories are built.
    with pytest.raises(ValueError):
        datasets.require_version("000003")


def test_get_dataset_unknown_id():
    with pytest.raises(KeyError):
        datasets.get_dataset("999999")


# --- URL cache (offline) -----------------------------------------------------


def test_url_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBENCH_CACHE_DIR", str(tmp_path))
    payload = {"000458@v::a.nwb": "https://example.org/a.nwb"}
    nwb_access._save_url_cache(payload)
    assert nwb_access._load_url_cache() == payload


def test_url_cache_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBENCH_CACHE_DIR", str(tmp_path))
    assert nwb_access._load_url_cache() == {}


def test_retry_reraises_as_access_error():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ValueError("boom")

    with pytest.raises(nwb_access.NWBAccessError):
        nwb_access._retry(always_fails, what="do the thing")
    assert calls["n"] == nwb_access._MAX_RETRIES


# --- manifest (offline) ------------------------------------------------------


def test_manifest_header_schema():
    with MANIFEST.open(newline="") as f:
        header = next(csv.reader(f))
    assert header == [
        "question_id",
        "dandi_id",
        "version",
        "asset_path",
        "time_window",
        "unit_subset",
        "notes",
    ]


# --- streaming (network) -----------------------------------------------------


@pytest.mark.network
def test_stream_pilot_asset_opens():
    ds = datasets.get_dataset(datasets.PILOT_DANDI_ID)
    paths = nwb_access.list_assets(ds.dandi_id, ds.version, "*.nwb")
    assert paths, "expected at least one NWB asset in the pilot dataset"
    with nwb_access.open_nwb(ds.dandi_id, ds.version, paths[0]) as nwb:
        assert nwb.identifier is not None

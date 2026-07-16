"""Stage 0 smoke tests.

Verifies the repository is importable and, when dependencies are installed,
that core third-party libraries and the DANDI client are usable. Third-party
checks skip gracefully before `make install-dev` so the suite is green from
a bare checkout.
"""

import importlib

import pytest

LOCAL_PACKAGES = ["data", "benchmark", "grading", "agent"]
CORE_DEPENDENCIES = ["numpy", "scipy", "pandas", "h5py"]


@pytest.mark.parametrize("pkg", LOCAL_PACKAGES)
def test_local_packages_import(pkg):
    assert importlib.import_module(pkg) is not None


@pytest.mark.parametrize("dep", CORE_DEPENDENCIES)
def test_core_dependencies_importable(dep):
    pytest.importorskip(dep, reason=f"{dep} not installed yet (run `make install-dev`)")


@pytest.mark.network
def test_dandi_client_instantiates():
    da = pytest.importorskip(
        "dandi.dandiapi", reason="dandi not installed yet (run `make install-dev`)"
    )
    client = da.DandiAPIClient()
    assert client is not None

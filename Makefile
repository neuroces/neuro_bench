.PHONY: install install-dev test test-offline lint format clean

install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[dev]"

# Full test suite (may hit the network for @network-marked tests).
test:
	python3 -m pytest -v

# Offline suite for CI: skips network/DANDI-dependent tests.
test-offline:
	python3 -m pytest -v -m "not network"

lint:
	python3 -m ruff check .

format:
	python3 -m ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist

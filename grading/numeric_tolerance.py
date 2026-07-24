"""Numeric answer grading with absolute tolerance.

Models often wrap numbers in prose ("about 1414 seconds"), so we extract the
first numeric token before comparing.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_number(value: object) -> float:
    """Extract a float from a number or from the first numeric token in a string."""
    if isinstance(value, bool):  # avoid True/False -> 1.0/0.0 surprises
        raise ValueError(f"expected a number, got bool {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError("expected a number, got None")
    match = _NUMBER_RE.search(str(value))
    if not match:
        raise ValueError(f"no number found in {value!r}")
    return float(match.group())


def within_tolerance(model: object, truth: float, tolerance: float) -> bool:
    """True if the model's number is within ``tolerance`` of ``truth``."""
    try:
        return abs(parse_number(model) - float(truth)) <= tolerance
    except ValueError:
        return False

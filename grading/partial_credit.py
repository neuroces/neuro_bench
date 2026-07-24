"""Partial-credit helpers for set-membership answers (e.g. cell assemblies).

Category 5 (assembly detection) grades on how well the model's identified set of
neurons overlaps the true assembly. We use the Jaccard index, with a threshold
for pass/partial as described in the benchmark plan.
"""

from __future__ import annotations

from collections.abc import Iterable


def jaccard(a: Iterable, b: Iterable) -> float:
    """Jaccard similarity |A∩B| / |A∪B| (1.0 if both empty)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def set_score(model: Iterable, truth: Iterable, threshold: float = 0.6) -> float:
    """Map Jaccard overlap to a 0 / 0.5 / 1.0 partial-credit score."""
    j = jaccard(model, truth)
    if j >= threshold:
        return 1.0
    return 0.5 if j > 0 else 0.0

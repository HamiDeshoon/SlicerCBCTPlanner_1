"""Shared print-layout rules for Step 5 slice series and the 8x10 film."""

from __future__ import annotations

from typing import Tuple

# Rule chosen for the 8x10 film: one tooth gets a fine local series, 2-3 teeth
# get a medium series, and a quadrant/full selection gets a wider series.
SLICE_RULES = (
    (1, 15, 0.75),
    (3, 21, 0.85),
    (10**9, 27, 1.00),
)


def slice_count_for_selection(n_teeth: int) -> Tuple[int, float]:
    """Return `(n_slices, step_mm)` for a selection of `n_teeth` teeth."""
    count = max(1, int(n_teeth))
    for max_teeth, slices, step in SLICE_RULES:
        if count <= max_teeth:
            return slices, step
    return 27, 1.00

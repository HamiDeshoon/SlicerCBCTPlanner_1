# v3.3 Guide Insertion, Sleeve Ring & Intaglio Body Task Checklist

- [x] **T1**: Implement mask-space `UndercutReport` / `compute_blocked_surface` / `suggest_insertion_axis` in `CBCTPlannerLib/core/insertion.py` (ROI crop + auto-stride, 6-neighbour surface, Bresenham sweep, coronal seating band, stair-step disclosure) and register it in `CMakeLists.txt`
- [x] **T2**: Wire `ImplantPlannerLogic.evaluate_insertion` (per-implant fraction + suggested axis, `coronal_band_mm`), guide-level `seating_axis_ras` / `undercut_blocked_fraction` on `GuideBuildReport`, Step 6 `Check Draw Direction` button and `step6HintLabel` reporting
- [x] **T3**: Add implant library schema `1.1` optional ring keys, `PlacedImplant` mirrors, and `SurgicalGuideGenerator.resolve_sleeve_dimensions` (`metal_ring` pocket vs `integral_tube`) with consistency warnings, leaving bundled entries numerically unchanged
- [x] **T4**: Implement the anatomy-conforming intaglio body (arch-band crop, undercut blockout, EDT shell band, fused open sleeve collars, island drop, manifold/volume report, `guide_build` JSON, analytic fallback) plus `Anatomy-conforming base` checkbox and mask/affine forwarding from the widget
- [x] **T5**: Add `tests/test_insertion_undercut.py`, `tests/test_guide_sleeve_schema.py`, `tests/test_guide_intaglio.py` (including the no-`skimage`/no-`vtk` surface fallback path)
- [x] **T6**: Verify `ruff check .`, `ruff format --check .` (pinned `0.17.6`) and `python -m unittest discover -s tests` (284 tests, 0 failures) and record BL-19..BL-21 in `BUSINESS_RULES.md`

# v3.3 Guide Insertion & Intaglio Body Implementation Plan

## Execution Sequence (Dependency-Ordered)

1. **Undercut primitive, no call sites touched (`CBCTPlannerLib/core/insertion.py`, `CBCTPlanner/CMakeLists.txt`)**:
   - Add `UndercutReport`, `GEOMETRIC_DISCLAIMER`, `MAX_CROPPED_GRID_DIM`, `_normalize_vec`, `_crop_and_downsample_mask`, `compute_blocked_surface`, `_orthonormal_basis`, `suggest_insertion_axis`. Pure NumPy: padded 6-neighbour erosion for the surface, central differences for normals, Bresenham sweep along the dominant voxel axis, ROI crop with auto-stride above 128 voxels.
   - Register `${MODULE_NAME}Lib/core/insertion.py` in `MODULE_PYTHON_SCRIPTS` so `test_packaging_completeness.py` stays green.
   - Cover with `tests/test_insertion_undercut.py`: box and coronal dome ≈ 0, stem-under-shelf blocked vertically and clear sideways, coronal band trimming the stem walls, 20° oblique cylinder clear along its own axis and blocked along RAS-Z (voxel-rotated **and** rotated-affine variants, proving the affine path), axis recovery within 5°, empty/single-voxel safety.

2. **Wiring: per-implant check, guide-level axis, Step 6 UI (`ImplantPlanner.py`, `Phase7Tools.py`, `CBCTPlanner.py`)**:
   - `ImplantPlannerLogic.evaluate_insertion(..., coronal_band_mm=None)` resolves the axis (explicit -> implant -> mean of placed axes -> `-Z`), stores `undercut_blocked_fraction` / `insertion_axis_suggested_ras` on the implant, and suggests a tilt cone only when the verdict is blocked.
   - `GuideBuildReport` gains `seating_axis_ras` + `undercut_blocked_fraction`; `seat_vec` is resolved once before either body branch so the analytic path reports the same shared axis; `generate_surgical_guide` records the fraction and appends the undercut warnings when blocked.
   - Step 6 gains `guideIntaglioCheck` ("Anatomy-conforming base") and `btnCheckDrawDirection`; export feedback moves from `step7HintLabel` to `step6HintLabel` (the guides live in Step 6). `CBCTPlannerLogic.generate_and_export_surgical_guide` gains `segmentationNode` / `body_mode` and forwards mask, affine and volume.

3. **Library schema v1.1 (`Resources/generic_implant_library.json`, `ImplantPlanner.py`, `Phase7Tools.py`)**:
   - Bump `schema_version` to `"1.1"` and load the five optional keys with `raw.get(...)` into `ImplantLibraryEntry`, mirrored onto `PlacedImplant` in `place_or_update_implant`. Do not touch the twelve bundled entries.
   - Add `resolve_sleeve_dimensions` (`metal_ring` pocket vs `integral_tube`) and the defaulted `GuideSleeveSpec` fields. Cover with `tests/test_guide_sleeve_schema.py`: bundled v1.1 loads with null rings, ring entry resolves `metal_ring` + pocket = `od + 2c` + the height/offset warnings, ringless entry keeps the legacy formula byte-for-byte, v1.0 without keys loads.

4. **Intaglio body (`Phase7Tools.py`, `CBCTPlanner.py`)**:
   - `_supporting_mask` (5-class / 77-class / phantom tolerant), `_build_intaglio_body_and_sleeves` (arch-band crop, projection blockout, EDT band with sub-voxel clamp, collar union then bore carve, island drop, isosurface + optional Laplacian smoothing, manifold/volume/island/channel metrics), and the analytic fallback path with recorded warnings.
   - Cover with `tests/test_guide_intaglio.py`: 0.25 mm slab gives the expected offset/thickness shell with an open through-hole and a single island, empty mask falls back to analytic with a warning, `check_edge_manifold` separates a closed tet from a broken face list, and surface extraction still emits a mesh with `skimage`/`vtk` blocked (the CI path).

5. **Verification & documentation close-out**:
   - `ruff check .`, `ruff format --check .` (ruff pinned to `0.17.6` in `pyproject.toml` + CI), `python -m unittest discover -s tests` (284 tests).
   - Update `analysis/slicercbctplanner_1/BUSINESS_RULES.md`: BL-19..BL-21 rows, module/test inventory counts, and the three new test-module rows.

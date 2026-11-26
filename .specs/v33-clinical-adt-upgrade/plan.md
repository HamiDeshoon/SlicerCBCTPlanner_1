# v3.3 Implementation Plan

## Execution Sequence (Dependency-Ordered)

1. **Clinical Cross-Section Geometry (`CBCTPlannerLib/core/measurements.py`, `CBCTPlannerLib/core/arch_detector.py`)**:
   - Add `_jaw_v_bounds_at_frame(...)` to measure local jawbone vertical extent along `B` around `P` (`mask == jaw_id`), expanding `v_coords` asymmetrically (`[-max(25, |v_min|+4), max(25, v_max+4)]`, clamped safely to `[-55, +55]`) so mandible bottom and maxilla top are never clipped while tiny synthetic tests remain inside `[-25, +25]`.
   - Enforce **`T`-junction (`CR-1`)**: in `_measure_from_cs`, when `0.0 in segs_uv`, anchor `crest_uv` directly onto `W0` midpoint `(0.5*(w0_l[0]+w0_r[0]), 0.5*(w0_l[1]+w0_r[1]))` without coronal back-projection past `W0`. Update `reanchor_h_start_to_w0` default when `coronal_offset_mm is None` to return `0.0` offset (`T`-junction), and update Pre-Print Editor (`CBCTPlanner.py`) + `PrintLayoutEngine.py` accordingly.
   - Enhance **`H` internal cortical end (`CR-3`)**: in `_compute_bone_height_in_cs`, when `img_cs` has real trabecular/cortical contrast near the apical jaw boundary, detect the inner medullary-cortical boundary (`0.75..3.5 mm` inside outer periosteal exit); otherwise keep `JAW_END_CORTEX_INSET_MM = 1.5 mm`.
   - Stabilize **`W0` angle & entry (`CR-2`)** across adjacent slices in `stabilize_entry_points`: preserve existing `crest_uv` jump clamping, AND detect outlier/jittery `W0` angles relative to concordant neighbors -> rotate/project `W0` cleanly onto bone boundaries around `w0_mid`, synchronize `ridge_axis_angle_deg`, and re-anchor `H` (`T`-junction) onto `w0_mid` without crossing any IAN canal.
   - Integrate fast landmark-assisted anatomical clipping into `arch_detector.py` (anterior midline + mandibular posterior ramus cutoff) so edentulous/partial arches never zig-zag.

2. **Cross-Section Rendering, Connected Labels & Periodic `W3`/`W5` (`core_print_render.py`, `PrintLayoutEngine.py`, `CBCTPlanner.py`)**:
   - Add `depth_key_depths: Optional[Sequence[float]] = None`, `show_depth_key: bool = True`, and helper `select_periodic_depth_key(slice_index, total_slices, is_center)` to `core_print_render.py`.
   - Update `_place_caption` / `_place_non_overlapping_pill` to draw crisp color-matched hairline **leader connector lines** (`1-2 px` + `2 px` dot at geometry anchor) from displaced or adjacent `W0`, `IAN`, `SINUS`, and `H` pills directly to `w0_mid` / canal ring / sinus floor.
   - Wire `depth_key_depths` into `PrintLayoutEngine.generate_8x10_layout` and `CBCTPlannerLogic.generate_cross_section_slice_series` (also setting `inline_line_labels=True` in gallery series).

3. **`CBCTPlannerLib/adt/` Complete Package Upgrade + Step 7 & Doc/Skill Integration**:
   - Build modular package `CBCTPlannerLib/adt/` (`__init__.py`, `geometry.py`, `ali.py`, `aso.py`, `clic.py`, `doc_shapeaxi.py`, `vface.py`).
   - Refactor `AutomatedDentalToolsEngine.py` to subclass/delegate cleanly to `CBCTPlannerLib/adt` while preserving 100% of existing public API symbols and dataclass fields.
   - Add `run_native_doc_shapeaxi` in `CBCTPlannerLogic` + `"DOCShapeAXI"` button/dropdown in Step 7 UI (`CBCTPlanner.py`), with 0 new PythonQt dangers (standard `QPushButton` + `QComboBox` + `QLabel`).
   - Connect landmark-assisted ASO orientation helper cleanly into `core/orientation.py` (optional landmark-guided RAS/Frankfort check & 5-class bilateral L-R symmetry refinement).

4. **Skill Packaging + Docs Sync + Comprehensive Verification**:
   - Create a concise local workflow skill `.agent/skills/adt-cbct-workflow/SKILL.md` (if helpful, validated) and sync `README.md`, `BUSINESS_RULES.md`, `SESSION_NOTES`, `.ecc/`.
   - Add unit tests covering: `T`-junction `H`-`W0` (zero coronal overhang), periodic single `W3`/`W5` across series, adjacent `W0` angle smoothness, full mandible/maxilla unclipped `v_coords` + internal cortical `H` end, connected `IAN`/`W0` leader connectors, `DOCShapeAXI` airway/condyle/cleft + `.vtk` saliency array, upgraded `CLIC` cortex-aware localization, upgraded `ASO` Kabsch SVD + outlier filtering, and upgraded `ALI` + landmark-guided arch fitting.
   - Run `ruff check CBCTPlanner tests`, `unittest`, `verify_real_ct.py`, `verify_mcp_slicer.py`, live Slicer 5.12.3 UI smoke test, and visually inspect rendered PNGs.

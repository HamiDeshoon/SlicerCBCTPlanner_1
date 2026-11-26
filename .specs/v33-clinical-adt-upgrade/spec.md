# v3.3 Clinical Cross-Section & AutomatedDentalTools Package Spec

## Context & Scope
Upgrade `CBCTPlanner` across two tightly scoped fronts using Karpathy & Ponytail discipline (orthogonal primitives, no speculative bloat, 100% offline/Slicer compatibility with explicit optional-checkpoint bridges):

1. **Clinical Cross-Section Geometry, Framing & Film Readability (`core/measurements.py`, `core_print_render.py`, `PrintLayoutEngine.py`, `CBCTPlanner.py`)**:
   - **CR-1 — Strict `T`-Junction Between `H` and `W0` (Never `+` Cross)**:
     - Whenever `W0` (`depth == 0.0`) is measured, `H` MUST start on the exact midpoint of `W0` (`w0_mid = 0.5 * (w0_l + w0_r)`) and proceed monotonically apically along `+d_apical` with zero coronal overhang past `W0` (true `T`, never cruciform `+`).
     - In `reanchor_h_start_to_w0(...)` and Pre-Print Editor drag synchronization, default to zero coronal overhang (`coronal_offset_mm=0.0` unless explicitly supplied), keeping `H` intersecting `W0` midpoint as a clean `T`.
   - **CR-2 — Adjacent-Slice `W0` Angle Consistency**:
     - Restrict per-slice PCA ridge axis estimation to alveolar anatomy and stabilize both `crest_uv` AND `W0` tilt angle across adjacent similar slices inside `stabilize_entry_points(items)` (bounded inter-slice angular jump `<= 8°` when adjacent ridge masks have high similarity, e.g., IoU >= 0.70 or concordant neighbors), while synchronizing `H` start (`T`-junction) and cortical-bounded endpoints.
   - **CR-3 — Unclipped Full-Mandible / Full-Maxilla Framing + Internal Cortical `H` End**:
     - Expand `_resample_arch_cross_section` and `measure_at_arc_mm` vertically using local jaw-mask vertical bounds (`mask == jaw_id`) so tall mandibles (never clipped inferiorly) and tall maxillae (never clipped superiorly) fit completely with `>= 4.0 mm` apical/coronal margin inside `v_coords`, while preserving `[-25.0, +25.0] mm` whenever the jaw already fits inside that window (preserving synthetic unit tests).
     - Stop `H` at the **internal border of the basal/maxillary cortical plate** (detecting the endosteal/medullary transition along the outer cortical rim `0.75..3.5 mm` inside `full_h` when CBCT bone contrast is present, falling back cleanly to `JAW_END_CORTEX_INSET_MM = 1.5 mm` on uniform synthetic phantoms) or ipsilateral vital structure (IAN / Sinus).
   - **CR-4 — Periodic Single `W3` or `W5` Readout**:
     - Add `depth_key_depths: Optional[Sequence[float]] = None` and `show_depth_key: bool = True` to `render_cross_section(...)`.
     - Across multi-slice galleries and 8x10 print layouts, display **one** compact depth reading (`W3` OR `W5`, never a 4-row `W3/W5/W6/W9` table on every frame) periodically every few slices (e.g., stride `4` alternating `W3` / `W5`, plus `W3` on center slice if not already covered), while keeping default single-call `render_cross_section` behavior safe for single-slice inspection.
   - **CR-5 — Connected `W0` and `IAN` Labels**:
     - Always attach `W0` and `IAN` (plus `H` and `SINUS`) captions directly to their line midpoint / nerve ring via crisp hairline leader connectors in their matching semantic color whenever `inline_line_labels=True`, and enable `inline_line_labels=True` in both interactive gallery series and Pre-Print / key print slices so captions are never floating detached.

2. **AutomatedDentalTools Full Package Upgrade (`DOCShapeAXI`, `CLIC`, `ASO`, `ALI`, `VFACE`, `AutoCrop3D`)**:
   - Audit baseline against `DCBIA-OrthoLab/SlicerAutomatedDentalTools` (`DOCShapeAXI_CLI`, `DOCShapeAXI`, `CLIC`, `ASO_CBCT`, `ALI_CBCT`, `VFACE`, `ADT/ADTLib`) and `DCBIA-OrthoLab/ShapeAXI`:
     - Upstream requires WSL2 + Conda + `pytorch3d` + `ocnn` + `monai`, and uses several slow Python per-vertex loops.
     - `PythonSlicer.exe` (5.12.3) natively includes `torch`, `torchvision`, `timm`, `vtk`, `scikit-image`, `scipy`, `SimpleITK`, `connected-components-3d (cc3d)`, `Pillow`, `nibabel`.
   - Package a unified, clean **`CBCTPlannerLib/adt/`** package (with 100% backwards-compatible re-exports from `CBCTPlannerLib/AutomatedDentalToolsEngine.py` and `CBCTPlannerLib/__init__.py`):
     - **`adt/geometry.py`**: C-level vectorized VTK<->NumPy surface helpers, `marching_cubes` / `vtkDiscreteMarchingCubes` surface extraction, vectorized `cKDTree` median surface smoothing (replacing upstream O(V*deg) Python loops), Kabsch/SVD rigid alignment, pairwise distance/direction landmark outlier rejection (ported from `ASO_CBCT`'s `GetLandmarkToRemove`).
     - **`adt/ali.py`**: Upgraded 3D ALI landmarks (`S`, `N`, `A`, `B`, `ANS`, `PNS`, `Me`, `Gn`, `Pg`, `Co_R/L`, `Go_R/L`, `Or_R/L`, `Po_R/L`, `MentalForamen_R/L`, per-tooth crowns/apices) with surface-gradient & HU cortical crest refinement + landmark-derived arch bounds (`anterior_symphysis`, `mental_foramina`, `gonial_posterior`) usable by `arch_detector.py` + `export_ali_landmarks_mrk_json`.
     - **`adt/aso.py`**: Upgraded 3D ASO rigid Kabsch/SVD + Frankfort Horizontal / Occlusal & Midsagittal symmetry plane solver with outlier rejection and optional `vtkIterativeClosestPointTransform` refinement; enriches `core/orientation.py` so landmark-assisted orientation works even on masks without individual FDI labels (including 5-class!).
     - **`adt/clic.py`**: Upgraded 3D cortex-aware `CLI-C` (`Buccal`, `Bicortical`, `Palatal`/`Lingual`) measuring exact signed offset against the fitted `DentalArch` normal & local jawbone cortical walls + 3D EDT root contact distance + optional `torchvision` Mask-R-CNN (`final_model.pth`) slice-wise inference bridge + collision-free Slicer slice legend helper + CLIC labelmap export (`1=Buccal, 2=Bicortical, 3=Palatal`).
     - **`adt/doc_shapeaxi.py`**: Complete `DOCShapeAXI` engine for **Nasopharyngeal Airway Obstruction** (binary / 4-class severity / obstruction % regression), **Mandibular Condyle** degeneration (4-class severity), and **Alveolar Cleft / Bone Defect** (4-class severity), outputting quantitative classification + per-vertex `ShapeAXI_Saliency` explainability scalar array on `vtkPolyData` (`.vtk` exportable for Slicer `ColdToHotRainbow` display) with optional `ShapeAXI`/`timm` multi-view Grad-CAM checkpoint bridge (`pytorch_grad_cam` / native PyTorch CAM hooks) and deterministic 3D geometric/curvature fallback.
     - **`adt/vface.py`**: Bilateral asymmetry metrics & sagittal mirror matrix cleanly modularized.
   - Integrate `ALI` landmarks into `core/arch_detector.py` (`_extract_jawbone_crest_points` anterior/foramen/ramus bounds when bone extends upward) and `DOCShapeAXI` into Step 7 UI + logic.

## Out of Scope
- Bundling 2 GB+ external Conda environments or auto-downloading external checkpoints without user action.
- Modifying git history / committing / pushing before user confirmation.

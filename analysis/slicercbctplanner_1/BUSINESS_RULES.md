# Code Modernization Assessment & Business Rules: `SlicerCBCTPlanner_1` → v3.1

## 1. Executive Assessment (`/modernize-assess`)
- **System**: `SlicerCBCTPlanner_1` (3D Slicer Scripted Loadable Extension in Python 3.9+ / VTK / Qt / CTK / NumPy / SciPy / PyTorch).
- **Codebase Inventory**: 27 Python modules across `CBCTPlanner/` and `CBCTPlannerLib/` + 13 test modules (**167 unit tests** as of 2026-11-24).
- **Modernization Method**: **Uplift + Feature Expansion** — preserve all behavioral guarantees (B1–B22, O1–O9, F1–F20) while fixing critical Slicer runtime/rendering defects and adding clinical subsystems (`RiseMarEngine`, `AutomatedDentalToolsEngine`, `SlicerDentalSegmentator` Vivid 3D/VR, `RaySum` panoramic, transparent RGBA exports).

---

## 2. Extracted Clinical & Engineering Rule Cards (`/modernize-extract-rules`)

### Rule Card RC-01: Honest Clinical Measurements & Dentate/Edentulous Ridge Envelope
- **Given**: A resampled cross-section `(u_coords, v_coords, mask_cs, img_cs)` orthogonal to the dental arch at `arc_mm`.
- **When**: Computing alveolar bone height `H`, crestal width `W0`, depth widths `W3`, `W5`, `W6`, `W9`, and vital nerve/sinus clearance.
- **Then**:
  1. **Socket-bridging `W0` on dentate ridges**: on a dentate site the outer-buccal and outer-lingual bone walls may be separated by an in-socket tooth root (or PDL gap ≤ 1.5 mm). The crest width `W0` **must bridge** from outer-lingual cortex to outer-buccal cortex across the socket — not stop at the mesial or distal surface of the tooth. Helpers `_same_jaw_tooth_mask`, `_find_socket_crest_rims`, `_select_ridge_run_indices`, and `_sample_band_profile` implement this bridge. Expected `W0` on a bridged molar: **6.0 – 9.5 mm** (regression test: `test_crest_and_w0_come_from_bone_not_the_tooth_crown` asserts `crest_width_mm ∈ [6.0, 9.5]`).
  2. **Large-angle axis stability**: `_detect_alveolar_crest_and_axis` uses PCA on bone voxels; at angles > 45° the minor eigenvector was selected instead of the major axis. The fix constrains `crest_axis_uv` to point generally buccolingually (small |u| component) even after PCA rotation flip, stabilized via `_stabilize_pca_axis`.
  3. **Height line endpoint guards**: `_compute_bone_height_in_cs` uses 3D spherical EDT only when a confirmed IAN canal is present in the ROI; on non-canal anterior slices the ray is extended to the inferior jaw boundary rather than stopping at a spurious EDT minimum. The H line always reaches the inferior jaw boundary unless the IAN canal is encountered first.
  4. **IAN canal clearance search radius**: `_select_local_vital_structure` now uses `u_ok = ±12.0 mm` (was ±8.5 mm) and 3D ROI = 20 mm (was 15 mm) so posterior canals > 15 mm from the crest are detected.
  5. **Width–crest synchronization**: after `_move_crest`, `width_endpoints_uv` is resynchronized at the new crest position, eliminating the slice-to-slice 2 mm single-plate spuriousness when the socket geometry shifts between adjacent slices.
  6. Cross-section rendering: grayscale X-ray film background, `H~` dashed caliper annotation for interpolated (non-IAN) heights, standard `H` solid caliper for IAN-limited heights. Root/implant/safety overlays removed. Remaining overlays: jaw contour (subtle gold), bone height (green), `W0` (yellow), `W3` (cyan), `W5` (orange), nerve clearance vector (red dashed).

### Rule Card RC-02: Print File Generation (`PDF` + `300 DPI PNG` + `HTML`) & Inline Measured Gallery
- **Given**: User selects 1 tooth, multiple teeth, or a quadrant (Q1–Q4) and clicks **Generate Print Layout** or **Measure & Show Slices**.
- **When**: `PrintLayoutEngine.generate_8x10_layout` executes.
- **Then**:
  1. Uses `self._parameterNode` (or `self.logic.getParameterNode()`), never `self.getParameterNode()` on the widget.
  2. Uses `segmentationNode.GetDisplayNode().GetSegmentVisibility(seg_id)` / `SetSegmentVisibility(seg_id, keep)` and 2-arg safe `GetTag`.
  3. Writes **three print files** (`<Scan>_Print_Layout.pdf`, `<Scan>_Print_Layout.png`, `<Scan>_Print_Layout.html`) to the user's selected/configured output directory (defaulting to `~/Documents/CBCTPlanner_Prints` with a folder picker & quick "Open Folder / Save As" button, or `out_dir` when passed programmatically).
  4. Immediately populates the **Inline Measured Cross-Section Gallery** inside the module panel AND opens the interactive high-resolution print preview dialog.
  5. **Pano/VR panels**: panoramic and VR tiles are rendered with **transparent RGBA backgrounds** (no `#020617` dark fill). `_paste_contain` composites RGBA over the film via alpha-channel; `.convert("RGB")` is never applied before pasting transparent tiles.
  6. **Panoramic projection**: `PrintLayoutEngine._pano_strip_uri` reads `PanoProjectionMode` from the param node — defaults to `"RaySum"`, never hardcodes `"Mean"`.

### Rule Card RC-03: Vivid Teeth & Roots 3D/VR Styling (`gaudot/SlicerDentalSegmentator`)
- **Given**: A 5-class (`DentalSegmentator`) or 77-class (`ToothFairy3`) segmentation node and/or CBCT volume node.
- **When**: Applying 3D styling or Volume Rendering presets.
- **Then**:
  1. Teeth (crowns & roots) receive vivid anatomical colors (`1.0` 3D opacity) with Phong specular surface properties (`Ambient=0.22`, `Diffuse=0.82`, `Specular=0.48`, `SpecularPower=28`) so roots and crowns are crisp and vivid.
  2. Maxilla and Mandible jawbones default to translucent 3D opacity (`0.25`) so **tooth roots and the mandibular canal are clearly visible through the bone**, controllable via a live **Bone Opacity Slider (`0.0 – 1.0`)** and **Vivid Presets** (`Vivid Teeth & Roots (Translucent Bone)`, `Roots & Nerve X-Ray Only`, `Endo Pulp & Canals`, `Opaque Bone & Teeth`).
  3. VR preset `"DentalSegmentator Vivid Teeth & Roots"` suppresses soft tissue and makes trabecular bone a subtle veil (`alpha <= 0.10`) while ramping root dentin (`950–1450 HU`) and crown enamel (`1500–2800 HU`) to high opacity (`0.75–1.0`).

### Rule Card RC-04: RISE-MAR Metal Artifact Reduction (`Masaaki-75/rise-mar`)
- **Given**: A CBCT volume affected by metallic dental restorations, implants, or brackets, and weights at a user-configured `.pkl` / `.pth` checkpoint (stored in Slicer settings).
- **When**: The user enables/runs **RISE-MAR Metal Artifact Reduction**.
- **Then**:
  1. Clips HU to `[-1024, 3072]` and normalizes to `[0, 1]` per `Ma et al., IEEE TMI 2025`.
  2. Loads user-configured RISE-MAR weights (`.pkl` / `.pth`) with sinogram-streak/frequency hybrid fallback when weights are absent in CI.
  3. Restores streak-corrupted peri-implant/interproximal regions while preserving true high-density metallic cores (`> 2500 HU`), computes pre/post Clinical Quality Assessor (`CQA`, 1–10 scale) scores, and outputs `<Volume>_RISE_MAR` (or updates in-place).

### Rule Card RC-05: RAS Auto-Standardization
- **Given**: A CBCT volume is loaded or the user triggers Step 1.
- **When**: `CBCTPlanner._setDefaultParameters` and `AutomatedDentalToolsEngine` initialize.
- **Then**:
  1. RAS auto-standardization is the **only** mode: Pitch/Roll/Yaw alignment to the occlusal plane and midsagittal plane is executed automatically on volume load.
  2. The legacy `autoStandardizeOnLoad` boolean Slicer settings key was removed in `chore(settings): drop the legacy autoStandardizeOnLoad key` (`96ee440`). No code should write or read it.

### Rule Card RC-06: Panoramic Default `RaySum` & Transparent Pano/VR
- **Given**: Any call to generate, display, or export the panoramic image or the 3D VR render.
- **When**: `PanoramicCurveLogic`, `CBCTPlanner`, `Phase7Tools`, `PrintLayoutEngine`, or `AdvancedTools` initializes or triggers panoramic generation.
- **Then**:
  1. `PanoProjectionMode` defaults to `"RaySum"` everywhere. `"Mean"` must never be hardcoded in `PrintLayoutEngine`, `Phase7Tools`, or param node default combo entries.
  2. `PanoramicCurveLogic._finalize_projection` scales RaySum as `accum / max(1, n_layers) * 1.25` and applies a percentile W/L mapping in `_populate_panoramic_node_from_array` to match clinical film density.
  3. **Transparent air in display**: `display_panoramic_in_slice_view` calls `SetApplyThreshold(True)` on the volume display node to make air (HU ≈ −1000) transparent; it also clears Green Foreground and Label layers to prevent 3D volume bleed behind the panoramic.
  4. **Isotropic tight-fill**: the panoramic viewport crops empty air margins and fills the slice view with the dental arch isotropically (no pillarbox/letterbox black bars, no stretch).
  5. **RGBA PNG capture**: `capture_3d_view_png_base64` and `VRVisualization._vr_capture` set `SetAlphaBitPlanes(1)`, `GradientBackgroundOff()`, `SetBackgroundAlpha(0.0)`, `SetInputBufferTypeToRGBA()` with a `finally:` restore block. The VR render node background is set to `(0.0, 0.0, 0.0)` with zero alpha before capture.
  6. `_data_uri_to_image` preserves the RGBA channel — it must NOT call `.convert("RGB")` before returning.

---

## 3. Known Limitations & Backlog

| ID | Item | Priority |
|----|------|----------|
| BL-01 | Smooth `W0`/`W3`/`W5` along arch (1-site Gaussian, suppress single-slice outliers between roots) | Medium |
| BL-02 | Tune `compute_panoramic_display_window_threshold` on more scans; expose percentile UI slider | Medium |
| BL-03 | RaySum quality: test `1.0×` vs `1.25×` scale; consider gamma/CLAHE post-processing | Low |
| BL-04 | Make `Zoom(1.40)` user-adjustable in VR tight-frame | Low |
| BL-05 | Dynamic `half_v_mm` for tall anterior mandibles (current 25.0 mm clips chin in some scans) | Medium |
| BL-06 | PDL gap tolerance validation on real molar CT (current tolerance: ≤ 1.5 mm) | Medium |
| BL-07 | Move `test_clinical_delivery.py` NIfTI parser to `tests/helpers/nifti_reader.py` | Low |
| BL-08 | Phase 7 batch report: consume RaySum pano array in HTML report (currently stores only `.npy`) | Low |
| BL-09 | Trim unused `core/diagnostics.py` code (~180 lines of `provenance_hash` + report writer) | Low |
| BL-10 | Real-CT smoke test with `ToothFairy3S_0021_0000.nii.gz` — **done 2026-11-24**: `verify_real_ct.py` headless in Slicer's Python, 19/19 checks, 205 sections (see §6) | Closed |
| BL-12 | Withhold `W0` when the opposite cortex is not in the section (185/205 real-CT sections cut interproximally) instead of reporting a 1–3 mm plate as a crest width | Medium |
| BL-11 | Decide grayscale default in `generate_8x10_layout(grayscale=...)` for CLI/batch callers | Medium |

---

## 4. Test Suite Reference

| File | Tests | Coverage Area |
|------|-------|---------------|
| `tests/test_anatomy_aware_measurements.py` | 38 | W0 socket-bridging, crest axis, height guards, IAN clearance |
| `tests/test_print_layout.py` | 22 | Pano/VR RGBA, transparent panels, pill placement |
| `tests/test_clinical_delivery.py` | 18 | End-to-end print file generation (PDF/PNG/HTML) |
| `tests/test_slicer_integration.py` | 21 | RaySum default, pano display, VR background |
| `tests/test_core_math.py` | 14 | EDT, axis math, plausibility bounds |
| `tests/test_rise_mar.py` | 12 | MAR inference, CQA scoring, fallback |
| `tests/test_vr_visualization.py` | 11 | VR preset, RGBA capture, background |
| `tests/test_automated_dental_tools.py` | 10 | ALI, ASO, CLI-C, VFACE |
| `tests/test_panoramic_curve.py` | 9 | RaySum projection, tight-fill, threshold |
| `tests/test_labels.py` | 7 | Label color, FDI disambiguation |
| `tests/test_core_print_render.py` | 7 | Grayscale film, caliper render |
| `tests/test_phase7.py` | 5 | Batch report, pano array |
| `tests/test_advanced_tools.py` | 3 | AdvancedTools RaySum combo default |
| **Total** | **167** | |

Run: `python -m unittest discover -s tests`

---

## 5. Data Pipeline & Module Handoff Contracts (`/data-engineering`)

```text
DICOM / NIfTI Volume
  └─► core/volume_loader.py + core/orientation.py (RAS standardization, zero-copy CBCTVolume)
        └─► core/labels.py (ToothFairy3 77-class vs 5-class DentalSegmentator disambiguation)
              ├─► core/arch_detector.py ──► PanoramicCurveLogic.py (RaySum CPR, Z-padding crop, threshold W/L)
              ├─► core/measurements.py  ──► socket-aware bone_component, crest/W0 bridging, ray-guarded H
              │     └─► core/diagnostics.py (provenance: label-derived vs H~ interpolated)
              └─► core_print_render.py  ──► PrintLayoutEngine.py (RGBA transparent pano/VR + 8x10 PDF/PNG/HTML)
```

### Optimization & Engineering Backlog (`OPT-01` – `OPT-05`)
| ID | Target | Optimization / Follow-Up | Impact |
|---|---|---|---|
| `OPT-01` | `PanoramicCurveLogic.py` | Cache `map_coordinates` 3D slab grid per `(arch_hash, spacing_mm, height_mm, offset_mm)` so toggling `RaySum`/`MIP`/`MinIP`/`Mean` avoids re-interpolating the CBCT volume. **Measured 2026-11-24: 0.16–0.27 s for 207 arcs × 240 rows at 0.30 mm (0.8–1.3 ms/column)** — already interactive, so only the 3D MIP card (0.64–0.95 s) is worth caching | High (interactive FPS) |
| `OPT-02` | `core/measurements.py` | Pre-index `local_vital_mask` bounding boxes and batch-vectorize `measure_all_sites` across adjacent slices (`-2 / 0 / +2 mm`) | Medium (multi-site speed) |
| `OPT-03` | `core_print_render.py` / `PrintLayoutEngine.py` | Pass `PIL.Image` / `ndarray` directly instead of `RGBA -> base64 -> PIL.Image` round-trips when composing 8×10″ film sheets | Medium (export latency) |
| `OPT-04` | `AdvancedTools.py` | Vectorize `finalize_transparent_png_bytes` border-connected flood-fill with `scipy.ndimage.label` on background mask instead of Python pixel BFS | Medium (high-DPI capture) |
| `OPT-05` | `Phase7Tools.py` / `cli_batch.py` | Emit structured JSON/Parquet-ready batch run manifest (`provenance_hash`, `calibration_stamp`, `withheld_reason`, site `H`/`W0`/`W3`/`W5`) for cohort QA | Medium (data pipeline QA) |

---

## 6. Real-CT Verification Record (2026-11-24)

Harness: `analysis/slicercbctplanner_1/verify_real_ct.py`, run with
`"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe"` (Python 3.12.10, VTK 9.6.2, NumPy 2.4.6, SciPy 1.17.1).
Data: `ToothFairy3S_0021_0000.nii.gz` + `ToothFairy3S_0021.nii.gz`, 370×370×169 @ 0.30 mm.
**19/19 checks pass over 205 cross-sections** (1 mm arc sweep, mandible + maxilla).

| Rule Card | What was verified | Result |
|---|---|---|
| RC-05 | Anatomical inversion auto-correction (`canonicalize_to_ras` → `auto_fix_anatomical_orientation`) | This scan is S-I **and** L-R inverted in voxel space; both axes corrected, post-fix report clean |
| RC-01.1 | Dentate `W0` bridges across the socket | 20/20 two-wall sections bridged, W0 5.1–10.2 mm (median 9.0), **12/12 molars ≥ 6 mm**, no sub-3 mm plate |
| RC-01.2 | Tilted ridges keep `W0` on bone | 79/79 sites (|axis| ≥ 15°, max 31.8°), both rims on cortex |
| RC-01.3/4 | `H` reaches the basal jaw end; stops before the IAN | 61/68 non-canal slices within 1.5 mm of the jaw border (median H 24.4 mm); 137/137 canal slices keep the caliper out of the canal (min lateral clearance 0.09 mm) |
| RC-06 | `RaySum` pano + RGBA transparent isotropic pano/VR | 240×206 RaySum in 0.16–0.27 s; strip 814×560 RGBA `alpha_min=0`, tight-filled; 3D MIP 613×280 RGBA in 0.64–0.95 s |
| RC-02 | 8×10 film export | PNG 2400×3000 @ 300 DPI, PDF `/MediaBox` 8.0×10.0 in, HTML `background: transparent` on pano/VR panels |

**Caveat:** the harness is headless — Slicer's Green view, 3D viewport and print-preview dialog were not rendered; only
the functions they call were exercised. See `SESSION_NOTES_2026-11-24.md` §8 for the full table and caveats
(incl. BL-12: 185/205 sections clip a single cortex, where `W0` is a plate rather than a bridge).


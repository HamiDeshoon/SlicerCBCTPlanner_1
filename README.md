# SlicerCBCTPlanner

**3D Slicer Scripted Extension for Dental CBCT Implant & Orthognathic Planning**

> **Not for diagnostic use / planning aid only.** All CBCT gray values are approximate and not calibrated Hounsfield Units (HU). Verify all anatomical structures, nerve proximity, and implant positions clinically.

## Features

### 1. Volume, Custom AI Model Checkpoint & 5-Class DentalSegmentator Integration
- **Place Your Own AI Checkpoint (`CheckpointInference.py`)**:
  - Load your own trained 3D segmentation checkpoint directly in **Step 1 -> AI Model Checkpoint (Place Your Checkpoint)** or via the headless CLI (`CBCTPlanner/cli_batch.py --checkpoint ...`).
  - **Supported formats**:
    - PyTorch `state_dict` / checkpoint dictionaries (`.pth`, `.pt`, `.ckpt`) — automatically unwraps `network_weights`, `state_dict`, `model_state_dict`, or `model`, strips `module.` / `_orig_mod.` prefixes, infers `(in_channels, num_classes, base_filters)`, and uses shape-safe weight loading (`load_state_dict_shape_safe`) so custom layers never crash with size mismatches.
    - Custom Python architecture scripts (`.py`) + class/builder name (e.g., `ResEncoderUNet`, `CascadingUNet`, `StripPoolUNet`).
    - TorchScript archives (`.pt`, `.ts`) and ONNX models (`.onnx` via `onnxruntime`).
    - `nnUNetv2` trained model folders (`dataset.json` + `plans.json` + `fold_0/checkpoint_final.pth`).
    - Pre-computed 3D labelmaps (`.nii`, `.nii.gz`, `.npy`, `.npz`).
  - **3D Sliding-Window Inference**: Configurable 3D patch size (e.g. `96,96,96` or `128,128,128`), overlap (`0.5`), 3D Gaussian importance weighting, intensity normalization (`ct_hu`, `zscore`, `minmax`), and automatic mapping to `DentalSegmentator` 5-class (`Maxilla`, `Mandible`, `Upper Teeth`, `Lower Teeth`, `Mandibular Canal`), `ToothFairy` multi-class FDI, or binary segmentation nodes.
- **5-Class DentalSegmentator Styling & VR Presets**:
  - Automatic 5-class color and 3D opacity presets (`Maxilla & Upper Skull`, `Mandible`, `Upper Teeth`, `Lower Teeth`, `Mandibular canal`) and `SlicerDentalSegmentator` integration.
  - Non-destructive 3-point Occlusal Plane alignment (`vtkMRMLLinearTransformNode` applied jointly to volume, segmentation, curves, fiducials, and rulers with one-click reset).

### 2. Dental Arch Curve & Fast In-Process Curved Planar Reformation (CPR)
- Right-to-Left (descending RAS X) arch curve parameterization (`18..11, 21..28` / `48..41, 31..38`) with smooth cubic B-spline auto-fit (`scipy.interpolate.splprep`) and user-edit protection.
- Fast in-process vectorized `scipy.ndimage.map_coordinates` panoramic straightening (~50–150 ms) with handedness verification (`det(N, -B, T) == +1`).

### 3. Interactive Cross-Section Navigation, 3-Slice Lightbox & +-30 deg Tilt
- Synchronized **Axial (Red)** perpendicular draggable line, **Panoramic (Green)** vertical cursor line, and **Cross-Section (Yellow)** orthogonal view.
- **3-Slice Cross-Sectional Lightbox** (`-2 mm, 0 mm, +2 mm` or custom step) and **$\pm 30^\circ$ Cross-Section Tilt** slider.

### 4. Inferior Alveolar Nerve (IAN) Canal Tracing & Safety Clearance (`F1`)
- Bilateral Left/Right mandibular canal curves (`CBCT_Canal_L`, `CBCT_Canal_R`), 2.0 mm 3D tube visualization, Yellow slice intersection ring, and 3D clearance warnings (`< 2.0 mm`).

### 5. Virtual Implant Placement, Prosthetic Library & Surgical Guide Export (`F2, F11, F14, F16, F19`)
- **Implant Library (`Resources/implant_catalog.json`)**: Generic and manufacturer systems (Straumann, Nobel Biocare, Zimmer Biomet, Megagen) with 1.5 mm radial + 2.0 mm apical safety zone meshes and bone-density (`D1–D4`) assessment along the implant body.
- **Prosthetic Library (`Resources/generic_prosthetic_library.json`, `F16`)**: Straight / 15° / 25° Ti-base abutments and tooth-group anatomic crown envelopes with restorative emergence angle and crown-to-implant (C/I) ratio evaluation.
- **Surgical Guide Generator (`Resources/printer_presets.json`, `F14`)**: Tooth- and mucosa-supported surgical guide baseline mesh generation with open cylindrical sleeve drill bores, printer offset presets (`SLA/DLP`, `MSLA/LCD`, `PolyJet`), and watertight `.stl` + `.json` export.
- **AI-Suggested Implant Plan (`F19`)**: Generates an initial plan marked `"Suggestion (unconfirmed)"` requiring explicit clinician confirmation before export.

### 6. Clinical Tools, 3D Cephalometry, AutomatedDentalTools & DICOM SEG/SR (`F4–F18, F20`)
- **Bone Density ROI (`F4`)**, **TMJ Bilateral Condyle Views (`F6`)**, **Maxillary Sinus Floor Analysis (`F7`)**, **Airway Pharyngeal Volume & Minimum Cross-Sectional Area (`F8`)**, **Superimposition (`F9`)**, **Endodontic Working-Length & Root-Canal Curvature (`F10`)**, **3D Cephalometry (`SNA, SNB, ANB, FMA, GoGn-SN, U1-L1`, `F17`)**, **AutomatedDentalTools Bridges (`ASO, AMASSS, ALI, AReg, AutoCrop3D, BatchDentalSegmentator, MRI2CBCT`, `F18`)**, **DICOM SEG & Structured Report Export (`F15`)**, and **Headless Batch CLI (`CBCTPlanner/cli_batch.py`, `F20`)**.

## Orientation, Auto-Fit & Print Layout Workflow

### RAS orientation (Step 1)
- **Detected axcodes** are shown next to **Standardize to RAS**, **Auto-detect issues**, and **Flip L-R / A-P / S-I**.
- Standardization is a lossless axis permutation/flip: voxel data, spacing and world coordinates are preserved. Oblique volumes are flagged and must be resampled explicitly with Slicer's `OrientScalarVolume` / `vtkImageReslice`; they are never silently resampled.
- **Auto-detect issues** uses the segmentation: mandible below maxilla (S-I), anterior arch convexity toward +Y (A-P), and FDI-labelled teeth on the correct side (L-R). With 5-class labels the left/right test reports *cannot determine* instead of guessing.
- The manual flip buttons are for wrong headers; they mirror the anatomy in the chosen axis. Auto-standardize on load is **off by default** (QSettings) so patient data is never mutated silently.

### Auto-Fit Arch (Step 2)
- Auto-Fit needs a **segmentation with tooth labels** (DentalSegmentator 5-class, ToothFairy-style FDI, or both). No segmentation triggers an explicit warning and manual point placement.
- The fit uses label voxels only (no surface meshes): per-FDI centroids when at least six teeth exist, otherwise a principal-curve fallback on the crown half of the 5-class tooth mask. The result is ordered Patient Right -> Patient Left and reported as `Fitted N pts, L mm, RMS r mm (method)`.
- Fits with RMS > 3 mm or arc length outside 60-200 mm are rejected with the reason shown; previous curve points are pushed to Undo. A manually edited curve is never overwritten by the mask pipeline.

### Step 5 slice series and 8x10 film
- Slice counts come from one rule: 1 tooth -> 15 slices @ 0.75 mm, 2-3 teeth -> 21 @ 0.85 mm, 4+ teeth -> 27 @ 1.00 mm. Selections are sorted anatomically, and teeth are grouped by quadrant into one page each.
- Every overlay is derived from measured millimetre coordinates in the arch cross-section; unmeasurable sites print `--` / `not measurable`, never a synthetic bone drawing.
- The panoramic strip is rendered from the panoramic volume with arc ticks (not a screenshot), and VR uses the real 3D view. If either is unavailable the page says so explicitly.
- PDF pages are composed at 8x10 in / 300 dpi (MediaBox 576x720 pt); the HTML file remains a browser-viewable report. Segment visibility is restored after generation, and opening the browser is left to the user.

## Installation in 3D Slicer
1. Open **3D Slicer** (5.6 or newer).
2. Navigate to **Edit -> Application Settings -> Modules**.
3. Under **Additional module paths**, click **Add** and select the `CBCTPlanner` directory inside this repository (the folder containing `CBCTPlanner.py`).
4. Click **OK** and restart 3D Slicer.
5. Open the module from **Modules -> Dental -> CBCT Implant Planner**.

## Running Headless Batch CLI (`F20`)
```bash
python3 CBCTPlanner/cli_batch.py \
  --input /path/to/cbct_volume.nii.gz \
  --checkpoint /path/to/my_3d_unet_checkpoint.pth \
  --label-scheme dental_segmentator_5 \
  --patch-size 96,96,96 \
  --teeth 36,46 \
  --output-dir /path/to/output_report
```

## Running Tests
```bash
python3 -m unittest discover -s tests -v
python3 tests/smoke.py
```

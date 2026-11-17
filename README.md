# SlicerCBCTPlanner v3.0 — Dental Radiology, Implantology & AI Suite for 3D Slicer

**SlicerCBCTPlanner v3.0** is a full-stack 3D Slicer scripted extension for automated pre-implant CBCT bone morphometry, 3D vivid tooth/root visualization, cephalometric & orthodontic analysis, and deep-learning Metal Artifact Reduction (MAR).

## Key Upgrades in v3.0

### 1. Fixed Print File Generation & Live Measured Cross-Section Gallery
- **Root-Cause Fixes**:
  - Resolved `CBCTPlannerWidget.getParameterNode()` `AttributeError` in `onGenerate8x10PrintLayout` by introducing a safe `getParameterNode()` helper on `CBCTPlannerWidget`.
  - Fixed `AnatomicalLabel.color` attribute availability across all 55 ToothFairy3 and DentalSegmentator labels (`CBCTPlannerLib/core/labels.py`).
  - Fixed 32-tooth `ToothFairy3` segmentations being misclassified as 5-class masks via `resolve_combined_mask_scheme()`, ensuring all FDI teeth (`11..48`), split Left/Right IAN canals, and maxillary sinuses are preserved during arch fitting and cross-sectional morphometry.
  - Fixed VTK 2-argument `segment.GetTag(key, vtk.reference(""))` compatibility via `safe_get_segment_tag()` and fixed `segmentationNode.GetDisplayNode().GetSegmentVisibility()` in `quadrant_vr_isolation`.
  - Fixed `vtkUnsignedCharArray` to `bytes` conversion in `capture_3d_view_png_base64()`.
- **Multi-Format Printable Clinical Film (`PrintLayoutEngine.py` & `core_print_render.py`)**:
  - Exports **PDF**, **300 DPI PNG**, and **HTML** print sheets to a user-visible default folder (`~/Documents/CBCTPlanner_Prints`) with a customizable output folder selector and **Open Print Folder** button.
  - Automatically synthesizes the in-process panoramic strip when the `PanoramicVolume` MRML node has not been pre-generated, and renders a 3D vivid MIP projection card when a 3D viewport screenshot is unavailable.
  - Renders vertical green **Bone Height (`H`)** lines with end-cap ticks, **Crestal/3mm/5mm Widths (`W0`/`W3`/`W5`)**, anatomical tooth/root/jaw contours, and IAN nerve safety rings on every cross-section tile.
- **Inline Measured Cross-Section Image Gallery (Step 5)**:
  - Displays rendered cross-sectional images with `H`, `W0`, `W3`, `W5`, and nerve clearance overlays directly inside the Step 5 module panel (`Refresh Measured Images` button + tooth site selector + 3-offset `-2 / 0 / +2 mm` strip).

### 2. Radiologist & Dentist Clinical Command Bar
- Role-aware workspace selector (**Radiologist**, **Implant Dentist**, **Orthodontist / Oral Surgeon**) and 4 one-click action buttons at the top of the module:
  - `🦷 Vivid Teeth & Roots 3D`
  - `🧲 RISE-MAR (Artifact Reduction)`
  - `📐 Measure & Show Slices`
  - `🖨️ Print 8×10″ Film (PDF+PNG)`

### 3. Vivid Teeth & Roots 3D Visualization ([gaudot/SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator))
- Implements **SlicerDentalSegmentator**-inspired 3D surface and Volume Rendering presets (`CBCTPlannerLib/VRVisualization.py`):
  - **Translucent Glass-Bone Jaw (`0.25` default opacity)** with a live `0.00–1.00` slider so tooth roots, pulp chambers, and the mandibular canal shine vividly through the cortical bone.
  - **Vivid Enamel & Root Surface Colors (`1.00` opacity)** with Phong specular highlights (`Specular=0.38`, `SpecularPower=28`, `Ambient=0.22`, `Diffuse=0.78`).
  - Dedicated **"DentalSegmentator Vivid Roots (VR)"** transfer function preset ramping root dentin (`700–1350 HU`) and crown enamel (`1500–2600 HU`) while keeping trabecular/cortical jawbone semi-transparent.

### 4. Automated Dental Tools Suite ([DCBIA-OrthoLab/SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools))
- Implemented in `CBCTPlannerLib/AutomatedDentalToolsEngine.py` and exposed in **Step 5B**:
  - **ALI (Automatic Landmark Identification)**: Extracts 3D cephalometric (`Me`, `Gn`, `Pg`, `B`, `ANS`, `PNS`, `U1`, `L1`, `Go_R/L`, `Co_R/L`, `MF_R/L`) and per-tooth crown/apex landmarks, creates a Slicer `vtkMRMLMarkupsFiducialNode`, and exports `.mrk.json`.
  - **ASO (Automated Standardized Orientation)**: Computes a 4×4 rigid RAS transformation matrix aligning the midsagittal plane and occlusal plane with Pitch/Roll/Yaw readout.
  - **CLI-C (Impacted Canine & Molar Classification)**: Evaluates bilateral asymmetry, sector, angulation, and Buccal/Bicortical/Palatal position for canines (`13, 23, 33, 43`) and third molars (`18, 28, 38, 48`).
  - **VFACE (Bilateral Mandibular & Facial Asymmetry)**: Quantifies right vs. left ramus height, mandibular body length, condylar height, hemimandibular volume, and Menton midline deviation.
  - **AutoCrop3D & AutoMatrix**: Fits a padded 3D ROI box around the dental arches and builds a sagittal mirror `vtkMRMLLinearTransformNode` for symmetry overlay.

### 5. RISE-MAR Metal Artifact Reduction ([Masaaki-75/rise-mar](https://github.com/masaaki-75/rise-mar))
- Implemented in `CBCTPlannerLib/RiseMarEngine.py` following [Ma et al., IEEE TMI 2025 (arXiv:2501.15610)](https://arxiv.org/html/2501.15610v1):
  - Default model weights pre-configured to `C:\Users\hamid\Downloads\risemar_dental_preview.pkl`.
  - Clips HU values to `[-1024, 3072]` and normalizes to `[0, 1]`, runs the 2D CBAM Attention U-Net generator paired with physics-guided radial streak & photon-starvation suppression, preserves high-density metal cores, and reports before/after **Clinical Quality Assessor (`CQA`)** scores (`1–10` Likert scale).

## Running Tests
```bash
python3 -m unittest discover -s tests -v
```

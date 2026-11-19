# Code Modernization Assessment & Business Rules: `SlicerCBCTPlanner_1` → v3.0

## 1. Executive Assessment (`/modernize-assess`)
- **System**: `SlicerCBCTPlanner_1` (3D Slicer Scripted Loadable Extension in Python 3.9+ / VTK / Qt / CTK / NumPy / SciPy / PyTorch).
- **Codebase Inventory**: 22 Python modules across `CBCTPlanner/` and `CBCTPlannerLib/` (~16,384 LOC) + 8 test modules (79 unit tests).
- **Modernization Method**: **Uplift + Feature Expansion** — preserve all 79 existing behavioral guarantees (B1–B22, O1–O9, F1–F20) while fixing the 6 critical Slicer runtime/rendering defects and adding 3 major clinical subsystems (`RiseMarEngine`, `AutomatedDentalToolsEngine`, and `SlicerDentalSegmentator` Vivid 3D/VR + Radiologist/Dentist UI Dashboard).

## 2. Extracted Clinical & Engineering Rule Cards (`/modernize-extract-rules`)

### Rule Card RC-01: Honest Clinical Measurements & Dentate/Edentulous Ridge Envelope
- **Given**: A resampled cross-section `(u_coords, v_coords, mask_cs, img_cs)` orthogonal to the dental arch at `arc_mm`.
- **When**: Computing alveolar bone height `H`, crestal width `W0`, depth widths `W3`, `W5`, `W6`, `W9`, and vital nerve/sinus clearance.
- **Then**:
  1. The alveolar ridge envelope is the union of jawbone voxels (`mask_cs == jaw_id`) and any in-socket tooth root (`mask_cs == tooth_id`) across `|u| <= 6.0 mm`, with an automatic HU cortical/trabecular bone fallback (`img_cs >= 250 HU`) when segmentation masks only cover teeth or when measuring unsegmented scans. Root voxels count only apical to the jawbone's occlusal boundary, so an erupted crown is never reported as the alveolar crest. Every withheld metric records its reason on `ImplantSiteMeasurement.withheld_reason` / `CrossSectionData.withheld_reason`, which the tile surfaces as a short badge and `treatment_plan.csv` exports in the `withheld_reason` column.
  2. `render_cross_section` draws:
     - Jawbone contour (subtle translucent gold/emerald outline), tooth contour (vivid cyan/ivory outline), and canal contour (vivid red circle + safety halo).
     - Vertical Green Bone Height (`H`) line with end-caps from `crest_v` to `basal_v` / canal top.
     - Horizontal Buccolingual Width lines at Crest (`W0`, yellow), `3 mm` (`W3`, cyan), and `5 mm` (`W5`, orange/gold) with millimeter callout badges.
     - Red dashed Nerve Clearance vector when a canal/sinus is present.

### Rule Card RC-02: Print File Generation (`PDF` + `300 DPI PNG` + `HTML`) & Inline Measured Gallery
- **Given**: User selects 1 tooth, multiple teeth, or a quadrant (Q1–Q4) and clicks **Generate Print Layout** or **Measure & Show Slices**.
- **When**: `PrintLayoutEngine.generate_8x10_layout` executes.
- **Then**:
  1. Uses `self._parameterNode` (or `self.logic.getParameterNode()`), never `self.getParameterNode()` on the widget.
  2. Uses `segmentationNode.GetDisplayNode().GetSegmentVisibility(seg_id)` / `SetSegmentVisibility(seg_id, keep)` and 2-arg safe `GetTag`.
  3. Writes **three print files** (`<Scan>_Print_Layout.pdf`, `<Scan>_Print_Layout.png`, `<Scan>_Print_Layout.html`) to the user's selected/configured output directory (defaulting to `~/Documents/CBCTPlanner_Exports` with a folder picker & quick "Open Folder / Save As" button, or `out_dir` when passed programmatically).
  4. Immediately populates the **Inline Measured Cross-Section Gallery** inside the module panel AND opens the interactive high-resolution print preview dialog.

### Rule Card RC-03: Vivid Teeth & Roots 3D/VR Styling (`gaudot/SlicerDentalSegmentator`)
- **Given**: A 5-class (`DentalSegmentator`) or 77-class (`ToothFairy3`) segmentation node and/or CBCT volume node.
- **When**: Applying 3D styling or Volume Rendering presets.
- **Then**:
  1. Teeth (crowns & roots) receive vivid anatomical colors (`1.0` 3D opacity) with Phong specular surface properties (`Ambient=0.22`, `Diffuse=0.82`, `Specular=0.48`, `SpecularPower=28`) so roots and crowns are crisp and vivid.
  2. Maxilla and Mandible jawbones default to translucent 3D opacity (`0.25`) so **tooth roots and the mandibular canal are clearly visible through the bone**, controllable via a live **Bone Opacity Slider (`0.0 – 1.0`)** and **Vivid Presets** (`Vivid Teeth & Roots (Translucent Bone)`, `Roots & Nerve X-Ray Only`, `Endo Pulp & Canals`, `Opaque Bone & Teeth`).
  3. VR preset `"DentalSegmentator Vivid Teeth & Roots"` suppresses soft tissue and makes trabecular bone a subtle veil (`alpha <= 0.10`) while ramping root dentin (`950–1450 HU`) and crown enamel (`1500–2800 HU`) to high opacity (`0.75–1.0`).

### Rule Card RC-04: RISE-MAR Metal Artifact Reduction (`Masaaki-75/rise-mar`)
- **Given**: A CBCT volume affected by metallic dental restorations, implants, or brackets, and weights at `C:\Users\hamid\Downloads\risemar_dental_preview.pkl`.
- **When**: The user enables/runs **RISE-MAR Metal Artifact Reduction**.
- **Then**:
  1. Clips HU to `[-1024, 3072]` and normalizes to `[0, 1]` per `Ma et al., IEEE TMI 2025`.
  2. Loads `risemar_dental_preview.pkl` (supporting pickled `nn.Module`, `state_dict` into the CBAM Attention U-Net generator, or dynamic layer reconstruction, with automatic sinogram-streak/frequency hybrid fallback when weights are absent in CI).
  3. Restores streak-corrupted peri-implant/interproximal regions while preserving true high-density metallic cores (`> 2500 HU`), computes pre/post Clinical Quality Assessor (`CQA`, 1–10 scale) scores, and outputs `<Volume>_RISE_MAR` (or updates in-place).

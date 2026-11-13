# SlicerCBCTPlanner

Dedicated, self-contained **3D Slicer** scripted extension for dental CBCT visualization, Curved Planar Reformation (CPR) panoramic radiographs, multi-slice cross-sectional lightbox navigation, and clinically honest implant site morphometry.

## 5-Step Clinical Workflow
1. **Step 1 — Load CBCT, Segmentation & 3D / VR View**
   - Select an input CBCT scalar volume and an optional segmentation (`DentalSegmentator` 5-class or `ToothFairy3` multi-label).
   - Switch between **3-Slice Lightbox (`-2 / 0 / +2 mm`)** (`CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID = 503`: Panoramic, Axial, and 3D on the top row; `CS-`, `CS0`, and `CS+` side-by-side on the bottom row), **Single CS Layout** (`CUSTOM_DENTAL_LAYOUT_ID = 502`), and **Standard MPR**.
   - Optional **3-Point Occlusal Plane** tool: place 3 fiducials (right molar, incisors, left molar) to compute the true occlusal plane normal and optionally harden an alignment transform.
   - **Volume Rendering (VR) & DentalSegmentator 3D Model**: [gaudot/SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) 5-class color/opacity styling (`#E3DD90` 0.45, `#D4A1E6` 0.45, `#DC9565` 1.0, `#EBDFB4` 1.0, `#D8654F` 1.0) and debounced (`150 ms`) CBCT Volume Rendering threshold shift.
2. **Step 2 — Dental Arch Curve (`Right -> Left` in RAS)**
   - Curves are oriented from **Patient Right (`+X` in RAS, FDI `18`/`48`) to Patient Left (`-X` in RAS, FDI `28`/`38`)** so $\mathbf{N} = \mathbf{T} \times \mathbf{Up}$ points **Buccal** everywhere and Cross-Section views display **Buccal on the left, Lingual on the right, and Superior up**.
   - **Auto-Fit Arch from Segmentation** fits a spline to segmented teeth/jawbone and asks for confirmation before overwriting any user-edited curve (`CBCTPlanner.UserEdited`).
   - When no segmentation is loaded, no points are invented: clicking **Place / Draw Arch Points on Axial** enters Slicer's Markups Place Mode.
3. **Step 3 — Panoramic Visualization (Curved Planar Reformat)**
   - Uses a right-handed ($\det = +1$) `vtkOrientedGridTransform` straightening matrix ([PerkLab/SlicerSandbox CurvedPlanarReformat](https://github.com/PerkLab/SlicerSandbox/blob/master/CurvedPlanarReformat/CurvedPlanarReformat.py)) and projects the straightened slab (`Mean`, `MIP`, or `MinIP`) with **Patient Right on screen left** (`pano_x(arc_mm) = -arc_mm`).
   - Runs via non-blocking asynchronous CLI with progress bar and cancel button; dragging curve control points triggers a debounced low-resolution preview and full-resolution update on release.
4. **Step 4 — Multi-Slice Cross-Sectional Lightbox (`-2 / 0 / +2 mm`) & $\pm 30^\circ$ Tilt Control**
   - Backed by a single-source-of-truth MRML `ParameterNode` observed via `VTKObservationMixin`.
   - **3-Slice Cross-Sectional Lightbox**: Simultaneously drives `YellowMinus` (`CS-`, amber `#F59E0B`, $-\Delta s$), `Yellow` (`CS0`, cyan `#06B6D4`, $0\text{ mm}$), and `YellowPlus` (`CS+`, emerald `#10B981`, $+\Delta s$, default $\Delta s = 2.0\text{ mm}$) and displays matching color-coded triplet cut lines on both the **Axial (`Red`)** and **Panoramic (`Green`)** views.
   - **$\pm 30^\circ$ Cross-Section Tilt Control**: Rotates the cross-section vertical axis $\mathbf{B}_{\text{tilted}} = \cos\theta\,\mathbf{B} + \sin\theta\,\mathbf{T}$ and slice normal $\mathbf{T}_{\text{tilted}} = \cos\theta\,\mathbf{T} - \sin\theta\,\mathbf{B}$ around the Buccal-Lingual normal $\mathbf{N}$ ($\det = +1$) and tilts the Panoramic cursor lines to match.
   - **Continuous `1 mm` Arc-Length Slider**: Drag the cyan line in Axial or Panoramic, nudge with `Left`/`Right` arrows ($\pm 1\text{ mm}$), jump with `PgUp`/`PgDn` ($\pm 5\text{ mm}$), or lock to any FDI tooth.
5. **Step 5 — Odontogram, Honest Morphometry & Report**
   - Clickable 32-tooth **Odontogram** (Q1/Q2 upper row, Q4/Q3 lower row): click to jump to a tooth; `Ctrl+Click` to toggle multi-selection with incremental per-tooth analysis.
   - **Zero Fabricated Numbers**: When a segmentation is selected, `MeasurementEngine` measures bone height, buccolingual widths, nerve clearance, and HU directly from Slicer volume/label arrays. When unsegmented, metrics report `"not measured"` unless the clinician places manual Markups rulers (`Tooth_<FDI>_HeightRuler`, `Tooth_<FDI>_WidthRuler`, `Tooth_<FDI>_NerveRuler`).
   - Exports HTML reports (with HTML-escaped fields and embedded Panoramic, Axial, `CS-`, `CS0`, and `CS+` screenshots) and CSV/JSON files (with flattened `pos_r_mm`, `pos_a_mm`, `pos_s_mm` columns).

## Running Tests
```bash
# Pure NumPy unit tests (no Slicer required):
python3 -m unittest discover -s tests -v

# Slicer headless smoke test:
Slicer --no-main-window --python-script tests/smoke.py
```

## Known Limitations
- Automatic arch curve fitting and automated bone/nerve morphometry require a loaded segmentation (`DentalSegmentator` 5-class or `ToothFairy3` multi-label); on unsegmented scans, arch curves and site rulers must be placed manually using the provided Markups tools.
- Full-resolution Curved Planar Reformation relies on 3D Slicer's built-in `ResampleScalarVectorDWIVolume` CLI module.

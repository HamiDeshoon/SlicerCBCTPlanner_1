# SlicerCBCTPlanner

Dedicated, self-contained **3D Slicer** extension for dental CBCT visualization, panoramic reformation, and cross-sectional implant planning.

## What Was Cleaned Up
This repository was separated from `CBCT_Planner` to provide a pure 3D Slicer extension without redundant web studio, FastAPI, Playwright E2E, scratch test scripts, or hardcoded local paths.

## Key Features
1. **Volume Rendering (VR) & DentalSegmentator 3D Model (`VRVisualization.py`)**
   - Ported 5-class anatomical surface styling from [gaudot/SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) (`Maxilla & Upper Skull` `#E3DD90` 0.45, `Mandible` `#D4A1E6` 0.45, `Upper Teeth` `#DC9565` 1.0, `Lower Teeth` `#EBDFB4` 1.0, `Mandibular canal` `#D8654F` 1.0) with white 3D background and hidden bounding box/axis labels.
   - Interactive CBCT Volume Rendering (`CT-Dental-CBCT`, `CT-Bone`, `CT-AAA`, `CT-MIP`, `CT-Cardiac`) with live **HU Threshold Shift** slider and **ROI Cropping**.
2. **Dental Arch Curve Visualization (`PanoramicCurveLogic.py`)**
   - Auto-fitted Upper and Lower `vtkMRMLMarkupsCurveNode` cardinal splines with toggleable 2D/3D visibility and interactive control-point handle editing.
3. **Panoramic Visualization (`PanoramicCurveLogic.py`)**
   - Curved Planar Reformation (CPR) using `vtkOrientedGridTransform` and upright Parallel Transport Frames, producing `<Vol>_Straightened_CPR` and `<Vol>_Panoramic` (Mean / MIP / MinIP) displayed in Slicer's Green slice view.
4. **Cross-Sectional Slice Navigation & Axial Indicators (`CrossSectionNavigator.py`)**
   - Synchronized **Yellow** (Cross-Section: Buccal Left / Lingual Right / Superior Up), **Green** (Tangential / Panoramic), and **Red** (Axial) views.
   - Real-time **Axial Cross-Section Slice Indicators**: amber multi-slice tick fan (`<Vol>_CrossSectionTicks`) plus a bright cyan active cut line (`<Vol>_ActiveCrossSectionLine`) labeled `CS #X / N (arc mm)` directly on the Axial image.

## Installation in 3D Slicer
1. Open **3D Slicer** (5.6+ / 5.8+).
2. Go to **Edit → Application Settings → Modules → Additional module paths**.
3. Add the `CBCTPlanner/` folder from this repository and restart 3D Slicer.
4. Open **Dental → CBCT Dental Planner**.

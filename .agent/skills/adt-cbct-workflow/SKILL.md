---
name: adt-cbct-workflow
description: Use when running or automating SlicerAutomatedDentalTools-compatible workflows (`ALI`, `ASO`, `CLI-C`, `DOCShapeAXI`, `VFACE`, `AutoCrop3D`) inside SlicerCBCTPlanner without external Conda/WSL2 dependencies.
---

# AutomatedDentalTools (ADT) Native & Hybrid CBCT Workflow

This skill documents the upgraded `CBCTPlannerLib.adt` package (`ALI`, `ASO`, `CLI-C`, `DOCShapeAXI`, `VFACE`, `AutoCrop3D`) integrated natively inside `SlicerCBCTPlanner` using Slicer's built-in `vtk`, `skimage`, `scipy`, `SimpleITK`, `cc3d`, and `torch`/`torchvision`/`timm` runtime.

## Quick Entrypoints (`CBCTPlannerLib.adt` / `AutomatedDentalToolsEngine`)

| Module | Capability | Primary Function / Method | Explainability / Export |
| :--- | :--- | :--- | :--- |
| **ALI** | 3D Skeletal, Cephalometric, Mental Foramen, Crown/Apex Landmarks | `identify_ali_landmarks(vol, mask, is_5class)` | `export_ali_landmarks_mrk_json` -> Slicer `.mrk.json` |
| **ASO** | Rigid Midsagittal + Frankfort Horizontal / Occlusal Orientation (+ Kabsch SVD & outlier filtering) | `compute_aso_standardized_orientation(vol, mask, reference_landmarks_ras=...)` | `4x4` RAS matrix + `landmark_guided_aso_frame` |
| **CLI-C** | Cortex-aware Impacted Canine/Molar (`Buccal=1, Bicortical=2, Palatal=3`) + 3D EDT root proximity | `analyze_impacted_teeth_clic(vol, mask, build_labelmap=True)` + optional `run_clic_maskrcnn_inference` | `last_clic_labelmap` (`{1, 2, 3}`) |
| **DOCShapeAXI** | Airway Obstruction / Condyle / Alveolar Cleft Classification & Regression + Vertex Saliency | `run_doc_shapeaxi(data_type, task, volume, mask, surface_path, out_vtk_path, checkpoint_path)` | `.vtk` with `ShapeAXI_Saliency` & `grad_cam_target_class_k` (`ColdToHotRainbow`) |
| **VFACE / Crop** | Bilateral Ramus/Body/Chin Asymmetry & AutoCrop3D ROI & Sagittal Mirror | `compute_vface_asymmetry_analysis`, `compute_autocrop_roi_bounds_ras`, `build_sagittal_mirror_matrix_4x4` | `VTK/MRML ROI` & `4x4` reflection |

## Operational Rules
1. Always standardize volumes cleanly via `core/orientation.py` (using `landmark_guided_aso_frame` when bilateral FDI tags are absent in 5-class masks) before fitting arches or cutting cross-sections.
2. When running `DOCShapeAXI` without external PyTorch3D weights, native morphometry + dihedral curvature / pharyngeal CSA saliency runs deterministically offline and exports VTK scalar arrays (`ShapeAXI_Saliency`). When a custom PyTorch `.ckpt`/`.pth` checkpoint is supplied, multi-view depth projection + Grad-CAM back-projects activations directly onto the VTK mesh vertices.
3. For edentulous or partially edentulous jaws, `extract_jaw_arch_anchor_bounds` uses `ALI` Gonion/Pg/PNS bounds inside `core/arch_detector.py` (`_extract_jawbone_crest_points`) to prevent arch splines from climbing ascending mandibular rami or zygomatic buttresses.

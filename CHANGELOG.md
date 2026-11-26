# Changelog

## [3.3.0] — 2026-11-26

### Packaging & Versioning (`P0`)
- Established single authoritative version source `CBCTPlanner/Resources/VERSION` (`3.3.0`), consumed dynamically by `CBCTPlannerLib/Utils.py` (`SOFTWARE_VERSION`), `CMakeLists.txt` (`EXTENSION_VERSION`), and `pyproject.toml` (`dynamic = ["version"]`, `package-dir = {"" = "CBCTPlanner"}`).
- Completed `CBCTPlanner/CMakeLists.txt` `MODULE_PYTHON_SCRIPTS` and `MODULE_PYTHON_RESOURCES` so all 40 Python modules (`WorkflowUIHelpers.py`, `adt/*`, `core/*`, `EndodonticEngine.py`) and resource manifests deploy together cleanly.
- Added `tests/test_packaging_completeness.py` verifying zero unpackaged files and isolated staged execution without repository root on `PYTHONPATH`.

### Architecture, Contracts & Performance (`P1`)
- Added `CBCTPlannerLib/core/contracts.py` (`UNIT_MM`, `UNIT_DEG`, `UNIT_HU`, normalized 6-value provenance taxonomy, `StructuredMeasurementEntry`, `UncertaintyClearance`, `StructuredReportEnvelope`).
- Extracted pure non-Qt case override/JSONL persistence into `CBCTPlannerLib/core/case_state.py`.
- Added `OPT-01` CRC32-keyed bounded LRU caches (`_SLAB_BASIS_CACHE`, `_SLAB_PROJECTION_CACHE`) in `PanoramicCurveLogic.py` (`pano_RaySum_cold`: 353.95 ms -> `pano_RaySum_warm`: 1.18 ms; copy-isolated with strict invalidation across mode/offset/affine/voxel edits).
- Added `OPT-03` direct PIL image hand-off for panoramic strip and VR capture during 8×10 film composition in `PrintLayoutEngine.py`.
- Added reproducible multi-pipeline benchmark `tests/bench_suite.py`.

### Clinical Geometry, Dental Arch & Panoramic CPR (`v3.3`)
- Fixed dentate `W0` socket bridging at thin coronal buccal rims (`CBCTPlannerLib/core/measurements.py`, `socket_rims` fallback), achieving 32/32 PASS on `verify_real_ct.py`.
- Added rotation-minimizing parallel-transport frame propagation (`compute_parallel_transport_frames`) in `CBCTPlannerLib/core/arch_detector.py` and `PanoramicCurveLogic.py` plus sparse-arch FDI outlier rejection (`len >= 5`), eliminating 180-deg frame flips on partial hemi-arches and curved ridges.
- Added half-voxel FOV edge clamping in `PanoramicCurveLogic.py` (`_sample_slab_layer`, `sample_panoramic_slab_array`) to avoid dark 1px border blending while retaining `-1000.0 HU` air outside FOV.
- Integrated native SlicerAutomatedDentalTools (`CBCTPlannerLib/adt/`: ALI, ASO, CLI-C, VFACE, DOCShapeAXI) and Step 3–5/7 clinical geometry invariants (`T`-junction, smooth `W0` angle, periodic depth key).

### Multi-Root Endodontics (`EndodonticEngine.py`)
- Replaced Z-sorted constant-`0.5 mm` tracing with physical ROI extraction, optional external checkpoint / conservatively labeled interior lumen extraction (`SOURCE_ESTIMATED`), anisotropic 3D EDT radius & outer-wall thickness profiling, 26-neighbor weighted geodesic skeleton graph with 3D spherical apical peak NMS, furcation/bifurcation detection, Schneider 3D curvature, and access cavity trajectory optimization (`CLINICAL_ENDO_STATUS`).
- Bridged cleanly into `CanalTracerLogic` and Step 6 UI (`btnAnalyzeEndoTooth`, `btnExportEndoJson`, ParameterNode `"EndoAnalyses"` persistence).

### Capsule Implant Geometry, Uncertainty & Structured Reporting (`ImplantPlanner.py`)
- Replaced coarse cylinder point clouds in `ImplantPlannerLogic` with closed-form segment-segment closest points, tapered capsule + hemispherical apical dome SDFs (`distance_points_to_capsule_surface`), vectorized VTK PolyData construction, neighboring tooth collision checks, IAN/incisive/lingual canal & sinus clearances, cortical containment/perforation checks, and `UncertaintyClearance` (reporting explicit quantization/missing-prior limitations without fabricated calibration).
- Reduced `implant_verify_and_stl` benchmark time from `301.39 ms` / `4972.8 KiB` to `148.45 ms` / `2533.2 KiB`.
- Enriched `treatment_plan.json` and `*_8x10_Report.json` with `StructuredReportEnvelope` provenance & unit metadata.

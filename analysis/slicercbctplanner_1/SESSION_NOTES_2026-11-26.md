# Session Notes — 2026-11-26 (`v3.3.0` Hardening & Architecture Modernization)

## Summary
Completed post-v3.2 clinical geometry verification (`32/32` on `ToothFairy3S_0021`), fixed critical CMake deployment packaging (`P0`), modernized core architecture with explicit units/provenance contracts and `OPT-01`/`OPT-03` optimizations (`P1`), implemented a true multi-root topology `EndodonticEngine`, upgraded `ImplantPlanner` to closed-form capsule geometry with multi-structure SDF & uncertainty reporting, and added structured JSON provenance reports (`P2`).

## External Open-Source Reference Comparison (Zero Blind Imports)
Inspected locally / via documentation (none committed or imported wholesale):
1. `gomesgustavoo/dentistry-implant-planner` (MIT): Inspired exact closed-form segment-segment capsule distances and explicit uncertainty limitation reporting (without importing arbitrary empirical calibration numbers).
2. `dnt102/Slicer-FSP`: Studied Slicer scripted surgical-guide/implant parameter workflows.
3. `ZhyBrian/SlicerEndoPlanner` (Apache-2.0): Studied multi-root graph canal reconstruction and coronal access corridor optimization, re-implemented cleanly in pure NumPy + SciPy (`EndodonticEngine.py`) without requiring OR-Tools/Numba/PyTorch3D.
4. `DCBIA-OrthoLab/SlicerAutomatedDentalTools` (Apache-2.0): Wrapped compatible native ALI, ASO, CLI-C, VFACE, DOCShapeAXI pipelines under `CBCTPlannerLib/adt/` with pure-Python VTK fallbacks.
5. `gaudot/SlicerDentalSegmentator` (Apache-2.0): Reviewed optional checkpoint/weights isolation patterns.
6. `ZoliQua/Dental-CBCT-Viewer` (MIT): Inspired half-voxel edge boundary clamping in panoramic sampling and pure `/core` math isolation.
7. `SlicerIGT/SlicerIGT` (BSD-style Slicer): Informed transform-aware implicit distance patterns.

## Verification Results (All Run on 2026-11-26 After Final Code Edits)
- `python -m ruff check CBCTPlanner tests analysis/slicercbctplanner_1/verify_real_ct.py`: **All checks passed**
- `python -m unittest discover -s tests` (Host Python 3.12): **265/265 passed in 24.211s** (`0` failures, `0` skipped)
- `"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe" -m unittest discover -s tests`: **265/265 passed in 28.419s** (`0` failures, `0` skipped)
- `"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe" analysis/slicercbctplanner_1/verify_real_ct.py`: **32/32 passed** (`205` lower rows; visually verified `ToothFairy3S_0021_8x10.png` and `ToothFairy3S_0021_cs_center.png`)
- `python analysis/mcp_slicer_check/verify_mcp_slicer.py`: **11/11 passed** on live 3D Slicer 5.12.3 rev 34627
- `"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe" tests/test_packaging_completeness.py`: **3/3 passed**, including isolated staged install execution (`setuptools` wheel metadata `slicer-cbct-planner-3.3.0.dist-info` also verified)
- `"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe" tests/bench_suite.py` (`96x160x160`, `0.5 mm`, 18 structures):
  - `arch_fit_lower`: `2624.21 ms` cold (first-call SciPy import ~1.0s; hot path ~29 ms)
  - `pano_RaySum_cold` / `warm`: `353.95 ms` -> `1.18 ms` (300x speedup; MIP/MinIP/Mean cold ~`124-130 ms` sharing basis cache)
  - `implant_verify_and_stl`: `301.39 ms` (`4972.8 KiB`) -> **`148.45 ms` (`2533.2 KiB`)** (~2.0x speedup + 49% lower peak memory)
  - `endo_multiroot_graph_46`: `160.65 ms` (`4800.7 KiB`)
  - `cross_section_21_sites`: `950.92 ms`; `print_8x10_q4`: `1663.86 ms`

## Build & Deployment Note
Host environment lacks `cmake`, `ninja`, and C++/Qt Slicer superbuild tools on `PATH`, so a native CMake superbuild compile/package was not executed on this machine; all CMake file/resource declarations were verified by `tests/test_packaging_completeness.py` copying exact `MODULE_PYTHON_SCRIPTS` and `MODULE_PYTHON_RESOURCES` into a standalone staged `qt-scripted-modules/` directory and executing core workflows in an isolated subprocess inside both host Python and `PythonSlicer.exe`.

Workspace changes remain **uncommitted** awaiting user confirmation before any `git commit` or `git push`.

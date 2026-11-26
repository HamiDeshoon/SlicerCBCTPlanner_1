# v3.3 Task Checklist

- [ ] **T1**: Implement jaw-aware `v_coords` framing & internal cortical `H` boundary in `CBCTPlannerLib/core/measurements.py`
- [ ] **T2**: Enforce strict `T`-junction (`H` starts on `W0` midpoint, no `+` cross) + `W0` inter-slice angle smoothing in `measurements.py` & `CBCTPlanner.py` Pre-Print Editor
- [ ] **T3**: Implement periodic single `W3` / `W5` corner readout & connected `W0` / `IAN` leader lines in `core_print_render.py`, `PrintLayoutEngine.py`, and `CBCTPlanner.py`
- [ ] **T4**: Build modular `CBCTPlannerLib/adt/` package (`geometry.py`, `ali.py`, `aso.py`, `clic.py`, `doc_shapeaxi.py`, `vface.py`) incorporating upstream `SlicerAutomatedDentalTools` (`DOCShapeAXI`, `CLIC`, `ASO`, `ALI`) & fast C-level fallbacks
- [ ] **T5**: Wire upgraded `adt` into `AutomatedDentalToolsEngine.py`, `core/orientation.py`, `core/arch_detector.py`, `Phase7Tools.py`, `CBCTPlanner.py` Step 7 UI (including DOCShapeAXI), and package `__init__.py`
- [ ] **T6**: Add unit & real-CT regression tests, sync documentation & skill card, and run 100% passing verification (`ruff`, `unittest`, `verify_real_ct.py`, `verify_mcp_slicer.py`, live Slicer 5.12.3 & PNG inspection)

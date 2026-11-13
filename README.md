# SlicerCBCTPlanner (v2.0.0)

> **CLINICAL DISCLAIMER — NOT FOR DIAGNOSTIC USE / PLANNING AID ONLY**
> `SlicerCBCTPlanner` is an open-source research and pre-surgical planning aid for 3D Slicer. It is **not** a certified medical device and must **not** be used for primary clinical diagnosis or unverified surgical execution. All CBCT gray-value intensity figures and Misch bone density classifications (`D1–D5`) are **approximate** because Cone-Beam CT voxel gray values are not calibrated Hounsfield Units (HU). Any AI-suggested implant dimensions are labeled as unconfirmed suggestions and require explicit clinician confirmation before entering exported reports.

---

## Overview & 6-Step Workflow

`SlicerCBCTPlanner` is a standalone 3D Slicer scripted loadable extension (`CBCTPlanner`) providing end-to-end dental CBCT visualization, Curved Planar Reformation (CPR) panoramic generation, synchronized multi-slice cross-sectional lightbox navigation, mandibular canal tracing, generic cylinder implant placement, and clinical reporting.

1. **Step 1 — Load CBCT, Segmentation & 3D / VR View**:
   - Select an input CBCT scalar volume and optional segmentation (`DentalSegmentator` 5-class or `ToothFairy3` FDI labels).
   - Switch between the **3-Slice Cross-Sectional Lightbox** (`CS-`, `CS0`, `CS+` on bottom; `Pano`, `Axial`, `3D` on top), **Single Cross-Section Dental Layout**, and **Standard MPR**.
   - Define or auto-fit the **Occlusal Plane** (3-point fiducials or automated SVD fit to segmented tooth cusps/incisal edges, **F6**) and apply a **non-destructive parent transform** (`<Vol>_OcclusalAlignmentTransform`, **B3**) keeping the volume, segmentation, curves, fiducials, and rulers co-registered, with one-click reset.
   - Apply **DentalSegmentator 3D surface styling** + **2D tooth outlines & `#FDI` overlays** (**F4**), or enable **3D Volume Rendering** with preset re-application (**B8**) and automatic percentile-based transfer functions when voxel values are non-HU (`min > -500` or `max > 10000`).
2. **Step 2 — Dental Arch Curve (`Right -> Left` in RAS)**:
   - Choose **Per-Jaw (`Lower` / `Upper`)** or **Shared Occlusal Arch** (**F3**, with automatic migration of legacy per-jaw curves).
   - Auto-fit a voxel-grid-downsampled principal horseshoe spline (**B20**) to segmented teeth (never falls back to fitting the whole skull), or enter interactive **Markups Place Mode** when unsegmented. Protects user-edited curves (`CBCTPlanner.UserEdited = "1"`) with confirmation before overwriting (**B14**).
3. **Step 3 — In-Process Panoramic CPR (`Mean` / `MIP` / `MinIP` / `RaySum`)**:
   - Generates Curved Planar Reformatted (CPR) panoramic radiographs **in-process** using `scipy.ndimage.map_coordinates(..., order=1)` across slab layers (**O1**) with right-handed displacement frames (`det([-N, B, T]) = +1`, **B5**) and fixed `<= 1.5 mm` frame sampling (**B6**).
   - Configurable slab thickness, height, and buccolingual offset (**F3**), chunked non-blocking `QTimer` progress/cancel, and zoom/pan preservation across preview/release refreshes (**B10**).
4. **Step 4 — Multi-Slice Cross-Sectional Lightbox & $\pm 30^\circ$ Tilt**:
   - Draggable cross-section lines on Axial (`Red`) and Panoramic (`Green`) views with sub-sample line-segment curve projection (**B13**) and 30 ms coalesced updates (**O2**).
   - **$\pm 30^\circ$ Cross-Section Tilt** around the Buccal-Lingual normal $\mathbf{N}$ (`det = +1`) and **Dynamic Lightbox** (`3, 5, 7, ...` slices, **F5**) with explicit **Slab Reconstruction** checkbox and mode (`Mean` default, `MIP`, `MinIP`, **B12**).
   - Preserves user Axial (`Red`) slice scrolling during cursor moves and syncs Axial Z only on explicit tooth jumps (**B11**).
   - Slice-view scoped keyboard shortcuts (`Left`/`Right` $\pm 1\text{ mm}$, `PgDn`/`PgUp` $\pm 5\text{ mm}$, `Ctrl+Z`/`Ctrl+Y` Undo/Redo, **B7, F13**).
5. **Step 5 — Odontogram, Site Morphometry, Implants & Clinical Report**:
   - **Odontogram** (**B17**): Plain click jumps the cursor to that FDI tooth without altering the plan; `Ctrl+Click` or **Toggle Active Tooth in Plan** adds/removes the tooth; **Lock to Tooth FDI** locks the cursor and releases on manual cursor drag.
   - **FDI-to-Arc Mapping** (**B2**): Projects segmented tooth centroids onto the arch curve (`source="segmentation"`), interpolates missing teeth between segmented neighbors, or falls back to standard adult mesiodistal crown widths (`source="approximate_width_table"`, flagged `"approximate"`) with support for calibrated tooth markers.
   - **Arch-Frame Cross-Section Morphometry** (**B4**): Resamples the cross-section plane perpendicular to the arch tangent along $(\mathbf{N}, \mathbf{B})$ in world space (invariant to flipped array axis `K` or oblique volume tilt).
   - **Honest Manual Rulers** (**B1**): Pre-created template rulers are flagged `CBCTPlanner.UserModified = "0"` and return `None` (`"not measured"`) until the user moves their endpoints or places them interactively.
   - **Implant Placement & Verification** (**F2, F11, F12, F19**): Extensible generic JSON cylinder library (`Resources/generic_implant_library.json`, zero manufacturer data), cross-section tilt/placement, live clearance to Mandibular Canal and Maxillary Sinus, inter-implant parallelism angles, approximate peri-implant gray-value & NNT concentric shell summary, **Implant-Centric View**, `.stl` + `.json` export, and clinician confirmation gate for AI suggestions.
   - **Clinical Reports** (**B21, B22, F10**): Exports printable **HTML** and **PDF** reports (with `vtkWindowToImageFilter` renderWindow screenshots, dynamic lightbox step labels, opt-in patient metadata anonymized by default, software version, and disclaimer) plus **CSV** (numeric columns empty `""` when unmeasured with a dedicated `measurement_source` column) and **JSON**.
6. **Step 6 — Mandibular Canal (IAN), Pharyngeal Airway & TMJ Views**:
   - **Mandibular Canal & Root-Canal Tracer** (**F1**): Automated centerline + local EDT radius extraction from class 5 / IAN masks, manual trace mode, 2D/3D tube display, configurable `2.0 mm` safety margin, and `OnDemand3D`-style proximity warnings.
   - **Pharyngeal Airway Analysis** (**F7**): Intensity threshold + 3D connected-component seeded region growing reporting total airway volume ($\text{mm}^3$ / $\text{cm}^3$) and minimum cross-sectional area ($\text{mm}^2$).
   - **TMJ Views** (**F8**): Bilateral Left and Right mandibular condyle sagittal and coronal slice presets.

---

## Running Tests & Smoke Verification

### Standalone Python Unit & Integration Suite (No 3D Slicer Required)
```bash
python3 -m unittest discover -s tests -v
python3 tests/smoke.py
```

### Live 3D Slicer Headless Smoke Test
```bash
Slicer --no-main-window --python-script tests/smoke.py
```

---

## Known Limitations
1. **Uncalibrated CBCT Gray Values**: Cone-Beam CT voxel intensities vary across scanners, FOVs, kVp/mAs settings, and scatter/beam-hardening artifacts. All reported gray values and Misch bone density classes (`D1–D5`) are explicitly labeled **approximate** and must not be treated as calibrated CT Hounsfield Units.
2. **Whole-Skull Arch Fallback Removed (B20)**: When a segmentation contains only full-cranium or full-mandible bone labels without segmented teeth, `Auto-Fit Arch` intentionally declines to fit to the whole skull/ramus and switches to interactive Markups curve placement.
3. **Phase 7 (P3 Features F14–F20)**: Surgical guide sleeve CAD (`F14`), DICOM SEG/SR (`F15`), abutment/crown library (`F16`), 3D cephalometry (`F17`), full `SlicerAutomatedDentalTools` execution wrappers (`F18`), and headless CLI batch runner (`F20`) are scoped for Phase 7 after Phases 1–6 merge.

# SlicerCBCTPlanner v3.1 — Dental Radiology, Implantology & AI Suite for 3D Slicer

**SlicerCBCTPlanner v3.1** is a full-stack 3D Slicer scripted extension for automated pre-implant CBCT bone morphometry, 3D vivid tooth/root visualization, cephalometric & orthodontic analysis, and deep-learning Metal Artifact Reduction (MAR).

## Key Features & Upgrades (v3.0 → v3.1)

### 1. Fixed Print File Generation & Live Measured Cross-Section Gallery
- **Root-Cause Fixes**:
  - Resolved `CBCTPlannerWidget.getParameterNode()` `AttributeError` in `onGenerate8x10PrintLayout` by introducing a safe `getParameterNode()` helper on `CBCTPlannerWidget`.
  - Fixed `AnatomicalLabel.color` attribute availability across all 77 ToothFairy3 and DentalSegmentator labels (`CBCTPlannerLib/core/labels.py`).
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
- Role-aware workspace selector (**Full Clinical Suite**, **Radiologist**, **Implant & Surgical Dentist**, **Ortho & Craniofacial**) and 4 one-click action buttons at the top of the module:
  - `🦷 Vivid Teeth & Roots 3D`
  - `🧲 RISE-MAR (Artifact Reduction)`
  - `📐 Measure & Show Slices`
  - `🖨️ Print 8×10″ Film (PDF+PNG)`

### 3. Vivid Teeth & Roots 3D Visualization ([gaudot/SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator))
- Implements **SlicerDentalSegmentator**-inspired 3D surface and Volume Rendering presets (`CBCTPlannerLib/VRVisualization.py`):
  - **Translucent Glass-Bone Jaw (`0.25` default opacity)** with a live `0.00–1.00` slider so tooth roots, pulp chambers, and the mandibular canal shine vividly through the cortical bone.
  - **Vivid Enamel & Root Surface Colors (`1.00` opacity)** with Phong specular VR shading (`Ambient=0.25`, `Diffuse=0.88`, `Specular=0.45`, `SpecularPower=26.0` for Vivid Roots VR; `Specular=0.35`, `SpecularPower=18.0` for Teeth & Bone).
  - Dedicated **"DentalSegmentator Vivid Roots (VR)"** transfer function preset ramping root dentin (`700–1350 HU`) and crown enamel (`1500–2600 HU`) while keeping trabecular/cortical jawbone semi-transparent.

### 4. Automated Dental Tools Suite ([DCBIA-OrthoLab/SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools))
- Implemented in `CBCTPlannerLib/AutomatedDentalToolsEngine.py` and exposed in **Step 7** (`Step 7: 3D Cephalometry & Automated Dental Tools`):
  - **ALI (Automatic Landmark Identification)**: Extracts 3D cephalometric (`Me`, `Gn`, `Pg`, `B`, `ANS`, `PNS`, `U1`, `L1`, `Go_R/L`, `Co_R/L`, `MF_R/L`) and per-tooth crown/apex landmarks, creates a Slicer `vtkMRMLMarkupsFiducialNode`, and exports `.mrk.json`.
  - **ASO (Automated Standardized Orientation)**: Computes a 4×4 rigid RAS transformation matrix aligning the midsagittal plane and occlusal plane with Pitch/Roll/Yaw readout.
  - **CLI-C (Impacted Canine & Molar Classification)**: Evaluates bilateral asymmetry, sector, angulation, and Buccal/Bicortical/Palatal position for canines (`13, 23, 33, 43`) and third molars (`18, 28, 38, 48`).
  - **VFACE (Bilateral Mandibular & Facial Asymmetry)**: Quantifies right vs. left ramus height, mandibular body length, condylar height, hemimandibular volume, and Menton midline deviation.
  - **AutoCrop3D & AutoMatrix**: Fits a padded 3D ROI box around the dental arches and builds a sagittal mirror `vtkMRMLLinearTransformNode` for symmetry overlay.

### 5. RISE-MAR Metal Artifact Reduction ([Masaaki-75/rise-mar](https://github.com/masaaki-75/rise-mar))
- Implemented in `CBCTPlannerLib/RiseMarEngine.py` following [Ma et al., IEEE TMI 2025 (arXiv:2501.15610)](https://arxiv.org/html/2501.15610v1):
  - Default model weights pre-configured to a user-configured `.pkl` / `.pth` checkpoint (stored in Slicer settings).
  - Clips HU values to `[-1024, 3072]` and normalizes to `[0, 1]`, runs the 2D CBAM Attention U-Net generator paired with physics-guided radial streak & photon-starvation suppression, preserves high-density metal cores, and reports before/after **Clinical Quality Assessor (`CQA`)** scores (`1–10` Likert scale).

### 6. Step 8: AI Agent Review, Annotations & Tangential Sections

A new wizard step (`Step 8`, after the canal/guide and cephalometry steps) turns the measured case into
something a remote review agent can work with, and adds a ridge-parallel section view.

**Review agent (URL + API key)**
- Any **OpenAI-compatible** path-based endpoint works: `https://api.openai.com/v1`, vLLM, Ollama's OpenAI shim
  (with a placeholder key), or an internal proxy. Enter an `http(s)://` base URL (a full `.../chat/completions` URL
  without query parameters is also accepted), the API key, the model/deployment name, and a timeout (query-string
  gateways such as Azure `?api-version=...` are tracked in `BL-17`).
- The request carries the *clinical context* (`core/agent_review.py`): jaw and arch geometry, every planned site with
  `H / W0 / W3 / W5`, IAN clearance, HU/density class, suggested implant, measurement provenance and withheld
  reasons, the automated warnings, the existing annotations, and an optional clinician question — plus up to 8
  rendered cross-section/tangential images as vision inputs.
- The agent answers with one JSON object: `summary`, `comments[]`, and `actions[]`. Actions are validated against a
  fixed whitelist (`add_annotation`, `update_annotation`, `remove_annotation`, `add_tooth`, `remove_tooth`,
  `set_implant_diameter`, `flag_site`) and **nothing is applied until you press “Apply agent changes”**.
- Annotations live in `core/annotations.py` (`AnnotationStore`): id, FDI, arc position, kind, severity, author,
  source, timestamps. Add clinician notes, remove rows, export the whole set to JSON.
- The key never leaves the machine except as an `Authorization: Bearer` header; it is not put in the request body,
  the annotations, or error messages (`redact_secrets`). Requests are anonymised (no patient identifiers) and run on
  a worker thread with a poll timer, so Slicer never freezes.

- **Tangential (ridge-parallel) sections** (`core/tangential.py`, embedded inside **Step 4** alongside the arch & cross-section viewports and shared with **Step 8**)
  - The plane spanned by the arch *tangent* and the occlusal axis at a given arc position — it looks **along** the ridge instead of across it, showing crest continuity between adjacent teeth (slab `mean`/`max`/`min`, 0.3 mm pixels).
  - Follows the cross-section cursor by default (debounced), or can be stepped manually; the cortical crest line and the crest drop are drawn on the section, and it can be saved as PNG.
- **Interactive Pre-Print Editor & Keypoint Overrides** (`CBCTPlanner.py`, `PrintLayoutEngine.py`, `core_print_render.py`)
  - Opens before 8×10″ film composition, auto-populating `H` and `W0` (and real cross-section images) from `MeasurementEngine` even when Step 5 has not yet been run.
  - Supports interactive handle dragging (`H` and `W0`, keeping `H` passing through `W0`'s midpoint while preserving its coronal crest offset), freehand inspection lines/circles, ParameterNode JSON persistence (`MeasurementOverrides`), and `.jsonl` export of corrected keypoints/overlays for ML training.
  - Cross-section corner legends (`W3/W5`) automatically relocate away from overlapping cortical bone, caliper overlays, vital structures, and the `10 mm` scale bar; `AnnotationStore` flags (`flag`) and agent comments (`ai_comment`) render into a compact clinical notes strip on both the 8×10″ PNG/PDF and HTML report (`BL-14`).
- **Edentulous & Partial-Jaw Arch Fitting (`jawbone_crest` + Jaw-Aware Odontogram)** (`core/arch_detector.py`, `CBCTPlanner.py`)
  - When < 30 tooth voxels exist, `fit_arch_from_label_mask` falls back to alveolar `jawbone_crest` (filtering ascending ramus Z spikes). When posterior/anterior teeth are missing on a jaw with segmented jawbone (`end_extension_mm > 0`), the fitted curve follows the segmented jawbone alveolar ridge posteriorly instead of stopping at the last surviving tooth centroid; `"shared"` mode fits lower and upper separately and blends them so lower+upper points never zig-zag in panoramic views.
  - Step 5's odontogram hides the absent jaw's quadrant/tooth buttons when only Mandible or Maxilla is present, marks missing teeth distinctly while keeping them clickable, and auto-selects + measures contiguous missing spans (e.g. clicking `#46` when `45/46/47` are missing measures all three).
  - Step 5's cross-section viewer supports one-click/button **Full Screen** inspection with keyboard `Prev`/`Next` navigation, compact thumbnail cards, and proportional 8×10″ print card sizing.

## MCP server for 3D Slicer ([zhaoyouj/mcp-slicer](https://github.com/zhaoyouj/mcp-slicer))

Lets an MCP client (Claude Desktop, Cline, Claude Code, …) drive Slicer directly: list MRML nodes, run Python in
Slicer, and take screenshots of the app, a slice view or the 3D view, so an agent can reason about the case.

**Install** (once, needs [uv](https://astral.sh/uv)):
```bash
uv tool install mcp-slicer      # or let the client resolve it: uvx mcp-slicer
uv tool update-shell            # puts mcp-slicer on PATH (not required when using .mcp.json)
```
The repository ships `.mcp.json` at its root, so clients that read that file pick the server up with `uvx mcp-slicer`.

**Start Slicer with its REST API ready** — the Web Server GUI module leaves `exec` disabled by default, so use the
launcher instead of enabling checkboxes by hand:
```bash
"D:/Program Files/3D Slicer 5.12.3/Slicer.exe" --no-splash \
  --additional-module-paths ./CBCTPlanner \
  --python-script analysis/mcp_slicer_check/slicer_webserver_boot.py
```
That binds Slicer's REST API to `127.0.0.1:2016` with the Slicer API + `exec` interfaces on (loopback only, so no
Windows firewall prompt) and registers **CBCTPlanner** as a module, which lets the agent call this module's own logic
through `execute_python_code`. The interactive equivalent is the **Web Server** module: *Advanced* → tick
`Slicer API` and `Slicer API exec` → *Start server* (port 2016).

**Verify the whole chain** (boots Slicer, speaks MCP over stdio, calls every tool):
```bash
python analysis/mcp_slicer_check/verify_mcp_slicer.py
```
Checked here against **Slicer 5.12.3** (README asks for 5.8+): 11/11 — handshake, `tools/list`, `execute_python_code`
(result and error path), `list_nodes` names/ids/properties on the live scene, `CBCTPlanner` visible as a module, and
JPEG captures of the application window, the Red slice view (matching the widget size) and the 3D view from axis A.

**Notes and limitations**
- The server hardcodes `http://localhost:2016/slicer`, so Slicer's web server must stay on port 2016.
- `execute_python_code` is remote code execution by design. Keep the API on loopback, and stop it (*Web Server → Stop
  server*) once the agent session is over.
- Slice screenshots grab the on-screen widget, so the image is only as big as that view; maximise Slicer for a usable
  capture, and prefer `view_type="application"` or `"3d"` for full-case context.

## Roadmap / Future Updates

Open items below are the remaining backlog after the v3.2+ hardening pass (`BL-12`, `BL-14`, `BL-15`, `BL-17`, and `BL-18` were closed on 2026-11-25); status and closed evidence live in `analysis/slicercbctplanner_1/BUSINESS_RULES.md`.

### Agent & annotations
- **Let the agent change implant selection** — `set_implant_diameter` is accepted and then reported as skipped, because
  the plan table is derived from measurements; the plan needs an explicit override column first (`BL-13`).
- **Token/image budget** — the payload is built once with up to 8 images; long cases would benefit from a per-image
  budget and a size estimate before sending.
- **Review history** — one request/response pair is kept in memory; a per-case log (request id, model, latency, applied
  actions) would make the agent auditable.

### Measurement & arch refinements
- **Smooth `W0`/`W3`/`W5` along the arch** (`BL-01`) — a 1-site Gaussian would stop isolated slices between roots from
  spiking.
- **Dynamic `half_v_mm`** (`BL-05`) for very tall anterior mandibles, and **PDL gap validation** on real molars (`BL-06`).

### Rendering & performance
- `1.0×` vs `1.25×` RaySum scale and gamma/CLAHE post-processing (`BL-03`); user-adjustable VR `Zoom` (`BL-04`).
- Reuse RGBA tiles instead of the `RGBA → base64 → PIL.Image` round-trip when composing the film (`OPT-03`).
- Vectorise `finalize_transparent_png_bytes` with `scipy.ndimage.label` (`OPT-04`).

### MCP
- The server hardcodes `http://localhost:2016/slicer`; a small vendored copy that reads `SLICER_WEB_SERVER_URL` would
  allow another port or host (`BL-16`). Loopback-only binding and the stop-server step stay mandatory either way.

## Running Tests
```bash
python -m unittest discover -s tests -v      # 242 tests across 17 modules (0 failures)
```

Two headless harnesses run inside 3D Slicer's *own* Python (they need VTK/Qt, so the plain interpreter cannot run them):

```bash
# measurement + edentulous/shared/partial jaw + BL-12 + 8x10 BL-14 + tangential + agent round trip (27 checks)
"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe" analysis/slicercbctplanner_1/verify_real_ct.py

# boot Slicer with the REST API and exercise every MCP tool (11 checks)
python analysis/mcp_slicer_check/verify_mcp_slicer.py
```

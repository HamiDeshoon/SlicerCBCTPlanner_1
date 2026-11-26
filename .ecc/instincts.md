# ECC Instincts & Karpathy Directives Registry (`SlicerCBCTPlanner`)

## 1. 3D Slicer VTK / MRML Python Binding Rules
- **Trigger**: Accessing `vtkSegment` tags or visibility on `vtkMRMLSegmentationNode`.
- **Observed Pattern**:
  - In 3D Slicer C++ VTK bindings, `segment.GetTag(name, vtk.reference(""))` requires 2 arguments, whereas test mocks may accept 1 argument. Always use a safe helper `safe_get_segment_tag(segment, key)` that tries 2-arg `vtk.reference("")` first and falls back to 1-arg for unit test mocks.
  - Segment visibility lives on `segmentationNode.GetDisplayNode().GetSegmentVisibility(seg_id)` and `SetSegmentVisibility(seg_id, bool)`, NOT on `vtkSegment` directly (keep fallback to `segmentationNode.SetSegmentVisibility` for unit test mocks).
- **Rationale**: Prevents runtime `TypeError` and `AttributeError` in real 3D Slicer that were masked by simplified unit test mocks.

## 2. ToothFairy3 vs. 5-Class DentalSegmentator Disambiguation
- **Trigger**: Combining segmentation segments into a unified label mask (`_extract_combined_label_mask`).
- **Observed Pattern**: Never set `is_5class = True` merely because a segment name contains `"Mandible"` or `"Maxilla"`. A segmentation is 5-class ONLY when it contains no individual FDI tooth labels (`11..48`), no split Left/Right IAN canals (`3, 4`), and at most 5 anatomical groups.
- **Rationale**: Prevents 32-tooth ToothFairy3 masks from being misinterpreted as 5-class masks.

## 3. Dual Pillow + Qt Rendering for Clinical Print & Measured Slices
- **Trigger**: Rendering cross-section slices (`render_cross_section`), panoramic strips (`render_panoramic_strip`), and printable PDF/PNG sheets (`PrintLayoutEngine`).
- **Observed Pattern**: Always draw all 5 clinical overlays (Green Bone Height `H` or `H~` dashed for interpolated, Yellow Crest Width `W0`, Cyan `W3`, Orange/Blue `W5`, Red/Cyan Nerve Canal & Clearance, plus jaw/bone contours and HU/Misch badge), and provide a native `qt.QImage`/`qt.QPainter` fallback if `Pillow` is absent. Save print files (`.pdf`, `.png`, `.html`) to an explicit user-accessible folder as well as displaying an inline Measured Image Gallery in the UI.
- **Rationale**: Guarantees measured images and print files are always generated and visible in 3D Slicer.

## 4. W0 Bridges Buccal↔Lingual Across Dentate Socket
- **Trigger**: Computing `W0` (crestal width) at any site where `mask_cs` contains both `jaw_id` and `tooth_id` voxels (dentate ridge).
- **Observed Pattern**: `W0` on a dentate ridge MUST bridge from outer-lingual cortex to outer-buccal cortex across the socket — not stop at the mesial or distal surface of the in-socket root (single plate ≈ 2 mm). Expected `W0` for a bridged molar socket: **6.0 – 9.5 mm**. Use helpers `_find_socket_crest_rims` and `_select_ridge_run_indices` to identify both cortical walls, then `_sample_band_profile` to confirm bone presence across the full width. PDL gap tolerance ≤ 1.5 mm (not empty air).
- **Rationale**: A 2 mm single-plate measurement is clinically meaningless for implant planning and impossible to report as consistent cross-slice data (adjacent slices without root contact return the true 6–9.5 mm).

## 5. `PanoProjectionMode` Default is `"RaySum"` Everywhere
- **Trigger**: Any code that initializes or reads `PanoProjectionMode` from the param node, combo box, or hardcoded string.
- **Observed Pattern**: `"RaySum"` is the ONLY allowed hardcoded default string. `"Mean"` must never appear as a default in `PrintLayoutEngine._pano_strip_uri`, `Phase7Tools`, `CBCTPlanner._setDefaultParameters`, or the panoramic combo box entries. `SlabMode="Mean"` (VTK slab rendering) is a completely separate parameter and unrelated.
- **Rationale**: `Mean` produces a washed-out panoramic that does not match clinical DPR film; `RaySum` with 1.25× scale and percentile W/L mapping matches film density and is the standard clinical rendering.

## 6. RGBA PNG Capture for Transparent Pano/VR Panels
- **Trigger**: Any function that captures a VTK render window, panoramic slice, or 3D VR viewport to PNG or data URI.
- **Observed Pattern**:
  1. Before capture, set `renderWindow.SetAlphaBitPlanes(1)`, call `renderer.GradientBackgroundOff()`, `renderer.SetBackgroundAlpha(0.0)`, and configure the image filter with `SetInputBufferTypeToRGBA()`.
  2. Use a `try/finally:` block to restore `SetAlphaBitPlanes(0)` and `GradientBackgroundOn()` after capture.
  3. `_data_uri_to_image` MUST NOT call `.convert("RGB")` — keep the RGBA channel intact for compositing.
  4. `_paste_contain` MUST alpha-composite the RGBA tile over the film using `Image.alpha_composite` or `paste(img, mask=img.split()[3])`, never draw a solid background color first.
- **Rationale**: Dropping the alpha channel turns the transparent air black (`#020617`), producing opaque dark pano/VR panels on the clinical 8×10 film which are clinically unacceptable and aesthetically broken.

## 7. Remote-Agent Output Is Untrusted Input
- **Trigger**: Parsing, storing or applying anything that came back from a review-agent endpoint (`core/agent_review.py`).
- **Observed Pattern**: Parse JSON tolerantly (fenced block, prose wrapper, brace-balanced substring) but **validate before
  acting**: only whitelisted ops (`add_annotation`, `update_annotation`, `remove_annotation`, `add_tooth`, `remove_tooth`,
  `set_implant_diameter`, `flag_site`) may run, and never automatically — the GUI applies them only on an explicit
  *Apply agent changes*. Count rejected ops and show them; do not silently drop them.
- **Also**: the API key is only ever an `Authorization: Bearer` header. It must not enter the request body, the annotation
  store, the MRML scene, a log line or an exception message (`redact_secrets`). Patient identifiers are replaced before
  the payload leaves the machine.
- **Rationale**: A model can hallucinate a tooth, a measurement or an FDI number; a whitelist plus an explicit Apply step is
  the difference between advice and an unaudited edit to a clinical plan.

## 8. Tangential Sections Use the Arch Frame Explicitly
- **Trigger**: Sampling a ridge-parallel (tangential) section in `core/tangential.py`.
- **Observed Pattern**: Span the image plane by the **arch tangent** (image columns) and the **occlusal axis** (image rows,
  **row 0 = superior**); average the slab along the **buccal normal**. Do not reuse the cross-section axis convention — a
  tangential view is 90° from a cross-section and mixing the two silently transposes the anatomy. Report `crest_drop_mm`
  from the cortical crest line (bone→air per column), and render RGBA with transparent air.
- **Rationale**: Cross-sections answer "how much bone at this site"; tangential sections answer "is the crest continuous
  along the ridge". Keeping the basis explicit makes the two views comparable instead of interchangeable.

## 9. No QThread — Network Work Is a Plain Thread Plus a QTimer Poll
- **Trigger**: Any blocking I/O started from a widget handler (agent HTTP request, exports).
- **Observed Pattern**: `threading.Thread` runs the call into a plain attribute (`_agentWorkerOutcome`), a `QTimer` (150 ms
  for the agent, 180 ms debounce for the tangential refresh) polls it on the GUI thread, and every timer/thread handle is
  stopped in `cleanup()`. Do not introduce `QThread`/`Signal` patterns into this codebase — none of the existing modules use
  them and headless tests cannot construct them.
- **Rationale**: Keeps 3D Slicer responsive without a second threading idiom, and keeps the handlers testable without Qt.
## 10. Parse Agent Endpoint URLs with `urlsplit` — Never Suffix Raw Strings
- **Trigger**: Normalizing, validating, or redacting user-configured OpenAI-compatible URLs (`AgentEndpointConfig.resolved_url`, `is_configured`, `redacted` in `core/agent_review.py`).
- **Observed Pattern**: Split the URL with `urllib.parse.urlsplit` before normalizing the path so query parameters (such as Azure `?api-version=...`) stay intact after `/chat/completions` is appended, strip fragments, validate a non-empty hostname (`netloc`), keep `is_configured()` aligned with `resolved_url()` (case-insensitive scheme), and strip/redact userinfo and query-string credentials (`?key=`, `?api_key=`) before logging or returning `redacted()`.
- **Rationale**: Appending `/chat/completions` to a raw string corrupts query-string gateways (`...?api-version=2024-06-01/chat/completions`), and unredacted URL query parameters can leak API keys into UI status labels or error logs.

## 11. Single-Cortex W0 Honesty (`BL-12`) & Coronal-Offset Re-anchoring (`BL-18`)
- **Trigger**: Measuring crestal width `W0` in `core/measurements.py` or dragging `W0` handles in the Interactive Pre-Print Editor / `PrintLayoutEngine.py`.
- **Observed Pattern**: After the anterior deeper-caliper rescue loop finishes, if `W0 < 3.0 mm` AND `_span_bone_runs(w0) <= 1`, withhold `W0` (`crest_width_mm = None`, `withheld_reason = WITHHELD_W0_SINGLE_CORTEX`, badge `"W0: 1-cortex"`) while retaining bone height `H` and crest-point anchoring (`crest_uv` fallback). Whenever `W0` is edited or dragged, re-anchor `h_start_uv` through `W0`'s midpoint via `reanchor_h_start_to_w0` preserving the signed distance (`coronal_offset_along_axis`) rather than snapping `h_start_uv` onto `W0`.
- **Rationale**: A single cortical wall plate (`1–3 mm`) is not an alveolar ridge width `W0`, and flattening `h_start_uv` onto `W0` destroys clinician crest-level adjustments on tilted or scalloped ridges.

## 12. Corner Legend Collision Avoidance (`W3/W5`) & Clinical Notes Print Strip (`BL-14`)
- **Trigger**: Rendering `W3/W5` boxes in `core_print_render.render_cross_section` or composing 8×10″ film sheets in `PrintLayoutEngine`.
- **Observed Pattern**: Register the `10 mm` scale bar in `occupied_rects` and score candidate legend corners (bottom-right → bottom-left above scale bar → top-left → top-right) against `occupied_rects`, drawn calipers/vital contours, and cortical bone/mask pixels while keeping pill bounds (`x >= 18, y >= 24`). Collect active `flag` / `ai_comment` entries from `AnnotationStore` and render a compact `Clinical Notes & Review Flags:` strip above the disclaimer on both PNG/PDF and HTML (grayscale and color).
- **Rationale**: Keeps quantitative width boxes readable off the mandibular/maxillary cortex and surfaces clinician/AI flags directly on the printed film.

## 13. Single Version Source & Complete CMake Script/Resource Enumeration
- **Trigger**: Adding Python modules or resources under `CBCTPlanner/` or updating release version metadata.
- **Observed Pattern**: Store the authoritative version in `CBCTPlanner/Resources/VERSION` (`3.3.0`), load it dynamically in `Utils.SOFTWARE_VERSION`, `pyproject.toml`, and root `CMakeLists.txt`, enumerate every `.py` and `Resources/*` file inside `CBCTPlanner/CMakeLists.txt`, and verify isolated staged imports via `tests/test_packaging_completeness.py` (`cwd=tempdir`, clean `PYTHONPATH`).
- **Rationale**: Prevents broken Slicer installs where source-cwd imports pass locally while installed extension directories miss subpackages (`WorkflowUIHelpers.py`, `adt/`, `core/`).

## 14. Parallel-Transport (Bishop) Arch Frames & Half-Voxel Pano Edge Clamp
- **Trigger**: Computing orthonormal `(T, N, B)` frames in `core/arch_detector.py` / `PanoramicCurveLogic.py` or sampling panoramic slabs near volume borders.
- **Observed Pattern**: Use `compute_parallel_transport_frames` (Rodrigues minimal rotation propagation) instead of per-point `chord_mid` outward flipping so partial hemi-arches and steep curves never suffer 180-deg frame flips. In panoramic `map_coordinates`, clamp in-FOV fractional boundary coordinates (`[-0.5, dim - 0.5]`) to `[0, dim - 1 - 1e-5]` with `mode='nearest'` while painting `~in_fov` pixels `-1000.0`, eliminating dark 1px border blending.
- **Rationale**: Ensures twist-free cross-sections/CPRs and crisp border slices across tilted/anisotropic CBCT volumes.

## 15. Multi-Root Endodontic Topology Graphs & Exact Capsule Implant SDFs
- **Trigger**: Analyzing root canals (`EndodonticEngine.py`, `CanalTracer.py`) or implant clearances (`ImplantPlanner.py`).
- **Observed Pattern**: Never extract canals by naive Z-sorting with constant `0.5 mm` radii; build a 26-neighbor EDT-ridge weighted geodesic tree + 3D spherical local-maxima NMS to recover multiple roots, bifurcations, local EDT radius profiles, Schneider 3D curvature, and outer dentin wall thickness. Use closed-form capsule segment-segment distance + tapered capsule SDFs + `UncertaintyClearance` (reporting explicit limitation notes instead of invented calibration constants).
- **Rationale**: Multi-root molars and tight surgical corridors require true topological branching, exact apical hemispherical geometry, and transparent uncertainty accounting.

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

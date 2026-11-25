# Code Modernization Assessment & Business Rules: `SlicerCBCTPlanner_1` → v3.1

## 1. Executive Assessment (`/modernize-assess`)
- **System**: `SlicerCBCTPlanner_1` (3D Slicer Scripted Loadable Extension in Python 3.9+ / VTK / Qt / CTK / NumPy / SciPy / PyTorch).
- **Codebase Inventory**: 51 Python modules (28 in `CBCTPlannerLib/`: 15 top-level + 13 in `CBCTPlannerLib/core/`) + 16 test modules (**219 unit tests**, 8 wizard steps / 8 step chips, 22 active `CBCTPlanner/*` QSettings keys as of 2026-11-25).
- **Modernization Method**: **Uplift + Feature Expansion** — preserve all behavioral guarantees (B1–B22, O1–O9, F1–F20) while fixing critical Slicer runtime/rendering defects and adding clinical subsystems (`RiseMarEngine`, `AutomatedDentalToolsEngine`, `SlicerDentalSegmentator` Vivid 3D/VR, `RaySum` panoramic, transparent RGBA exports).

---

## 2. Extracted Clinical & Engineering Rule Cards (`/modernize-extract-rules`)

### Rule Card RC-01: Honest Clinical Measurements & Dentate/Edentulous Ridge Envelope
- **Given**: A resampled cross-section `(u_coords, v_coords, mask_cs, img_cs)` orthogonal to the dental arch at `arc_mm`.
- **When**: Computing alveolar bone height `H`, crestal width `W0`, depth widths `W3`, `W5`, `W6`, `W9`, and vital nerve/sinus clearance.
- **Then**:
  1. **Socket-bridging `W0` on dentate ridges**: on a dentate site the outer-buccal and outer-lingual bone walls may be separated by an in-socket tooth root (or PDL gap ≤ 1.5 mm). The crest width `W0` **must bridge** from outer-lingual cortex to outer-buccal cortex across the socket — not stop at the mesial or distal surface of the tooth. Helpers `_same_jaw_tooth_mask`, `_find_socket_crest_rims`, `_select_ridge_run_indices`, and `_sample_band_profile` implement this bridge. Expected `W0` on a bridged molar: **6.0 – 9.5 mm** (regression test: `test_crest_and_w0_come_from_bone_not_the_tooth_crown` asserts `crest_width_mm ∈ [6.0, 9.5]`).
  2. **Large-angle axis stability**: `_detect_alveolar_crest_and_axis` uses PCA on bone voxels clipped to `ridge_mask` (≤ 16 mm apical) with an elongated-axis guard; at angles > 45° this prevents selecting the minor eigenvector and keeps the ridge axis aligned with the alveolar wall.
  3. **Height line endpoint guards**: `_compute_bone_height_in_cs` uses 3D spherical EDT only when a confirmed IAN canal is present in the ROI; on non-canal anterior slices the ray is extended to the inferior jaw boundary rather than stopping at a spurious EDT minimum. The H line always reaches the inferior jaw boundary unless the IAN canal is encountered first.
  4. **IAN canal clearance search radius**: `_select_local_vital_structure` uses `u_ok = (|u| <= 11.0 mm) | (|u - u_ref| <= 8.5 mm)` and `local_radius_mm = 28.0 mm` (was 15 mm) so posterior canals > 15 mm from the crest are detected.
  5. **Width–crest synchronization**: after `_move_crest`, `width_endpoints_uv` is resynchronized at the new crest position, eliminating the slice-to-slice 2 mm single-plate spuriousness when the socket geometry shifts between adjacent slices.
  6. Cross-section rendering: grayscale X-ray film background, `H~` dashed caliper annotation for interpolated (non-IAN) heights, standard `H` solid caliper for IAN-limited heights. Root/implant/safety overlays removed. Remaining overlays: jaw contour (subtle gold), bone height (green), `W0` (yellow), `W3` (cyan), `W5` (orange), nerve clearance vector (red dashed).

### Rule Card RC-02: Print File Generation (`PDF` + `300 DPI PNG` + `HTML`) & Inline Measured Gallery
- **Given**: User selects 1 tooth, multiple teeth, or a quadrant (Q1–Q4) and clicks **Generate Print Layout** or **Measure & Show Slices**.
- **When**: `PrintLayoutEngine.generate_8x10_layout` executes.
- **Then**:
  1. Uses `self._parameterNode` (or `self.logic.getParameterNode()`), never `self.getParameterNode()` on the widget.
  2. Uses `segmentationNode.GetDisplayNode().GetSegmentVisibility(seg_id)` / `SetSegmentVisibility(seg_id, keep)` and 2-arg safe `GetTag`.
  3. Writes **three print files** (`<Scan>_Print_Layout.pdf`, `<Scan>_Print_Layout.png`, `<Scan>_Print_Layout.html`) to the user's selected/configured output directory (defaulting to `~/Documents/CBCTPlanner_Prints` with a folder picker & quick "Open Folder / Save As" button, or `out_dir` when passed programmatically).
  4. Immediately populates the **Inline Measured Cross-Section Gallery** inside the module panel AND opens the interactive high-resolution print preview dialog.
  5. **Pano/VR panels**: panoramic and VR tiles are rendered with **transparent RGBA backgrounds** (no `#020617` dark fill). `_paste_contain` composites RGBA over the film via alpha-channel; `.convert("RGB")` is never applied before pasting transparent tiles.
  6. **Panoramic projection**: `PrintLayoutEngine._pano_strip_uri` reads `PanoProjectionMode` from the param node — defaults to `"RaySum"`, never hardcodes `"Mean"`.

### Rule Card RC-03: Vivid Teeth & Roots 3D/VR Styling (`gaudot/SlicerDentalSegmentator`)
- **Given**: A 5-class (`DentalSegmentator`) or 77-class (`ToothFairy3`) segmentation node and/or CBCT volume node.
- **When**: Applying 3D styling or Volume Rendering presets.
- **Then**:
  1. Teeth (crowns & roots) receive vivid anatomical colors (`1.0` 3D opacity), and the VR volume property applies Phong shading (`Ambient=0.25`, `Diffuse=0.88`, `Specular=0.45`, `SpecularPower=26.0` for Vivid Roots VR; `Specular=0.35`, `SpecularPower=18.0` for Teeth & Bone).
  2. Maxilla and Mandible jawbones default to translucent 3D opacity (`0.25`) so **tooth roots and the mandibular canal are clearly visible through the bone**, controllable via a live **Bone Opacity Slider (`0.0 – 1.0`)** and **Vivid Presets** (`DentalSegmentator Vivid (Teeth & Roots)`, `Roots & Nerve X-Ray (Hide Bone)`, `Endodontic Root-Canal View (Translucent Teeth)`, `Surgical Bone & Teeth (Opaque Bone)`).
  3. VR preset `"DentalSegmentator Vivid Roots (VR)"` suppresses soft tissue and makes trabecular bone a subtle veil while ramping root dentin and crown enamel to high opacity.

### Rule Card RC-04: RISE-MAR Metal Artifact Reduction (`Masaaki-75/rise-mar`)
- **Given**: A CBCT volume affected by metallic dental restorations, implants, or brackets, and weights at a user-configured `.pkl` / `.pth` checkpoint (stored in Slicer settings).
- **When**: The user enables/runs **RISE-MAR Metal Artifact Reduction**.
- **Then**:
  1. Clips HU to `[-1024, 3072]` and normalizes to `[0, 1]` per `Ma et al., IEEE TMI 2025`.
  2. Loads user-configured RISE-MAR weights (`.pkl` / `.pth`) with sinogram-streak/frequency hybrid fallback when weights are absent in CI.
  3. Restores streak-corrupted peri-implant/interproximal regions while preserving true high-density metallic cores (`> 2500 HU`), computes pre/post Clinical Quality Assessor (`CQA`, 1–10 scale) scores, and outputs `<Volume>_RISE_MAR` (or updates in-place).

### Rule Card RC-05: RAS Auto-Standardization
- **Given**: A CBCT volume is loaded or the user triggers Step 1.
- **When**: `CBCTPlanner._setDefaultParameters` and `AutomatedDentalToolsEngine` initialize.
- **Then**:
  1. RAS auto-standardization is the **only** mode: Pitch/Roll/Yaw alignment to the occlusal plane and midsagittal plane is executed automatically on volume load.
  2. The legacy `autoStandardizeOnLoad` boolean Slicer settings key was removed in `chore(settings): drop the legacy autoStandardizeOnLoad key` (`96ee440`). No code should write or read it.

### Rule Card RC-06: Panoramic Default `RaySum` & Transparent Pano/VR
- **Given**: Any call to generate, display, or export the panoramic image or the 3D VR render.
- **When**: `PanoramicCurveLogic`, `CBCTPlanner`, `Phase7Tools`, `PrintLayoutEngine`, or `AdvancedTools` initializes or triggers panoramic generation.
- **Then**:
  1. `PanoProjectionMode` defaults to `"RaySum"` everywhere. `"Mean"` must never be hardcoded in `PrintLayoutEngine`, `Phase7Tools`, or param node default combo entries.
  2. `PanoramicCurveLogic._finalize_projection` scales RaySum as `accum / max(1, n_layers) * 1.25` and applies a percentile W/L mapping in `_populate_panoramic_node_from_array` to match clinical film density.
  3. **Transparent air in display**: `display_panoramic_in_slice_view` calls `SetApplyThreshold(True)` on the volume display node to make air (HU ≈ −1000) transparent; it also clears Green Foreground and Label layers to prevent 3D volume bleed behind the panoramic.
  4. **Isotropic tight-fill**: the panoramic viewport crops empty air margins and fills the slice view with the dental arch isotropically (no pillarbox/letterbox black bars, no stretch).
  5. **RGBA PNG capture**: `capture_3d_view_png_base64` and `VRVisualization._vr_capture` set `SetAlphaBitPlanes(1)`, `GradientBackgroundOff()`, `SetBackgroundAlpha(0.0)`, `SetInputBufferTypeToRGBA()` with a `finally:` restore block. The VR render node background is set to `(0.0, 0.0, 0.0)` with zero alpha before capture.
  6. `_data_uri_to_image` preserves the RGBA channel — it must NOT call `.convert("RGB")` before returning.

### Rule Card RC-07: Remote Review Agent & Annotation Store
- **Given**: A measured case and a clinician-configured review endpoint (any OpenAI-compatible URL + API key).
- **When**: The clinician presses **Send case to agent**, then **Apply agent changes**.
- **Then**:
  1. **Wire format**: `POST <base>/chat/completions` with `Authorization: Bearer <key>`. `AgentEndpointConfig` normalises
     a bare host, `/v1`, `/openai`, `/v1/chat/completions` or a full `/chat/completions` URL to one endpoint, and the key
     is a header value only — never in the body, the annotations, the scene, a log line or an error message
     (`redact_secrets`).
  2. **Context**: `build_review_payload` sends jaw/arch geometry, every planned site (`H`/`W0`/`W3`/`W5`, IAN clearance,
     HU class, suggested implant, provenance and withheld reasons), the automated warnings, the existing annotations, an
     optional clinician question, and up to 8 rendered sections as images. Patient identifiers are replaced
     (`name`/`id` = `"Anonymous"`).
  3. **Output is untrusted**: `parse_agent_response` accepts fenced, prose-wrapped or brace-balanced JSON and extracts one
     `summary`, `comments[]` and `actions[]`. Actions are validated against a fixed whitelist — `add_annotation`,
     `update_annotation`, `remove_annotation`, `add_tooth`, `remove_tooth`, `set_implant_diameter`, `flag_site` — and
     **nothing is applied until the user presses Apply**. Unknown ops are counted as rejected, not silently dropped.
  4. **Plan mutation is narrow**: only `add_tooth`/`remove_tooth` reach the plan, through `toggleToothPlanMembership` via
     `_AgentPlanAdapter`. `set_implant_diameter` is whitelisted but reported as *skipped*, because the plan table is
     measurement-derived (see `BL-13`).
  5. **Annotations** live in `core/annotations.py` (`AnnotationStore`): id, FDI, arc mm, kind, severity, author, source,
     created/edited timestamps; add/update/upsert/remove/clear, per-FDI grouping, JSON round trip.
  6. **No UI freeze**: the request runs on a `threading.Thread` polled by a 150 ms `QTimer`; the tangential-refresh debounce
     uses 180 ms. Both timers are stopped in `cleanup()`.
- **Rationale**: A remote model is an advisory reviewer, not an authority. The whitelist + explicit Apply step keep a bad
  or hallucinated answer from silently editing a clinical plan or losing the clinician's notes.

### Rule Card RC-08: Tangential (Ridge-Parallel) Sections
- **Given**: The arch frame (tangent, buccal normal, occlusal axis) and an arc position in mm.
- **When**: `core/tangential.py` samples the tangential view in Step 8.
- **Then**:
  1. The image plane is spanned by the **arch tangent** (columns) and the **occlusal axis** (rows, **row 0 = superior**),
     with the slab averaged along the buccal normal (`mean`/`max`/`min`, 0.3 mm pixels) — it looks **along** the ridge
     rather than across it, so crest continuity between adjacent teeth is visible.
  2. The cortical crest line (bone → air transition per column) and `crest_drop_mm` are reported with the section; the
     renderer emits RGBA with transparent air, a crest line, HUD and scale bar.
  3. It follows the cross-section cursor by default through a debounced refresh, and can be stepped manually or saved
     as PNG.
- **Rationale**: Cross-sections answer "how much bone here"; tangential sections answer "is the crest continuous along
  the ridge", which a buccolingual slice cannot show.

---

## 3. Known Limitations & Backlog

| ID | Item | Priority |
|----|------|----------|
| BL-01 | Smooth `W0`/`W3`/`W5` along arch (1-site Gaussian, suppress single-slice outliers between roots) | Medium |
| BL-02 | Tune `compute_panoramic_display_window_threshold` on more scans; expose percentile UI slider | Medium |
| BL-03 | RaySum quality: test `1.0×` vs `1.25×` scale; consider gamma/CLAHE post-processing | Low |
| BL-04 | Make `Zoom(1.40)` user-adjustable in VR tight-frame | Low |
| BL-05 | Dynamic `half_v_mm` for tall anterior mandibles (current 25.0 mm clips chin in some scans) | Medium |
| BL-06 | PDL gap tolerance validation on real molar CT (current tolerance: ≤ 1.5 mm) | Medium |
| BL-07 | Move `test_clinical_delivery.py` NIfTI parser to `tests/helpers/nifti_reader.py` | Low |
| BL-08 | Phase 7 batch report: consume RaySum pano array in HTML report (currently stores only `.npy`) | Low |
| BL-09 | Trim unused `core/diagnostics.py` code (~180 lines of `provenance_hash` + report writer) | Low |
| BL-10 | Real-CT smoke test with `ToothFairy3S_0021_0000.nii.gz` — **done 2026-11-24**: `verify_real_ct.py` headless in Slicer's Python, 23/23 checks after the 2026-11-25 extension, 205 sections (see §6) | Closed |
| BL-12 | Withhold `W0` when the opposite cortex is not in the section (185/205 real-CT sections cut interproximally) instead of reporting a 1–3 mm plate as a crest width | Medium |
| BL-11 | Decide grayscale default in `generate_8x10_layout(grayscale=...)` for CLI/batch callers | Medium |
| BL-13 | Agent actions: apply `set_implant_diameter` / implant length instead of reporting them as skipped (needs a plan override column) | Medium |
| BL-14 | Print annotations and agent comments into the 8×10 film and the HTML report | Medium |
| BL-15 | Retry/backoff on agent rate limits (429/5xx; `AgentError.is_retryable` already exposed) | Low |
| BL-16 | Vendored `mcp-slicer` copy with `SLICER_WEB_SERVER_URL` so Slicer's port is configurable | Low |
| BL-17 | Fix `AgentEndpointConfig.resolved_url()` for query-string URLs (Azure `?api-version=...`), align `is_configured()` with `resolved_url()`, allow keyless localhost shims, redact URL credentials/query keys, and send an explicit `User-Agent` | Medium |
| BL-18 | Anchor bone-height `H` line start at the `W0` midline midpoint `(pt_l + pt_r) / 2` instead of `crest_uv` | Medium |

---

## 4. Test Suite Reference

| File | Tests | Coverage Area |
|------|-------|---------------|
| `tests/test_agent_review.py` | 27 | URL normalisation, payload/context, tolerant JSON, whitelist, redaction, mock transport |
| `tests/test_agent_step_glue.py` | 14 | Step 8 handlers headless (send/apply/remove/export, row→id map) |
| `tests/test_tangential_view.py` | 11 | ridge-parallel sampling, crest drop, RGBA render |
| `tests/test_anatomy_aware_measurements.py` | 29 | W0 socket-bridging, crest axis, height guards, IAN clearance |
| `tests/test_print_layout.py` | 10 | Pano/VR RGBA, transparent panels, pill placement |
| `tests/test_clinical_delivery.py` | 25 | End-to-end print file generation (PDF/PNG/HTML) |
| `tests/test_core_math.py` | 30 | EDT, axis math, plausibility bounds |
| `tests/test_slicer_integration.py` | 17 | RaySum default, pano display, VR background |
| `tests/test_orientation.py` | 12 | RAS standardisation, anatomical auto-fix |
| `tests/test_arch_fit.py` | 10 | Arch detection and curve fitting |
| `tests/test_autofit_widget.py` | 8 | Workspace role profiles, 8 step chips |
| `tests/test_cbct_planner_v4_fixes.py` | 8 | v4 regression fixes |
| `tests/test_v3_upgrades.py` | 5 | v3 upgrade paths |
| `tests/test_slicer_glue.py` | 5 | Widget / parameter-node glue |
| `tests/test_static_attrs.py` | 4 | Static attribute audit (no dangling `self.*`, no silent `except: pass`) |
| `tests/test_orientation_and_cross_sections.py` | 4 | Orientation + section geometry |
| **Total** | **219** | as of 2026-11-25 (1 skipped if external ToothFairy3 NIfTI is absent) |

Run: `python -m unittest discover -s tests`

Headless Slicer harnesses (need VTK/Qt — last run 2026-11-25 on Windows Slicer 5.12.3; not re-run in headless Linux audit):
`PythonSlicer.exe analysis/slicercbctplanner_1/verify_real_ct.py` (23/23) and
`python analysis/mcp_slicer_check/verify_mcp_slicer.py` (11/11).

---

## 5. Data Pipeline & Module Handoff Contracts (`/data-engineering`)

```text
DICOM / NIfTI Volume
  └─► core/volume_loader.py + core/orientation.py (RAS standardization, zero-copy CBCTVolume)
        └─► core/labels.py (ToothFairy3 77-class vs 5-class DentalSegmentator disambiguation)
              ├─► core/arch_detector.py ──► PanoramicCurveLogic.py (RaySum CPR, Z-padding crop, threshold W/L)
              ├─► core/measurements.py  ──► socket-aware bone_component, crest/W0 bridging, ray-guarded H
              │     └─► core/diagnostics.py (provenance: label-derived vs H~ interpolated)
              ├─► core/annotations.py   ──► clinician + agent annotations (JSON round trip)
              ├─► core/agent_review.py  ──► OpenAI-compatible endpoint ──► whitelisted actions
              ├─► core/tangential.py    ──► ridge-parallel sections (arch tangent + occlusal axis)
              └─► core_print_render.py  ──► PrintLayoutEngine.py (RGBA transparent pano/VR + 8x10 PDF/PNG/HTML)
```

### Optimization & Engineering Backlog (`OPT-01` – `OPT-05`)
| ID | Target | Optimization / Follow-Up | Impact |
|---|---|---|---|
| `OPT-01` | `PanoramicCurveLogic.py` | Cache `map_coordinates` 3D slab grid per `(arch_hash, spacing_mm, height_mm, offset_mm)` so toggling `RaySum`/`MIP`/`MinIP`/`Mean` avoids re-interpolating the CBCT volume. **Measured 2026-11-24: 0.16–0.27 s for 207 arcs × 240 rows at 0.30 mm (0.8–1.3 ms/column)** — already interactive, so only the 3D MIP card (0.64–0.95 s) is worth caching | High (interactive FPS) |
| `OPT-02` | `core/measurements.py` | Pre-index `local_vital_mask` bounding boxes and batch-vectorize `measure_all_sites` across adjacent slices (`-2 / 0 / +2 mm`) | Medium (multi-site speed) |
| `OPT-03` | `core_print_render.py` / `PrintLayoutEngine.py` | Pass `PIL.Image` / `ndarray` directly instead of `RGBA -> base64 -> PIL.Image` round-trips when composing 8×10″ film sheets | Medium (export latency) |
| `OPT-04` | `AdvancedTools.py` | Vectorize `finalize_transparent_png_bytes` border-connected flood-fill with `scipy.ndimage.label` on background mask instead of Python pixel BFS | Medium (high-DPI capture) |
| `OPT-05` | `Phase7Tools.py` / `cli_batch.py` | Emit structured JSON/Parquet-ready batch run manifest (`provenance_hash`, `calibration_stamp`, `withheld_reason`, site `H`/`W0`/`W3`/`W5`) for cohort QA | Medium (data pipeline QA) |

---

## 6. Real-CT Verification Record (2026-11-24)

Harness: `analysis/slicercbctplanner_1/verify_real_ct.py`, run with
`"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe"` (Python 3.12.10, VTK 9.6.2, NumPy 2.4.6, SciPy 1.17.1).
Data: `ToothFairy3S_0021_0000.nii.gz` + `ToothFairy3S_0021.nii.gz`, 370×370×169 @ 0.30 mm.
**23/23 checks pass over 205 cross-sections** (1 mm arc sweep, mandible + maxilla), plus the tangential-section and
agent round-trip checks added on 2026-11-25.

| Rule Card | What was verified | Result |
|---|---|---|
| RC-05 | Anatomical inversion auto-correction (`canonicalize_to_ras` → `auto_fix_anatomical_orientation`) | This scan is S-I **and** L-R inverted in voxel space; both axes corrected, post-fix report clean |
| RC-01.1 | Dentate `W0` bridges across the socket | 20/20 two-wall sections bridged, W0 5.1–10.2 mm (median 9.0), **12/12 molars ≥ 6 mm**, no sub-3 mm plate |
| RC-01.2 | Tilted ridges keep `W0` on bone | 79/79 sites (|axis| ≥ 15°, max 31.8°), both rims on cortex |
| RC-01.3/4 | `H` reaches the basal jaw end; stops before the IAN | 61/68 non-canal slices within 1.5 mm of the jaw border (median H 24.4 mm); 137/137 canal slices keep the caliper out of the canal (min lateral clearance 0.09 mm) |
| RC-06 | `RaySum` pano + RGBA transparent isotropic pano/VR | 240×206 RaySum in 0.16–0.27 s; strip 814×560 RGBA `alpha_min=0`, tight-filled; 3D MIP 613×280 RGBA in 0.64–0.95 s |
| RC-02 | 8×10 film export | PNG 2400×3000 @ 300 DPI, PDF `/MediaBox` 8.0×10.0 in, HTML `background: transparent` on pano/VR panels |
| RC-08 | Tangential section + crest drop | `(101, 141)` at 0.30 mm/px, arc 20.0 mm, 57 % bone pixels, 125 crest samples, drop 11.1 mm, 0.03 s; RGBA render with transparent air |
| RC-07 | Agent round trip with `MockTransport` | 49 KB payload from the real case, key only in `Authorization`, 1 comment parsed, 3 changes applied, 1 invalid op rejected, plan adapter called |

**Caveat:** the harness is headless — Slicer's Green view, 3D viewport and print-preview dialog were not rendered; only
the functions they call were exercised. See `SESSION_NOTES_2026-11-24.md` §8 for the full table and caveats
(incl. BL-12: 185/205 sections clip a single cortex, where `W0` is a plate rather than a bridge).

---

## 7. MCP Verification Record (2026-11-25)

[`zhaoyouj/mcp-slicer`](https://github.com/zhaoyouj/mcp-slicer) **0.3.1** (`uv tool install mcp-slicer`) against the local
**Slicer 5.12.3** (upstream asks for 5.8+): **11/11 checks pass**.

| Check | Result |
|---|---|
| MCP handshake + `tools/list` | PASS — server `SlicerMCP`, protocol `2025-06-18`, tools `list_nodes`, `execute_python_code`, `capture_screenshot` |
| `execute_python_code` result **and** error path | PASS — Slicer 5.12 accepts a raw text body on `/exec`; errors return `{"success": false, "message": "Server error: …"}` and surface as a tool error |
| `list_nodes` by names / ids / properties | PASS on the live scene — `/mrml/properties` keys come from `node.__str__()` (`Name`, not `name`) |
| `CBCTPlanner` visible to the agent | PASS — registered via `--additional-module-paths ./CBCTPlanner`, so `execute_python_code` can drive this module's logic |
| Screenshots | PASS — application (41 KiB), Red slice grab equal to the slice **view** widget size, 3D view from axis A |

Repo additions: `.mcp.json` (`uvx mcp-slicer`), `analysis/mcp_slicer_check/slicer_webserver_boot.py` (REST API on
`127.0.0.1:2016` with `exec` enabled, window maximised) and `analysis/mcp_slicer_check/verify_mcp_slicer.py` (11 checks).

Known limitation (`BL-16`): the server hardcodes `http://localhost:2016/slicer`, so a different web-server port needs a
patched copy. `execute_python_code` is remote code execution by design — keep the API on loopback and stop the server when
the agent session ends.

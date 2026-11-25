# Session Notes — 2026-11-24 / 2026-11-25: W0 measurement, panoramic/VR, and merge

Branch: `feat/new_ver_gpt` → merged into `main` this session.

---

## 1. What happened (2026-11-24 earlier session)

The working tree held one mixed batch (8 files, +159/−81) on top of `29f0cb8`. It was cut into four
commits, each independently revertable and each with a single purpose.

| # | Commit | Purpose | Files |
|---|--------|---------|-------|
| 1 | `419e5e2` docs(cbct): record bone-only crest and width derivation | Documents that crest / `W0` / widths come from jaw bone only, envelope feeds the height ray; hardens mask-scheme inference | `core/measurements.py`, `core/labels.py`, `BUSINESS_RULES.md` (RC-01 item 1) |
| 2 | `54f52f4` feat(print): render the GUI clinical film in grayscale | GUI print path passes `grayscale=True`, drops the quadrant VR isolation pass, logs a dropped VR panel | `CBCTPlanner.py`, `PrintLayoutEngine.py` (docstring + VR log) |
| 3 | `5bee833` fix(cbct): remove root/implant overlays and flag interpolated heights | Removes root-extent / implant-envelope / 2 mm safety drawing, adds `H~` provenance marking and `min_height_estimated` | `core_print_render.py`, `core/measurements.py`, `core/diagnostics.py`, `PrintLayoutEngine.py`, `tests/test_clinical_delivery.py`, `BUSINESS_RULES.md` (RC-01 items 1b, 2) |
| 4 | `96ee440` chore(settings): drop the legacy autoStandardizeOnLoad key | Deletes the obsolete settings key on startup so a stale `false` cannot be read as the current choice | `CBCTPlanner.py` |

## 2. What happened (2026-11-24 → 2026-11-25 this session)

Three focused fix commits were added to `feat/new_ver_gpt` and merged into `main`:

### 2a. `dc7591d` — fix(measurements): bridge dentate socket W0/widths, stabilize large-angle axis, and guard canal/jaw-end height

**Root causes diagnosed and fixed in `core/measurements.py`:**

| Bug | Root cause | Fix |
|-----|-----------|-----|
| W0 drops 5→2→6 mm across tooth socket | `_extract_primary_ridge_component` kept only one 8-connected bone region, discarding the opposing cortical wall | `_same_jaw_tooth_mask` fills the socket gap so both lingual & buccal walls stay in `bone_component` |
| W0 placed inside or over the crown at large angles | `_detect_alveolar_crest_and_axis` plateau snapped `crest_uv` into the socket horn or averaged across the void; PCA at ≈45° flipped to the minor eigenvector | `_find_socket_crest_rims` bridges to the midpoint of buccal/lingual bone crest; PCA clipped on `ridge_mask` (≤16 mm apical) with elongated-axis guard eliminates the sign flip |
| Height stops short of jaw base on non-canal slices | 3D spherical EDT truncation misclassified as canal stop when no 2D canal exists | 3D EDT truncation disabled when `local_vital_mask_2d` is `None`; non-canal slices now trace all the way to `basal_uv` |
| Height crosses IAN canal | `_select_local_vital_structure` missed canals > 15 mm from crest-world or when crest jumped off-axis | `u_ok` widened to ±11 mm (ridge-centered), 3D ROI expanded to 28 mm; direct ray-voxel guard in `_compute_bone_height_in_cs` |
| Stabilized `crest_uv` desynchronized from `width_endpoints_uv` | `_move_crest` updated crest + H without resetting widths | `_move_crest` now re-syncs all width endpoints and re-checks IAN clearance |

**New helper functions:** `_same_jaw_tooth_mask`, `_extract_coronal_bone_surface`, `_find_socket_crest_rims`, `_sample_band_profile`, `_refine_run_endpoints`, `_select_ridge_run_indices`.

**Tests added** (`tests/test_anatomy_aware_measurements.py`): `test_19` (W0 socket bridge), `test_20` (large-angle axis stability), `test_21` (non-canal height + IAN stop), `test_22` (stabilize_entry_points width-sync & canal guard).

### 2b. `15e1bbb` — fix(render): RaySum default, RGBA transparent isotropic pano/VR, and safe pill/footer overlays

**Changes in `core_print_render.py` and `PrintLayoutEngine.py`:**

- `render_panoramic_strip`: new `transparent_bg`, `crop_empty_vertical`, `preserve_aspect`, `target_size` params (backward-compat defaults). When enabled: trims empty vertical air bands, scales isotropically, outputs RGBA with `alpha=0` for air/background and opaque tooth/bone/ticks.
- `render_3d_mip_card`: new function — RGBA transparent air background, tight foreground crop, preserves physical `(sz, sx)` aspect, correct ToothFairy3 + 5-class canal IDs.
- `_place_non_overlapping_pill`: fallback clamped inside `[18, w-4] × [24, h-footer-4]`; pills can no longer clip into ruler or badge region.
- Footer bar drawn before calipers so deep basal `H` end-caps are never occluded.
- `is_estimated_provenance(w_prov)` used consistently (removed inline tuple check at L655).
- `PrintLayoutEngine._pano_strip_uri`: reads `PanoProjectionMode` from param node, defaults to `"RaySum"`; samples at 0.5 mm; renders with `transparent_bg=True, crop_empty_vertical=True, preserve_aspect=True`.
- `PrintLayoutEngine._vr_capture`: uses `render_3d_mip_card(..., transparent_bg=True)`.
- `PrintLayoutEngine._data_uri_to_image` / `_paste_contain`: preserve RGBA alpha channel; alpha-composite onto panel via `Image.alpha_composite`.
- `PrintLayoutEngine._build_8x10_html`: removed `#020617` hardcoded dark backgrounds on `.pano-section`, `.pano-img`, `.vr-card`, `.vr-img` → `background: transparent`; removed `max-height: 220px` on `.pano-img`.

**Tests added** (`tests/test_print_layout.py`, `tests/test_clinical_delivery.py`): 6 new regression tests covering clamped pills, RGBA panoramic, RaySum URI default, 3D MIP card RGBA, `_paste_contain` RGBA preservation, transparent HTML backgrounds. `test_crest_and_w0_come_from_bone_not_the_tooth_crown` updated to assert dentate `W0 ∈ [6.0, 9.5] mm` bridging buccal↔lingual bone crests (not 2 mm single plate).

### 2c. `6ece4e1` — feat(panoview): default RaySum panoramic and transparent tight-fill pano/VR

**Changes in `PanoramicCurveLogic.py`, `VRVisualization.py`, `AdvancedTools.py`, `CBCTPlanner.py`, `Phase7Tools.py`:**

- `PanoramicCurveLogic.sample_panoramic_slab_array` and `generate_panoramic_volume`: default `projection_mode="RaySum"`.
- `CBCTPlanner.py`: `DEFAULTS`, `panoProjCombo`, `updateGUIFromParameterNode`, `CBCTPlannerLogic._setDefaultParameters` all default to `"RaySum"` (SlabMode stays `"Mean"`); `_setDefaultParameters["VRPreset"]` aligned to `"DentalSegmentator (Teeth & Bone)"`.
- `Phase7Tools.py`: headless batch pipeline defaults to `projection_mode="RaySum"`.
- `PanoramicCurveLogic._crop_panoramic_vertical_padding`: trims out-of-volume air bands (2 mm margin), updates `CBCTPlanner.HeightMm` cleanly.
- `PanoramicCurveLogic.compute_panoramic_display_window_threshold`: percentile W/L tuned for RaySum `1.25×` scaling + `SetApplyThreshold(True)` / `SetLowerThreshold` for transparent air.
- `PanoramicCurveLogic.display_panoramic_in_slice_view`: clears Green Foreground/Label layers, sets flat black slice background, preserves `FitSliceToAll()` call count == 1 (`B10`), applies limiting-axis isotropic `SetFieldOfView` tight-fill.
- `CBCTPlanner.py`: `refit_panoramic_view` added; re-fits Green after `onSetDentalLayout`, `onSetDentalLightboxLayout`, and `reset_slice_views`.
- `VRVisualization.py`: `_extract_ras_bounds`, `frame_3d_views_tight` (ResetCamera + Zoom 1.40); clean flat background; `configure_3d_view` called from `set_volume_rendering`; axis labels + 3D box hidden.
- `AdvancedTools.py`: `finalize_transparent_png_bytes` (flood-fill alpha=0 preserving dark cavities + ~2% tight crop); `capture_vtk_render_window_png_base64` (RGBA VTK capture with `SetAlphaBitPlanes`, `GradientBackgroundOff`, restore in `finally`); `capture_slice_view_png_base64` extended with backward-compatible `transparent_bg: bool = False`.
- `capture_3d_view_png_base64`: defaults `transparent_bg=True, tight_crop=True`; Green slice capture enables both.

**Tests added** (`tests/test_slicer_integration.py`): 5 new tests in `PanoramicRaySumAndVrFramingTests` covering RaySum defaults, vertical crop + W/L threshold, Green layer cleanup + tight-fill + layout refit, VR tight framing, PNG RGBA flood-fill + VTK state restoration.

## 3. Checks run this session

| Check | Result |
|-------|--------|
| `python -m ruff check CBCTPlanner tests` | All checks passed |
| `python -m ruff format --check CBCTPlanner tests` | 42 files already formatted |
| `python -m unittest discover -s tests` | **167 tests, OK** (13.5 s) |
| `git merge feat/new_ver_gpt → main` | Clean no-conflict merge, pushed to origin |
| Old stale branches deleted | 3 local + 7 origin/* branches removed |

Not verified in this session (unchanged limitation — no running Slicer GPU process):
- No 3D Slicer / VTK / Qt, so panoramic RaySum rendering inside the Green view, VR tight framing in the 3D viewport, RGBA PNG captures, and the printed film were not live-rendered.
- The 38 MB real CT at `C:/Users/hamid/Downloads/inspect_ToothFairy3S_0021/` was used for unit-level assertions only.

## 4. What still needs to be done

1. **Visual check in Slicer** (`D:\Program Files\3D Slicer 5.12.3`) — load a scan with segmentation, trigger Step 3 (Arch) → Step 4 (Measure) on dentate + edentulous sites:
   - Verify `RaySum` pano in Green view: transparent air, no vertical bands, teeth crisp and full-width.
   - Verify cross-section overlays: `W0` bridges buccal↔lingual across molar socket; `H` reaches basal jaw on non-canal slices; no IAN crossing.
   - Verify 3D viewport: clean flat background, jaw/teeth tight-framed (no 40% empty margin).
   - Verify exported 8×10 PDF+PNG: panoramic strip fills its panel (no huge side margins), VR card has transparent background, `H~` dashed caliper on interpolated slices.
2. **Decide grayscale default** — `generate_8x10_layout(...)` currently defaults to `grayscale=True`. If programmatic callers (not GUI) must keep colour, set default back to `False`.
3. **Trim unused `core/diagnostics.py`** — `provenance_hash`, the preflight checklist, and `write_diagnostics_text` (~180 lines) are only used by `write_diagnostics_text`; only `calibration_stamp` and `summarize_span_worst_case` are used by the film.
4. **Refactor brittle tests in `tests/test_clinical_delivery.py`** — RAS-string assertion, display-name regex, and `resample_segmentation_to_volume` `hasattr` check test source text, not behaviour. The ~90-line NIfTI parser should move to a test helper.
5. **IAN canal stop fine-tuning on real scans** — `half_v_mm=30.0` was tested and caused plausibility failures; `25.0 mm` kept. For very tall anterior mandibles (>26 mm Z below arch) a scan-specific clamp or dynamic `half_v_mm` may be needed.
6. **Multi-root / PDL gap tolerance** — depth-width bridging currently allows ≤1.5 mm PDL gap. Needs visual validation on real molar roots with thin PDL widening.

## 5. Future follow-ups and optimizations

| Area | Idea |
|------|------|
| `W0` stability | Smooth `W0`/`W3`/`W5` along arch with a 1-site Gaussian so isolated messy slices (e.g., between roots) don't spike to outliers |
| Pano W/L | Tune `compute_panoramic_display_window_threshold` on more scans; expose percentile UI controls |
| RaySum quality | Experiment with `1.0×` vs `1.25×` scale factor on CBCT HU range; consider gamma/CLAHE post-processing for caries/crown visibility |
| VR tight-frame | Make `Zoom(1.40)` a user-adjustable setting; consider `vtkPolyData` bounding from segmentation labels for tighter fit without clipping implants |
| RGBA slice export | Extend `capture_slice_view_png_base64` with `tight_crop=True` and test the Green view panoramic PNG export via GUI |
| Test brittleness | Move `test_clinical_delivery.py` NIfTI parser to `tests/helpers/nifti_reader.py`; replace RAS-string and display-name regex assertions with behavioural ones |
| Phase 7 batch | Update `Phase7Tools.py` to consume `RaySum` panoramic output in its HTML report (currently stores `.npy` only) |
| `diagnostics.py` trim | Remove the unused `provenance_hash` + report-writer code if no audit trail is required |

## 6. Where to touch what (unchanged from previous session)

| Change | Location |
|--------|----------|
| Crest / `W0` / width rule | `CBCTPlannerLib/core/measurements.py`: `_detect_alveolar_crest_and_axis`, `_compute_bone_widths_with_endpoints_in_cs`, `_find_socket_crest_rims` |
| Socket-aware mask | `measurements.py`: `_same_jaw_tooth_mask`, `_ridge_envelope_mask` |
| Height ray & IAN guard | `measurements.py`: `_compute_bone_height_in_cs`, `_select_local_vital_structure` |
| Stabilization sync | `measurements.py`: `stabilize_entry_points` / `_move_crest` |
| Panoramic projection default | `PanoramicCurveLogic.py:646,790`; `CBCTPlanner.py DEFAULTS/panoProjCombo/4338`; `Phase7Tools.py:733`; `PrintLayoutEngine._pano_strip_uri` |
| Panoramic display | `PanoramicCurveLogic.display_panoramic_in_slice_view`, `_crop_panoramic_vertical_padding`, `compute_panoramic_display_window_threshold` |
| VR display | `VRVisualization.set_volume_rendering`, `configure_3d_view`, `frame_3d_views_tight` |
| PNG RGBA capture | `AdvancedTools.capture_vtk_render_window_png_base64`, `finalize_transparent_png_bytes`; `CBCTPlanner.capture_3d_view_png_base64` |
| Overlay draw list | `core_print_render.render_cross_section` + `_resolve_engine_overlay_data` |
| Panoramic print strip | `core_print_render.render_panoramic_strip`; `PrintLayoutEngine._pano_strip_uri` + `_paste_contain` + `_build_8x10_html` |
| 3D MIP card | `core_print_render.render_3d_mip_card`; `PrintLayoutEngine._vr_capture` |
| Grayscale film | `PrintLayoutEngine.generate_8x10_layout(grayscale=...)`; `CBCTPlanner.py` ~line 3590 |
| Clinical rules doc | `analysis/slicercbctplanner_1/BUSINESS_RULES.md` RC-01 through RC-06 |

## 7. Codebase graph (Graphify-style) — updated

**Highest in-degree (god nodes / ripple risk)** — `core/labels` (25), `core/volume_loader` (11), `core/orientation` (8), `core/measurements` (8), `core/arch_detector` (6), `Utils` (6), `PanoramicCurveLogic` (5), `core/print_layout` (5).

**Touched-module edges (this session)**
```
core/measurements     → arch_detector, labels, tooth_localizer, volume_loader
PanoramicCurveLogic   → (called by) CrossSectionNavigator, Phase7Tools, PrintLayoutEngine, CBCTPlanner
core_print_render     → labels, measurements
PrintLayoutEngine     → PanoramicCurveLogic, arch_detector, diagnostics, labels,
                        measurements, print_layout, core_print_render
VRVisualization       → Utils, labels
AdvancedTools         → (AdvancedTools self-contained for VTK/RGBA capture)
```

**Session record (ECC-style)** — verified: 167 tests OK (`unittest discover -s tests`), Ruff check & format clean (`42 files already formatted`), merged onto `main` (`92c9e20`).
Open threads: Slicer visual check (`3D Slicer 5.12.3` + `ToothFairy3S_0021`), `OPT-01..OPT-05` performance/data-pipeline optimizations, `BL-01..BL-11` backlog (`diagnostics.py` trim, brittle-test refactor, dynamic `half_v_mm`, PDL gap validation).

## 8. Real-CT / Slicer-Python verification (closes §4 item 1, BL-10)

Harness: `analysis/slicercbctplanner_1/verify_real_ct.py`, run headlessly with
`"D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe"` (Python 3.12.10, VTK 9.6.2,
NumPy 2.4.6, SciPy 1.17.1) on `ToothFairy3S_0021_0000.nii.gz` + `ToothFairy3S_0021.nii.gz`
(370×370×169 @ 0.30 mm; tooth labels are FDI, canals `3/4`, lingual canal `103`).
**19/19 checks pass, 205 cross-sections measured** (1 mm sweep, both jaws).

| Check | Result |
|-------|--------|
| `CBCTPlannerLib` + `PanoramicCurveLogic` / `core_print_render` / `PrintLayoutEngine` import in Slicer's Python | PASS |
| Anatomical inversion detected and auto-corrected (`standardize_orientation` order) | PASS — this scan is **S-I and L-R inverted** in voxel space; `canonicalize_to_ras` alone leaves the mandible above the maxilla, so the app's `auto_fix_anatomical_orientation` step is mandatory. Without it every crest/H result is nonsense (H ran upward into the maxillary teeth). |
| Dentate `W0` bridges buccal↔lingual across the socket | PASS — 20/20 sections that show two cortical walls at the crest; W0 min 5.1 / median 9.0 / max 10.2 mm; **12/12 molars ≥ 6 mm**; zero sub-3 mm plates |
| `W0` rims land on jaw bone | PASS — 205/205 |
| Tilted ridges keep `W0` on bone (|ridge axis| ≥ 15°) | PASS — 79/79 sites, max tilt 31.8°, W0 7.3–11.1 mm |
| `H` reaches the basal jaw end (non-canal slices) | PASS — 61/68 within 1.5 mm of an independently measured jaw border; median H 24.4 mm |
| `H` stops strictly before the IAN | PASS — 137/137 canal slices: no IAN voxel under the caliper corridor; min lateral clearance 0.09 mm |
| RaySum panoramic on the real volume | PASS — 240×206 columns, 0.16–0.27 s (≈0.8 ms/column, 25 mm slab), std 835, all finite |
| Panoramic strip RGBA + transparent air + isotropic tight-fill | PASS — 814×560 RGBA, `alpha_min=0`, limiting axis filled 100% |
| 3D MIP/VR card RGBA + transparent air | PASS — 613×280 RGBA, `alpha_min=0`, 0.64–0.95 s on the full volume |
| 8×10 film export | PASS — PNG 2400×3000 @ 300 DPI (585 KiB), PDF `/MediaBox` exactly 8.0×10.0 in (361 KiB), HTML keeps `background: transparent` on pano/VR |

### Findings / caveats

1. **The S-I/L-R flip is data-level, not a code defect** — but it means a scan whose header claims RAS while the voxels
   are inverted *silently* produces an inverted crest/H unless the auto-fix runs. `standardize_orientation(auto_fix_anatomy=True)`
   handles it (verified), so the load path is correct; any headless/batch path must call `auto_fix_anatomical_orientation` too.
2. **Single-wall sections** — 185/205 sections cut interproximally/obliquely and clip only one cortical plate; `W0` there is
   a plate measurement (1–3 mm), not a bridge. It is reported as a number rather than withheld. Candidate follow-up:
   withhold `W0` when the opposite cortex is not in the section (same class of honesty as `H~`).
3. **Mental-foramen region** — 1 of 68 non-canal slices (FDI 43, arc 41.5 mm) ends 2.04 mm short of the jaw border (no canal
   reported). All other outliers (FDI 45/35) turned out to be canal-stopped slices that a plane-restricted canal test misread.
4. Slicer GUI panes (Green view RaySum, 3D viewport framing, print preview dialog) were **not** rendered — this harness is
   headless; it exercises the same functions the GUI calls, but the VTK viewport/Qt dialog themselves are unverified.

### Profiling numbers (OPT-01 / OPT-03)

| Path | Cost on 370×370×169 |
|------|---------------------|
| `sample_panoramic_slab_array` (RaySum, 207 arcs × 240 rows, 25 mm slab) | 0.16–0.27 s (0.8–1.3 ms/column) |
| `render_panoramic_strip` (RGBA, tight-fill, 300 DPI tile) | 0.01–0.04 s |
| `render_3d_mip_card` (full-volume MIP, RGBA, tight crop) | 0.64–0.95 s |
| 8×10 page compose | 0.1–0.2 s |

The panoramic sampler is already interactive on a full-FOV CBCT; the 3D MIP card is the only second-scale path
(one downward `max` over 169 slices + PIL mask composition), so OPT-01 caching is not currently warranted.

---

## 9. Follow-up refinements (UI + frontal W0), 2026-11-24

User feedback after the first merge: frontal teeth still produced a one-sided `W0`, measurement captions sat on the
anatomy, `W3`/`W5` calipers crossed the jaw, `H` ran to the periosteal jaw surface, the footer overlapped annotations,
and the 8×10 band gave the whole width to the OPG.

### Measurement engine (`core/measurements.py`)

| Change | Why |
|--------|-----|
| Crestal `W0` samples a 0.5/1.0/1.5 mm band and keeps the **widest bridged** span | One crestal height reported whichever plate the plane happened to catch |
| New post-pass: when `W0` is a single cortical run < 3 mm and a deeper caliper does span both walls, `W0` is re-anchored to that caliper's two wall u-endpoints at **each wall's own crest rim** | Frontal incisors: the incisal crown's labial/lingual air gap stopped the crestal bridge, collapsing `W0` to one plate |
| `SOCKET_RIM_MAX_V_OFFSET_MM = 2.5` (was a hard-coded 4.5) + vertical bone-support test per rim | Keeps the rim-pair branch near the crest and rejects stray voxels |
| `JAW_END_CORTEX_INSET_MM = 1.5` — the jaw-end stop is inset along `d_apical` | `H` now ends at the **internal** cortical border, never flush with the periosteal surface. Canal-limited stops are untouched |

### Print renderer (`core_print_render.py`)

- Tile canvas is `img_h + footer_h`: the footer has its own band, so it can never cover a caption or a deep basal end-cap.
- A reserved right-hand caption gutter (`max(64, 14·scale)` px) holds the `H`, `W0`, `IAN` and `SINUS` pills, stacked with
  collision avoidance and a short colour stub; nothing is drawn on the tooth or jaw any more.
- `W3`/`W5` are no longer calipers across the anatomy: they render as a compact corner depth-key box with coloured dashes.

### 8×10 layout (`PrintLayoutEngine.py`)

- `_compose_page_image`: the reference band is split 50/50 — OPG left, 3D VR right — and the VR now also renders on the
  grayscale film page; the colour key keeps the full width.
- `_build_8x10_html`: `.pano-section` is a two-column row (`.pano-card` + `.vr-card` at `flex: 1 1 50%`); the VR card left
  the bottom section.

### Verification (same real CT, `verify_real_ct.py`, 20 checks)

- `W0` two-wall sockets: 20/20 bridged (molars 12/12 ≥ 6 mm, W0 median 9.0 mm). 185/205 sections still clip a single wall.
- **Frontal sites: 31 = 7.19, 32 = 6.56, 33 = 8.26, 41 = 6.79, 42 = 7.16, 43 = 7.90 mm** (before: 32 = 1.53, 43 = 0.98).
  New permanent check: `frontal sockets never report one side of the tooth as W0`.
- `H`: 65/68 non-canal slices end inside the jaw, gap p10/p50/p90 = 1.26/2.53/3.15 mm against the 1.5 mm cortical inset;
  IAN clearance unchanged (137/137 clear, min 0.09 mm).
- `python -m unittest discover -s tests` → 167 tests OK (the `H`-to-jaw-base assertions now subtract
  `JAW_END_CORTEX_INSET_MM`, and the tile-size assertion allows the extra footer band).
- Visual: the exported 8×10 page and a 300 DPI cross-section tile were inspected — captions in the gutter, depth key in the
  corner, footer in its own band, OPG/VR side by side.

Residual: one interproximal section at the FDI 43 edge (arc 39.9 mm) still reads a 1.75 mm plate because its buccal plate
top lies below the sampled crest band; the site itself and its neighbours bridge 7.6–7.9 mm.

---

## 10. Step 8: review agent, annotations & tangential sections (2026-11-24)

Big feature pass: a remote review agent, an annotation store, and ridge-parallel sections.

### New core modules (Qt-free, unit-tested)

| Module | Responsibility |
|--------|----------------|
| `core/annotations.py` | `Annotation` + `AnnotationStore`: id, FDI (11-48), arc mm, kind (comment/ai_comment/flag/question/plan_change), severity, author, source, created/edited; add/update/upsert/remove/clear, per-FDI grouping, JSON round-trip |
| `core/agent_review.py` | `AgentEndpointConfig` (URL normalisation for `/`, `/v1`, `/v1/chat/completions`, bare hosts; key redaction), `build_review_payload` (text + image parts), `parse_agent_response` (fenced/prose/brace-balanced JSON, comment + action extraction), `AgentAction` validation, `apply_agent_actions` (whitelist → store/plan), `AgentClient` over `urllib` with `MockTransport`, `context_warnings` |
| `core/tangential.py` | `sample_tangential_section` (plane spanned by arch tangent + occlusal binormal, slab mean/max/min along the buccal normal, row 0 = superior, optional cortical crest line), `crest_drop_mm`, `render_tangential_section` (RGBA with transparent air, crest line, HUD, scale bar, optional footer) |

### GUI (`CBCTPlanner.py`)

- New collapsible **Step 8: AI Agent Review, Annotations & Tangential Sections** (built last, after Step 7), plus an
  8th step-status chip. `agentStepBox` added to all four workspace role profiles in `WorkflowUIHelpers.py`.
- Endpoint form (URL, key with `Password` echo, model, timeout, image toggle/max), `Test connection`, `Save as default`
  (QSettings only), question field, `Send case to agent`, `Apply agent changes`, `Remove agent notes`.
- Annotation table (FDI/Kind/Severity/Author/Text) with add/remove/export; row → id mapping kept Qt-free in
  `_annotationRowIds` so the handlers are testable headless.
- Network work runs on a `threading.Thread` with a 150 ms `QTimer` poll (`_agentThread`, `_agentPollTimer`,
  `_agentWorkerOutcome`); both timers are stopped in `cleanup()`. The tangential view refreshes through a 180 ms
  debounce timer and follows the cross-section cursor.
- `_AgentPlanAdapter` exposes only `add_tooth`/`remove_tooth`; implant diameter edits are reported as skipped instead of
  pretending to change a measurement-derived plan row.
- Context is anonymised (`patient.name = "Anonymous"`), and the key is only ever a header value.

### Verification

- `python -m unittest discover -s tests` → **219 tests OK** (new: `test_agent_review.py` 27, `test_tangential_view.py` 11,
  `test_agent_step_glue.py` 14; `test_autofit_widget.py` updated for the 8th chip).
- `PythonSlicer verify_real_ct.py` → **23/23 PASS**, including on the real ToothFairy3S_0021 CT:
  - tangential section `(101, 141)` at 0.30 mm/px, arc 20.0 mm, 57 % bone pixels, 125 crest samples, crest drop 11.1 mm, 0.03 s;
  - tangential RGBA render with transparent air;
  - full agent round trip with a mock transport: real context + image → 49 KB payload, key only in `Authorization`,
    1 comment parsed, 3 changes applied (comment, annotation, tooth add), 1 invalid op rejected, plan adapter called.
- Not verified: the Qt panel was never rendered (headless environment). Handlers are covered by
  `test_agent_step_glue.py`; the widget construction path is covered by `tests/test_static_attrs.py`.

### Known gaps / follow-ups

- Implant diameter/length edits from the agent are skipped (the plan table is derived from measurements).
- Annotations are not yet printed into the 8x10 film or the HTML report.
- No retry/backoff loop on rate limits (`AgentError.is_retryable` is exposed for that).

---

## 11. MCP server for 3D Slicer (2026-11-25)

Installed [zhaoyouj/mcp-slicer](https://github.com/zhaoyouj/mcp-slicer) 0.3.1 (`uv tool install mcp-slicer`) and
verified it end to end against the local **Slicer 5.12.3** (upstream asks for 5.8+).

### What the package is

Three MCP tools over Slicer's built-in Web Server REST API (`http://localhost:2016/slicer`, hardcoded):
`list_nodes` (`/mrml/names|ids|properties`), `execute_python_code` (`/exec`, raw POST body), `capture_screenshot`
(`/screenshot`, `/threeD`, and a `/exec`-based slice-widget grab). Python >= 3.13, deps `mcp[cli]`, `requests`, `pillow`.

### Repo additions

| File | Purpose |
|------|---------|
| `.mcp.json` | Root MCP client config: `uvx mcp-slicer` (portable, no absolute machine paths) |
| `analysis/mcp_slicer_check/slicer_webserver_boot.py` | Slicer launcher: REST API on **127.0.0.1:2016** with `exec` enabled (the GUI leaves it off), maximises the window, and optionally seeds a probe volume + quit sentinel. Python-only, no firewall prompt |
| `analysis/mcp_slicer_check/verify_mcp_slicer.py` | 11-check harness: boots Slicer with `--additional-module-paths ./CBCTPlanner`, speaks MCP JSON-RPC over stdio, calls every tool, prints a PASS/FAIL table, exits non-zero on failure |

### Findings against Slicer 5.12.3 (11/11 PASS)

- `/exec` accepts a **raw text body** in 5.12 (`if requestBody: source = requestBody`), which is what the package
  sends — the older `__exec__` JSON envelope is not needed. Errors come back as `{"success": false,
  "message": "Server error: division by zero"}` and the tool surfaces them instead of raising.
- `list_nodes` works for all three filter types. `/mrml/properties` parses `node.__str__()`, so keys are the printed
  ones (`Name`, not `name`) — callers filtering by name must use `Name`.
- Screenshots: application 41 KiB; Red slice grab equals the slice *view* widget size (307x68 in this session); 3D from
  axis A returns a valid JPEG. **A restored 965x400 window gives a 300x70 slice capture**, so the launcher maximises
  the window — an agent gets a usable image only when the view is actually big on screen.
- `CBCTPlanner` is registered and visible to the agent (`slicer.util.moduleNames()`), so `execute_python_code` can
  drive this module's logic directly.
- Unchanged upstream limitation: the Slicer URL is hardcoded to `localhost:2016`, so a non-default web-server port
  needs a patched copy (not vendored here).

### Harness notes (bugs that were in the harness, not the product)

- `sys` is not in Slicer's `/exec` globals: probe code must `import sys` itself.
- First launch of a fresh `.exe` can exceed 30 s (antivirus scan) — the initialize timeout is 120 s.

---

## 12. Documentation sync & commit record (2026-11-25)

**Committed:** `9411c7b` — *Add comprehensive unit tests for agent review and annotation handling*.
20 files, **+4 779 / −113**: `CBCTPlanner.py` (+1 020), `core/agent_review.py` (839), `core/annotations.py` (392),
`core/tangential.py` (324), three new test modules (972), `WorkflowUIHelpers.py`, `verify_real_ct.py` (+190),
`analysis/mcp_slicer_check/` (548), `.mcp.json`, plus the measurement/print adjustments from §9.
Working tree is clean on `main`; nothing pushed beyond `main`'s existing upstream.

This session then re-synced the documentation to the code as merged:

| File | Update |
|------|--------|
| `README.md` | Header now reads “v3.0 → v3.1”; new **Roadmap / Future Updates** section (agent, measurement honesty, rendering, MCP); *Running Tests* shows 219 tests and both Slicer harnesses; Step 8 + MCP sections already documented in `9411c7b` |
| `analysis/slicercbctplanner_1/BUSINESS_RULES.md` | Inventory 167→**219** tests / 51 modules; new **RC-07** (review agent + annotation store) and **RC-08** (tangential sections); backlog extended with `BL-13`–`BL-16`; test table replaced with the real per-file counts (16 files, 219); pipeline graph gains the three new core modules; §6 raised to **23/23** with the tangential + agent rows; new **§7 MCP verification record** (11/11) |
| `.ecc/instincts.md` | Three new instincts: untrusted agent output / key handling, tangential plane basis, thread + `QTimer` instead of `QThread` |
| `.ecc/context.json` | Milestone, decisions, completed and next tasks updated to the merged Step 8 + MCP state |

### Checks re-run for this sync

| Check | Result |
|-------|--------|
| `python -m unittest discover -s tests` | **219 tests, OK** (16.8 s) |
| `PythonSlicer verify_real_ct.py` | 23/23 PASS, exit 0 (run before the doc sync; code unchanged since) |
| `python analysis/mcp_slicer_check/verify_mcp_slicer.py` | 11/11 PASS, exit 0 |
| `git status` | only the five documentation files below are modified; no code changed |

---

## 13. Doc-vs-code audit pass & API URL / W0-midline review (2026-11-25)

**Source snapshot:** GitHub zipball of commit `552d2a89b7d08113d80a7cc7f49adf0ef08a2af6` (no `.git` history bundled; no source, test, or config files modified).

### Doc-vs-code contradictions found and fixed

| File | Stale doc claim | Ground truth in code |
|------|-----------------|----------------------|
| `README.md` | 55 labels; 3 workspace roles; ADT in Step 5B; VR Phong `Specular=0.38/Power=28/Ambient=0.22/Diffuse=0.78`; Azure gateway advertised as working out of the box | `core/labels.py` `LABELS_BY_ID` has **77** labels; `WorkflowUIHelpers.WORKSPACE_ROLE_PROFILES` has **4** roles (`full`, `radiologist`, `implant`, `ortho`); ADT is in **Step 7** (`CBCTPlanner.py:1553`); `VRVisualization.py` uses `Ambient=0.25`, `Diffuse=0.88`, `Specular=0.45`, `SpecularPower=26.0` (Vivid Roots VR); Azure `?api-version=...` is currently mangled by `resolved_url()` (`BL-17`) |
| `BUSINESS_RULES.md` | Referenced nonexistent `_stabilize_pca_axis`; `u_ok=±12.0 mm / 3D ROI=20 mm`; stale VR Phong values and preset names | `_detect_alveolar_crest_and_axis` clips PCA on `ridge_mask` (≤ 16 mm apical) with elongated-axis guard; `_select_local_vital_structure` uses `u_ok=(|u|<=11.0) | (|u-u_ref|<=8.5)` and `local_radius_mm=28.0`; `VIVID_3D_STYLE_PRESETS` and `"DentalSegmentator Vivid Roots (VR)"` names/values aligned |
| `.ecc/instincts.md` | No URL-normalization invariant recorded | Appended Instinct 10 covering `urllib.parse.urlsplit` normalization, `is_configured()` parity, and URL secret redaction |
| `.ecc/context.json` | Carried `_stabilize_pca_axis`, `u_ok=±12 mm / 3D ROI=20 mm`, `Power=25`, and hardcoded `C:\Users\hamid\Downloads\risemar_dental_preview.pkl` | Aligned all four with code and added `BL-17` / `BL-18` to `next_tasks` |

### Checks re-run for this audit pass

| Check | Result |
|-------|--------|
| `python3 -m unittest discover -s tests` | **219 tests, OK (skipped=1)** in 10.3 s (`test_real_volume_file_matches_the_label_grid` skipped — external ToothFairy3 NIfTI not present) |
| Per-file test count (`grep -c '    def test_' tests/test_*.py`) | 16 files, sum = **219** (exact match with `BUSINESS_RULES.md` §4) |
| Module / UI / settings counts | **51** `.py` files (**28** in `CBCTPlannerLib/`: 15 top-level + 13 in `CBCTPlannerLib/core/`); **8** wizard steps / **8** step chips; **22** active `CBCTPlanner/*` QSettings keys (+ 1 removed legacy key `autoStandardizeOnLoad`) |
| `PythonSlicer verify_real_ct.py` | **Not verified in this pass** — `PythonSlicer` / 3D Slicer 5.12.3 is not installed in this Linux container (last verified 23/23 on Windows Slicer 5.12.3) |
| `python analysis/mcp_slicer_check/verify_mcp_slicer.py` | **Not verified in this pass** — requires a running 3D Slicer instance (last verified 11/11 on Windows Slicer 5.12.3) |

---

## 14. Consolidated future updates

The same list lives in `README.md` → *Roadmap* and `BUSINESS_RULES.md` → §3; IDs are stable, so a future session can
pick one and go.

### Step 8 (agent, annotations, tangential)
| ID | Item | Why it matters | Effort |
|----|------|----------------|--------|
| `BL-14` | Print annotations/agent comments into the 8×10 film and the HTML report | The store is invisible on the clinical deliverable today | S–M |
| `BL-13` | Apply `set_implant_diameter`/length instead of skipping | Requires a plan override column so a measurement-derived row is not overwritten | M |
| `BL-15` | Retry/backoff on 429/5xx | `AgentError.is_retryable` is exposed but unused; one transient rate limit loses the review | S |
| `BL-17` | Fix `AgentEndpointConfig.resolved_url()` query-string handling, `is_configured()` parity, keyless localhost, URL redaction, and `User-Agent` | Query-string gateways like Azure `?api-version=...` are currently corrupted by string suffixing, and WAFs can block `Python-urllib` | S |
| — | Token/image budget + pre-send size estimate | Up to 8 images; long cases can exceed a context window | S |
| — | Per-case review log (request id, model, latency, applied count) | Auditability of an AI-authored comment on a clinical case | S |
| — | Tangential: multi-slab thickness and a buccal/lingual offset control | Current slab is a single averaged band | S |

### Measurement honesty
| ID | Item | Why it matters | Effort |
|----|------|----------------|--------|
| `BL-12` | Withhold `W0` when the opposite cortex is absent | 185/205 real-CT sections cut one plate and still report 1–3 mm as a crest width | M |
| `BL-18` | Anchor bone-height `H` line start at the `W0` midline midpoint `(pt_l + pt_r) / 2` | Ensures `H` starts visually and geometrically at the center of the bridged `W0` crest span rather than an off-center `crest_uv` | S |
| `BL-01` | 1-site Gaussian smoothing of `W0`/`W3`/`W5` along the arch | Removes isolated inter-root spikes | S |
| `BL-05` | Dynamic `half_v_mm` for tall anterior mandibles | 25.0 mm clips the chin on some scans; 30.0 broke phantom tests | M |
| `BL-06` | PDL gap tolerance validation on real molars | 1.5 mm bridge is untested against thin widened PDL | S |
| — | FDI 43 edge section reads 1.75 mm | Single interproximal outlier; neighbours bridge 7.6–7.9 mm | S |

### Rendering, packaging, MCP
| ID | Item | Why it matters | Effort |
|----|------|----------------|--------|
| `OPT-03` | Pass `PIL.Image`/`ndarray` directly when composing the film | Avoids `RGBA → base64 → PIL.Image` round-trips | S |
| `OPT-04` | `scipy.ndimage.label` flood-fill in `finalize_transparent_png_bytes` | Removes the Python per-pixel BFS on high-DPI captures | S |
| `BL-03`/`BL-04` | RaySum scale/gamma exploration; user-adjustable VR `Zoom` | Rendering taste, no correctness impact | S |
| `BL-16` | Vendored `mcp-slicer` with `SLICER_WEB_SERVER_URL` | Port 2016 is hardcoded upstream | S |
| `BL-07`/`BL-09` | Test-helper extraction; trim unused `core/diagnostics.py` | Maintenance only | S |
| `OPT-05` | Structured (JSON/Parquet) batch manifest for cohort QA | Phase 7 stores `.npy` only | M |

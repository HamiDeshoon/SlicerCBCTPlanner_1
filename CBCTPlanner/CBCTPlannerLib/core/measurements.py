"""
Clinical Dental CBCT Measurement Engine — Single Source of Truth.

Performs millimeter-precise calculations in the arch-frame cross-section plane from real
volume & segmentation masks ONLY. The alveolar crest, `W0` and every buccolingual width come
from jaw-bone voxels only; the ridge envelope (jaw bone plus the in-socket root within
`ENVELOPE_HALF_WIDTH_MM`) is used solely for the available-bone-height ray. Never returns
fabricated or hardcoded fallback clinical values, and never substitutes arbitrary
image-intensity thresholds for missing jaw segmentation.
All Misch bone density classifications are explicitly labeled approximate because CBCT gray values
are not calibrated Hounsfield Units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .arch_detector import (
    DentalArch,
    compute_fdi_arc_positions_mm,
    project_point_to_polyline_arc_mm,
)
from .labels import (
    CANAL_IDS,
    DENTALSEGMENTATOR_CANAL_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    get_fdi_quadrant,
    get_ipsilateral_canal_ids,
    get_ipsilateral_sinus_ids,
    get_jaw_label_id,
    get_tooth_group_label_id,
    get_tooth_label,
    infer_mask_is_5class,
    is_mandibular_fdi,
    jaw_key_from_text,
    resolve_fdi_number,
)
from .tooth_localizer import ToothInfo, ToothLocalizer
from .volume_loader import CBCTVolume

MISCH_APPROXIMATE_DISCLAIMER = "approximate (CBCT gray values are not calibrated HU)"
IMPLANT_SUGGESTION_DISCLAIMER = "Planning assistance only — clinician verification required"

# Transparent measurement provenance constants (Part 7.3)
PROVENANCE_LABEL_DERIVED = "label_derived"
PROVENANCE_INTERPOLATED = "interpolated_boundary"
PROVENANCE_ESTIMATED = "estimated"
PROVENANCE_UNAVAILABLE = "unavailable"

# Measurement quality / reliability status
QUALITY_VALID = "valid"
QUALITY_WARNING = "warning"
QUALITY_ESTIMATED = "estimated"
QUALITY_UNAVAILABLE = "unavailable"

# Clinical safety margins (preserved consistently across GUI, gallery, and print)
STANDARD_IMPLANT_DIAMETERS_MM: Tuple[float, ...] = (3.0, 3.5, 3.75, 4.0, 4.3, 4.5, 5.0)
STANDARD_IMPLANT_LENGTHS_MM: Tuple[float, ...] = (6.0, 8.0, 10.0, 11.5, 13.0)
IMPLANT_BUCCOLINGUAL_MARGIN_MM: float = 3.0  # 1.5 mm buccal + 1.5 mm lingual
IMPLANT_MESIODISTAL_MARGIN_MM: float = 3.0  # 1.5 mm mesial + 1.5 mm distal
IMPLANT_APICAL_SAFETY_MARGIN_MM: float = 2.0

logger = logging.getLogger("CBCTPlanner.measurements")

# A measurable alveolar ridge sits inside the cross-section field. A bone region that reaches
# the lateral border, or spans most of the field, means the jaw mask is missing or misaligned
# at this site: measuring it would print invented millimetres on the clinical film.
_CS_BORDER_TOL_MM = 0.75

# The height caliper stops at the INTERNAL border of the basal cortex (the medullary side of the
# inferior cortical shell) instead of the periosteal surface, so an osteotomy is never planned
# flush against the outer cortical plate.
JAW_END_CORTEX_INSET_MM = 1.5

# Front teeth often carry one crest rim several millimetres deeper than the other (thin buccal
# plate). Bridging is allowed across that height difference as long as both rims rest on real bone.
SOCKET_RIM_MAX_V_OFFSET_MM = 2.5
_MAX_RIDGE_WIDTH_FRACTION = 0.80
_MAX_RIDGE_HEIGHT_FRACTION = 0.90

# The height-ray envelope adds the in-socket root inside this buccolingual window.
# Crest, W0 and the buccolingual widths must use the jaw-bone-only mask instead.
ENVELOPE_HALF_WIDTH_MM = 6.0

# An osteotomy entry (alveolar crest) that moves further than this between neighbouring
# slices is interpolated from its stable neighbours instead of being drawn as a step.
MAX_ENTRY_JUMP_MM = 2.5

# Recorded whenever a gate withholds a metric, so a rendered `--` stays distinguishable
# from missing segmentation and from a metric that failed outright.
WITHHELD_NO_SEGMENTATION = "no segmentation at site"
WITHHELD_NO_BONE = "no jaw bone in cross-section"
WITHHELD_NO_MANDIBLE_SEG = "missing mandibular jaw segmentation at site"
WITHHELD_NO_MAXILLA_SEG = "missing maxillary jaw segmentation at site"
WITHHELD_NO_CANAL_SEG = "ipsilateral inferior alveolar canal segmentation unavailable in local region"
WITHHELD_NO_SINUS_SEG = "ipsilateral maxillary sinus segmentation unavailable in local region"
WITHHELD_WIDTH_BORDER = "widths withheld: jaw mask touches cross-section border"
WITHHELD_WIDTH_SPAN = "widths withheld: jaw mask spans the cross-section field"
WITHHELD_HEIGHT_SPAN = "height withheld: jaw mask exceeds cross-section height"
WITHHELD_NO_CREST = "crest not resolved in cross-section"
WITHHELD_W0_SINGLE_CORTEX = (
    "W0 withheld: cross-section cuts only one cortical plate (missing opposite cortex)"
)
WITHHELD_NO_ARCH = "no fitted dental arch for this jaw; fit or draw the arch first"
WITHHELD_NO_SITE = "no tooth segmentation and no fitted arch at this planning site"
WITHHELD_METRIC_FAILED = "metric computation failed"

_WITHHELD_BADGES = (
    ("no segmentation", "no seg"),
    ("missing mandibular", "no jaw seg"),
    ("missing maxillary", "no jaw seg"),
    ("no jaw bone", "no bone"),
    ("touches cross-section border", "W: border"),
    ("spans the cross-section", "W: field-wide"),
    ("exceeds cross-section height", "H: field-tall"),
    ("crest not resolved", "no crest"),
    ("only one cortical plate", "W0: 1-cortex"),
    ("metric computation failed", "metric failed"),
)


def withheld_badge(reason: Optional[str]) -> str:
    """Return a short tile badge for a withheld-measurement `reason` (empty when none)."""
    text = str(reason or "").strip().lower()
    if not text:
        return ""
    for needle, badge in _WITHHELD_BADGES:
        if needle in text:
            return badge
    return "withheld"


def suggest_implant_dimensions(
    crest_w: Optional[float],
    mid_w: Optional[float],
    bone_height: Optional[float],
    mesiodistal: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Single source of truth for conservative implant dimension suggestions.

    Preserves clinical safety deductions:
      - Buccolingual: 3.0 mm total (1.5 mm buccal + 1.5 mm lingual)
      - Mesiodistal: 3.0 mm total (1.5 mm per adjacent tooth)
      - Apical: 2.0 mm clearance coronal to canal / sinus / basal limit
    Returns `(diameter_mm, length_mm, angulation_deg)` or `(None, None, None)`.
    Planning assistance only — clinician verification required.
    """
    if crest_w is None or bone_height is None:
        return None, None, None
    eff_w = min(float(crest_w), float(mid_w)) if mid_w is not None else float(crest_w)
    max_allowed_diam = eff_w - IMPLANT_BUCCOLINGUAL_MARGIN_MM
    if mesiodistal is not None:
        max_allowed_diam = min(max_allowed_diam, float(mesiodistal) - IMPLANT_MESIODISTAL_MARGIN_MM)

    valid_diams = [d for d in STANDARD_IMPLANT_DIAMETERS_MM if d <= max_allowed_diam]
    chosen_diam = valid_diams[-1] if valid_diams else None

    safe_height = float(bone_height) - IMPLANT_APICAL_SAFETY_MARGIN_MM
    valid_lens = [length for length in STANDARD_IMPLANT_LENGTHS_MM if length <= safe_height]
    chosen_len = valid_lens[-1] if valid_lens else None

    if chosen_diam is None or chosen_len is None:
        return None, None, None
    return float(chosen_diam), float(chosen_len), 0.0


def _uv_to_world(
    P: np.ndarray, N: np.ndarray, B: np.ndarray, u: float, v: float
) -> Tuple[float, float, float]:
    """Map 2D cross-section `(u, v)` in mm to 3D physical RAS coordinates."""
    pt = (
        np.asarray(P, dtype=np.float64)
        + float(u) * np.asarray(N, dtype=np.float64)
        + float(v) * np.asarray(B, dtype=np.float64)
    )
    return (float(pt[0]), float(pt[1]), float(pt[2]))


def coronal_offset_along_axis(
    h_start_uv: Sequence[float],
    h_end_uv: Sequence[float],
    w0_start_uv: Sequence[float],
    w0_end_uv: Sequence[float],
) -> float:
    """Signed distance (mm) of `h_start_uv` coronal to `W0` midpoint along the H ray."""
    mid_u = 0.5 * (float(w0_start_uv[0]) + float(w0_end_uv[0]))
    mid_v = 0.5 * (float(w0_start_uv[1]) + float(w0_end_uv[1]))
    du = mid_u - float(h_end_uv[0])
    dv = mid_v - float(h_end_uv[1])
    seg_len = math.hypot(du, dv)
    if seg_len <= 1e-4:
        return float(math.hypot(float(h_start_uv[0]) - mid_u, float(h_start_uv[1]) - mid_v))
    return ((float(h_start_uv[0]) - mid_u) * du + (float(h_start_uv[1]) - mid_v) * dv) / seg_len


def reanchor_h_start_to_w0(
    h_start_uv: Sequence[float],
    h_end_uv: Sequence[float],
    w0_start_uv: Sequence[float],
    w0_end_uv: Sequence[float],
    coronal_offset_mm: Optional[float] = None,
) -> Tuple[float, float]:
    """Keep H passing through W0's midpoint while preserving its coronal crest offset."""
    mid_u = 0.5 * (float(w0_start_uv[0]) + float(w0_end_uv[0]))
    mid_v = 0.5 * (float(w0_start_uv[1]) + float(w0_end_uv[1]))
    offset = (
        coronal_offset_along_axis(h_start_uv, h_end_uv, w0_start_uv, w0_end_uv)
        if coronal_offset_mm is None
        else float(coronal_offset_mm)
    )
    if abs(offset) <= 1e-4:
        return (round(mid_u, 2), round(mid_v, 2))
    du = mid_u - float(h_end_uv[0])
    dv = mid_v - float(h_end_uv[1])
    seg_len = math.hypot(du, dv)
    if seg_len <= 1e-4:
        return (round(mid_u, 2), round(mid_v, 2))
    return (
        round(mid_u + offset * (du / seg_len), 2),
        round(mid_v + offset * (dv / seg_len), 2),
    )


def _same_jaw_tooth_mask(
    mask_cs: np.ndarray,
    is_mandible: bool,
    is_5class: bool,
    tooth_id: Optional[int] = None,
) -> np.ndarray:
    """Return boolean mask of same-jaw teeth and pulps in `mask_cs` (never opposing teeth)."""
    ids: List[int] = []
    if is_5class:
        ids.append(get_tooth_group_label_id(is_mandible, is_5class=True))
    else:
        base = MANDIBULAR_TEETH_IDS if is_mandible else MAXILLARY_TEETH_IDS
        ids.extend(int(t) for t in base)
        ids.extend(100 + int(t) for t in base)
    if tooth_id is not None:
        tid = int(tooth_id)
        if tid not in ids:
            ids.append(tid)
        if not is_5class and 11 <= tid <= 48 and (100 + tid) not in ids:
            ids.append(100 + tid)
    if not ids:
        return np.zeros(mask_cs.shape, dtype=bool)
    return np.isin(mask_cs, ids)


def _ridge_envelope_mask(
    mask_cs: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    jaw_id: int,
    tooth_id: Optional[int],
    is_mandible: bool,
    tooth_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Height-ray envelope: jawbone label mask plus the in-socket root inside `|u| <= 6 mm` apical to jaw crest.

    Only the available-bone-height ray (and primary ridge socket connectivity) may use this mask.
    The alveolar crest, `W0` and every buccolingual width must be derived from `mask_cs == jaw_id`
    (never measuring a tooth crown or single-plate root wall as bone).

    Never falls back to arbitrary image-intensity thresholds when jawbone segmentation is
    missing: returns an empty boolean mask so callers explicitly withhold measurements.
    """
    jaw_mask = mask_cs == jaw_id
    if not jaw_mask.any():
        return np.zeros_like(jaw_mask, dtype=bool)
    envelope = jaw_mask.copy()
    t_mask = (
        np.asarray(tooth_mask, dtype=bool)
        if tooth_mask is not None
        else ((mask_cs == int(tooth_id)) if tooth_id is not None else np.zeros_like(jaw_mask, dtype=bool))
    )
    if t_mask.any():
        # Guard occlusal bound against tiny stray bone speckles above the crown.
        jaw_clean = _extract_primary_ridge_component(jaw_mask, u_coords, v_coords)
        jaw_ref = jaw_clean if jaw_clean.any() else jaw_mask
        v_jaw = v_coords[np.flatnonzero(jaw_ref.any(axis=1))]
        occlusal_v = float(np.max(v_jaw)) if is_mandible else float(np.min(v_jaw))
        in_socket = t_mask & (np.abs(u_coords)[None, :] <= ENVELOPE_HALF_WIDTH_MM)
        occlusal_side = v_coords[:, None] <= occlusal_v if is_mandible else v_coords[:, None] >= occlusal_v
        envelope |= in_socket & occlusal_side
    return envelope


def _extract_primary_ridge_component(
    bone_mask: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> np.ndarray:
    """Isolate the primary connected alveolar ridge component, rejecting stray voxels & islands.

    Uses 8-connected 2D labeling with physical spacing-aware noise rejection while preserving
    thin legitimate cortical walls attached to the ridge.
    """
    mask = np.asarray(bone_mask, dtype=bool)
    if not mask.any():
        return mask

    du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
    dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
    px_area_mm2 = max(1e-6, du * dv)
    min_area_px = max(4, int(round(1.0 / px_area_mm2)))

    try:
        from scipy import ndimage  # type: ignore

        labeled, n_comp = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int32))
    except ImportError:
        return mask

    if n_comp <= 1:
        if int(np.count_nonzero(mask)) < max(2, min_area_px // 2):
            return np.zeros_like(mask, dtype=bool)
        return mask

    counts = np.bincount(labeled.ravel())
    central_u = np.abs(u_coords) <= 6.0
    best_id = 0
    best_score = (-1, -1.0, -1)

    for comp_id in range(1, n_comp + 1):
        area_px = int(counts[comp_id])
        if area_px < min_area_px:
            continue
        comp = labeled == comp_id
        central_hits = int(np.count_nonzero(comp[:, central_u])) if central_u.any() else 0
        # Distance of component centroid to planning axis u=0
        c_rows, c_cols = np.where(comp)
        u_dist = float(np.min(np.abs(u_coords[c_cols]))) if c_cols.size else 1e6
        # Prefer components overlapping the central planning corridor, then closest to u=0, then larger area
        in_corridor = 1 if central_hits > 0 else 0
        score = (in_corridor, -u_dist if not in_corridor else float(central_hits), area_px)
        if score > best_score:
            best_score = score
            best_id = comp_id

    if best_id == 0:
        # Relax area threshold only if all components were small (very thin cortical bone)
        valid_ids = [cid for cid in range(1, n_comp + 1) if int(counts[cid]) >= 2]
        if not valid_ids:
            return np.zeros_like(mask, dtype=bool)
        best_id = max(valid_ids, key=lambda cid: int(counts[cid]))

    return labeled == best_id


def _join_reasons(*reasons: Optional[str]) -> Optional[str]:
    """Join distinct non-empty withheld reasons with `; ` (None when nothing was withheld)."""
    unique: List[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return "; ".join(unique) or None


def _ridge_region_withheld_reasons(
    bone: np.ndarray, u_coords: np.ndarray, v_coords: np.ndarray
) -> Tuple[Optional[str], Optional[str]]:
    """Return `(width_reason, height_reason)`; each is None when that dimension is measurable.

    The width reason names the failing gate (field border vs. field-spanning mask) and the
    height reason covers a region taller than the cross-section field. An empty region
    withholds both dimensions as missing bone.
    """
    if not bone.any():
        return WITHHELD_NO_BONE, WITHHELD_NO_BONE
    u_bone = u_coords[np.flatnonzero(bone.any(axis=0))]
    v_bone = v_coords[np.flatnonzero(bone.any(axis=1))]
    u_lo, u_hi = float(u_coords.min()), float(u_coords.max())
    v_lo, v_hi = float(v_coords.min()), float(v_coords.max())
    width_reason: Optional[str] = None
    touches_left = bool(u_bone.min() <= u_lo + _CS_BORDER_TOL_MM)
    touches_right = bool(u_bone.max() >= u_hi - _CS_BORDER_TOL_MM)
    max_row_span = 0.0
    for r in np.flatnonzero(bone.any(axis=1)):
        cols = np.flatnonzero(bone[r, :])
        if cols.size >= 2:
            span = float(u_coords[cols[-1]] - u_coords[cols[0]])
            if span > max_row_span:
                max_row_span = span
    if touches_left and touches_right:
        width_reason = WITHHELD_WIDTH_BORDER
    elif max_row_span > _MAX_RIDGE_WIDTH_FRACTION * (u_hi - u_lo):
        width_reason = WITHHELD_WIDTH_SPAN
    height_reason = (
        None
        if v_bone.max() - v_bone.min() <= _MAX_RIDGE_HEIGHT_FRACTION * (v_hi - v_lo)
        else WITHHELD_HEIGHT_SPAN
    )
    return width_reason, height_reason


@dataclass
class MeasurementRecord:
    """Canonical measurement geometry + value record (Single Source of Truth)."""

    measurement_type: str
    value_mm: Optional[float]
    start_world_mm: Optional[Tuple[float, float, float]] = None
    end_world_mm: Optional[Tuple[float, float, float]] = None
    start_uv_mm: Optional[Tuple[float, float]] = None
    end_uv_mm: Optional[Tuple[float, float]] = None
    source_labels: Tuple[int, ...] = ()
    provenance: str = PROVENANCE_UNAVAILABLE
    quality: str = QUALITY_UNAVAILABLE
    withheld_reason: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.value_mm is not None and self.start_uv_mm is not None and self.end_uv_mm is not None


@dataclass
class ImplantSiteMeasurement:
    """Quantitative clinical metrics for dental implant planning at a specific tooth site."""

    tooth_fdi: str
    bone_height_mm: Optional[float]
    bone_width_at_depths_mm: Dict[float, Optional[float]]
    crest_width_mm: Optional[float]
    mid_root_width_mm: Optional[float]
    apical_width_mm: Optional[float]
    mesiodistal_space_mm: Optional[float]
    mean_density_hu: Optional[float]
    misch_density_class: Optional[str]
    distance_to_vital_structure_mm: Optional[float]
    vital_structure_name: str
    safety_margin_mm: Optional[float]
    suggested_diameter_mm: Optional[float]
    suggested_length_mm: Optional[float]
    suggested_angulation_deg: Optional[float]
    is_ai_suggested: bool = True
    ai_suggestion_confirmed: bool = False
    site_position_source: str = "segmentation"
    is_approximate_site: bool = False
    buccal_bone_thickness_mm: Optional[float] = None
    lingual_bone_thickness_mm: Optional[float] = None
    socket_depth_mm: Optional[float] = None
    apical_bone_reserve_mm: Optional[float] = None
    jumping_distance_mm: Optional[float] = None
    immediate_placement_type: Optional[str] = None
    immediate_placement_recommendation: str = ""
    subantral_bone_height_mm: Optional[float] = None
    sinus_lift_recommendation: Optional[str] = None
    insertion_torque_ncm: Optional[str] = None
    tooth_length_mm: Optional[float] = None
    crown_length_mm: Optional[float] = None
    root_length_mm: Optional[float] = None
    crown_to_root_ratio: Optional[float] = None
    tooth_width_md_mm: Optional[float] = None
    tooth_width_bl_mm: Optional[float] = None
    cervical_width_md_mm: Optional[float] = None
    cervical_width_bl_mm: Optional[float] = None
    canal_entrance_depth_mm: Optional[float] = None
    canal_entrance_coord_mm: Optional[List[float]] = None
    apical_foramen_coord_mm: Optional[List[float]] = None
    nnt_density_summary: Dict[str, Any] = field(default_factory=dict)
    # Separate anatomical measurements & provenance (Parts 1.1, 1.2, 7.3)
    root_extent_mm: Optional[float] = None
    vital_clearance_mm: Optional[float] = None
    measurements: Dict[str, MeasurementRecord] = field(default_factory=dict)
    provenance: str = PROVENANCE_LABEL_DERIVED
    quality_status: str = QUALITY_VALID
    implant_suggestion_disclaimer: str = IMPLANT_SUGGESTION_DISCLAIMER
    # Why a metric is `None`; empty when every requested metric was measured honestly.
    withheld_reason: Optional[str] = None


@dataclass
class CrossSectionData:
    """One resampled arch cross-section plus its canonical engine-computed geometry."""

    u_coords: np.ndarray
    v_coords: np.ndarray
    mask_cs: np.ndarray
    img_cs: np.ndarray
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    crest_v: Optional[float]
    width_segments: Dict[float, Tuple[float, float]]
    canal_centroid_uv: Optional[Tuple[float, float]]
    basal_v: Optional[float] = None
    root_apex_v: Optional[float] = None
    sinus_floor_uv: Optional[List[Tuple[float, float]]] = None
    ridge_axis_angle_deg: Optional[float] = None
    buccal_plate_mm: Optional[float] = None
    lingual_plate_mm: Optional[float] = None
    height_u: float = 0.0
    is_mandible: bool = True
    # Canonical engine geometry consumed verbatim by renderers (Part 1.1 & Part 4.1)
    crest_uv: Optional[Tuple[float, float]] = None
    basal_uv: Optional[Tuple[float, float]] = None
    crest_profile_uv: Optional[List[Tuple[float, float]]] = None
    width_endpoints_uv: Dict[float, Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=dict
    )
    root_start_uv: Optional[Tuple[float, float]] = None
    root_apex_uv: Optional[Tuple[float, float]] = None
    vital_boundary_uv: Optional[Tuple[float, float]] = None
    vital_contour_uv: Optional[List[Tuple[float, float]]] = None
    vital_structure_name: Optional[str] = None
    vital_withheld_reason: Optional[str] = None
    measurements: Dict[str, MeasurementRecord] = field(default_factory=dict)
    suggested_implant_diameter_mm: Optional[float] = None
    suggested_implant_length_mm: Optional[float] = None
    suggested_implant_disclaimer: str = IMPLANT_SUGGESTION_DISCLAIMER
    misch_code: Optional[str] = None
    density_disclaimer: str = MISCH_APPROXIMATE_DISCLAIMER
    provenance: str = PROVENANCE_UNAVAILABLE
    quality_status: str = QUALITY_UNAVAILABLE
    # Why this cross-section withheld a metric (implausible ridge, no crest, failure).
    withheld_reason: Optional[str] = None


class MeasurementEngine:
    """Master engine for calculating dental CBCT clinical metrics in the arch-frame cross-section."""

    def __init__(
        self,
        volume: CBCTVolume,
        mask: np.ndarray,
        arch: Optional[DentalArch] = None,
        arches: Optional[Dict[str, DentalArch]] = None,
        is_dental_segmentator_5class: bool = False,
        extract_teeth_eagerly: bool = True,
    ):
        self.volume = volume
        self.mask = mask
        self.arch = arch
        self.arches: Dict[str, DentalArch] = dict(arches or {})
        self.is_dental_segmentator_5class = is_dental_segmentator_5class
        self.localizer = ToothLocalizer(volume, mask, arch=arch, arches=self.arches)
        self.teeth: Dict[str, ToothInfo] = (
            {}
            if (is_dental_segmentator_5class or not extract_teeth_eagerly)
            else self.localizer.extract_all_teeth()
        )
        self._tooth_span_cache: Dict[Tuple[str, int], Optional[Tuple[float, float]]] = {}
        self._site_cache: Dict[Tuple[str, float], Optional[ImplantSiteMeasurement]] = {}
        self._arc_cache: Dict[
            Tuple[str, float, Optional[str]],
            Tuple[Optional[ImplantSiteMeasurement], Optional[CrossSectionData]],
        ] = {}
        self._present_labels_cache: Optional[set] = None
        self.last_cross_sections: Dict[str, CrossSectionData] = {}
        # Reason recorded by the most recent site/cross-section measurement (None when all metrics measured).
        self.last_withheld_reason: Optional[str] = None

    def invalidate_caches(self, *, mask_changed: bool = True) -> None:
        """Clear cached measurements, spans, and (optionally) mask-derived distance transforms."""
        for attr in ("_site_cache", "_arc_cache", "_tooth_span_cache", "last_cross_sections"):
            c = getattr(self, attr, None)
            if c is not None:
                c.clear()
        if mask_changed:
            self._present_labels_cache = None
            if getattr(self, "localizer", None) is not None:
                self.localizer._label_slices = None
            teeth = getattr(self, "teeth", None)
            if teeth is not None:
                teeth.clear()

    def _jaw_id(self, is_mandible: bool) -> int:
        return get_jaw_label_id(is_mandible, is_5class=self.is_dental_segmentator_5class)

    def _vital_ids(self, quadrant: int, is_mandible: bool) -> List[int]:
        """Return strictly ipsilateral vital-structure label IDs for the given quadrant."""
        if is_mandible:
            return get_ipsilateral_canal_ids(quadrant, is_5class=self.is_dental_segmentator_5class)
        return get_ipsilateral_sinus_ids(quadrant, is_5class=self.is_dental_segmentator_5class)

    def _arch_for_jaw(self, is_mandible: bool) -> Optional[DentalArch]:
        return self.arches.get("lower" if is_mandible else "upper") or self.arch

    def _present_label_ids(self) -> set:
        """Return set of non-zero label IDs present in `self.mask` using tight bbox pass."""
        if self._present_labels_cache is None:
            slices = self.localizer._get_label_slices()
            self._present_labels_cache = set(slices.keys())
        return self._present_labels_cache

    def _tooth_bearing_span_mm(
        self, arch: Optional[DentalArch], jaw_type: str
    ) -> Optional[Tuple[float, float]]:
        """Return the arc interval actually occupied by tooth voxels (5-class masks)."""
        if arch is None or not self.is_dental_segmentator_5class:
            return None
        key = (jaw_type, id(arch))
        if key in self._tooth_span_cache:
            return self._tooth_span_cache[key]
        is_mand = jaw_type.lower().startswith("low")
        label = get_tooth_group_label_id(is_mand, is_5class=True)
        slc = self.localizer._get_label_slices().get(label)
        if slc is None:
            self._tooth_span_cache[key] = None
            return None
        sub = self.mask[slc] == label
        vox_local = np.argwhere(sub)
        if len(vox_local) < 20:
            self._tooth_span_cache[key] = None
            return None
        step = max(1, len(vox_local) // 4000)
        vox = vox_local[::step].copy()
        vox[:, 0] += int(slc[0].start or 0)
        vox[:, 1] += int(slc[1].start or 0)
        vox[:, 2] += int(slc[2].start or 0)
        world = self.volume.voxel_to_world(vox[:, ::-1])  # zyx -> xyz
        arcs = np.asarray(
            [project_point_to_polyline_arc_mm(arch.pts_dense, arch.cum_arc_lengths, pt)[0] for pt in world],
            dtype=np.float64,
        )
        if len(arcs) < 10:
            self._tooth_span_cache[key] = None
            return None
        lo, hi = np.percentile(arcs, [0.5, 99.5])
        span = (float(lo), float(hi)) if hi - lo >= 20.0 else None
        self._tooth_span_cache[key] = span
        return span

    def set_arches(self, arches: Dict[str, DentalArch], arch: Optional[DentalArch] = None) -> None:
        """Update arch geometry without rebuilding the mask or tooth localizer."""
        self.invalidate_caches(mask_changed=False)
        self.arches = dict(arches)
        if arch is not None:
            self.arch = arch
        else:
            self.arch = self.arches.get("lower") or self.arches.get("upper") or self.arch
        localizer = getattr(self, "localizer", None)
        if localizer is not None:
            localizer.arches = self.arches
            localizer.arch = self.arch

    def _arch_for_fdi(self, fdi: str) -> Optional[DentalArch]:
        info = get_tooth_label(fdi)
        if info is None:
            return self.arch
        return self._arch_for_jaw(is_mandibular_fdi(fdi))

    def _get_tooth_info(self, fdi: str) -> Optional[ToothInfo]:
        fdi_str = resolve_fdi_number(fdi)
        if fdi_str is None:
            return None
        if fdi_str in self.teeth:
            return self.teeth[fdi_str]
        if not self.is_dental_segmentator_5class:
            extracted = self.localizer.extract_tooth(fdi_str)
            if extracted is not None:
                self.teeth[fdi_str] = extracted
            return extracted
        return None

    def _get_cross_section_frame(
        self, site_center_mm: np.ndarray, is_mandible: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return orthonormal `(P, T, N, B)` frame at `site_center_mm` for cross-section resampling."""
        arch = self._arch_for_jaw(is_mandible)
        if arch is not None:
            t_param, _ = arch.find_closest_parameter(site_center_mm)
            frame = arch.get_frame(t_param)
            return np.asarray(site_center_mm, dtype=np.float64), frame.tangent, frame.normal, frame.binormal
        return (
            np.asarray(site_center_mm, dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

    def _jaw_v_bounds_at_frame(
        self,
        P: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        *,
        is_mandible: bool,
        base_half_v_mm: float = 25.0,
        max_half_v_mm: float = 55.0,
        step_mm: float = 0.25,
        margin_mm: float = 4.5,
    ) -> Tuple[float, float]:
        """Compute `(v_min, v_max)` bounds along binormal `B` so the full mandible/maxilla fits unclipped."""
        v_lo = -float(base_half_v_mm)
        v_hi = float(base_half_v_mm)
        jaw_id = self._jaw_id(is_mandible)
        slc = self.localizer._get_label_slices().get(int(jaw_id))
        if slc is None or self.mask is None or self.volume is None:
            return v_lo, v_hi

        sz, sy, sx = self.volume.spacing
        vox = self.volume.world_to_voxel(P)
        vx, vy = int(round(vox[0])), int(round(vox[1]))
        rx = max(4, int(round(16.0 / max(1e-3, sx))))
        ry = max(4, int(round(16.0 / max(1e-3, sy))))
        z0, z1 = int(slc[0].start or 0), int(slc[0].stop or self.mask.shape[0])
        y0 = max(int(slc[1].start or 0), vy - ry)
        y1 = min(int(slc[1].stop or self.mask.shape[1]), vy + ry + 1)
        x0 = max(int(slc[2].start or 0), vx - rx)
        x1 = min(int(slc[2].stop or self.mask.shape[2]), vx + rx + 1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return v_lo, v_hi

        # Do not expand dynamically if the jaw label spans the entire scan FOV from border to border
        # (e.g. pathological full-volume masks that must still trip height/width plausibility gates).
        slc_z0, slc_z1 = int(slc[0].start or 0), int(slc[0].stop or self.mask.shape[0])
        slc_y0, slc_y1 = int(slc[1].start or 0), int(slc[1].stop or self.mask.shape[1])
        if (slc_z1 - slc_z0) >= self.mask.shape[0] - 2 and (slc_y1 - slc_y0) >= self.mask.shape[1] - 2:
            return v_lo, v_hi

        sub = self.mask[z0:z1, y0:y1, x0:x1] == int(jaw_id)
        if not np.any(sub):
            return v_lo, v_hi
        lz, ly, lx = np.where(sub)
        step = max(1, len(lz) // 4000)
        vox_pts = np.column_stack([lx[::step] + x0, ly[::step] + y0, lz[::step] + z0])
        world_pts = self.volume.voxel_to_world(vox_pts)
        rel = world_pts - np.asarray(P, dtype=np.float64)[None, :]
        u_vals = rel @ np.asarray(N, dtype=np.float64)
        in_plane = np.abs(u_vals) <= 15.0
        if int(np.count_nonzero(in_plane)) < 6:
            return v_lo, v_hi
        v_vals = rel[in_plane] @ np.asarray(B, dtype=np.float64)
        jaw_min_v = float(np.min(v_vals)) - float(margin_mm)
        jaw_max_v = float(np.max(v_vals)) + float(margin_mm)

        def _snap_neg(val: float) -> float:
            mag = min(float(max_half_v_mm), max(float(base_half_v_mm), abs(min(0.0, val))))
            return -float(math.ceil(mag / max(1e-3, step_mm)) * step_mm)

        def _snap_pos(val: float) -> float:
            mag = min(float(max_half_v_mm), max(float(base_half_v_mm), max(0.0, val)))
            return float(math.ceil(mag / max(1e-3, step_mm)) * step_mm)

        if is_mandible:
            if jaw_min_v < v_lo:
                v_lo = _snap_neg(jaw_min_v)
            if jaw_max_v > v_hi:
                v_hi = _snap_pos(jaw_max_v)
        else:
            if jaw_max_v > v_hi:
                v_hi = _snap_pos(jaw_max_v)
            if jaw_min_v < v_lo:
                v_lo = _snap_neg(jaw_min_v)
        return v_lo, v_hi

    def _resample_arch_cross_section(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        half_u_mm: float = 15.0,
        half_v_mm: float = 25.0,
        step_mm: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resample label mask and image intensity on a 2D cross-section grid perpendicular to arch tangent."""
        P, T, N, B = self._get_cross_section_frame(site_center_mm, is_mandible)
        v_lo, v_hi = self._jaw_v_bounds_at_frame(
            P, N, B, is_mandible=is_mandible, base_half_v_mm=half_v_mm, step_mm=step_mm
        )
        n_u = int(round(2.0 * half_u_mm / step_mm)) + 1
        n_v = int(round((v_hi - v_lo) / step_mm)) + 1
        u_coords = np.linspace(-half_u_mm, half_u_mm, n_u)
        v_coords = np.linspace(v_lo, v_hi, n_v)

        uu, vv = np.meshgrid(u_coords, v_coords)  # shape (n_v, n_u)
        world_grid = P[None, None, :] + uu[:, :, None] * N[None, None, :] + vv[:, :, None] * B[None, None, :]
        mask_cs = self.volume.sample_mask_world_points(self.mask, world_grid)
        img_cs = self.volume.sample_world_points(world_grid, order=1, cval=-1000.0)
        return u_coords, v_coords, mask_cs, img_cs, P, N, B

    def _site_center_with_provenance(
        self, fdi: str, offset_mm: float = 0.0
    ) -> Optional[Tuple[np.ndarray, str, bool]]:
        """Resolve the arch cross-section center for `fdi` plus its provenance."""
        fdi_str = resolve_fdi_number(fdi)
        if not fdi_str:
            return None
        arch = self._arch_for_fdi(fdi_str)
        tooth = self._get_tooth_info(fdi_str)
        if tooth is not None:
            base_site_mm = np.asarray(tooth.centroid_mm, dtype=np.float64)
            site_source = "segmentation"
            is_approx_site = False
        elif arch is not None:
            base_site_mm, site_source, is_approx_site = self._estimate_missing_tooth_site_with_provenance(
                fdi_str
            )
        else:
            return None

        if offset_mm != 0.0 and arch is not None:
            t_center, _ = arch.find_closest_parameter(base_site_mm)
            span = max(float(arch.total_arc_length), 1e-6)
            t_target = float(np.clip(t_center + (offset_mm / span), 0.0, 1.0))
            site_center_mm = arch.evaluate_at(t_target)
        else:
            site_center_mm = base_site_mm
        return np.asarray(site_center_mm, dtype=np.float64), site_source, is_approx_site

    def get_site_center_ras(self, fdi: str, offset_mm: float = 0.0) -> Optional[np.ndarray]:
        """Return the RAS center of the arch cross-section for `fdi`."""
        resolved = self._site_center_with_provenance(str(fdi), offset_mm=offset_mm)
        return None if resolved is None else resolved[0]

    def is_tooth_missing(self, fdi: int) -> bool:
        """Return True when `fdi` has no segmented tooth mask (edentulous site within an existing jaw)."""
        fdi_str = resolve_fdi_number(fdi)
        if fdi_str is None:
            return False
        is_mand = is_mandibular_fdi(fdi_str)
        if getattr(self, "mask", None) is not None:
            arch = self._arch_for_fdi(fdi_str)
            jaw_id = self._jaw_id(is_mand)
            present_ids = self._present_label_ids()
            if arch is None and jaw_id not in present_ids:
                return False
        if not self.is_dental_segmentator_5class:
            return self._get_tooth_info(fdi_str) is None
        center = self.get_site_center_ras(fdi_str)
        if center is None:
            return False
        tooth_lbl = get_tooth_group_label_id(is_mand, is_5class=True)
        vox = self.volume.world_to_voxel(center)
        vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        sz, sy, sx = self.volume.spacing
        rx = max(2, int(round(3.5 / max(1e-3, sx))))
        ry = max(2, int(round(3.5 / max(1e-3, sy))))
        rz = max(2, int(round(5.0 / max(1e-3, sz))))
        sub = self.mask[
            max(0, vz - rz) : min(self.mask.shape[0], vz + rz + 1),
            max(0, vy - ry) : min(self.mask.shape[1], vy + ry + 1),
            max(0, vx - rx) : min(self.mask.shape[2], vx + rx + 1),
        ]
        return int(np.count_nonzero(sub == tooth_lbl)) < 25

    def expand_missing_edentulous_span(self, requested_fdis: List[int]) -> Tuple[List[int], List[int]]:
        """Smartly expand requested tooth site(s) to include all contiguous missing/empty teeth."""
        if not requested_fdis:
            return [], []
        req_set = {int(f) for f in requested_fdis}
        expanded = set(req_set)
        missing_detected: set = set()

        for fdi in sorted(req_set):
            q = get_fdi_quadrant(fdi)
            if q is None:
                continue
            pos = int(str(fdi)[1])
            if self.is_tooth_missing(fdi):
                missing_detected.add(fdi)
                for p in range(pos - 1, 0, -1):
                    cand = q * 10 + p
                    if self.is_tooth_missing(cand):
                        expanded.add(cand)
                        missing_detected.add(cand)
                    else:
                        break
                for p in range(pos + 1, 8):
                    cand = q * 10 + p
                    if self.is_tooth_missing(cand):
                        expanded.add(cand)
                        missing_detected.add(cand)
                    else:
                        break
        return sorted(expanded), sorted(missing_detected)

    def measure_implant_site(self, fdi: str, offset_mm: float = 0.0) -> Optional[ImplantSiteMeasurement]:
        """Compute implant planning metrics in the arch-frame cross-section; returns None if unmeasurable."""
        self.last_withheld_reason = None
        fdi_str = resolve_fdi_number(fdi)
        if not fdi_str:
            return None
        cache_key = (fdi_str, round(float(offset_mm), 4))
        if cache_key in self._site_cache:
            cached = self._site_cache[cache_key]
            self.last_withheld_reason = (
                cached.withheld_reason if cached is not None else self.last_withheld_reason
            )
            return cached

        info = get_tooth_label(fdi_str)
        if not info:
            return None

        is_mandible = is_mandibular_fdi(fdi_str)
        jaw_id = self._jaw_id(is_mandible)
        tooth = self._get_tooth_info(fdi_str)

        resolved = self._site_center_with_provenance(fdi_str, offset_mm=offset_mm)
        if resolved is None:
            # Never fail silently: the GUI/print/gallery show this reason instead of a bare "not measured".
            self.last_withheld_reason = WITHHELD_NO_SITE
            return None
        site_center_mm, site_source, is_approx_site = resolved

        u_coords, v_coords, mask_cs, img_cs, P, N, B = self._resample_arch_cross_section(
            site_center_mm, is_mandible=is_mandible, half_u_mm=15.0, half_v_mm=25.0, step_mm=0.25
        )
        cs = CrossSectionData(
            u_coords=u_coords,
            v_coords=v_coords,
            mask_cs=mask_cs,
            img_cs=img_cs,
            frame=(P, np.cross(B, N), N, B),
            crest_v=None,
            width_segments={},
            canal_centroid_uv=None,
            is_mandible=is_mandible,
        )
        try:
            res = self._measure_from_cs(
                fdi_str, info, is_mandible, jaw_id, tooth, cs, site_source, is_approx_site
            )
            self.last_cross_sections[fdi_str] = cs
            self._site_cache[cache_key] = res
            return res
        except Exception:
            logger.exception("Implant-site metrics failed for FDI %s", fdi_str)
            cs.withheld_reason = _join_reasons(cs.withheld_reason, WITHHELD_METRIC_FAILED)
            self.last_withheld_reason = WITHHELD_METRIC_FAILED
            return None

    # ------------------------------------------------------------------
    # Anatomy-Aware Local Vital-Structure Selection (Part 2.2)
    # ------------------------------------------------------------------
    def _select_local_vital_structure(
        self,
        mask_cs: np.ndarray,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        P: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        quadrant: int,
        is_mandible: bool,
        crest_uv: Optional[Tuple[float, float]],
    ) -> Tuple[
        Optional[int],
        Optional[np.ndarray],
        Optional[Tuple[float, float]],
        Optional[List[Tuple[float, float]]],
        Optional[str],
    ]:
        """Select the ipsilateral canal/sinus ONLY if present in the local 3D/2D planning region.

        Returns `(vital_id, local_vital_mask_2d, centroid_uv, contour_uv, unavailable_reason)`.
        Rejects contralateral, distant, or unrelated structures.
        """
        candidates = self._vital_ids(quadrant, is_mandible)
        unavail_reason = WITHHELD_NO_CANAL_SEG if is_mandible else WITHHELD_NO_SINUS_SEG
        if not candidates:
            return None, None, None, None, unavail_reason

        present = self._present_label_ids()
        active_candidates = [vid for vid in candidates if vid in present]
        if not active_candidates:
            return None, None, None, None, unavail_reason

        u_ref = float(crest_uv[0]) if crest_uv is not None else 0.0
        v_ref = float(crest_uv[1]) if crest_uv is not None else 0.0

        # 1. Check directly inside the local cross-section plane first
        for vid in active_candidates:
            v_mask = mask_cs == int(vid)
            if not v_mask.any():
                continue
            # Restrict to ipsilateral local corridor centered around the ridge (|u| <= 11.0 mm
            # or |u - u_ref| <= 8.5 mm) and apical to crest (with 1.0 mm tolerance).
            u_ok = (np.abs(u_coords) <= 11.0) | (np.abs(u_coords - u_ref) <= 8.5)
            v_ok = (v_coords <= v_ref + 1.0) if is_mandible else (v_coords >= v_ref - 1.0)
            local_2d = v_mask & np.outer(v_ok, u_ok)
            if not local_2d.any():
                continue

            # Connected-component cleanup: drop 1-pixel noise while preserving all valid >=2px
            # vital components in the corridor; pick the closest component for centroid/contour.
            marker_2d = local_2d
            try:
                from scipy import ndimage  # type: ignore

                lbl, n_c = ndimage.label(local_2d, structure=np.ones((3, 3), dtype=np.int32))
                if n_c > 1:
                    valid_mask = np.zeros_like(local_2d, dtype=bool)
                    best_c = 0
                    best_d = float("inf")
                    for cid in range(1, n_c + 1):
                        comp = lbl == cid
                        rs, cs_i = np.where(comp)
                        if rs.size < 2:
                            continue
                        valid_mask |= comp
                        cu = float(np.mean(u_coords[cs_i]))
                        cv = float(np.mean(v_coords[rs]))
                        d = math.hypot(cu - u_ref, cv - v_ref)
                        if d < best_d:
                            best_d = d
                            best_c = cid
                    if best_c > 0:
                        local_2d = valid_mask
                        marker_2d = lbl == best_c
            except ImportError:
                pass

            rows, cols = np.where(marker_2d)
            if rows.size == 0:
                continue
            centroid_uv = (float(np.mean(u_coords[cols])), float(np.mean(v_coords[rows])))

            # Extract segmentation boundary contour in (u, v)
            contour_uv: List[Tuple[float, float]] = []
            if is_mandible:
                # Boundary pixels of the canal cross-section
                padded = np.pad(marker_2d, 1, mode="constant", constant_values=False)
                interior = (
                    padded[1:-1, 1:-1]
                    & padded[:-2, 1:-1]
                    & padded[2:, 1:-1]
                    & padded[1:-1, :-2]
                    & padded[1:-1, 2:]
                )
                edge = marker_2d & (~interior)
                er, ec = np.where(edge)
                if er.size > 0:
                    angles = np.arctan2(v_coords[er] - centroid_uv[1], u_coords[ec] - centroid_uv[0])
                    order = np.argsort(angles)
                    contour_uv = [(float(u_coords[int(ec[i])]), float(v_coords[int(er[i])])) for i in order]
            else:
                # Sinus floor contour: inferior-most sinus voxel per column
                for u_col in np.unique(cols):
                    col_vs = rows[cols == u_col]
                    if col_vs.size > 0:
                        contour_uv.append((float(u_coords[int(u_col)]), float(v_coords[int(col_vs.min())])))

            return int(vid), local_2d, centroid_uv, (contour_uv if len(contour_uv) >= 2 else None), None

        # 2. If not in the 2D cross-section plane, check a local 3D ROI (<= 28 mm around P)
        #    for metadata/availability reporting (never used to shorten non-canal slices).
        local_radius_mm = 28.0
        vox = self.volume.world_to_voxel(P)
        vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        sz, sy, sx = self.volume.spacing
        rz = max(2, int(round(local_radius_mm / max(1e-3, sz))))
        ry = max(2, int(round(local_radius_mm / max(1e-3, sy))))
        rx = max(2, int(round(local_radius_mm / max(1e-3, sx))))
        z0, z1 = max(0, vz - rz), min(self.mask.shape[0], vz + rz + 1)
        y0, y1 = max(0, vy - ry), min(self.mask.shape[1], vy + ry + 1)
        x0, x1 = max(0, vx - rx), min(self.mask.shape[2], vx + rx + 1)
        local_vol = self.mask[z0:z1, y0:y1, x0:x1]

        for vid in active_candidates:
            if np.any(local_vol == int(vid)):
                return int(vid), None, None, None, None

        return None, None, None, None, unavail_reason

    # ------------------------------------------------------------------
    # Anatomy-Aware Alveolar Crest & Ridge Axis Detection (Part 3.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_socket_crest_rims(
        surf_u: np.ndarray,
        smooth_v: np.ndarray,
        bone_mask: np.ndarray,
        tooth_mask: Optional[np.ndarray],
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        is_mandible: bool,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Locate `(u_L_crest, v_L_crest)` and `(u_B_crest, v_B_crest)` flanking an in-socket tooth.

        Returns `None` when the crest is unbroken or when only a single cortical plate exists
        (e.g. a missing buccal wall), preventing bridging across missing bone.
        """
        if tooth_mask is None or not np.any(tooth_mask):
            return None
        central = np.flatnonzero(np.abs(surf_u) <= 8.5)
        if central.size < 2:
            return None
        peak_v = float(np.max(smooth_v[central])) if is_mandible else float(np.min(smooth_v[central]))

        # Check for same-jaw tooth voxels in the coronal socket band around the bone crest.
        # Prefer tooth columns within +/-1.75 mm of `peak_v` so tilted roots deeper down do not
        # drag u_t_min / u_t_max horizontally past the coronal crest rim.
        if is_mandible:
            v_band_crest = (v_coords >= peak_v - 1.75) & (v_coords <= peak_v + 1.5)
            v_band = (v_coords >= peak_v - 4.5) & (v_coords <= peak_v + 1.5)
        else:
            v_band_crest = (v_coords <= peak_v + 1.75) & (v_coords >= peak_v - 1.5)
            v_band = (v_coords <= peak_v + 4.5) & (v_coords >= peak_v - 1.5)
        u_central_mask = np.abs(u_coords) <= 7.0
        t_band = np.asarray(tooth_mask, dtype=bool) & np.outer(v_band, u_central_mask)
        if not t_band.any():
            return None
        t_band_coronal = np.asarray(tooth_mask, dtype=bool) & np.outer(v_band_crest, u_central_mask)
        # Prefer the coronal band near `peak_v` when tilted teeth span widely, but never clip a narrow
        # canine/premolar socket below `u_t_vals` of the socket neck.
        t_src = t_band_coronal if int(np.count_nonzero(t_band_coronal)) >= 6 else t_band
        _tr, t_cols = np.where(t_src)
        u_t_vals = u_coords[t_cols]
        u_t_min = float(np.min(u_t_vals))
        u_t_max = float(np.max(u_t_vals))
        u_t_mid = float(np.median(u_t_vals))

        def _rim_peak(side_indices: np.ndarray) -> Optional[Tuple[float, float]]:
            if side_indices.size == 0:
                return None
            side_v = smooth_v[side_indices]
            best_local = int(np.argmax(side_v)) if is_mandible else int(np.argmin(side_v))
            rim_v = float(side_v[best_local])
            # Reject deep basal bone below a missing wall (e.g. single-plate defect). Frontal teeth
            # legitimately have rims at different heights, so the tolerance is wider than the old
            # 4.5 mm; a vertical bone-support test below keeps stray voxels out.
            if abs(peak_v - rim_v) > SOCKET_RIM_MAX_V_OFFSET_MM:
                return None
            ui = int(np.argmin(np.abs(u_coords - float(surf_u[side_indices[best_local]]))))
            dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
            if float(np.count_nonzero(np.asarray(bone_mask)[:, ui])) * dv < 0.75:
                return None
            if is_mandible:
                plat = side_indices[side_v >= rim_v - 0.15]
            else:
                plat = side_indices[side_v <= rim_v + 0.15]
            if plat.size >= 2 and int(plat[-1] - plat[0] + 1) == int(plat.size):
                return float(0.5 * (surf_u[plat[0]] + surf_u[plat[-1]])), float(np.median(smooth_v[plat]))
            idx = int(side_indices[best_local])
            return float(surf_u[idx]), rim_v

        left_idx = central[(surf_u[central] <= u_t_min + 0.75) & (surf_u[central] < u_t_mid)]
        right_idx = central[(surf_u[central] >= u_t_max - 0.75) & (surf_u[central] > u_t_mid)]
        left_rim = _rim_peak(left_idx)
        right_rim = _rim_peak(right_idx)
        if left_rim is None or right_rim is None:
            return None
        if not (2.0 <= (right_rim[0] - left_rim[0]) <= 14.0):
            return None
        return left_rim, right_rim

    @staticmethod
    def _extract_coronal_bone_surface(
        surface_mask: np.ndarray,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        is_mandible: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract coronal-most continuous bone surface `(surf_u, surf_v)` per column."""
        dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
        min_support_px = max(2, int(round(0.75 / max(1e-3, dv))))
        surf_cols: List[int] = []
        surf_vs: List[float] = []
        for iu in range(len(u_coords)):
            col = surface_mask[:, iu]
            if not col.any():
                continue
            v_indices = np.flatnonzero(col)
            chosen_v: Optional[int] = None
            run = 0
            prev = -1
            seq = v_indices[::-1] if is_mandible else v_indices
            for idx in seq:
                if prev < 0 or abs(prev - int(idx)) == 1:
                    run += 1
                else:
                    run = 1
                prev = int(idx)
                if run >= min_support_px:
                    chosen_v = int(idx) + run - 1 if is_mandible else int(idx) - run + 1
                    break
            if chosen_v is None:
                chosen_v = int(v_indices[-1] if is_mandible else v_indices[0])
            surf_cols.append(iu)
            surf_vs.append(float(v_coords[chosen_v]))
        if not surf_cols:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        return (
            np.asarray([float(u_coords[i]) for i in surf_cols], dtype=np.float64),
            np.asarray(surf_vs, dtype=np.float64),
        )

    @staticmethod
    def _detect_alveolar_crest_and_axis(
        ridge_mask: np.ndarray,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        is_mandible: bool,
        bone_mask: Optional[np.ndarray] = None,
        tooth_mask: Optional[np.ndarray] = None,
    ) -> Tuple[
        Optional[Tuple[float, float]],
        Tuple[float, float],
        Tuple[float, float],
        Optional[float],
        List[Tuple[float, float]],
    ]:
        """Detect local alveolar crest `(u_crest, v_crest)` and ridge-aligned unit axes `(d_apical, d_bl)`.

        Algorithm:
          1. Extract the coronal-most continuous bone surface `v_surf(u)` per column from bone-only
             `surface_mask`, rejecting thin vertical spikes (< 0.75 mm vertical support).
          2. Apply spacing-aware robust 1D median + light Gaussian smoothing to `v_surf(u)`.
          3. When a tooth socket splits lingual and buccal jawbone rims, bridge `crest_uv` to the
             midpoint between the two bone rims; otherwise locate the ridge peak/flat plateau.
          4. Estimate the local alveolar long axis via PCA on the primary alveolar ridge,
             avoiding eigenvector sign flips near 45 degrees.
        """
        default_apical = (0.0, -1.0) if is_mandible else (0.0, 1.0)
        default_bl = (1.0, 0.0)
        if not ridge_mask.any():
            return None, default_apical, default_bl, None, []

        du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
        # The crest must come from BONE only (`bone_mask`), never the tooth crown.
        surface_mask = bone_mask if (bone_mask is not None and bool(np.any(bone_mask))) else ridge_mask
        surf_u, surf_v = MeasurementEngine._extract_coronal_bone_surface(
            surface_mask, u_coords, v_coords, is_mandible
        )
        if surf_u.size == 0:
            return None, default_apical, default_bl, None, []

        # Spacing-aware robust smoothing (median filter ~1.0 mm + light Gaussian sigma ~0.4 mm)
        smooth_v = surf_v.copy()
        if len(smooth_v) >= 5:
            try:
                from scipy import ndimage  # type: ignore

                med_k = max(3, int(round(1.0 / max(1e-3, du))) | 1)
                med_k = min(med_k, len(smooth_v) if len(smooth_v) % 2 == 1 else len(smooth_v) - 1)
                if med_k >= 3:
                    smooth_v = ndimage.median_filter(smooth_v, size=med_k, mode="nearest")
                sigma_px = 0.4 / max(1e-3, du)
                if sigma_px >= 0.4:
                    smooth_v = ndimage.gaussian_filter1d(smooth_v, sigma=sigma_px, mode="nearest")
            except ImportError:
                pass

        crest_profile_uv = [(float(u), float(v)) for u, v in zip(surf_u, smooth_v)]

        socket_rims = MeasurementEngine._find_socket_crest_rims(
            surf_u, smooth_v, surface_mask, tooth_mask, u_coords, v_coords, is_mandible
        )
        if socket_rims is not None:
            left_rim, right_rim = socket_rims
            u_crest = float(0.5 * (left_rim[0] + right_rim[0]))
            v_crest = float(0.5 * (left_rim[1] + right_rim[1]))
        else:
            # Search the whole central corridor (|u| <= 6 mm), not just the narrow planning axis:
            # bone-only columns can be missing over a hollow socket, and limiting the search to
            # |u| <= 1.5 mm there put the entry on the basal cortex several millimetres too deep.
            central_mask = np.abs(surf_u) <= 6.0
            search_indices = np.flatnonzero(central_mask) if central_mask.any() else np.arange(len(surf_u))
            search_v = smooth_v[search_indices]
            if is_mandible:
                local_best = int(np.argmax(search_v))
            else:
                local_best = int(np.argmin(search_v))
            best_idx = int(search_indices[local_best])

            # For flat-topped ridges (multiple columns tied within 0.15 mm of peak without a dip
            # > 0.5 mm between them), pick the horizontal center of the plateau.
            peak_val = float(smooth_v[best_idx])
            if is_mandible:
                plateau = search_indices[smooth_v[search_indices] >= peak_val - 0.15]
                no_dip = bool(
                    plateau.size >= 2 and np.all(smooth_v[plateau[0] : plateau[-1] + 1] >= peak_val - 0.5)
                )
            else:
                plateau = search_indices[smooth_v[search_indices] <= peak_val + 0.15]
                no_dip = bool(
                    plateau.size >= 2 and np.all(smooth_v[plateau[0] : plateau[-1] + 1] <= peak_val + 0.5)
                )

            if len(plateau) >= 2 and no_dip:
                u_crest = float(0.5 * (surf_u[plateau[0]] + surf_u[plateau[-1]]))
                v_crest = float(np.median(smooth_v[plateau]))
            else:
                u_crest = float(surf_u[best_idx])
                v_crest = peak_val
                # Subvoxel parabolic refinement for isolated interior peak
                if 0 < best_idx < len(smooth_v) - 1:
                    y0, y1, y2 = (
                        float(smooth_v[best_idx - 1]),
                        float(smooth_v[best_idx]),
                        float(smooth_v[best_idx + 1]),
                    )
                    denom = y0 - 2.0 * y1 + y2
                    if abs(denom) > 1e-6:
                        delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
                        u_crest = float(surf_u[best_idx] + delta * du)
                        v_crest = float(y1 - 0.25 * (y0 - y2) * delta)

        # Height axis = coronal midline bisector of the intact `ridge_mask` (`0..12 mm` apical
        # to crest), rejecting basal flaring / external oblique shelves (> 1.45x crestal width).
        # Using `ridge_mask` (with tooth root) avoids hollow single-plate socket walls skewing
        # molars by 30-45 deg, while preserving uniformly tilted ridges (`test_20`).
        ridge_angle_deg: Optional[float] = None
        d_apical = default_apical
        d_bl = default_bl

        dir_sign = -1.0 if is_mandible else 1.0
        v_band_lo = float(v_crest) - 12.0 if is_mandible else float(v_crest) + 0.5
        v_band_hi = float(v_crest) - 0.5 if is_mandible else float(v_crest) + 12.0
        row_idxs = np.flatnonzero((v_coords >= v_band_lo) & (v_coords <= v_band_hi))
        if row_idxs.size > 0 and not is_mandible:
            row_idxs = row_idxs[::-1]  # process apically outward -> coronal-first for maxilla too
        elif row_idxs.size > 0 and is_mandible:
            row_idxs = row_idxs[::-1]  # high v (coronal) -> low v (apical)

        mid_rows: List[Tuple[float, float, float, float, float]] = []
        for r in row_idxs:
            cols = np.flatnonzero(ridge_mask[int(r), :])
            if cols.size < 2:
                continue
            u_l_row = float(u_coords[int(cols[0])])
            u_r_row = float(u_coords[int(cols[-1])])
            r_span = u_r_row - u_l_row
            if r_span < 1.0:
                continue
            u_mid = 0.5 * (u_l_row + u_r_row)
            mid_rows.append((u_mid, float(v_coords[int(r)]), r_span, u_l_row, u_r_row))

        if len(mid_rows) >= 6:
            crest_spans = [item[2] for item in mid_rows[: max(3, len(mid_rows) // 2)]]
            med_span = float(np.median(crest_spans))
            kept = [item for item in mid_rows if item[2] <= max(med_span * 1.45, med_span + 4.5)]
            if len(kept) < 6:
                kept = mid_rows
            vs_arr = np.asarray([k[1] for k in kept], dtype=np.float64)
            us_arr = np.asarray([k[0] for k in kept], dtype=np.float64)
            ul_arr = np.asarray([k[3] for k in kept], dtype=np.float64)
            ur_arr = np.asarray([k[4] for k in kept], dtype=np.float64)
            v_span = float(np.max(vs_arr) - np.min(vs_arr))
            if v_span >= 4.0:
                # Coronal linear regression of u(s_apical) with bilateral wall agreement check:
                # true tilted ridges (`test_20`) tilt BOTH walls in the same direction, whereas
                # palatal vault / mylohyoid flares move only one wall while the other stays steep.
                s_ap = dir_sign * (vs_arr - float(v_crest))
                s_c = s_ap - float(np.mean(s_ap))
                denom = float(np.dot(s_c, s_c))
                if denom > 1e-6:
                    slope_mid = float(np.dot(s_c, us_arr - float(np.mean(us_arr))) / denom)
                    slope_l = float(np.dot(s_c, ul_arr - float(np.mean(ul_arr))) / denom)
                    slope_r = float(np.dot(s_c, ur_arr - float(np.mean(ur_arr))) / denom)
                    if slope_l * slope_r > 0.0:
                        s_abs_min = min(abs(slope_l), abs(slope_r))
                        s_abs_max = max(abs(slope_l), abs(slope_r))
                        if s_abs_max <= 2.5 * max(0.08, s_abs_min):
                            slope = slope_mid
                        else:
                            slope = math.copysign(
                                min(abs(slope_mid), s_abs_min + 0.35 * (s_abs_max - s_abs_min)), slope_mid
                            )
                    else:
                        slope = 0.35 * slope_mid
                    raw_deg = math.degrees(math.atan2(slope, 1.0))
                    clamped_deg = float(np.clip(raw_deg, -45.0, 45.0))
                    ridge_angle_deg = round(clamped_deg, 1)
                    rad = math.radians(clamped_deg)
                    d_apical = (
                        (math.sin(rad), -math.cos(rad)) if is_mandible else (math.sin(rad), math.cos(rad))
                    )
                    d_bl = (math.cos(rad), math.sin(rad) if is_mandible else -math.sin(rad))

        if ridge_angle_deg is None:
            if is_mandible:
                alv_band = (v_coords >= float(v_crest) - 14.0) & (v_coords <= float(v_crest) + 3.0)
            else:
                alv_band = (v_coords <= float(v_crest) + 14.0) & (v_coords >= float(v_crest) - 3.0)
            pca_src = (
                ridge_mask & alv_band[:, None]
                if int(np.count_nonzero(ridge_mask & alv_band[:, None])) >= 14
                else ridge_mask
            )
            rows_p, cols_p = np.where(pca_src)
            if rows_p.size >= 8:
                cloud = np.column_stack([u_coords[cols_p], v_coords[rows_p]]).astype(np.float64)
                cloud -= cloud.mean(axis=0, keepdims=True)
                cov = (cloud.T @ cloud) / float(max(1, cloud.shape[0]))
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    v_major = np.asarray(eigvecs[:, 1], dtype=np.float64)
                    if (
                        float(eigvals[1]) >= 1.4 * max(1e-6, float(eigvals[0]))
                        and abs(float(v_major[1])) >= 0.57
                    ):
                        axis = v_major
                    else:
                        axis = max(
                            (np.asarray(eigvecs[:, index], dtype=np.float64) for index in range(2)),
                            key=lambda vec: abs(float(vec[1])),
                        )
                    if (is_mandible and axis[1] > 0.0) or ((not is_mandible) and axis[1] < 0.0):
                        axis = -axis
                    norm = float(math.hypot(float(axis[0]), float(axis[1])))
                    if norm > 1e-6:
                        axis = axis / norm
                        raw_deg = math.degrees(
                            math.atan2(float(axis[0]), -float(axis[1]))
                            if is_mandible
                            else math.atan2(float(axis[0]), float(axis[1]))
                        )
                        clamped_deg = float(np.clip(raw_deg, -45.0, 45.0))
                        ridge_angle_deg = round(clamped_deg, 1)
                        rad = math.radians(clamped_deg)
                        d_apical = (
                            (math.sin(rad), -math.cos(rad)) if is_mandible else (math.sin(rad), math.cos(rad))
                        )
                        d_bl = (math.cos(rad), math.sin(rad) if is_mandible else -math.sin(rad))
                except Exception:  # pragma: no cover - degenerate covariance only
                    logger.debug("PCA height axis unavailable", exc_info=True)

        return (u_crest, v_crest), d_apical, d_bl, ridge_angle_deg, crest_profile_uv

    # ------------------------------------------------------------------
    # Anatomy-Aware Available Bone Height (Part 1.2 & Part 3.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _endosteal_cortex_inset_mm(
        img_cs: Optional[np.ndarray],
        ray_u: np.ndarray,
        ray_v: np.ndarray,
        s_vals: np.ndarray,
        last_idx: int,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        full_h: float,
    ) -> float:
        """Locate the internal endosteal border of the basal/maxillary cortex when HU contrast exists."""
        if img_cs is None or img_cs.shape != (len(v_coords), len(u_coords)) or full_h < 6.0:
            return JAW_END_CORTEX_INSET_MM
        try:
            u_min, u_max = float(u_coords[0]), float(u_coords[-1])
            v_min, v_max = float(v_coords[0]), float(v_coords[-1])
            s_sub = s_vals[: last_idx + 1]
            if s_sub.size < 10:
                return JAW_END_CORTEX_INSET_MM
            cf = np.clip(
                (ray_u[: last_idx + 1] - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1),
                0.0,
                len(u_coords) - 1.0,
            )
            rf = np.clip(
                (ray_v[: last_idx + 1] - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1),
                0.0,
                len(v_coords) - 1.0,
            )
            from scipy import ndimage  # type: ignore

            hu_prof = ndimage.map_coordinates(
                np.asarray(img_cs, dtype=np.float32),
                np.vstack([rf, cf]),
                order=1,
                mode="nearest",
            )
            rim_mask = (s_sub >= max(0.0, full_h - 4.0)) & (s_sub <= full_h)
            marrow_mask = (s_sub >= 0.35 * full_h) & (s_sub <= max(0.45 * full_h, full_h - 4.0))
            if not rim_mask.any() or not marrow_mask.any():
                return JAW_END_CORTEX_INSET_MM
            rim_peak = float(np.max(hu_prof[rim_mask]))
            marrow_med = float(np.median(hu_prof[marrow_mask]))
            # Only refine on scans with genuine hyperdense cortex vs trabecular interior (>= 180 HU contrast).
            if rim_peak < 350.0 or (rim_peak - marrow_med) < 180.0:
                return JAW_END_CORTEX_INSET_MM
            thresh = marrow_med + 0.55 * (rim_peak - marrow_med)
            rim_indices = np.flatnonzero(rim_mask)
            # Walk inward (apical -> coronal) from peak rim to find the inner endosteal border.
            peak_rel = int(np.argmax(hu_prof[rim_indices]))
            peak_idx = int(rim_indices[peak_rel])
            for k in range(
                peak_idx, max(-1, peak_idx - int(round(2.6 / max(0.1, float(s_sub[1] - s_sub[0]))))), -1
            ):
                if float(hu_prof[k]) <= thresh:
                    inset = float(full_h) - float(s_sub[k])
                    return float(np.clip(inset, JAW_END_CORTEX_INSET_MM, 2.4))
        except Exception:
            logger.debug("Endosteal cortical border refinement fell back to inset", exc_info=True)
        return JAW_END_CORTEX_INSET_MM

    def _compute_bone_height_in_cs(
        self,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        mask_cs: np.ndarray,
        P: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        vital_id: Optional[int],
        *,
        ridge_mask: np.ndarray,
        crest_uv: Optional[Tuple[float, float]],
        d_apical: Tuple[float, float],
        local_vital_mask_2d: Optional[np.ndarray] = None,
        img_cs: Optional[np.ndarray] = None,
    ) -> Tuple[
        Optional[float], Optional[float], Optional[Tuple[float, float]], Optional[Tuple[float, float]], str
    ]:
        """Compute available bone height along `d_apical` from `crest_uv` to vital or basal cortical boundary.

        Returns `(bone_height_mm, vital_dist_mm, basal_uv, vital_boundary_uv, provenance)`.
        NEVER uses a root apex as the basal endpoint.
        """
        if crest_uv is None or not ridge_mask.any():
            return None, None, None, None, PROVENANCE_UNAVAILABLE

        u0, v0 = float(crest_uv[0]), float(crest_uv[1])
        au, av = float(d_apical[0]), float(d_apical[1])
        du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
        dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
        step = min(du, dv, 0.25)
        max_len = float(abs(v_coords[-1] - v_coords[0]))

        # 1. Trace ray along d_apical through the primary connected ridge_mask to find basal cortex
        s_vals = np.arange(0.0, max_len + step, step, dtype=np.float64)
        ray_u = u0 + s_vals * au
        ray_v = v0 + s_vals * av

        u_min, u_max = float(u_coords[0]), float(u_coords[-1])
        v_min, v_max = float(v_coords[0]), float(v_coords[-1])
        in_bounds = (ray_u >= u_min) & (ray_u <= u_max) & (ray_v >= v_min) & (ray_v <= v_max)
        s_vals = s_vals[in_bounds]
        ray_u = ray_u[in_bounds]
        ray_v = ray_v[in_bounds]
        if s_vals.size < 2:
            return None, None, None, None, PROVENANCE_UNAVAILABLE

        col_f = (ray_u - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1)
        row_f = (ray_v - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1)

        try:
            from scipy import ndimage  # type: ignore

            bone_profile = ndimage.map_coordinates(
                ridge_mask.astype(np.float32),
                np.vstack([row_f, col_f]),
                order=1,
                mode="constant",
                cval=0.0,
            )
        except ImportError:
            ri = np.clip(np.round(row_f).astype(int), 0, len(v_coords) - 1)
            ci = np.clip(np.round(col_f).astype(int), 0, len(u_coords) - 1)
            bone_profile = ridge_mask[ri, ci].astype(np.float32)

        in_bone = bone_profile >= 0.5
        if not in_bone.any():
            # Fallback to nearest bone column if crest_uv slightly grazed the surface
            col_hits = np.flatnonzero(ridge_mask.any(axis=1))
            if col_hits.size < 2:
                return None, None, None, None, PROVENANCE_UNAVAILABLE
            basal_v = float(v_coords[int(col_hits[0])]) if is_mandible else float(v_coords[int(col_hits[-1])])
            full_h = abs(v0 - basal_v)
            basal_uv = (u0 + full_h * au, v0 + full_h * av)
        else:
            bone_indices = np.flatnonzero(in_bone)
            last_idx = int(bone_indices[-1])
            # When no nerve/sinus bounds the site, the caliper continues to the far end of the
            # jaw along the PCA axis instead of stopping at the ridge component's entry point.
            full_h = float(max(0.0, s_vals[last_idx]))
            basal_uv = (u0 + full_h * au, v0 + full_h * av)

        # 2. Check ipsilateral vital structure boundary inside the 2D cross-section plane.
        #    Never truncate height via 3D Euclidean distance when the canal/sinus does not
        #    exist in this cross-section plane (prevents stopping short mid-bone anterior to mental foramen).
        vital_dist: Optional[float] = None
        vital_boundary_uv: Optional[Tuple[float, float]] = None
        provenance = PROVENANCE_LABEL_DERIVED

        if local_vital_mask_2d is not None and local_vital_mask_2d.any():
            vr, vc = np.where(local_vital_mask_2d)
            vu = u_coords[vc]
            vv = v_coords[vr]
            du_v = vu - u0
            dv_v = vv - v0
            along = du_v * au + dv_v * av
            perp = np.abs(du_v * (-av) + dv_v * au)

            cand_dists: List[float] = []
            # Implant-footprint safety corridor (±3.0 mm laterally) prevents distant foramina/sinus
            # recesses from prematurely truncating H while keeping direct crossings strictly blocked.
            in_corridor = (along >= 0.0) & (perp <= 3.0)
            if in_corridor.any():
                cand_dists.append(float(max(0.0, np.min(along[in_corridor]))))

            # Direct ray hit check ensures the caliper line never enters or crosses any vital pixel.
            ri_v = np.clip(np.round(row_f).astype(int), 0, len(v_coords) - 1)
            ci_v = np.clip(np.round(col_f).astype(int), 0, len(u_coords) - 1)
            ray_hits = np.flatnonzero(local_vital_mask_2d[ri_v, ci_v])
            if ray_hits.size > 0:
                cand_dists.append(float(max(0.0, s_vals[int(ray_hits[0])] - 0.5 * step)))

            if cand_dists:
                vital_dist = min(cand_dists)
                if vital_dist <= full_h + 1.0:
                    eff_h = min(full_h, vital_dist)
                    vital_boundary_uv = (u0 + eff_h * au, v0 + eff_h * av)
                    return (
                        round(eff_h, 2),
                        round(eff_h, 2),
                        vital_boundary_uv,
                        vital_boundary_uv,
                        PROVENANCE_LABEL_DERIVED,
                    )

        # Stop at the internal (endosteal) border of the basal/maxillary cortex, never flush with periosteum.
        eff_inset = (
            self._endosteal_cortex_inset_mm(
                img_cs, ray_u, ray_v, s_vals, last_idx, u_coords, v_coords, full_h
            )
            if in_bone.any()
            else JAW_END_CORTEX_INSET_MM
        )
        inset_h = max(0.0, float(full_h) - eff_inset)
        basal_uv = (u0 + inset_h * au, v0 + inset_h * av)
        return (
            round(inset_h, 2),
            (round(vital_dist, 2) if vital_dist is not None else None),
            basal_uv,
            vital_boundary_uv,
            provenance,
        )

    # ------------------------------------------------------------------
    # Anatomy-Aware Buccal & Lingual Cortical Boundary Detection (Part 3.4)
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_band_profile(
        arr_f32: np.ndarray,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        cx: float,
        cy: float,
        s_grid: np.ndarray,
        au: float,
        av: float,
        bu: float,
        bv: float,
        step: float,
        *,
        offsets: Tuple[float, ...] = (-1.0, 0.0, 1.0),
    ) -> np.ndarray:
        """Sample a thin band perpendicular to `d_bl` along `s_grid` with bilinear interpolation."""
        u_min, u_max = float(u_coords[0]), float(u_coords[-1])
        v_min, v_max = float(v_coords[0]), float(v_coords[-1])
        try:
            from scipy import ndimage  # type: ignore

            has_scipy = True
        except ImportError:
            has_scipy = False

        band_profiles = []
        for mult in offsets:
            d_off = mult * step
            su = (cx + d_off * au) + s_grid * bu
            sv = (cy + d_off * av) + s_grid * bv
            in_b = (su >= u_min) & (su <= u_max) & (sv >= v_min) & (sv <= v_max)
            prof = np.zeros_like(s_grid, dtype=np.float32)
            if in_b.any():
                cf = (su[in_b] - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1)
                rf = (sv[in_b] - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1)
                if has_scipy:
                    prof[in_b] = ndimage.map_coordinates(
                        arr_f32, np.vstack([rf, cf]), order=1, mode="constant", cval=0.0
                    )
                else:
                    ri = np.clip(np.round(rf).astype(int), 0, len(v_coords) - 1)
                    ci = np.clip(np.round(cf).astype(int), 0, len(u_coords) - 1)
                    prof[in_b] = arr_f32[ri, ci]
            band_profiles.append(prof)
        return np.mean(np.vstack(band_profiles), axis=0)

    @staticmethod
    def _refine_run_endpoints(
        profile: np.ndarray, s_grid: np.ndarray, i_left: int, i_right: int, step: float
    ) -> Tuple[float, float]:
        """Apply subvoxel 0.5-isosurface refinement at outer cortical edges."""
        s_left = float(s_grid[i_left])
        s_right = float(s_grid[i_right])
        if i_left > 0:
            p_out, p_in = float(profile[i_left - 1]), float(profile[i_left])
            if p_in >= 0.5 > p_out and (p_out > 0.0 or p_in < 1.0) and (p_in - p_out) > 1e-6:
                s_left = float(s_grid[i_left - 1] + ((0.5 - p_out) / (p_in - p_out)) * step)
        if i_right + 1 < len(profile):
            p_in, p_out = float(profile[i_right]), float(profile[i_right + 1])
            if p_in >= 0.5 > p_out and (p_out > 0.0 or p_in < 1.0) and (p_in - p_out) > 1e-6:
                s_right = float(s_grid[i_right] + ((p_in - 0.5) / (p_in - p_out)) * step)
        return s_left, s_right

    @staticmethod
    def _select_ridge_run_indices(
        profile: np.ndarray,
        tooth_profile: Optional[np.ndarray],
        s_grid: np.ndarray,
        step: float,
    ) -> Optional[Tuple[int, int]]:
        """Select outer `(i_left, i_right)` indices of jawbone along `s_grid`, bridging tooth sockets.

        If a tooth/pulp separates two jawbone runs straddling `s = 0` with no empty air gap
        (allowing <= 0.75 mm PDL margin), links outer-lingual to outer-buccal jawbone edges.
        Otherwise returns only the single contiguous run closest to `s = 0`.
        """
        fg = profile >= 0.5
        if not fg.any():
            return None
        idx_fg = np.flatnonzero(fg)
        splits = np.where(np.diff(idx_fg) > 1)[0] + 1
        raw_runs = list(np.split(idx_fg, splits))
        runs = [r for r in raw_runs if r.size >= 2]
        # Preserve genuine sub-millimetre (1 px) cortical walls ONLY when tooth bridging can pair them
        # across an in-socket root with another >=2 px bone run; never accept isolated 1 px noise.
        if tooth_profile is not None and len(runs) >= 1 and len(raw_runs) > len(runs):
            runs = [r for r in raw_runs if r.size >= 1]
        if not runs:
            return None

        zero_idx = int(np.argmin(np.abs(s_grid)))
        # Locate the anchor bone run containing or closest to s = 0 (within 6.5 mm when tooth-bridging
        # is available, else within 4.0 mm), preferring >=2 px walls.
        anchor_idx: Optional[int] = None
        best_dist = float("inf")
        for idx, run in enumerate(runs):
            if run.size < 2 and tooth_profile is None:
                continue
            if run[0] <= zero_idx <= run[-1]:
                anchor_idx = idx
                best_dist = 0.0
                break
            d_to_zero = min(abs(float(s_grid[run[0]])), abs(float(s_grid[run[-1]])))
            if d_to_zero < best_dist:
                best_dist = d_to_zero
                anchor_idx = idx
        if anchor_idx is None:
            return None

        max_anchor_dist = 6.5 if (tooth_profile is not None and len(runs) >= 2) else 4.0
        if anchor_idx is None or best_dist > max_anchor_dist:
            return None

        left_idx = anchor_idx
        right_idx = anchor_idx
        if tooth_profile is not None and len(runs) >= 2:

            def _tooth_bridges_gap(r_left: np.ndarray, r_right: np.ndarray) -> bool:
                if (float(s_grid[int(r_right[-1])]) - float(s_grid[int(r_left[0])])) > 16.0:
                    return False
                gap_sl = slice(int(r_left[-1]) + 1, int(r_right[0]))
                gap_t = tooth_profile[gap_sl] >= 0.35
                if gap_t.size == 0 or not np.any(gap_t):
                    return False
                air_idx = np.flatnonzero(~gap_t)
                max_air_run = 0
                if air_idx.size > 0:
                    air_splits = np.where(np.diff(air_idx) > 1)[0] + 1
                    max_air_run = max(len(a) for a in np.split(air_idx, air_splits))
                return max_air_run * step <= 1.5

            while left_idx > 0 and _tooth_bridges_gap(runs[left_idx - 1], runs[right_idx]):
                if not _tooth_bridges_gap(runs[left_idx - 1], runs[left_idx]):
                    break
                left_idx -= 1
            while right_idx + 1 < len(runs) and _tooth_bridges_gap(runs[left_idx], runs[right_idx + 1]):
                if not _tooth_bridges_gap(runs[right_idx], runs[right_idx + 1]):
                    break
                right_idx += 1
            if left_idx == right_idx and (best_dist > 4.0 or runs[left_idx].size < 2):
                return None
            if not any(r.size >= 2 for r in runs[left_idx : right_idx + 1]):
                return None

        return int(runs[left_idx][0]), int(runs[right_idx][-1])

    @staticmethod
    def _refine_width_with_gray_edges(
        img_cs: Optional[np.ndarray],
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        pt_l: Tuple[float, float],
        pt_r: Tuple[float, float],
        *,
        max_shift_mm: float = 0.75,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Refine outer cortical endpoints to the falling half-max HU edge when real bone contrast exists."""
        if img_cs is None or img_cs.shape != (len(v_coords), len(u_coords)):
            return pt_l, pt_r
        dx = float(pt_r[0]) - float(pt_l[0])
        dy = float(pt_r[1]) - float(pt_l[1])
        seg_len = math.hypot(dx, dy)
        if seg_len < 2.5:
            return pt_l, pt_r
        ux, uy = dx / seg_len, dy / seg_len
        du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
        step = min(du, 0.25)
        margin = 2.0
        s_arr = np.arange(-margin, seg_len + margin + step, step, dtype=np.float64)
        su = float(pt_l[0]) + s_arr * ux
        sv = float(pt_l[1]) + s_arr * uy
        u_min, u_max = float(u_coords[0]), float(u_coords[-1])
        v_min, v_max = float(v_coords[0]), float(v_coords[-1])
        if np.any((su < u_min) | (su > u_max) | (sv < v_min) | (sv > v_max)):
            return pt_l, pt_r
        try:
            from scipy import ndimage  # type: ignore

            cf = (su - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1)
            rf = (sv - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1)
            hu = ndimage.map_coordinates(
                np.asarray(img_cs, dtype=np.float32), np.vstack([rf, cf]), order=1, mode="nearest"
            )
        except Exception:
            return pt_l, pt_r

        ext_l = hu[s_arr <= -0.75]
        ext_r = hu[s_arr >= seg_len + 0.75]
        in_band = hu[(s_arr >= 0.0) & (s_arr <= seg_len)]
        if ext_l.size < 2 or ext_r.size < 2 or in_band.size < 4:
            return pt_l, pt_r
        # Skip uniform/binary synthetic images (already exact at mask edge) or uncalibrated flat contrast.
        if float(np.std(in_band)) < 15.0:
            return pt_l, pt_r

        def _shift_edge(
            anchor_s: float, outward_sign: float, ext_vals: np.ndarray, near_vals_mask: np.ndarray
        ) -> float:
            peak_vals = hu[near_vals_mask]
            if peak_vals.size < 2:
                return anchor_s
            peak_hu = float(np.percentile(peak_vals, 90))
            ext_hu = float(np.median(ext_vals))
            if peak_hu < 350.0 or (peak_hu - ext_hu) < 180.0:
                return anchor_s
            thresh = ext_hu + 0.50 * (peak_hu - ext_hu)
            # Search outward/inward within +-max_shift_mm around anchor_s
            w_mask = np.abs(s_arr - anchor_s) <= max_shift_mm
            idxs = np.flatnonzero(w_mask)
            if idxs.size < 3:
                return anchor_s
            vals = hu[idxs]
            if float(np.max(vals)) <= thresh or float(np.min(vals)) >= thresh:
                return anchor_s
            # Pick crossing closest to anchor_s where intensity falls outward
            order_idx = idxs[::-1] if outward_sign < 0.0 else idxs
            for k in range(len(order_idx) - 1):
                i0, i1 = int(order_idx[k]), int(order_idx[k + 1])
                v0_h, v1_h = float(hu[i0]), float(hu[i1])
                if v0_h >= thresh > v1_h and abs(v0_h - v1_h) > 1e-3:
                    frac = (v0_h - thresh) / (v0_h - v1_h)
                    return float(s_arr[i0] + frac * (s_arr[i1] - s_arr[i0]))
            return anchor_s

        new_s_l = _shift_edge(0.0, -1.0, ext_l, (s_arr >= -0.25) & (s_arr <= min(2.0, 0.4 * seg_len)))
        new_s_r = _shift_edge(
            seg_len,
            1.0,
            ext_r,
            (s_arr >= max(0.0, seg_len - min(2.0, 0.4 * seg_len))) & (s_arr <= seg_len + 0.25),
        )
        if new_s_r - new_s_l < max(2.5, seg_len - 1.5):
            return pt_l, pt_r
        return (float(pt_l[0]) + new_s_l * ux, float(pt_l[1]) + new_s_l * uy), (
            float(pt_l[0]) + new_s_r * ux,
            float(pt_l[1]) + new_s_r * uy,
        )

    @staticmethod
    def _compute_bone_widths_with_endpoints_in_cs(
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        mask_cs: np.ndarray,
        crest_v: Optional[float],
        is_mandible: bool,
        jaw_id: int,
        depths_mm: List[float],
        bone_mask: Optional[np.ndarray] = None,
        *,
        crest_uv: Optional[Tuple[float, float]] = None,
        d_apical: Optional[Tuple[float, float]] = None,
        d_bl: Optional[Tuple[float, float]] = None,
        tooth_mask: Optional[np.ndarray] = None,
        withheld_notes: Optional[List[str]] = None,
        img_cs: Optional[np.ndarray] = None,
        out_ortho_frame: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Dict[float, Tuple[float, float]], Dict[float, Tuple[Tuple[float, float], Tuple[float, float]]]
    ]:
        """Detect buccal and lingual cortical boundaries on jaw-bone-only `bone_mask`.

        Ensures `W0` and all deeper widths (`W3`, `W5`, `W6`, `W9`) share ONE orthonormal
        `(d_bl, d_apical)` frame so `W0 ⟂ H` and `W0 ∥ W3 ∥ W5` on every site, bridges
        in-socket teeth cleanly across both cortical plates, and never reports a single
        cortical socket wall beside a root as ridge width.
        """
        segments: Dict[float, Tuple[float, float]] = {}
        endpoints_uv: Dict[float, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        if crest_v is None and crest_uv is None:
            return segments, endpoints_uv

        if bone_mask is not None:
            ridge_mask = np.asarray(bone_mask, dtype=bool)
        else:
            ridge_mask = _extract_primary_ridge_component(mask_cs == jaw_id, u_coords, v_coords)
        if not ridge_mask.any():
            return segments, endpoints_uv

        u0, v0 = crest_uv if crest_uv is not None else (0.0, float(crest_v))
        au, av = d_apical if d_apical is not None else ((0.0, -1.0) if is_mandible else (0.0, 1.0))
        bu, bv = d_bl if d_bl is not None else (1.0, 0.0)

        u_min, u_max = float(u_coords[0]), float(u_coords[-1])
        v_min, v_max = float(v_coords[0]), float(v_coords[-1])
        du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
        dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
        step = min(du, dv, 0.25)
        max_half_s = float(0.5 * (u_max - u_min))
        s_grid = np.arange(-max_half_s, max_half_s + step, step, dtype=np.float64)

        ridge_f32 = ridge_mask.astype(np.float32)
        tooth_f32 = (
            np.asarray(tooth_mask, dtype=np.float32)
            if (tooth_mask is not None and np.any(tooth_mask))
            else None
        )

        # Pre-check dentate socket rims for W0 (depth == 0.0)
        socket_rims: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        if tooth_f32 is not None and 0.0 in (float(d) for d in depths_mm):
            su_arr, sv_arr = MeasurementEngine._extract_coronal_bone_surface(
                ridge_mask, u_coords, v_coords, is_mandible
            )
            if su_arr.size > 0:
                socket_rims = MeasurementEngine._find_socket_crest_rims(
                    su_arr, sv_arr, ridge_mask, tooth_mask, u_coords, v_coords, is_mandible
                )

        def _measure_caliper(
            cx: float, cy: float, c_au: float, c_av: float, c_bu: float, c_bv: float, *, allow_bridge: bool
        ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
            prof = MeasurementEngine._sample_band_profile(
                ridge_f32, u_coords, v_coords, cx, cy, s_grid, c_au, c_av, c_bu, c_bv, step
            )
            t_prof = (
                MeasurementEngine._sample_band_profile(
                    tooth_f32, u_coords, v_coords, cx, cy, s_grid, c_au, c_av, c_bu, c_bv, step
                )
                if (allow_bridge and tooth_f32 is not None)
                else None
            )
            span = MeasurementEngine._select_ridge_run_indices(prof, t_prof, s_grid, step)
            if span is None:
                return None
            s_l, s_r = MeasurementEngine._refine_run_endpoints(prof, s_grid, span[0], span[1], step)
            if s_r <= s_l:
                return None
            pt_l = (cx + s_l * c_bu, cy + s_l * c_bv)
            pt_r = (cx + s_r * c_bu, cy + s_r * c_bv)
            if pt_l[0] <= u_min + _CS_BORDER_TOL_MM or pt_r[0] >= u_max - _CS_BORDER_TOL_MM:
                return None
            return pt_l, pt_r

        def _span_bone_runs(seg: Tuple[Tuple[float, float], Tuple[float, float]]) -> int:
            """Count separate cortical runs crossed by a caliper span (1 == a single plate)."""
            (x0, y0), (x1, y1) = seg
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / max(step, 1e-3)) + 1)
            ts = np.linspace(0.0, 1.0, n)
            cols = np.rint(np.interp(x0 + ts * (x1 - x0), u_coords, np.arange(len(u_coords)))).astype(int)
            rows = np.rint(np.interp(y0 + ts * (y1 - y0), v_coords, np.arange(len(v_coords)))).astype(int)
            flags = ridge_mask[
                np.clip(rows, 0, len(v_coords) - 1), np.clip(cols, 0, len(u_coords) - 1)
            ].astype(np.int8)
            if flags.size == 0:
                return 0
            edges = np.diff(np.concatenate(([0], flags, [0])))
            return int(np.count_nonzero(edges == 1))

        def _is_single_socket_wall(seg: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
            """True when a short (<=4.5mm) 1-run segment stops right beside a segmented tooth root."""
            if tooth_f32 is None or _span_bone_runs(seg) > 1:
                return False
            (x0, y0), (x1, y1) = seg
            w_len = math.hypot(x1 - x0, y1 - y0)
            if w_len >= 4.5:
                return False
            du_seg, dv_seg = (x1 - x0) / max(1e-6, w_len), (y1 - y0) / max(1e-6, w_len)
            mid_y = 0.5 * (y0 + y1)
            for sign, end_x, end_y in ((-1.0, x0, y0), (1.0, x1, y1)):
                t_hits = 0
                for d_p in (0.5, 1.0, 1.5, 2.0, 2.5):
                    px_t = end_x + sign * d_p * du_seg
                    py_t = end_y + sign * d_p * dv_seg
                    ci_t = int(
                        np.clip(
                            np.round((px_t - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1)),
                            0,
                            len(u_coords) - 1,
                        )
                    )
                    ri_t = int(
                        np.clip(
                            np.round((py_t - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1)),
                            0,
                            len(v_coords) - 1,
                        )
                    )
                    if float(tooth_f32[ri_t, ci_t]) >= 0.5:
                        t_hits += 1
                if t_hits >= 2 and abs(mid_y - v0) <= 12.0:
                    return True
            return False

        def _wall_crest_at(col_u: float) -> Optional[float]:
            if len(u_coords) == 0:
                return None
            ci = int(np.argmin(np.abs(u_coords - col_u)))
            col = ridge_mask[:, max(0, ci - 1) : ci + 2].any(axis=1)
            idx = np.where(col)[0]
            if idx.size == 0:
                return None
            lo, hi = (v0 - 8.0, v0 + 2.0) if is_mandible else (v0 - 2.0, v0 + 8.0)
            band = [int(i) for i in idx if lo <= float(v_coords[i]) <= hi]
            if not band:
                return None
            return float(v_coords[band[-1] if is_mandible else band[0]])

        # Phase 1: Determine W0 (depth == 0.0) along the coronal ridge-normal `d_bl` FIRST so W0
        # stays perpendicular to `d_apical` even on steeply tilted jaws (`test_20`) and horizontal
        # top slices never hijack tilted ridge axes. Fall back to `socket_rims` / `c_horiz` /
        # deeper two-wall bridging only when `d_bl` cannot span both walls at the crest.
        solid_tapered_crest = False
        ordered_depths = [float(d) for d in depths_mm]
        if 0.0 in ordered_depths:
            w0_got: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
            deeper_solid_w = 0.0
            if tooth_f32 is None:
                d3_cand = _measure_caliper(u0 + 3.0 * au, v0 + 3.0 * av, au, av, bu, bv, allow_bridge=False)
                if d3_cand is not None and _span_bone_runs(d3_cand) == 1:
                    deeper_solid_w = math.hypot(d3_cand[1][0] - d3_cand[0][0], d3_cand[1][1] - d3_cand[0][1])

            best: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
            best_w = 0.0
            saw_single_wall = False
            for eff_d in (0.25, 0.5, 1.0, 1.5, 2.0):
                cx = u0 + eff_d * au
                cy = v0 + eff_d * av
                if cy < v_min - 0.5 or cy > v_max + 0.5:
                    continue
                c_primary = _measure_caliper(cx, cy, au, av, bu, bv, allow_bridge=True)
                lvl_best = c_primary
                lvl_w = (
                    math.hypot(c_primary[1][0] - c_primary[0][0], c_primary[1][1] - c_primary[0][1])
                    if c_primary is not None
                    else 0.0
                )
                if lvl_best is not None and (_is_single_socket_wall(lvl_best) or lvl_w < 3.0):
                    saw_single_wall = True
                if lvl_best is not None and not _is_single_socket_wall(lvl_best) and lvl_w > best_w:
                    best, best_w = lvl_best, lvl_w
                if (
                    lvl_best is not None
                    and not _is_single_socket_wall(lvl_best)
                    and lvl_w >= (6.0 if abs(bv) >= 0.25 else 4.6)
                ):
                    best, best_w = lvl_best, lvl_w
                    break
                if (
                    tooth_f32 is None
                    and eff_d >= 0.5
                    and lvl_best is not None
                    and 2.0 <= lvl_w < 3.0
                    and deeper_solid_w >= 4.5
                    and _span_bone_runs(lvl_best) == 1
                ):
                    best, best_w = lvl_best, lvl_w
                    solid_tapered_crest = True
                    break
            w0_got = best

            # Uneven-height dentate socket rims rescue when orthogonal `d_bl` could not bridge both walls
            # or on extreme >=40 deg slants (`|d_bl_y| >= 0.65`) where perpendicular shaft width < 6 mm.
            if (
                w0_got is None or best_w < 3.0 or (best_w < 6.0 and abs(bv) >= 0.65)
            ) and socket_rims is not None:
                l_rim, r_rim = socket_rims
                rim_dx = float(r_rim[0]) - float(l_rim[0])
                rim_dy = float(r_rim[1]) - float(l_rim[1])
                rim_deg = abs(math.degrees(math.atan2(rim_dy, max(1e-6, rim_dx))))
                if rim_deg <= 32.0:
                    v_l_s = l_rim[1] - 0.5 if is_mandible else l_rim[1] + 0.5
                    v_r_s = r_rim[1] - 0.5 if is_mandible else r_rim[1] + 0.5
                    c_l = _measure_caliper(l_rim[0], v_l_s, au, av, bu, bv, allow_bridge=False)
                    c_r = _measure_caliper(r_rim[0], v_r_s, au, av, bu, bv, allow_bridge=False)
                    pt_l = c_l[0] if c_l is not None else l_rim
                    pt_r = c_r[1] if c_r is not None else r_rim
                    if (
                        (c_l is not None or c_r is not None)
                        and pt_r[0] > pt_l[0]
                        and pt_l[0] <= l_rim[0] + 0.75
                        and pt_r[0] >= r_rim[0] - 0.75
                        and pt_l[0] > u_min + _CS_BORDER_TOL_MM
                        and pt_r[0] < u_max - _CS_BORDER_TOL_MM
                    ):
                        rim_w = math.hypot(pt_r[0] - pt_l[0], pt_r[1] - pt_l[1])
                        if rim_w >= 3.0 and (best_w < 3.0 or rim_w >= best_w + 1.5):
                            w0_got = (pt_l, pt_r)
                            best_w = rim_w

            # Horizontal crestal fallback only if tilted `d_bl` and rims both failed to span >= 3 mm
            if (w0_got is None or best_w < 3.0) and abs(bv) > 0.05 and not solid_tapered_crest:
                for eff_d in (0.5, 1.0, 1.5):
                    cy = v0 + eff_d * av
                    if cy < v_min - 0.5 or cy > v_max + 0.5:
                        continue
                    c_horiz = _measure_caliper(
                        u0, cy, 0.0, -1.0 if is_mandible else 1.0, 1.0, 0.0, allow_bridge=True
                    )
                    if c_horiz is not None:
                        w_h = math.hypot(c_horiz[1][0] - c_horiz[0][0], c_horiz[1][1] - c_horiz[0][1])
                        if _is_single_socket_wall(c_horiz) or w_h < 3.0:
                            saw_single_wall = True
                        elif w_h >= 3.0 and w_h > best_w:
                            w0_got = c_horiz
                            best_w = w_h
                            break

            # If W0 missed without encountering a single-wall dehiscence (or hit a short single run),
            # rescue from a deeper two-wall bridge (`3..6 mm`)
            if not solid_tapered_crest and (
                (w0_got is None and not saw_single_wall)
                or (
                    w0_got is not None
                    and math.hypot(w0_got[1][0] - w0_got[0][0], w0_got[1][1] - w0_got[0][1]) < 3.0
                    and _span_bone_runs(w0_got) <= 1
                )
            ):
                for rescue_d in (3.0, 4.5, 5.0, 6.0):
                    cx_r, cy_r = u0 + rescue_d * au, v0 + rescue_d * av
                    if cy_r < v_min - 0.5 or cy_r > v_max + 0.5:
                        continue
                    cand = _measure_caliper(cx_r, cy_r, au, av, bu, bv, allow_bridge=True)
                    if cand is None:
                        continue
                    c_len = math.hypot(cand[1][0] - cand[0][0], cand[1][1] - cand[0][1])
                    if c_len < 3.0 or _span_bone_runs(cand) < 2:
                        continue
                    v_l = _wall_crest_at(float(cand[0][0]))
                    v_r = _wall_crest_at(float(cand[1][0]))
                    if v_l is None or v_r is None:
                        continue
                    interp = ((float(cand[0][0]), v_l), (float(cand[1][0]), v_r))
                    i_len = math.hypot(interp[1][0] - interp[0][0], interp[1][1] - interp[0][1])
                    if (
                        i_len >= 3.0
                        and interp[0][0] > u_min + _CS_BORDER_TOL_MM
                        and interp[1][0] < u_max - _CS_BORDER_TOL_MM
                        and _span_bone_runs(interp) >= 2
                    ):
                        w0_got = interp
                        break

            # Grayscale outer-cortex refinement followed by BL-12 < 3.0 mm single-plate guard
            if w0_got is not None:
                w0_got = MeasurementEngine._refine_width_with_gray_edges(
                    img_cs, u_coords, v_coords, w0_got[0], w0_got[1]
                )
            if w0_got is not None and not solid_tapered_crest:
                w0_len = math.hypot(w0_got[1][0] - w0_got[0][0], w0_got[1][1] - w0_got[0][1])
                if w0_len < 3.0 or _is_single_socket_wall(w0_got):
                    w0_got = None
                    if withheld_notes is not None:
                        withheld_notes.append(WITHHELD_W0_SINGLE_CORTEX)
            elif w0_got is None and saw_single_wall and withheld_notes is not None:
                withheld_notes.append(WITHHELD_W0_SINGLE_CORTEX)
            if w0_got is not None:
                segments[0.0] = (float(w0_got[0][0]), float(w0_got[1][0]))
                endpoints_uv[0.0] = w0_got
                # Synchronize the orthogonal caliper/height frame directly to the verified W0 chord
                # whenever the chord agrees within 20 deg of the coronal shaft axis (keeping steeply
                # slanted >22 deg shafts pointed cleanly down the ridge body).
                w_dx = float(w0_got[1][0]) - float(w0_got[0][0])
                w_dy = float(w0_got[1][1]) - float(w0_got[0][1])
                w_l = math.hypot(w_dx, w_dy)
                if w_l >= 2.0:
                    w_deg_abs = abs(math.degrees(math.atan2(w_dy, max(1e-6, w_dx))))
                    w_deg_signed = (
                        math.degrees(math.atan2(w_dy, max(1e-6, w_dx)))
                        if is_mandible
                        else -math.degrees(math.atan2(w_dy, max(1e-6, w_dx)))
                    )
                    init_axis_deg = (
                        math.degrees(math.atan2(au, -av)) if is_mandible else math.degrees(math.atan2(au, av))
                    )
                    u0 = 0.5 * (float(w0_got[0][0]) + float(w0_got[1][0]))
                    v0 = 0.5 * (float(w0_got[0][1]) + float(w0_got[1][1]))
                    if out_ortho_frame is not None:
                        out_ortho_frame["crest_uv"] = (u0, v0)
                    if w_deg_abs <= 45.0 and abs(w_deg_signed - init_axis_deg) <= 10.0:
                        bu, bv = w_dx / w_l, w_dy / w_l
                        if is_mandible:
                            au, av = bv, -bu
                        else:
                            au, av = -bv, bu
                        if out_ortho_frame is not None:
                            out_ortho_frame["d_apical"] = (au, av)
                            out_ortho_frame["d_bl"] = (bu, bv)
                            ang_val = (
                                math.degrees(math.atan2(au, -av))
                                if is_mandible
                                else math.degrees(math.atan2(au, av))
                            )
                            out_ortho_frame["ridge_angle_deg"] = round(
                                float(np.clip(ang_val, -45.0, 45.0)), 1
                            )
                    elif abs(w_deg_signed - init_axis_deg) > 10.0 and not (
                        abs(init_axis_deg) > 22.0 and w_deg_abs <= 12.0
                    ):
                        # Enforce strict orthogonality W0 ⟂ H unless keeping a flat crest on a >22 deg slanted shaft.
                        snapped = None
                        for step_d in (0.0, 0.5, 1.0):
                            cand_s = _measure_caliper(
                                u0 + step_d * au, v0 + step_d * av, au, av, bu, bv, allow_bridge=True
                            )
                            if cand_s is not None and not _is_single_socket_wall(cand_s):
                                s_len = math.hypot(cand_s[1][0] - cand_s[0][0], cand_s[1][1] - cand_s[0][1])
                                if s_len >= 3.0 or solid_tapered_crest:
                                    snapped = cand_s
                                    break
                        if snapped is not None:
                            w0_got = snapped
                            u0 = 0.5 * (float(w0_got[0][0]) + float(w0_got[1][0]))
                            v0 = 0.5 * (float(w0_got[0][1]) + float(w0_got[1][1]))
                        else:
                            proj_half = 0.5 * abs(w_dx * bu + w_dy * bv)
                            if proj_half >= (1.5 if not solid_tapered_crest else 0.75):
                                w0_got = (
                                    (u0 - proj_half * bu, v0 - proj_half * bv),
                                    (u0 + proj_half * bu, v0 + proj_half * bv),
                                )
                        segments[0.0] = (float(w0_got[0][0]), float(w0_got[1][0]))
                        endpoints_uv[0.0] = w0_got
                        if out_ortho_frame is not None:
                            out_ortho_frame["crest_uv"] = (u0, v0)

        # Phase 2: Measure all deeper widths strictly parallel to W0 along (u0, v0) + depth * d_apical
        for fdepth in ordered_depths:
            if fdepth == 0.0:
                continue
            cx = u0 + fdepth * au
            cy = v0 + fdepth * av
            if cy < v_min - 0.5 or cy > v_max + 0.5:
                continue
            res = _measure_caliper(cx, cy, au, av, bu, bv, allow_bridge=True)
            if res is not None and _is_single_socket_wall(res):
                # Probe +/-0.5 mm apically before withholding a collapsed single-wall dentate cut
                rescued = None
                for delta_d in (-0.5, 0.5):
                    alt = _measure_caliper(
                        cx + delta_d * au, cy + delta_d * av, au, av, bu, bv, allow_bridge=True
                    )
                    if alt is not None and not _is_single_socket_wall(alt):
                        rescued = alt
                        break
                res = rescued
            if res is not None:
                res = MeasurementEngine._refine_width_with_gray_edges(
                    img_cs, u_coords, v_coords, res[0], res[1]
                )
                segments[fdepth] = (float(res[0][0]), float(res[1][0]))
                endpoints_uv[fdepth] = res

        return segments, endpoints_uv

    # ------------------------------------------------------------------
    # Root Extent Detection (Strictly Separate from Available Bone Height — Part 1.2)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_root_extent_in_cs(
        mask_cs: np.ndarray,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        tooth_id: Optional[int],
        is_mandible: bool,
        crest_uv: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Extract tooth reference point to actual root apex from the tooth label ONLY."""
        if tooth_id is None:
            return None, None
        t_mask = mask_cs == int(tooth_id)
        if not t_mask.any():
            return None, None

        # Isolate tooth connected component closest to u = 0
        t_comp = _extract_primary_ridge_component(t_mask, u_coords, v_coords)
        rows, cols = np.where(t_comp)
        if rows.size < 3:
            return None, None

        if is_mandible:
            apex_row = int(rows.min())
            crown_row = int(rows.max())
        else:
            apex_row = int(rows.max())
            crown_row = int(rows.min())

        apex_cols = cols[rows == apex_row]
        crown_cols = cols[rows == crown_row]
        apex_uv = (float(np.mean(u_coords[apex_cols])), float(v_coords[apex_row]))

        # Anchor root extent start at crest level if tooth passes through crest, else coronal tooth reference
        if crest_uv is not None:
            v_crest = float(crest_uv[1])
            if (is_mandible and v_crest > apex_uv[1] + 0.5) or (
                (not is_mandible) and v_crest < apex_uv[1] - 0.5
            ):
                c_row = int(np.argmin(np.abs(v_coords - v_crest)))
                c_cols = cols[np.abs(rows - c_row) <= 2]
                start_u = float(np.mean(u_coords[c_cols])) if c_cols.size > 0 else apex_uv[0]
                return (start_u, v_crest), apex_uv

        start_uv = (float(np.mean(u_coords[crown_cols])), float(v_coords[crown_row]))
        return start_uv, apex_uv

    def _measure_from_cs(
        self,
        fdi: str,
        info: Any,
        is_mandible: bool,
        jaw_id: int,
        tooth: Optional[ToothInfo],
        cs: CrossSectionData,
        site_source: str,
        is_approx_site: bool,
    ) -> Optional[ImplantSiteMeasurement]:
        """Compute all canonical clinical measurements and geometry from one resampled cross-section."""
        u_coords = cs.u_coords
        v_coords = cs.v_coords
        mask_cs = cs.mask_cs
        img_cs = cs.img_cs
        P, _T, N, B = cs.frame
        cs.is_mandible = is_mandible
        self.last_withheld_reason = None

        tooth_id: Optional[int] = None
        if tooth is not None:
            tooth_id = int(tooth.label_id)
        elif self.is_dental_segmentator_5class:
            tooth_id = get_tooth_group_label_id(is_mandible, is_5class=True)

        # 1. Require real jawbone segmentation — never fall back to intensity thresholds (Parts 3.2, 7.2)
        same_jaw_tooth = _same_jaw_tooth_mask(
            mask_cs, is_mandible, self.is_dental_segmentator_5class, tooth_id=tooth_id
        )
        raw_bone = _ridge_envelope_mask(
            mask_cs, u_coords, v_coords, jaw_id, tooth_id, is_mandible, tooth_mask=same_jaw_tooth
        )
        if not raw_bone.any():
            missing_reason = WITHHELD_NO_MANDIBLE_SEG if is_mandible else WITHHELD_NO_MAXILLA_SEG
            cs.withheld_reason = missing_reason
            cs.provenance = PROVENANCE_UNAVAILABLE
            cs.quality_status = QUALITY_UNAVAILABLE
            self.last_withheld_reason = missing_reason
            return None

        # 2. Extract primary connected alveolar ridge (keeping both socket walls via tooth envelope,
        #    while rejecting stray voxels and disconnected lateral bone islands).
        ridge_mask = _extract_primary_ridge_component(raw_bone, u_coords, v_coords)
        # Bone-only mask within the primary ridge: keeps both lingual and buccal socket walls,
        # never includes tooth/pulp voxels or disconnected lateral islands.
        bone_component = (mask_cs == int(jaw_id)) & ridge_mask
        width_reason, height_reason = _ridge_region_withheld_reasons(ridge_mask, u_coords, v_coords)
        width_ok, height_ok = width_reason is None, height_reason is None
        if not width_ok and not height_ok:
            cs.withheld_reason = _join_reasons(width_reason, height_reason)
            cs.provenance = PROVENANCE_UNAVAILABLE
            cs.quality_status = QUALITY_UNAVAILABLE
            self.last_withheld_reason = cs.withheld_reason
            logger.info("Cross-section at FDI %s rejected: %s", fdi, cs.withheld_reason)
            return None

        # 3. Anatomy-aware local alveolar crest & ridge axis detection (Part 3.3)
        crest_uv, d_apical, d_bl, ridge_angle_deg, crest_profile_uv = self._detect_alveolar_crest_and_axis(
            ridge_mask, u_coords, v_coords, is_mandible, bone_mask=bone_component, tooth_mask=same_jaw_tooth
        )
        cs.crest_uv = crest_uv
        cs.crest_v = crest_uv[1] if crest_uv is not None else None
        cs.height_u = crest_uv[0] if crest_uv is not None else 0.0
        cs.crest_profile_uv = crest_profile_uv if len(crest_profile_uv) >= 2 else None
        cs.ridge_axis_angle_deg = ridge_angle_deg

        records: Dict[str, MeasurementRecord] = {}
        withheld_reason: Optional[str] = None

        # 4. Anatomy-aware buccal & lingual widths (Part 3.4) — computed before height so H starts at the W0 midline
        depths = [0.0, 3.0, 5.0, 6.0, 9.0]
        if width_ok:
            w0_notes: List[str] = []
            ortho_frame: Dict[str, Any] = {}
            segs_u, segs_uv = self._compute_bone_widths_with_endpoints_in_cs(
                u_coords,
                v_coords,
                mask_cs,
                cs.crest_v,
                is_mandible,
                jaw_id,
                depths,
                bone_mask=bone_component,
                crest_uv=crest_uv,
                d_apical=d_apical,
                d_bl=d_bl,
                tooth_mask=same_jaw_tooth,
                withheld_notes=w0_notes,
                img_cs=img_cs,
                out_ortho_frame=ortho_frame,
            )
            if ortho_frame:
                crest_uv = ortho_frame.get("crest_uv", crest_uv)
                d_apical = ortho_frame.get("d_apical", d_apical)
                d_bl = ortho_frame.get("d_bl", d_bl)
                if "ridge_angle_deg" in ortho_frame:
                    cs.ridge_axis_angle_deg = ortho_frame["ridge_angle_deg"]
                if crest_uv is not None:
                    cs.crest_uv = crest_uv
                    cs.crest_v = crest_uv[1]
                    cs.height_u = crest_uv[0]
            cs.width_segments = segs_u
            cs.width_endpoints_uv = segs_uv
            widths: Dict[float, Optional[float]] = {}
            for depth in depths:
                if depth in segs_uv:
                    pt_l, pt_r = segs_uv[depth]
                    w_sw = _uv_to_world(P, N, B, pt_l[0], pt_l[1])
                    w_ew = _uv_to_world(P, N, B, pt_r[0], pt_r[1])
                    w_val = round(float(np.linalg.norm(np.asarray(w_ew) - np.asarray(w_sw))), 2)
                    widths[depth] = w_val
                    records[f"width_{depth:g}"] = MeasurementRecord(
                        measurement_type=f"bone_width_w{depth:g}",
                        value_mm=w_val,
                        start_world_mm=w_sw,
                        end_world_mm=w_ew,
                        start_uv_mm=pt_l,
                        end_uv_mm=pt_r,
                        source_labels=(jaw_id,),
                        provenance=PROVENANCE_LABEL_DERIVED,
                        quality=QUALITY_VALID,
                    )
                else:
                    widths[depth] = None
                    if depth == 0.0 and w0_notes:
                        records["width_0"] = MeasurementRecord(
                            measurement_type="bone_width_w0",
                            value_mm=None,
                            source_labels=(jaw_id,),
                            provenance=PROVENANCE_UNAVAILABLE,
                            quality=QUALITY_UNAVAILABLE,
                            withheld_reason=w0_notes[0],
                        )
            if w0_notes:
                withheld_reason = _join_reasons(withheld_reason, *w0_notes)
            if 0.0 in segs_uv and crest_uv is not None:
                w0_l, w0_r = segs_uv[0.0]
                w0_mid_u = 0.5 * (float(w0_l[0]) + float(w0_r[0]))
                w0_mid_v = 0.5 * (float(w0_l[1]) + float(w0_r[1]))
                # Enforce strict T-junction: H starts directly at W0 midpoint and proceeds strictly
                # apically along d_apical with zero coronal overhang past W0 (T shape, never '+').
                crest_uv = (w0_mid_u, w0_mid_v)
                cs.crest_uv = crest_uv
                cs.crest_v = crest_uv[1]
                cs.height_u = crest_uv[0]
        else:
            cs.width_segments = {}
            cs.width_endpoints_uv = {}
            widths = {depth: None for depth in depths}
            withheld_reason = _join_reasons(withheld_reason, width_reason)

        # 5. Local ipsilateral vital-structure selection (Part 2.2)
        vital_name = "Inferior Alveolar Canal" if is_mandible else "Maxillary Sinus Floor"
        vital_id, local_vital_2d, vital_centroid_uv, vital_contour_uv, vital_unavail = (
            self._select_local_vital_structure(
                mask_cs, u_coords, v_coords, P, N, B, info.quadrant, is_mandible, crest_uv
            )
        )
        cs.vital_structure_name = vital_name
        cs.vital_withheld_reason = vital_unavail
        cs.vital_contour_uv = vital_contour_uv
        if is_mandible:
            cs.canal_centroid_uv = vital_centroid_uv
        else:
            cs.sinus_floor_uv = vital_contour_uv

        # 6. Available bone height (W0 midline -> ipsilateral vital boundary or endosteal basal cortex; NEVER root apex!)
        bone_height_mm, vital_dist_mm, basal_uv, vital_boundary_uv, height_prov = (
            self._compute_bone_height_in_cs(
                u_coords,
                v_coords,
                mask_cs,
                P,
                N,
                B,
                is_mandible,
                jaw_id,
                vital_id,
                ridge_mask=ridge_mask,
                crest_uv=crest_uv,
                d_apical=d_apical,
                local_vital_mask_2d=local_vital_2d,
                img_cs=img_cs,
            )
        )
        cs.vital_boundary_uv = vital_boundary_uv

        if not height_ok:
            bone_height_mm = None
            basal_uv = None
            withheld_reason = _join_reasons(height_reason, withheld_reason)

        cs.basal_uv = basal_uv
        cs.basal_v = basal_uv[1] if basal_uv is not None else None

        if crest_uv is not None and basal_uv is not None and bone_height_mm is not None:
            h_start_w = _uv_to_world(P, N, B, crest_uv[0], crest_uv[1])
            h_end_w = _uv_to_world(P, N, B, basal_uv[0], basal_uv[1])
            exact_h = round(float(np.linalg.norm(np.asarray(h_end_w) - np.asarray(h_start_w))), 2)
            bone_height_mm = exact_h
            src_labels = (
                (jaw_id, vital_id) if (vital_id is not None and vital_boundary_uv is not None) else (jaw_id,)
            )
            records["available_bone_height"] = MeasurementRecord(
                measurement_type="available_bone_height",
                value_mm=exact_h,
                start_world_mm=h_start_w,
                end_world_mm=h_end_w,
                start_uv_mm=crest_uv,
                end_uv_mm=basal_uv,
                source_labels=tuple(int(x) for x in src_labels if x is not None),
                provenance=height_prov,
                quality=QUALITY_VALID if height_prov == PROVENANCE_LABEL_DERIVED else QUALITY_ESTIMATED,
            )
        else:
            records["available_bone_height"] = MeasurementRecord(
                measurement_type="available_bone_height",
                value_mm=None,
                provenance=PROVENANCE_UNAVAILABLE,
                quality=QUALITY_UNAVAILABLE,
                withheld_reason=height_reason or WITHHELD_NO_CREST,
            )

        # 7. Separate root extent measurement (tooth reference -> actual root apex)
        root_start_uv, root_apex_uv = self._detect_root_extent_in_cs(
            mask_cs, u_coords, v_coords, tooth_id, is_mandible, crest_uv
        )
        cs.root_start_uv = root_start_uv
        cs.root_apex_uv = root_apex_uv
        cs.root_apex_v = root_apex_uv[1] if root_apex_uv is not None else None
        root_extent_mm: Optional[float] = None
        if root_start_uv is not None and root_apex_uv is not None:
            r_sw = _uv_to_world(P, N, B, root_start_uv[0], root_start_uv[1])
            r_ew = _uv_to_world(P, N, B, root_apex_uv[0], root_apex_uv[1])
            root_extent_mm = round(float(np.linalg.norm(np.asarray(r_ew) - np.asarray(r_sw))), 2)
            if root_extent_mm > 0.0:
                records["root_extent"] = MeasurementRecord(
                    measurement_type="root_extent",
                    value_mm=root_extent_mm,
                    start_world_mm=r_sw,
                    end_world_mm=r_ew,
                    start_uv_mm=root_start_uv,
                    end_uv_mm=root_apex_uv,
                    source_labels=(int(tooth_id),) if tooth_id is not None else (),
                    provenance=PROVENANCE_LABEL_DERIVED,
                    quality=QUALITY_VALID,
                )
            else:
                root_extent_mm = None

        if cs.crest_v is None:
            withheld_reason = _join_reasons(withheld_reason, WITHHELD_NO_CREST)
        cs.withheld_reason = withheld_reason
        self.last_withheld_reason = withheld_reason

        crest_w = widths.get(0.0)
        mid_w = widths.get(6.0) if widths.get(6.0) is not None else widths.get(3.0)
        apical_w = widths.get(9.0)
        mesiodistal_mm = self._compute_mesiodistal_space(fdi)
        mean_density_hu, nnt_summary = self._compute_site_bone_density_in_cs(
            u_coords, v_coords, mask_cs, img_cs, cs.crest_v, is_mandible, bone_height_mm, jaw_id
        )
        density_class = self._classify_misch_density(mean_density_hu) if mean_density_hu is not None else None
        cs.misch_code = self.extract_misch_code(density_class)

        # 8. Centralized implant suggestion (Part 5)
        suggested_diam, suggested_len, suggested_ang = suggest_implant_dimensions(
            crest_w, mid_w, bone_height_mm, mesiodistal_mm
        )
        cs.suggested_implant_diameter_mm = suggested_diam
        cs.suggested_implant_length_mm = suggested_len
        safety_margin = (
            round(float(max(0.0, bone_height_mm - suggested_len)), 2)
            if (bone_height_mm is not None and suggested_len is not None)
            else None
        )

        # 9. Separate vital-structure clearance from proposed implant apex (Part 1.2)
        vital_clearance_mm: Optional[float] = None
        if vital_dist_mm is not None and suggested_len is not None and crest_uv is not None:
            vital_clearance_mm = round(float(max(0.0, vital_dist_mm - suggested_len)), 2)
            imp_tip_uv = (
                crest_uv[0] + suggested_len * d_apical[0],
                crest_uv[1] + suggested_len * d_apical[1],
            )
            v_end_uv = (
                vital_boundary_uv
                if vital_boundary_uv is not None
                else (
                    crest_uv[0] + vital_dist_mm * d_apical[0],
                    crest_uv[1] + vital_dist_mm * d_apical[1],
                )
            )
            vc_sw = _uv_to_world(P, N, B, imp_tip_uv[0], imp_tip_uv[1])
            vc_ew = _uv_to_world(P, N, B, v_end_uv[0], v_end_uv[1])
            records["vital_clearance"] = MeasurementRecord(
                measurement_type="vital_clearance",
                value_mm=vital_clearance_mm,
                start_world_mm=vc_sw,
                end_world_mm=vc_ew,
                start_uv_mm=imp_tip_uv,
                end_uv_mm=v_end_uv,
                source_labels=(int(vital_id),) if vital_id is not None else (),
                provenance=height_prov,
                quality=QUALITY_VALID
                if vital_clearance_mm >= IMPLANT_APICAL_SAFETY_MARGIN_MM
                else QUALITY_WARNING,
            )
        elif vital_unavail:
            records["vital_clearance"] = MeasurementRecord(
                measurement_type="vital_clearance",
                value_mm=None,
                provenance=PROVENANCE_UNAVAILABLE,
                quality=QUALITY_UNAVAILABLE,
                withheld_reason=vital_unavail,
            )

        buccal_plate_w, lingual_plate_w = self._compute_cortical_plates_in_cs(
            u_coords, v_coords, mask_cs, cs.crest_v, is_mandible, jaw_id, tooth=tooth
        )
        cs.buccal_plate_mm = buccal_plate_w
        cs.lingual_plate_mm = lingual_plate_w

        overall_prov = (
            height_prov
            if bone_height_mm is not None
            else (PROVENANCE_LABEL_DERIVED if crest_w is not None else PROVENANCE_UNAVAILABLE)
        )
        if is_approx_site and overall_prov == PROVENANCE_LABEL_DERIVED:
            overall_quality = QUALITY_ESTIMATED
        elif withheld_reason is not None:
            overall_quality = QUALITY_WARNING
        elif overall_prov == PROVENANCE_UNAVAILABLE:
            overall_quality = QUALITY_UNAVAILABLE
        else:
            overall_quality = QUALITY_VALID

        cs.measurements = records
        cs.provenance = overall_prov
        cs.quality_status = overall_quality

        return ImplantSiteMeasurement(
            tooth_fdi=fdi,
            bone_height_mm=bone_height_mm,
            bone_width_at_depths_mm=widths,
            crest_width_mm=crest_w,
            mid_root_width_mm=mid_w,
            apical_width_mm=apical_w,
            mesiodistal_space_mm=mesiodistal_mm,
            mean_density_hu=mean_density_hu,
            misch_density_class=density_class,
            distance_to_vital_structure_mm=vital_dist_mm,
            vital_structure_name=vital_name,
            safety_margin_mm=safety_margin,
            suggested_diameter_mm=suggested_diam,
            suggested_length_mm=suggested_len,
            suggested_angulation_deg=suggested_ang,
            is_ai_suggested=True,
            ai_suggestion_confirmed=False,
            site_position_source=site_source,
            is_approximate_site=is_approx_site,
            buccal_bone_thickness_mm=buccal_plate_w,
            lingual_bone_thickness_mm=lingual_plate_w,
            tooth_length_mm=tooth.total_length_mm if tooth is not None else None,
            crown_length_mm=tooth.crown_length_mm if tooth is not None else None,
            root_length_mm=tooth.root_length_mm if tooth is not None else None,
            crown_to_root_ratio=tooth.crown_to_root_ratio if tooth is not None else None,
            nnt_density_summary=nnt_summary,
            root_extent_mm=root_extent_mm,
            vital_clearance_mm=vital_clearance_mm,
            measurements=records,
            provenance=overall_prov,
            quality_status=overall_quality,
            implant_suggestion_disclaimer=IMPLANT_SUGGESTION_DISCLAIMER,
            withheld_reason=withheld_reason,
        )

    def _resample_frame_data(
        self,
        frame: Any,
        *,
        half_u_mm: float = 15.0,
        half_v_mm: float = 25.0,
        step_mm: float = 0.25,
        is_mandible: Optional[bool] = None,
    ) -> CrossSectionData:
        """Resample label mask and image once on a cross-section at an explicit arch frame."""
        P, T, N, B = frame
        if is_mandible is not None:
            v_lo, v_hi = self._jaw_v_bounds_at_frame(
                np.asarray(P, dtype=np.float64),
                np.asarray(N, dtype=np.float64),
                np.asarray(B, dtype=np.float64),
                is_mandible=bool(is_mandible),
                base_half_v_mm=half_v_mm,
                step_mm=step_mm,
            )
        else:
            v_lo, v_hi = -float(half_v_mm), float(half_v_mm)
        n_u = int(round(2.0 * half_u_mm / step_mm)) + 1
        n_v = int(round((v_hi - v_lo) / step_mm)) + 1
        u_coords = np.linspace(-half_u_mm, half_u_mm, n_u)
        v_coords = np.linspace(v_lo, v_hi, n_v)
        uu, vv = np.meshgrid(u_coords, v_coords)
        world_grid = (
            np.asarray(P)[None, None, :]
            + uu[:, :, None] * np.asarray(N)[None, None, :]
            + vv[:, :, None] * np.asarray(B)[None, None, :]
        )
        mask_cs = self.volume.sample_mask_world_points(self.mask, world_grid)
        img_cs = self.volume.sample_world_points(world_grid, order=1, cval=-1000.0)
        return CrossSectionData(
            u_coords=u_coords,
            v_coords=v_coords,
            mask_cs=mask_cs,
            img_cs=img_cs,
            frame=(np.asarray(P), np.asarray(T), np.asarray(N), np.asarray(B)),
            crest_v=None,
            width_segments={},
            canal_centroid_uv=None,
        )

    def measure_at_arc_mm(
        self, jaw: str, arc_mm: float, fdi_hint: Optional[str] = None
    ) -> Tuple[Optional[ImplantSiteMeasurement], Optional[CrossSectionData]]:
        """Measure a single cross-section at `arc_mm` along the arch with one resample."""
        norm_hint = resolve_fdi_number(fdi_hint) if fdi_hint is not None else None
        # Accept display names ("Mandible"/"Maxilla") as well as arch keys ("lower"/"upper"):
        # reading "Mandible" as the maxilla made every mandibular cross-section withhold its bone.
        jaw_key = jaw_key_from_text(jaw)
        if jaw_key is None:
            jaw_key = "lower" if is_mandibular_fdi(norm_hint) else "upper"
        cache_key = (jaw_key, round(float(arc_mm), 4), norm_hint)
        if cache_key in self._arc_cache:
            cached_meas, cached_cs = self._arc_cache[cache_key]
            self.last_withheld_reason = (
                cached_meas.withheld_reason
                if cached_meas is not None
                else (cached_cs.withheld_reason if cached_cs is not None else None)
            )
            return cached_meas, cached_cs

        arch = self.arches.get(jaw_key) or self.arch
        if arch is None:
            self.last_withheld_reason = WITHHELD_NO_ARCH
            return None, None
        frame = arch.get_frame_at_mm(float(arc_mm))
        is_mandible = jaw_key == "lower"
        cs = self._resample_frame_data(
            (frame.point, frame.tangent, frame.normal, frame.binormal),
            is_mandible=is_mandible,
        )
        cs.is_mandible = is_mandible

        fdi = norm_hint or ""
        if not fdi:
            seg_centroids = {
                key: value.centroid_mm
                for key, value in self.teeth.items()
                if (is_mandibular_fdi(key) == is_mandible)
            }
            site_map = compute_fdi_arc_positions_mm(
                total_arc_length_mm=arch.total_arc_length,
                jaw_type=jaw_key,
                arch=arch,
                segmented_centroids=seg_centroids,
                tooth_bearing_span_mm=self._tooth_bearing_span_mm(arch, jaw_key),
            )
            fdi = min(site_map, key=lambda key: abs(site_map[key].arc_mm - float(arc_mm)))
        info = get_tooth_label(fdi)
        if info is None:
            return None, cs
        jaw_id = self._jaw_id(is_mandible)
        try:
            meas = self._measure_from_cs(
                fdi,
                info,
                is_mandible,
                jaw_id,
                self._get_tooth_info(fdi),
                cs,
                "arch",
                False,
            )
        except Exception:
            logger.exception("Cross-section metrics failed at %.2f mm", arc_mm)
            if cs is not None:
                cs.withheld_reason = _join_reasons(cs.withheld_reason, WITHHELD_METRIC_FAILED)
            self.last_withheld_reason = WITHHELD_METRIC_FAILED
            meas = None
        self._arc_cache[cache_key] = (meas, cs)
        return meas, cs

    # ------------------------------------------------------------------
    # Entry (crest) & W0 angle continuity across a slice series
    # ------------------------------------------------------------------
    @staticmethod
    def _jaw_slices_similar(cs_a: CrossSectionData, cs_b: CrossSectionData) -> bool:
        """True when two adjacent cross-sections depict similar jawbone morphology."""
        w_eps_a = getattr(cs_a, "width_endpoints_uv", None) or {}
        w_eps_b = getattr(cs_b, "width_endpoints_uv", None) or {}
        if 0.0 not in w_eps_a or 0.0 not in w_eps_b:
            return False
        mask_a = getattr(cs_a, "mask_cs", None)
        mask_b = getattr(cs_b, "mask_cs", None)
        is_mand_a = bool(getattr(cs_a, "is_mandible", True))
        is_mand_b = bool(getattr(cs_b, "is_mandible", True))
        if is_mand_a != is_mand_b:
            return False
        if (
            mask_a is not None
            and mask_b is not None
            and getattr(mask_a, "shape", None) == getattr(mask_b, "shape", None)
            and bool(np.any(mask_a))
            and bool(np.any(mask_b))
        ):
            use_5 = infer_mask_is_5class(mask_a)
            jaw_id = get_jaw_label_id(is_mand_a, is_5class=use_5)
            ja = mask_a == jaw_id
            jb = mask_b == jaw_id
            u_cnt = int(np.count_nonzero(ja | jb))
            if u_cnt > 0:
                iou = float(np.count_nonzero(ja & jb)) / float(u_cnt)
                return iou >= 0.58
        # Shape-invariant geometry check (covers jaw-aware dynamic v_bounds differing by a few rows)
        a0, a1 = w_eps_a[0.0]
        b0, b1 = w_eps_b[0.0]
        wa = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
        wb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
        mid_va = 0.5 * (float(a0[1]) + float(a1[1]))
        mid_vb = 0.5 * (float(b0[1]) + float(b1[1]))
        h_a = getattr(cs_a, "measurements", {}).get("available_bone_height")
        h_b = getattr(cs_b, "measurements", {}).get("available_bone_height")
        if h_a is not None and h_b is not None and h_a.value_mm is not None and h_b.value_mm is not None:
            if abs(float(h_a.value_mm) - float(h_b.value_mm)) > 2.5:
                return False
        return abs(wa - wb) <= 3.0 and abs(mid_va - mid_vb) <= 3.0

    def _smooth_adjacent_w0_angles(
        self,
        series: List[Tuple[Optional[ImplantSiteMeasurement], CrossSectionData]],
        *,
        max_angle_jump_deg: float = 8.0,
    ) -> None:
        """Suppress jagged slice-to-slice W0 angle swings across similar adjacent jaw cross-sections."""
        if len(series) < 2:
            return
        angles: Dict[int, float] = {}
        for idx, (_m, cs) in enumerate(series):
            w_eps = getattr(cs, "width_endpoints_uv", None) or {}
            if 0.0 not in w_eps:
                continue
            p0, p1 = w_eps[0.0]
            dx = float(p1[0]) - float(p0[0])
            dy = float(p1[1]) - float(p0[1])
            if dx < 0.0:
                dx, dy = -dx, -dy
            if math.hypot(dx, dy) >= 1.5:
                angles[idx] = float(np.clip(math.degrees(math.atan2(dy, dx)), -45.0, 45.0))
        if len(angles) < 2:
            return

        smoothed = dict(angles)
        # Step 1: Smooth isolated 3-slice angle spikes when both neighbors agree cleanly.
        for idx in sorted(angles.keys()):
            if (idx - 1) in angles and (idx + 1) in angles:
                cs_prev, cs_cur, cs_next = series[idx - 1][1], series[idx][1], series[idx + 1][1]
                if self._jaw_slices_similar(cs_prev, cs_cur) and self._jaw_slices_similar(cs_cur, cs_next):
                    smoothed[idx] = 0.25 * angles[idx - 1] + 0.50 * angles[idx] + 0.25 * angles[idx + 1]

        # Step 2: Enforce adjacent pairwise delta <= max_angle_jump_deg across similar consecutive slices
        # using bidirectional relaxation so neither direction retains a step cliff.
        max_step = max(1.0, float(max_angle_jump_deg))
        keys_asc = sorted(smoothed.keys())
        for _pass in range(2):
            for idx in keys_asc[1:]:
                if (idx - 1) in smoothed and self._jaw_slices_similar(series[idx - 1][1], series[idx][1]):
                    diff = smoothed[idx] - smoothed[idx - 1]
                    if abs(diff) > max_step:
                        smoothed[idx] = smoothed[idx - 1] + math.copysign(max_step, diff)
            for idx in reversed(keys_asc[:-1]):
                if (idx + 1) in smoothed and self._jaw_slices_similar(series[idx][1], series[idx + 1][1]):
                    diff = smoothed[idx] - smoothed[idx + 1]
                    if abs(diff) > max_step:
                        smoothed[idx] = smoothed[idx + 1] + math.copysign(max_step, diff)

        for idx, new_deg in smoothed.items():
            old_deg = angles.get(idx)
            if old_deg is None or abs(new_deg - old_deg) < 0.75:
                continue
            meas, cs = series[idx]
            w_eps = getattr(cs, "width_endpoints_uv", None) or {}
            if 0.0 not in w_eps or cs.frame is None:
                continue
            p0, p1 = w_eps[0.0]
            mid_u = 0.5 * (float(p0[0]) + float(p1[0]))
            mid_v = 0.5 * (float(p0[1]) + float(p1[1]))
            half_w = 0.5 * math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
            rad = math.radians(new_deg)
            c_bu, c_bv = math.cos(rad), math.sin(rad)
            is_mand = bool(getattr(cs, "is_mandible", True))

            new_p0: Optional[Tuple[float, float]] = None
            new_p1: Optional[Tuple[float, float]] = None
            # Re-trace cortical bone boundaries along smoothed W0 direction when mask_cs exists.
            mask_cs = getattr(cs, "mask_cs", None)
            u_coords = getattr(cs, "u_coords", None)
            v_coords = getattr(cs, "v_coords", None)
            if mask_cs is not None and u_coords is not None and v_coords is not None and np.any(mask_cs):
                use_5 = self.is_dental_segmentator_5class or infer_mask_is_5class(mask_cs)
                jaw_id = get_jaw_label_id(is_mand, is_5class=use_5)
                jaw_mask = mask_cs == jaw_id
                if jaw_mask.any():
                    du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
                    dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
                    step = min(du, dv, 0.25)
                    max_half = 0.5 * float(u_coords[-1] - u_coords[0])
                    s_grid = np.arange(-max_half, max_half + step, step, dtype=np.float64)
                    t_id = None
                    if meas is not None and getattr(meas, "tooth_fdi", None):
                        t_lbl = get_tooth_label(str(meas.tooth_fdi))
                        if t_lbl is not None:
                            t_id = int(t_lbl.id)
                    same_t = _same_jaw_tooth_mask(mask_cs, is_mand, use_5, tooth_id=t_id)
                    raw_env = _ridge_envelope_mask(
                        mask_cs, u_coords, v_coords, jaw_id, t_id, is_mand, tooth_mask=same_t
                    )
                    b_comp = jaw_mask & _extract_primary_ridge_component(raw_env, u_coords, v_coords)
                    if b_comp.any():
                        prof = self._sample_band_profile(
                            b_comp.astype(np.float32),
                            u_coords,
                            v_coords,
                            mid_u,
                            mid_v,
                            s_grid,
                            0.0,
                            -1.0 if is_mand else 1.0,
                            c_bu,
                            c_bv,
                            step,
                        )
                        t_prof = (
                            self._sample_band_profile(
                                same_t.astype(np.float32),
                                u_coords,
                                v_coords,
                                mid_u,
                                mid_v,
                                s_grid,
                                0.0,
                                -1.0 if is_mand else 1.0,
                                c_bu,
                                c_bv,
                                step,
                            )
                            if same_t.any()
                            else None
                        )
                        span = self._select_ridge_run_indices(prof, t_prof, s_grid, step)
                        if span is not None:
                            s_l, s_r = self._refine_run_endpoints(prof, s_grid, span[0], span[1], step)
                            cand_l = (mid_u + s_l * c_bu, mid_v + s_l * c_bv)
                            cand_r = (mid_u + s_r * c_bu, mid_v + s_r * c_bv)
                            if (
                                math.hypot(cand_r[0] - cand_l[0], cand_r[1] - cand_l[1]) >= 3.0
                                and cand_l[0] > float(u_coords[0]) + _CS_BORDER_TOL_MM
                                and cand_r[0] < float(u_coords[-1]) - _CS_BORDER_TOL_MM
                            ):
                                new_p0, new_p1 = cand_l, cand_r
            if new_p0 is None or new_p1 is None:
                new_p0 = (mid_u - half_w * c_bu, mid_v - half_w * c_bv)
                new_p1 = (mid_u + half_w * c_bu, mid_v + half_w * c_bv)

            cs.width_endpoints_uv[0.0] = (new_p0, new_p1)
            if getattr(cs, "width_segments", None) is not None:
                cs.width_segments[0.0] = (float(new_p0[0]), float(new_p1[0]))
            P, _T, N, B = cs.frame
            w_sw = _uv_to_world(P, N, B, new_p0[0], new_p0[1])
            w_ew = _uv_to_world(P, N, B, new_p1[0], new_p1[1])
            w_val = round(float(np.linalg.norm(np.asarray(w_ew) - np.asarray(w_sw))), 2)
            records = getattr(cs, "measurements", None) or {}
            w0_rec = records.get("width_0")
            if w0_rec is not None:
                w0_rec.start_uv_mm = new_p0
                w0_rec.end_uv_mm = new_p1
                w0_rec.start_world_mm = w_sw
                w0_rec.end_world_mm = w_ew
                w0_rec.value_mm = w_val
            if meas is not None:
                meas.crest_width_mm = w_val
                if getattr(meas, "bone_width_at_depths_mm", None) is not None:
                    meas.bone_width_at_depths_mm[0.0] = w_val
            cs.ridge_axis_angle_deg = round(new_deg if is_mand else -new_deg, 1)

            # Maintain strict T-junction: re-anchor H start onto smoothed W0 midpoint AND align
            # H's apical ray perpendicular to smoothed W0 so width and height never go skew.
            w_mid = (0.5 * (new_p0[0] + new_p1[0]), 0.5 * (new_p0[1] + new_p1[1]))
            h_rec = records.get("available_bone_height")
            if h_rec is not None and h_rec.is_available and h_rec.end_world_mm is not None:
                cs.crest_uv = w_mid
                cs.crest_v = w_mid[1]
                cs.height_u = w_mid[0]
                h_start_w = _uv_to_world(P, N, B, w_mid[0], w_mid[1])
                h_rec.start_uv_mm = w_mid
                h_rec.start_world_mm = h_start_w
                # Re-trace apical H along smoothed normal when mask_cs exists so H ⟂ W0 strictly.
                new_end_uv = None
                if mask_cs is not None and u_coords is not None and v_coords is not None and np.any(mask_cs):
                    use_5_h = self.is_dental_segmentator_5class or infer_mask_is_5class(mask_cs)
                    jaw_id_h = get_jaw_label_id(is_mand, is_5class=use_5_h)
                    t_id_h = None
                    if meas is not None and getattr(meas, "tooth_fdi", None):
                        t_lbl_h = get_tooth_label(str(meas.tooth_fdi))
                        if t_lbl_h is not None:
                            t_id_h = int(t_lbl_h.id)
                    same_t_h = _same_jaw_tooth_mask(mask_cs, is_mand, use_5_h, tooth_id=t_id_h)
                    raw_env_h = _ridge_envelope_mask(
                        mask_cs, u_coords, v_coords, jaw_id_h, t_id_h, is_mand, tooth_mask=same_t_h
                    )
                    r_mask_h = _extract_primary_ridge_component(raw_env_h, u_coords, v_coords)
                    if r_mask_h.any():
                        d_ap_sm = (c_bv, -c_bu) if is_mand else (-c_bv, c_bu)
                        canal_2d_h = None
                        if is_mand:
                            c_ids = [DENTALSEGMENTATOR_CANAL_ID] if use_5_h else list(CANAL_IDS)
                            if meas is not None and getattr(meas, "tooth_fdi", None):
                                ipsi = get_ipsilateral_canal_ids(meas.tooth_fdi, is_5class=use_5_h)
                                if ipsi:
                                    c_ids = ipsi
                            u_ok_h = (np.abs(u_coords) <= 11.0) | (np.abs(u_coords - w_mid[0]) <= 8.5)
                            v_ok_h = v_coords <= w_mid[1] + 1.0
                            c_loc = np.isin(mask_cs, c_ids) & np.outer(v_ok_h, u_ok_h)
                            if c_loc.any():
                                canal_2d_h = c_loc
                        elif not use_5_h:
                            s_ids = (
                                get_ipsilateral_sinus_ids(meas.tooth_fdi, is_5class=False)
                                if (meas is not None and getattr(meas, "tooth_fdi", None))
                                else None
                            )
                            if s_ids:
                                u_ok_h = (np.abs(u_coords) <= 11.0) | (np.abs(u_coords - w_mid[0]) <= 8.5)
                                v_ok_h = v_coords >= w_mid[1] - 1.0
                                s_loc = np.isin(mask_cs, s_ids) & np.outer(v_ok_h, u_ok_h)
                                if s_loc.any():
                                    canal_2d_h = s_loc
                        sm_h, _, sm_basal, _, _ = self._compute_bone_height_in_cs(
                            u_coords,
                            v_coords,
                            mask_cs,
                            P,
                            N,
                            B,
                            is_mand,
                            jaw_id_h,
                            None,
                            ridge_mask=r_mask_h,
                            crest_uv=w_mid,
                            d_apical=d_ap_sm,
                            local_vital_mask_2d=canal_2d_h,
                            img_cs=getattr(cs, "img_cs", None),
                        )
                        if sm_h is not None and sm_basal is not None:
                            new_end_uv = sm_basal
                if new_end_uv is not None:
                    h_end_w = _uv_to_world(P, N, B, new_end_uv[0], new_end_uv[1])
                    cs.basal_uv = new_end_uv
                    cs.basal_v = new_end_uv[1]
                    h_rec.end_uv_mm = new_end_uv
                    h_rec.end_world_mm = h_end_w
                h_val = round(
                    float(np.linalg.norm(np.asarray(h_rec.end_world_mm) - np.asarray(h_start_w))), 2
                )
                h_rec.value_mm = h_val
                if meas is not None:
                    meas.bone_height_mm = h_val

    def stabilize_entry_points(
        self,
        items: Sequence[Tuple[Optional[ImplantSiteMeasurement], Optional[CrossSectionData]]],
        *,
        max_jump_mm: float = MAX_ENTRY_JUMP_MM,
        max_angle_jump_deg: float = 8.0,
    ) -> None:
        """Interpolate entry (crest) points and smooth W0 angles across similar adjacent slices.

        Each slice is resampled independently, so one noisy cut can move the osteotomy entry
        or tilt W0 between adjacent cuts of the same jaw segment. Crests that differ from
        both neighbours by more than ``max_jump_mm`` are replaced by neighbour midpoints, and
        concordant adjacent W0 angles are clamped to ``<= max_angle_jump_deg`` while keeping a
        strict T-junction between H and W0.
        """
        series = [
            (meas, cs) for meas, cs in items if cs is not None and getattr(cs, "crest_uv", None) is not None
        ]
        if len(series) >= 3 and float(max_jump_mm) > 0.0:
            crests = [np.asarray(cs.crest_uv, dtype=np.float64) for _meas, cs in series]
            adjusted = [crest.copy() for crest in crests]
            for index in range(1, len(crests) - 1):
                neighbour_mid = 0.5 * (crests[index - 1] + crests[index + 1])
                if float(np.linalg.norm(crests[index] - neighbour_mid)) > float(max_jump_mm):
                    adjusted[index] = neighbour_mid
            for (meas, cs), new_crest in zip(series, adjusted):
                self._move_crest(cs, meas, new_crest, is_5class=self.is_dental_segmentator_5class)
        self._smooth_adjacent_w0_angles(series, max_angle_jump_deg=max_angle_jump_deg)

    @staticmethod
    def _move_crest(
        cs: CrossSectionData,
        meas: Optional[ImplantSiteMeasurement],
        crest_uv: np.ndarray,
        *,
        is_5class: Optional[bool] = None,
    ) -> None:
        """Move a cross-section's crest, keep height clear of vital canals, and re-sync widths."""
        old = cs.crest_uv
        if old is None or cs.frame is None:
            return
        new_uv = (float(crest_uv[0]), float(crest_uv[1]))
        if float(math.hypot(new_uv[0] - old[0], new_uv[1] - old[1])) < 1e-6:
            return
        records = getattr(cs, "measurements", None) or {}
        record = records.get("available_bone_height")
        if record is None or not record.is_available or record.end_world_mm is None:
            return

        P, _T, N, B = cs.frame
        start_world = _uv_to_world(P, N, B, new_uv[0], new_uv[1])
        end_uv = tuple(float(v) for v in record.end_uv_mm) if record.end_uv_mm is not None else None
        end_world = record.end_world_mm

        # Never allow a stabilized height segment to cross an IAN canal in the cross-section.
        is_mand = bool(getattr(cs, "is_mandible", True))
        mask_cs = getattr(cs, "mask_cs", None)
        u_coords = getattr(cs, "u_coords", None)
        v_coords = getattr(cs, "v_coords", None)
        use_5class = (
            bool(is_5class)
            if is_5class is not None
            else (infer_mask_is_5class(mask_cs) if (mask_cs is not None and np.any(mask_cs)) else False)
        )
        if (
            is_mand
            and end_uv is not None
            and mask_cs is not None
            and u_coords is not None
            and v_coords is not None
            and np.any(mask_cs)
        ):
            canal_ids = [DENTALSEGMENTATOR_CANAL_ID] if use_5class else list(CANAL_IDS)
            # If the record recorded a specific vital_id, prefer that exact ID.
            src_lbls = tuple(getattr(record, "source_labels", ()) or ())
            if len(src_lbls) >= 2:
                canal_ids = [int(src_lbls[1])]
            elif meas is not None and getattr(meas, "tooth_fdi", None):
                ipsi = get_ipsilateral_canal_ids(meas.tooth_fdi, is_5class=use_5class)
                if ipsi:
                    canal_ids = ipsi
            canal_2d = np.isin(mask_cs, canal_ids)
            if canal_2d.any():
                du_seg = end_uv[0] - new_uv[0]
                dv_seg = end_uv[1] - new_uv[1]
                seg_len = math.hypot(du_seg, dv_seg)
                if seg_len > 1e-6:
                    n_samp = max(4, int(math.ceil(seg_len / 0.25)) + 1)
                    ts = np.linspace(0.0, 1.0, n_samp)
                    su = new_uv[0] + ts * du_seg
                    sv = new_uv[1] + ts * dv_seg
                    u_min, u_max = float(u_coords[0]), float(u_coords[-1])
                    v_min, v_max = float(v_coords[0]), float(v_coords[-1])
                    ci = np.clip(
                        np.round((su - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1)).astype(int),
                        0,
                        len(u_coords) - 1,
                    )
                    ri = np.clip(
                        np.round((sv - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1)).astype(int),
                        0,
                        len(v_coords) - 1,
                    )
                    hits = np.flatnonzero(canal_2d[ri, ci])
                    if hits.size > 0:
                        t_stop = max(0.0, float(ts[int(hits[0])]) - (0.25 / seg_len))
                        end_uv = (new_uv[0] + t_stop * du_seg, new_uv[1] + t_stop * dv_seg)
                        end_world = _uv_to_world(P, N, B, end_uv[0], end_uv[1])
                        record.end_uv_mm = end_uv
                        record.end_world_mm = end_world
                        cs.basal_uv = end_uv
                        cs.basal_v = end_uv[1]
                        if getattr(cs, "vital_boundary_uv", None) is not None:
                            cs.vital_boundary_uv = end_uv

        value = round(float(np.linalg.norm(np.asarray(end_world) - np.asarray(start_world))), 2)
        record.value_mm = value
        record.start_uv_mm = new_uv
        record.start_world_mm = start_world
        # The crest no longer comes from this slice's own label boundary, so the height is an
        # interpolated estimate and must not be reported (or drawn) as label-exact.
        record.provenance = PROVENANCE_INTERPOLATED
        record.quality = QUALITY_ESTIMATED
        cs.crest_uv = new_uv
        cs.crest_v = new_uv[1]
        cs.height_u = new_uv[0]

        # Re-sync width calipers with the stabilized crest when real mask geometry is present.
        had_widths = bool(getattr(cs, "width_endpoints_uv", None) or getattr(cs, "width_segments", None))
        if (
            had_widths
            and mask_cs is not None
            and u_coords is not None
            and v_coords is not None
            and np.any(mask_cs)
        ):
            src_lbls = tuple(getattr(record, "source_labels", ()) or ())
            jaw_id = int(src_lbls[0]) if src_lbls else get_jaw_label_id(is_mand, is_5class=use_5class)
            jaw_mask = mask_cs == jaw_id
            if jaw_mask.any():
                t_id: Optional[int] = None
                if meas is not None and getattr(meas, "tooth_fdi", None):
                    info_t = get_tooth_label(str(meas.tooth_fdi))
                    if info_t is not None:
                        t_id = int(info_t.id)
                same_tooth = _same_jaw_tooth_mask(mask_cs, is_mand, use_5class, tooth_id=t_id)
                raw_env = _ridge_envelope_mask(
                    mask_cs, u_coords, v_coords, jaw_id, t_id, is_mand, tooth_mask=same_tooth
                )
                r_mask = _extract_primary_ridge_component(raw_env, u_coords, v_coords)
                b_comp = jaw_mask & r_mask
                ang = getattr(cs, "ridge_axis_angle_deg", None)
                rad = math.radians(float(ang)) if ang is not None else 0.0
                d_apical = (math.sin(rad), -math.cos(rad)) if is_mand else (math.sin(rad), math.cos(rad))
                d_bl = (math.cos(rad), math.sin(rad) if is_mand else -math.sin(rad))
                depths = [0.0, 3.0, 5.0, 6.0, 9.0]
                w0_move_notes: List[str] = []
                segs_u, segs_uv = MeasurementEngine._compute_bone_widths_with_endpoints_in_cs(
                    u_coords,
                    v_coords,
                    mask_cs,
                    new_uv[1],
                    is_mand,
                    jaw_id,
                    depths,
                    bone_mask=b_comp if b_comp.any() else jaw_mask,
                    crest_uv=new_uv,
                    d_apical=d_apical,
                    d_bl=d_bl,
                    tooth_mask=same_tooth,
                    withheld_notes=w0_move_notes,
                )
                if segs_uv or w0_move_notes:
                    cs.width_segments = segs_u
                    cs.width_endpoints_uv = segs_uv
                    new_widths: Dict[float, Optional[float]] = {}
                    for depth in depths:
                        if depth in segs_uv:
                            pt_l, pt_r = segs_uv[depth]
                            w_sw = _uv_to_world(P, N, B, pt_l[0], pt_l[1])
                            w_ew = _uv_to_world(P, N, B, pt_r[0], pt_r[1])
                            w_val = round(float(np.linalg.norm(np.asarray(w_ew) - np.asarray(w_sw))), 2)
                            new_widths[depth] = w_val
                            records[f"width_{depth:g}"] = MeasurementRecord(
                                measurement_type=f"bone_width_w{depth:g}",
                                value_mm=w_val,
                                start_world_mm=w_sw,
                                end_world_mm=w_ew,
                                start_uv_mm=pt_l,
                                end_uv_mm=pt_r,
                                source_labels=(jaw_id,),
                                provenance=PROVENANCE_INTERPOLATED,
                                quality=QUALITY_ESTIMATED,
                            )
                        else:
                            new_widths[depth] = None
                            records.pop(f"width_{depth:g}", None)
                    if w0_move_notes:
                        cs.withheld_reason = _join_reasons(
                            getattr(cs, "withheld_reason", None), *w0_move_notes
                        )
                    if meas is not None:
                        meas.bone_width_at_depths_mm = new_widths
                        meas.crest_width_mm = new_widths.get(0.0)
                        meas.mid_root_width_mm = (
                            new_widths.get(6.0) if new_widths.get(6.0) is not None else new_widths.get(3.0)
                        )
                        meas.apical_width_mm = new_widths.get(9.0)
                        if w0_move_notes:
                            meas.withheld_reason = _join_reasons(meas.withheld_reason, *w0_move_notes)

        if meas is not None:
            meas.bone_height_mm = value
            meas.provenance = PROVENANCE_INTERPOLATED
            meas.quality_status = QUALITY_ESTIMATED
            if value is not None:
                meas.safety_margin_mm = (
                    round(float(max(0.0, value - meas.suggested_length_mm)), 2)
                    if meas.suggested_length_mm is not None
                    else None
                )
                diameter, length, angulation = suggest_implant_dimensions(
                    meas.crest_width_mm, meas.mid_root_width_mm, value, meas.mesiodistal_space_mm
                )
                meas.suggested_diameter_mm = diameter
                meas.suggested_length_mm = length
                meas.suggested_angulation_deg = angulation

    @staticmethod
    def _compute_cortical_plates_in_cs(
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        mask_cs: np.ndarray,
        crest_v: Optional[float],
        is_mandible: bool,
        jaw_id: int,
        tooth: Optional[ToothInfo] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        if crest_v is None or tooth is None:
            return None, None
        target_v = crest_v - 2.0 if is_mandible else crest_v + 2.0
        row_idx = int(np.argmin(np.abs(v_coords - target_v)))
        row_labels = mask_cs[row_idx, :]
        bone_u = u_coords[row_labels == jaw_id]
        tooth_u = u_coords[row_labels == tooth.label_id]
        if len(bone_u) > 0 and len(tooth_u) > 0:
            buccal_th = float(max(0.0, float(np.max(bone_u)) - float(np.max(tooth_u))))
            lingual_th = float(max(0.0, float(np.min(tooth_u)) - float(np.min(bone_u))))
            return round(buccal_th, 2), round(lingual_th, 2)
        return None, None

    def _compute_mesiodistal_space(self, fdi: str) -> Optional[float]:
        info = get_tooth_label(fdi)
        if not info:
            return None
        quad, t_num = info.quadrant, info.tooth_number
        mesial_fdi = (
            f"{quad}{t_num - 1}"
            if t_num > 1
            else (f"{2 if quad == 1 else 1}1" if quad in (1, 2) else f"{3 if quad == 4 else 4}1")
        )
        distal_fdi = f"{quad}{t_num + 1}" if t_num < 8 else None
        t_mesial = self._get_tooth_info(mesial_fdi)
        t_distal = self._get_tooth_info(distal_fdi) if distal_fdi else None
        if t_mesial and t_distal:
            dist = float(np.linalg.norm(t_mesial.crown_point_mm - t_distal.crown_point_mm))
            return round(max(0.0, dist - 4.0), 2)
        return None

    def _compute_site_bone_density_in_cs(
        self,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        mask_cs: np.ndarray,
        img_cs: np.ndarray,
        crest_v: Optional[float],
        is_mandible: bool,
        bone_height_mm: Optional[float],
        jaw_id: int,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Compute approximate implant-zone bone intensity and concentric shell summary.

        Never describes uncalibrated CBCT gray values as calibrated Hounsfield Units.
        """
        if crest_v is None or bone_height_mm is None or bone_height_mm <= 0:
            return None, {}
        depth_span = min(float(bone_height_mm), 10.0)
        if is_mandible:
            v_mask = (v_coords <= crest_v) & (v_coords >= crest_v - depth_span)
        else:
            v_mask = (v_coords >= crest_v) & (v_coords <= crest_v + depth_span)

        core_u = np.abs(u_coords) <= 2.0
        shell_u = (np.abs(u_coords) > 2.0) & (np.abs(u_coords) <= 3.5)

        core_region = np.outer(v_mask, core_u) & (mask_cs == jaw_id)
        if not np.any(core_region):
            return None, {}

        core_vals = img_cs[core_region]
        mean_val = round(float(np.mean(core_vals)), 1)
        std_val = round(float(np.std(core_vals)), 1)

        shell_region = np.outer(v_mask, shell_u) & (mask_cs == jaw_id)
        shell_mean = round(float(np.mean(img_cs[shell_region])), 1) if np.any(shell_region) else mean_val
        misch_full = self._classify_misch_density(mean_val)
        misch_code = self.extract_misch_code(misch_full)
        summary = {
            "core_mean_gray": mean_val,
            "core_std_gray": std_val,
            "outer_shell_mean_gray": shell_mean,
            "misch_code": misch_code,
            "misch_class": misch_full,
            "is_approximate": True,
            "calibration_warning": MISCH_APPROXIMATE_DISCLAIMER,
        }
        return mean_val, summary

    @staticmethod
    def _classify_misch_density(gray_val: float) -> str:
        """Classify approximate Misch bone quality and explicitly label it uncalibrated."""
        if gray_val > 1250:
            code = "D1"
        elif gray_val >= 850:
            code = "D2"
        elif gray_val >= 350:
            code = "D3"
        elif gray_val >= 150:
            code = "D4"
        else:
            code = "D5"
        return f"{code} ({MISCH_APPROXIMATE_DISCLAIMER})"

    @staticmethod
    def extract_misch_code(misch_str: Optional[str]) -> Optional[str]:
        if not misch_str:
            return None
        return str(misch_str).split()[0].strip()

    def _estimate_missing_tooth_site_with_provenance(self, fdi: str) -> Tuple[np.ndarray, str, bool]:
        """Estimate missing tooth site using segmented neighbors or mesiodistal width table."""
        info = get_tooth_label(fdi)
        arch = self._arch_for_fdi(fdi)
        if arch is None or not info:
            return np.zeros(3, dtype=np.float64), "approximate_width_table", True
        jaw_type = "lower" if is_mandibular_fdi(fdi) else "upper"
        seg_centroids = {
            k: t_info.centroid_mm
            for k, t_info in self.teeth.items()
            if (is_mandibular_fdi(k) == (jaw_type == "lower"))
        }
        site_map = compute_fdi_arc_positions_mm(
            total_arc_length_mm=arch.total_arc_length,
            jaw_type=jaw_type,
            arch=arch,
            segmented_centroids=seg_centroids,
            tooth_bearing_span_mm=self._tooth_bearing_span_mm(arch, jaw_type),
        )
        site = site_map.get(str(fdi))
        if site is not None:
            return site.point_ras, site.source, site.is_approximate
        return arch.evaluate_at(0.5), "approximate_width_table", True

    def _estimate_missing_tooth_site(self, fdi: str) -> np.ndarray:
        pt, _, _ = self._estimate_missing_tooth_site_with_provenance(fdi)
        return pt

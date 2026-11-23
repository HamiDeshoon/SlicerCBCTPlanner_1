"""
Clinical Dental CBCT Measurement Engine — Single Source of Truth.

Performs millimeter-precise calculations in the arch-frame cross-section plane from real
volume & segmentation masks ONLY. Never returns fabricated or hardcoded fallback clinical values,
and never substitutes arbitrary image-intensity thresholds for missing jaw segmentation.
All Misch bone density classifications are explicitly labeled approximate because CBCT gray values
are not calibrated Hounsfield Units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import ensure_optional_package
from .arch_detector import (
    DentalArch,
    compute_fdi_arc_positions_mm,
    project_point_to_polyline_arc_mm,
)
from .labels import (
    DENTALSEGMENTATOR_LOWER_TEETH_ID,
    DENTALSEGMENTATOR_UPPER_TEETH_ID,
    get_fdi_quadrant,
    get_ipsilateral_canal_ids,
    get_ipsilateral_sinus_ids,
    get_jaw_label_id,
    get_tooth_group_label_id,
    get_tooth_label,
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
IMPLANT_MESIODISTAL_MARGIN_MM: float = 3.0   # 1.5 mm mesial + 1.5 mm distal
IMPLANT_APICAL_SAFETY_MARGIN_MM: float = 2.0

logger = logging.getLogger("CBCTPlanner.measurements")

# A measurable alveolar ridge sits inside the cross-section field. A bone region that reaches
# the lateral border, or spans most of the field, means the jaw mask is missing or misaligned
# at this site: measuring it would print invented millimetres on the clinical film.
_CS_BORDER_TOL_MM = 0.75
_MAX_RIDGE_WIDTH_FRACTION = 0.80
_MAX_RIDGE_HEIGHT_FRACTION = 0.90

# The alveolar ridge envelope adds the in-socket root inside this buccolingual window.
ENVELOPE_HALF_WIDTH_MM = 6.0

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
    pt = np.asarray(P, dtype=np.float64) + float(u) * np.asarray(N, dtype=np.float64) + float(v) * np.asarray(B, dtype=np.float64)
    return (float(pt[0]), float(pt[1]), float(pt[2]))


def _ridge_envelope_mask(
    mask_cs: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    jaw_id: int,
    tooth_id: Optional[int],
    is_mandible: bool,
) -> np.ndarray:
    """Jawbone label mask plus the in-socket root inside `|u| <= 6 mm` apical to jaw crest.

    Never falls back to arbitrary image-intensity thresholds when jawbone segmentation is
    missing: returns an empty boolean mask so callers explicitly withhold measurements.
    """
    jaw_mask = mask_cs == jaw_id
    if not jaw_mask.any():
        return np.zeros_like(jaw_mask, dtype=bool)
    envelope = jaw_mask.copy()
    if tooth_id is not None:
        v_jaw = v_coords[np.flatnonzero(jaw_mask.any(axis=1))]
        occlusal_v = float(np.max(v_jaw)) if is_mandible else float(np.min(v_jaw))
        in_socket = (mask_cs == int(tooth_id)) & (
            np.abs(u_coords)[None, :] <= ENVELOPE_HALF_WIDTH_MM
        )
        occlusal_side = (
            v_coords[:, None] <= occlusal_v if is_mandible else v_coords[:, None] >= occlusal_v
        )
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
    if u_bone.min() <= u_lo + _CS_BORDER_TOL_MM or u_bone.max() >= u_hi - _CS_BORDER_TOL_MM:
        width_reason = WITHHELD_WIDTH_BORDER
    elif u_bone.max() - u_bone.min() > _MAX_RIDGE_WIDTH_FRACTION * (u_hi - u_lo):
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
    width_endpoints_uv: Dict[float, Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=dict)
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


@dataclass
class NerveProximityMetric:
    """Distance from a tooth's root apex to critical neurological and anatomical canals."""

    tooth_fdi: str
    structure_id: int
    structure_name: str
    shortest_distance_mm: float
    apex_coord_mm: np.ndarray
    closest_canal_point_mm: np.ndarray
    is_safe: bool


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
        # ROI-cropped distance transforms keyed by structure_id -> (dt_sub, z_slice, y_slice, x_slice)
        self._cropped_dt_cache: Dict[int, Tuple[np.ndarray, slice, slice, slice]] = {}
        self._tooth_span_cache: Dict[Tuple[str, int], Optional[Tuple[float, float]]] = {}
        self._site_cache: Dict[Tuple[str, float], Optional[ImplantSiteMeasurement]] = {}
        self._arc_cache: Dict[Tuple[str, float, Optional[str]], Tuple[Optional[ImplantSiteMeasurement], Optional[CrossSectionData]]] = {}
        self._present_labels_cache: Optional[set] = None
        # Reason recorded by the most recent site/cross-section measurement (None when all metrics measured).
        self.last_withheld_reason: Optional[str] = None

    def invalidate_caches(self, *, mask_changed: bool = True) -> None:
        """Clear cached measurements, spans, and (optionally) mask-derived distance transforms."""
        for attr in ("_site_cache", "_arc_cache", "_tooth_span_cache"):
            c = getattr(self, attr, None)
            if c is not None:
                c.clear()
        if mask_changed:
            dt_c = getattr(self, "_cropped_dt_cache", None)
            if dt_c is not None:
                dt_c.clear()
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

    def get_distance_at_world_point(self, structure_id: int, world_pt_ras: np.ndarray) -> Optional[float]:
        """Evaluate 3D Euclidean distance transform (in mm) to `structure_id` using an ROI-cropped EDT."""
        if structure_id not in self._cropped_dt_cache:
            slc = self.localizer._get_label_slices().get(int(structure_id))
            if slc is None:
                return None
            scipy_nd = ensure_optional_package("scipy.ndimage", "scipy", "3D distance transform")
            sz, sy, sx = self.volume.spacing
            margin_z = int(max(4, round(25.0 / max(1e-3, sz))))
            margin_y = int(max(4, round(25.0 / max(1e-3, sy))))
            margin_x = int(max(4, round(25.0 / max(1e-3, sx))))

            z0 = max(0, int(slc[0].start or 0) - margin_z)
            z1 = min(self.mask.shape[0], int(slc[0].stop or self.mask.shape[0]) + margin_z)
            y0 = max(0, int(slc[1].start or 0) - margin_y)
            y1 = min(self.mask.shape[1], int(slc[1].stop or self.mask.shape[1]) + margin_y)
            x0 = max(0, int(slc[2].start or 0) - margin_x)
            x1 = min(self.mask.shape[2], int(slc[2].stop or self.mask.shape[2]) + margin_x)

            zs, ys, xs = slice(z0, z1), slice(y0, y1), slice(x0, x1)
            sub_bin = self.mask[zs, ys, xs] == int(structure_id)
            if not sub_bin.any():
                return None
            dt_sub = scipy_nd.distance_transform_edt(~sub_bin, sampling=(sz, sy, sx))
            self._cropped_dt_cache[structure_id] = (dt_sub, zs, ys, xs)

        dt_sub, zs, ys, xs = self._cropped_dt_cache[structure_id]
        vox = self.volume.world_to_voxel(world_pt_ras)
        vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        if zs.start <= vz < zs.stop and ys.start <= vy < ys.stop and xs.start <= vx < xs.stop:
            return float(dt_sub[vz - zs.start, vy - ys.start, vx - xs.start])
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
        n_u = int(round(2.0 * half_u_mm / step_mm)) + 1
        n_v = int(round(2.0 * half_v_mm / step_mm)) + 1
        u_coords = np.linspace(-half_u_mm, half_u_mm, n_u)
        v_coords = np.linspace(-half_v_mm, half_v_mm, n_v)

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
        """Return True when `fdi` has no segmented tooth mask (edentulous site)."""
        fdi_str = resolve_fdi_number(fdi)
        if fdi_str is None:
            return False
        if not self.is_dental_segmentator_5class:
            return self._get_tooth_info(fdi_str) is None
        center = self.get_site_center_ras(fdi_str)
        if center is None:
            return False
        is_mand = is_mandibular_fdi(fdi_str)
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

    def expand_missing_edentulous_span(
        self, requested_fdis: List[int]
    ) -> Tuple[List[int], List[int]]:
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
            self.last_withheld_reason = cached.withheld_reason if cached is not None else self.last_withheld_reason
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
    ) -> Tuple[Optional[int], Optional[np.ndarray], Optional[Tuple[float, float]], Optional[List[Tuple[float, float]]], Optional[str]]:
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
            # Restrict to ipsilateral local corridor around the ridge axis (|u - u_ref| <= 8.5 mm)
            # and apical to crest (with 1.0 mm tolerance).
            u_ok = np.abs(u_coords - u_ref) <= 8.5
            v_ok = (v_coords <= v_ref + 1.0) if is_mandible else (v_coords >= v_ref - 1.0)
            local_2d = v_mask & np.outer(v_ok, u_ok)
            if not local_2d.any():
                continue

            # Connected-component cleanup to drop 1-pixel noise or distant unrelated blobs
            try:
                from scipy import ndimage  # type: ignore

                lbl, n_c = ndimage.label(local_2d, structure=np.ones((3, 3), dtype=np.int32))
                if n_c > 1:
                    best_c = 0
                    best_d = float("inf")
                    for cid in range(1, n_c + 1):
                        rs, cs_i = np.where(lbl == cid)
                        if rs.size < 2:
                            continue
                        cu = float(np.mean(u_coords[cs_i]))
                        cv = float(np.mean(v_coords[rs]))
                        d = math.hypot(cu - u_ref, cv - v_ref)
                        if d < best_d:
                            best_d = d
                            best_c = cid
                    if best_c > 0:
                        local_2d = lbl == best_c
            except ImportError:
                pass

            rows, cols = np.where(local_2d)
            if rows.size == 0:
                continue
            centroid_uv = (float(np.mean(u_coords[cols])), float(np.mean(v_coords[rows])))

            # Extract segmentation boundary contour in (u, v)
            contour_uv: List[Tuple[float, float]] = []
            if is_mandible:
                # Boundary pixels of the canal cross-section
                padded = np.pad(local_2d, 1, mode="constant", constant_values=False)
                interior = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
                edge = local_2d & (~interior)
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

        # 2. If not in the 2D cross-section plane, check a local 3D ROI (<= 15 mm around P)
        #    Reject contralateral or distant structures outside 15 mm of the planning site.
        local_radius_mm = 15.0
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
    def _detect_alveolar_crest_and_axis(
        ridge_mask: np.ndarray,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        is_mandible: bool,
    ) -> Tuple[Optional[Tuple[float, float]], Tuple[float, float], Tuple[float, float], Optional[float], List[Tuple[float, float]]]:
        """Detect local alveolar crest `(u_crest, v_crest)` and ridge-aligned unit axes `(d_apical, d_bl)`.

        Algorithm:
          1. Extract the coronal-most continuous bone surface `v_surf(u)` per column, rejecting
             thin vertical spikes (< 0.75 mm vertical support) while preserving valid sloped/curved ridges.
          2. Apply spacing-aware robust 1D median + light Gaussian smoothing to `v_surf(u)`.
          3. Locate the alveolar crest peak `(u_crest, v_crest)` on the smoothed ridge surface
             with subvoxel parabolic interpolation when interior.
          4. Estimate the local ridge long axis from bone midpoints at 1..8 mm apical depths,
             yielding `(d_apical, d_bl)` adapted to tilted/oblique ridges (or upright if |angle| < 2 deg).
        """
        default_apical = (0.0, -1.0) if is_mandible else (0.0, 1.0)
        default_bl = (1.0, 0.0)
        if not ridge_mask.any():
            return None, default_apical, default_bl, None, []

        du = abs(float(u_coords[1] - u_coords[0])) if len(u_coords) > 1 else 0.25
        dv = abs(float(v_coords[1] - v_coords[0])) if len(v_coords) > 1 else 0.25
        min_support_px = max(2, int(round(0.75 / max(1e-3, dv))))

        n_v, n_u = ridge_mask.shape
        surf_cols: List[int] = []
        surf_vs: List[float] = []

        for iu in range(n_u):
            col = ridge_mask[:, iu]
            if not col.any():
                continue
            v_indices = np.flatnonzero(col)
            if is_mandible:
                # Scan from superior (high index) downward for a continuous run of >= min_support_px
                chosen_v: Optional[int] = None
                run = 0
                prev = -1
                for idx in v_indices[::-1]:
                    if prev < 0 or prev - idx == 1:
                        run += 1
                    else:
                        run = 1
                    prev = int(idx)
                    if run >= min_support_px:
                        chosen_v = int(idx) + run - 1
                        break
                if chosen_v is None:
                    chosen_v = int(v_indices[-1])
            else:
                # Scan from inferior (low index) upward for a continuous run of >= min_support_px
                chosen_v = None
                run = 0
                prev = -1
                for idx in v_indices:
                    if prev < 0 or idx - prev == 1:
                        run += 1
                    else:
                        run = 1
                    prev = int(idx)
                    if run >= min_support_px:
                        chosen_v = int(idx) - run + 1
                        break
                if chosen_v is None:
                    chosen_v = int(v_indices[0])
            surf_cols.append(iu)
            surf_vs.append(float(v_coords[chosen_v]))

        if not surf_cols:
            return None, default_apical, default_bl, None, []

        surf_u = np.asarray([float(u_coords[i]) for i in surf_cols], dtype=np.float64)
        surf_v = np.asarray(surf_vs, dtype=np.float64)

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

        # Prefer central planning corridor (|u| <= 2.5 mm) when bone exists there, else |u| <= 6.0 mm
        inner_mask = np.abs(surf_u) <= 1.5
        central_mask = np.abs(surf_u) <= 6.0
        if inner_mask.any():
            search_indices = np.flatnonzero(inner_mask)
        elif central_mask.any():
            search_indices = np.flatnonzero(central_mask)
        else:
            search_indices = np.arange(len(surf_u))
        search_v = smooth_v[search_indices]
        if is_mandible:
            local_best = int(np.argmax(search_v))
        else:
            local_best = int(np.argmin(search_v))
        best_idx = int(search_indices[local_best])

        # For flat-topped ridges (multiple columns tied within 0.15 mm of peak), pick the horizontal
        # center of the plateau so the crest point sits centered on the ridge rather than at one edge!
        peak_val = float(smooth_v[best_idx])
        if is_mandible:
            plateau = search_indices[smooth_v[search_indices] >= peak_val - 0.15]
        else:
            plateau = search_indices[smooth_v[search_indices] <= peak_val + 0.15]

        if len(plateau) >= 2:
            u_crest = float(0.5 * (surf_u[plateau[0]] + surf_u[plateau[-1]]))
            v_crest = float(np.median(smooth_v[plateau]))
        else:
            u_crest = float(surf_u[best_idx])
            v_crest = peak_val
            # Subvoxel parabolic refinement for isolated interior peak
            if 0 < best_idx < len(smooth_v) - 1:
                y0, y1, y2 = float(smooth_v[best_idx - 1]), float(smooth_v[best_idx]), float(smooth_v[best_idx + 1])
                denom = (y0 - 2.0 * y1 + y2)
                if abs(denom) > 1e-6:
                    delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
                    u_crest = float(surf_u[best_idx] + delta * du)
                    v_crest = float(y1 - 0.25 * (y0 - y2) * delta)

        # Estimate local anatomical ridge long axis from horizontal ridge midpoints at depths 1..8 mm
        mid_depths: List[float] = []
        mid_us: List[float] = []
        for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            tv = v_crest - d if is_mandible else v_crest + d
            r_idx = int(np.argmin(np.abs(v_coords - tv)))
            if abs(float(v_coords[r_idx]) - tv) > 0.75:
                continue
            row_hits = np.flatnonzero(ridge_mask[r_idx, :])
            if row_hits.size >= 2:
                mid_depths.append(float(d))
                mid_us.append(0.5 * (float(u_coords[int(row_hits[0])]) + float(u_coords[int(row_hits[-1])])))

        ridge_angle_deg: Optional[float] = None
        d_apical = default_apical
        d_bl = default_bl

        if len(mid_depths) >= 3 and (max(mid_depths) - min(mid_depths)) >= 3.0:
            d_arr = np.asarray(mid_depths, dtype=np.float64)
            u_arr = np.asarray(mid_us, dtype=np.float64)
            # Linear fit: u(depth) = slope * depth + intercept
            d_mean = float(np.mean(d_arr))
            u_mean = float(np.mean(u_arr))
            var_d = float(np.sum((d_arr - d_mean) ** 2))
            if var_d > 1e-6:
                slope = float(np.sum((d_arr - d_mean) * (u_arr - u_mean)) / var_d)
                raw_deg = math.degrees(math.atan(slope))
                clamped_deg = float(np.clip(raw_deg, -35.0, 35.0))
                ridge_angle_deg = round(clamped_deg, 1)
                if abs(clamped_deg) >= 2.0:
                    rad = math.radians(clamped_deg)
                    if is_mandible:
                        d_apical = (math.sin(rad), -math.cos(rad))
                        d_bl = (math.cos(rad), math.sin(rad))
                    else:
                        d_apical = (math.sin(rad), math.cos(rad))
                        d_bl = (math.cos(rad), -math.sin(rad))
                else:
                    ridge_angle_deg = round(raw_deg, 1)

        return (u_crest, v_crest), d_apical, d_bl, ridge_angle_deg, crest_profile_uv

    # ------------------------------------------------------------------
    # Anatomy-Aware Available Bone Height (Part 1.2 & Part 3.3)
    # ------------------------------------------------------------------
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
    ) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]], Optional[Tuple[float, float]], str]:
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
            first_idx = int(bone_indices[0])
            last_idx = int(bone_indices[-1])
            # Use symmetric span between entry and exit of the ridge component along d_apical
            full_h = float(max(0.0, s_vals[last_idx] - s_vals[first_idx]))
            basal_uv = (u0 + full_h * au, v0 + full_h * av)

        # 2. Check ipsilateral vital structure boundary (in 2D cross-section first, then local 3D EDT)
        vital_dist: Optional[float] = None
        vital_boundary_uv: Optional[Tuple[float, float]] = None
        provenance = PROVENANCE_LABEL_DERIVED

        if local_vital_mask_2d is not None and local_vital_mask_2d.any():
            vr, vc = np.where(local_vital_mask_2d)
            vu = u_coords[vc]
            vv = v_coords[vr]
            # Project vital pixels onto the local ridge axis `(d_apical, d_bl)`
            du_v = vu - u0
            dv_v = vv - v0
            along = du_v * au + dv_v * av
            # Perpendicular distance to the measurement ray
            perp = np.abs(du_v * (-av) + dv_v * au)
            in_corridor = (along >= 0.0) & (perp <= 5.5)
            if in_corridor.any():
                cand_along = along[in_corridor]
                min_i = int(np.argmin(cand_along))
                vital_dist = float(max(0.0, cand_along[min_i]))
                # Exact nearest vital boundary point along the ray
                vital_boundary_uv = (u0 + vital_dist * au, v0 + vital_dist * av)
                if vital_dist <= full_h + 1.0:
                    return (
                        round(vital_dist, 2),
                        round(vital_dist, 2),
                        vital_boundary_uv,
                        vital_boundary_uv,
                        PROVENANCE_LABEL_DERIVED,
                    )
        elif vital_id is not None:
            crest_world = np.asarray(_uv_to_world(P, N, B, u0, v0), dtype=np.float64)
            edt_val = self.get_distance_at_world_point(int(vital_id), crest_world)
            if edt_val is not None and edt_val <= 25.0:
                vital_dist = float(edt_val)
                eff_h = min(full_h, vital_dist)
                eff_basal = (u0 + eff_h * au, v0 + eff_h * av)
                return (
                    round(eff_h, 2),
                    round(vital_dist, 2),
                    eff_basal,
                    eff_basal if vital_dist <= full_h else None,
                    PROVENANCE_INTERPOLATED,
                )

        return round(full_h, 2), (round(vital_dist, 2) if vital_dist is not None else None), basal_uv, vital_boundary_uv, provenance

    # ------------------------------------------------------------------
    # Anatomy-Aware Buccal & Lingual Cortical Boundary Detection (Part 3.4)
    # ------------------------------------------------------------------
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
    ) -> Tuple[Dict[float, Tuple[float, float]], Dict[float, Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """Detect buccal and lingual cortical boundaries independently on the primary ridge component.

        Samples along the local buccolingual caliper `d_bl` perpendicular to the ridge axis `d_apical`,
        isolating the single contiguous bone segment around the ridge axis so disconnected islands
        never inflate bone widths. Applies subvoxel 0.5-isosurface refinement at both cortical edges.

        Returns `(width_segments_u, width_endpoints_uv)` where:
          - `width_segments_u[d] = (u_lingual, u_buccal)` for backward compatibility
          - `width_endpoints_uv[d] = ((u_lingual, v_lingual), (u_buccal, v_buccal))` exact 2D endpoints
        """
        segments: Dict[float, Tuple[float, float]] = {}
        endpoints_uv: Dict[float, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        if crest_v is None and crest_uv is None:
            return segments, endpoints_uv

        raw_bone = bone_mask if bone_mask is not None else (mask_cs == jaw_id)
        ridge_mask = _extract_primary_ridge_component(raw_bone, u_coords, v_coords)
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

        try:
            from scipy import ndimage  # type: ignore
            has_scipy = True
        except ImportError:
            has_scipy = False

        ridge_f32 = ridge_mask.astype(np.float32)

        for depth in depths_mm:
            eff_d = max(0.25, float(depth)) if float(depth) == 0.0 else float(depth)
            cx = u0 + eff_d * au
            cy = v0 + eff_d * av
            if cy < v_min - 0.5 or cy > v_max + 0.5:
                continue

            # Sample a thin +/- 0.25 mm band along d_apical for surface robustness
            band_profiles = []
            for d_off in (-step, 0.0, step):
                su = (cx + d_off * au) + s_grid * bu
                sv = (cy + d_off * av) + s_grid * bv
                in_b = (su >= u_min) & (su <= u_max) & (sv >= v_min) & (sv <= v_max)
                prof = np.zeros_like(s_grid, dtype=np.float32)
                if in_b.any():
                    cf = (su[in_b] - u_min) / max(1e-6, u_max - u_min) * (len(u_coords) - 1)
                    rf = (sv[in_b] - v_min) / max(1e-6, v_max - v_min) * (len(v_coords) - 1)
                    if has_scipy:
                        prof[in_b] = ndimage.map_coordinates(
                            ridge_f32, np.vstack([rf, cf]), order=1, mode="constant", cval=0.0
                        )
                    else:
                        ri = np.clip(np.round(rf).astype(int), 0, len(v_coords) - 1)
                        ci = np.clip(np.round(cf).astype(int), 0, len(u_coords) - 1)
                        prof[in_b] = ridge_f32[ri, ci]
                band_profiles.append(prof)

            profile = np.mean(np.vstack(band_profiles), axis=0)
            fg = profile >= 0.5
            if not fg.any():
                continue

            # Find contiguous foreground runs along the caliper line and pick the run
            # belonging to the alveolar ridge (containing or closest to s = 0).
            idx_fg = np.flatnonzero(fg)
            splits = np.where(np.diff(idx_fg) > 1)[0] + 1
            runs = np.split(idx_fg, splits)
            zero_idx = int(np.argmin(np.abs(s_grid)))

            chosen_run = None
            best_dist = float("inf")
            for run in runs:
                if run.size < 2:
                    continue
                if run[0] <= zero_idx <= run[-1]:
                    chosen_run = run
                    break
                d_to_zero = min(abs(float(s_grid[run[0]])), abs(float(s_grid[run[-1]])))
                if d_to_zero < best_dist:
                    best_dist = d_to_zero
                    chosen_run = run

            if chosen_run is None or (best_dist != float("inf") and best_dist > 4.0):
                continue

            i_left = int(chosen_run[0])
            i_right = int(chosen_run[-1])
            s_left = float(s_grid[i_left])
            s_right = float(s_grid[i_right])

            # Subvoxel refinement at oblique/partial-volume cortical boundaries without dilating binary steps
            if i_left > 0:
                p_out, p_in = float(profile[i_left - 1]), float(profile[i_left])
                if p_in >= 0.5 > p_out and (p_out > 0.0 or p_in < 1.0) and (p_in - p_out) > 1e-6:
                    frac = (0.5 - p_out) / (p_in - p_out)
                    s_left = float(s_grid[i_left - 1] + frac * step)
            if i_right + 1 < len(profile):
                p_in, p_out = float(profile[i_right]), float(profile[i_right + 1])
                if p_in >= 0.5 > p_out and (p_out > 0.0 or p_in < 1.0) and (p_in - p_out) > 1e-6:
                    frac = (p_in - 0.5) / (p_in - p_out)
                    s_right = float(s_grid[i_right] + frac * step)

            if s_right <= s_left:
                continue

            pt_l = (cx + s_left * bu, cy + s_left * bv)
            pt_r = (cx + s_right * bu, cy + s_right * bv)

            # Reject if either cortical boundary touches the cross-section lateral border
            if (
                pt_l[0] <= u_min + _CS_BORDER_TOL_MM
                or pt_r[0] >= u_max - _CS_BORDER_TOL_MM
            ):
                continue

            segments[float(depth)] = (float(pt_l[0]), float(pt_r[0]))
            endpoints_uv[float(depth)] = (pt_l, pt_r)

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
            if (is_mandible and v_crest > apex_uv[1] + 0.5) or ((not is_mandible) and v_crest < apex_uv[1] - 0.5):
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
        raw_bone = _ridge_envelope_mask(mask_cs, u_coords, v_coords, jaw_id, tooth_id, is_mandible)
        if not raw_bone.any():
            missing_reason = WITHHELD_NO_MANDIBLE_SEG if is_mandible else WITHHELD_NO_MAXILLA_SEG
            cs.withheld_reason = missing_reason
            cs.provenance = PROVENANCE_UNAVAILABLE
            cs.quality_status = QUALITY_UNAVAILABLE
            self.last_withheld_reason = missing_reason
            return None

        # 2. Extract primary connected jawbone component (reject stray voxels & disconnected islands)
        ridge_mask = _extract_primary_ridge_component(raw_bone, u_coords, v_coords)
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
            ridge_mask, u_coords, v_coords, is_mandible
        )
        cs.crest_uv = crest_uv
        cs.crest_v = crest_uv[1] if crest_uv is not None else None
        cs.height_u = crest_uv[0] if crest_uv is not None else 0.0
        cs.crest_profile_uv = crest_profile_uv if len(crest_profile_uv) >= 2 else None
        cs.ridge_axis_angle_deg = ridge_angle_deg

        # 4. Local ipsilateral vital-structure selection (Part 2.2)
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

        # 5. Available bone height (crest -> ipsilateral vital boundary or basal cortex; NEVER root apex!)
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
            )
        )
        cs.vital_boundary_uv = vital_boundary_uv

        withheld_reason: Optional[str] = None
        if not height_ok:
            bone_height_mm = None
            basal_uv = None
            withheld_reason = height_reason

        cs.basal_uv = basal_uv
        cs.basal_v = basal_uv[1] if basal_uv is not None else None

        records: Dict[str, MeasurementRecord] = {}
        if crest_uv is not None and basal_uv is not None and bone_height_mm is not None:
            h_start_w = _uv_to_world(P, N, B, crest_uv[0], crest_uv[1])
            h_end_w = _uv_to_world(P, N, B, basal_uv[0], basal_uv[1])
            exact_h = round(float(np.linalg.norm(np.asarray(h_end_w) - np.asarray(h_start_w))), 2)
            bone_height_mm = exact_h
            src_labels = (jaw_id, vital_id) if (vital_id is not None and vital_boundary_uv is not None) else (jaw_id,)
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

        # 6. Separate root extent measurement (tooth reference -> actual root apex)
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

        # 7. Anatomy-aware buccal & lingual widths (Part 3.4)
        depths = [0.0, 3.0, 5.0, 6.0, 9.0]
        if width_ok:
            segs_u, segs_uv = self._compute_bone_widths_with_endpoints_in_cs(
                u_coords,
                v_coords,
                mask_cs,
                cs.crest_v,
                is_mandible,
                jaw_id,
                depths,
                bone_mask=ridge_mask,
                crest_uv=crest_uv,
                d_apical=d_apical,
                d_bl=d_bl,
            )
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
        else:
            cs.width_segments = {}
            cs.width_endpoints_uv = {}
            widths = {depth: None for depth in depths}
            withheld_reason = _join_reasons(withheld_reason, width_reason)

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
            imp_tip_uv = (crest_uv[0] + suggested_len * d_apical[0], crest_uv[1] + suggested_len * d_apical[1])
            v_end_uv = vital_boundary_uv if vital_boundary_uv is not None else (
                crest_uv[0] + vital_dist_mm * d_apical[0],
                crest_uv[1] + vital_dist_mm * d_apical[1],
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
                quality=QUALITY_VALID if vital_clearance_mm >= IMPLANT_APICAL_SAFETY_MARGIN_MM else QUALITY_WARNING,
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

        overall_prov = height_prov if bone_height_mm is not None else (
            PROVENANCE_LABEL_DERIVED if crest_w is not None else PROVENANCE_UNAVAILABLE
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
        self, frame: Any, *, half_u_mm: float = 15.0, half_v_mm: float = 25.0, step_mm: float = 0.25
    ) -> CrossSectionData:
        """Resample label mask and image once on a cross-section at an explicit arch frame."""
        P, T, N, B = frame
        n_u = int(round(2.0 * half_u_mm / step_mm)) + 1
        n_v = int(round(2.0 * half_v_mm / step_mm)) + 1
        u_coords = np.linspace(-half_u_mm, half_u_mm, n_u)
        v_coords = np.linspace(-half_v_mm, half_v_mm, n_v)
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
        cs = self._resample_frame_data(
            (frame.point, frame.tangent, frame.normal, frame.binormal)
        )
        is_mandible = jaw_key == "lower"
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

    @staticmethod
    def _suggest_implant(
        crest_w: Optional[float],
        mid_w: Optional[float],
        bone_height: Optional[float],
        mesiodistal: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Delegate to the canonical `suggest_implant_dimensions` single source of truth."""
        return suggest_implant_dimensions(crest_w, mid_w, bone_height, mesiodistal)

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

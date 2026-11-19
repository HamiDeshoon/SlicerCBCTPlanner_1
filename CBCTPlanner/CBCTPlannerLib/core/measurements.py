"""
Clinical Dental CBCT Measurement Engine.
Performs millimeter-precise calculations in the arch-frame cross-section plane (B4) from real
volume & segmentation masks ONLY. Never returns fabricated or hardcoded fallback clinical values;
returns None when a structure or measurement cannot be determined from the segmentation/volume.
All Misch bone density classifications are explicitly labeled approximate because CBCT gray values
are not calibrated Hounsfield Units (B4, F12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import ensure_optional_package
from .arch_detector import (
    DentalArch,
    compute_fdi_arc_positions_mm,
    project_point_to_polyline_arc_mm,
)
from .labels import (
    DENTALSEGMENTATOR_CANAL_ID,
    DENTALSEGMENTATOR_LOWER_TEETH_ID,
    DENTALSEGMENTATOR_UPPER_TEETH_ID,
    LEFT_IAN_ID,
    LEFT_SINUS_ID,
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    RIGHT_SINUS_ID,
    UPPER_JAW_ID,
    get_tooth_label,
)
from .tooth_localizer import ToothInfo, ToothLocalizer
from .volume_loader import CBCTVolume

MISCH_APPROXIMATE_DISCLAIMER = "approximate (CBCT gray values are not calibrated HU)"

logger = logging.getLogger("CBCTPlanner.measurements")

# A measurable alveolar ridge sits inside the cross-section field. A bone region that reaches
# the lateral border, or spans most of the field, means the jaw mask is missing or misaligned
# at this site: measuring it would print invented millimetres on the clinical film.
_CS_BORDER_TOL_MM = 0.75
_MAX_RIDGE_WIDTH_FRACTION = 0.80
_MAX_RIDGE_HEIGHT_FRACTION = 0.90

# RC-01: the alveolar ridge envelope adds the in-socket root inside this buccolingual
# window, and falls back to gray-value bone when the segmentation carries no jawbone.
ENVELOPE_HALF_WIDTH_MM = 6.0
BONE_HU_FALLBACK = 250.0

# Recorded whenever a gate withholds a metric, so a rendered `--` stays distinguishable
# from missing segmentation and from a metric that failed outright.
WITHHELD_NO_SEGMENTATION = "no segmentation at site"
WITHHELD_NO_BONE = "no jaw bone in cross-section"
WITHHELD_WIDTH_BORDER = "widths withheld: jaw mask touches cross-section border"
WITHHELD_WIDTH_SPAN = "widths withheld: jaw mask spans the cross-section field"
WITHHELD_HEIGHT_SPAN = "height withheld: jaw mask exceeds cross-section height"
WITHHELD_NO_CREST = "crest not resolved in cross-section"
WITHHELD_METRIC_FAILED = "metric computation failed"

_WITHHELD_BADGES = (
    ("no segmentation", "no seg"),
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


def _ridge_envelope_mask(
    mask_cs: np.ndarray,
    img_cs: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    jaw_id: int,
    tooth_id: Optional[int],
    is_mandible: bool,
) -> np.ndarray:
    """RC-01: jawbone voxels plus the in-socket root inside `|u| <= 6 mm`.

    Root voxels count only apical to the jawbone's occlusal boundary, so an erupted crown
    can never pose as the alveolar crest. When the cross-section carries no jawbone voxels
    at all (teeth-only segmentation or an unsegmented scan), the envelope falls back to
    voxels at or above `BONE_HU_FALLBACK` so real gray-value bone is measured instead of
    the site being rejected as missing bone.
    """
    jaw_mask = mask_cs == jaw_id
    if not jaw_mask.any():
        return np.asarray(img_cs, dtype=np.float32) >= BONE_HU_FALLBACK
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


def _ridge_region_is_plausible(
    bone: np.ndarray, u_coords: np.ndarray, v_coords: np.ndarray
) -> Tuple[bool, bool]:
    """Return `(width_ok, height_ok)` for a bone region inside the cross-section field.

    `width_ok` is False when the region touches the buccolingual field border or spans
    more than `_MAX_RIDGE_WIDTH_FRACTION` of the field width; `height_ok` is False when
    the region spans more than `_MAX_RIDGE_HEIGHT_FRACTION` of the field height. An
    empty region is implausible in both dimensions.
    """
    width_reason, height_reason = _ridge_region_withheld_reasons(bone, u_coords, v_coords)
    return width_reason is None, height_reason is None


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
    # Why a metric is `None`; empty when every requested metric was measured honestly.
    withheld_reason: Optional[str] = None


@dataclass
class CrossSectionData:
    """One resampled arch cross-section plus its derived geometry."""

    u_coords: np.ndarray
    v_coords: np.ndarray
    mask_cs: np.ndarray
    img_cs: np.ndarray
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    crest_v: Optional[float]
    width_segments: Dict[float, Tuple[float, float]]
    canal_centroid_uv: Optional[Tuple[float, float]]
    basal_v: Optional[float] = None
    height_u: float = 0.0
    is_mandible: bool = True
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
    """Master engine for calculating dental CBCT clinical metrics in the arch-frame cross-section (B4)."""

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
        # O3: Cache ROI-cropped distance transforms keyed by structure_id -> (dt_sub, z_slice, y_slice, x_slice)
        self._cropped_dt_cache: Dict[int, Tuple[np.ndarray, slice, slice, slice]] = {}
        self._tooth_span_cache: Dict[Tuple[str, int], Optional[Tuple[float, float]]] = {}
        # Reason recorded by the most recent site/cross-section measurement (None when all metrics measured).
        self.last_withheld_reason: Optional[str] = None

    def _jaw_id(self, is_mandible: bool) -> int:
        if self.is_dental_segmentator_5class:
            return 2 if is_mandible else 1
        return LOWER_JAW_ID if is_mandible else UPPER_JAW_ID

    def _vital_ids(self, quadrant: int, is_mandible: bool) -> List[int]:
        if self.is_dental_segmentator_5class:
            return [DENTALSEGMENTATOR_CANAL_ID] if is_mandible else []
        if is_mandible:
            primary = LEFT_IAN_ID if quadrant == 3 else RIGHT_IAN_ID
            return [primary, LEFT_IAN_ID, RIGHT_IAN_ID, DENTALSEGMENTATOR_CANAL_ID]
        return [LEFT_SINUS_ID if quadrant == 2 else RIGHT_SINUS_ID]

    def _arch_for_jaw(self, is_mandible: bool) -> Optional[DentalArch]:
        return self.arches.get("lower" if is_mandible else "upper") or self.arch

    def _tooth_bearing_span_mm(
        self, arch: Optional[DentalArch], jaw_type: str
    ) -> Optional[Tuple[float, float]]:
        """Return the arc interval actually occupied by tooth voxels (5-class masks).

        The arch curve is intentionally extended ~12 mm past the last molar for
        panoramic coverage. Without anchors, the width table must not be spread over
        that retromolar extension, or every approximate site shifts posteriorly.
        """
        if arch is None or not self.is_dental_segmentator_5class:
            return None
        key = (jaw_type, id(arch))
        if key in self._tooth_span_cache:
            return self._tooth_span_cache[key]
        label = (
            DENTALSEGMENTATOR_LOWER_TEETH_ID
            if jaw_type.lower().startswith("low")
            else DENTALSEGMENTATOR_UPPER_TEETH_ID
        )
        vox = np.argwhere(self.mask == label)
        if len(vox) < 20:
            self._tooth_span_cache[key] = None
            return None
        # Bound the projection work irrespective of mask resolution.
        step = max(1, len(vox) // 4000)
        world = self.volume.voxel_to_world(vox[::step, ::-1])  # zyx -> xyz
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
        cache = getattr(self, "_tooth_span_cache", None)
        if cache is not None:
            cache.clear()
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
        return self._arch_for_jaw(info.quadrant in (3, 4))

    def _get_tooth_info(self, fdi: str) -> Optional[ToothInfo]:
        fdi_str = str(fdi)
        if fdi_str in self.teeth:
            return self.teeth[fdi_str]
        if not self.is_dental_segmentator_5class:
            extracted = self.localizer.extract_tooth(fdi_str)
            if extracted is not None:
                self.teeth[fdi_str] = extracted
            return extracted
        return None

    def get_distance_at_world_point(self, structure_id: int, world_pt_ras: np.ndarray) -> Optional[float]:
        """O3: Evaluate 3D Euclidean distance transform (in mm) to `structure_id` using an ROI-cropped EDT."""
        if structure_id not in self._cropped_dt_cache:
            binary = self.mask == structure_id
            if not binary.any():
                return None
            scipy_nd = ensure_optional_package("scipy.ndimage", "scipy", "3D distance transform")
            z_idx, y_idx, x_idx = np.where(binary)
            sz, sy, sx = self.volume.spacing
            margin_z = int(max(4, round(30.0 / max(1e-3, sz))))
            margin_y = int(max(4, round(30.0 / max(1e-3, sy))))
            margin_x = int(max(4, round(30.0 / max(1e-3, sx))))

            z0 = max(0, int(z_idx.min()) - margin_z)
            z1 = min(self.mask.shape[0], int(z_idx.max()) + margin_z + 1)
            y0 = max(0, int(y_idx.min()) - margin_y)
            y1 = min(self.mask.shape[1], int(y_idx.max()) + margin_y + 1)
            x0 = max(0, int(x_idx.min()) - margin_x)
            x1 = min(self.mask.shape[2], int(x_idx.max()) + margin_x + 1)

            zs, ys, xs = slice(z0, z1), slice(y0, y1), slice(x0, x1)
            sub_bin = binary[zs, ys, xs]
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
        """Return orthonormal `(P, T, N, B)` frame at `site_center_mm` for cross-section resampling (B4)."""
        arch = self._arch_for_jaw(is_mandible)
        if arch is not None:
            t_param, _ = arch.find_closest_parameter(site_center_mm)
            frame = arch.get_frame(t_param)
            return np.asarray(site_center_mm, dtype=np.float64), frame.tangent, frame.normal, frame.binormal
        # Fallback orthonormal RAS axes if no arch is provided
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
        """B4: Resample label mask and image intensity on a 2D cross-section grid perpendicular to the arch tangent.

        Grid coordinates:
          - `u_coords`: Buccal-Lingual offsets along `N` (`-half_u_mm` to `+half_u_mm`)
          - `v_coords`: Inferior-Superior offsets along occlusal normal `B` (`-half_v_mm` to `+half_v_mm`)
          - `world_grid[iv, iu] = P + u_coords[iu] * N + v_coords[iv] * B`
        """
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
        """Resolve the arch cross-section center for `fdi` plus its provenance.

        Returns `(center_ras, source, is_approximate)` or `None` when neither a segmented
        tooth nor an arch is available. `offset_mm` moves along the arch curve.
        """
        info = get_tooth_label(str(fdi))
        if not info:
            return None
        arch = self._arch_for_fdi(str(fdi))
        tooth = self._get_tooth_info(str(fdi))
        if tooth is not None:
            base_site_mm = np.asarray(tooth.centroid_mm, dtype=np.float64)
            site_source = "segmentation"
            is_approx_site = False
        elif arch is not None:
            base_site_mm, site_source, is_approx_site = self._estimate_missing_tooth_site_with_provenance(
                str(fdi)
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
        """Return the RAS center of the arch cross-section for `fdi` (segmented centroid or estimated site)."""
        resolved = self._site_center_with_provenance(str(fdi), offset_mm=offset_mm)
        return None if resolved is None else resolved[0]

    def measure_implant_site(self, fdi: str, offset_mm: float = 0.0) -> Optional[ImplantSiteMeasurement]:
        """Compute implant planning metrics in the arch-frame cross-section (B4); returns None if unmeasurable."""
        self.last_withheld_reason = None
        info = get_tooth_label(str(fdi))
        if not info:
            return None

        is_mandible = info.quadrant in (3, 4)
        jaw_id = self._jaw_id(is_mandible)
        tooth = self._get_tooth_info(str(fdi))

        resolved = self._site_center_with_provenance(str(fdi), offset_mm=offset_mm)
        if resolved is None:
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
            return self._measure_from_cs(
                str(fdi), info, is_mandible, jaw_id, tooth, cs, site_source, is_approx_site
            )
        except Exception:
            # A failing metric must surface as a recorded reason, not a silently blank tile.
            logger.exception("Implant-site metrics failed for FDI %s", fdi)
            cs.withheld_reason = _join_reasons(cs.withheld_reason, WITHHELD_METRIC_FAILED)
            self.last_withheld_reason = WITHHELD_METRIC_FAILED
            return None

    def _measure_from_cs(
        self,
        fdi: str,
        info: Any,
        is_mandible: bool,
        jaw_id: int,
        tooth: Optional[ToothInfo],
        cs: "CrossSectionData",
        site_source: str,
        is_approx_site: bool,
    ) -> Optional[ImplantSiteMeasurement]:
        """Compute implant metrics from one already-resampled cross-section."""
        u_coords = cs.u_coords
        v_coords = cs.v_coords
        mask_cs = cs.mask_cs
        img_cs = cs.img_cs
        P, _T, _N, B = cs.frame
        cs.is_mandible = is_mandible
        self.last_withheld_reason = None
        tooth_id: Optional[int] = None
        if tooth is not None:
            tooth_id = int(tooth.label_id)
        elif self.is_dental_segmentator_5class:
            tooth_id = (
                DENTALSEGMENTATOR_LOWER_TEETH_ID if is_mandible else DENTALSEGMENTATOR_UPPER_TEETH_ID
            )
        bone_mask = _ridge_envelope_mask(
            mask_cs, img_cs, u_coords, v_coords, jaw_id, tooth_id, is_mandible
        )
        width_reason, height_reason = _ridge_region_withheld_reasons(
            bone_mask, u_coords, v_coords
        )
        width_ok, height_ok = width_reason is None, height_reason is None
        if not width_ok and not height_ok:
            # Not measurable: leave crest/widths unset so the tile renders the ruler only,
            # and record why so a blank H/W is auditable instead of ambiguous.
            cs.withheld_reason = _join_reasons(width_reason, height_reason)
            self.last_withheld_reason = cs.withheld_reason
            logger.info("Cross-section at FDI %s rejected: %s", fdi, cs.withheld_reason)
            return None
        vital_candidates = self._vital_ids(info.quadrant, is_mandible)
        vital_id = next((vid for vid in vital_candidates if (self.mask == vid).any()), None)
        vital_name = "Inferior Alveolar Canal" if is_mandible else "Maxillary Sinus Floor"

        bone_height_mm, vital_dist_mm, crest_v = self._compute_bone_height_in_cs(
            u_coords, v_coords, mask_cs, P, B, is_mandible, jaw_id, vital_id, bone_mask=bone_mask
        )
        cs.crest_v = crest_v
        withheld_reason: Optional[str] = None
        if not height_ok:
            # The crest still anchors the widths; only the implausible span is withheld.
            bone_height_mm = None
            withheld_reason = height_reason
        if crest_v is not None and bone_height_mm is not None:
            cs.basal_v = crest_v - bone_height_mm if is_mandible else crest_v + bone_height_mm

        depths = [0.0, 3.0, 5.0, 6.0, 9.0]
        if width_ok:
            widths = self._compute_bone_widths_in_cs(
                u_coords, v_coords, mask_cs, crest_v, is_mandible, jaw_id, depths, bone_mask=bone_mask
            )
            cs.width_segments = self._compute_bone_widths_with_endpoints_in_cs(
                u_coords, v_coords, mask_cs, crest_v, is_mandible, jaw_id, depths, bone_mask=bone_mask
            )
        else:
            widths = {depth: None for depth in depths}
            withheld_reason = _join_reasons(withheld_reason, width_reason)
        if crest_v is None:
            withheld_reason = _join_reasons(withheld_reason, WITHHELD_NO_CREST)
        cs.withheld_reason = withheld_reason
        self.last_withheld_reason = withheld_reason
        if 0.0 in cs.width_segments:
            u_l, u_r = cs.width_segments[0.0]
            cs.height_u = 0.5 * (u_l + u_r)
        elif 3.0 in cs.width_segments:
            u_l, u_r = cs.width_segments[3.0]
            cs.height_u = 0.5 * (u_l + u_r)

        crest_w = widths.get(0.0)
        mid_w = widths.get(6.0) if widths.get(6.0) is not None else widths.get(3.0)
        apical_w = widths.get(9.0)

        mesiodistal_mm = self._compute_mesiodistal_space(fdi)
        mean_density_hu, nnt_summary = self._compute_site_bone_density_in_cs(
            u_coords, v_coords, mask_cs, img_cs, crest_v, is_mandible, bone_height_mm, jaw_id
        )
        density_class = self._classify_misch_density(mean_density_hu) if mean_density_hu is not None else None

        suggested_diam, suggested_len, suggested_ang = self._suggest_implant(
            crest_w, mid_w, bone_height_mm, mesiodistal_mm
        )
        safety_margin = (
            round(float(max(0.0, bone_height_mm - suggested_len)), 2)
            if (bone_height_mm is not None and suggested_len is not None)
            else None
        )
        buccal_plate_w, lingual_plate_w = self._compute_cortical_plates_in_cs(
            u_coords, v_coords, mask_cs, crest_v, is_mandible, jaw_id, tooth=tooth
        )
        if vital_id is not None:
            canal_v, canal_u = np.where(mask_cs == vital_id)
            if len(canal_u) > 0:
                cs.canal_centroid_uv = (
                    float(np.mean(u_coords[canal_u])),
                    float(np.mean(v_coords[canal_v])),
                )

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
            withheld_reason=withheld_reason,
        )

    def _resample_frame_data(
        self, frame: Any, *, half_u_mm: float = 15.0, half_v_mm: float = 25.0, step_mm: float = 0.25
    ) -> "CrossSectionData":
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
    ) -> Tuple[Optional[ImplantSiteMeasurement], Optional["CrossSectionData"]]:
        """Measure a single cross-section at `arc_mm` along the arch with one resample."""
        jaw_key = "lower" if jaw.lower().startswith("low") else "upper"
        arch = self.arches.get(jaw_key) or self.arch
        if arch is None:
            return None, None
        frame = arch.get_frame_at_mm(float(arc_mm))
        cs = self._resample_frame_data(
            (frame.point, frame.tangent, frame.normal, frame.binormal)
        )
        is_mandible = jaw_key == "lower"

        fdi = str(fdi_hint) if fdi_hint is not None else ""
        if not fdi:
            seg_centroids = {
                key: value.centroid_mm
                for key, value in self.teeth.items()
                if ((int(key) >= 31) == is_mandible)
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
            # A single failing metric must not discard the cross-section: the crest and
            # width geometry resolved before the failure still renders with the tile.
            logger.exception("Cross-section metrics failed at %.2f mm", arc_mm)
            if cs is not None:
                cs.withheld_reason = _join_reasons(cs.withheld_reason, WITHHELD_METRIC_FAILED)
            self.last_withheld_reason = WITHHELD_METRIC_FAILED
            meas = None
        return meas, cs

    def _compute_bone_height_in_cs(
        self,
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        mask_cs: np.ndarray,
        P: np.ndarray,
        B: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        vital_id: Optional[int],
        bone_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """B4: Compute bone height and vital clearance along occlusal normal `B` in the cross-section plane."""
        bone = bone_mask if bone_mask is not None else (mask_cs == jaw_id)
        central_u = np.abs(u_coords) <= 1.5
        jaw_central = bone[:, central_u].any(axis=1)
        if not jaw_central.any():
            wider_u = np.abs(u_coords) <= 4.0
            jaw_central = bone[:, wider_u].any(axis=1)
        if not jaw_central.any():
            # Dentate socket fallback: alveolar cortical plates may sit at |u| up to 7.5 mm around a wide molar root
            socket_u = np.abs(u_coords) <= 7.5
            jaw_central = bone[:, socket_u].any(axis=1)
        if not jaw_central.any():
            return None, None, None

        v_jaw = v_coords[jaw_central]
        if is_mandible:
            crest_v = float(np.max(v_jaw))
            inf_v = float(np.min(v_jaw))
            full_jaw_height = float(max(0.0, crest_v - inf_v))
            if vital_id is not None:
                canal_u = np.abs(u_coords) <= 5.5
                canal_rows = (mask_cs[:, canal_u] == vital_id).any(axis=1) & (v_coords <= crest_v)
                if canal_rows.any():
                    canal_top_v = float(np.max(v_coords[canal_rows]))
                    vital_dist = float(max(0.0, crest_v - canal_top_v))
                    return round(vital_dist, 2), round(vital_dist, 2), crest_v
                crest_world = P + crest_v * B
                edt_val = self.get_distance_at_world_point(vital_id, crest_world)
                if edt_val is not None:
                    return round(min(full_jaw_height, edt_val), 2), round(edt_val, 2), crest_v
            return round(full_jaw_height, 2), None, crest_v
        else:
            crest_v = float(np.min(v_jaw))
            sup_v = float(np.max(v_jaw))
            full_jaw_height = float(max(0.0, sup_v - crest_v))
            if vital_id is not None:
                sinus_u = np.abs(u_coords) <= 5.5
                sinus_rows = (mask_cs[:, sinus_u] == vital_id).any(axis=1) & (v_coords >= crest_v)
                if sinus_rows.any():
                    sinus_floor_v = float(np.min(v_coords[sinus_rows]))
                    vital_dist = float(max(0.0, sinus_floor_v - crest_v))
                    return round(vital_dist, 2), round(vital_dist, 2), crest_v
                crest_world = P + crest_v * B
                edt_val = self.get_distance_at_world_point(vital_id, crest_world)
                if edt_val is not None:
                    return round(min(full_jaw_height, edt_val), 2), round(edt_val, 2), crest_v
            return round(full_jaw_height, 2), None, crest_v

    @staticmethod
    def _compute_bone_widths_in_cs(
        u_coords: np.ndarray,
        v_coords: np.ndarray,
        mask_cs: np.ndarray,
        crest_v: Optional[float],
        is_mandible: bool,
        jaw_id: int,
        depths_mm: List[float],
        bone_mask: Optional[np.ndarray] = None,
    ) -> Dict[float, Optional[float]]:
        """B4: Measure buccolingual bone width along `N` at specified apical depths from `crest_v`."""
        widths: Dict[float, Optional[float]] = {d: None for d in depths_mm}
        if crest_v is None:
            return widths
        bone = bone_mask if bone_mask is not None else (mask_cs == jaw_id)

        for depth in depths_mm:
            # At depth 0 (crest), sample 0.5 mm sub-crestal to avoid surface partial-volume grazing
            eff_d = max(0.25, float(depth)) if depth == 0.0 else float(depth)
            target_v = crest_v - eff_d if is_mandible else crest_v + eff_d
            row_idx = int(np.argmin(np.abs(v_coords - target_v)))
            if abs(float(v_coords[row_idx]) - target_v) > 1.0:
                continue
            # Allow +/- 1 row band for robustness
            r0 = max(0, row_idx - 1)
            r1 = min(len(v_coords), row_idx + 2)
            in_jaw = bone[r0:r1, :].any(axis=0)
            if in_jaw.any():
                u_in = u_coords[in_jaw]
                w = float(np.max(u_in) - np.min(u_in))
                widths[depth] = round(w, 2) if w > 0.0 else None
        return widths

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
    ) -> Dict[float, Tuple[float, float]]:
        """Return `(u_left, u_right)` bone endpoints at each depth (empty when unmeasured)."""
        segments: Dict[float, Tuple[float, float]] = {}
        if crest_v is None:
            return segments
        bone = bone_mask if bone_mask is not None else (mask_cs == jaw_id)
        for depth in depths_mm:
            effective_depth = max(0.25, float(depth)) if depth == 0.0 else float(depth)
            target_v = crest_v - effective_depth if is_mandible else crest_v + effective_depth
            row_idx = int(np.argmin(np.abs(v_coords - target_v)))
            if abs(float(v_coords[row_idx]) - target_v) > 1.0:
                continue
            rows = slice(max(0, row_idx - 1), min(len(v_coords), row_idx + 2))
            in_jaw = bone[rows, :].any(axis=0)
            if in_jaw.any():
                u_in = u_coords[in_jaw]
                left = float(np.min(u_in))
                right = float(np.max(u_in))
                if right > left:
                    segments[float(depth)] = (left, right)
        return segments

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
        """B4 & F12: Compute implant-zone bone intensity and NNT-style concentric shell summary."""
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
            core_region = np.outer(v_mask, core_u)
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
    def _classify_misch_density(hu: float) -> str:
        """B4 & F12: Classify Misch bone quality and explicitly label it approximate."""
        if hu > 1250:
            code = "D1"
        elif hu >= 850:
            code = "D2"
        elif hu >= 350:
            code = "D3"
        elif hu >= 150:
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
        if crest_w is None or bone_height is None:
            return None, None, None
        standard_diameters = [3.0, 3.5, 3.75, 4.0, 4.3, 4.5, 5.0]
        standard_lengths = [6.0, 8.0, 10.0, 11.5, 13.0]

        eff_w = min(crest_w, mid_w) if mid_w is not None else crest_w
        max_allowed_diam = eff_w - 3.0
        if mesiodistal is not None:
            max_allowed_diam = min(max_allowed_diam, mesiodistal - 3.0)

        valid_diams = [d for d in standard_diameters if d <= max_allowed_diam]
        chosen_diam = valid_diams[-1] if valid_diams else None

        safe_height = bone_height - 2.0
        valid_lens = [length for length in standard_lengths if length <= safe_height]
        chosen_len = valid_lens[-1] if valid_lens else None

        if chosen_diam is None or chosen_len is None:
            return None, None, None
        return chosen_diam, chosen_len, 0.0

    def _estimate_missing_tooth_site_with_provenance(self, fdi: str) -> Tuple[np.ndarray, str, bool]:
        """B2: Estimate missing tooth site using segmented neighbors or mesiodistal width table."""
        info = get_tooth_label(fdi)
        arch = self._arch_for_fdi(fdi)
        if arch is None or not info:
            return np.zeros(3, dtype=np.float64), "approximate_width_table", True
        jaw_type = "lower" if info.quadrant in (3, 4) else "upper"
        seg_centroids = {
            k: t_info.centroid_mm
            for k, t_info in self.teeth.items()
            if ((int(k) >= 31) == (jaw_type == "lower"))
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

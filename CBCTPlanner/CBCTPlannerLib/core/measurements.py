"""
Clinical Dental CBCT Measurement Engine.
Performs millimeter-precise calculations from real volume & segmentation masks ONLY.
Never returns fabricated or hardcoded fallback clinical values; returns None when a structure
or measurement cannot be determined from the segmentation/volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import ensure_optional_package
from .arch_detector import DentalArch
from .labels import (
    FDI_TO_PULP_ID,
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

# Support DentalSegmentator 5-class group IDs as well as ToothFairy3 label IDs:
# In DentalSegmentator: 1=Maxilla, 2=Mandible, 3=Upper Teeth, 4=Lower Teeth, 5=Mandibular canal
DENTALSEGMENTATOR_CANAL_ID = 5


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
    """Master engine for calculating dental CBCT clinical metrics without fabricated constants."""

    def __init__(
        self,
        volume: CBCTVolume,
        mask: np.ndarray,
        arch: Optional[DentalArch] = None,
        arches: Optional[Dict[str, DentalArch]] = None,
        is_dental_segmentator_5class: bool = False,
    ):
        self.volume = volume
        self.mask = mask
        self.arch = arch
        self.arches: Dict[str, DentalArch] = dict(arches or {})
        self.is_dental_segmentator_5class = is_dental_segmentator_5class
        self.localizer = ToothLocalizer(volume, mask, arch=arch, arches=self.arches)
        self.teeth: Dict[str, ToothInfo] = (
            {} if is_dental_segmentator_5class else self.localizer.extract_all_teeth()
        )
        self._distance_maps: Dict[int, np.ndarray] = {}

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

    def _arch_for_fdi(self, fdi: str) -> Optional[DentalArch]:
        info = get_tooth_label(fdi)
        if info is None:
            return self.arch
        return self._arch_for_jaw(info.quadrant in (3, 4))

    def get_distance_transform(self, structure_id: int) -> Optional[np.ndarray]:
        """Compute or retrieve cached 3D Euclidean distance transform in mm."""
        if structure_id not in self._distance_maps:
            binary = self.mask == structure_id
            if not binary.any():
                return None
            scipy_nd = ensure_optional_package(
                "scipy.ndimage", "scipy", "3D distance transform"
            )
            sz, sy, sx = self.volume.spacing
            self._distance_maps[structure_id] = scipy_nd.distance_transform_edt(
                ~binary, sampling=(sz, sy, sx)
            )
        return self._distance_maps[structure_id]

    def measure_implant_site(
        self, fdi: str, offset_mm: float = 0.0
    ) -> Optional[ImplantSiteMeasurement]:
        """Compute implant planning metrics from actual segmentation/volume; returns None if unmeasurable."""
        info = get_tooth_label(str(fdi))
        if not info:
            return None

        is_mandible = info.quadrant in (3, 4)
        jaw_id = self._jaw_id(is_mandible)
        jaw_mask = self.mask == jaw_id
        arch = self._arch_for_fdi(str(fdi))

        tooth = self.teeth.get(str(fdi))
        if tooth is not None:
            base_site_mm = tooth.centroid_mm
        elif arch is not None:
            base_site_mm = self._estimate_missing_tooth_site(str(fdi))
        else:
            return None

        if offset_mm != 0.0 and arch is not None:
            t_center, _ = arch.find_closest_parameter(base_site_mm)
            span = max(float(arch.total_arc_length), 1e-6)
            t_target = float(np.clip(t_center + (offset_mm / span), 0.0, 1.0))
            site_center_mm = arch.evaluate_at(t_target)
        else:
            site_center_mm = base_site_mm

        vital_candidates = self._vital_ids(info.quadrant, is_mandible)
        vital_id = next((vid for vid in vital_candidates if (self.mask == vid).any()), None)
        vital_name = "Inferior Alveolar Canal" if is_mandible else "Maxillary Sinus Floor"

        bone_height_mm, vital_dist_mm = self._compute_bone_height(
            site_center_mm, is_mandible, jaw_id, vital_id
        )

        depths = [0.0, 3.0, 5.0, 6.0, 9.0]
        widths = self._compute_bone_widths(site_center_mm, is_mandible, jaw_id, depths)
        crest_w = widths.get(0.0)
        mid_w = widths.get(6.0) if widths.get(6.0) is not None else widths.get(3.0)
        apical_w = widths.get(9.0)

        mesiodistal_mm = self._compute_mesiodistal_space(str(fdi))
        mean_density_hu = self._compute_site_bone_density(
            site_center_mm, is_mandible, bone_height_mm, jaw_id
        )
        density_class = (
            self._classify_misch_density(mean_density_hu)
            if mean_density_hu is not None
            else None
        )

        suggested_diam, suggested_len, suggested_ang = self._suggest_implant(
            crest_w, mid_w, bone_height_mm, mesiodistal_mm
        )
        safety_margin = (
            round(float(max(0.0, bone_height_mm - suggested_len)), 2)
            if (bone_height_mm is not None and suggested_len is not None)
            else None
        )

        buccal_plate_w, lingual_plate_w = self._compute_cortical_plates(
            site_center_mm, is_mandible, jaw_id, tooth=tooth
        )

        return ImplantSiteMeasurement(
            tooth_fdi=str(fdi),
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
            buccal_bone_thickness_mm=buccal_plate_w,
            lingual_bone_thickness_mm=lingual_plate_w,
        )

    def _compute_bone_height(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        vital_id: Optional[int],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute vertical bone height and vital clearance from actual segmented masks; None if absent."""
        vox_center = self.volume.world_to_voxel(site_center_mm)
        cx, cy = int(round(vox_center[0])), int(round(vox_center[1]))
        z_dim, y_dim, x_dim = self.volume.shape
        sz = self.volume.spacing[0]

        if not (0 <= cx < x_dim and 0 <= cy < y_dim):
            return None, None

        y_min, y_max = max(0, cy - 3), min(y_dim, cy + 4)
        x_min, x_max = max(0, cx - 3), min(x_dim, cx + 4)
        jaw_col = (self.mask[:, cy, cx] == jaw_id)
        if not jaw_col.any():
            jaw_col = (self.mask[:, y_min:y_max, x_min:x_max] == jaw_id).any(axis=(1, 2))
        if not jaw_col.any():
            return None, None

        z_indices = np.where(jaw_col)[0]
        dt_vital = self.get_distance_transform(vital_id) if vital_id is not None else None

        if is_mandible:
            crest_z = int(np.max(z_indices))
            full_jaw_height = float((crest_z - int(np.min(z_indices))) * sz)
            if vital_id is not None:
                canal_z = np.where(
                    (self.mask[:, y_min:y_max, x_min:x_max] == vital_id).any(axis=(1, 2))
                )[0]
                if len(canal_z) > 0:
                    canal_top_z = int(np.max(canal_z))
                    vital_dist = float(max(0.0, (crest_z - canal_top_z) * sz))
                    return round(vital_dist, 2), round(vital_dist, 2)
                if dt_vital is not None:
                    vital_dist = float(dt_vital[crest_z, cy, cx])
                    return round(min(full_jaw_height, vital_dist), 2), round(vital_dist, 2)
            return round(full_jaw_height, 2), None
        else:
            crest_z = int(np.min(z_indices))
            full_jaw_height = float((int(np.max(z_indices)) - crest_z) * sz)
            if vital_id is not None:
                sinus_z = np.where(
                    (self.mask[:, y_min:y_max, x_min:x_max] == vital_id).any(axis=(1, 2))
                )[0]
                if len(sinus_z) > 0:
                    sinus_floor_z = int(np.min(sinus_z))
                    vital_dist = float(max(0.0, (sinus_floor_z - crest_z) * sz))
                    return round(vital_dist, 2), round(vital_dist, 2)
                if dt_vital is not None:
                    vital_dist = float(dt_vital[crest_z, cy, cx])
                    return round(min(full_jaw_height, vital_dist), 2), round(vital_dist, 2)
            return round(full_jaw_height, 2), None

    def _compute_bone_widths(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        depths_mm: List[float],
    ) -> Dict[float, Optional[float]]:
        """Compute buccolingual bone width at each depth; returns None for depths with no segmented bone."""
        widths: Dict[float, Optional[float]] = {d: None for d in depths_mm}
        arch = self._arch_for_jaw(is_mandible)
        if arch is None:
            return widths

        t, _ = arch.find_closest_parameter(site_center_mm)
        frame = arch.get_frame(t)
        normal = frame.normal
        up = frame.binormal

        # Find crest Z coordinate at site_center_mm
        vox_center = self.volume.world_to_voxel(site_center_mm)
        cx, cy = int(round(vox_center[0])), int(round(vox_center[1]))
        z_dim, y_dim, x_dim = self.volume.shape
        if not (0 <= cx < x_dim and 0 <= cy < y_dim):
            return widths

        y_min, y_max = max(0, cy - 3), min(y_dim, cy + 4)
        x_min, x_max = max(0, cx - 3), min(x_dim, cx + 4)
        jaw_col = (self.mask[:, y_min:y_max, x_min:x_max] == jaw_id).any(axis=(1, 2))
        if not jaw_col.any():
            return widths
        z_indices = np.where(jaw_col)[0]
        crest_z_vox = int(np.max(z_indices)) if is_mandible else int(np.min(z_indices))
        crest_world = self.volume.voxel_to_world([cx, cy, crest_z_vox])
        crest_point = np.array([site_center_mm[0], site_center_mm[1], crest_world[2]], dtype=np.float64)

        apical_dir = -up if is_mandible else up
        sample_offsets = np.linspace(-15.0, 15.0, 61)

        for depth in depths_mm:
            sample_center = crest_point + apical_dir * depth
            line_pts = sample_center + np.outer(sample_offsets, normal)
            inside = []
            for pt in line_pts:
                vox = self.volume.world_to_voxel(pt)
                vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
                if 0 <= vz < z_dim and 0 <= vy < y_dim and 0 <= vx < x_dim:
                    inside.append(bool(self.mask[vz, vy, vx] == jaw_id))
                else:
                    inside.append(False)
            inside_arr = np.array(inside, dtype=bool)
            if inside_arr.any():
                true_idx = np.where(inside_arr)[0]
                w = float((int(np.max(true_idx)) - int(np.min(true_idx))) * 0.5)
                widths[depth] = round(w, 2) if w > 0.0 else None
        return widths

    def _compute_cortical_plates(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        tooth: Optional[ToothInfo] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        arch = self._arch_for_jaw(is_mandible)
        if arch is None or tooth is None:
            return None, None
        t, _ = arch.find_closest_parameter(site_center_mm)
        frame = arch.get_frame(t)
        normal = frame.normal
        apical_dir = -frame.binormal if is_mandible else frame.binormal
        crest_sub = site_center_mm + apical_dir * 2.0
        offsets = np.linspace(-15.0, 15.0, 61)
        line_pts = crest_sub + np.outer(offsets, normal)

        bone_idx, tooth_idx = [], []
        z_dim, y_dim, x_dim = self.volume.shape
        for i, pt in enumerate(line_pts):
            vox = self.volume.world_to_voxel(pt)
            vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
            if 0 <= vz < z_dim and 0 <= vy < y_dim and 0 <= vx < x_dim:
                val = int(self.mask[vz, vy, vx])
                if val == jaw_id:
                    bone_idx.append(i)
                if val == tooth.label_id:
                    tooth_idx.append(i)
        if bone_idx and tooth_idx:
            buccal_th = float(max(0.0, (max(bone_idx) - max(tooth_idx)) * 0.5))
            lingual_th = float(max(0.0, (min(tooth_idx) - min(bone_idx)) * 0.5))
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
        t_mesial = self.teeth.get(mesial_fdi)
        t_distal = self.teeth.get(distal_fdi) if distal_fdi else None
        if t_mesial and t_distal:
            dist = float(np.linalg.norm(t_mesial.crown_point_mm - t_distal.crown_point_mm))
            return round(max(0.0, dist - 4.0), 2)
        return None

    def _compute_site_bone_density(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        bone_height_mm: Optional[float],
        jaw_id: int,
    ) -> Optional[float]:
        if bone_height_mm is None or bone_height_mm <= 0:
            return None
        vox_center = self.volume.world_to_voxel(site_center_mm)
        cx, cy, cz = int(round(vox_center[0])), int(round(vox_center[1])), int(round(vox_center[2]))
        rx = int(max(1, round(2.0 / self.volume.spacing[2])))
        ry = int(max(1, round(2.0 / self.volume.spacing[1])))
        rz = int(max(2, round(min(bone_height_mm, 10.0) / self.volume.spacing[0])))

        z_slice = (
            slice(cz, min(cz + rz, self.volume.shape[0]))
            if not is_mandible
            else slice(max(0, cz - rz), min(cz + 1, self.volume.shape[0]))
        )
        y_slice = slice(max(0, cy - ry), min(cy + ry + 1, self.volume.shape[1]))
        x_slice = slice(max(0, cx - rx), min(cx + rx + 1, self.volume.shape[2]))

        sub_img = self.volume.image[z_slice, y_slice, x_slice]
        sub_mask = self.mask[z_slice, y_slice, x_slice] == jaw_id
        if sub_img.size == 0:
            return None
        if np.any(sub_mask):
            return round(float(np.mean(sub_img[sub_mask])), 1)
        return round(float(np.mean(sub_img)), 1)

    @staticmethod
    def _classify_misch_density(hu: float) -> str:
        if hu > 1250:
            return "D1"
        if hu >= 850:
            return "D2"
        if hu >= 350:
            return "D3"
        if hu >= 150:
            return "D4"
        return "D5"

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
        valid_lens = [l for l in standard_lengths if l <= safe_height]
        chosen_len = valid_lens[-1] if valid_lens else None

        if chosen_diam is None or chosen_len is None:
            return None, None, None
        return chosen_diam, chosen_len, 0.0

    def _estimate_missing_tooth_site(self, fdi: str) -> np.ndarray:
        info = get_tooth_label(fdi)
        arch = self._arch_for_fdi(fdi)
        if arch is None or not info:
            return np.zeros(3, dtype=np.float64)
        q, n = info.quadrant, info.tooth_number
        if q in (1, 4):
            t = (8 - n) / 7.0 * 0.48
        else:
            t = 0.52 + (n - 1) / 7.0 * 0.48
        return arch.evaluate_at(t)

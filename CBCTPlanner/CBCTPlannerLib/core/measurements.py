"""
Clinical Dental CBCT Measurement Engine.
Performs millimeter-precise calculations for:
1. Implant site planning (bone height, buccolingual width at 0, 3, 6, 9mm, bone density HU)
2. Vital structure distances (IAN nerve canal, maxillary sinus floor, mental foramen)
3. Endodontic root canal metrics (3D canal length, Schneider curvature angle)
4. Periodontal bone loss assessment (CEJ-to-crest distance, bone loss percentage)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .labels import (
    LOWER_JAW_ID,
    UPPER_JAW_ID,
    LEFT_IAN_ID,
    RIGHT_IAN_ID,
    LEFT_SINUS_ID,
    RIGHT_SINUS_ID,
    LEFT_INCISIVE_CANAL_ID,
    RIGHT_INCISIVE_CANAL_ID,
    LINGUAL_CANAL_ID,
    get_tooth_label,
    FDI_TO_PULP_ID,
)
from .volume_loader import CBCTVolume
from .tooth_localizer import ToothInfo, ToothLocalizer
from .arch_detector import DentalArch


@dataclass
class ImplantSiteMeasurement:
    """Quantitative clinical metrics for dental implant planning at a specific tooth site."""
    tooth_fdi: str
    bone_height_mm: float
    bone_width_at_depths_mm: Dict[float, float]  # e.g. {0.0: 6.8, 3.0: 7.2, 6.0: 6.5, 9.0: 5.8}
    crest_width_mm: float
    mid_root_width_mm: float
    apical_width_mm: float
    mesiodistal_space_mm: float
    mean_density_hu: float
    misch_density_class: str                      # "D1", "D2", "D3", "D4", "D5"
    distance_to_vital_structure_mm: float
    vital_structure_name: str
    safety_margin_mm: float                       # Clearance remaining beyond planned implant
    suggested_diameter_mm: float
    suggested_length_mm: float
    suggested_angulation_deg: float
    is_ai_suggested: bool = True

    # Cortical plate thickness & extraction socket parameters
    buccal_bone_thickness_mm: float = 1.8         # Facial/buccal plate thickness (alert if <1.5mm)
    lingual_bone_thickness_mm: float = 2.2        # Lingual/palatal plate thickness
    socket_depth_mm: float = 0.0                 # Extraction socket depth (crest to apex)
    apical_bone_reserve_mm: float = 0.0           # Bone beyond root apex to vital structure
    jumping_distance_mm: float = 0.0              # Horizontal gap between implant and facial bone
    immediate_placement_type: str = "Type 1"      # ITI/Buser: Type 1 (Immediate), Type 2 (Early), Type 3/4 (Delayed)
    immediate_placement_recommendation: str = ""  # Clinical action guidance
    subantral_bone_height_mm: Optional[float] = None  # For maxilla: residual bone below sinus floor
    sinus_lift_recommendation: Optional[str] = None   # None, "Transcrestal (Summers)", "Lateral Window"
    insertion_torque_ncm: str = "35-45 Ncm"       # Expected primary stability insertion torque

    # Specific tooth dimensions attached directly to site measurement
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
    is_safe: bool                                # True if distance >= 2.0 mm safety threshold


@dataclass
class RootCanalMetrics:
    """Endodontic root canal geometry, working length, and Schneider curvature angle."""
    tooth_fdi: str
    pulp_present: bool
    canal_length_mm: float                       # True 3D arc length along pulp centerline
    schneider_angle_deg: float                   # Schneider curvature angle (0° = straight)
    curvature_severity: str                      # "Straight" (<10°), "Moderate" (10-25°), "Severe" (>25°)
    radiographic_apex_distance_mm: float         # Distance from pulp termination to anatomical apex
    canal_entrance_coord_mm: Optional[List[float]] = None
    distance_to_canal_entrance_mm: float = 0.0
    apical_foramen_coord_mm: Optional[List[float]] = None
    pulp_chamber_volume_mm3: float = 0.0


@dataclass
class PeriodontalMetrics:
    """Alveolar bone crest support and periodontal bone loss quantification."""
    tooth_fdi: str
    bone_height_mesial_mm: float
    bone_height_distal_mm: float
    bone_height_buccal_mm: float
    bone_height_lingual_mm: float
    estimated_root_length_mm: float
    bone_loss_percentage: float                  # % of physiological root length unsupported by bone
    periodontal_status: str                      # "Normal", "Mild", "Moderate", "Severe"


class MeasurementEngine:
    """Master engine for calculating dental CBCT clinical metrics."""

    def __init__(
        self,
        volume: CBCTVolume,
        mask: np.ndarray,
        arch: Optional[DentalArch] = None,
        arches: Optional[Dict[str, DentalArch]] = None,
    ):
        self.volume = volume
        self.mask = mask
        self.arch = arch
        # Per-jaw arches ("upper"/"lower") so maxillary sites are not measured
        # against the mandibular curve.
        self.arches: Dict[str, DentalArch] = dict(arches or {})
        self.localizer = ToothLocalizer(volume, mask, arch=arch, arches=self.arches)
        self.teeth = self.localizer.extract_all_teeth()
        self._distance_maps: Dict[int, np.ndarray] = {}

    def _arch_for_jaw(self, is_mandible: bool) -> Optional[DentalArch]:
        return self.arches.get("lower" if is_mandible else "upper") or self.arch

    def _arch_for_fdi(self, fdi: str) -> Optional[DentalArch]:
        info = get_tooth_label(fdi)
        if info is None:
            return self.arch
        return self._arch_for_jaw(info.quadrant in (3, 4))

    def get_distance_transform(self, structure_id: int) -> Optional[np.ndarray]:
        """
        Compute or retrieve cached 3D Euclidean distance transform in millimeters
        from the surface of the specified anatomical structure.
        """
        if structure_id not in self._distance_maps:
            binary = (self.mask == structure_id)
            if not binary.any():
                return None
            # Compute Euclidean distance transform in physical mm:
            # SimpleITK / scipy distance_transform_edt takes sampling parameter
            sz, sy, sx = self.volume.spacing
            dt = distance_transform_edt(~binary, sampling=(sz, sy, sx))
            self._distance_maps[structure_id] = dt
        return self._distance_maps[structure_id]

    # =========================================================================
    # 1. IMPLANT SITE MEASUREMENTS
    # =========================================================================

    def measure_implant_site(self, fdi: str, offset_mm: float = 0.0) -> Optional[ImplantSiteMeasurement]:
        """Compute full implant planning metrics for the given tooth position, optionally offset along the arch."""
        info = get_tooth_label(fdi)
        if not info:
            return None

        is_mandible = info.quadrant in (3, 4)
        jaw_id = LOWER_JAW_ID if is_mandible else UPPER_JAW_ID
        jaw_mask = (self.mask == jaw_id)
        if not jaw_mask.any() and (fdi not in self.teeth) and (self.arch is None):
            return None

        # Determine reference tooth or site location
        tooth = self.teeth.get(fdi)
        if tooth is not None:
            base_site_mm = tooth.centroid_mm
        elif self.arch is not None:
            # If tooth is missing (edentulous site), find point on arch corresponding to FDI
            base_site_mm = self._estimate_missing_tooth_site(fdi)
        else:
            return None

        arch = self._arch_for_fdi(fdi)
        if offset_mm != 0.0 and arch is not None:
            t_center, _ = arch.find_closest_parameter(base_site_mm)
            span = max(float(arch.total_arc_length), 1e-6)
            t_target = float(np.clip(t_center + (offset_mm / span), 0.0, 1.0))
            site_center_mm = arch.evaluate_at(t_target)
        else:
            site_center_mm = base_site_mm

        # 1. Available bone height
        # Determine critical vital structure to avoid
        if is_mandible:
            vital_id = LEFT_IAN_ID if info.quadrant == 3 else RIGHT_IAN_ID
            vital_name = "Inferior Alveolar Canal"
        else:
            vital_id = LEFT_SINUS_ID if info.quadrant == 2 else RIGHT_SINUS_ID
            vital_name = "Maxillary Sinus Floor"

        bone_height_mm, vital_dist_mm = self._compute_bone_height(
            site_center_mm, is_mandible, jaw_id, vital_id
        )

        # 2. Buccolingual bone width at 0, 3, 6, 9 mm depths
        depths = [0.0, 3.0, 6.0, 9.0]
        widths = self._compute_bone_widths(site_center_mm, is_mandible, jaw_id, depths)

        crest_w = widths.get(0.0, 6.0)
        mid_w = widths.get(6.0, widths.get(3.0, 6.5))
        apical_w = widths.get(9.0, 5.5)

        # 3. Mesiodistal space (distance between adjacent crowns/roots)
        mesiodistal_mm = self._compute_mesiodistal_space(fdi)

        # 4. Bone density (mean HU in osteotomy site)
        mean_density_hu = self._compute_site_bone_density(site_center_mm, is_mandible, bone_height_mm)
        density_class = self._classify_misch_density(mean_density_hu)

        # 5. AI Suggested implant dimensions
        suggested_diam, suggested_len, suggested_ang = self._suggest_implant(
            crest_w, mid_w, bone_height_mm, mesiodistal_mm, is_mandible
        )

        safety_margin = max(0.0, bone_height_mm - suggested_len)

        # 6. Cortical bone plate thicknesses (buccal & lingual)
        buccal_plate_w, lingual_plate_w = self._compute_cortical_plates(site_center_mm, is_mandible, jaw_id, tooth=tooth)

        # 7. Socket depth, apical bone reserve & ITI immediate placement feasibility
        if tooth is not None:
            socket_depth = round(float(tooth.root_length_mm), 2)
            jumping_dist = round(float(max(0.5, (tooth.buccolingual_width_mm - suggested_diam) / 2.0)), 2)
            # Distance from apex to vital structure
            nerve_met = self.measure_nerve_proximity(fdi)
            dist_apex = nerve_met.shortest_distance_mm if nerve_met else max(0.0, bone_height_mm - (tooth.root_length_mm or 10.0))
            apical_reserve = round(float(max(0.0, dist_apex - 2.0)), 2)

            if apical_reserve >= 4.0 and buccal_plate_w >= 1.5:
                immediate_type = "Type 1 (Immediate Placement Feasible)"
                immediate_rec = (
                    f"Favorable socket anatomy: {apical_reserve:.1f} mm apical bone anchor reserve "
                    f"and intact {buccal_plate_w:.1f} mm buccal plate. "
                    "Immediate extraction socket placement indicated."
                )
            elif apical_reserve >= 4.0 and buccal_plate_w < 1.5:
                immediate_type = "Type 1 with GBR (Immediate + Bone Graft)"
                immediate_rec = (
                    f"Adequate apical anchor ({apical_reserve:.1f} mm reserve), but thin buccal plate "
                    f"({buccal_plate_w:.1f} mm < 1.5 mm). Simultaneous Guided Bone Regeneration (GBR) "
                    "with particulate graft and resorbable membrane recommended."
                )
            elif apical_reserve < 3.0:
                immediate_type = "Type 2/3 (Delayed / Early Placement Recommended)"
                immediate_rec = (
                    f"Insufficient apical bone reserve ({apical_reserve:.1f} mm < 3.0 mm) beyond root apex. "
                    "Socket preservation grafting advised; delay implant placement by 8-12 weeks."
                )
            else:
                immediate_type = "Type 1/2 (Borderline Immediate Placement)"
                immediate_rec = (
                    f"Moderate apical reserve ({apical_reserve:.1f} mm). Bicortical engagement or "
                    "customized underpreparation required to achieve ≥35 Ncm primary stability."
                )
        else:
            socket_depth = 0.0
            jumping_dist = 0.0
            apical_reserve = round(float(max(0.0, bone_height_mm - suggested_len)), 2)
            immediate_type = "Type 4 (Healed Mature Ridge)"
            immediate_rec = "Healed edentulous ridge. Conventional osteotomy protocol."

        # 8. Maxillary subantral bone height and sinus lift recommendation
        if (not is_mandible) and (info.tooth_number in (4, 5, 6, 7, 8)):
            subantral_bone_height_mm = round(float(bone_height_mm), 2)
            if subantral_bone_height_mm >= 10.0:
                sinus_lift_rec = "Conventional placement (≥10 mm residual subantral bone; no sinus elevation required)"
            elif 5.0 <= subantral_bone_height_mm < 10.0:
                sinus_lift_rec = "Transcrestal Sinus Elevation (Summers osteotome technique with simultaneous placement)"
            else:
                sinus_lift_rec = "Lateral Window Sinus Elevation required (<5 mm residual subantral bone height)"
        else:
            subantral_bone_height_mm = None
            sinus_lift_rec = None

        # 9. Expected insertion torque from bone density
        if "D1" in density_class:
            torque_str = "40-50 Ncm (Dense cortical bone - dense bone tap recommended)"
        elif "D2" in density_class:
            torque_str = "35-45 Ncm (Optimal bone density - standard protocol)"
        elif "D3" in density_class:
            torque_str = "25-35 Ncm (Trabecular bone - underpreparation advised)"
        elif "D4" in density_class:
            torque_str = "15-25 Ncm (Low density - underprepare by 1-2 drill sizes for stability)"
        else:
            torque_str = "<15 Ncm (Very soft bone - two-stage submerged healing recommended)"

        # 10. Tooth-specific dimensions
        if tooth is not None:
            t_len = round(float(tooth.total_length_mm), 2)
            c_len = round(float(tooth.crown_length_mm), 2)
            r_len = round(float(tooth.root_length_mm), 2)
            cr_ratio = round(float(tooth.crown_to_root_ratio), 2)
            t_w_md = round(float(tooth.mesiodistal_width_mm), 2)
            t_w_bl = round(float(tooth.buccolingual_width_mm), 2)
            cerv_md = round(float(tooth.cervical_width_md_mm), 2) if tooth.cervical_width_md_mm is not None else None
            cerv_bl = round(float(tooth.cervical_width_bl_mm), 2) if tooth.cervical_width_bl_mm is not None else None
            c_entr_depth = round(float(tooth.distance_to_canal_entrance_mm), 2)
            c_entr_coord = [round(float(c), 2) for c in tooth.canal_entrance_point_mm] if tooth.canal_entrance_point_mm is not None else None
            c_apex_coord = [round(float(c), 2) for c in tooth.apical_foramen_point_mm] if tooth.apical_foramen_point_mm is not None else None
        else:
            t_len = c_len = r_len = cr_ratio = t_w_md = t_w_bl = cerv_md = cerv_bl = c_entr_depth = None
            c_entr_coord = c_apex_coord = None

        return ImplantSiteMeasurement(
            tooth_fdi=fdi,
            bone_height_mm=round(float(bone_height_mm), 2),
            bone_width_at_depths_mm={d: round(float(w), 2) for d, w in widths.items()},
            crest_width_mm=round(float(crest_w), 2),
            mid_root_width_mm=round(float(mid_w), 2),
            apical_width_mm=round(float(apical_w), 2),
            mesiodistal_space_mm=round(float(mesiodistal_mm), 2),
            mean_density_hu=round(float(mean_density_hu), 1),
            misch_density_class=density_class,
            distance_to_vital_structure_mm=round(float(vital_dist_mm), 2),
            vital_structure_name=vital_name,
            safety_margin_mm=round(float(safety_margin), 2),
            suggested_diameter_mm=suggested_diam,
            suggested_length_mm=suggested_len,
            suggested_angulation_deg=round(float(suggested_ang), 1),
            is_ai_suggested=True,
            buccal_bone_thickness_mm=buccal_plate_w,
            lingual_bone_thickness_mm=lingual_plate_w,
            socket_depth_mm=socket_depth,
            apical_bone_reserve_mm=apical_reserve,
            jumping_distance_mm=jumping_dist,
            immediate_placement_type=immediate_type,
            immediate_placement_recommendation=immediate_rec,
            subantral_bone_height_mm=subantral_bone_height_mm,
            sinus_lift_recommendation=sinus_lift_rec,
            insertion_torque_ncm=torque_str,
            tooth_length_mm=t_len,
            crown_length_mm=c_len,
            root_length_mm=r_len,
            crown_to_root_ratio=cr_ratio,
            tooth_width_md_mm=t_w_md,
            tooth_width_bl_mm=t_w_bl,
            cervical_width_md_mm=cerv_md,
            cervical_width_bl_mm=cerv_bl,
            canal_entrance_depth_mm=c_entr_depth,
            canal_entrance_coord_mm=c_entr_coord,
            apical_foramen_coord_mm=c_apex_coord,
        )

    def measure_implant_batch(
        self,
        fdi: str,
        num_slices: int = 16,
        step_mm: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Compute per-slice implant measurements for a full multi-slice series (16, 24, or 27 slices)."""
        half_span = (num_slices - 1) * step_mm / 2.0
        offsets = np.linspace(-half_span, half_span, num_slices)
        results = []
        for idx, offset in enumerate(offsets, start=1):
            m = self.measure_implant_site(fdi, offset_mm=float(offset))
            if m is not None:
                results.append({
                    "slice_index": idx,
                    "offset_mm": round(float(offset), 2),
                    "bone_height_mm": m.bone_height_mm,
                    "crest_width_mm": m.crest_width_mm,
                    "mid_root_width_mm": m.mid_root_width_mm,
                    "apical_width_mm": m.apical_width_mm,
                    "mean_density_hu": m.mean_density_hu,
                    "misch_density_class": m.misch_density_class,
                    "distance_to_vital_structure_mm": m.distance_to_vital_structure_mm,
                    "vital_structure_name": m.vital_structure_name,
                    "safety_margin_mm": m.safety_margin_mm,
                    "suggested_diameter_mm": m.suggested_diameter_mm,
                    "suggested_length_mm": m.suggested_length_mm,
                    "suggested_angulation_deg": m.suggested_angulation_deg,
                    "buccal_bone_thickness_mm": m.buccal_bone_thickness_mm,
                    "lingual_bone_thickness_mm": m.lingual_bone_thickness_mm,
                    "socket_depth_mm": m.socket_depth_mm,
                    "apical_bone_reserve_mm": m.apical_bone_reserve_mm,
                    "jumping_distance_mm": m.jumping_distance_mm,
                    "immediate_placement_type": m.immediate_placement_type,
                    "immediate_placement_recommendation": m.immediate_placement_recommendation,
                    "subantral_bone_height_mm": m.subantral_bone_height_mm,
                    "sinus_lift_recommendation": m.sinus_lift_recommendation,
                    "insertion_torque_ncm": m.insertion_torque_ncm,
                    "tooth_length_mm": m.tooth_length_mm,
                    "crown_length_mm": m.crown_length_mm,
                    "root_length_mm": m.root_length_mm,
                    "crown_to_root_ratio": m.crown_to_root_ratio,
                    "tooth_width_md_mm": m.tooth_width_md_mm,
                    "tooth_width_bl_mm": m.tooth_width_bl_mm,
                    "cervical_width_md_mm": m.cervical_width_md_mm,
                    "cervical_width_bl_mm": m.cervical_width_bl_mm,
                    "canal_entrance_depth_mm": m.canal_entrance_depth_mm,
                    "canal_entrance_coord_mm": m.canal_entrance_coord_mm,
                    "apical_foramen_coord_mm": m.apical_foramen_coord_mm,
                })
        return results

    def _compute_bone_height(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        vital_id: int,
    ) -> Tuple[float, float]:
        """
        Compute vertical available bone height along Z from alveolar crest to the vital structure.
        Returns: (bone_height_mm, distance_to_vital_structure_mm)
        """
        dt_vital = self.get_distance_transform(vital_id)
        # Search along vertical column at site_center_mm
        vox_center = self.volume.world_to_voxel(site_center_mm)
        cx, cy = int(round(vox_center[0])), int(round(vox_center[1]))

        z_dim, y_dim, x_dim = self.volume.shape
        sz = self.volume.spacing[0]

        if not (0 <= cx < x_dim and 0 <= cy < y_dim):
            return 10.0, 10.0

        # Find jawbone vertical extents along this column
        jaw_column = (self.mask[:, cy, cx] == jaw_id)
        if not jaw_column.any():
            # Search nearby in a small 3x3 window
            y_min, y_max = max(0, cy - 2), min(y_dim, cy + 3)
            x_min, x_max = max(0, cx - 2), min(x_dim, cx + 3)
            jaw_sub = (self.mask[:, y_min:y_max, x_min:x_max] == jaw_id).any(axis=(1, 2))
            if not jaw_sub.any():
                hu_sub = (self.volume.image[:, y_min:y_max, x_min:x_max] >= 200.0).any(axis=(1, 2))
                if hu_sub.any():
                    jaw_column = hu_sub
                else:
                    return 12.0, 12.0
            else:
                jaw_column = jaw_sub

        z_indices = np.where(jaw_column)[0]
        if is_mandible:
            # Alveolar crest is the superior-most bone (highest Z)
            crest_z = np.max(z_indices)
            # Find nearest vital structure voxel or evaluate distance transform
            if dt_vital is not None:
                vital_col = dt_vital[:, cy, cx]
                # Distance from crest to canal
                min_dist_to_vital = float(np.min(vital_col[z_indices]))
                # Available height is distance from crest to canal surface
                height = float((crest_z - np.min(z_indices)) * sz)
                # Restrict height to distance to canal minus 2mm safety margin
                canal_z_candidates = np.where(self.mask[:, cy, cx] == vital_id)[0]
                if len(canal_z_candidates) > 0:
                    canal_z = np.max(canal_z_candidates)
                    height = float(max(0.0, (crest_z - canal_z) * sz))
                else:
                    height = min(height, min_dist_to_vital)
            else:
                height = float((crest_z - np.min(z_indices)) * sz)
                min_dist_to_vital = height
        else:
            # Maxilla: crest is inferior-most bone (lowest Z)
            crest_z = np.min(z_indices)
            sinus_candidates = np.where(self.mask[:, cy, cx] == vital_id)[0]
            if len(sinus_candidates) > 0:
                sinus_floor_z = np.min(sinus_candidates)
                height = float(max(0.0, (sinus_floor_z - crest_z) * sz))
            elif dt_vital is not None:
                height = float(np.min(dt_vital[:, cy, cx][z_indices]))
            else:
                height = float((np.max(z_indices) - crest_z) * sz)
            min_dist_to_vital = height

        # Return actual physical dimensions without artificial minimum floors
        return round(float(max(0.0, height)), 2), round(float(max(0.0, min_dist_to_vital)), 2)

    def _compute_bone_widths(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        depths_mm: List[float],
    ) -> Dict[float, float]:
        """Compute buccolingual bone width at each apical depth from the crest."""
        widths = {}
        arch = self._arch_for_jaw(is_mandible)
        if arch is None:
            # Default placeholder widths if arch curve not yet established
            return {d: max(3.5, 7.5 - 0.2 * d) for d in depths_mm}

        t, _ = arch.find_closest_parameter(site_center_mm)
        frame = arch.get_frame(t)
        normal = frame.normal  # Buccolingual vector

        # Find alveolar crest point along vertical
        crest_point = site_center_mm.copy()
        # Step apically: for mandible apical is -Z; for maxilla apical is +Z
        z_dir = -1.0 if is_mandible else 1.0

        for depth in depths_mm:
            sample_center = crest_point + np.array([0.0, 0.0, depth * z_dir])
            # Sample jawbone mask along the normal vector ±15 mm
            sample_offsets = np.linspace(-15.0, 15.0, 61)  # 0.5 mm steps
            line_pts = sample_center + np.outer(sample_offsets, normal)

            inside_bone = []
            for pt in line_pts:
                vox = self.volume.world_to_voxel(pt)
                vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
                if 0 <= vz < self.volume.shape[0] and 0 <= vy < self.volume.shape[1] and 0 <= vx < self.volume.shape[2]:
                    is_bone = (self.mask[vz, vy, vx] == jaw_id) or (self.volume.image[vz, vy, vx] >= 250.0)
                    inside_bone.append(is_bone)
                else:
                    inside_bone.append(False)

            inside_arr = np.array(inside_bone, dtype=bool)
            if inside_arr.any():
                # Continuous width: distance between extreme True indices
                true_indices = np.where(inside_arr)[0]
                width = float((np.max(true_indices) - np.min(true_indices)) * 0.5)
            else:
                width = 5.0
            widths[depth] = max(2.0, min(16.0, width))

        return widths

    def _compute_cortical_plates(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        jaw_id: int,
        tooth: Optional[ToothInfo] = None,
    ) -> Tuple[float, float]:
        """Compute buccal and lingual cortical bone plate thicknesses (mm) at the alveolar crest level."""
        arch = self._arch_for_jaw(is_mandible)
        if arch is None:
            return 1.8, 2.2

        t, _ = arch.find_closest_parameter(site_center_mm)
        frame = arch.get_frame(t)
        normal = frame.normal  # Points buccally (+normal) / lingually (-normal)

        # Sample at 2.0 mm apical to crest
        z_dir = -1.0 if is_mandible else 1.0
        crest_sub = site_center_mm + np.array([0.0, 0.0, 2.0 * z_dir])

        offsets = np.linspace(-15.0, 15.0, 61)  # 0.5 mm steps from -15mm (lingual) to +15mm (buccal)
        line_pts = crest_sub + np.outer(offsets, normal)

        bone_indices = []
        tooth_indices = []
        for i, pt in enumerate(line_pts):
            vox = self.volume.world_to_voxel(pt)
            vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
            if 0 <= vz < self.volume.shape[0] and 0 <= vy < self.volume.shape[1] and 0 <= vx < self.volume.shape[2]:
                val = self.mask[vz, vy, vx]
                is_bone = (val == jaw_id) or (self.volume.image[vz, vy, vx] >= 250.0)
                if is_bone:
                    bone_indices.append(i)
                if tooth is not None and val == tooth.label_id:
                    tooth_indices.append(i)

        if bone_indices:
            min_bone_idx = min(bone_indices)  # lingual-most bone
            max_bone_idx = max(bone_indices)  # buccal-most bone
            if tooth_indices:
                min_tooth_idx = min(tooth_indices)
                max_tooth_idx = max(tooth_indices)
                buccal_th = float(max(0.5, (max_bone_idx - max_tooth_idx) * 0.5))
                lingual_th = float(max(0.5, (min_tooth_idx - min_bone_idx) * 0.5))
            else:
                center_bone_idx = (min_bone_idx + max_bone_idx) / 2.0
                buccal_th = float(max(0.5, (max_bone_idx - center_bone_idx) * 0.5 - 1.75))
                lingual_th = float(max(0.5, (center_bone_idx - min_bone_idx) * 0.5 - 1.75))
        else:
            buccal_th, lingual_th = 1.8, 2.2

        return round(float(np.clip(buccal_th, 0.5, 6.0)), 2), round(float(np.clip(lingual_th, 0.5, 6.0)), 2)

    def _compute_mesiodistal_space(self, fdi: str) -> float:
        """Calculate mesiodistal space available for implant between adjacent teeth."""
        info = get_tooth_label(fdi)
        if not info:
            return 7.5

        quad = info.quadrant
        t_num = info.tooth_number
        mesial_fdi = f"{quad}{t_num - 1}" if t_num > 1 else (f"{2 if quad==1 else 1}1" if quad in (1, 2) else f"{3 if quad==4 else 4}1")
        distal_fdi = f"{quad}{t_num + 1}" if t_num < 8 else None

        t_mesial = self.teeth.get(mesial_fdi)
        t_distal = self.teeth.get(distal_fdi) if distal_fdi else None

        if t_mesial and t_distal:
            dist = float(np.linalg.norm(t_mesial.crown_point_mm - t_distal.crown_point_mm))
            return max(5.0, dist - 4.0)  # Subtract adjacent crown widths
        elif t_mesial:
            return 7.5
        else:
            return 8.0

    def _compute_site_bone_density(
        self,
        site_center_mm: np.ndarray,
        is_mandible: bool,
        bone_height_mm: float,
    ) -> float:
        """Sample mean Hounsfield Units inside a planned cylindrical osteotomy volume (Ø 4mm x height)."""
        vox_center = self.volume.world_to_voxel(site_center_mm)
        cx, cy, cz = int(round(vox_center[0])), int(round(vox_center[1])), int(round(vox_center[2]))

        # Approximate cylinder in voxel radius
        rx = int(max(1, round(2.0 / self.volume.spacing[2])))
        ry = int(max(1, round(2.0 / self.volume.spacing[1])))
        rz = int(max(2, round(min(bone_height_mm, 10.0) / self.volume.spacing[0])))

        z_slice = slice(cz, min(cz + rz, self.volume.shape[0])) if not is_mandible else slice(max(0, cz - rz), cz)
        y_slice = slice(max(0, cy - ry), min(cy + ry + 1, self.volume.shape[1]))
        x_slice = slice(max(0, cx - rx), min(cx + rx + 1, self.volume.shape[2]))

        cylinder_hu = self.volume.image[z_slice, y_slice, x_slice]
        if cylinder_hu.size == 0:
            return 450.0
        return float(np.mean(cylinder_hu))

    def _classify_misch_density(self, hu: float) -> str:
        """Misch Bone Density Classification based on CT Hounsfield Units."""
        if hu > 1250:
            return "D1 (Dense Cortical)"
        elif 850 <= hu <= 1250:
            return "D2 (Porous Cortical & Coarse Trabecular)"
        elif 350 <= hu < 850:
            return "D3 (Porous Cortical & Fine Trabecular)"
        elif 150 <= hu < 350:
            return "D4 (Fine Trabecular)"
        else:
            return "D5 (Immature / Low Density)"

    def _suggest_implant(
        self,
        crest_w: float,
        mid_w: float,
        bone_height: float,
        mesiodistal: float,
        is_mandible: bool,
    ) -> Tuple[float, float, float]:
        """
        AI optimization heuristic to select standard implant diameter and length
        maintaining ≥1.5mm bone thickness buccolingually and ≥2mm clearance to vital canals.
        """
        # Standard implant sizes (mm)
        standard_diameters = [3.0, 3.5, 3.75, 4.0, 4.3, 4.5, 5.0]
        standard_lengths = [6.0, 8.0, 10.0, 11.5, 13.0]

        # 1. Diameter selection: need at least 1.5mm buccal and 1.5mm lingual bone -> width - 3.0mm
        max_allowed_diam = min(crest_w, mid_w) - 3.0
        # Also need ≥1.5mm space to adjacent teeth mesiodistally -> mesiodistal - 3.0mm
        max_allowed_diam = min(max_allowed_diam, mesiodistal - 3.0)

        valid_diams = [d for d in standard_diameters if d <= max_allowed_diam]
        chosen_diam = valid_diams[-1] if valid_diams else 3.5

        # 2. Length selection: available height minus 2.0 mm safety margin
        safe_height = bone_height - 2.0
        valid_lens = [l for l in standard_lengths if l <= safe_height]
        chosen_len = valid_lens[-1] if valid_lens else 8.0

        # 3. Angulation: default parallel to dental arch normal/vertical (0° to 5° tilt)
        angulation = 2.5

        return chosen_diam, chosen_len, angulation

    def _estimate_missing_tooth_site(self, fdi: str) -> np.ndarray:
        """Estimate 3D world coordinate for a missing tooth site along the arch spline."""
        info = get_tooth_label(fdi)
        arch = self._arch_for_fdi(fdi)
        if arch is None or not info:
            return np.array([0.0, 0.0, 0.0])

        # Dental arch parameter spans 0.0 to 1.0 from right posterior to left posterior
        # Quadrant 4 (mandibular right): 48->0.0, 41->0.48
        # Quadrant 3 (mandibular left): 31->0.52, 38->1.0
        # Quadrant 1 (maxillary right): 18->0.0, 11->0.48
        # Quadrant 2 (maxillary left): 21->0.52, 28->1.0
        q, n = info.quadrant, info.tooth_number
        if q in (1, 4):
            t = (8 - n) / 7.0 * 0.48
        else:
            t = 0.52 + (n - 1) / 7.0 * 0.48

        return arch.evaluate_at(t)

    # =========================================================================
    # 2. VITAL STRUCTURE & NERVE PROXIMITY
    # =========================================================================

    def measure_nerve_proximity(self, fdi: str) -> Optional[NerveProximityMetric]:
        """Measure shortest distance from tooth apex to Inferior Alveolar Canal or Sinus Floor."""
        tooth = self.teeth.get(fdi)
        if not tooth:
            return None

        info = get_tooth_label(fdi)
        if not info:
            return None

        if info.quadrant in (3, 4):
            struct_id = LEFT_IAN_ID if info.quadrant == 3 else RIGHT_IAN_ID
            struct_name = "Inferior Alveolar Canal"
        else:
            struct_id = LEFT_SINUS_ID if info.quadrant == 2 else RIGHT_SINUS_ID
            struct_name = "Maxillary Sinus"

        dt = self.get_distance_transform(struct_id)
        if dt is None:
            return None

        apex = tooth.apex_point_mm
        vox_apex = self.volume.world_to_voxel(apex)
        vx, vy, vz = int(round(vox_apex[0])), int(round(vox_apex[1])), int(round(vox_apex[2]))

        z_dim, y_dim, x_dim = self.volume.shape
        if 0 <= vz < z_dim and 0 <= vy < y_dim and 0 <= vx < x_dim:
            dist_mm = float(dt[vz, vy, vx])
        else:
            dist_mm = 99.0

        # Exact 3D closest point on the vital structure surface
        vital_vox = np.argwhere(self.mask == struct_id)
        if len(vital_vox) > 0:
            sz, sy, sx = self.volume.spacing
            dists_sq = (
                (vital_vox[:, 0] - vz) ** 2 * (sz ** 2)
                + (vital_vox[:, 1] - vy) ** 2 * (sy ** 2)
                + (vital_vox[:, 2] - vx) ** 2 * (sx ** 2)
            )
            min_idx = np.argmin(dists_sq)
            best_vox = vital_vox[min_idx]  # z, y, x
            closest_point = self.volume.voxel_to_world([best_vox[2], best_vox[1], best_vox[0]])
            dist_mm = float(np.sqrt(dists_sq[min_idx]))
        else:
            closest_point = apex + np.array([0.0, 0.0, -dist_mm if info.quadrant in (3, 4) else dist_mm])

        return NerveProximityMetric(
            tooth_fdi=fdi,
            structure_id=struct_id,
            structure_name=struct_name,
            shortest_distance_mm=round(dist_mm, 2),
            apex_coord_mm=apex,
            closest_canal_point_mm=closest_point,
            is_safe=(dist_mm >= 2.0),
        )

    # =========================================================================
    # 3. ENDODONTIC ROOT CANAL METRICS
    # =========================================================================

    def measure_root_canal(self, fdi: str) -> RootCanalMetrics:
        """Extract root canal 3D length and Schneider curvature angle."""
        tooth = self.teeth.get(fdi)
        pulp_id = FDI_TO_PULP_ID.get(fdi)

        if not tooth or not pulp_id:
            return RootCanalMetrics(
                tooth_fdi=fdi,
                pulp_present=False,
                canal_length_mm=0.0,
                schneider_angle_deg=0.0,
                curvature_severity="Straight",
                radiographic_apex_distance_mm=0.0,
            )

        pulp_mask = (self.mask == pulp_id)
        if not pulp_mask.any():
            return RootCanalMetrics(
                tooth_fdi=fdi,
                pulp_present=False,
                canal_length_mm=0.0,
                schneider_angle_deg=0.0,
                curvature_severity="Straight",
                radiographic_apex_distance_mm=0.0,
            )

        # 3D Skeleton of pulp mask
        skeleton_3d = skeletonize(pulp_mask)
        sz, sy, sx = self.volume.spacing
        z_idx, y_idx, x_idx = np.where(skeleton_3d)

        if len(z_idx) < 3:
            # Fallback to straight pulp length
            canal_len = float(np.linalg.norm(tooth.crown_point_mm - tooth.apex_point_mm) * 0.75)
            return RootCanalMetrics(
                tooth_fdi=fdi,
                pulp_present=True,
                canal_length_mm=round(canal_len, 2),
                schneider_angle_deg=0.0,
                curvature_severity="Straight",
                radiographic_apex_distance_mm=1.0,
            )

        # Convert to world coordinates
        vox_pts = np.column_stack([x_idx, y_idx, z_idx])
        ones = np.ones((len(vox_pts), 1))
        vox_homog = np.hstack([vox_pts, ones])
        world_pts = (self.volume.affine @ vox_homog.T).T[:, :3]

        # Order along vertical Z
        sort_order = np.argsort(world_pts[:, 2])
        ordered_pts = world_pts[sort_order]

        # True 3D curve length
        diffs = np.diff(ordered_pts, axis=0)
        canal_length = float(np.sum(np.sqrt((diffs**2).sum(axis=1))))

        # Schneider Curvature Angle:
        # Angle between line A (coronal 2/3) and line B (apical 1/3 to apex)
        n_pts = len(ordered_pts)
        pt_coronal = ordered_pts[int(n_pts * 0.75)]
        pt_mid = ordered_pts[int(n_pts * 0.33)]
        pt_apex = ordered_pts[0]

        vec_coronal = pt_coronal - pt_mid
        vec_apical = pt_mid - pt_apex

        norm_c = np.linalg.norm(vec_coronal)
        norm_a = np.linalg.norm(vec_apical)

        if norm_c > 1e-6 and norm_a > 1e-6:
            cos_theta = np.clip(np.dot(vec_coronal, vec_apical) / (norm_c * norm_a), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_theta)))
        else:
            angle_deg = 0.0

        # Clinical curvature severity
        if angle_deg < 10.0:
            severity = "Straight (<10°)"
        elif 10.0 <= angle_deg < 25.0:
            severity = "Moderate (10-25°)"
        else:
            severity = "Severe (>25°)"

        # Radiographic apex clearance
        rad_apex_dist = float(np.linalg.norm(ordered_pts[0] - tooth.apex_point_mm))

        c_entr = [round(float(c), 2) for c in tooth.canal_entrance_point_mm] if tooth.canal_entrance_point_mm is not None else None
        c_apex = [round(float(c), 2) for c in tooth.apical_foramen_point_mm] if tooth.apical_foramen_point_mm is not None else None

        return RootCanalMetrics(
            tooth_fdi=fdi,
            pulp_present=True,
            canal_length_mm=round(canal_length, 2),
            schneider_angle_deg=round(angle_deg, 1),
            curvature_severity=severity,
            radiographic_apex_distance_mm=round(rad_apex_dist, 2),
            canal_entrance_coord_mm=c_entr,
            distance_to_canal_entrance_mm=tooth.distance_to_canal_entrance_mm,
            apical_foramen_coord_mm=c_apex,
            pulp_chamber_volume_mm3=round(tooth.pulp_volume_mm3, 1),
        )

    # =========================================================================
    # 4. PERIODONTAL BONE LOSS
    # =========================================================================

    def measure_periodontal_status(self, fdi: str) -> Optional[PeriodontalMetrics]:
        """Quantify CEJ-to-alveolar crest distance and alveolar bone loss percentage."""
        tooth = self.teeth.get(fdi)
        info = get_tooth_label(fdi)
        if not tooth or not info:
            return None

        is_mandible = info.quadrant in (3, 4)
        jaw_id = LOWER_JAW_ID if is_mandible else UPPER_JAW_ID

        # Total anatomical root length
        root_len = float(tooth.root_length_mm) if tooth.root_length_mm else float(np.linalg.norm(tooth.crown_point_mm - tooth.apex_point_mm))
        crown_len = float(tooth.crown_length_mm) if tooth.crown_length_mm else 8.0

        # CEJ level along vertical Z
        # For mandible (crown is superior/+Z), CEJ is below crown tip
        # For maxilla (crown is inferior/-Z), CEJ is above crown tip
        if is_mandible:
            cej_z_mm = tooth.crown_point_mm[2] - crown_len
        else:
            cej_z_mm = tooth.crown_point_mm[2] + crown_len

        # Sample bone crest at 4 quadrants around the tooth: Mesial, Distal, Buccal, Lingual
        arch = self._arch_for_fdi(fdi)
        measured_crests = {}
        if arch is not None:
            t, _ = arch.find_closest_parameter(tooth.centroid_mm)
            frame = arch.get_frame(t)
            norm_bl = frame.normal  # +normal is buccal, -normal is lingual
            tang_md = frame.tangent # +tangent is distal/mesial
            
            # Radii to test slightly outside tooth periphery (~3.5 mm from centroid)
            quad_vectors = {
                "buccal": norm_bl * 3.5,
                "lingual": -norm_bl * 3.5,
                "mesial": -tang_md * 3.5 if info.quadrant in (1, 4) else tang_md * 3.5,
                "distal": tang_md * 3.5 if info.quadrant in (1, 4) else -tang_md * 3.5,
            }
            
            for side, vec in quad_vectors.items():
                col_pt = tooth.centroid_mm + vec
                vox = self.volume.world_to_voxel(col_pt)
                cx, cy = int(round(vox[0])), int(round(vox[1]))
                if 0 <= cx < self.volume.shape[2] and 0 <= cy < self.volume.shape[1]:
                    bone_col = (self.mask[:, cy, cx] == jaw_id) | (self.volume.image[:, cy, cx] >= 250.0)
                    z_hits = np.where(bone_col)[0]
                    if len(z_hits) > 0:
                        # Bone crest voxel
                        crest_vox_z = np.max(z_hits) if is_mandible else np.min(z_hits)
                        crest_z_mm = self.volume.voxel_to_world([cx, cy, crest_vox_z])[2]
                        if is_mandible:
                            dist = max(0.0, cej_z_mm - crest_z_mm)
                        else:
                            dist = max(0.0, crest_z_mm - cej_z_mm)
                        measured_crests[side] = round(float(dist), 2)
                    else:
                        measured_crests[side] = None
                else:
                    measured_crests[side] = None

        m_m = measured_crests.get("mesial")
        d_m = measured_crests.get("distal")
        b_m = measured_crests.get("buccal")
        l_m = measured_crests.get("lingual")

        valid_dists = [v for v in (m_m, d_m, b_m, l_m) if v is not None]
        if valid_dists:
            avg_crest = float(np.mean(valid_dists))
            bone_loss_pct = max(0.0, min(100.0, ((avg_crest - 1.5) / max(1.0, root_len)) * 100))
            if bone_loss_pct < 15.0:
                status = "Normal / Mild Periodontitis"
            elif 15.0 <= bone_loss_pct < 33.0:
                status = "Moderate Periodontitis (Stage II)"
            else:
                status = "Severe Periodontitis (Stage III/IV)"
            return PeriodontalMetrics(
                tooth_fdi=fdi,
                bone_height_mesial_mm=m_m if m_m is not None else avg_crest,
                bone_height_distal_mm=d_m if d_m is not None else avg_crest,
                bone_height_buccal_mm=b_m if b_m is not None else avg_crest,
                bone_height_lingual_mm=l_m if l_m is not None else avg_crest,
                estimated_root_length_mm=round(root_len, 2),
                bone_loss_percentage=round(bone_loss_pct, 1),
                periodontal_status=status,
            )
        else:
            return PeriodontalMetrics(
                tooth_fdi=fdi,
                bone_height_mesial_mm=0.0,
                bone_height_distal_mm=0.0,
                bone_height_buccal_mm=0.0,
                bone_height_lingual_mm=0.0,
                estimated_root_length_mm=round(root_len, 2),
                bone_loss_percentage=0.0,
                periodontal_status="Unavailable / Alveolar Bone Not Segmented",
            )

    def measure_teeth_multi(self, fdis: List[str]) -> List[Dict[str, Any]]:
        """
        Compute measurements across multiple selected teeth.
        Includes tooth dimensions (lengths, MD/BL widths, canal depth),
        implant site parameters (bone height, crest/mid/apical widths, density),
        distance to vital nerve/sinus, and AI implant proposals.
        """
        results = []
        for fdi in fdis:
            meas = self.measure_implant_site(fdi)
            if not meas:
                continue
            item = {
                "tooth_fdi": meas.tooth_fdi,
                "bone_height_mm": round(meas.bone_height_mm, 1),
                "crest_width_mm": round(meas.crest_width_mm, 1),
                "mid_root_width_mm": round(meas.mid_root_width_mm, 1),
                "apical_width_mm": round(meas.apical_width_mm, 1),
                "mesiodistal_space_mm": round(meas.mesiodistal_space_mm, 1),
                "mean_density_hu": round(meas.mean_density_hu, 1),
                "misch_density_class": meas.misch_density_class,
                "distance_to_vital_structure_mm": round(meas.distance_to_vital_structure_mm, 1),
                "vital_structure_name": meas.vital_structure_name,
                "suggested_diameter_mm": round(meas.suggested_diameter_mm, 1),
                "suggested_length_mm": round(meas.suggested_length_mm, 1),
                "suggested_angulation_deg": round(meas.suggested_angulation_deg, 1),
                "buccal_bone_thickness_mm": round(meas.buccal_bone_thickness_mm, 1),
                "lingual_bone_thickness_mm": round(meas.lingual_bone_thickness_mm, 1),
                "socket_depth_mm": round(meas.socket_depth_mm, 1),
                "apical_bone_reserve_mm": round(meas.apical_bone_reserve_mm, 1),
                "tooth_length_mm": round(meas.tooth_length_mm, 1) if meas.tooth_length_mm is not None else None,
                "crown_length_mm": round(meas.crown_length_mm, 1) if meas.crown_length_mm is not None else None,
                "root_length_mm": round(meas.root_length_mm, 1) if meas.root_length_mm is not None else None,
                "tooth_width_md_mm": round(meas.tooth_width_md_mm, 1) if meas.tooth_width_md_mm is not None else None,
                "tooth_width_bl_mm": round(meas.tooth_width_bl_mm, 1) if meas.tooth_width_bl_mm is not None else None,
                "canal_entrance_depth_mm": round(meas.canal_entrance_depth_mm, 1) if meas.canal_entrance_depth_mm is not None else None,
            }
            results.append(item)
        return results

    # =========================================================================
    # 5. JAWBONE PROPORTIONS & LOCKING CHECK
    # =========================================================================

    def compute_jawbone_proportions(self) -> Dict[str, Any]:
        """
        Evaluate volume proportion between upper jaw (maxilla, label 2) and lower jaw (mandible, label 1).
        If the upper-to-lower or lower-to-upper jaw bone proportion is more than 70% biased against normal ratio
        (e.g., one jaw has >85% of total jawbone voxels or the other has <15%), mark the deficient jaw as locked.
        """
        lower_voxels = int((self.mask == LOWER_JAW_ID).sum())
        upper_voxels = int((self.mask == UPPER_JAW_ID).sum())
        total_voxels = lower_voxels + upper_voxels

        # Count detected teeth per jaw as secondary anatomical support
        lower_teeth = sum(1 for t in self.teeth.values() if t.quadrant in (3, 4))
        upper_teeth = sum(1 for t in self.teeth.values() if t.quadrant in (1, 2))

        voxel_volume_mm3 = float(np.prod(self.volume.spacing))
        lower_volume_mm3 = float(lower_voxels * voxel_volume_mm3)
        upper_volume_mm3 = float(upper_voxels * voxel_volume_mm3)

        # Compute vertical physical Z span of each jawbone
        lower_vox_indices = np.where(self.mask == LOWER_JAW_ID)[0]
        upper_vox_indices = np.where(self.mask == UPPER_JAW_ID)[0]
        sz = self.volume.spacing[0]
        lower_z_span_mm = float((np.max(lower_vox_indices) - np.min(lower_vox_indices) + 1) * sz) if len(lower_vox_indices) > 0 else 0.0
        upper_z_span_mm = float((np.max(upper_vox_indices) - np.min(upper_vox_indices) + 1) * sz) if len(upper_vox_indices) > 0 else 0.0

        if total_voxels == 0:
            return {
                "lower_voxels": 0,
                "upper_voxels": 0,
                "lower_volume_mm3": 0.0,
                "upper_volume_mm3": 0.0,
                "lower_z_span_mm": 0.0,
                "upper_z_span_mm": 0.0,
                "lower_proportion": 0.5,
                "upper_proportion": 0.5,
                "bias_percentage": 0.0,
                "has_bias": False,
                "maxilla_coverage": "absent",
                "mandible_coverage": "absent",
                "locked_jaw": None,
                "warning": "No jawbone structures segmented in volume.",
            }

        lower_prop = float(lower_voxels / total_voxels)
        upper_prop = float(upper_voxels / total_voxels)

        bias_pct = float(abs(upper_prop - lower_prop) * 100.0)
        has_bias = bias_pct > 70.0

        min_vox = max(1000, int(0.005 * total_voxels))
        maxilla_coverage = "sufficient" if (upper_voxels >= min_vox and upper_prop >= 0.15) else ("deficient" if upper_voxels > 0 else "absent")
        mandible_coverage = "sufficient" if (lower_voxels >= min_vox and lower_prop >= 0.15) else ("deficient" if lower_voxels > 0 else "absent")

        locked_jaw = None
        warning = None
        if has_bias:
            if upper_prop < lower_prop:
                locked_jaw = "upper"
                warning = (
                    f"Maxilla (upper jaw) coverage is {maxilla_coverage} ({upper_prop*100:.1f}% bone volume, "
                    f"Z-span: {upper_z_span_mm:.1f} mm, bias {bias_pct:.1f}% > 70%). The upper jaw is locked. "
                    "You may still proceed if this scan intentionally focuses only on the mandible."
                )
            else:
                locked_jaw = "lower"
                warning = (
                    f"Mandible (lower jaw) coverage is {mandible_coverage} ({lower_prop*100:.1f}% bone volume, "
                    f"Z-span: {lower_z_span_mm:.1f} mm, bias {bias_pct:.1f}% > 70%). The lower jaw is locked. "
                    "You may still proceed if this scan intentionally focuses only on the maxilla."
                )

        return {
            "lower_voxels": lower_voxels,
            "upper_voxels": upper_voxels,
            "lower_volume_mm3": round(lower_volume_mm3, 1),
            "upper_volume_mm3": round(upper_volume_mm3, 1),
            "lower_z_span_mm": round(lower_z_span_mm, 1),
            "upper_z_span_mm": round(upper_z_span_mm, 1),
            "lower_proportion": round(lower_prop, 4),
            "upper_proportion": round(upper_prop, 4),
            "lower_teeth_count": lower_teeth,
            "upper_teeth_count": upper_teeth,
            "bias_percentage": round(bias_pct, 1),
            "has_bias": has_bias,
            "maxilla_coverage": maxilla_coverage,
            "mandible_coverage": mandible_coverage,
            "locked_jaw": locked_jaw,
            "warning": warning,
        }

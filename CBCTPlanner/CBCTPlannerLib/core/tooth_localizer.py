"""
Per-Tooth Localization, Orientation, and Impaction Analysis.
Maps segmentation masks to FDI notation, extracts centroids, bounding boxes,
PCA long axes (crown-to-apex), and clinical impaction classification (Winter's classification).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .labels import (
    FDI_TO_PULP_ID,
    FDI_TO_TOOTH_ID,
    WISDOM_TEETH_FDI,
    get_tooth_label,
)
from .volume_loader import CBCTVolume


@dataclass
class ToothInfo:
    """Detailed anatomical, geometric, and clinical information for a single tooth."""

    fdi: str
    label_id: int
    name: str
    quadrant: int
    tooth_number: int
    centroid_mm: np.ndarray
    bbox_mm: Tuple[np.ndarray, np.ndarray]
    volume_mm3: float
    long_axis: np.ndarray
    crown_point_mm: np.ndarray
    apex_point_mm: np.ndarray
    angulation_deg: float
    total_length_mm: float = 0.0
    crown_length_mm: float = 0.0
    root_length_mm: float = 0.0
    crown_to_root_ratio: float = 0.0
    mesiodistal_width_mm: float = 0.0
    buccolingual_width_mm: float = 0.0
    cervical_width_md_mm: Optional[float] = None
    cervical_width_bl_mm: Optional[float] = None
    pulp_present: bool = False
    pulp_label_id: Optional[int] = None
    pulp_volume_mm3: float = 0.0
    canal_entrance_point_mm: Optional[np.ndarray] = None
    distance_to_canal_entrance_mm: Optional[float] = None
    apical_foramen_point_mm: Optional[np.ndarray] = None
    impaction_class: Optional[str] = None
    adjacent_fdi: Optional[str] = None


class ToothLocalizer:
    """Analyzes a 3D dental CBCT segmentation mask to localize and characterize all dentition."""

    def __init__(
        self,
        volume: CBCTVolume,
        mask: np.ndarray,
        arch: Optional[Any] = None,
        arches: Optional[Dict[str, Any]] = None,
    ):
        self.volume = volume
        self.mask = mask
        self.arch = arch
        self.arches = arches or {}
        sz, sy, sx = volume.spacing
        self.voxel_volume_mm3 = float(sx * sy * sz)

    def extract_tooth(self, fdi: str) -> Optional[ToothInfo]:
        """Localize and extract geometric properties of a specific tooth by FDI code."""
        info = get_tooth_label(fdi)
        if not info:
            return None

        tooth_mask = self.mask == info.id
        voxel_count = int(np.count_nonzero(tooth_mask))
        if voxel_count < 4:
            return None

        z_idx, y_idx, x_idx = np.where(tooth_mask)
        vox_pts = np.column_stack([x_idx, y_idx, z_idx])
        ones = np.ones((len(vox_pts), 1), dtype=np.float64)
        vox_homog = np.hstack([vox_pts, ones])
        world_pts = (self.volume.affine @ vox_homog.T).T[:, :3]

        centroid_mm = np.mean(world_pts, axis=0)
        min_corner = np.min(world_pts, axis=0)
        max_corner = np.max(world_pts, axis=0)
        vol_mm3 = float(voxel_count * self.voxel_volume_mm3)

        centered = world_pts - centroid_mm
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sort_order = np.argsort(eigenvalues)[::-1]
        primary_axis = eigenvectors[:, sort_order[0]]

        if info.quadrant in (1, 2):
            if primary_axis[2] > 0:
                primary_axis = -primary_axis
        else:
            if primary_axis[2] < 0:
                primary_axis = -primary_axis

        projections = centered @ primary_axis
        apex_offset = float(np.min(projections))
        crown_offset = float(np.max(projections))
        apex_point = centroid_mm + apex_offset * primary_axis
        crown_point = centroid_mm + crown_offset * primary_axis
        total_length_mm = float(max(0.0, crown_offset - apex_offset))
        if total_length_mm <= 1e-6:
            return None

        cand_offsets = np.linspace(
            apex_offset + 0.50 * total_length_mm,
            apex_offset + 0.75 * total_length_mm,
            25,
        )
        min_area = float("inf")
        best_cej_offset = apex_offset + 0.62 * total_length_mm
        for c_off in cand_offsets:
            in_slice = np.abs(projections - c_off) <= 0.6
            count = int(np.count_nonzero(in_slice))
            if 0 < count < min_area:
                min_area = count
                best_cej_offset = float(c_off)

        cej_offset = float(best_cej_offset)
        crown_length_mm = float(max(0.0, crown_offset - cej_offset))
        root_length_mm = float(max(0.0, total_length_mm - crown_length_mm))
        crown_to_root_ratio = round(crown_length_mm / max(0.1, root_length_mm), 2)

        target_jaw = "lower" if info.quadrant in (3, 4) else "upper"
        arch = self.arches.get(target_jaw) or self.arch
        if arch is not None:
            t_pos, _ = arch.find_closest_parameter(centroid_mm)
            frame = arch.get_frame(t_pos)
            axis_md = frame.tangent
            axis_bl = frame.normal
        else:
            axis_md = eigenvectors[:, sort_order[1]]
            axis_bl = eigenvectors[:, sort_order[2]]

        norm_md = np.linalg.norm(axis_md)
        norm_bl = np.linalg.norm(axis_bl)
        axis_md = axis_md / (norm_md if norm_md > 1e-6 else 1.0)
        axis_bl = axis_bl / (norm_bl if norm_bl > 1e-6 else 1.0)

        proj_md = centered @ axis_md
        proj_bl = centered @ axis_bl
        mesiodistal_width_mm = float(np.max(proj_md) - np.min(proj_md))
        buccolingual_width_mm = float(np.max(proj_bl) - np.min(proj_bl))

        cerv_mask = np.abs(projections - cej_offset) <= 1.0
        if np.count_nonzero(cerv_mask) >= 6:
            cervical_width_md_mm: Optional[float] = round(
                float(np.max(proj_md[cerv_mask]) - np.min(proj_md[cerv_mask])), 2
            )
            cervical_width_bl_mm: Optional[float] = round(
                float(np.max(proj_bl[cerv_mask]) - np.min(proj_bl[cerv_mask])), 2
            )
        else:
            cervical_width_md_mm = None
            cervical_width_bl_mm = None

        z_comp = np.abs(primary_axis[2])
        angulation_deg = float(np.degrees(np.arcsin(np.clip(z_comp, 0.0, 1.0))))

        pulp_id = FDI_TO_PULP_ID.get(fdi)
        pulp_present = False
        pulp_vol_mm3 = 0.0
        canal_entrance_point = None
        distance_to_canal_entrance: Optional[float] = None
        apical_foramen_point = None

        if pulp_id is not None:
            pulp_vox = int(np.count_nonzero(self.mask == pulp_id))
            if pulp_vox > 0:
                pulp_present = True
                pulp_vol_mm3 = float(pulp_vox * self.voxel_volume_mm3)
                pz, py, px = np.where(self.mask == pulp_id)
                p_vox = np.column_stack([px, py, pz])
                p_ones = np.ones((len(p_vox), 1), dtype=np.float64)
                p_world = (self.volume.affine @ np.hstack([p_vox, p_ones]).T).T[:, :3]
                p_proj = (p_world - centroid_mm) @ primary_axis

                cej_pulp_mask = np.abs(p_proj - cej_offset) <= 1.5
                if np.any(cej_pulp_mask):
                    canal_entrance_point = np.mean(p_world[cej_pulp_mask], axis=0)
                else:
                    closest_idx = int(np.argmin(np.abs(p_proj - cej_offset)))
                    canal_entrance_point = p_world[closest_idx]

                distance_to_canal_entrance = round(
                    float(np.linalg.norm(crown_point - canal_entrance_point)), 2
                )
                apical_idx = int(np.argmin(p_proj))
                apical_foramen_point = p_world[apical_idx]

        impaction_class = None
        adjacent_fdi = None
        if fdi in WISDOM_TEETH_FDI:
            adjacent_fdi = f"{info.quadrant}7"
            impaction_class = self._classify_impaction(fdi, primary_axis, adjacent_fdi)

        return ToothInfo(
            fdi=fdi,
            label_id=info.id,
            name=info.name,
            quadrant=info.quadrant,
            tooth_number=info.tooth_number,
            centroid_mm=centroid_mm,
            bbox_mm=(min_corner, max_corner),
            volume_mm3=vol_mm3,
            long_axis=primary_axis,
            crown_point_mm=crown_point,
            apex_point_mm=apex_point,
            angulation_deg=angulation_deg,
            total_length_mm=round(total_length_mm, 2),
            crown_length_mm=round(crown_length_mm, 2),
            root_length_mm=round(root_length_mm, 2),
            crown_to_root_ratio=crown_to_root_ratio,
            mesiodistal_width_mm=round(mesiodistal_width_mm, 2),
            buccolingual_width_mm=round(buccolingual_width_mm, 2),
            cervical_width_md_mm=cervical_width_md_mm,
            cervical_width_bl_mm=cervical_width_bl_mm,
            pulp_present=pulp_present,
            pulp_label_id=pulp_id,
            pulp_volume_mm3=pulp_vol_mm3,
            canal_entrance_point_mm=canal_entrance_point,
            distance_to_canal_entrance_mm=distance_to_canal_entrance,
            apical_foramen_point_mm=apical_foramen_point,
            impaction_class=impaction_class,
            adjacent_fdi=adjacent_fdi,
        )

    def extract_all_teeth(self) -> Dict[str, ToothInfo]:
        """Find and characterize all teeth present in the segmentation mask."""
        detected: Dict[str, ToothInfo] = {}
        for fdi in FDI_TO_TOOTH_ID:
            tooth = self.extract_tooth(fdi)
            if tooth is not None:
                detected[fdi] = tooth
        return detected

    def _classify_impaction(
        self,
        wisdom_fdi: str,
        wisdom_long_axis: np.ndarray,
        adjacent_fdi: str,
    ) -> Optional[str]:
        adj_tooth = self.extract_tooth(adjacent_fdi)
        if adj_tooth is None:
            return None
        ref_axis = adj_tooth.long_axis
        dot_prod = float(np.clip(np.dot(wisdom_long_axis, ref_axis), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot_prod)))
        if angle_deg < 11.0:
            return "Vertical"
        if 11.0 <= angle_deg < 70.0:
            return "Mesioangular" if wisdom_long_axis[1] > ref_axis[1] else "Distoangular"
        if 70.0 <= angle_deg <= 100.0:
            return "Horizontal"
        return "Inverted"

"""VFACE (Visual Facial Asymmetry Classification Engine), AutoCrop3D ROI, and sagittal mirroring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from ..core.labels import (
    DENTALSEGMENTATOR_MANDIBLE_ID,
    DENTALSEGMENTATOR_MAXILLA_ID,
    LOWER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    UPPER_JAW_ID,
)
from ..core.volume_loader import CBCTVolume
from .ali import ALILandmarkResult, _sample_mask_world_pts, identify_ali_landmarks


@dataclass
class VFACEAsymmetryResult:
    """Bilateral craniofacial & mandibular asymmetry quantification."""

    right_ramus_height_mm: float
    left_ramus_height_mm: float
    ramus_height_diff_mm: float
    right_mandibular_length_mm: float
    left_mandibular_length_mm: float
    mandibular_length_diff_mm: float
    chin_deviation_mm: float
    chin_deviation_side: str
    asymmetry_index_pct: float
    classification: str  # Symmetric, Mild Asymmetry, Moderate Asymmetry, Severe Asymmetry
    midsagittal_plane_x_ras: float
    metrics_detail: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""


def compute_vface_asymmetry_analysis(
    volume: CBCTVolume,
    mask: Optional[np.ndarray] = None,
    is_5class: bool = False,
    landmarks: Optional[Dict[str, ALILandmarkResult]] = None,
) -> VFACEAsymmetryResult:
    """Compute bilateral Left-vs-Right mandibular & facial symmetry metrics (`VFACE`)."""
    lms = landmarks or identify_ali_landmarks(
        volume, mask=mask, is_5class=is_5class, include_per_tooth_landmarks=False
    )

    def _pt(key: str, default: Sequence[float]) -> np.ndarray:
        if key in lms:
            return np.asarray(lms[key].position_ras, dtype=np.float64)
        return np.asarray(default, dtype=np.float64)

    co_r = _pt("Co_R", [48.0, -25.0, 20.0])
    co_l = _pt("Co_L", [-48.0, -25.0, 20.0])
    go_r = _pt("Go_R", [45.0, -20.0, -30.0])
    go_l = _pt("Go_L", [-45.0, -20.0, -30.0])
    me = _pt("Me", [0.0, 35.0, -45.0])
    n_pt = _pt("N", [0.0, 45.0, 35.0])
    ans = _pt("ANS", [0.0, 42.0, 0.0])
    pns = _pt("PNS", [0.0, 0.0, 0.0])

    midline_x = float(np.median([n_pt[0], ans[0], pns[0]]))

    r_ramus = round(float(np.linalg.norm(co_r - go_r)), 2)
    l_ramus = round(float(np.linalg.norm(co_l - go_l)), 2)
    ramus_diff = round(abs(r_ramus - l_ramus), 2)

    r_mand_len = round(float(np.linalg.norm(go_r - me)), 2)
    l_mand_len = round(float(np.linalg.norm(go_l - me)), 2)
    mand_diff = round(abs(r_mand_len - l_mand_len), 2)

    chin_signed = float(me[0] - midline_x)
    chin_dev = round(abs(chin_signed), 2)
    if chin_dev < 0.8:
        side = "Midline"
    elif chin_signed > 0:
        side = "Right"
    else:
        side = "Left"

    r_asym = 100.0 * ramus_diff / max(1.0, 0.5 * (r_ramus + l_ramus))
    m_asym = 100.0 * mand_diff / max(1.0, 0.5 * (r_mand_len + l_mand_len))
    asym_idx = round(0.45 * r_asym + 0.35 * m_asym + 0.20 * min(20.0, chin_dev * 2.5), 2)

    if chin_dev < 2.0 and ramus_diff < 2.0 and mand_diff < 2.5:
        cls_label = "Symmetric"
    elif chin_dev < 4.0 and ramus_diff < 4.0 and mand_diff < 5.0:
        cls_label = "Mild Asymmetry"
    elif chin_dev < 7.0 and ramus_diff < 7.0:
        cls_label = "Moderate Asymmetry"
    else:
        cls_label = "Severe Asymmetry"

    detail = {
        "right_ramus_mm": r_ramus,
        "left_ramus_mm": l_ramus,
        "ramus_diff_mm": ramus_diff,
        "right_mand_body_mm": r_mand_len,
        "left_mand_body_mm": l_mand_len,
        "mand_body_diff_mm": mand_diff,
        "chin_deviation_mm": chin_dev,
        "condylar_height_diff_mm": round(abs(float(co_r[2] - co_l[2])), 2),
        "occlusal_cant_deg": round(
            float(np.degrees(np.arctan2(abs(go_r[2] - go_l[2]), max(10.0, abs(go_r[0] - go_l[0]))))),
            2,
        ),
    }
    summary = (
        f"{cls_label} | Chin dev: {chin_dev:.1f} mm ({side}) | "
        f"Ramus L/R diff: {ramus_diff:.1f} mm | Body diff: {mand_diff:.1f} mm | Index: {asym_idx:.1f}%"
    )
    return VFACEAsymmetryResult(
        right_ramus_height_mm=r_ramus,
        left_ramus_height_mm=l_ramus,
        ramus_height_diff_mm=ramus_diff,
        right_mandibular_length_mm=r_mand_len,
        left_mandibular_length_mm=l_mand_len,
        mandibular_length_diff_mm=mand_diff,
        chin_deviation_mm=chin_dev,
        chin_deviation_side=side,
        asymmetry_index_pct=asym_idx,
        classification=cls_label,
        midsagittal_plane_x_ras=round(midline_x, 2),
        metrics_detail=detail,
        summary_text=summary,
    )


def compute_autocrop_roi_bounds_ras(
    volume: CBCTVolume,
    mask: Optional[np.ndarray] = None,
    padding_mm: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a tight AutoCrop3D RAS bounding box `[min_ras, max_ras]` around the dental structures."""
    pts = np.empty((0, 3), dtype=np.float64)
    if mask is not None and np.any(mask > 0):
        dental_ids = (
            [LOWER_JAW_ID, UPPER_JAW_ID, DENTALSEGMENTATOR_MANDIBLE_ID, DENTALSEGMENTATOR_MAXILLA_ID]
            + list(MANDIBULAR_TEETH_IDS)
            + list(MAXILLARY_TEETH_IDS)
        )
        pts = _sample_mask_world_pts(volume, mask, dental_ids, max_pts=4000)
        if len(pts) < 10:
            pts = _sample_mask_world_pts(
                volume, mask, [int(x) for x in np.unique(mask) if x > 0], max_pts=4000
            )
    if len(pts) < 10:
        bone_zyx = np.where(volume.image >= 400.0)
        if len(bone_zyx[0]) >= 20:
            step = max(1, len(bone_zyx[0]) // 2000)
            vox = np.column_stack([bone_zyx[2][::step], bone_zyx[1][::step], bone_zyx[0][::step]])
            pts = volume.voxel_to_world(vox)

    if len(pts) < 4:
        corners = np.array(
            [
                [0, 0, 0],
                [volume.shape[2] - 1, volume.shape[1] - 1, volume.shape[0] - 1],
            ],
            dtype=np.float64,
        )
        w = volume.voxel_to_world(corners)
        return np.min(w, axis=0), np.max(w, axis=0)

    pad = np.array([padding_mm, padding_mm, padding_mm], dtype=np.float64)
    return np.min(pts, axis=0) - pad, np.max(pts, axis=0) + pad


def build_sagittal_mirror_matrix_4x4(midline_x_ras: float = 0.0) -> np.ndarray:
    """Build an AutoMatrix 4x4 homogeneous reflection matrix across the sagittal X = `midline_x_ras` plane."""
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = -1.0
    mat[0, 3] = 2.0 * float(midline_x_ras)
    return mat

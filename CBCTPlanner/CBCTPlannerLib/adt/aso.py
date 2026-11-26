"""ASO (Automated Standardized Orientation) upgraded with Kabsch SVD, FH plane & outlier rejection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence
import numpy as np

from ..core.volume_loader import CBCTVolume
from .ali import ALILandmarkResult, identify_ali_landmarks
from .geometry import filter_outlier_landmarks, kabsch_rigid_alignment, vtk_icp_refine


@dataclass
class ASOOrientationResult:
    """3D Standardized Cranial/Occlusal Orientation (`4x4` RAS rigid transform)."""

    matrix_4x4_ras: np.ndarray
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    midsagittal_normal_ras: List[float]
    occlusal_normal_ras: List[float]
    reference_center_ras: List[float]
    landmarks_used: List[str] = field(default_factory=list)
    rejected_landmarks: List[str] = field(default_factory=list)
    alignment_mode: str = "anatomical_fh_occlusal"


def compute_aso_standardized_orientation(
    volume: CBCTVolume,
    mask: Optional[np.ndarray] = None,
    is_5class: bool = False,
    landmarks: Optional[Dict[str, ALILandmarkResult]] = None,
    reference_landmarks_ras: Optional[Dict[str, Sequence[float]]] = None,
    use_icp_refinement: bool = True,
) -> ASOOrientationResult:
    """Compute a 4x4 rigid RAS orientation matrix aligning Midsagittal and Frankfort/Occlusal planes.

    When `reference_landmarks_ras` (gold model landmarks, ASO_CBCT style) is provided with >= 3
    matching labels, performs pairwise geometric outlier filtering (`filter_outlier_landmarks`)
    followed by exact rigid Kabsch/SVD alignment and optional VTK ICP refinement.
    Otherwise solves the canonical Midsagittal + Frankfort/Occlusal orthogonal basis from
    bilateral and midline landmarks.
    """
    lms = landmarks or identify_ali_landmarks(
        volume, mask=mask, is_5class=is_5class, include_per_tooth_landmarks=False
    )

    if reference_landmarks_ras:
        src_map = {k: lm.position_ras for k, lm in lms.items() if k in reference_landmarks_ras}
        if len(src_map) >= 3:
            kept, rejected = filter_outlier_landmarks(src_map, reference_landmarks_ras)
            src_pts = np.asarray([src_map[k] for k in kept], dtype=np.float64)
            tgt_pts = np.asarray([reference_landmarks_ras[k] for k in kept], dtype=np.float64)
            mat = kabsch_rigid_alignment(src_pts, tgt_pts)
            mode = "kabsch_reference"
            if use_icp_refinement:
                aligned = (mat[:3, :3] @ src_pts.T).T + mat[:3, 3]
                icp_m = vtk_icp_refine(aligned, tgt_pts)
                if icp_m is not None:
                    mat = icp_m @ mat
                    mode = "kabsch_icp_reference"
            R = mat[:3, :3]
            x_axis = R[0, :]
            z_axis = R[2, :]
            center = src_pts.mean(axis=0)
            return ASOOrientationResult(
                matrix_4x4_ras=mat,
                pitch_deg=round(math.degrees(math.atan2(float(z_axis[1]), float(z_axis[2]))), 2),
                roll_deg=round(math.degrees(math.atan2(float(-z_axis[0]), float(z_axis[2]))), 2),
                yaw_deg=round(math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))), 2),
                midsagittal_normal_ras=[round(float(v), 5) for v in x_axis],
                occlusal_normal_ras=[round(float(v), 5) for v in z_axis],
                reference_center_ras=[round(float(v), 2) for v in center],
                landmarks_used=sorted(kept),
                rejected_landmarks=sorted(rejected),
                alignment_mode=mode,
            )

    used: List[str] = []
    # 1. Weighted bilateral pairs for Right-to-Left (+X) axis (foramina, molars, condyles, gonia, orbits)
    rl_vectors: List[np.ndarray] = []
    for r_key, l_key, weight in (
        ("U6_R", "U6_L", 1.2),
        ("L6_R", "L6_L", 1.2),
        ("MentalForamen_R", "MentalForamen_L", 1.2),
        ("Co_R", "Co_L", 1.0),
        ("Go_R", "Go_L", 1.0),
        ("Or_R", "Or_L", 0.7),
        ("Po_R", "Po_L", 0.7),
    ):
        if r_key in lms and l_key in lms:
            vec = np.asarray(lms[r_key].position_ras, dtype=np.float64) - np.asarray(
                lms[l_key].position_ras, dtype=np.float64
            )
            norm = float(np.linalg.norm(vec))
            if norm > 5.0:
                rl_vectors.append((vec / norm) * weight)
                used.extend([r_key, l_key])

    if rl_vectors:
        x_axis = np.sum(rl_vectors, axis=0)
        x_axis /= max(1e-8, float(np.linalg.norm(x_axis)))
        if x_axis[0] < 0:
            x_axis = -x_axis
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # 2. Anterior (+Y) axis from Frankfort Horizontal (Po_mid -> Or_mid), palatal (PNS -> ANS), and Go_mid -> Pg/Gn
    ap_vectors: List[np.ndarray] = []
    if "Po_R" in lms and "Po_L" in lms and "Or_R" in lms and "Or_L" in lms:
        po_mid = 0.5 * (np.asarray(lms["Po_R"].position_ras) + np.asarray(lms["Po_L"].position_ras))
        or_mid = 0.5 * (np.asarray(lms["Or_R"].position_ras) + np.asarray(lms["Or_L"].position_ras))
        v_fh = or_mid - po_mid
        if float(np.linalg.norm(v_fh)) > 10.0:
            ap_vectors.append((v_fh / float(np.linalg.norm(v_fh))) * 1.1)
            used.extend(["Po_R", "Po_L", "Or_R", "Or_L"])
    if "ANS" in lms and "PNS" in lms:
        v_pal = np.asarray(lms["ANS"].position_ras) - np.asarray(lms["PNS"].position_ras)
        if float(np.linalg.norm(v_pal)) > 3.0:
            ap_vectors.append(v_pal / float(np.linalg.norm(v_pal)))
            used.extend(["ANS", "PNS"])
    if "Go_R" in lms and "Go_L" in lms and "Gn" in lms:
        go_mid = 0.5 * (np.asarray(lms["Go_R"].position_ras) + np.asarray(lms["Go_L"].position_ras))
        v_mand = np.asarray(lms["Gn"].position_ras) - go_mid
        v_mand[2] = 0.0
        if float(np.linalg.norm(v_mand)) > 3.0:
            ap_vectors.append((v_mand / float(np.linalg.norm(v_mand))) * 0.8)
            used.append("Gn")

    y_raw = np.sum(ap_vectors, axis=0) if ap_vectors else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = y_raw - np.dot(y_raw, x_axis) * x_axis
    y_norm = float(np.linalg.norm(y_axis))
    y_axis = y_axis / y_norm if y_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if y_axis[1] < 0:
        y_axis = -y_axis

    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    z_axis = z_axis / z_norm if z_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if z_axis[2] < 0:
        z_axis = -z_axis
        y_axis = np.cross(z_axis, x_axis)

    R = np.vstack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(R)) < 0:
        R[0, :] = -R[0, :]

    all_pts = [np.asarray(lm.position_ras, dtype=np.float64) for lm in lms.values()]
    center = np.mean(all_pts, axis=0) if all_pts else np.zeros(3, dtype=np.float64)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3] = center - R @ center

    return ASOOrientationResult(
        matrix_4x4_ras=mat,
        pitch_deg=round(math.degrees(math.atan2(float(z_axis[1]), float(z_axis[2]))), 2),
        roll_deg=round(math.degrees(math.atan2(float(-z_axis[0]), float(z_axis[2]))), 2),
        yaw_deg=round(math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))), 2),
        midsagittal_normal_ras=[round(float(v), 5) for v in x_axis],
        occlusal_normal_ras=[round(float(v), 5) for v in z_axis],
        reference_center_ras=[round(float(v), 2) for v in center],
        landmarks_used=sorted(set(used)),
        alignment_mode="anatomical_fh_occlusal",
    )

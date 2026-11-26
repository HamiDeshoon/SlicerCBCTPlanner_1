"""CLI-C (Classification & Localization of Impacted Canines and Molars).

Upgrades upstream Mask-R-CNN concept (`Buccal=1, Bicortical=2, Palatal/Lingual=3`) with:
1. Arch-normal cortical boundary localization (exact signed distance along fitted `DentalArch.normal`
   vs local jawbone buccal & palatal/lingual cortical surfaces + 3D EDT contact distance to adjacent roots).
2. Optional `torchvision` Mask-R-CNN (`final_model.pth`) 2D slice inference runner bridge.
3. 3D `CLIC` localization labelmap generation (`1=Buccal, 2=Bicortical, 3=Palatal/Lingual`).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np

from ..core.arch_detector import DentalArch, fit_arch_from_label_mask
from ..core.labels import LOWER_JAW_ID, UPPER_JAW_ID
from ..core.tooth_localizer import ToothLocalizer
from ..core.volume_loader import CBCTVolume

logger = logging.getLogger("CBCTPlanner.adt.clic")

CLIC_CLASS_NAMES = {1: "Buccal", 2: "Bicortical", 3: "Palatal"}
CLIC_CLASS_CODES = {"Buccal": 1, "Bicortical": 2, "Palatal": 3, "Lingual": 3}


@dataclass
class ImpactedToothCLICResult:
    """Impacted canine / wisdom tooth localization and root resorption risk (`CLI-C`)."""

    fdi: str
    tooth_name: str
    intra_osseous_zone: str  # Buccal, Bicortical, Palatal, Lingual
    winter_angulation_class: str  # Vertical, Mesioangular, Distoangular, Horizontal, Inverted
    angulation_to_midline_deg: float
    distance_to_occlusal_plane_mm: float
    buccolingual_offset_mm: float
    nearest_adjacent_fdi: str
    adjacent_root_distance_mm: Optional[float]
    resorption_risk: str
    crown_ras: List[float]
    apex_ras: List[float]
    class_id: int = 2
    buccal_cortex_clearance_mm: Optional[float] = None
    lingual_cortex_clearance_mm: Optional[float] = None


def _exact_surface_distance_mm(
    volume: CBCTVolume,
    mask: np.ndarray,
    id_a: int,
    id_b: int,
) -> Optional[float]:
    """Exact 3D surface-to-surface minimum distance (mm) between two labels via cropped EDT."""
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore

        a_idx = np.where(mask == int(id_a))
        b_idx = np.where(mask == int(id_b))
        if len(a_idx[0]) == 0 or len(b_idx[0]) == 0:
            return None
        pad = 6
        z0 = max(0, int(min(a_idx[0].min(), b_idx[0].min())) - pad)
        z1 = min(mask.shape[0], int(max(a_idx[0].max(), b_idx[0].max())) + pad + 1)
        y0 = max(0, int(min(a_idx[1].min(), b_idx[1].min())) - pad)
        y1 = min(mask.shape[1], int(max(a_idx[1].max(), b_idx[1].max())) + pad + 1)
        x0 = max(0, int(min(a_idx[2].min(), b_idx[2].min())) - pad)
        x1 = min(mask.shape[2], int(max(a_idx[2].max(), b_idx[2].max())) + pad + 1)
        sub_a = mask[z0:z1, y0:y1, x0:x1] == int(id_a)
        sub_b = mask[z0:z1, y0:y1, x0:x1] == int(id_b)
        sz, sy, sx = volume.spacing
        dt = distance_transform_edt(~sub_b, sampling=(sz, sy, sx))
        return round(float(np.min(dt[sub_a])), 2)
    except Exception:
        return None


def _jawbone_cortical_signed_distances(
    volume: CBCTVolume,
    mask: np.ndarray,
    jaw_id: int,
    crown_ras: np.ndarray,
    buccal_normal: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Measure crown-center distances (mm) to outer buccal & outer palatal/lingual bone borders along `buccal_normal`."""
    try:
        steps = np.arange(-16.0, 16.25, 0.25, dtype=np.float64)
        pts_ras = np.asarray(crown_ras, dtype=np.float64)[None, :] + steps[:, None] * np.asarray(buccal_normal, dtype=np.float64)[None, :]
        vox = volume.world_to_voxel(pts_ras)
        rx = np.rint(vox[:, 0]).astype(int)
        ry = np.rint(vox[:, 1]).astype(int)
        rz = np.rint(vox[:, 2]).astype(int)
        valid = (
            (rx >= 0)
            & (rx < mask.shape[2])
            & (ry >= 0)
            & (ry < mask.shape[1])
            & (rz >= 0)
            & (rz < mask.shape[0])
        )
        if not valid.any():
            return None, None
        in_bone = np.zeros(len(steps), dtype=bool)
        # Treat jawbone OR tooth voxels in the alveolus as bone corridor; find outer edges of bone.
        sampled = mask[np.clip(rz, 0, mask.shape[0] - 1), np.clip(ry, 0, mask.shape[1] - 1), np.clip(rx, 0, mask.shape[2] - 1)]
        in_bone[valid] = (sampled[valid] == int(jaw_id)) | (sampled[valid] >= 11)
        hits = np.flatnonzero(in_bone)
        if hits.size < 2:
            return None, None
        s_ling = float(steps[int(hits[0])])
        s_bucc = float(steps[int(hits[-1])])
        return round(max(0.0, s_bucc), 2), round(max(0.0, -s_ling), 2)
    except Exception:
        return None, None


def run_clic_maskrcnn_inference(
    volume_zyx: np.ndarray,
    model_pth: Path,
    *,
    score_threshold: float = 0.7,
) -> np.ndarray:
    """Optional `CLIC_runner` compatible Mask-R-CNN ResNet50-FPN inference producing 3D `{1,2,3}` mask."""
    import torch  # type: ignore
    from torchvision.models.detection import maskrcnn_resnet50_fpn  # type: ignore
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # type: ignore
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor  # type: ignore

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = maskrcnn_resnet50_fpn(weights=None)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 4)
    in_fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, 4)
    state = torch.load(str(model_pth), map_location=dev)
    m.load_state_dict(state)
    m.to(dev).eval()

    vol = np.asarray(volume_zyx, dtype=np.float32)
    seg = np.zeros(vol.shape, dtype=np.int16)
    with torch.no_grad():
        for z in range(vol.shape[0]):
            sl = vol[z]
            lo, hi = float(sl.min()), float(sl.max())
            if hi - lo <= 1e-6:
                continue
            norm = (sl - lo) / (hi - lo)
            t = torch.from_numpy(norm).unsqueeze(0).repeat(3, 1, 1).float().to(dev)
            pred = m([t])[0]
            keep = pred["scores"] >= float(score_threshold)
            for mk, lb in zip(
                (pred["masks"][keep] > 0.5).squeeze(1).cpu().numpy(),
                pred["labels"][keep].cpu().numpy(),
            ):
                seg[z][mk] = int(lb)
    return seg


def analyze_impacted_teeth_clic(
    volume: CBCTVolume,
    mask: np.ndarray,
    is_5class: bool = False,
    target_fdis: Sequence[str] = ("13", "23", "33", "43", "18", "28", "38", "48"),
    arch_lower: Optional[DentalArch] = None,
    arch_upper: Optional[DentalArch] = None,
    build_labelmap: bool = False,
) -> Tuple[List[ImpactedToothCLICResult], Optional[np.ndarray]]:
    """Run upgraded CLI-C localization (`Buccal / Bicortical / Palatal-Lingual`, exact EDT root contact & optional labelmap)."""
    if mask is None or is_5class:
        return [], None

    localizer = ToothLocalizer(volume, mask)
    all_teeth = localizer.extract_all_teeth()
    if not all_teeth:
        return [], None

    if arch_lower is None:
        fit_l = fit_arch_from_label_mask(mask, volume.affine, jaw="lower", is_5class=False)
        if fit_l is not None and fit_l.accepted and len(fit_l.points_ras) >= 4:
            arch_lower = DentalArch(fit_l.points_ras, jaw_type="lower", ras_convention=True)
    if arch_upper is None:
        fit_u = fit_arch_from_label_mask(mask, volume.affine, jaw="upper", is_5class=False)
        if fit_u is not None and fit_u.accepted and len(fit_u.points_ras) >= 4:
            arch_upper = DentalArch(fit_u.points_ras, jaw_type="upper", ras_convention=True)

    all_crowns = np.array([t.crown_point_mm for t in all_teeth.values()], dtype=np.float64)
    occlusal_z = float(np.median(all_crowns[:, 2]))
    clic_labelmap = np.zeros(mask.shape, dtype=np.int16) if build_labelmap else None

    results: List[ImpactedToothCLICResult] = []
    for fdi_str in target_fdis:
        tooth = all_teeth.get(str(fdi_str))
        if tooth is None:
            continue

        q = tooth.quadrant
        is_mand = q in (3, 4)
        arch = arch_lower if is_mand else arch_upper
        neighbors = [
            t
            for k, t in all_teeth.items()
            if t.quadrant == q and k != fdi_str and abs(t.tooth_number - tooth.tooth_number) <= 2
        ]

        # True arch-frame buccal normal when arch is fitted; radial fallback otherwise
        if arch is not None:
            t_param, _ = arch.find_closest_parameter(tooth.crown_point_mm)
            arch_pt = arch.evaluate_at(t_param)
            frame = arch.get_frame(t_param)
            buccal_normal = np.asarray(frame.normal, dtype=np.float64)
            bl_offset = float(np.dot(tooth.crown_point_mm - arch_pt, buccal_normal))
        elif neighbors:
            ref_center = np.mean([n.centroid_mm for n in neighbors], axis=0)
            radial_vec = np.array([ref_center[0], ref_center[1] + 20.0, 0.0], dtype=np.float64)
            buccal_normal = radial_vec / max(1e-6, float(np.linalg.norm(radial_vec[:2])))
            bl_offset = float(np.dot(tooth.crown_point_mm - ref_center, buccal_normal))
        else:
            buccal_normal = np.array([1.0 if q in (1, 4) else -1.0, 0.0, 0.0], dtype=np.float64)
            bl_offset = 0.0

        jaw_id = LOWER_JAW_ID if is_mand else UPPER_JAW_ID
        b_clear, l_clear = _jawbone_cortical_signed_distances(
            volume, mask, jaw_id, tooth.crown_point_mm, buccal_normal
        )

        # Cortical proximity + signed arch offset classify Buccal vs Bicortical vs Palatal/Lingual
        if b_clear is not None and l_clear is not None:
            if bl_offset > 1.0 or (b_clear <= 2.2 and l_clear >= 3.2):
                zone = "Buccal"
                cls_id = 1
            elif bl_offset < -1.0 or (l_clear <= 2.2 and b_clear >= 3.2):
                zone = "Palatal" if q in (1, 2) else "Lingual"
                cls_id = 3
            else:
                zone = "Bicortical"
                cls_id = 2
        elif bl_offset > 1.2:
            zone = "Buccal"
            cls_id = 1
        elif bl_offset < -1.2:
            zone = "Palatal" if q in (1, 2) else "Lingual"
            cls_id = 3
        else:
            zone = "Bicortical"
            cls_id = 2

        if clic_labelmap is not None:
            clic_labelmap[mask == int(tooth.label_id)] = int(cls_id)

        vertical_vec = np.array([0.0, 0.0, -1.0 if q in (1, 2) else 1.0], dtype=np.float64)
        cos_v = float(np.clip(abs(np.dot(tooth.long_axis, vertical_vec)), 0.0, 1.0))
        ang_midline = round(math.degrees(math.acos(cos_v)), 1)

        winter = tooth.impaction_class
        if not winter:
            if ang_midline < 15.0:
                winter = "Vertical"
            elif ang_midline < 65.0:
                winter = "Mesioangular"
            elif ang_midline <= 105.0:
                winter = "Horizontal"
            else:
                winter = "Inverted"

        dist_occ = round(abs(float(tooth.crown_point_mm[2]) - occlusal_z), 2)

        nearest_fdi = "None"
        min_root_dist: Optional[float] = None
        if neighbors:
            best_n = min(
                neighbors,
                key=lambda n: float(np.linalg.norm(tooth.crown_point_mm - n.apex_point_mm)),
            )
            nearest_fdi = best_n.fdi
            edt_d = _exact_surface_distance_mm(volume, mask, int(tooth.label_id), int(best_n.label_id))
            if edt_d is not None:
                min_root_dist = edt_d
            else:
                raw_d = float(
                    min(
                        np.linalg.norm(tooth.crown_point_mm - best_n.centroid_mm),
                        np.linalg.norm(tooth.crown_point_mm - best_n.apex_point_mm),
                        np.linalg.norm(tooth.centroid_mm - best_n.centroid_mm),
                    )
                )
                min_root_dist = round(max(0.0, raw_d - 3.2), 2)

        if min_root_dist is None or min_root_dist >= 1.5:
            risk = "Low (>= 1.5 mm)"
        elif min_root_dist >= 0.8:
            risk = "Moderate (0.8 - 1.5 mm)"
        else:
            risk = "High (< 0.8 mm Contact Risk)"

        results.append(
            ImpactedToothCLICResult(
                fdi=str(fdi_str),
                tooth_name=tooth.name,
                intra_osseous_zone=zone,
                winter_angulation_class=winter,
                angulation_to_midline_deg=ang_midline,
                distance_to_occlusal_plane_mm=dist_occ,
                buccolingual_offset_mm=round(bl_offset, 2),
                nearest_adjacent_fdi=nearest_fdi,
                adjacent_root_distance_mm=min_root_dist,
                resorption_risk=risk,
                crown_ras=[round(float(x), 2) for x in tooth.crown_point_mm],
                apex_ras=[round(float(x), 2) for x in tooth.apex_point_mm],
                class_id=cls_id,
                buccal_cortex_clearance_mm=b_clear,
                lingual_cortex_clearance_mm=l_clear,
            )
        )

    return results, clic_labelmap

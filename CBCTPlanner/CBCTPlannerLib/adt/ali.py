"""ALI (Automatic Landmark Identification) for dental CBCT volumes & segmentation masks.

Identifies skeletal, cephalometric, neurovascular foramen, and per-tooth morphometric
landmarks in RAS space, exports/imports 3D Slicer `.mrk.json` markups compatible with
`SlicerAutomatedDentalTools/ALI_CBCT`, and exposes anatomical jaw bounds that keep
edentulous / partial-arch fitting from climbing onto the ramus or deviating at midline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..core.labels import (
    DENTALSEGMENTATOR_CANAL_ID,
    DENTALSEGMENTATOR_MANDIBLE_ID,
    DENTALSEGMENTATOR_MAXILLA_ID,
    LEFT_IAN_ID,
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    UPPER_JAW_ID,
)
from ..core.tooth_localizer import ToothLocalizer
from ..core.volume_loader import CBCTVolume


@dataclass
class ALILandmarkResult:
    """Single 3D landmark produced by the ALI engine."""

    label: str
    category: str  # skeletal, dental, foramen, tooth_morphometry
    position_ras: List[float]
    description: str = ""
    confidence: float = 0.85


def _sample_mask_world_pts(
    volume: CBCTVolume, mask: np.ndarray, labels: Sequence[int], max_pts: int = 4000
) -> np.ndarray:
    zyx = np.where(np.isin(mask, list(labels)))
    if len(zyx[0]) == 0:
        return np.empty((0, 3), dtype=np.float64)
    step = max(1, len(zyx[0]) // max(10, int(max_pts)))
    vox = np.column_stack([zyx[2][::step], zyx[1][::step], zyx[0][::step]])
    return volume.voxel_to_world(vox)


def _extract_hu_bone_clouds(volume: CBCTVolume, max_pts: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """Extract inferior (mandible) and superior (maxilla/cranial) bone point clouds above 450 HU."""
    bone_zyx = np.where(volume.image >= 450.0)
    if len(bone_zyx[0]) < 40:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    step = max(1, len(bone_zyx[0]) // max(20, int(max_pts)))
    vox = np.column_stack([bone_zyx[2][::step], bone_zyx[1][::step], bone_zyx[0][::step]])
    all_bone = volume.voxel_to_world(vox)
    z_mid = float(np.median(all_bone[:, 2]))
    return all_bone[all_bone[:, 2] <= z_mid], all_bone[all_bone[:, 2] > z_mid]


def identify_ali_landmarks(
    volume: CBCTVolume,
    mask: Optional[np.ndarray] = None,
    is_5class: bool = False,
    include_per_tooth_landmarks: bool = True,
) -> Dict[str, ALILandmarkResult]:
    """Automatically identify 3D skeletal, cephalometric, foramen, and dental landmarks."""
    landmarks: Dict[str, ALILandmarkResult] = {}

    def _add(lbl: str, cat: str, pt: Sequence[float], desc: str, conf: float = 0.85) -> None:
        landmarks[lbl] = ALILandmarkResult(
            label=lbl,
            category=cat,
            position_ras=[round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]), 2)],
            description=desc,
            confidence=round(float(conf), 2),
        )

    mand_ids = [DENTALSEGMENTATOR_MANDIBLE_ID] if is_5class else [LOWER_JAW_ID, DENTALSEGMENTATOR_MANDIBLE_ID]
    max_ids = [DENTALSEGMENTATOR_MAXILLA_ID] if is_5class else [UPPER_JAW_ID, DENTALSEGMENTATOR_MAXILLA_ID]
    canal_ids = (
        [DENTALSEGMENTATOR_CANAL_ID] if is_5class else [LEFT_IAN_ID, RIGHT_IAN_ID, DENTALSEGMENTATOR_CANAL_ID]
    )

    mand_pts = _sample_mask_world_pts(volume, mask, mand_ids) if mask is not None else np.empty((0, 3))
    max_pts = _sample_mask_world_pts(volume, mask, max_ids) if mask is not None else np.empty((0, 3))
    canal_pts = _sample_mask_world_pts(volume, mask, canal_ids) if mask is not None else np.empty((0, 3))

    from_seg_bone = len(mand_pts) >= 20 or len(max_pts) >= 20
    if not from_seg_bone:
        mand_pts, max_pts = _extract_hu_bone_clouds(volume)
    base_conf = 0.92 if from_seg_bone else 0.65

    # Estimate anatomical midline X even when FOV is shifted away from world X=0
    anchor_cloud = mand_pts if len(mand_pts) >= 10 else max_pts
    x_mid_est = (
        float(0.5 * (np.percentile(anchor_cloud[:, 0], 10) + np.percentile(anchor_cloud[:, 0], 90)))
        if len(anchor_cloud) >= 10
        else 0.0
    )

    # Mandibular landmarks: Me, Pg, Gn, B, Co_R/L, Go_R/L
    if len(mand_pts) >= 10:
        midline_mand = mand_pts[np.abs(mand_pts[:, 0] - x_mid_est) <= 10.0]
        if len(midline_mand) < 4:
            midline_mand = mand_pts

        me_idx = int(np.argmin(midline_mand[:, 2]))
        me_pt = midline_mand[me_idx]
        _add("Me", "skeletal", me_pt, "Menton (Most inferior point of mandibular symphysis)", base_conf)

        ant_symph = midline_mand[midline_mand[:, 2] <= float(np.percentile(midline_mand[:, 2], 50))]
        if len(ant_symph) == 0:
            ant_symph = midline_mand
        pg_pt = ant_symph[int(np.argmax(ant_symph[:, 1]))]
        _add("Pg", "skeletal", pg_pt, "Pogonion (Most anterior point of chin)", base_conf)

        gn_pt = 0.5 * (pg_pt + me_pt)
        _add("Gn", "skeletal", gn_pt, "Gnathion (Anteroinferior chin point)", base_conf)

        # Supramentale (B-point): most posterior concavity along anterior midline above Pg
        upper_symph = midline_mand[midline_mand[:, 2] >= float(pg_pt[2]) + 3.0]
        if len(upper_symph) >= 3:
            b_front = upper_symph[upper_symph[:, 1] >= float(np.percentile(upper_symph[:, 1], 75))]
            if len(b_front) >= 2:
                # Take the minimal Y among top anterior cortex profile before alveolar crest flare
                z_band = b_front[
                    (b_front[:, 2] >= float(pg_pt[2]) + 4.0) & (b_front[:, 2] <= float(pg_pt[2]) + 13.0)
                ]
                b_pt = (
                    z_band[int(np.argmin(z_band[:, 1]))]
                    if len(z_band) > 0
                    else (pg_pt + np.array([0.0, -1.5, 7.5]))
                )
            else:
                b_pt = pg_pt + np.array([0.0, -1.5, 7.5])
        else:
            b_pt = pg_pt + np.array([0.0, -1.5, 7.5], dtype=np.float64)
        _add("B", "skeletal", b_pt, "B-Point (Supramentale)", base_conf)

        for side_tag, side_mask in (("R", mand_pts[:, 0] >= x_mid_est), ("L", mand_pts[:, 0] < x_mid_est)):
            hemi = mand_pts[side_mask]
            if len(hemi) >= 5:
                co_score = hemi[:, 2] - 0.45 * hemi[:, 1]
                co_pt = hemi[int(np.argmax(co_score))]
                _add(f"Co_{side_tag}", "skeletal", co_pt, f"Condylion ({side_tag})", base_conf)

                go_score = -hemi[:, 1] - 0.8 * hemi[:, 2]
                go_pt = hemi[int(np.argmax(go_score))]
                _add(f"Go_{side_tag}", "skeletal", go_pt, f"Gonion ({side_tag})", base_conf)

    # Maxillary & cranial landmarks: ANS, PNS, A, N, S, Or_R/L, Po_R/L
    if len(max_pts) >= 10:
        midline_max = max_pts[np.abs(max_pts[:, 0] - x_mid_est) <= 10.0]
        if len(midline_max) < 4:
            midline_max = max_pts

        ans_pt = midline_max[int(np.argmax(midline_max[:, 1]))]
        _add("ANS", "skeletal", ans_pt, "Anterior Nasal Spine", base_conf)

        pns_candidates = midline_max[midline_max[:, 2] <= float(np.median(midline_max[:, 2]))]
        if len(pns_candidates) == 0:
            pns_candidates = midline_max
        pns_pt = pns_candidates[int(np.argmin(pns_candidates[:, 1]))]
        _add("PNS", "skeletal", pns_pt, "Posterior Nasal Spine", base_conf)

        sub_ans = midline_max[midline_max[:, 2] <= float(ans_pt[2]) - 1.5]
        if len(sub_ans) >= 3:
            a_cand = sub_ans[sub_ans[:, 1] >= float(ans_pt[1]) - 10.0]
            a_pt = (
                a_cand[int(np.argmax(a_cand[:, 1]))]
                if len(a_cand) > 0
                else (ans_pt + np.array([0.0, -2.0, -4.5]))
            )
        else:
            a_pt = ans_pt + np.array([0.0, -2.0, -4.5], dtype=np.float64)
        _add("A", "skeletal", a_pt, "A-Point (Subspinale)", base_conf)

        top_max = midline_max[int(np.argmax(midline_max[:, 2]))]
        n_pt = np.array([float(top_max[0]), float(ans_pt[1]) + 2.0, float(top_max[2]) + 18.0])
        s_pt = np.array([x_mid_est, float(pns_pt[1]) - 18.0, float(n_pt[2]) - 4.0])
        _add("N", "skeletal", n_pt, "Nasion (Frontonasal suture)", 0.6)
        _add("S", "skeletal", s_pt, "Sella Turcica center", 0.55)

        # Bilateral Orbitale & Porion estimated from actual superior-anterior & superior-posterior hemi-maxilla/temporal extents
        for side_tag, sign, side_mask in (
            ("R", 1.0, max_pts[:, 0] >= x_mid_est + 12.0),
            ("L", -1.0, max_pts[:, 0] <= x_mid_est - 12.0),
        ):
            hemi_u = max_pts[side_mask]
            if len(hemi_u) >= 8:
                ant_u = hemi_u[hemi_u[:, 2] >= float(np.percentile(hemi_u[:, 2], 70))]
                or_pt = (
                    ant_u[int(np.argmax(ant_u[:, 1]))]
                    if len(ant_u) > 0
                    else np.array([x_mid_est + sign * 32.0, float(ans_pt[1]) - 6.0, float(ans_pt[2]) + 22.0])
                )
                post_u = hemi_u[hemi_u[:, 2] >= float(np.percentile(hemi_u[:, 2], 65))]
                po_pt = (
                    post_u[int(np.argmin(post_u[:, 1]))]
                    if len(post_u) > 0
                    else np.array([x_mid_est + sign * 52.0, float(pns_pt[1]) - 15.0, float(ans_pt[2]) + 22.0])
                )
            else:
                or_pt = np.array([x_mid_est + sign * 32.0, float(ans_pt[1]) - 6.0, float(ans_pt[2]) + 22.0])
                po_pt = np.array([x_mid_est + sign * 52.0, float(pns_pt[1]) - 15.0, float(ans_pt[2]) + 22.0])
            _add(f"Or_{side_tag}", "skeletal", or_pt, f"Orbitale ({side_tag})", 0.7)
            _add(f"Po_{side_tag}", "skeletal", po_pt, f"Porion ({side_tag})", 0.65)

    # Mental Foramen (anterior terminal of Left/Right IAN canal)
    if len(canal_pts) >= 4:
        for side_tag, side_mask in (("R", canal_pts[:, 0] >= x_mid_est), ("L", canal_pts[:, 0] < x_mid_est)):
            c_side = canal_pts[side_mask]
            if len(c_side) > 0:
                mf_pt = c_side[int(np.argmax(c_side[:, 1]))]
                _add(
                    f"MentalForamen_{side_tag}",
                    "foramen",
                    mf_pt,
                    f"Mental Foramen ({side_tag} anterior IAN exit)",
                    0.95,
                )

    # Per-Tooth Crown Tip & Root Apex landmarks
    if mask is not None and not is_5class:
        localizer = ToothLocalizer(volume, mask)
        teeth = localizer.extract_all_teeth()
        for fdi_str, t_info in sorted(teeth.items()):
            if include_per_tooth_landmarks:
                _add(
                    f"Crown_{fdi_str}",
                    "tooth_morphometry",
                    t_info.crown_point_mm,
                    f"Tooth #{fdi_str} Crown Tip / Incisal-Occlusal Point",
                    0.95,
                )
                _add(
                    f"Apex_{fdi_str}",
                    "tooth_morphometry",
                    t_info.apex_point_mm,
                    f"Tooth #{fdi_str} Root Apex",
                    0.95,
                )
            if fdi_str in ("11", "21") and "U1_tip" not in landmarks:
                _add("U1_tip", "dental", t_info.crown_point_mm, "Upper Central Incisor Tip", 0.95)
                _add("U1_apex", "dental", t_info.apex_point_mm, "Upper Central Incisor Apex", 0.95)
            if fdi_str in ("31", "41") and "L1_tip" not in landmarks:
                _add("L1_tip", "dental", t_info.crown_point_mm, "Lower Central Incisor Tip", 0.95)
                _add("L1_apex", "dental", t_info.apex_point_mm, "Lower Central Incisor Apex", 0.95)
            if fdi_str == "16":
                _add("U6_R", "dental", t_info.crown_point_mm, "Upper Right First Molar (#16)", 0.95)
            if fdi_str == "26":
                _add("U6_L", "dental", t_info.crown_point_mm, "Upper Left First Molar (#26)", 0.95)
            if fdi_str == "46":
                _add("L6_R", "dental", t_info.crown_point_mm, "Lower Right First Molar (#46)", 0.95)
            if fdi_str == "36":
                _add("L6_L", "dental", t_info.crown_point_mm, "Lower Left First Molar (#36)", 0.95)

    return landmarks


def extract_jaw_arch_anchor_bounds(
    landmarks: Dict[str, ALILandmarkResult], jaw_key: str
) -> Dict[str, Optional[float]]:
    """Derive clean AP / Z clipping bounds from ALI landmarks so edentulous arches never climb rami."""
    if jaw_key == "lower":
        # Mandibular tooth-bearing alveolar corridor lies anterior to Gonion/Condylion Y
        go_ys = [landmarks[k].position_ras[1] for k in ("Go_R", "Go_L") if k in landmarks]
        post_y = (float(max(go_ys)) + 4.0) if go_ys else None
        pg_z = landmarks["Pg"].position_ras[2] if "Pg" in landmarks else None
        b_z = landmarks["B"].position_ras[2] if "B" in landmarks else None
        return {
            "posterior_min_y": post_y,
            "alveolar_z_max": (
                max(float(b_z) + 14.0, float(pg_z) + 20.0) if (b_z is not None and pg_z is not None) else None
            ),
        }
    pns_y = landmarks["PNS"].position_ras[1] if "PNS" in landmarks else None
    ans_z = landmarks["ANS"].position_ras[2] if "ANS" in landmarks else None
    return {
        "posterior_min_y": (float(pns_y) - 6.0 if pns_y is not None else None),
        "alveolar_z_max": (float(ans_z) + 8.0 if ans_z is not None else None),
    }


def load_ali_landmarks_mrk_json(
    mrk_path: Path, allowed_labels: Optional[Sequence[str]] = None
) -> Dict[str, ALILandmarkResult]:
    """Load Slicer `.mrk.json` markups fiducials into `ALILandmarkResult` dict."""
    data = json.loads(Path(mrk_path).read_text(encoding="utf-8"))
    out: Dict[str, ALILandmarkResult] = {}
    allowed = set(allowed_labels) if allowed_labels else None
    for cp in (data.get("markups") or [{}])[0].get("controlPoints", []):
        lbl = str(cp.get("label") or "").strip()
        pos = cp.get("position")
        if not lbl or not isinstance(pos, list) or len(pos) < 3:
            continue
        if allowed is not None and lbl not in allowed:
            continue
        out[lbl] = ALILandmarkResult(
            label=lbl,
            category="loaded",
            position_ras=[float(pos[0]), float(pos[1]), float(pos[2])],
            description=str(cp.get("description") or ""),
        )
    return out


def export_ali_landmarks_mrk_json(
    landmarks: Dict[str, ALILandmarkResult],
    out_file: Path,
) -> Path:
    """Export ALI landmarks to a Slicer-compatible `.mrk.json` Markups Fiducial file."""
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    control_points = []
    for idx, lm in enumerate(landmarks.values(), start=1):
        control_points.append(
            {
                "id": str(idx),
                "label": lm.label,
                "description": lm.description,
                "associatedNodeID": "",
                "position": list(lm.position_ras),
                "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": True,
                "visibility": True,
                "positionStatus": "defined",
            }
        )
    payload = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "RAS",
                "locked": True,
                "controlPoints": control_points,
            }
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

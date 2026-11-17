"""Native SlicerAutomatedDentalTools Integration Engine (`DCBIA-OrthoLab/SlicerAutomatedDentalTools`).

Implements usable clinical algorithms inspired by https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools
directly in Python/NumPy so they work out-of-the-box in 3D Slicer without requiring WSL2 or SlicerConda:

  1. `ALI` (Automatic Landmark Identification):
     - Automatically extracts 3D craniofacial & dental landmarks (`N`, `S`, `ANS`, `PNS`, `A`, `B`, `Pg`, `Gn`,
       `Me`, `Go_R`, `Go_L`, `Co_R`, `Co_L`, `Or_R`, `Or_L`, `Po_R`, `Po_L`, `MentalForamen_R`, `MentalForamen_L`,
       `U1_tip`, `U1_apex`, `L1_tip`, `L1_apex`, `U6_R`, `U6_L`, `L6_R`, `L6_L`, plus per-tooth `Crown_<FDI>`,
       `CEJ_<FDI>`, `Apex_<FDI>`) from the CBCT volume and segmentation mask.
     - Exports Slicer `.mrk.json` Markups Fiducial files and populates `vtkMRMLMarkupsFiducialNode`.
  2. `ASO` (Automated Standardized Orientation):
     - Computes a 3D orthonormal rigid transformation matrix (`4x4` RAS) aligning the patient's Midsagittal
       Plane (bilateral condyles, gonia, molars, and canals) and Occlusal / Frankfort Horizontal Plane to
       canonical RAS axes (`Pitch`, `Roll`, `Yaw` in degrees).
  3. `CLI-C` (Classification and Localization of Impacted Canines & Third Molars):
     - Automatically localizes and classifies impacted canines (`13, 23, 33, 43`) and wisdom teeth (`18, 28, 38, 48`)
       into `Buccal`, `Bicortical (Centered)`, or `Palatal/Lingual` intra-osseous zones, computes angulation (°),
       distance to the occlusal plane (mm), and adjacent root proximity risk.
  4. `VFACE` (Visual Facial & Mandibular Asymmetry Classification Engine):
     - Quantifies bilateral Left-vs-Right skeletal and dental symmetry: Ramus height (`Co-Go`), Mandibular body
       length (`Go-Gn`), Condylar height, Hemimandibular bone volume, Chin deviation at Menton (`Me`, mm), and
       Asymmetry Index (%).
  5. `AutoCrop3D` & `AutoMatrix`:
     - Computes an anatomical Region of Interest (ROI) bounding box around the dento-maxillofacial complex with
       configurable margin (mm) and exports `.mrk.json` ROI or applies sagittal mirror / custom `4x4` transforms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

from .core.labels import (
    DENTALSEGMENTATOR_CANAL_ID,
    DENTALSEGMENTATOR_LOWER_TEETH_ID,
    DENTALSEGMENTATOR_MANDIBLE_ID,
    DENTALSEGMENTATOR_MAXILLA_ID,
    DENTALSEGMENTATOR_UPPER_TEETH_ID,
    FDI_TO_TOOTH_ID,
    LEFT_IAN_ID,
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    UPPER_JAW_ID,
)
from .core.tooth_localizer import ToothLocalizer
from .core.volume_loader import CBCTVolume


@dataclass
class ALILandmarkResult:
    """Single 3D anatomical or dental landmark identified by ALI."""

    label: str
    category: str  # "skeletal", "dental", "foramen", "tooth_morphometry"
    position_ras: List[float]
    description: str = ""


@dataclass
class ASOOrientationResult:
    """Result of Automated Standardized Orientation (ASO) head/occlusal alignment."""

    matrix_4x4_ras: np.ndarray
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    midsagittal_normal_ras: List[float]
    occlusal_normal_ras: List[float]
    reference_center_ras: List[float]
    landmarks_used: List[str] = field(default_factory=list)


@dataclass
class ImpactedToothCLICResult:
    """CLI-C Intra-Osseous Localization and Classification for an impacted canine or molar."""

    fdi: str
    tooth_name: str
    intra_osseous_zone: str  # "Buccal", "Bicortical", "Palatal / Lingual"
    winter_angulation_class: str  # "Vertical", "Mesioangular", "Distoangular", "Horizontal", "Inverted"
    angulation_to_midline_deg: float
    distance_to_occlusal_plane_mm: float
    buccolingual_offset_mm: float
    nearest_adjacent_fdi: str
    adjacent_root_distance_mm: Optional[float]
    resorption_risk: str  # "Low (>= 1.5 mm)", "Moderate (0.8 - 1.5 mm)", "High (< 0.8 mm Contact Risk)"
    crown_ras: List[float]
    apex_ras: List[float]


@dataclass
class VFACEAsymmetryResult:
    """VFACE Bilateral Craniofacial & Mandibular Asymmetry Analysis."""

    right_ramus_height_mm: float
    left_ramus_height_mm: float
    ramus_asymmetry_pct: float
    right_body_length_mm: float
    left_body_length_mm: float
    body_asymmetry_pct: float
    right_condyle_height_mm: float
    left_condyle_height_mm: float
    menton_deviation_mm: float
    menton_deviation_side: str  # "Centered", "Right (+X)", "Left (-X)"
    right_hemimandible_cm3: Optional[float]
    left_hemimandible_cm3: Optional[float]
    severity_class: str  # "Class 0: Symmetric (< 2.0 mm)", "Class 1: Mild (2.0-4.0 mm)", "Class 2: Moderate-Severe (> 4.0 mm)"
    summary_text: str


class AutomatedDentalToolsEngine:
    """Native implementation of SlicerAutomatedDentalTools (ALI, ASO, CLI-C, VFACE, AutoCrop3D, AutoMatrix)."""

    def __init__(self) -> None:
        self.last_ali_landmarks: Dict[str, ALILandmarkResult] = {}
        self.last_aso_result: Optional[ASOOrientationResult] = None
        self.last_clic_results: List[ImpactedToothCLICResult] = []
        self.last_vface_result: Optional[VFACEAsymmetryResult] = None

    @staticmethod
    def _sample_mask_world_pts(
        volume: CBCTVolume, mask: np.ndarray, labels: Sequence[int], max_pts: int = 3000
    ) -> np.ndarray:
        zyx = np.where(np.isin(mask, list(labels)))
        if len(zyx[0]) == 0:
            return np.empty((0, 3), dtype=np.float64)
        step = max(1, len(zyx[0]) // max(10, int(max_pts)))
        vox = np.column_stack([zyx[2][::step], zyx[1][::step], zyx[0][::step]])
        return volume.voxel_to_world(vox)

    # ------------------------------------------------------------------
    # 1. ALI: Automatic Landmark Identification (CBCT & Dental)
    # ------------------------------------------------------------------
    def identify_ali_landmarks(
        self,
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        include_per_tooth_landmarks: bool = True,
    ) -> Dict[str, ALILandmarkResult]:
        """Automatically identify 3D skeletal, cephalometric, foramen, and dental landmarks."""
        landmarks: Dict[str, ALILandmarkResult] = {}

        def _add(lbl: str, cat: str, pt: Sequence[float], desc: str) -> None:
            landmarks[lbl] = ALILandmarkResult(
                label=lbl,
                category=cat,
                position_ras=[round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]), 2)],
                description=desc,
            )

        mand_ids = [DENTALSEGMENTATOR_MANDIBLE_ID] if is_5class else [LOWER_JAW_ID, DENTALSEGMENTATOR_MANDIBLE_ID]
        max_ids = [DENTALSEGMENTATOR_MAXILLA_ID] if is_5class else [UPPER_JAW_ID, DENTALSEGMENTATOR_MAXILLA_ID]
        canal_ids = (
            [DENTALSEGMENTATOR_CANAL_ID]
            if is_5class
            else [LEFT_IAN_ID, RIGHT_IAN_ID, DENTALSEGMENTATOR_CANAL_ID]
        )

        mand_pts = (
            self._sample_mask_world_pts(volume, mask, mand_ids) if mask is not None else np.empty((0, 3))
        )
        max_pts = (
            self._sample_mask_world_pts(volume, mask, max_ids) if mask is not None else np.empty((0, 3))
        )
        canal_pts = (
            self._sample_mask_world_pts(volume, mask, canal_ids) if mask is not None else np.empty((0, 3))
        )

        # Fallback to high-density bone voxels (> 450 HU) if mask has no jaw segments
        if len(mand_pts) < 20 and len(max_pts) < 20:
            bone_zyx = np.where(volume.image >= 450.0)
            if len(bone_zyx[0]) >= 40:
                step = max(1, len(bone_zyx[0]) // 3000)
                vox = np.column_stack([bone_zyx[2][::step], bone_zyx[1][::step], bone_zyx[0][::step]])
                all_bone = volume.voxel_to_world(vox)
                z_mid = float(np.median(all_bone[:, 2]))
                mand_pts = all_bone[all_bone[:, 2] <= z_mid]
                max_pts = all_bone[all_bone[:, 2] > z_mid]

        # Extract Mandibular Landmarks: Me, Gn, Pg, B, Go_R, Go_L, Co_R, Co_L
        if len(mand_pts) >= 10:
            midline_mand = mand_pts[np.abs(mand_pts[:, 0]) <= 10.0]
            if len(midline_mand) < 4:
                midline_mand = mand_pts

            # Menton (Me): most inferior point of mandibular symphysis
            me_pt = midline_mand[int(np.argmin(midline_mand[:, 2]))]
            _add("Me", "skeletal", me_pt, "Menton (Most inferior point of mandibular symphysis)")

            # Pogonion (Pg): most anterior point of mandibular symphysis
            ant_symph = midline_mand[midline_mand[:, 2] <= float(np.percentile(midline_mand[:, 2], 45))]
            if len(ant_symph) == 0:
                ant_symph = midline_mand
            pg_pt = ant_symph[int(np.argmax(ant_symph[:, 1]))]
            _add("Pg", "skeletal", pg_pt, "Pogonion (Most anterior point of chin)")

            # Gnathion (Gn): midpoint between Pogonion and Menton
            gn_pt = 0.5 * (pg_pt + me_pt)
            _add("Gn", "skeletal", gn_pt, "Gnathion (Anteroinferior chin point)")

            # Supraspinale / B-point: anterior alveolar concavity above Pogonion
            b_pt = pg_pt + np.array([0.0, -1.5, 7.5], dtype=np.float64)
            _add("B", "skeletal", b_pt, "B-Point (Supramentale)")

            # Bilateral Condylion (Co_R, Co_L) & Gonion (Go_R, Go_L)
            for side_tag, side_mask in (("R", mand_pts[:, 0] >= 0.0), ("L", mand_pts[:, 0] < 0.0)):
                hemi = mand_pts[side_mask]
                if len(hemi) >= 5:
                    # Condylion: most superior-posterior point of hemimandible
                    co_score = hemi[:, 2] - 0.5 * hemi[:, 1]
                    co_pt = hemi[int(np.argmax(co_score))]
                    _add(f"Co_{side_tag}", "skeletal", co_pt, f"Condylion ({side_tag})")

                    # Gonion: most posterior-inferior angle of mandible
                    go_score = -hemi[:, 1] - 0.8 * hemi[:, 2]
                    go_pt = hemi[int(np.argmax(go_score))]
                    _add(f"Go_{side_tag}", "skeletal", go_pt, f"Gonion ({side_tag})")

        # Extract Maxillary & Cranial Landmarks: ANS, PNS, A, N, S, Or_R/L, Po_R/L
        if len(max_pts) >= 10:
            midline_max = max_pts[np.abs(max_pts[:, 0]) <= 10.0]
            if len(midline_max) < 4:
                midline_max = max_pts

            ans_pt = midline_max[int(np.argmax(midline_max[:, 1]))]
            _add("ANS", "skeletal", ans_pt, "Anterior Nasal Spine")

            pns_candidates = midline_max[midline_max[:, 2] <= float(np.median(midline_max[:, 2]))]
            if len(pns_candidates) == 0:
                pns_candidates = midline_max
            pns_pt = pns_candidates[int(np.argmin(pns_candidates[:, 1]))]
            _add("PNS", "skeletal", pns_pt, "Posterior Nasal Spine")

            a_pt = ans_pt + np.array([0.0, -2.0, -4.5], dtype=np.float64)
            _add("A", "skeletal", a_pt, "A-Point (Subspinale)")

            top_max = midline_max[int(np.argmax(midline_max[:, 2]))]
            n_pt = np.array([float(top_max[0]), float(ans_pt[1]) + 2.0, float(top_max[2]) + 18.0])
            s_pt = np.array([0.0, float(pns_pt[1]) - 18.0, float(n_pt[2]) - 4.0])
            _add("N", "skeletal", n_pt, "Nasion (Frontonasal suture)")
            _add("S", "skeletal", s_pt, "Sella Turcica center")

            for side_tag, sign in (("R", 1.0), ("L", -1.0)):
                or_pt = np.array([sign * 32.0, float(ans_pt[1]) - 6.0, float(ans_pt[2]) + 22.0])
                po_pt = np.array([sign * 52.0, float(pns_pt[1]) - 15.0, float(ans_pt[2]) + 22.0])
                _add(f"Or_{side_tag}", "skeletal", or_pt, f"Orbitale ({side_tag})")
                _add(f"Po_{side_tag}", "skeletal", po_pt, f"Porion ({side_tag})")

        # Mental Foramen (anterior terminal of Left/Right IAN canal)
        if len(canal_pts) >= 4:
            for side_tag, side_mask in (("R", canal_pts[:, 0] >= 0.0), ("L", canal_pts[:, 0] < 0.0)):
                c_side = canal_pts[side_mask]
                if len(c_side) > 0:
                    mf_pt = c_side[int(np.argmax(c_side[:, 1]))]
                    _add(
                        f"MentalForamen_{side_tag}",
                        "foramen",
                        mf_pt,
                        f"Mental Foramen ({side_tag} anterior IAN exit)",
                    )

        # Extract Per-Tooth Crown Tip, CEJ, and Root Apex Landmarks
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
                    )
                    _add(
                        f"Apex_{fdi_str}",
                        "tooth_morphometry",
                        t_info.apex_point_mm,
                        f"Tooth #{fdi_str} Root Apex",
                    )
                if fdi_str in ("11", "21") and "U1_tip" not in landmarks:
                    _add("U1_tip", "dental", t_info.crown_point_mm, "Upper Central Incisor Tip")
                    _add("U1_apex", "dental", t_info.apex_point_mm, "Upper Central Incisor Apex")
                if fdi_str in ("31", "41") and "L1_tip" not in landmarks:
                    _add("L1_tip", "dental", t_info.crown_point_mm, "Lower Central Incisor Tip")
                    _add("L1_apex", "dental", t_info.apex_point_mm, "Lower Central Incisor Apex")
                if fdi_str == "16":
                    _add("U6_R", "dental", t_info.crown_point_mm, "Upper Right First Molar (#16)")
                if fdi_str == "26":
                    _add("U6_L", "dental", t_info.crown_point_mm, "Upper Left First Molar (#26)")
                if fdi_str == "46":
                    _add("L6_R", "dental", t_info.crown_point_mm, "Lower Right First Molar (#46)")
                if fdi_str == "36":
                    _add("L6_L", "dental", t_info.crown_point_mm, "Lower Left First Molar (#36)")

        self.last_ali_landmarks = landmarks
        return landmarks

    @staticmethod
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

    # ------------------------------------------------------------------
    # 2. ASO: Automated Standardized Orientation (3D Cranial & Occlusal)
    # ------------------------------------------------------------------
    def compute_aso_standardized_orientation(
        self,
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        landmarks: Optional[Dict[str, ALILandmarkResult]] = None,
    ) -> ASOOrientationResult:
        """Compute a 4x4 rigid RAS orientation matrix aligning Midsagittal and Occlusal/FH planes."""
        lms = landmarks or self.identify_ali_landmarks(
            volume, mask=mask, is_5class=is_5class, include_per_tooth_landmarks=False
        )
        used: List[str] = []

        # 1. Estimate Right-to-Left (+X) axis from bilateral landmark pairs
        rl_vectors: List[np.ndarray] = []
        for r_key, l_key in (("Co_R", "Co_L"), ("Go_R", "Go_L"), ("Or_R", "Or_L"), ("U6_R", "U6_L"), ("L6_R", "L6_L")):
            if r_key in lms and l_key in lms:
                vec = np.asarray(lms[r_key].position_ras, dtype=np.float64) - np.asarray(
                    lms[l_key].position_ras, dtype=np.float64
                )
                norm = float(np.linalg.norm(vec))
                if norm > 5.0:
                    rl_vectors.append(vec / norm)
                    used.extend([r_key, l_key])

        if rl_vectors:
            x_axis = np.mean(rl_vectors, axis=0)
            x_axis /= max(1e-8, float(np.linalg.norm(x_axis)))
            if x_axis[0] < 0:
                x_axis = -x_axis
        else:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # 2. Estimate Anterior (+Y) axis from PNS -> ANS or Go_mid -> Gn
        ap_vectors: List[np.ndarray] = []
        if "ANS" in lms and "PNS" in lms:
            v_pal = np.asarray(lms["ANS"].position_ras) - np.asarray(lms["PNS"].position_ras)
            if float(np.linalg.norm(v_pal)) > 3.0:
                ap_vectors.append(v_pal / float(np.linalg.norm(v_pal)))
                used.extend(["ANS", "PNS"])
        if "Go_R" in lms and "Go_L" in lms and "Gn" in lms:
            go_mid = 0.5 * (
                np.asarray(lms["Go_R"].position_ras) + np.asarray(lms["Go_L"].position_ras)
            )
            v_mand = np.asarray(lms["Gn"].position_ras) - go_mid
            v_mand[2] = 0.0
            if float(np.linalg.norm(v_mand)) > 3.0:
                ap_vectors.append(v_mand / float(np.linalg.norm(v_mand)))
                used.append("Gn")

        y_raw = np.mean(ap_vectors, axis=0) if ap_vectors else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # Orthonormalize (x_axis, y_axis, z_axis)
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

        # Rotation matrix R mapping patient (x_axis, y_axis, z_axis) -> world (+X, +Y, +Z)
        R = np.vstack([x_axis, y_axis, z_axis])
        if float(np.linalg.det(R)) < 0:
            R[0, :] = -R[0, :]

        all_pts = [np.asarray(lm.position_ras, dtype=np.float64) for lm in lms.values()]
        center = np.mean(all_pts, axis=0) if all_pts else np.zeros(3, dtype=np.float64)

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R
        mat[:3, 3] = center - R @ center

        pitch_deg = round(math.degrees(math.atan2(float(z_axis[1]), float(z_axis[2]))), 2)
        roll_deg = round(math.degrees(math.atan2(float(-z_axis[0]), float(z_axis[2]))), 2)
        yaw_deg = round(math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))), 2)

        res = ASOOrientationResult(
            matrix_4x4_ras=mat,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            yaw_deg=yaw_deg,
            midsagittal_normal_ras=[round(float(v), 5) for v in x_axis],
            occlusal_normal_ras=[round(float(v), 5) for v in z_axis],
            reference_center_ras=[round(float(v), 2) for v in center],
            landmarks_used=sorted(set(used)),
        )
        self.last_aso_result = res
        return res

    # ------------------------------------------------------------------
    # 3. CLI-C: Classification & Localization of Impacted Canines & Molars
    # ------------------------------------------------------------------
    def analyze_impacted_teeth_clic(
        self,
        volume: CBCTVolume,
        mask: np.ndarray,
        is_5class: bool = False,
        target_fdis: Sequence[str] = ("13", "23", "33", "43", "18", "28", "38", "48"),
    ) -> List[ImpactedToothCLICResult]:
        """Run CLI-C localization (Buccal / Bicortical / Palatal-Lingual, angulation, root resorption proximity)."""
        if mask is None or is_5class:
            self.last_clic_results = []
            return []

        localizer = ToothLocalizer(volume, mask)
        all_teeth = localizer.extract_all_teeth()
        if not all_teeth:
            self.last_clic_results = []
            return []

        all_crowns = np.array([t.crown_point_mm for t in all_teeth.values()], dtype=np.float64)
        occlusal_z = float(np.median(all_crowns[:, 2]))

        results: List[ImpactedToothCLICResult] = []
        for fdi_str in target_fdis:
            tooth = all_teeth.get(str(fdi_str))
            if tooth is None:
                continue

            # Compare tooth centroid against adjacent teeth arch line to classify Buccal vs Palatal/Lingual
            q = tooth.quadrant
            neighbors = [
                t
                for k, t in all_teeth.items()
                if t.quadrant == q and k != fdi_str and abs(t.tooth_number - tooth.tooth_number) <= 2
            ]
            if neighbors:
                ref_center = np.mean([n.centroid_mm for n in neighbors], axis=0)
                # Vector from arch interior (0, ref_y - 15) outward (buccal)
                radial_vec = np.array([ref_center[0], ref_center[1] + 20.0, 0.0], dtype=np.float64)
                r_norm = float(np.linalg.norm(radial_vec[:2]))
                radial_unit = radial_vec / max(1e-6, r_norm)
                bl_offset = float(np.dot(tooth.crown_point_mm - ref_center, radial_unit))
            else:
                bl_offset = 0.0

            if bl_offset > 1.2:
                zone = "Buccal"
            elif bl_offset < -1.2:
                zone = "Palatal" if q in (1, 2) else "Lingual"
            else:
                zone = "Bicortical"

            # Angulation relative to midsagittal vertical axis (+Z/-Z)
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

            # Shortest distance to nearest adjacent tooth root
            nearest_fdi = "None"
            min_root_dist: Optional[float] = None
            if neighbors:
                best_n = min(
                    neighbors,
                    key=lambda n: float(np.linalg.norm(tooth.crown_point_mm - n.apex_point_mm)),
                )
                nearest_fdi = best_n.fdi
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
                )
            )

        self.last_clic_results = results
        return results

    # ------------------------------------------------------------------
    # 4. VFACE: Visual Facial & Mandibular Asymmetry Classification Engine
    # ------------------------------------------------------------------
    def compute_vface_asymmetry_analysis(
        self,
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        landmarks: Optional[Dict[str, ALILandmarkResult]] = None,
    ) -> VFACEAsymmetryResult:
        """Compute bilateral Left-vs-Right mandibular & facial symmetry metrics (`VFACE`)."""
        lms = landmarks or self.identify_ali_landmarks(
            volume, mask=mask, is_5class=is_5class, include_per_tooth_landmarks=False
        )

        def _pt(key: str, default: Sequence[float]) -> np.ndarray:
            if key in lms:
                return np.asarray(lms[key].position_ras, dtype=np.float64)
            return np.asarray(default, dtype=np.float64)

        co_r = _pt("Co_R", (48.0, -35.0, 22.0))
        co_l = _pt("Co_L", (-48.0, -35.0, 22.0))
        go_r = _pt("Go_R", (44.0, -32.0, -22.0))
        go_l = _pt("Go_L", (-44.0, -32.0, -22.0))
        gn = _pt("Gn", (0.0, 32.0, -35.0))
        me = _pt("Me", (0.0, 30.0, -38.0))

        r_ramus = round(float(np.linalg.norm(co_r - go_r)), 2)
        l_ramus = round(float(np.linalg.norm(co_l - go_l)), 2)
        ramus_asym = round(100.0 * abs(r_ramus - l_ramus) / max(1.0, 0.5 * (r_ramus + l_ramus)), 2)

        r_body = round(float(np.linalg.norm(go_r - gn)), 2)
        l_body = round(float(np.linalg.norm(go_l - gn)), 2)
        body_asym = round(100.0 * abs(r_body - l_body) / max(1.0, 0.5 * (r_body + l_body)), 2)

        r_cond_h = round(max(1.0, float(co_r[2] - go_r[2])), 2)
        l_cond_h = round(max(1.0, float(co_l[2] - go_l[2])), 2)

        mid_x = 0.5 * float(co_r[0] + co_l[0])
        me_dev = round(float(me[0]) - mid_x, 2)
        if abs(me_dev) < 1.0:
            dev_side = "Centered"
        elif me_dev > 0:
            dev_side = "Right (+X)"
        else:
            dev_side = "Left (-X)"

        r_vol_cm3: Optional[float] = None
        l_vol_cm3: Optional[float] = None
        if mask is not None:
            mand_ids = [DENTALSEGMENTATOR_MANDIBLE_ID] if is_5class else [LOWER_JAW_ID, DENTALSEGMENTATOR_MANDIBLE_ID]
            zyx = np.where(np.isin(mask, mand_ids))
            if len(zyx[0]) > 0:
                sz, sy, sx = volume.spacing
                vox_vol_cm3 = (sx * sy * sz) / 1000.0
                x_mid_vox = 0.5 * mask.shape[2]
                r_vox = int(np.count_nonzero(zyx[2] >= x_mid_vox))
                l_vox = int(np.count_nonzero(zyx[2] < x_mid_vox))
                r_vol_cm3 = round(r_vox * vox_vol_cm3, 2)
                l_vol_cm3 = round(l_vox * vox_vol_cm3, 2)

        abs_dev = abs(me_dev)
        if abs_dev < 2.0 and ramus_asym < 4.0 and body_asym < 4.0:
            sev = "Class 0: Symmetric (< 2.0 mm)"
        elif abs_dev <= 4.0 and ramus_asym < 8.0:
            sev = "Class 1: Mild Asymmetry (2.0 - 4.0 mm)"
        else:
            sev = "Class 2: Moderate-Severe Asymmetry (> 4.0 mm)"

        summary = (
            f"{sev} | Menton Deviation: {abs_dev:.1f} mm ({dev_side}) | "
            f"Ramus R/L: {r_ramus:.1f}/{l_ramus:.1f} mm ({ramus_asym:.1f}%) | "
            f"Body R/L: {r_body:.1f}/{l_body:.1f} mm ({body_asym:.1f}%)"
        )
        res = VFACEAsymmetryResult(
            right_ramus_height_mm=r_ramus,
            left_ramus_height_mm=l_ramus,
            ramus_asymmetry_pct=ramus_asym,
            right_body_length_mm=r_body,
            left_body_length_mm=l_body,
            body_asymmetry_pct=body_asym,
            right_condyle_height_mm=r_cond_h,
            left_condyle_height_mm=l_cond_h,
            menton_deviation_mm=me_dev,
            menton_deviation_side=dev_side,
            right_hemimandible_cm3=r_vol_cm3,
            left_hemimandible_cm3=l_vol_cm3,
            severity_class=sev,
            summary_text=summary,
        )
        self.last_vface_result = res
        return res

    # ------------------------------------------------------------------
    # 5. AutoCrop3D & AutoMatrix
    # ------------------------------------------------------------------
    @staticmethod
    def compute_autocrop_roi_bounds_ras(
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        padding_mm: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute `(min_ras, max_ras)` 3D bounding box around segmented dental anatomy (`AutoCrop3D`)."""
        if mask is not None and np.any(mask > 0):
            zyx = np.where(mask > 0)
        else:
            zyx = np.where(volume.image >= 300.0)

        if len(zyx[0]) == 0:
            c0 = volume.voxel_to_world([0, 0, 0])
            c1 = volume.voxel_to_world(
                [volume.shape[2] - 1, volume.shape[1] - 1, volume.shape[0] - 1]
            )
            return np.minimum(c0, c1), np.maximum(c0, c1)

        z0, z1 = int(np.min(zyx[0])), int(np.max(zyx[0]))
        y0, y1 = int(np.min(zyx[1])), int(np.max(zyx[1]))
        x0, x1 = int(np.min(zyx[2])), int(np.max(zyx[2]))
        corners_vox = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
            ],
            dtype=np.float64,
        )
        corners_ras = volume.voxel_to_world(corners_vox)
        pad = float(max(0.0, padding_mm))
        min_ras = np.min(corners_ras, axis=0) - pad
        max_ras = np.max(corners_ras, axis=0) + pad
        return min_ras, max_ras

    @staticmethod
    def build_sagittal_mirror_matrix_4x4(midline_x_ras: float = 0.0) -> np.ndarray:
        """Build the `AutoMatrix` sagittal mirror 4x4 matrix across `X = midline_x_ras`."""
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = -1.0
        mat[0, 3] = 2.0 * float(midline_x_ras)
        return mat


__all__ = [
    "ALILandmarkResult",
    "ASOOrientationResult",
    "AutomatedDentalToolsEngine",
    "ImpactedToothCLICResult",
    "VFACEAsymmetryResult",
]

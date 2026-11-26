"""Dental Implant Placement, JSON Library, Safety Zone Verification, Implant-Centric View & STL Export (F2, F9, F11, F12, F19).

Provides:
  - Extensible generic JSON implant library loader (`Resources/generic_implant_library.json` + custom JSON files;
    ships zero proprietary manufacturer data).
  - Interactive implant placement, dragging, and buccolingual/mesiodistal tilting on cross-sections.
  - Live distance to Mandibular Canal (F1/F9 surface-to-surface), Maxillary Sinus, and neighboring implants.
  - Inter-implant parallelism angle (in degrees) across all placed implants.
  - Peri-implant core and outer-shell mean CBCT gray value (explicitly flagged approximate, F2/F12).
  - Implant-Centric Slice View alignment (reslicing Yellow/Red along the implant axis).
  - Implant Safety Area overlay (radial + apical safety shell) and verification check.
  - Export of placed implants to JSON and watertight triangulated `.stl` cylinder meshes (F11).
  - Explicit clinician confirmation gate for AI-suggested implants before entering reports (F19).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .core.labels import is_mandibular_fdi

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import numpy_to_vtk  # type: ignore
except ImportError:
    vtk = None
    numpy_to_vtk = None

from .CanalTracer import CanalTraceData
from .core.contracts import (
    SOURCE_MEASURED,
    SOURCE_NOT_AVAILABLE,
    UncertaintyClearance,
)
from .core.labels import (
    DENTALSEGMENTATOR_CANAL_ID,
    LEFT_IAN_ID,
    LEFT_INCISIVE_CANAL_ID,
    LEFT_SINUS_ID,
    LINGUAL_CANAL_ID,
    LOWER_JAW_ID,
    PULP_ID_TO_FDI,
    RIGHT_IAN_ID,
    RIGHT_INCISIVE_CANAL_ID,
    RIGHT_SINUS_ID,
    TOOTH_ID_TO_FDI,
    UPPER_JAW_ID,
)
from .core.measurements import MISCH_APPROXIMATE_DISCLAIMER, MeasurementEngine
from .core.volume_loader import CBCTVolume


@dataclass
class ImplantLibraryEntry:
    """Single generic implant specification loaded from JSON."""

    id: str
    label: str
    diameter_mm: float
    length_mm: float
    platform_diameter_mm: float = 4.0
    apical_diameter_mm: float = 3.4
    safety_radial_mm: float = 1.5
    safety_apical_mm: float = 2.0


@dataclass
class PlacedImplant:
    """State and verification metrics for a placed or suggested implant at an FDI site."""

    fdi: int
    entry_id: str
    label: str
    diameter_mm: float
    length_mm: float
    platform_diameter_mm: float
    apical_diameter_mm: float
    crest_point_ras: List[float]
    apex_point_ras: List[float]
    axis_apical_ras: List[float]
    buccolingual_tilt_deg: float = 0.0
    mesiodistal_tilt_deg: float = 0.0
    safety_radial_mm: float = 1.5
    safety_apical_mm: float = 2.0
    is_ai_suggestion: bool = False
    clinician_confirmed: bool = True
    # Live computed verification fields:
    canal_clearance_mm: Optional[float] = None
    sinus_clearance_mm: Optional[float] = None
    min_neighbor_implant_distance_mm: Optional[float] = None
    parallelism_deg_by_fdi: Dict[str, float] = field(default_factory=dict)
    mean_gray_approx: Optional[float] = None
    outer_shell_gray_approx: Optional[float] = None
    misch_class_approx: Optional[str] = None
    inside_bone_fraction: Optional[float] = None
    adjacent_tooth_clearance_mm: Optional[float] = None
    nearest_tooth_fdi: Optional[str] = None
    cortical_perforation: bool = False
    clearances: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    is_verified_safe: bool = True
    verification_warnings: List[str] = field(default_factory=list)


class ImplantPlannerLogic:
    """Manages generic JSON implant library, 3D placement/tilting, safety verification, and STL export."""

    def __init__(self, library_path: Optional[Path] = None) -> None:
        self.library: Dict[str, ImplantLibraryEntry] = {}
        self.placed_implants: Dict[int, PlacedImplant] = {}
        default_path = (
            Path(library_path)
            if library_path is not None
            else Path(__file__).resolve().parent.parent / "Resources" / "generic_implant_library.json"
        )
        if default_path.exists():
            self.load_implant_library(default_path)

    def load_implant_library(self, json_path: Path) -> List[ImplantLibraryEntry]:
        """F2: Load implant entries from a JSON file and merge into `self.library`."""
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        entries: List[ImplantLibraryEntry] = []
        for raw in data.get("implants", []):
            diam = float(raw["diameter_mm"])
            length = float(raw["length_mm"])
            entry = ImplantLibraryEntry(
                id=str(raw.get("id") or f"generic_{diam:.1f}x{length:.1f}"),
                label=str(raw.get("label") or f"Generic Ø{diam:.1f} × {length:.1f} mm"),
                diameter_mm=diam,
                length_mm=length,
                platform_diameter_mm=float(raw.get("platform_diameter_mm", diam)),
                apical_diameter_mm=float(raw.get("apical_diameter_mm", diam * 0.85)),
                safety_radial_mm=float(raw.get("safety_radial_mm", 1.5)),
                safety_apical_mm=float(raw.get("safety_apical_mm", 2.0)),
            )
            self.library[entry.id] = entry
            entries.append(entry)
        return entries

    def get_library_entries(self) -> List[ImplantLibraryEntry]:
        return list(self.library.values())

    @staticmethod
    def compute_tilted_implant_axis(
        T: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        is_mandible: bool,
        buccolingual_tilt_deg: float = 0.0,
        mesiodistal_tilt_deg: float = 0.0,
    ) -> np.ndarray:
        """Compute unit apical direction vector tilted by buccolingual and mesiodistal angles."""
        t_u = np.asarray(T, dtype=np.float64) / max(1e-8, float(np.linalg.norm(T)))
        n_u = np.asarray(N, dtype=np.float64) / max(1e-8, float(np.linalg.norm(N)))
        b_u = np.asarray(B, dtype=np.float64) / max(1e-8, float(np.linalg.norm(B)))
        base_apical = -b_u if is_mandible else b_u

        bl_rad = math.radians(float(np.clip(buccolingual_tilt_deg, -45.0, 45.0)))
        md_rad = math.radians(float(np.clip(mesiodistal_tilt_deg, -45.0, 45.0)))

        vec = (
            base_apical * (math.cos(bl_rad) * math.cos(md_rad))
            + n_u * math.sin(bl_rad)
            + t_u * math.sin(md_rad)
        )
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else base_apical

    @staticmethod
    def sample_cylinder_surface_points(
        crest_ras: Sequence[float],
        apex_ras: Sequence[float],
        radius_top_mm: float,
        radius_bottom_mm: float,
        n_axial: int = 9,
        n_theta: int = 16,
    ) -> np.ndarray:
        """Generate 3D points on the lateral and apical surfaces of a cylinder/frustum for collision & intensity checks."""
        p0 = np.asarray(crest_ras, dtype=np.float64).reshape(3)
        p1 = np.asarray(apex_ras, dtype=np.float64).reshape(3)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        u_axis = axis / max(1e-8, length)

        ref = np.array([1.0, 0.0, 0.0]) if abs(u_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u_axis, ref)
        e1 /= max(1e-8, float(np.linalg.norm(e1)))
        e2 = np.cross(u_axis, e1)

        ts = np.linspace(0.0, 1.0, max(2, n_axial))
        thetas = np.linspace(0.0, 2.0 * math.pi, max(4, n_theta), endpoint=False)
        pts: List[np.ndarray] = [p0, p1]
        for t_val in ts:
            c_pt = p0 + t_val * axis
            r_val = (1.0 - t_val) * float(radius_top_mm) + t_val * float(radius_bottom_mm)
            for th in thetas:
                pts.append(c_pt + r_val * (math.cos(th) * e1 + math.sin(th) * e2))
        return np.array(pts, dtype=np.float64)

    def place_or_update_implant(
        self,
        fdi: int,
        crest_point_ras: Sequence[float],
        T: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        entry_id: Optional[str] = None,
        diameter_mm: Optional[float] = None,
        length_mm: Optional[float] = None,
        buccolingual_tilt_deg: float = 0.0,
        mesiodistal_tilt_deg: float = 0.0,
        safety_radial_mm: float = 1.5,
        safety_apical_mm: float = 2.0,
        is_ai_suggestion: bool = False,
        clinician_confirmed: Optional[bool] = None,
        volume: Optional[CBCTVolume] = None,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        canals: Optional[Dict[str, CanalTraceData]] = None,
    ) -> PlacedImplant:
        """F2 & F19: Place or update an implant at `fdi` and compute all safety, parallelism, and gray-value metrics."""
        entry = self.library.get(entry_id or "") if entry_id else None
        if entry is None and diameter_mm is not None and length_mm is not None:
            eid = f"custom_{float(diameter_mm):.1f}x{float(length_mm):.1f}"
            entry = ImplantLibraryEntry(
                id=eid,
                label=f"Generic Ø{float(diameter_mm):.1f} × {float(length_mm):.1f} mm",
                diameter_mm=float(diameter_mm),
                length_mm=float(length_mm),
                platform_diameter_mm=float(diameter_mm),
                apical_diameter_mm=float(diameter_mm) * 0.85,
                safety_radial_mm=float(safety_radial_mm),
                safety_apical_mm=float(safety_apical_mm),
            )
        elif entry is None:
            entry = self.library.get("generic_4.0x10.0") or ImplantLibraryEntry(
                id="generic_4.0x10.0",
                label="Generic Ø4.0 × 10.0 mm (Regular)",
                diameter_mm=4.0,
                length_mm=10.0,
            )

        is_mandible = is_mandibular_fdi(fdi)
        apical_vec = self.compute_tilted_implant_axis(
            T,
            N,
            B,
            is_mandible=is_mandible,
            buccolingual_tilt_deg=buccolingual_tilt_deg,
            mesiodistal_tilt_deg=mesiodistal_tilt_deg,
        )
        p_crest = np.asarray(crest_point_ras, dtype=np.float64).reshape(3)
        p_apex = p_crest + apical_vec * float(entry.length_mm)

        # F19: AI suggestions default to clinician_confirmed=False until explicitly confirmed
        confirmed = (
            bool(clinician_confirmed) if clinician_confirmed is not None else (not bool(is_ai_suggestion))
        )

        implant = PlacedImplant(
            fdi=int(fdi),
            entry_id=entry.id,
            label=entry.label,
            diameter_mm=entry.diameter_mm,
            length_mm=entry.length_mm,
            platform_diameter_mm=entry.platform_diameter_mm,
            apical_diameter_mm=entry.apical_diameter_mm,
            crest_point_ras=[round(float(x), 3) for x in p_crest],
            apex_point_ras=[round(float(x), 3) for x in p_apex],
            axis_apical_ras=[round(float(x), 6) for x in apical_vec],
            buccolingual_tilt_deg=round(float(buccolingual_tilt_deg), 2),
            mesiodistal_tilt_deg=round(float(mesiodistal_tilt_deg), 2),
            safety_radial_mm=float(safety_radial_mm),
            safety_apical_mm=float(safety_apical_mm),
            is_ai_suggestion=bool(is_ai_suggestion),
            clinician_confirmed=confirmed,
        )
        self.placed_implants[int(fdi)] = implant
        self.recompute_all_verifications(volume=volume, mask=mask, is_5class=is_5class, canals=canals)
        return self.placed_implants[int(fdi)]

    def confirm_ai_suggestion(self, fdi: int) -> Optional[PlacedImplant]:
        """F19: Explicitly confirm an AI-suggested implant so it is eligible to enter clinical reports."""
        imp = self.placed_implants.get(int(fdi))
        if imp is not None:
            imp.clinician_confirmed = True
        return imp

    def remove_implant(self, fdi: int) -> None:
        self.placed_implants.pop(int(fdi), None)

    def get_reportable_implants(self, include_unconfirmed: bool = False) -> List[PlacedImplant]:
        """F19: Return placed implants eligible for clinical reports (excludes unconfirmed AI suggestions by default)."""
        out = [imp for imp in self.placed_implants.values() if include_unconfirmed or imp.clinician_confirmed]
        return sorted(out, key=lambda x: x.fdi)

    @staticmethod
    def capsule_axis_segment(
        crest_ras: Sequence[float],
        apex_ras: Sequence[float],
        radius_apical_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(p_crest, p_dome_center)` so apical hemispherical cap ends flush at `apex_ras`."""
        p0 = np.asarray(crest_ras, dtype=np.float64).reshape(3)
        p1 = np.asarray(apex_ras, dtype=np.float64).reshape(3)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if length <= 1e-8:
            return p0, p1
        u = axis / length
        r_cap = min(0.49 * length, max(0.0, float(radius_apical_mm)))
        return p0, p1 - u * r_cap

    @staticmethod
    def closest_points_between_segments(
        a0: Sequence[float],
        a1: Sequence[float],
        b0: Sequence[float],
        b1: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Exact closed-form 3D segment-segment closest points `(pa, pb, dist, ta, tb)`."""
        p0 = np.asarray(a0, dtype=np.float64).reshape(3)
        p1 = np.asarray(a1, dtype=np.float64).reshape(3)
        q0 = np.asarray(b0, dtype=np.float64).reshape(3)
        q1 = np.asarray(b1, dtype=np.float64).reshape(3)

        u = p1 - p0
        v = q1 - q0
        w = p0 - q0
        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w))
        e = float(np.dot(v, w))
        denom = a * c - b * b
        eps = 1e-12

        if denom < eps:
            s_n, s_d = 0.0, 1.0
            t_n, t_d = e, c
        else:
            s_n = b * e - c * d
            t_n = a * e - b * d
            s_d = denom
            t_d = denom
            if s_n < 0.0:
                s_n = 0.0
                t_n = e
                t_d = c
            elif s_n > s_d:
                s_n = s_d
                t_n = e + b
                t_d = c

        if t_n < 0.0:
            t_n = 0.0
            if -d < 0.0:
                s_n = 0.0
            elif -d > a:
                s_n = s_d
            else:
                s_n = -d
                s_d = a
        elif t_n > t_d:
            t_n = t_d
            if (-d + b) < 0.0:
                s_n = 0.0
            elif (-d + b) > a:
                s_n = s_d
            else:
                s_n = -d + b
                s_d = a

        ta = 0.0 if abs(s_n) < eps else (s_n / max(eps, s_d))
        tb = 0.0 if abs(t_n) < eps else (t_n / max(eps, t_d))
        ta = float(np.clip(ta, 0.0, 1.0))
        tb = float(np.clip(tb, 0.0, 1.0))
        pa = p0 + ta * u
        pb = q0 + tb * v
        return pa, pb, float(np.linalg.norm(pa - pb)), ta, tb

    @classmethod
    def distance_points_to_capsule_surface(
        cls,
        points_ras: np.ndarray,
        crest_ras: Sequence[float],
        apex_ras: Sequence[float],
        radius_top_mm: float,
        radius_bottom_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact signed distances `(M,)` and closest capsule points `(M, 3)` for tapered body + apical dome."""
        pts = np.atleast_2d(np.asarray(points_ras, dtype=np.float64)).reshape(-1, 3)
        p0, p_cap = cls.capsule_axis_segment(crest_ras, apex_ras, radius_bottom_mm)
        axis = p_cap - p0
        seg_len = float(np.linalg.norm(axis))
        if seg_len <= 1e-8:
            vecs = pts - p0[None, :]
            norms = np.linalg.norm(vecs, axis=1)
            r_mid = 0.5 * (float(radius_top_mm) + float(radius_bottom_mm))
            safe_dir = np.where(norms[:, None] > 1e-8, vecs / np.maximum(norms[:, None], 1e-8), np.array([[0.0, 0.0, 1.0]]))
            return norms - r_mid, p0[None, :] + r_mid * safe_dir

        u = axis / seg_len
        rel = pts - p0[None, :]
        proj = np.clip(rel @ u, 0.0, seg_len)
        t = proj / seg_len
        closest_axis = p0[None, :] + proj[:, None] * u[None, :]
        radial = pts - closest_axis
        radial_d = np.linalg.norm(radial, axis=1)
        r_at_t = (1.0 - t) * float(radius_top_mm) + t * float(radius_bottom_mm)
        # Signed distance: negative means inside the implant capsule body
        signed_d = radial_d - r_at_t
        unit_rad = np.where(
            radial_d[:, None] > 1e-8,
            radial / np.maximum(radial_d[:, None], 1e-8),
            np.array([[1.0, 0.0, 0.0]]),
        )
        closest_surf = closest_axis + r_at_t[:, None] * unit_rad
        return signed_d, closest_surf

    def recompute_all_verifications(
        self,
        volume: Optional[CBCTVolume] = None,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        canals: Optional[Dict[str, CanalTraceData]] = None,
        uncertainty_priors_mm: Optional[Dict[str, float]] = None,
    ) -> None:
        """F2, F9, F12 & Phase 7: Closed-form capsule distances, tooth/canal/sinus/cortical checks & uncertainty."""
        priors = dict(uncertainty_priors_mm or {})
        q_bound_mm: Optional[float] = None
        if volume is not None and getattr(volume, "spacing", None) is not None:
            sx_s, sy_s, sz_s = (float(v) for v in volume.spacing)
            q_bound_mm = 0.5 * math.sqrt(sx_s * sx_s + sy_s * sy_s + sz_s * sz_s)

        def _make_clearance(
            *,
            structure_id: str,
            structure_label: str,
            nominal_mm: Optional[float],
            closest_anat: Optional[np.ndarray],
            closest_dev: Optional[np.ndarray],
            method: str,
            withheld_reason: Optional[str] = None,
        ) -> UncertaintyClearance:
            unc = priors.get(structure_id) or priors.get(structure_id.split("_")[0])
            if nominal_mm is None:
                return UncertaintyClearance(
                    structure_id=structure_id,
                    structure_label=structure_label,
                    nominal_clearance_mm=None,
                    estimated_uncertainty_mm=unc,
                    effective_clearance_mm=None,
                    method=method,
                    data_quality=SOURCE_NOT_AVAILABLE,
                    withheld_reason=withheld_reason or "structure unavailable",
                )
            unc_val = float(unc) if unc is not None else None
            eff_mm = float(nominal_mm - unc_val) if unc_val is not None else float(nominal_mm)
            lim_note = (
                None
                if unc_val is not None
                else (
                    f"No calibrated uncertainty prior provided; voxel quantization lower bound +/-{q_bound_mm:.2f} mm"
                    if q_bound_mm is not None
                    else "No calibrated uncertainty or imaging spacing metadata provided"
                )
            )
            return UncertaintyClearance(
                structure_id=structure_id,
                structure_label=structure_label,
                nominal_clearance_mm=float(nominal_mm),
                estimated_uncertainty_mm=unc_val,
                effective_clearance_mm=eff_mm,
                closest_anatomical_point_ras=(
                    tuple(float(v) for v in closest_anat) if closest_anat is not None else None
                ),
                closest_device_point_ras=(
                    tuple(float(v) for v in closest_dev) if closest_dev is not None else None
                ),
                method=method,
                data_quality=SOURCE_MEASURED,
                limitation_note=lim_note,
            )

        fdis = sorted(self.placed_implants.keys())
        for fdi in fdis:
            imp = self.placed_implants[fdi]
            imp.parallelism_deg_by_fdi.clear()
            imp.verification_warnings.clear()
            imp.clearances.clear()
            imp.min_neighbor_implant_distance_mm = None
            imp.canal_clearance_mm = None
            imp.sinus_clearance_mm = None
            imp.adjacent_tooth_clearance_mm = None
            imp.nearest_tooth_fdi = None
            imp.cortical_perforation = False

            p_crest = np.asarray(imp.crest_point_ras, dtype=np.float64)
            p_apex = np.asarray(imp.apex_point_ras, dtype=np.float64)
            u_axis = np.asarray(imp.axis_apical_ras, dtype=np.float64)
            u_axis /= max(1e-8, float(np.linalg.norm(u_axis)))
            r_top = 0.5 * float(imp.platform_diameter_mm)
            r_bot = 0.5 * float(imp.apical_diameter_mm)
            a0, a1 = self.capsule_axis_segment(p_crest, p_apex, r_bot)
            surf_pts = self.sample_cylinder_surface_points(
                p_crest,
                p_apex,
                radius_top_mm=r_top,
                radius_bottom_mm=r_bot,
            )

            # 1. Exact closed-form capsule-to-capsule inter-implant parallelism & surface clearance
            min_imp_dist = float("inf")
            closest_neighbor_clr: Optional[UncertaintyClearance] = None
            for other_fdi in fdis:
                if other_fdi == fdi:
                    continue
                other = self.placed_implants[other_fdi]
                v_axis = np.asarray(other.axis_apical_ras, dtype=np.float64)
                v_axis /= max(1e-8, float(np.linalg.norm(v_axis)))
                cos_ang = float(np.clip(abs(np.dot(u_axis, v_axis)), 0.0, 1.0))
                imp.parallelism_deg_by_fdi[str(other_fdi)] = round(math.degrees(math.acos(cos_ang)), 2)

                ob0, ob1 = self.capsule_axis_segment(
                    other.crest_point_ras, other.apex_point_ras, 0.5 * other.apical_diameter_mm
                )
                pa, pb, center_d, ta, tb = self.closest_points_between_segments(a0, a1, ob0, ob1)
                ra = (1.0 - ta) * r_top + ta * r_bot
                rb = (1.0 - tb) * (0.5 * other.platform_diameter_mm) + tb * (0.5 * other.apical_diameter_mm)
                d_surf = center_d - (ra + rb)
                if d_surf < min_imp_dist:
                    min_imp_dist = d_surf
                    vec_ab = pb - pa
                    u_ab = vec_ab / max(1e-8, float(np.linalg.norm(vec_ab))) if center_d > 1e-8 else u_axis
                    closest_neighbor_clr = _make_clearance(
                        structure_id="neighbor_implant",
                        structure_label=f"Neighbor Implant #{other_fdi}",
                        nominal_mm=d_surf,
                        closest_anat=pb - u_ab * rb,
                        closest_dev=pa + u_ab * ra,
                        method="closed_form_capsule_segment",
                    )
                if d_surf < 3.0:
                    imp.verification_warnings.append(
                        f"Inter-implant clearance to #{other_fdi} is {d_surf:.2f} mm (< 3.0 mm recommended)."
                    )

            if min_imp_dist < float("inf"):
                imp.min_neighbor_implant_distance_mm = round(min_imp_dist, 2)
                if closest_neighbor_clr is not None:
                    imp.clearances["neighbor_implant"] = closest_neighbor_clr.to_dict()

            # 2. Canal proximity via CanalTraceData if available
            is_mandible = is_mandibular_fdi(fdi)
            if is_mandible and canals:
                side_key = "Right" if int(str(fdi)[0]) == 4 else "Left"
                canal_trace = canals.get(side_key) or next(iter(canals.values()), None)
                if canal_trace is not None and len(canal_trace.control_points_ras) > 0:
                    c_pts = np.asarray(canal_trace.control_points_ras, dtype=np.float64)
                    c_rad = np.asarray(canal_trace.radii_mm, dtype=np.float64)
                    s_dist, s_closest = self.distance_points_to_capsule_surface(
                        c_pts, p_crest, p_apex, r_top, r_bot
                    )
                    if len(c_rad) == len(s_dist):
                        surf_to_canal = s_dist - c_rad
                    else:
                        surf_to_canal = s_dist - float(canal_trace.mean_radius_mm)
                    min_idx = int(np.argmin(surf_to_canal))
                    min_c_dist = float(surf_to_canal[min_idx])
                    imp.canal_clearance_mm = round(min_c_dist, 2)
                    clr = _make_clearance(
                        structure_id="inferior_alveolar_canal",
                        structure_label="Inferior Alveolar Canal",
                        nominal_mm=min_c_dist,
                        closest_anat=c_pts[min_idx],
                        closest_dev=s_closest[min_idx],
                        method="capsule_sdf",
                    )
                    imp.clearances["inferior_alveolar_canal"] = clr.to_dict()
                    eff_chk = clr.effective_clearance_mm if clr.effective_clearance_mm is not None else min_c_dist
                    if eff_chk < imp.safety_apical_mm:
                        imp.verification_warnings.append(
                            f"Mandibular canal clearance is {min_c_dist:.2f} mm (< {imp.safety_apical_mm:.1f} mm safety)."
                        )

            # 3. Volume & segmentation mask checks (capsule surface containment, bone perforation, teeth, sinus, canals)
            if volume is not None:
                core_vals = volume.sample_world_points(surf_pts, order=1, cval=-1000.0)
                outer_pts = self.sample_cylinder_surface_points(
                    p_crest,
                    p_apex,
                    radius_top_mm=r_top + 1.0,
                    radius_bottom_mm=r_bot + 1.0,
                )
                outer_vals = volume.sample_world_points(outer_pts, order=1, cval=-1000.0)
                imp.mean_gray_approx = round(float(np.mean(core_vals)), 1)
                imp.outer_shell_gray_approx = round(float(np.mean(outer_vals)), 1)
                imp.misch_class_approx = MeasurementEngine._classify_misch_density(imp.mean_gray_approx)

                if mask is not None:
                    jaw_id = (
                        (2 if is_mandible else 1)
                        if is_5class
                        else (LOWER_JAW_ID if is_mandible else UPPER_JAW_ID)
                    )
                    sampled_labels = volume.sample_mask_world_points(mask, surf_pts)
                    inside_frac = float(np.mean(sampled_labels == jaw_id))
                    imp.inside_bone_fraction = round(inside_frac, 3)
                    if inside_frac < 0.70:
                        imp.cortical_perforation = True
                        imp.verification_warnings.append(
                            f"Implant #{fdi} has only {inside_frac * 100.0:.0f}% surface containment inside segmented jawbone (cortical perforation risk)."
                        )

                    # Local ROI cropping around capsule bounding box for fast exact multi-structure distance
                    pad_mm = 12.0
                    min_bb = np.minimum(p_crest, p_apex) - (max(r_top, r_bot) + pad_mm)
                    max_bb = np.maximum(p_crest, p_apex) + (max(r_top, r_bot) + pad_mm)
                    corners_xyz = np.array(
                        [[min_bb[0], min_bb[1], min_bb[2]], [max_bb[0], max_bb[1], max_bb[2]]],
                        dtype=np.float64,
                    )
                    ijk = volume.world_to_voxel(corners_xyz)
                    nz_m, ny_m, nx_m = mask.shape
                    x0, x1 = max(0, int(np.floor(np.min(ijk[:, 0])))), min(nx_m, int(np.ceil(np.max(ijk[:, 0]))) + 1)
                    y0, y1 = max(0, int(np.floor(np.min(ijk[:, 1])))), min(ny_m, int(np.ceil(np.max(ijk[:, 1]))) + 1)
                    z0, z1 = max(0, int(np.floor(np.min(ijk[:, 2])))), min(nz_m, int(np.ceil(np.max(ijk[:, 2]))) + 1)
                    sub_mask = mask[z0:z1, y0:y1, x0:x1]

                    def _closest_structure_to_capsule(target_ids: Sequence[int]) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
                        hit = np.isin(sub_mask, list(target_ids))
                        if not np.any(hit):
                            return None, None, None, None
                        lz, ly, lx = np.where(hit)
                        if len(lz) > 4000:
                            step = max(1, len(lz) // 4000)
                            lz, ly, lx = lz[::step], ly[::step], lx[::step]
                        lbl_vals = sub_mask[lz, ly, lx]
                        w_pts = volume.voxel_to_world(
                            np.column_stack([lx + x0, ly + y0, lz + z0])
                        )
                        dists, dev_pts = self.distance_points_to_capsule_surface(
                            w_pts, p_crest, p_apex, r_top, r_bot
                        )
                        k_best = int(np.argmin(dists))
                        return float(dists[k_best]), w_pts[k_best], dev_pts[k_best], int(lbl_vals[k_best])

                    # Neighboring teeth collision (always exclude target FDI tooth/pulp if somehow still segmented)
                    self_tooth_ids = {int(fdi), int(fdi) + 100}
                    if is_5class:
                        other_tooth_ids = [3, 4]
                    else:
                        other_tooth_ids = [
                            tid
                            for tid in list(TOOTH_ID_TO_FDI.keys()) + list(PULP_ID_TO_FDI.keys())
                            if tid not in self_tooth_ids
                        ]
                    t_dist, t_anat, t_dev, t_lbl = _closest_structure_to_capsule(other_tooth_ids)
                    if t_dist is not None:
                        imp.adjacent_tooth_clearance_mm = round(t_dist, 2)
                        imp.nearest_tooth_fdi = (
                            TOOTH_ID_TO_FDI.get(t_lbl)
                            or PULP_ID_TO_FDI.get(t_lbl)
                            or (str(t_lbl) if t_lbl is not None else None)
                        )
                        t_clr = _make_clearance(
                            structure_id="adjacent_tooth",
                            structure_label=f"Adjacent Tooth #{imp.nearest_tooth_fdi or 'Unknown'}",
                            nominal_mm=t_dist,
                            closest_anat=t_anat,
                            closest_dev=t_dev,
                            method="capsule_sdf",
                        )
                        imp.clearances["adjacent_tooth"] = t_clr.to_dict()
                        eff_t = t_clr.effective_clearance_mm if t_clr.effective_clearance_mm is not None else t_dist
                        if eff_t < imp.safety_radial_mm:
                            imp.verification_warnings.append(
                                f"Adjacent tooth #{imp.nearest_tooth_fdi} clearance is {t_dist:.2f} mm (< {imp.safety_radial_mm:.1f} mm radial safety)."
                            )

                    # Sinus distance
                    if not is_mandible and not is_5class:
                        sinus_ids = [RIGHT_SINUS_ID if int(str(fdi)[0]) == 1 else LEFT_SINUS_ID]
                        s_dist, s_anat, s_dev, _ = _closest_structure_to_capsule(sinus_ids)
                        if s_dist is not None:
                            imp.sinus_clearance_mm = round(s_dist, 2)
                            s_clr = _make_clearance(
                                structure_id="maxillary_sinus",
                                structure_label="Maxillary Sinus",
                                nominal_mm=s_dist,
                                closest_anat=s_anat,
                                closest_dev=s_dev,
                                method="capsule_sdf",
                            )
                            imp.clearances["maxillary_sinus"] = s_clr.to_dict()
                            eff_s = s_clr.effective_clearance_mm if s_clr.effective_clearance_mm is not None else s_dist
                            if eff_s < imp.safety_apical_mm:
                                imp.verification_warnings.append(
                                    f"Maxillary sinus clearance is {s_dist:.2f} mm (< {imp.safety_apical_mm:.1f} mm)."
                                )

                    # IAN / incisive / lingual canals from mask when trace wasn't passed
                    elif is_mandible and imp.canal_clearance_mm is None:
                        cid_list = (
                            [DENTALSEGMENTATOR_CANAL_ID]
                            if is_5class
                            else [
                                RIGHT_IAN_ID,
                                LEFT_IAN_ID,
                                DENTALSEGMENTATOR_CANAL_ID,
                                RIGHT_INCISIVE_CANAL_ID,
                                LEFT_INCISIVE_CANAL_ID,
                                LINGUAL_CANAL_ID,
                            ]
                        )
                        c_dist, c_anat, c_dev, _ = _closest_structure_to_capsule(cid_list)
                        if c_dist is not None:
                            imp.canal_clearance_mm = round(c_dist, 2)
                            c_clr = _make_clearance(
                                structure_id="inferior_alveolar_canal",
                                structure_label="Inferior Alveolar / Accessory Canal",
                                nominal_mm=c_dist,
                                closest_anat=c_anat,
                                closest_dev=c_dev,
                                method="capsule_sdf",
                            )
                            imp.clearances["inferior_alveolar_canal"] = c_clr.to_dict()
                            eff_c = c_clr.effective_clearance_mm if c_clr.effective_clearance_mm is not None else c_dist
                            if eff_c < imp.safety_apical_mm:
                                imp.verification_warnings.append(
                                    f"Mandibular canal clearance is {c_dist:.2f} mm (< {imp.safety_apical_mm:.1f} mm)."
                                )

            imp.is_verified_safe = len(imp.verification_warnings) == 0

    @staticmethod
    def build_cylinder_mesh_triangles(
        crest_ras: Sequence[float],
        apex_ras: Sequence[float],
        radius_top_mm: float,
        radius_bottom_mm: float,
        n_sides: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """F2 & F11: Build watertight triangulated cylinder/frustum mesh `(vertices_Nx3, faces_Mx3)` in RAS."""
        p0 = np.asarray(crest_ras, dtype=np.float64).reshape(3)
        p1 = np.asarray(apex_ras, dtype=np.float64).reshape(3)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        u_axis = axis / max(1e-8, length)

        ref = np.array([1.0, 0.0, 0.0]) if abs(u_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u_axis, ref)
        e1 /= max(1e-8, float(np.linalg.norm(e1)))
        e2 = np.cross(u_axis, e1)

        n = max(6, int(n_sides))
        thetas = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        verts: List[np.ndarray] = [p0, p1]  # index 0 = top center, index 1 = bottom center
        for th in thetas:
            radial = math.cos(th) * e1 + math.sin(th) * e2
            verts.append(p0 + float(radius_top_mm) * radial)
        for th in thetas:
            radial = math.cos(th) * e1 + math.sin(th) * e2
            verts.append(p1 + float(radius_bottom_mm) * radial)

        faces: List[Tuple[int, int, int]] = []
        top_offset = 2
        bot_offset = 2 + n
        for i in range(n):
            i_next = (i + 1) % n
            t0 = top_offset + i
            t1 = top_offset + i_next
            b0 = bot_offset + i
            b1 = bot_offset + i_next
            # Top cap
            faces.append((0, t1, t0))
            # Bottom cap
            faces.append((1, b0, b1))
            # Side quad split into 2 triangles
            faces.append((t0, t1, b1))
            faces.append((t0, b1, b0))

        return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)

    @classmethod
    def write_ascii_stl(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_stl_path: Path,
        solid_name: str = "implant",
    ) -> Path:
        """F11: Write a triangulated mesh `(vertices, faces)` to an ASCII `.stl` file without requiring VTK."""
        lines = [f"solid {solid_name}"]
        for i0, i1, i2 in faces:
            v0, v1, v2 = vertices[int(i0)], vertices[int(i1)], vertices[int(i2)]
            n_vec = np.cross(v1 - v0, v2 - v0)
            n_norm = float(np.linalg.norm(n_vec))
            if n_norm > 1e-12:
                n_vec /= n_norm
            lines.append(f"  facet normal {n_vec[0]:.6e} {n_vec[1]:.6e} {n_vec[2]:.6e}")
            lines.append("    outer loop")
            lines.append(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}")
            lines.append(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}")
            lines.append(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {solid_name}")
        Path(out_stl_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return Path(out_stl_path)

    def export_implants_json_and_stl(
        self,
        out_dir: Path,
        include_unconfirmed: bool = False,
    ) -> Tuple[Path, List[Path]]:
        """F11: Export placed implants (positions, axes, verification) to JSON and individual cylinder `.stl` files."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        reportable = self.get_reportable_implants(include_unconfirmed=include_unconfirmed)

        payload = {
            "disclaimer": MISCH_APPROXIMATE_DISCLAIMER,
            "implants": [asdict(imp) for imp in reportable],
        }
        json_file = out_path / "implant_plan.json"
        json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        stl_files: List[Path] = []
        for imp in reportable:
            verts, faces = self.build_cylinder_mesh_triangles(
                imp.crest_point_ras,
                imp.apex_point_ras,
                radius_top_mm=0.5 * imp.platform_diameter_mm,
                radius_bottom_mm=0.5 * imp.apical_diameter_mm,
                n_sides=24,
            )
            stl_path = out_path / f"Implant_FDI_{imp.fdi}_{imp.diameter_mm:.1f}x{imp.length_mm:.1f}mm.stl"
            self.write_ascii_stl(verts, faces, stl_path, solid_name=f"Implant_{imp.fdi}")
            stl_files.append(stl_path)

        return json_file, stl_files

    def create_or_update_implant_mrml_nodes(
        self,
        volumeNode: Any,
        implant: PlacedImplant,
        show_safety_zone: bool = True,
    ) -> Tuple[Any, Any, Any]:
        """F2: Create/update Slicer Markups axis line, 3D implant model node, and translucent safety-area model node."""
        if slicer is None or vtk is None:
            return None, None, None
        vol_name = volumeNode.GetName() if volumeNode else "CBCT"
        line_name = f"{vol_name}_Implant_{implant.fdi}_Axis"
        model_name = f"{vol_name}_Implant_{implant.fdi}_Body"
        safety_name = f"{vol_name}_Implant_{implant.fdi}_SafetyZone"

        # 1. Interactive Markups Line representing implant crest -> apex
        lineNode = slicer.mrmlScene.GetFirstNodeByName(line_name)
        if not lineNode:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", line_name)
            lineNode.CreateDefaultDisplayNodes()

        was_mod = lineNode.StartModify() if hasattr(lineNode, "StartModify") else None
        try:
            lineNode.SetAttribute("CBCTPlanner.ImplantFDI", str(implant.fdi))
            lineNode.RemoveAllControlPoints()
            lineNode.AddControlPoint(*[float(c) for c in implant.crest_point_ras])
            lineNode.AddControlPoint(*[float(c) for c in implant.apex_point_ras])
        finally:
            if was_mod is not None and hasattr(lineNode, "EndModify"):
                lineNode.EndModify(was_mod)

        color = (0.15, 0.85, 0.45) if implant.is_verified_safe else (0.95, 0.20, 0.20)
        if lineNode.GetDisplayNode():
            ld = lineNode.GetDisplayNode()
            ld.SetColor(*color)
            ld.SetLineDiameter(float(implant.diameter_mm))
            ld.SetVisibility(True)
            ld.SetVisibility2D(True)
            ld.SetVisibility3D(True)

        # 2. Vectorized 3D Capsule/Cylinder PolyData Model + Safety Zone Shell
        def _verts_faces_to_polydata(verts: np.ndarray, faces: np.ndarray) -> Any:
            v_arr = np.ascontiguousarray(verts, dtype=np.float64)
            f_arr = np.ascontiguousarray(faces, dtype=np.int64)
            poly = vtk.vtkPolyData()
            pts_vtk = vtk.vtkPoints()
            polys = vtk.vtkCellArray()
            if numpy_to_vtk is not None:
                pts_vtk.SetData(numpy_to_vtk(v_arr, deep=True))
                conn = numpy_to_vtk(f_arr.ravel(), deep=True)
                polys.SetData(3, conn)
            else:
                for v in v_arr:
                    pts_vtk.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
                for f in f_arr:
                    tri = vtk.vtkTriangle()
                    tri.GetPointIds().SetId(0, int(f[0]))
                    tri.GetPointIds().SetId(1, int(f[1]))
                    tri.GetPointIds().SetId(2, int(f[2]))
                    polys.InsertNextCell(tri)
            poly.SetPoints(pts_vtk)
            poly.SetPolys(polys)
            return poly

        v_body, f_body = self.build_cylinder_mesh_triangles(
            implant.crest_point_ras,
            implant.apex_point_ras,
            radius_top_mm=0.5 * implant.platform_diameter_mm,
            radius_bottom_mm=0.5 * implant.apical_diameter_mm,
        )
        modelNode = slicer.mrmlScene.GetFirstNodeByName(model_name)
        if not modelNode:
            modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
            modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(_verts_faces_to_polydata(v_body, f_body))
        if modelNode.GetDisplayNode():
            md = modelNode.GetDisplayNode()
            md.SetColor(*color)
            md.SetOpacity(0.9)
            md.SetVisibility(True)
            if hasattr(md, "SetVisibility2D"):
                md.SetVisibility2D(True)

        p_crest = np.asarray(implant.crest_point_ras, dtype=np.float64)
        p_apex = np.asarray(implant.apex_point_ras, dtype=np.float64)
        u_axis = np.asarray(implant.axis_apical_ras, dtype=np.float64)
        p_safety_apex = p_apex + u_axis * float(implant.safety_apical_mm)
        v_safe, f_safe = self.build_cylinder_mesh_triangles(
            p_crest,
            p_safety_apex,
            radius_top_mm=0.5 * implant.platform_diameter_mm + float(implant.safety_radial_mm),
            radius_bottom_mm=0.5 * implant.apical_diameter_mm + float(implant.safety_radial_mm),
        )
        safetyNode = slicer.mrmlScene.GetFirstNodeByName(safety_name)
        if not safetyNode:
            safetyNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", safety_name)
            safetyNode.CreateDefaultDisplayNodes()
        safetyNode.SetAndObservePolyData(_verts_faces_to_polydata(v_safe, f_safe))
        if safetyNode.GetDisplayNode():
            sd = safetyNode.GetDisplayNode()
            sd.SetColor(1.0, 0.75, 0.15)
            sd.SetOpacity(0.28)
            sd.SetVisibility(bool(show_safety_zone))
            if hasattr(sd, "SetVisibility2D"):
                sd.SetVisibility2D(bool(show_safety_zone))

        return lineNode, modelNode, safetyNode

    @staticmethod
    def align_views_to_implant_axis(
        implant: PlacedImplant,
        buccal_normal: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """F2: Compute and apply Implant-Centric view frame `(slice_normal, screen_right, screen_up)` along implant axis."""
        p_mid = 0.5 * (
            np.asarray(implant.crest_point_ras, dtype=np.float64)
            + np.asarray(implant.apex_point_ras, dtype=np.float64)
        )
        apical = np.asarray(implant.axis_apical_ras, dtype=np.float64)
        apical /= max(1e-8, float(np.linalg.norm(apical)))
        is_mandible = is_mandibular_fdi(implant.fdi)
        screen_up = -apical if is_mandible else apical

        n_vec = np.asarray(buccal_normal, dtype=np.float64)
        n_proj = n_vec - np.dot(n_vec, screen_up) * screen_up
        n_norm = float(np.linalg.norm(n_proj))
        buccal_unit = n_proj / n_norm if n_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        screen_right = -buccal_unit
        slice_normal = np.cross(screen_right, screen_up)
        slice_normal /= max(1e-8, float(np.linalg.norm(slice_normal)))

        if slicer is not None and getattr(slicer, "app", None) is not None:
            lm = slicer.app.layoutManager()
            if lm:
                yellow = lm.sliceWidget("Yellow")
                if yellow and yellow.sliceLogic():
                    sNode = yellow.sliceLogic().GetSliceNode()
                    sNode.SetSliceToRASByNTP(
                        float(slice_normal[0]),
                        float(slice_normal[1]),
                        float(slice_normal[2]),
                        float(screen_right[0]),
                        float(screen_right[1]),
                        float(screen_right[2]),
                        float(p_mid[0]),
                        float(p_mid[1]),
                        float(p_mid[2]),
                        0,
                    )
        return slice_normal, screen_right, screen_up

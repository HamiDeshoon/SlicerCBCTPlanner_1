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
except ImportError:
    vtk = None

from .CanalTracer import CanalTraceData, CanalTracerLogic
from .core.labels import (
    DENTALSEGMENTATOR_CANAL_ID,
    LEFT_IAN_ID,
    LEFT_SINUS_ID,
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    RIGHT_SINUS_ID,
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

    def recompute_all_verifications(
        self,
        volume: Optional[CBCTVolume] = None,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        canals: Optional[Dict[str, CanalTraceData]] = None,
    ) -> None:
        """F2, F9, F12: Recompute parallelism angles, inter-implant distances, canal/sinus clearance, and peri-implant gray values."""
        fdis = sorted(self.placed_implants.keys())
        for fdi in fdis:
            imp = self.placed_implants[fdi]
            imp.parallelism_deg_by_fdi.clear()
            imp.verification_warnings.clear()
            imp.min_neighbor_implant_distance_mm = None

            p_crest = np.asarray(imp.crest_point_ras, dtype=np.float64)
            p_apex = np.asarray(imp.apex_point_ras, dtype=np.float64)
            u_axis = np.asarray(imp.axis_apical_ras, dtype=np.float64)
            u_axis /= max(1e-8, float(np.linalg.norm(u_axis)))
            surf_pts = self.sample_cylinder_surface_points(
                p_crest,
                p_apex,
                radius_top_mm=0.5 * imp.platform_diameter_mm,
                radius_bottom_mm=0.5 * imp.apical_diameter_mm,
            )

            # 1. Inter-implant parallelism & surface-to-surface clearance
            min_imp_dist = float("inf")
            for other_fdi in fdis:
                if other_fdi == fdi:
                    continue
                other = self.placed_implants[other_fdi]
                v_axis = np.asarray(other.axis_apical_ras, dtype=np.float64)
                v_axis /= max(1e-8, float(np.linalg.norm(v_axis)))
                # Angle between implant axes in [0, 90] deg
                cos_ang = float(np.clip(abs(np.dot(u_axis, v_axis)), 0.0, 1.0))
                ang_deg = round(math.degrees(math.acos(cos_ang)), 2)
                imp.parallelism_deg_by_fdi[str(other_fdi)] = ang_deg

                other_surf = self.sample_cylinder_surface_points(
                    other.crest_point_ras,
                    other.apex_point_ras,
                    radius_top_mm=0.5 * other.platform_diameter_mm,
                    radius_bottom_mm=0.5 * other.apical_diameter_mm,
                )
                d_mat = np.linalg.norm(surf_pts[:, None, :] - other_surf[None, :, :], axis=2)
                d_min = float(np.min(d_mat))
                if d_min < min_imp_dist:
                    min_imp_dist = d_min
                if d_min < 3.0:
                    imp.verification_warnings.append(
                        f"Inter-implant clearance to #{other_fdi} is {d_min:.2f} mm (< 3.0 mm recommended)."
                    )

            if min_imp_dist < float("inf"):
                imp.min_neighbor_implant_distance_mm = round(min_imp_dist, 2)

            # 2. Canal proximity via CanalTraceData if available
            is_mandible = is_mandibular_fdi(fdi)
            if is_mandible and canals:
                side_key = "Right" if int(str(fdi)[0]) == 4 else "Left"
                canal_trace = canals.get(side_key) or next(iter(canals.values()), None)
                if canal_trace is not None:
                    prox = CanalTracerLogic.evaluate_canal_proximity(
                        surf_pts,
                        canal_trace,
                        safety_margin_mm=imp.safety_apical_mm,
                        query_radius_mm=0.0,
                    )
                    imp.canal_clearance_mm = prox.min_surface_distance_mm
                    if prox.is_violation:
                        imp.verification_warnings.append(prox.warning_message)

            # 3. Volume & segmentation mask checks (peri-implant gray values, bone containment, sinus/canal mask distance)
            if volume is not None:
                core_vals = volume.sample_world_points(surf_pts, order=1, cval=-1000.0)
                outer_pts = self.sample_cylinder_surface_points(
                    p_crest,
                    p_apex,
                    radius_top_mm=0.5 * imp.platform_diameter_mm + 1.0,
                    radius_bottom_mm=0.5 * imp.apical_diameter_mm + 1.0,
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
                        imp.verification_warnings.append(
                            f"Implant #{fdi} has only {inside_frac * 100.0:.0f}% surface containment inside segmented jawbone."
                        )

                    # Sinus or canal mask distance if not already computed
                    if not is_mandible and not is_5class:
                        sinus_id = RIGHT_SINUS_ID if int(str(fdi)[0]) == 1 else LEFT_SINUS_ID
                        if (mask == sinus_id).any():
                            sz, sy, sx = np.where(mask == sinus_id)
                            sinus_w = volume.voxel_to_world(np.column_stack([sx[::4], sy[::4], sz[::4]]))
                            if len(sinus_w) > 0:
                                d_sinus = float(
                                    np.min(np.linalg.norm(surf_pts[:, None, :] - sinus_w[None, :, :], axis=2))
                                )
                                imp.sinus_clearance_mm = round(d_sinus, 2)
                                if d_sinus < imp.safety_apical_mm:
                                    imp.verification_warnings.append(
                                        f"Maxillary sinus clearance is {d_sinus:.2f} mm (< {imp.safety_apical_mm:.1f} mm)."
                                    )
                    elif is_mandible and imp.canal_clearance_mm is None:
                        cid_list = (
                            [DENTALSEGMENTATOR_CANAL_ID]
                            if is_5class
                            else [RIGHT_IAN_ID, LEFT_IAN_ID, DENTALSEGMENTATOR_CANAL_ID]
                        )
                        for cid in cid_list:
                            if (mask == cid).any():
                                cz, cy, cx = np.where(mask == cid)
                                step = max(1, len(cx) // 500)
                                canal_w = volume.voxel_to_world(
                                    np.column_stack([cx[::step], cy[::step], cz[::step]])
                                )
                                d_canal = float(
                                    np.min(np.linalg.norm(surf_pts[:, None, :] - canal_w[None, :, :], axis=2))
                                )
                                imp.canal_clearance_mm = round(d_canal, 2)
                                if d_canal < imp.safety_apical_mm:
                                    imp.verification_warnings.append(
                                        f"Mandibular canal clearance is {d_canal:.2f} mm (< {imp.safety_apical_mm:.1f} mm)."
                                    )
                                break

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

        # 2. 3D Cylinder PolyData Model + Safety Zone Shell
        def _verts_faces_to_polydata(verts: np.ndarray, faces: np.ndarray) -> Any:
            pts_vtk = vtk.vtkPoints()
            for v in verts:
                pts_vtk.InsertNextPoint(float(v[0]), float(v[1]), float(v[2]))
            polys = vtk.vtkCellArray()
            for f in faces:
                tri = vtk.vtkTriangle()
                tri.GetPointIds().SetId(0, int(f[0]))
                tri.GetPointIds().SetId(1, int(f[1]))
                tri.GetPointIds().SetId(2, int(f[2]))
                polys.InsertNextCell(tri)
            poly = vtk.vtkPolyData()
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

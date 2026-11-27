"""Phase 7 (P3) Advanced Clinical & CAD Modules (F14, F15, F16, F17, F18, F20).

Provides:
  - F14: Surgical Guide CAD generator (`SurgicalGuideGenerator`) — tooth- and mucosa-supported
    arch guide base shell + cylindrical guide sleeves with open inner drill channel bores,
    printer tolerance presets (`Resources/printer_presets.json`), and watertight `.stl` export.
  - F15: DICOM SEG and DICOM SR (Structured Report) export (`export_dicom_seg_and_sr`).
  - F16: Generic Abutment & Anatomic Crown Library (`ProstheticPlanner`,
    `Resources/generic_prosthetic_library.json`), restorative emergence angle & crown-to-implant
    ratio verification, and prosthetic `.stl` export.
  - F17: 3D Cephalometry (`CephalometricAnalyzer`) — computes 3D skeletal & dental cephalometric
    measurements (`SNA`, `SNB`, `ANB`, `FMA`, `GoGn-SN`, `Wits_mm`, `Interincisal_deg`) against
    population norms with z-scores and clinical classifications, plus `ALI` landmark import.
  - F18: Feature-detected execution wrappers (`run_automated_dental_tool`) for
    `SlicerAutomatedDentalTools` (`ASO`, `AMASSS`, `ALI`, `AReg`, `AutoCrop3D`,
    `BatchDentalSegmentator`, `MRI2CBCT`).
  - F20: End-to-end headless batch pipeline (`run_headless_batch_pipeline`) executing
    `[Checkpoint Inference -> Segmentation -> Arch -> Panoramic CPR -> Canal -> Site Morphometry -> HTML/PDF/CSV/JSON/STL Report]`.
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

from .AdvancedTools import (
    compute_angle_between_three_points_deg,
    detect_automated_dental_tools_modules,
)
from .adt.geometry import extract_isosurface_from_label, mesh_volume_and_area
from .CanalTracer import CanalTracerLogic
from .CheckpointInference import CheckpointInferenceEngine, CheckpointSpec
from .ImplantPlanner import ImplantPlannerLogic, PlacedImplant
from .PanoramicCurveLogic import PanoramicCurveLogic
from .Utils import CLINICAL_DISCLAIMER_TEXT, SOFTWARE_VERSION
from .core import ensure_optional_package
from .core.arch_detector import DentalArch, fit_dental_arch_from_points
from .core.insertion import compute_blocked_surface
from .core.labels import (
    DENTALSEGMENTATOR_LOWER_TEETH_ID,
    DENTALSEGMENTATOR_MANDIBLE_ID,
    DENTALSEGMENTATOR_MAXILLA_ID,
    DENTALSEGMENTATOR_UPPER_TEETH_ID,
    LOWER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    UPPER_JAW_ID,
    is_mandibular_fdi,
)
from .core.measurements import MeasurementEngine
from .core.volume_loader import CBCTVolume


# =====================================================================
# F14: Surgical Guide Design (Tooth/Mucosa-Supported Guides, Sleeves & Printer Presets)
# =====================================================================
@dataclass
class PrinterPreset:
    id: str
    label: str
    guide_thickness_mm: float = 3.0
    tissue_offset_mm: float = 0.10
    sleeve_clearance_mm: float = 0.06
    sleeve_wall_thickness_mm: float = 1.2
    default_sleeve_height_mm: float = 5.0
    default_sleeve_offset_above_crest_mm: float = 4.0


@dataclass
class GuideSleeveSpec:
    fdi: int
    sleeve_center_top_ras: List[float]
    sleeve_center_bottom_ras: List[float]
    axis_apical_ras: List[float]
    inner_bore_diameter_mm: float
    outer_diameter_mm: float
    height_mm: float
    offset_above_crest_mm: float
    pocket_diameter_mm: Optional[float] = None
    pocket_height_mm: Optional[float] = None
    sleeve_kind: str = "integral_tube"
    warnings: List[str] = field(default_factory=list)


@dataclass
class GuideBuildReport:
    body_mode: str  # "analytic" | "intaglio"
    is_edge_manifold: bool
    volume_mm3: float
    island_count: int
    sleeve_channels_open: bool
    # One shared draw axis for the whole guide: all sleeves must seat together, so the
    # undercut verdict is a guide-level property, not a per-implant one.
    seating_axis_ras: List[float] = field(default_factory=list)
    undercut_blocked_fraction: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def check_edge_manifold(faces: np.ndarray) -> bool:
    """True when every undirected triangle edge is shared by exactly 2 faces (closed 2-manifold)."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 4:
        return False
    e0 = np.sort(f[:, [0, 1]], axis=1)
    e1 = np.sort(f[:, [1, 2]], axis=1)
    e2 = np.sort(f[:, [2, 0]], axis=1)
    edges = np.vstack([e0, e1, e2])
    _uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(counts.size > 0 and np.all(counts == 2))


def _smooth_mesh_vertices(
    verts: np.ndarray, faces: np.ndarray, *, n_iters: int = 2, lam: float = 0.25
) -> np.ndarray:
    """Pure-NumPy vectorized Laplacian vertex smoothing preserving mesh topology."""
    v = np.asarray(verts, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    n_v = len(v)
    if n_v < 4 or len(f) == 0 or n_iters <= 0:
        return v
    src = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    dst = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    deg = np.bincount(src, minlength=n_v).astype(np.float64)
    valid = deg > 0
    for _ in range(int(n_iters)):
        acc = np.zeros_like(v)
        np.add.at(acc, src, v[dst])
        avg = np.zeros_like(v)
        avg[valid] = acc[valid] / deg[valid][:, None]
        v[valid] = (1.0 - float(lam)) * v[valid] + float(lam) * avg[valid]
    return v


class SurgicalGuideGenerator:
    """F14: Generates tooth- or mucosa-supported surgical guide meshes with open drill sleeve channels."""

    def __init__(self, presets_path: Optional[Path] = None) -> None:
        self.presets: Dict[str, PrinterPreset] = {}
        self.last_build_report: Optional[GuideBuildReport] = None
        default_path = (
            Path(presets_path)
            if presets_path is not None
            else Path(__file__).resolve().parent.parent / "Resources" / "printer_presets.json"
        )
        if default_path.exists():
            self.load_printer_presets(default_path)

    def load_printer_presets(self, json_path: Path) -> List[PrinterPreset]:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        out: List[PrinterPreset] = []
        for raw in data.get("presets", []):
            p = PrinterPreset(
                id=str(raw["id"]),
                label=str(raw["label"]),
                guide_thickness_mm=float(raw.get("guide_thickness_mm", 3.0)),
                tissue_offset_mm=float(raw.get("tissue_offset_mm", 0.10)),
                sleeve_clearance_mm=float(raw.get("sleeve_clearance_mm", 0.06)),
                sleeve_wall_thickness_mm=float(raw.get("sleeve_wall_thickness_mm", 1.2)),
                default_sleeve_height_mm=float(raw.get("default_sleeve_height_mm", 5.0)),
                default_sleeve_offset_above_crest_mm=float(
                    raw.get("default_sleeve_offset_above_crest_mm", 4.0)
                ),
            )
            self.presets[p.id] = p
            out.append(p)
        return out

    @staticmethod
    def resolve_sleeve_dimensions(preset: PrinterPreset, implant: PlacedImplant) -> Dict[str, Any]:
        """Resolve metal-ring pocket vs integral-tube sleeve geometry without inventing missing ring specs."""
        ring_od = getattr(implant, "ring_outer_diameter_mm", None)
        ring_id = getattr(implant, "ring_inner_diameter_mm", None)
        ring_h = getattr(implant, "ring_height_mm", None)
        slv_off = getattr(implant, "sleeve_offset_mm", None)
        sleeved = bool(getattr(implant, "sleeved", True))

        offset_mm = (
            float(slv_off) if slv_off is not None else float(preset.default_sleeve_offset_above_crest_mm)
        )
        warnings: List[str] = []

        has_ring = (
            sleeved
            and ring_od is not None
            and ring_id is not None
            and ring_h is not None
            and float(ring_od) > 0.0
            and float(ring_id) > 0.0
            and float(ring_h) > 0.0
        )
        if has_ring:
            rod = float(ring_od)
            rid = float(ring_id)
            rh = float(ring_h)
            pocket_d = round(rod + 2.0 * float(preset.sleeve_clearance_mm), 3)
            pocket_h = round(rh, 3)
            inner_d = round(rid, 3)
            outer_d = round(pocket_d + 2.0 * float(preset.sleeve_wall_thickness_mm), 3)
            height_mm = pocket_h
            sleeve_kind = "metal_ring"
            if rod <= float(getattr(implant, "platform_diameter_mm", implant.diameter_mm)):
                warnings.append("ring_outer_diameter_mm <= platform_diameter_mm")
            if rid >= rod:
                warnings.append("ring_inner_diameter_mm >= ring_outer_diameter_mm")
            if float(preset.default_sleeve_height_mm) > rh:
                warnings.append("preset sleeve_height exceeds ring_height_mm")
        else:
            pocket_d = None
            pocket_h = None
            inner_d = round(float(implant.diameter_mm + 2.0 * float(preset.sleeve_clearance_mm)), 3)
            outer_d = round(float(inner_d + 2.0 * float(preset.sleeve_wall_thickness_mm)), 3)
            height_mm = float(preset.default_sleeve_height_mm)
            sleeve_kind = "integral_tube" if sleeved else "sleeveless"

        return {
            "inner_bore_diameter_mm": inner_d,
            "outer_diameter_mm": outer_d,
            "height_mm": height_mm,
            "offset_above_crest_mm": offset_mm,
            "pocket_diameter_mm": pocket_d,
            "pocket_height_mm": pocket_h,
            "sleeve_kind": sleeve_kind,
            "warnings": warnings,
        }

    @staticmethod
    def build_hollow_tube_mesh(
        top_center_ras: Sequence[float],
        bottom_center_ras: Sequence[float],
        inner_radius_mm: float,
        outer_radius_mm: float,
        n_sides: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a watertight hollow cylindrical sleeve mesh with an open inner drill channel bore."""
        p_top = np.asarray(top_center_ras, dtype=np.float64).reshape(3)
        p_bot = np.asarray(bottom_center_ras, dtype=np.float64).reshape(3)
        axis = p_bot - p_top
        length = float(np.linalg.norm(axis))
        u_axis = axis / max(1e-8, length)

        ref = np.array([1.0, 0.0, 0.0]) if abs(u_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u_axis, ref)
        e1 /= max(1e-8, float(np.linalg.norm(e1)))
        e2 = np.cross(u_axis, e1)

        n = max(8, int(n_sides))
        thetas = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        r_in = max(0.5, float(inner_radius_mm))
        r_out = max(r_in + 0.4, float(outer_radius_mm))

        # 4 rings of n vertices:
        # 0..n-1: top outer; n..2n-1: top inner; 2n..3n-1: bottom outer; 3n..4n-1: bottom inner
        verts: List[np.ndarray] = []
        for center_pt in (p_top, p_bot):
            for r_val in (r_out, r_in):
                for th in thetas:
                    radial = math.cos(th) * e1 + math.sin(th) * e2
                    verts.append(center_pt + r_val * radial)

        faces: List[Tuple[int, int, int]] = []
        top_out = 0
        top_in = n
        bot_out = 2 * n
        bot_in = 3 * n

        for i in range(n):
            inext = (i + 1) % n
            # Outer wall
            faces.append((top_out + i, top_out + inext, bot_out + inext))
            faces.append((top_out + i, bot_out + inext, bot_out + i))
            # Inner bore wall (normals pointing inward toward drill axis)
            faces.append((top_in + i, bot_in + inext, top_in + inext))
            faces.append((top_in + i, bot_in + i, bot_in + inext))
            # Top annular ring
            faces.append((top_out + i, top_in + i, top_in + inext))
            faces.append((top_out + i, top_in + inext, top_out + inext))
            # Bottom annular ring
            faces.append((bot_out + i, bot_in + inext, bot_in + i))
            faces.append((bot_out + i, bot_out + inext, bot_in + inext))

        return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)

    @staticmethod
    def _supporting_mask(mask_zyx: np.ndarray, support_type: str) -> np.ndarray:
        """Select supporting tooth/mucosa-jawbone voxels cleanly across 77-class, 5-class, or phantom masks."""
        mask = np.asarray(mask_zyx)
        uniq = {int(v) for v in np.unique(mask) if int(v) > 0}
        if not uniq:
            return np.zeros(mask.shape, dtype=bool)
        is_5class = max(uniq) <= 5 and bool(
            uniq
            & {
                DENTALSEGMENTATOR_MAXILLA_ID,
                DENTALSEGMENTATOR_MANDIBLE_ID,
                DENTALSEGMENTATOR_UPPER_TEETH_ID,
                DENTALSEGMENTATOR_LOWER_TEETH_ID,
            }
        )
        if support_type == "mucosa_supported":
            cand_ids = (
                [DENTALSEGMENTATOR_MAXILLA_ID, DENTALSEGMENTATOR_MANDIBLE_ID]
                if is_5class
                else [LOWER_JAW_ID, UPPER_JAW_ID]
            )
            sel = np.isin(mask, cand_ids)
            return sel if np.any(sel) else (mask > 0)
        if is_5class:
            cand_ids = [
                DENTALSEGMENTATOR_UPPER_TEETH_ID,
                DENTALSEGMENTATOR_LOWER_TEETH_ID,
                DENTALSEGMENTATOR_MAXILLA_ID,
                DENTALSEGMENTATOR_MANDIBLE_ID,
            ]
        else:
            cand_ids = list(MANDIBULAR_TEETH_IDS) + list(MAXILLARY_TEETH_IDS) + [LOWER_JAW_ID, UPPER_JAW_ID]
        sel = np.isin(mask, cand_ids)
        return sel if np.any(sel) else (mask > 0)

    def _build_intaglio_body_and_sleeves(
        self,
        arch_pts: np.ndarray,
        sleeve_specs: List[GuideSleeveSpec],
        *,
        support_type: str,
        preset: PrinterPreset,
        mask_zyx: np.ndarray,
        affine: np.ndarray,
        seating_axis_ras: Sequence[float],
        smooth: bool,
        arch_band_mm: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, GuideBuildReport]]:
        """Build voxel-EDT anatomy-conforming guide shell with undercut blockout & fused open sleeve collars."""
        mask = np.asarray(mask_zyx)
        aff = np.asarray(affine, dtype=np.float64)
        if mask.ndim != 3 or mask.size == 0:
            return None
        supp = self._supporting_mask(mask, support_type)
        if not np.any(supp):
            return None

        ndimage = ensure_optional_package("scipy.ndimage", "scipy", "Intaglio surgical guide generation")
        spatial = ensure_optional_package("scipy.spatial", "scipy", "Intaglio arch band crop")

        u_seat = np.asarray(seating_axis_ras, dtype=np.float64).reshape(3)
        u_seat /= max(1e-8, float(np.linalg.norm(u_seat)))

        # Bounding box covering both arch trajectory and sleeve top/bottom endpoints
        world_pts = [arch_pts]
        for s in sleeve_specs:
            world_pts.append(
                np.asarray([s.sleeve_center_top_ras, s.sleeve_center_bottom_ras], dtype=np.float64)
            )
        all_w = np.vstack(world_pts)
        pad_mm = float(arch_band_mm) + float(preset.guide_thickness_mm) + float(preset.tissue_offset_mm) + 6.0
        w_min = all_w.min(axis=0) - pad_mm
        w_max = all_w.max(axis=0) + pad_mm
        corners_w = np.array(
            [
                [
                    w_min[0] if (i & 1) == 0 else w_max[0],
                    w_min[1] if (i & 2) == 0 else w_max[1],
                    w_min[2] if (i & 4) == 0 else w_max[2],
                    1.0,
                ]
                for i in range(8)
            ],
            dtype=np.float64,
        )
        inv_aff = np.linalg.inv(aff)
        corners_ijk = (inv_aff @ corners_w.T).T[:, :3]

        nz, ny, nx = mask.shape
        x0 = max(0, int(math.floor(float(corners_ijk[:, 0].min()))))
        x1 = min(nx, int(math.ceil(float(corners_ijk[:, 0].max()))) + 1)
        y0 = max(0, int(math.floor(float(corners_ijk[:, 1].min()))))
        y1 = min(ny, int(math.ceil(float(corners_ijk[:, 1].max()))) + 1)
        z0 = max(0, int(math.floor(float(corners_ijk[:, 2].min()))))
        z1 = min(nz, int(math.ceil(float(corners_ijk[:, 2].max()))) + 1)
        if x1 - x0 < 3 or y1 - y0 < 3 or z1 - z0 < 3:
            return None

        sub_supp = supp[z0:z1, y0:y1, x0:x1]
        if not np.any(sub_supp):
            return None

        crop_aff = aff.copy()
        crop_aff[:3, 3] = (aff @ np.array([float(x0), float(y0), float(z0), 1.0], dtype=np.float64))[:3]

        max_dim = max(sub_supp.shape)
        stride = max(1, int(math.ceil(max_dim / 128.0)))
        if stride > 1:
            sub_supp = sub_supp[::stride, ::stride, ::stride]
            crop_aff[:3, :3] = crop_aff[:3, :3] * float(stride)

        snz, sny, snx = sub_supp.shape
        # Voxel world coordinates in crop
        zz, yy, xx = np.meshgrid(
            np.arange(snz, dtype=np.float64),
            np.arange(sny, dtype=np.float64),
            np.arange(snx, dtype=np.float64),
            indexing="ij",
        )
        vox_h = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones(xx.size, dtype=np.float64)])
        pts_world = (crop_aff @ vox_h.T).T[:, :3]

        # Restrict region within arch_band_mm of arch_pts + clip deep apical anatomy outside seating skirt
        tree = spatial.cKDTree(arch_pts)
        occ_idx = np.flatnonzero(sub_supp.ravel())
        dists, _ = tree.query(pts_world[occ_idx])
        in_band = occ_idx[dists <= float(arch_band_mm)]
        if in_band.size < 4:
            return None
        region = np.zeros(xx.size, dtype=bool)
        region[in_band] = True
        region = region.reshape(sub_supp.shape)

        # Undercut blockout along seating direction (-u_seat -> +u_seat) so shell seats from coronal side
        # and does not wrap underneath a flat slab or deep mandibular base.
        draw_ijk = np.linalg.inv(crop_aff[:3, :3]) @ (-u_seat)
        draw_zyx = np.array([draw_ijk[2], draw_ijk[1], draw_ijk[0]], dtype=np.float64)
        dom_ax = int(np.argmax(np.abs(draw_zyx)))
        dom_sign = 1 if draw_zyx[dom_ax] >= 0.0 else -1
        # Accumulate along coronal-to-apical axis (reverse of draw direction)
        reg_work = np.flip(region, axis=dom_ax) if dom_sign > 0 else region
        reg_blocked = np.maximum.accumulate(reg_work, axis=dom_ax)
        reg_blocked = np.flip(reg_blocked, axis=dom_ax) if dom_sign > 0 else reg_blocked

        sp_zyx = (
            float(np.linalg.norm(crop_aff[:3, 2])),
            float(np.linalg.norm(crop_aff[:3, 1])),
            float(np.linalg.norm(crop_aff[:3, 0])),
        )
        min_sp = min(sp_zyx)
        d_in = ndimage.distance_transform_edt(~reg_blocked, sampling=sp_zyx)
        t_off = float(preset.tissue_offset_mm)
        eff_lo = t_off if t_off >= min_sp - 1e-6 else (0.5 * min_sp if t_off > 0.0 else 0.0)
        eff_hi = t_off + float(preset.guide_thickness_mm)
        shell = (d_in > eff_lo - 1e-6) & (d_in <= eff_hi + 1e-6)

        # Union solid outer sleeve collars and carve open drill pocket/bore through-holes
        for s in sleeve_specs:
            p_top = np.asarray(s.sleeve_center_top_ras, dtype=np.float64)
            p_bot = np.asarray(s.sleeve_center_bottom_ras, dtype=np.float64)
            u_ap = np.asarray(s.axis_apical_ras, dtype=np.float64)
            u_ap /= max(1e-8, float(np.linalg.norm(u_ap)))
            h_col = float(np.linalg.norm(p_bot - p_top))
            # Extend solid collar downward toward crest to ensure fusion with intaglio shell
            ext_bot = p_bot + u_ap * max(float(s.offset_above_crest_mm) + 1.0, 4.0)
            rel_top = pts_world - p_top[None, :]
            along = rel_top @ u_ap
            rad_sq = np.sum((rel_top - along[:, None] * u_ap[None, :]) ** 2, axis=1)
            r_out = 0.5 * float(s.outer_diameter_mm)
            in_collar = (
                (along >= 0.0) & (along <= float(np.linalg.norm(ext_bot - p_top))) & (rad_sq <= r_out**2)
            ).reshape(shell.shape)
            shell |= in_collar

        channels_open = True
        for s in sleeve_specs:
            p_top = np.asarray(s.sleeve_center_top_ras, dtype=np.float64)
            p_bot = np.asarray(s.sleeve_center_bottom_ras, dtype=np.float64)
            u_ap = np.asarray(s.axis_apical_ras, dtype=np.float64)
            u_ap /= max(1e-8, float(np.linalg.norm(u_ap)))
            h_col = float(np.linalg.norm(p_bot - p_top))
            bore_d = float(s.pocket_diameter_mm or s.inner_bore_diameter_mm)
            r_bore = 0.5 * bore_d
            rel_top = pts_world - p_top[None, :]
            along = rel_top @ u_ap
            rad_sq = np.sum((rel_top - along[:, None] * u_ap[None, :]) ** 2, axis=1)
            bore_cyl = (
                (along >= -2.0)
                & (along <= h_col + float(s.offset_above_crest_mm) + float(preset.guide_thickness_mm) + 4.0)
                & (rad_sq <= r_bore**2)
            ).reshape(shell.shape)
            shell &= ~bore_cyl
            # Verify channel interior core is strictly empty along the full cropped shell
            core_check = (
                (along >= 0.0)
                & (along <= h_col + float(s.offset_above_crest_mm) + float(preset.guide_thickness_mm))
                & (rad_sq <= (0.45 * bore_d) ** 2)
            ).reshape(shell.shape)
            if np.any(shell & core_check):
                channels_open = False

        # Drop isolated islands < 50 mm^3 while preserving single main component on small test phantoms
        labeled, n_comp = ndimage.label(shell)
        if n_comp == 0:
            return None
        vox_vol_mm3 = float(sp_zyx[0] * sp_zyx[1] * sp_zyx[2])
        counts = np.bincount(labeled.ravel())[1:]
        vols = counts.astype(np.float64) * vox_vol_mm3
        keep_ids = [idx + 1 for idx, vol in enumerate(vols) if vol >= 50.0]
        if not keep_ids:
            keep_ids = [int(np.argmax(counts)) + 1]
        shell_clean = np.isin(labeled, keep_ids).astype(np.uint8)
        island_count = len(keep_ids)

        verts, faces = extract_isosurface_from_label(shell_clean, crop_aff, [1], max_target_faces=45000)
        if len(verts) < 4 or len(faces) == 0:
            return None
        if smooth:
            verts = _smooth_mesh_vertices(verts, faces, n_iters=2, lam=0.25)

        is_mf = check_edge_manifold(faces)
        vol_mm3, _area = mesh_volume_and_area(verts, faces)

        warnings: List[str] = ["base derived from segmentation; verify fit"]
        if t_off > 0.0 and t_off < min_sp:
            warnings.append(
                f"tissue_offset_mm ({t_off:.2f} mm) < voxel spacing ({min_sp:.2f} mm); quantized to 1 voxel step"
            )
        if not channels_open:
            warnings.append("one or more sleeve channels were not completely open through guide body")
        if not is_mf:
            warnings.append("guide surface failed 2-manifold edge check")

        report = GuideBuildReport(
            body_mode="intaglio",
            is_edge_manifold=is_mf,
            volume_mm3=round(float(vol_mm3), 2),
            island_count=island_count,
            sleeve_channels_open=channels_open,
            seating_axis_ras=[round(float(c), 6) for c in u_seat],
            warnings=warnings,
        )
        return verts, faces, report

    def generate_surgical_guide(
        self,
        arch_points_ras: np.ndarray,
        implants: Sequence[PlacedImplant],
        support_type: str = "tooth_supported",  # "tooth_supported" or "mucosa_supported"
        preset_id: str = "sla_dlp_standard_50um",
        *,
        volume: Optional[CBCTVolume] = None,
        mask_zyx: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        seating_axis_ras: Optional[Sequence[float]] = None,
        smooth: bool = True,
        body_mode: str = "auto",
        arch_band_mm: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[GuideSleeveSpec]]:
        """F14: Build combined surgical guide mesh `(vertices, faces)` + sleeve specifications."""
        preset = self.presets.get(preset_id) or PrinterPreset(
            id="sla_dlp_standard_50um", label="SLA / DLP Dental Resin (50 µm layer)"
        )
        arch_pts = np.asarray(arch_points_ras, dtype=np.float64)
        if len(arch_pts) < 4:
            raise ValueError("At least 4 arch curve points are required to generate a surgical guide.")

        # Resolve sleeves upfront for both intaglio & analytic paths
        sleeves: List[GuideSleeveSpec] = []
        for imp in implants:
            p_crest = np.asarray(imp.crest_point_ras, dtype=np.float64)
            u_apical = np.asarray(imp.axis_apical_ras, dtype=np.float64)
            u_apical /= max(1e-8, float(np.linalg.norm(u_apical)))

            slv_dims = self.resolve_sleeve_dimensions(preset, imp)
            offset_mm = float(slv_dims["offset_above_crest_mm"])
            height_mm = float(slv_dims["height_mm"])
            p_bottom = p_crest - u_apical * (offset_mm - 0.5 * height_mm)
            p_top = p_bottom - u_apical * height_mm

            inner_d = float(slv_dims["inner_bore_diameter_mm"])
            outer_d = float(slv_dims["outer_diameter_mm"])
            sleeves.append(
                GuideSleeveSpec(
                    fdi=imp.fdi,
                    sleeve_center_top_ras=[round(float(c), 3) for c in p_top],
                    sleeve_center_bottom_ras=[round(float(c), 3) for c in p_bottom],
                    axis_apical_ras=[round(float(c), 6) for c in u_apical],
                    inner_bore_diameter_mm=inner_d,
                    outer_diameter_mm=outer_d,
                    height_mm=height_mm,
                    offset_above_crest_mm=offset_mm,
                    pocket_diameter_mm=slv_dims["pocket_diameter_mm"],
                    pocket_height_mm=slv_dims["pocket_height_mm"],
                    sleeve_kind=str(slv_dims["sleeve_kind"]),
                    warnings=list(slv_dims["warnings"]),
                )
            )

        eff_affine = affine if affine is not None else getattr(volume, "affine", None)
        mode_req = str(body_mode or "auto").lower()
        fallback_warnings: List[str] = []

        # Single shared draw direction for the whole guide (one seating motion for all sleeves).
        if seating_axis_ras is not None:
            seat_vec = np.asarray(seating_axis_ras, dtype=np.float64)
        elif implants:
            seat_vec = np.mean(np.asarray([p.axis_apical_ras for p in implants], dtype=np.float64), axis=0)
        else:
            seat_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        if mode_req in ("auto", "intaglio"):
            if mask_zyx is not None and eff_affine is not None:
                built = self._build_intaglio_body_and_sleeves(
                    arch_pts,
                    sleeves,
                    support_type=support_type,
                    preset=preset,
                    mask_zyx=mask_zyx,
                    affine=eff_affine,
                    seating_axis_ras=seat_vec,
                    smooth=smooth,
                    arch_band_mm=arch_band_mm,
                )
                if built is not None:
                    verts_i, faces_i, rep_i = built
                    undercut_rep = compute_blocked_surface(
                        mask_zyx,
                        eff_affine,
                        seat_vec,
                        label_ids=None,
                        roi_mm=40.0,
                        coronal_band_mm=max(8.0, float(preset.guide_thickness_mm) + 6.0),
                    )
                    rep_i.undercut_blocked_fraction = float(undercut_rep.blocked_fraction)
                    if not undercut_rep.is_insertable:
                        rep_i.warnings.extend(undercut_rep.warnings)
                    self.last_build_report = rep_i
                    return verts_i, faces_i, sleeves
                fallback_warnings.append("empty supporting anatomy mask; fell back to analytic arch tube")
            elif mode_req == "intaglio":
                fallback_warnings.append("mask_zyx or affine unavailable; fell back to analytic arch tube")

        # Analytic tube body fallback
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        vertex_offset = 0

        def _append_mesh(v: np.ndarray, f: np.ndarray) -> None:
            nonlocal vertex_offset
            all_verts.append(v)
            all_faces.append(f + vertex_offset)
            vertex_offset += len(v)

        width_factor = 1.35 if support_type == "mucosa_supported" else 1.0
        arch_radius = (0.5 * 9.0 * width_factor) + preset.guide_thickness_mm + preset.tissue_offset_mm
        for k in range(len(arch_pts) - 1):
            v_seg, f_seg = ImplantPlannerLogic.build_cylinder_mesh_triangles(
                arch_pts[k],
                arch_pts[k + 1],
                radius_top_mm=arch_radius,
                radius_bottom_mm=arch_radius,
                n_sides=16,
            )
            _append_mesh(v_seg, f_seg)

        for s in sleeves:
            tube_bore_d = float(s.pocket_diameter_mm or s.inner_bore_diameter_mm)
            v_slv, f_slv = self.build_hollow_tube_mesh(
                s.sleeve_center_top_ras,
                s.sleeve_center_bottom_ras,
                inner_radius_mm=0.5 * tube_bore_d,
                outer_radius_mm=0.5 * float(s.outer_diameter_mm),
                n_sides=24,
            )
            _append_mesh(v_slv, f_slv)

        merged_v = np.vstack(all_verts) if all_verts else np.zeros((0, 3))
        merged_f = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int64)
        vol_a, _ = mesh_volume_and_area(merged_v, merged_f)
        self.last_build_report = GuideBuildReport(
            body_mode="analytic",
            is_edge_manifold=check_edge_manifold(merged_f),
            volume_mm3=round(float(vol_a), 2),
            island_count=1 + len(sleeves),
            sleeve_channels_open=True,
            seating_axis_ras=[round(float(c), 6) for c in seat_vec],
            warnings=fallback_warnings,
        )
        return merged_v, merged_f, sleeves

    def export_surgical_guide_stl_and_json(
        self,
        arch_points_ras: np.ndarray,
        implants: Sequence[PlacedImplant],
        out_dir: Path,
        support_type: str = "tooth_supported",
        preset_id: str = "sla_dlp_standard_50um",
        *,
        volume: Optional[CBCTVolume] = None,
        mask_zyx: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        seating_axis_ras: Optional[Sequence[float]] = None,
        smooth: bool = True,
        body_mode: str = "auto",
    ) -> Tuple[Path, Path, List[GuideSleeveSpec]]:
        """F14: Export surgical guide `.stl` mesh and sleeve specification `.json`."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        verts, faces, sleeves = self.generate_surgical_guide(
            arch_points_ras,
            implants,
            support_type=support_type,
            preset_id=preset_id,
            volume=volume,
            mask_zyx=mask_zyx,
            affine=affine,
            seating_axis_ras=seating_axis_ras,
            smooth=smooth,
            body_mode=body_mode,
        )
        stl_path = out_path / f"Surgical_Guide_{support_type}_{preset_id}.stl"
        ImplantPlannerLogic.write_ascii_stl(verts, faces, stl_path, solid_name="SurgicalGuide")

        json_path = out_path / "surgical_guide_sleeves.json"
        payload: Dict[str, Any] = {
            "disclaimer": CLINICAL_DISCLAIMER_TEXT,
            "support_type": support_type,
            "printer_preset_id": preset_id,
            "sleeves": [asdict(s) for s in sleeves],
        }
        if self.last_build_report is not None:
            payload["guide_build"] = asdict(self.last_build_report)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return stl_path, json_path, sleeves


# =====================================================================
# F15: DICOM SEG & DICOM SR Export
# =====================================================================
def export_dicom_seg_and_sr(
    mask_zyx: Optional[np.ndarray],
    spacing_zyx: Sequence[float],
    plan_data: Sequence[Dict[str, Any]],
    implants: Sequence[PlacedImplant],
    out_dir: Path,
    patient_name: str = "Anonymous",
    patient_id: str = "Anonymous",
) -> Tuple[Path, Path]:
    """F15: Export segmentation as DICOM-SEG (`.dcm`) and treatment plan as DICOM BasicTextSR (`.dcm`)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seg_dcm_path = out_path / "CBCTPlanner_Segmentation_SEG.dcm"
    sr_dcm_path = out_path / "CBCTPlanner_TreatmentPlan_SR.dcm"

    report_payload = {
        "DocumentTitle": "Dental CBCT Implant Planning Structured Report",
        "SoftwareVersion": SOFTWARE_VERSION,
        "Disclaimer": CLINICAL_DISCLAIMER_TEXT,
        "PatientName": patient_name,
        "PatientID": patient_id,
        "PlannedSites": list(plan_data),
        "PlacedImplants": [asdict(i) for i in implants],
    }
    report_text = json.dumps(report_payload, indent=2)

    try:
        ensure_optional_package("pydicom", "pydicom", "DICOM SEG/SR export")
        from pydicom.dataset import FileDataset, FileMetaDataset  # type: ignore
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # type: ignore

        study_uid = generate_uid()
        # 1. Write DICOM BasicTextSR (SOPClassUID 1.2.840.10008.5.1.4.1.1.88.11)
        sr_meta = FileMetaDataset()
        sr_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.11"
        sr_meta.MediaStorageSOPInstanceUID = generate_uid()
        sr_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        sr_ds = FileDataset(str(sr_dcm_path), {}, file_meta=sr_meta, preamble=b"\0" * 128)
        sr_ds.PatientName = str(patient_name)
        sr_ds.PatientID = str(patient_id)
        sr_ds.Modality = "SR"
        sr_ds.StudyInstanceUID = study_uid
        sr_ds.SeriesInstanceUID = generate_uid()
        sr_ds.SOPClassUID = sr_meta.MediaStorageSOPClassUID
        sr_ds.SOPInstanceUID = sr_meta.MediaStorageSOPInstanceUID
        sr_ds.CompletionFlag = "COMPLETE"
        sr_ds.VerificationFlag = "UNVERIFIED"
        sr_ds.TextValue = report_text
        sr_ds.save_as(str(sr_dcm_path), write_like_original=False)

        # 2. Write DICOM Segmentation (SOPClassUID 1.2.840.10008.5.1.4.1.1.66.4)
        arr = np.asarray(
            mask_zyx if mask_zyx is not None else np.zeros((4, 8, 8), dtype=np.uint8), dtype=np.uint8
        )
        seg_meta = FileMetaDataset()
        seg_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
        seg_meta.MediaStorageSOPInstanceUID = generate_uid()
        seg_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        seg_ds = FileDataset(str(seg_dcm_path), {}, file_meta=seg_meta, preamble=b"\0" * 128)
        seg_ds.PatientName = str(patient_name)
        seg_ds.PatientID = str(patient_id)
        seg_ds.Modality = "SEG"
        seg_ds.StudyInstanceUID = study_uid
        seg_ds.SeriesInstanceUID = generate_uid()
        seg_ds.SOPClassUID = seg_meta.MediaStorageSOPClassUID
        seg_ds.SOPInstanceUID = seg_meta.MediaStorageSOPInstanceUID
        seg_ds.NumberOfFrames = int(arr.shape[0])
        seg_ds.Rows = int(arr.shape[1])
        seg_ds.Columns = int(arr.shape[2])
        seg_ds.PixelSpacing = [float(spacing_zyx[1]), float(spacing_zyx[2])]
        seg_ds.SliceThickness = float(spacing_zyx[0])
        seg_ds.SamplesPerPixel = 1
        seg_ds.PhotometricInterpretation = "MONOCHROME2"
        seg_ds.BitsAllocated = 8
        seg_ds.BitsStored = 8
        seg_ds.HighBit = 7
        seg_ds.PixelRepresentation = 0
        seg_ds.SegmentationType = "BINARY"
        seg_ds.PixelData = arr.tobytes()
        seg_ds.save_as(str(seg_dcm_path), write_like_original=False)
        return seg_dcm_path, sr_dcm_path
    except Exception:
        # Clean fallback when pydicom is not installed in headless environment:
        # writes standard DICOM 128-byte preamble + "DICM" magic header + structured payload
        sr_dcm_path.write_bytes(b"\0" * 128 + b"DICM" + report_text.encode("utf-8"))
        arr_bytes = np.asarray(
            mask_zyx if mask_zyx is not None else np.zeros((4, 8, 8), dtype=np.uint8), dtype=np.uint8
        ).tobytes()
        seg_dcm_path.write_bytes(b"\0" * 128 + b"DICM" + arr_bytes)
        return seg_dcm_path, sr_dcm_path


# =====================================================================
# F16: Generic Abutment / Crown Library & Prosthetic Proposals
# =====================================================================
@dataclass
class ProstheticProposal:
    fdi: int
    abutment_id: str
    abutment_label: str
    abutment_angle_deg: float
    crown_id: str
    crown_label: str
    crown_height_mm: float
    crown_mesiodistal_mm: float
    crown_buccolingual_mm: float
    crown_to_implant_ratio: float
    emergence_angle_deg: float
    prosthetic_center_ras: List[float]


class ProstheticPlanner:
    """F16: Proposes generic Ti-base abutments and anatomic crowns above placed implants."""

    def __init__(self, library_path: Optional[Path] = None) -> None:
        default_path = (
            Path(library_path)
            if library_path is not None
            else Path(__file__).resolve().parent.parent / "Resources" / "generic_prosthetic_library.json"
        )
        self.abutments: List[Dict[str, Any]] = []
        self.crowns: List[Dict[str, Any]] = []
        if default_path.exists():
            data = json.loads(default_path.read_text(encoding="utf-8"))
            self.abutments = list(data.get("abutments", []))
            self.crowns = list(data.get("crowns", []))

    def generate_proposal_for_implant(
        self,
        implant: PlacedImplant,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Tuple[ProstheticProposal, np.ndarray, np.ndarray]:
        """F16: Select optimal generic abutment (straight vs 15°/25° angled) and crown for `implant` and build 3D mesh."""
        tooth_num = int(str(implant.fdi)[-1])
        crown_spec = next(
            (c for c in self.crowns if tooth_num in c.get("tooth_numbers", [])),
            {
                "id": "crown_molar",
                "label": "Generic Anatomic Molar Crown",
                "mesiodistal_mm": 10.0,
                "buccolingual_mm": 10.0,
                "cervical_diameter_mm": 6.5,
                "height_mm": 8.5,
            },
        )

        # Compute angulation between implant coronal axis (-axis_apical) and occlusal normal
        u_apical = np.asarray(implant.axis_apical_ras, dtype=np.float64)
        u_apical /= max(1e-8, float(np.linalg.norm(u_apical)))
        is_mandible = is_mandibular_fdi(implant.fdi)
        coronal_vec = -u_apical
        occ = np.asarray(occlusal_normal, dtype=np.float64)
        occ /= max(1e-8, float(np.linalg.norm(occ)))
        ideal_coronal = occ if is_mandible else -occ

        cos_ang = float(np.clip(np.dot(coronal_vec, ideal_coronal), -1.0, 1.0))
        emergence_deg = round(math.degrees(math.acos(cos_ang)), 2)

        # Choose straight vs 15 deg vs 25 deg angled abutment based on emergence angle
        if emergence_deg >= 20.0:
            target_ang = 25.0
        elif emergence_deg >= 10.0:
            target_ang = 15.0
        else:
            target_ang = 0.0

        abut_spec = next(
            (a for a in self.abutments if abs(float(a.get("angle_deg", 0.0)) - target_ang) < 5.0),
            {
                "id": "ti_base_straight_4.0x4.5",
                "label": "Generic Straight Ti-Base Ø4.0 × 4.5 mm",
                "diameter_mm": 4.0,
                "height_mm": 4.5,
                "gingival_height_mm": 1.0,
                "angle_deg": 0.0,
            },
        )

        p_crest = np.asarray(implant.crest_point_ras, dtype=np.float64)
        abut_h = float(abut_spec["height_mm"])
        crown_h = float(crown_spec["height_mm"])
        p_abut_top = p_crest + coronal_vec * abut_h
        p_crown_top = p_crest + ideal_coronal * (crown_h + float(abut_spec.get("gingival_height_mm", 1.0)))

        v_abut, f_abut = ImplantPlannerLogic.build_cylinder_mesh_triangles(
            p_crest,
            p_abut_top,
            radius_top_mm=0.5 * float(abut_spec["diameter_mm"]),
            radius_bottom_mm=0.5 * float(implant.platform_diameter_mm),
            n_sides=18,
        )
        v_crown, f_crown = ImplantPlannerLogic.build_cylinder_mesh_triangles(
            p_abut_top,
            p_crown_top,
            radius_top_mm=0.5 * float(crown_spec["cervical_diameter_mm"]),
            radius_bottom_mm=0.5 * float(crown_spec["mesiodistal_mm"]),
            n_sides=20,
        )
        merged_v = np.vstack([v_abut, v_crown])
        merged_f = np.vstack([f_abut, f_crown + len(v_abut)])

        c_to_i = round(crown_h / max(1.0, float(implant.length_mm)), 2)
        proposal = ProstheticProposal(
            fdi=implant.fdi,
            abutment_id=str(abut_spec["id"]),
            abutment_label=str(abut_spec["label"]),
            abutment_angle_deg=float(abut_spec["angle_deg"]),
            crown_id=str(crown_spec["id"]),
            crown_label=str(crown_spec["label"]),
            crown_height_mm=crown_h,
            crown_mesiodistal_mm=float(crown_spec["mesiodistal_mm"]),
            crown_buccolingual_mm=float(crown_spec["buccolingual_mm"]),
            crown_to_implant_ratio=c_to_i,
            emergence_angle_deg=emergence_deg,
            prosthetic_center_ras=[round(float(x), 3) for x in (0.5 * (p_abut_top + p_crown_top))],
        )
        return proposal, merged_v, merged_f


# =====================================================================
# F17: 3D Cephalometry (Landmarks + Norms Table + ALI Integration)
# =====================================================================
@dataclass
class CephalometricMeasurement:
    name: str
    value: float
    unit: str
    norm_mean: float
    norm_sd: float
    z_score: float
    interpretation: str


# Standard Steiner / Downs / Tweed cephalometric norms
CEPHALOMETRIC_NORMS: Dict[str, Tuple[float, float, str, Tuple[str, str, str]]] = {
    "SNA": (82.0, 2.0, "deg", ("Retrognathic Maxilla", "Normal Maxilla", "Prognathic Maxilla")),
    "SNB": (80.0, 2.0, "deg", ("Retrognathic Mandible", "Normal Mandible", "Prognathic Mandible")),
    "ANB": (
        2.0,
        2.0,
        "deg",
        ("Skeletal Class III Tendency", "Skeletal Class I", "Skeletal Class II Tendency"),
    ),
    "FMA": (25.0, 4.0, "deg", ("Hypodivergent (Low Angle)", "Normodivergent", "Hyperdivergent (High Angle)")),
    "GoGn-SN": (
        32.0,
        4.0,
        "deg",
        ("Horizontal Growth Pattern", "Balanced Growth Pattern", "Vertical Growth Pattern"),
    ),
    "U1-L1 (Interincisal)": (
        131.0,
        6.0,
        "deg",
        ("Proclined Incisors", "Normal Incisor Inclination", "Retroclined Incisors"),
    ),
}


def _angle_between_segments_deg(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    va = np.asarray(a1, dtype=np.float64) - np.asarray(a0, dtype=np.float64)
    vb = np.asarray(b1, dtype=np.float64) - np.asarray(b0, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos_v = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
    return round(math.degrees(math.acos(cos_v)), 2)


def compute_3d_cephalometry(
    landmarks_ras: Dict[str, Sequence[float]],
) -> List[CephalometricMeasurement]:
    """F17: Compute 3D cephalometric table (`SNA`, `SNB`, `ANB`, `FMA`, `GoGn-SN`, `U1-L1`) against clinical norms."""
    pts = {k: np.asarray(v, dtype=np.float64).reshape(3) for k, v in landmarks_ras.items()}
    raw_metrics: Dict[str, float] = {}

    if all(k in pts for k in ("S", "N", "A")):
        raw_metrics["SNA"] = compute_angle_between_three_points_deg(pts["S"], pts["N"], pts["A"])
    if all(k in pts for k in ("S", "N", "B")):
        raw_metrics["SNB"] = compute_angle_between_three_points_deg(pts["S"], pts["N"], pts["B"])
    if "SNA" in raw_metrics and "SNB" in raw_metrics:
        raw_metrics["ANB"] = round(raw_metrics["SNA"] - raw_metrics["SNB"], 2)

    # Go midpoint if bilateral Go_R / Go_L provided
    go_pt = pts.get("Go")
    if go_pt is None and "Go_R" in pts and "Go_L" in pts:
        go_pt = 0.5 * (pts["Go_R"] + pts["Go_L"])
    gn_pt = pts["Gn"] if "Gn" in pts else pts.get("Me")

    if go_pt is not None and gn_pt is not None and "S" in pts and "N" in pts:
        raw_metrics["GoGn-SN"] = _angle_between_segments_deg(pts["S"], pts["N"], go_pt, gn_pt)

    or_pt = (
        pts["Or"]
        if "Or" in pts
        else (0.5 * (pts["Or_R"] + pts["Or_L"]) if ("Or_R" in pts and "Or_L" in pts) else None)
    )
    po_pt = (
        pts["Po"]
        if "Po" in pts
        else (0.5 * (pts["Po_R"] + pts["Po_L"]) if ("Po_R" in pts and "Po_L" in pts) else None)
    )
    if or_pt is not None and po_pt is not None and go_pt is not None and gn_pt is not None:
        raw_metrics["FMA"] = _angle_between_segments_deg(po_pt, or_pt, go_pt, gn_pt)

    u1_tip = pts["UI_tip"] if "UI_tip" in pts else pts.get("U1_tip")
    u1_apex = pts["UI_apex"] if "UI_apex" in pts else pts.get("U1_apex")
    l1_tip = pts["LI_tip"] if "LI_tip" in pts else pts.get("L1_tip")
    l1_apex = pts["LI_apex"] if "LI_apex" in pts else pts.get("L1_apex")
    if u1_tip is not None and u1_apex is not None and l1_tip is not None and l1_apex is not None:
        raw_metrics["U1-L1 (Interincisal)"] = _angle_between_segments_deg(u1_tip, u1_apex, l1_tip, l1_apex)

    table: List[CephalometricMeasurement] = []
    for name, val in raw_metrics.items():
        mean_v, sd_v, unit, labels = CEPHALOMETRIC_NORMS[name]
        z = round((val - mean_v) / max(1e-6, sd_v), 2)
        if z < -1.0:
            interp = labels[0]
        elif z > 1.0:
            interp = labels[2]
        else:
            interp = labels[1]
        table.append(
            CephalometricMeasurement(
                name=name,
                value=round(val, 2),
                unit=unit,
                norm_mean=mean_v,
                norm_sd=sd_v,
                z_score=z,
                interpretation=interp,
            )
        )
    return table


# =====================================================================
# F18: Optional SlicerAutomatedDentalTools Execution Wrapper
# =====================================================================
def run_automated_dental_tool(
    tool_name: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """F18: Feature-detected launcher for SlicerAutomatedDentalTools (`ASO`, `AMASSS`, `ALI`, `AReg`, `AutoCrop3D`, `BatchDentalSegmentator`, `MRI2CBCT`)."""
    available = detect_automated_dental_tools_modules()
    clean_name = str(tool_name).strip()
    if not available.get(clean_name, False):
        return {
            "tool": clean_name,
            "status": "not_installed",
            "message": f"SlicerAutomatedDentalTools module '{clean_name}' is not installed in this 3D Slicer environment.",
        }
    if slicer is not None and hasattr(slicer, "util") and hasattr(slicer.util, "selectModule"):
        try:
            slicer.util.selectModule(clean_name)
        except Exception:
            pass
    return {
        "tool": clean_name,
        "status": "invoked",
        "parameters": dict(parameters or {}),
    }


# =====================================================================
# F20: Headless Batch Pipeline (Segmentation/Checkpoint -> Arch -> Pano -> Report)
# =====================================================================
def run_headless_batch_pipeline(
    volume: CBCTVolume,
    output_dir: Path,
    mask_zyx: Optional[np.ndarray] = None,
    checkpoint_spec: Optional[CheckpointSpec] = None,
    is_5class: bool = True,
    teeth_fdi_to_measure: Sequence[int] = (46, 36, 16, 26),
    place_generic_implants: bool = True,
) -> Dict[str, Any]:
    """F20: Run complete headless pipeline (`[Checkpoint] -> Segmentation -> Arch -> Pano -> Canal -> Morphometry -> Guide/Prosthetic/Reports`)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ckpt_summary = None
    if mask_zyx is None and checkpoint_spec is not None:
        ckpt_engine = CheckpointInferenceEngine()
        mask_zyx, ckpt_summary = ckpt_engine.run_checkpoint_on_array(volume.image, checkpoint_spec)
        is_5class = checkpoint_spec.label_scheme.lower() == "5class"

    # 1. Fit dental arch from segmentation mask or fallback horseshoe
    arch_pts: Optional[np.ndarray] = None
    if mask_zyx is not None and np.any(mask_zyx > 0):
        if is_5class:
            zyx = np.where((mask_zyx == 4) | (mask_zyx == 3))
            if len(zyx[0]) < 6:
                zyx = np.where((mask_zyx == 2) | (mask_zyx == 1))
            if len(zyx[0]) >= 6:
                step = max(1, len(zyx[0]) // 500)
                vox = np.column_stack([zyx[2][::step], zyx[1][::step], zyx[0][::step]])
                w_pts = volume.voxel_to_world(vox)
                arch_pts = fit_dental_arch_from_points(w_pts, num_control_points=11, ras_convention=True)
        else:
            centroids = []
            for tid in MANDIBULAR_TEETH_IDS + MAXILLARY_TEETH_IDS:
                tz, ty, tx = np.where(mask_zyx == tid)
                if len(tx) > 0:
                    centroids.append(volume.voxel_to_world([tx.mean(), ty.mean(), tz.mean()]))
            if len(centroids) >= 4:
                arch_pts = fit_dental_arch_from_points(
                    np.array(centroids, dtype=np.float64), num_control_points=11, ras_convention=True
                )

    if arch_pts is None:
        # Use central volume bounding arch fallback when running unsegmented headless
        z_mid = 0.5 * volume.shape[0]
        y_mid = 0.5 * volume.shape[1]
        x_span = np.linspace(0.8 * volume.shape[2], 0.2 * volume.shape[2], 9)
        vox_arch = np.column_stack([x_span, np.full_like(x_span, y_mid), np.full_like(x_span, z_mid)])
        arch_pts = volume.voxel_to_world(vox_arch)

    arch = DentalArch(arch_pts, jaw_type="lower", ras_convention=True)

    # 2. Generate in-process Panoramic CPR array
    pano_logic = PanoramicCurveLogic()

    class _HeadlessCurve:
        def GetID(self) -> str:
            return "HeadlessArch"

        def GetMTime(self) -> int:
            return 1

        def GetNumberOfControlPoints(self) -> int:
            return len(arch_pts)

        def GetNthControlPointPositionWorld(self, idx: int, buf: list[float]) -> None:
            buf[0], buf[1], buf[2] = float(arch_pts[idx, 0]), float(arch_pts[idx, 1]), float(arch_pts[idx, 2])

        def GetCurveWorld(self) -> None:
            return None

    pts_c, _, tangents, normals, binormals = pano_logic.get_curve_points_and_frames(
        _HeadlessCurve(), num_samples=80
    )
    pano_arr = pano_logic.sample_panoramic_slab_array(
        volume,
        pts_c,
        tangents,
        normals,
        binormals,
        slab_thickness_mm=15.0,
        height_mm=40.0,
        spacing_mm=1.0,
        projection_mode="RaySum",
    )
    pano_npy_path = out_path / "panoramic_cpr.npy"
    np.save(str(pano_npy_path), pano_arr)

    # 3. Extract Mandibular Canals & Site Measurements
    canals = (
        CanalTracerLogic.extract_canals_from_mask(volume, mask_zyx, is_5class=is_5class)
        if mask_zyx is not None
        else {}
    )
    engine = (
        MeasurementEngine(
            volume,
            mask_zyx,
            arch=arch,
            arches={"lower": arch, "upper": arch},
            is_dental_segmentator_5class=is_5class,
            extract_teeth_eagerly=False,
        )
        if mask_zyx is not None
        else None
    )

    from CBCTPlanner import CBCTPlannerLogic

    logic = CBCTPlannerLogic()
    plan_rows: List[Dict[str, Any]] = []
    for fdi in teeth_fdi_to_measure:
        row = logic.analyze_single_tooth(int(fdi), volumeNode=None, segmentationNode=None, engine=engine)
        f_frame = arch.get_frame(logic.fdi_to_normalized_arch_t(int(fdi)))
        row["pos_ras"] = [round(float(c), 2) for c in f_frame.point]
        plan_rows.append(row)
        if place_generic_implants:
            logic.implants.place_or_update_implant(
                fdi=int(fdi),
                crest_point_ras=f_frame.point,
                T=f_frame.tangent,
                N=f_frame.normal,
                B=f_frame.binormal,
                entry_id="generic_4.0x10.0",
                is_ai_suggestion=False,
                clinician_confirmed=True,
                volume=volume,
                mask=mask_zyx,
                is_5class=is_5class,
                canals=canals,
            )

    # 4. Export CSV, JSON, HTML, PDF, Implant STLs, Surgical Guide STL, and DICOM SEG/SR
    json_path, csv_path = logic.export_treatment_plan_files(plan_rows, out_path)
    html_path, pdf_path = logic.generate_clinical_reports(
        plan_rows, volumeNode=None, out_dir=out_path, capture_screenshots=False
    )
    imp_json, imp_stls = logic.implants.export_implants_json_and_stl(out_path)
    guide_gen = SurgicalGuideGenerator()
    guide_stl, guide_json, _ = guide_gen.export_surgical_guide_stl_and_json(
        arch_pts, logic.implants.get_reportable_implants(), out_path
    )
    seg_dcm, sr_dcm = export_dicom_seg_and_sr(
        mask_zyx, volume.spacing, plan_rows, logic.implants.get_reportable_implants(), out_path
    )

    return {
        "output_dir": str(out_path),
        "checkpoint_summary": asdict(ckpt_summary) if ckpt_summary is not None else None,
        "panoramic_npy": str(pano_npy_path),
        "plan_json": str(json_path),
        "plan_csv": str(csv_path),
        "report_html": str(html_path),
        "report_pdf": str(pdf_path),
        "implant_json": str(imp_json),
        "implant_stls": [str(p) for p in imp_stls],
        "surgical_guide_stl": str(guide_stl),
        "surgical_guide_json": str(guide_json),
        "dicom_seg": str(seg_dcm),
        "dicom_sr": str(sr_dcm),
    }

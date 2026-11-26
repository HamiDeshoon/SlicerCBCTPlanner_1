"""3D Topology-Aware Multi-Root Endodontic Canal & Access Cavity Planning Engine (Phase 6).

Architecture inspired by SlicerEndoPlanner (Apache-2.0) and anatomical EDT morphometry:
  - Physical-space tooth ROI extraction preserving CBCT voxel-to-RAS affine.
  - Explicit pulp mask ingestion (segmented pulp IDs, optional external checkpoint adapter,
    or conservatively labeled intensity-refined lumen strictly flagged as `estimated`).
  - 3D anisotropic Euclidean Distance Transform (EDT) radius profiling + 26-neighbor
    weighted geodesic tree / skeleton graph extracting MULTIPLE roots, multiple canals,
    bifurcations/furcation junctions, canal length (mm), Schneider 3D curvature (deg),
    external root wall clearance (mm), and apical terminus localization.
  - Access cavity corridor optimizer with manual override support and clinical disclaimer
    (`automated_planning_aid — not clinically approved`).
  - Pure NumPy + SciPy core (no mandatory OR-Tools, Numba, PyTorch, or scikit-image).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core.contracts import (
    SOURCE_ESTIMATED,
    SOURCE_MANUALLY_CORRECTED,
    SOURCE_MEASURED,
    SOURCE_MODEL_PREDICTED,
    SOURCE_NOT_AVAILABLE,
    UNIT_DEG,
    UNIT_MM,
)
from .core.labels import (
    FDI_TO_PULP_ID,
    FDI_TO_TOOTH_ID,
    is_mandibular_fdi,
    resolve_fdi_number,
)

CLINICAL_ENDO_STATUS = "automated_planning_aid — not clinically approved"


@dataclass
class EndoToothROI:
    """Physical sub-volume cropped around a target tooth with preserved RAS affine."""

    fdi: str
    is_mandible: bool
    spacing_zyx_mm: Tuple[float, float, float]
    affine_ras: np.ndarray
    ijk_offset_zyx: Tuple[int, int, int]
    image_zyx: Optional[np.ndarray]
    tooth_mask: np.ndarray
    pulp_mask: np.ndarray
    pulp_source_type: str
    pulp_method: str
    pulp_confidence: Optional[float] = None
    caveats: List[str] = field(default_factory=list)


@dataclass
class CanalBifurcation:
    """3D junction point where root canal trunks split or merge."""

    position_ras_mm: List[float]
    geodesic_depth_mm: float
    radius_mm: float
    child_canal_ids: List[str] = field(default_factory=list)


@dataclass
class RootCanalBranch:
    """Complete 3D morphometric representation of one root-canal branch."""

    canal_id: str
    root_label: str
    points_ras_mm: np.ndarray
    radii_mm: np.ndarray
    wall_clearance_mm: np.ndarray
    length_mm: float
    curvature_deg: float
    min_radius_mm: float
    mean_radius_mm: float
    max_radius_mm: float
    min_wall_distance_mm: Optional[float]
    orifice_ras_mm: List[float]
    apex_ras_mm: List[float]
    confidence: float
    source_type: str = SOURCE_MEASURED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canal_id": self.canal_id,
            "root_label": self.root_label,
            "length_mm": round(float(self.length_mm), 3),
            "length_unit": UNIT_MM,
            "curvature_deg": round(float(self.curvature_deg), 2),
            "curvature_unit": UNIT_DEG,
            "min_radius_mm": round(float(self.min_radius_mm), 3),
            "mean_radius_mm": round(float(self.mean_radius_mm), 3),
            "max_radius_mm": round(float(self.max_radius_mm), 3),
            "min_wall_distance_mm": (
                None
                if self.min_wall_distance_mm is None
                else round(float(self.min_wall_distance_mm), 3)
            ),
            "orifice_ras_mm": [round(float(v), 3) for v in self.orifice_ras_mm],
            "apex_ras_mm": [round(float(v), 3) for v in self.apex_ras_mm],
            "confidence": round(float(self.confidence), 3),
            "source_type": self.source_type,
            "points_ras_mm": [[round(float(v), 3) for v in pt] for pt in self.points_ras_mm],
            "radii_mm": [round(float(r), 3) for r in self.radii_mm],
            "wall_clearance_mm": [round(float(w), 3) for w in self.wall_clearance_mm],
        }


@dataclass
class AccessCavityPlan:
    """Planned or manually adjusted straight-line coronal access corridor."""

    entry_ras_mm: List[float]
    target_ras_mm: List[float]
    direction_ras: List[float]
    recommended_diameter_mm: float
    min_wall_clearance_mm: Optional[float]
    orifice_alignment_deg: float
    confidence: float
    source_type: str = SOURCE_MEASURED
    manual_edit: bool = False
    status_flag: str = CLINICAL_ENDO_STATUS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EndoToothAnalysis:
    """Unified multi-root endodontic analysis result for one FDI tooth."""

    fdi: str
    is_mandible: bool
    status_flag: str = CLINICAL_ENDO_STATUS
    available: bool = False
    withheld_reason: Optional[str] = None
    pulp_source_type: str = SOURCE_NOT_AVAILABLE
    pulp_method: str = "none"
    pulp_volume_mm3: float = 0.0
    chamber_center_ras_mm: Optional[List[float]] = None
    canals: List[RootCanalBranch] = field(default_factory=list)
    bifurcations: List[CanalBifurcation] = field(default_factory=list)
    access_cavity: Optional[AccessCavityPlan] = None
    caveats: List[str] = field(default_factory=list)
    external_model_status: Optional[str] = None

    @property
    def total_canals(self) -> int:
        return len(self.canals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fdi": self.fdi,
            "is_mandible": bool(self.is_mandible),
            "status_flag": self.status_flag,
            "available": bool(self.available),
            "withheld_reason": self.withheld_reason,
            "pulp_source_type": self.pulp_source_type,
            "pulp_method": self.pulp_method,
            "pulp_volume_mm3": round(float(self.pulp_volume_mm3), 3),
            "chamber_center_ras_mm": (
                [round(float(v), 3) for v in self.chamber_center_ras_mm]
                if self.chamber_center_ras_mm is not None
                else None
            ),
            "total_canals": self.total_canals,
            "canals": [c.to_dict() for c in self.canals],
            "bifurcations": [
                {
                    "position_ras_mm": [round(float(v), 3) for v in b.position_ras_mm],
                    "geodesic_depth_mm": round(float(b.geodesic_depth_mm), 3),
                    "radius_mm": round(float(b.radius_mm), 3),
                    "child_canal_ids": list(b.child_canal_ids),
                }
                for b in self.bifurcations
            ],
            "access_cavity": self.access_cavity.to_dict() if self.access_cavity is not None else None,
            "caveats": list(self.caveats),
            "external_model_status": self.external_model_status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EndoToothAnalysis":
        canals: List[RootCanalBranch] = []
        for cd in payload.get("canals") or []:
            canals.append(
                RootCanalBranch(
                    canal_id=str(cd.get("canal_id", "")),
                    root_label=str(cd.get("root_label", "")),
                    points_ras_mm=np.asarray(cd.get("points_ras_mm", []), dtype=np.float64),
                    radii_mm=np.asarray(cd.get("radii_mm", []), dtype=np.float64),
                    wall_clearance_mm=np.asarray(cd.get("wall_clearance_mm", []), dtype=np.float64),
                    length_mm=float(cd.get("length_mm", 0.0)),
                    curvature_deg=float(cd.get("curvature_deg", 0.0)),
                    min_radius_mm=float(cd.get("min_radius_mm", 0.0)),
                    mean_radius_mm=float(cd.get("mean_radius_mm", 0.0)),
                    max_radius_mm=float(cd.get("max_radius_mm", 0.0)),
                    min_wall_distance_mm=(
                        None
                        if cd.get("min_wall_distance_mm") is None
                        else float(cd["min_wall_distance_mm"])
                    ),
                    orifice_ras_mm=[float(v) for v in cd.get("orifice_ras_mm", [0.0, 0.0, 0.0])],
                    apex_ras_mm=[float(v) for v in cd.get("apex_ras_mm", [0.0, 0.0, 0.0])],
                    confidence=float(cd.get("confidence", 0.0)),
                    source_type=str(cd.get("source_type", SOURCE_MEASURED)),
                )
            )
        bifs = [
            CanalBifurcation(
                position_ras_mm=[float(v) for v in b.get("position_ras_mm", [0.0, 0.0, 0.0])],
                geodesic_depth_mm=float(b.get("geodesic_depth_mm", 0.0)),
                radius_mm=float(b.get("radius_mm", 0.0)),
                child_canal_ids=[str(c) for c in (b.get("child_canal_ids") or [])],
            )
            for b in (payload.get("bifurcations") or [])
        ]
        ac_raw = payload.get("access_cavity")
        ac = AccessCavityPlan(**ac_raw) if isinstance(ac_raw, dict) else None
        return cls(
            fdi=str(payload.get("fdi", "")),
            is_mandible=bool(payload.get("is_mandible", True)),
            status_flag=str(payload.get("status_flag", CLINICAL_ENDO_STATUS)),
            available=bool(payload.get("available", False)),
            withheld_reason=payload.get("withheld_reason"),
            pulp_source_type=str(payload.get("pulp_source_type", SOURCE_NOT_AVAILABLE)),
            pulp_method=str(payload.get("pulp_method", "none")),
            pulp_volume_mm3=float(payload.get("pulp_volume_mm3", 0.0)),
            chamber_center_ras_mm=payload.get("chamber_center_ras_mm"),
            canals=canals,
            bifurcations=bifs,
            access_cavity=ac,
            caveats=[str(c) for c in (payload.get("caveats") or [])],
            external_model_status=payload.get("external_model_status"),
        )


def _ijk_zyx_to_ras(ijk_zyx: np.ndarray, affine_ras: np.ndarray) -> np.ndarray:
    """Convert `(N, 3)` `[z, y, x]` voxel indices to RAS world `(N, 3)` `[R, A, S]`."""
    pts = np.asarray(ijk_zyx, dtype=np.float64).reshape(-1, 3)
    xyz = pts[:, [2, 1, 0]]
    return xyz @ affine_ras[:3, :3].T + affine_ras[:3, 3]


def _ras_to_ijk_zyx(ras: np.ndarray, affine_ras: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(affine_ras, dtype=np.float64))
    pts = np.asarray(ras, dtype=np.float64).reshape(-1, 3)
    xyz = pts @ inv[:3, :3].T + inv[:3, 3]
    return xyz[:, [2, 1, 0]]


def _voxel_spacing_zyx(affine_ras: np.ndarray) -> Tuple[float, float, float]:
    sx = float(np.linalg.norm(affine_ras[:3, 0]))
    sy = float(np.linalg.norm(affine_ras[:3, 1]))
    sz = float(np.linalg.norm(affine_ras[:3, 2]))
    return max(1e-3, sz), max(1e-3, sy), max(1e-3, sx)


def _compute_polyline_curvature_deg(points_ras: np.ndarray) -> float:
    """Compute cumulative turning curvature angle (Schneider-aligned 3D deflection, degrees)."""
    pts = np.asarray(points_ras, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    # Resample at ~1.0 mm steps so voxel stair-stepping does not inflate turning angles
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total < 2.0:
        return 0.0
    n_res = max(3, min(40, int(round(total / 1.0)) + 1))
    s_grid = np.linspace(0.0, total, n_res)
    smooth = np.column_stack([np.interp(s_grid, cum, pts[:, d]) for d in range(3)])
    segs = np.diff(smooth, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    valid = segs[seg_lens > 1e-6] / seg_lens[seg_lens > 1e-6, None]
    if len(valid) < 2:
        return 0.0
    # Max deflection from coronal entry vector (Schneider angle generalization in 3D)
    v0 = valid[0]
    dots = np.clip(valid @ v0, -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dots))
    return float(np.max(angles_deg))


def _resample_branch_smooth(
    pts_ras: np.ndarray,
    radii: np.ndarray,
    wall_dists: np.ndarray,
    step_mm: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pts = np.asarray(pts_ras, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    w = np.asarray(wall_dists, dtype=np.float64)
    if len(pts) < 2:
        return pts, r, w, 0.0
    # Light 3-tap triangular smoothing preserving endpoints to suppress grid staircases
    if len(pts) >= 5:
        sm = pts.copy()
        sm[1:-1] = 0.25 * pts[:-2] + 0.50 * pts[1:-1] + 0.25 * pts[2:]
        pts = sm
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total <= 1e-6:
        return pts, r, w, 0.0
    n_out = max(4, int(round(total / max(0.25, float(step_mm)))) + 1)
    s_new = np.linspace(0.0, total, n_out)
    pts_out = np.column_stack([np.interp(s_new, cum, pts[:, d]) for d in range(3)])
    r_out = np.interp(s_new, cum, r)
    w_out = np.interp(s_new, cum, w)
    return pts_out, r_out, w_out, total


class EndodonticEngine:
    """Multi-root topology graph canal analyzer and access cavity trajectory optimizer."""

    def __init__(self, min_branch_spur_mm: float = 2.0) -> None:
        self.min_branch_spur_mm = float(min_branch_spur_mm)

    @staticmethod
    def extract_tooth_roi(
        fdi: int | str,
        mask_zyx: np.ndarray,
        affine_ras: np.ndarray,
        image_zyx: Optional[np.ndarray] = None,
        *,
        margin_mm: float = 3.5,
        allow_intensity_refinement: bool = False,
        optional_checkpoint_path: Optional[Path] = None,
    ) -> Tuple[Optional[EndoToothROI], Optional[str]]:
        """Crop physical-space subvolume around `fdi` tooth and pulp without dropping transform accuracy."""
        fdi_str = resolve_fdi_number(fdi)
        if not fdi_str:
            return None, f"Invalid FDI tooth code: {fdi!r}"

        tooth_id = FDI_TO_TOOTH_ID.get(fdi_str)
        pulp_id = FDI_TO_PULP_ID.get(fdi_str)
        if tooth_id is None and pulp_id is None:
            return None, f"Unsupported FDI tooth identifier: #{fdi_str}"

        sz, sy, sx = _voxel_spacing_zyx(affine_ras)
        has_tooth = tooth_id is not None and bool(np.any(mask_zyx == tooth_id))
        has_pulp = pulp_id is not None and bool(np.any(mask_zyx == pulp_id))
        if not has_tooth and not has_pulp:
            return None, f"Neither tooth #{fdi_str} nor its segmented pulp exists in label mask"

        union_sel = np.zeros(mask_zyx.shape, dtype=bool)
        if has_tooth:
            union_sel |= mask_zyx == tooth_id
        if has_pulp:
            union_sel |= mask_zyx == pulp_id
        zz, yy, xx = np.where(union_sel)
        pad_z = max(2, int(math.ceil(margin_mm / sz)))
        pad_y = max(2, int(math.ceil(margin_mm / sy)))
        pad_x = max(2, int(math.ceil(margin_mm / sx)))
        z0 = max(0, int(zz.min()) - pad_z)
        z1 = min(mask_zyx.shape[0], int(zz.max()) + pad_z + 1)
        y0 = max(0, int(yy.min()) - pad_y)
        y1 = min(mask_zyx.shape[1], int(yy.max()) + pad_y + 1)
        x0 = max(0, int(xx.min()) - pad_x)
        x1 = min(mask_zyx.shape[2], int(xx.max()) + pad_x + 1)

        roi_mask = mask_zyx[z0:z1, y0:y1, x0:x1]
        roi_img = (
            np.asarray(image_zyx[z0:z1, y0:y1, x0:x1], dtype=np.float32)
            if image_zyx is not None
            else None
        )
        pulp_mask = (roi_mask == pulp_id) if has_pulp else np.zeros(roi_mask.shape, dtype=bool)
        tooth_only = (roi_mask == tooth_id) if has_tooth else np.zeros(roi_mask.shape, dtype=bool)
        # Tooth solid = crown/dentin + enclosed pulp chamber/canals for outer wall EDT distance
        tooth_solid = tooth_only | pulp_mask

        roi_affine = np.asarray(affine_ras, dtype=np.float64).copy()
        offset_xyz = np.array([float(x0), float(y0), float(z0)], dtype=np.float64)
        roi_affine[:3, 3] = affine_ras[:3, :3] @ offset_xyz + affine_ras[:3, 3]

        caveats: List[str] = []
        pulp_source = SOURCE_MEASURED if has_pulp else SOURCE_NOT_AVAILABLE
        pulp_method = "segmented_pulp_label" if has_pulp else "none"
        confidence: Optional[float] = 1.0 if has_pulp else None

        ckpt_status = None
        if not has_pulp and optional_checkpoint_path is not None:
            ckpt = Path(optional_checkpoint_path)
            if not ckpt.is_file():
                ckpt_status = f"External endodontic checkpoint unavailable at {ckpt}; no weights loaded"
                caveats.append(ckpt_status)
            elif roi_img is not None:
                try:
                    from .CheckpointInference import CheckpointSpec, infer_volume_with_checkpoint

                    spec = CheckpointSpec(checkpoint_path=ckpt, label_scheme="5class")
                    pred = infer_volume_with_checkpoint(roi_img, spec)
                    if pred is not None and np.any(pred > 0):
                        pulp_mask = (pred > 0) & (tooth_solid if tooth_solid.any() else True)
                        has_pulp = bool(np.any(pulp_mask))
                        if has_pulp:
                            pulp_source = SOURCE_MODEL_PREDICTED
                            pulp_method = f"external_checkpoint:{ckpt.name}"
                            confidence = 0.75
                except Exception as exc:
                    ckpt_status = f"External checkpoint inference skipped: {exc}"
                    caveats.append(ckpt_status)

        if not has_pulp and allow_intensity_refinement and roi_img is not None and tooth_solid.any():
            refined = EndodonticEngine._conservative_lumen_from_intensity(
                roi_img, tooth_solid, spacing_zyx=(sz, sy, sx)
            )
            if int(np.count_nonzero(refined)) >= 12:
                pulp_mask = refined
                has_pulp = True
                pulp_source = SOURCE_ESTIMATED
                pulp_method = "intensity_lumen_within_eroded_dentin"
                confidence = 0.45
                caveats.append(
                    "Estimated pulp lumen from interior tooth radiolucency without ground-truth pulp label"
                )
            else:
                caveats.append("No contiguous interior radiolucency met lumen threshold inside tooth")

        roi = EndoToothROI(
            fdi=fdi_str,
            is_mandible=is_mandibular_fdi(fdi_str),
            spacing_zyx_mm=(sz, sy, sx),
            affine_ras=roi_affine,
            ijk_offset_zyx=(z0, y0, x0),
            image_zyx=roi_img,
            tooth_mask=tooth_solid,
            pulp_mask=pulp_mask,
            pulp_source_type=pulp_source,
            pulp_method=pulp_method,
            pulp_confidence=confidence,
            caveats=caveats,
        )
        return roi, ckpt_status

    @staticmethod
    def _conservative_lumen_from_intensity(
        image_zyx: np.ndarray,
        tooth_mask: np.ndarray,
        spacing_zyx: Tuple[float, float, float],
    ) -> np.ndarray:
        """Strictly conservative dark-lumen extraction >0.7 mm inside dentin wall."""
        from scipy import ndimage  # type: ignore

        edt = ndimage.distance_transform_edt(tooth_mask, sampling=spacing_zyx)
        interior = tooth_mask & (edt >= 0.75)
        if int(np.count_nonzero(interior)) < 20:
            return np.zeros(tooth_mask.shape, dtype=bool)
        vals = image_zyx[interior]
        p2, p50, p95 = np.percentile(vals, [2.0, 50.0, 95.0])
        contrast = float(p95 - p2)
        if contrast < 250.0 or float(p50 - p2) < 200.0:
            return np.zeros(tooth_mask.shape, dtype=bool)
        thresh = float(min(0.5 * (p2 + p50), p50 - 0.35 * contrast, 900.0))
        cand = interior & (image_zyx <= thresh)
        lbl, n_comp = ndimage.label(cand)
        if n_comp == 0:
            return np.zeros(tooth_mask.shape, dtype=bool)
        sizes = np.bincount(lbl.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        return lbl == keep

    def analyze_tooth(
        self,
        fdi: int | str,
        mask_zyx: np.ndarray,
        affine_ras: np.ndarray,
        image_zyx: Optional[np.ndarray] = None,
        *,
        allow_intensity_refinement: bool = False,
        optional_checkpoint_path: Optional[Path] = None,
        manual_access_entry_ras: Optional[Sequence[float]] = None,
        manual_access_target_ras: Optional[Sequence[float]] = None,
    ) -> EndoToothAnalysis:
        """Run full multi-root canal topology reconstruction + access cavity optimization."""
        fdi_str = resolve_fdi_number(fdi) or str(fdi)
        roi, ckpt_status = self.extract_tooth_roi(
            fdi_str,
            mask_zyx,
            affine_ras,
            image_zyx=image_zyx,
            allow_intensity_refinement=allow_intensity_refinement,
            optional_checkpoint_path=optional_checkpoint_path,
        )
        if roi is None:
            return EndoToothAnalysis(
                fdi=fdi_str,
                is_mandible=is_mandibular_fdi(fdi_str) if fdi_str.isdigit() else True,
                available=False,
                withheld_reason=ckpt_status or f"Tooth #{fdi_str} not present",
                external_model_status=ckpt_status,
            )

        pulp_vox = int(np.count_nonzero(roi.pulp_mask))
        voxel_vol = float(np.prod(roi.spacing_zyx_mm))
        pulp_vol_mm3 = pulp_vox * voxel_vol
        if pulp_vox < 6:
            return EndoToothAnalysis(
                fdi=roi.fdi,
                is_mandible=roi.is_mandible,
                available=False,
                withheld_reason=(
                    ckpt_status
                    or f"Insufficient segmented pulp volume for #{roi.fdi} ({pulp_vox} voxels)"
                ),
                pulp_source_type=SOURCE_NOT_AVAILABLE,
                pulp_method=roi.pulp_method,
                pulp_volume_mm3=pulp_vol_mm3,
                caveats=list(roi.caveats),
                external_model_status=ckpt_status,
            )

        from scipy import ndimage  # type: ignore

        edt_pulp_mm = ndimage.distance_transform_edt(roi.pulp_mask, sampling=roi.spacing_zyx_mm)
        edt_wall_mm = (
            ndimage.distance_transform_edt(roi.tooth_mask, sampling=roi.spacing_zyx_mm)
            if np.any(roi.tooth_mask)
            else np.zeros_like(edt_pulp_mm)
        )

        # Restrict to primary connected component(s) with >= 6 voxels
        comp_lbl, n_comp = ndimage.label(roi.pulp_mask, structure=np.ones((3, 3, 3)))
        comp_sizes = np.bincount(comp_lbl.ravel())
        valid_comps = [cid for cid in range(1, n_comp + 1) if comp_sizes[cid] >= 6]
        if not valid_comps:
            valid_comps = [int(np.argmax(comp_sizes[1:])) + 1] if n_comp > 0 else []

        main_comp = int(max(valid_comps, key=lambda c: int(comp_sizes[c])))
        pulp_primary = comp_lbl == main_comp
        canals, bifurcations, chamber_ras = self._extract_topology_from_component(
            pulp_primary,
            edt_pulp_mm,
            edt_wall_mm,
            roi,
        )

        access_plan = self._optimize_access_cavity(
            roi=roi,
            edt_wall_mm=edt_wall_mm,
            canals=canals,
            chamber_center_ras=chamber_ras,
            manual_entry_ras=manual_access_entry_ras,
            manual_target_ras=manual_access_target_ras,
        )

        return EndoToothAnalysis(
            fdi=roi.fdi,
            is_mandible=roi.is_mandible,
            available=len(canals) > 0,
            withheld_reason=None if canals else "Failed to extract contiguous canal paths",
            pulp_source_type=roi.pulp_source_type,
            pulp_method=roi.pulp_method,
            pulp_volume_mm3=pulp_vol_mm3,
            chamber_center_ras_mm=chamber_ras,
            canals=canals,
            bifurcations=bifurcations,
            access_cavity=access_plan,
            caveats=list(roi.caveats),
            external_model_status=ckpt_status,
        )

    def _extract_topology_from_component(
        self,
        pulp_comp: np.ndarray,
        edt_pulp_mm: np.ndarray,
        edt_wall_mm: np.ndarray,
        roi: EndoToothROI,
    ) -> Tuple[List[RootCanalBranch], List[CanalBifurcation], List[float]]:
        """26-neighbor EDT-ridge shortest-path tree extracting multi-root canals & furcations."""
        nodes_zyx = np.argwhere(pulp_comp)
        n_vox = len(nodes_zyx)
        if n_vox == 0:
            return [], [], [0.0, 0.0, 0.0]

        radii_all = edt_pulp_mm[pulp_comp]
        walls_all = edt_wall_mm[pulp_comp]
        max_r = max(0.25, float(np.max(radii_all)))

        # Crown chamber seed: coronal end weighted toward thickest pulp chamber core
        pts_ras = _ijk_zyx_to_ras(nodes_zyx, roi.affine_ras)
        z_world = pts_ras[:, 2]
        coronal_score = (z_world if roi.is_mandible else -z_world) - np.min(
            z_world if roi.is_mandible else -z_world
        )
        c_span = max(1.0, float(np.max(coronal_score)))
        seed_score = 0.65 * (coronal_score / c_span) + 0.35 * (radii_all / max_r)
        root_idx = int(np.argmax(seed_score))
        chamber_ras = [float(v) for v in pts_ras[root_idx]]

        # Build lookup volume mapping voxel (z, y, x) -> node index [0..n_vox-1]
        idx_vol = np.full(pulp_comp.shape, -1, dtype=np.int32)
        idx_vol[nodes_zyx[:, 0], nodes_zyx[:, 1], nodes_zyx[:, 2]] = np.arange(n_vox, dtype=np.int32)

        # Precompute 26-neighbor steps with physical edge length
        sz, sy, sx = roi.spacing_zyx_mm
        steps = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    dist = math.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2)
                    steps.append((dz, dy, dx, dist))

        # Dijkstra from chamber seed with center-biased edge cost keeping paths along medial ridge
        nz, ny, nx = pulp_comp.shape
        g_cost = np.full(n_vox, float("inf"), dtype=np.float64)
        g_euclid = np.zeros(n_vox, dtype=np.float64)
        pred = np.full(n_vox, -1, dtype=np.int32)
        g_cost[root_idx] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, root_idx)]

        while pq:
            cur_c, u = heapq.heappop(pq)
            if cur_c > g_cost[u]:
                continue
            z, y, x = int(nodes_zyx[u, 0]), int(nodes_zyx[u, 1]), int(nodes_zyx[u, 2])
            r_u = float(radii_all[u])
            for dz, dy, dx, step_d in steps:
                zz, yy, xx = z + dz, y + dy, x + dx
                if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                    v = int(idx_vol[zz, yy, xx])
                    if v < 0:
                        continue
                    r_edge = 0.5 * (r_u + float(radii_all[v]))
                    ridge_pen = 1.0 + 3.0 * max(0.0, (max_r - r_edge) / max_r)
                    cand_c = cur_c + step_d * ridge_pen
                    if cand_c < g_cost[v]:
                        g_cost[v] = cand_c
                        g_euclid[v] = g_euclid[u] + step_d
                        pred[v] = u
                        heapq.heappush(pq, (cand_c, v))

        # Identify genuine root apex terminal candidates: local maxima of Euclidean geodesic
        # distance within a 3.0 mm spherical neighborhood in RAS (distinct root branches).
        max_len = float(np.max(g_euclid))
        if max_len < 1.5:
            # Tiny focal chamber only: build single coronal-to-apical segment
            end_idx = int(np.argmax(g_euclid))
            path = self._trace_path(root_idx, end_idx, pred)
            if len(path) < 2:
                return [], [], chamber_ras
            pts_b, r_b, w_b, b_len = _resample_branch_smooth(
                pts_ras[path], radii_all[path], walls_all[path]
            )
            return (
                [
                    RootCanalBranch(
                        canal_id=f"{roi.fdi}_C1",
                        root_label="Root 1",
                        points_ras_mm=pts_b,
                        radii_mm=r_b,
                        wall_clearance_mm=w_b,
                        length_mm=b_len,
                        curvature_deg=_compute_polyline_curvature_deg(pts_b),
                        min_radius_mm=float(np.min(r_b)),
                        mean_radius_mm=float(np.mean(r_b)),
                        max_radius_mm=float(np.max(r_b)),
                        min_wall_distance_mm=float(np.min(w_b)) if float(np.max(w_b)) > 0 else None,
                        orifice_ras_mm=chamber_ras,
                        apex_ras_mm=[float(v) for v in pts_b[-1]],
                        confidence=float(roi.pulp_confidence or 0.9),
                        source_type=roi.pulp_source_type,
                    )
                ],
                [],
                chamber_ras,
            )

        # Identify true root apex endpoints: strict local geodesic-distance maxima over a 3D
        # spherical ball slightly wider than one canal lumen diameter (`~3.0 mm`). Mid-shaft wall
        # voxels always have larger geodesic neighbors further apically and are eliminated in C.
        from scipy import ndimage  # type: ignore

        r_nms_mm = max(2.8, min(4.5, 2.0 * float(np.median(radii_all)) + 1.4))
        kz = max(2, int(math.ceil(r_nms_mm / sz)))
        ky = max(2, int(math.ceil(r_nms_mm / sy)))
        kx = max(2, int(math.ceil(r_nms_mm / sx)))
        oz, oy, ox = np.ogrid[-kz : kz + 1, -ky : ky + 1, -kx : kx + 1]
        sphere = ((oz * sz) ** 2 + (oy * sy) ** 2 + (ox * sx) ** 2) <= (r_nms_mm**2)

        g_vol = np.full(pulp_comp.shape, -1.0, dtype=np.float64)
        g_vol[nodes_zyx[:, 0], nodes_zyx[:, 1], nodes_zyx[:, 2]] = g_euclid
        g_local_max = ndimage.maximum_filter(g_vol, footprint=sphere, mode="constant", cval=-1.0)

        min_root_sep_mm = max(2.8, self.min_branch_spur_mm)
        is_peak = (g_euclid > 0.0) & (
            g_euclid >= g_local_max[nodes_zyx[:, 0], nodes_zyx[:, 1], nodes_zyx[:, 2]] - 1e-6
        )
        peak_candidates = np.flatnonzero(is_peak)
        order_far = peak_candidates[np.argsort(g_euclid[peak_candidates])[::-1]]

        selected_tips: List[int] = []
        for cand in order_far:
            if float(g_euclid[cand]) < max(2.0, 0.35 * max_len):
                break
            p_cand = pts_ras[int(cand)]
            if any(
                float(np.linalg.norm(p_cand - pts_ras[t])) < min_root_sep_mm for t in selected_tips
            ):
                continue
            path_c = self._trace_path(root_idx, int(cand), pred)
            if selected_tips:
                seen_earlier = set()
                for t in selected_tips:
                    seen_earlier.update(self._trace_path(root_idx, t, pred))
                unshared = [u for u in path_c if u not in seen_earlier]
                if len(unshared) < 3:
                    continue
                unshared_pts = pts_ras[unshared]
                unshared_len = float(np.sum(np.linalg.norm(np.diff(unshared_pts, axis=0), axis=1)))
                if unshared_len < self.min_branch_spur_mm:
                    continue
            selected_tips.append(int(cand))

        if not selected_tips:
            selected_tips = [int(np.argmax(g_euclid))]

        raw_paths = [self._trace_path(root_idx, tip, pred) for tip in selected_tips]

        # Order roots reproducibly: Patient Right/Mesial -> Left/Distal, Buccal -> Lingual
        paired = sorted(
            zip(selected_tips, raw_paths),
            key=lambda tp: (-float(pts_ras[tp[0], 0]), -float(pts_ras[tp[0], 1])),
        )

        # Detect bifurcation/furcation junctions where paths diverge
        junction_map: Dict[int, List[str]] = {}
        node_usage: Dict[int, List[str]] = {}
        for idx, (_tip, path) in enumerate(paired):
            cid = f"{roi.fdi}_C{idx + 1}"
            for node in path:
                node_usage.setdefault(node, []).append(cid)

        for _idx, path in enumerate(raw_paths):
            for k in range(len(path) - 1):
                u_n = path[k]
                v_next = path[k + 1]
                if len(node_usage.get(u_n, [])) > len(node_usage.get(v_next, [])):
                    junction_map[u_n] = list(node_usage[u_n])

        bifurcations = [
            CanalBifurcation(
                position_ras_mm=[float(v) for v in pts_ras[j_node]],
                geodesic_depth_mm=round(float(g_euclid[j_node]), 3),
                radius_mm=round(float(radii_all[j_node]), 3),
                child_canal_ids=sorted(set(cids)),
            )
            for j_node, cids in junction_map.items()
        ]
        bifurcations.sort(key=lambda b: b.geodesic_depth_mm)

        branches: List[RootCanalBranch] = []
        has_wall_data = bool(np.any(edt_wall_mm > 0.0))
        for idx, (_tip, path) in enumerate(paired):
            cid = f"{roi.fdi}_C{idx + 1}"
            root_lbl = f"Canal {idx + 1}" if len(paired) > 1 else "Main Canal"
            pts_b, r_b, w_b, b_len = _resample_branch_smooth(
                pts_ras[path], radii_all[path], walls_all[path]
            )
            curv = _compute_polyline_curvature_deg(pts_b)
            conf = float(roi.pulp_confidence or 0.9)
            if b_len < 3.0:
                conf *= 0.8
            branches.append(
                RootCanalBranch(
                    canal_id=cid,
                    root_label=root_lbl,
                    points_ras_mm=pts_b,
                    radii_mm=r_b,
                    wall_clearance_mm=w_b,
                    length_mm=b_len,
                    curvature_deg=curv,
                    min_radius_mm=float(np.min(r_b)),
                    mean_radius_mm=float(np.mean(r_b)),
                    max_radius_mm=float(np.max(r_b)),
                    min_wall_distance_mm=float(np.min(w_b)) if has_wall_data else None,
                    orifice_ras_mm=[float(v) for v in pts_b[0]],
                    apex_ras_mm=[float(v) for v in pts_b[-1]],
                    confidence=conf,
                    source_type=roi.pulp_source_type,
                )
            )

        return branches, bifurcations, chamber_ras

    @staticmethod
    def _trace_path(root_idx: int, tip_idx: int, pred: np.ndarray) -> List[int]:
        path: List[int] = []
        cur = int(tip_idx)
        guard = 0
        max_steps = int(len(pred)) + 2
        while cur >= 0 and guard < max_steps:
            path.append(cur)
            if cur == root_idx:
                break
            cur = int(pred[cur])
            guard += 1
        path.reverse()
        return path

    @staticmethod
    def _optimize_access_cavity(
        *,
        roi: EndoToothROI,
        edt_wall_mm: np.ndarray,
        canals: Sequence[RootCanalBranch],
        chamber_center_ras: Sequence[float],
        manual_entry_ras: Optional[Sequence[float]] = None,
        manual_target_ras: Optional[Sequence[float]] = None,
    ) -> Optional[AccessCavityPlan]:
        """Optimize straight-line coronal access corridor minimizing perforation risk."""
        if manual_entry_ras is not None and manual_target_ras is not None:
            e = np.asarray(manual_entry_ras, dtype=np.float64).reshape(3)
            t = np.asarray(manual_target_ras, dtype=np.float64).reshape(3)
            vec = t - e
            norm = float(np.linalg.norm(vec))
            u_dir = vec / max(1e-8, norm)
            wall_clr = EndodonticEngine._sample_corridor_wall_clearance(
                e, t, roi, edt_wall_mm
            )
            align_deg = EndodonticEngine._mean_orifice_alignment_deg(u_dir, canals)
            return AccessCavityPlan(
                entry_ras_mm=[float(v) for v in e],
                target_ras_mm=[float(v) for v in t],
                direction_ras=[float(v) for v in u_dir],
                recommended_diameter_mm=2.2 if len(canals) > 1 else 1.6,
                min_wall_clearance_mm=wall_clr,
                orifice_alignment_deg=align_deg,
                confidence=1.0,
                source_type=SOURCE_MANUALLY_CORRECTED,
                manual_edit=True,
                status_flag=CLINICAL_ENDO_STATUS,
            )

        if not np.any(roi.tooth_mask):
            return None

        target = np.asarray(chamber_center_ras, dtype=np.float64).reshape(3)
        tooth_zyx = np.argwhere(roi.tooth_mask)
        tooth_ras = _ijk_zyx_to_ras(tooth_zyx, roi.affine_ras)
        z_t = tooth_ras[:, 2]
        crown_mask = (
            z_t >= np.percentile(z_t, 82.0)
            if roi.is_mandible
            else z_t <= np.percentile(z_t, 18.0)
        )
        crown_pts = tooth_ras[crown_mask]
        if len(crown_pts) == 0:
            return None

        nominal_entry = np.mean(crown_pts, axis=0)
        nom_vec = target - nominal_entry
        corridor_len = float(np.linalg.norm(nom_vec))
        if corridor_len < 1.0:
            nominal_entry = target + np.array([0.0, 0.0, 4.0 if roi.is_mandible else -4.0])
            nom_vec = target - nominal_entry
            corridor_len = float(np.linalg.norm(nom_vec))
        u_nom = nom_vec / max(1e-8, corridor_len)

        # Build local tangent plane around u_nom to sweep small clinical tilt variations (+/- 12 deg)
        ref = np.array([1.0, 0.0, 0.0]) if abs(u_nom[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(u_nom, ref)
        b1 /= max(1e-8, float(np.linalg.norm(b1)))
        b2 = np.cross(u_nom, b1)

        best_score = -1e9
        best_entry = nominal_entry
        best_dir = u_nom
        best_wall = EndodonticEngine._sample_corridor_wall_clearance(
            nominal_entry, target, roi, edt_wall_mm
        )
        best_align = EndodonticEngine._mean_orifice_alignment_deg(u_nom, canals)

        for ax_deg in (-10.0, -5.0, 0.0, 5.0, 10.0):
            for ay_deg in (-10.0, -5.0, 0.0, 5.0, 10.0):
                rx = math.tan(math.radians(ax_deg))
                ry = math.tan(math.radians(ay_deg))
                u_cand = u_nom + rx * b1 + ry * b2
                u_cand /= max(1e-8, float(np.linalg.norm(u_cand)))
                cand_entry = target - u_cand * corridor_len
                w_clr = EndodonticEngine._sample_corridor_wall_clearance(
                    cand_entry, target, roi, edt_wall_mm
                )
                if w_clr is None:
                    continue
                align = EndodonticEngine._mean_orifice_alignment_deg(u_cand, canals)
                # Objective balances wall safety distance minus misalignment penalty
                score = w_clr - 0.04 * align
                if score > best_score:
                    best_score = score
                    best_entry = cand_entry
                    best_dir = u_cand
                    best_wall = w_clr
                    best_align = align

        return AccessCavityPlan(
            entry_ras_mm=[float(v) for v in best_entry],
            target_ras_mm=[float(v) for v in target],
            direction_ras=[float(v) for v in best_dir],
            recommended_diameter_mm=2.2 if len(canals) > 1 else 1.6,
            min_wall_clearance_mm=best_wall,
            orifice_alignment_deg=round(float(best_align), 2),
            confidence=float(roi.pulp_confidence or 0.85),
            source_type=roi.pulp_source_type,
            manual_edit=False,
            status_flag=CLINICAL_ENDO_STATUS,
        )

    @staticmethod
    def _sample_corridor_wall_clearance(
        entry_ras: np.ndarray,
        target_ras: np.ndarray,
        roi: EndoToothROI,
        edt_wall_mm: np.ndarray,
    ) -> Optional[float]:
        if not np.any(edt_wall_mm > 0):
            return None
        ts = np.linspace(0.05, 1.0, 16)[:, None]
        pts_ras = entry_ras[None, :] + ts * (target_ras - entry_ras)[None, :]
        zyx = np.rint(_ras_to_ijk_zyx(pts_ras, roi.affine_ras)).astype(int)
        nz, ny, nx = edt_wall_mm.shape
        in_b = (
            (zyx[:, 0] >= 0)
            & (zyx[:, 0] < nz)
            & (zyx[:, 1] >= 0)
            & (zyx[:, 1] < ny)
            & (zyx[:, 2] >= 0)
            & (zyx[:, 2] < nx)
        )
        if not np.any(in_b):
            return None
        vals = edt_wall_mm[zyx[in_b, 0], zyx[in_b, 1], zyx[in_b, 2]]
        return float(np.min(vals))

    @staticmethod
    def _mean_orifice_alignment_deg(
        access_dir: np.ndarray, canals: Sequence[RootCanalBranch]
    ) -> float:
        if not canals:
            return 0.0
        angles = []
        for c in canals:
            if len(c.points_ras_mm) >= 2:
                k_end = min(len(c.points_ras_mm) - 1, max(1, len(c.points_ras_mm) // 3))
                v = np.asarray(c.points_ras_mm[k_end] - c.points_ras_mm[0], dtype=np.float64)
                n = float(np.linalg.norm(v))
                if n > 1e-6:
                    dot = float(np.clip(np.dot(access_dir, v / n), -1.0, 1.0))
                    angles.append(math.degrees(math.acos(dot)))
        return float(np.mean(angles)) if angles else 0.0

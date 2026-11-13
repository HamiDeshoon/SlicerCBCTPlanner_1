"""Mandibular Canal (IAN) & Root-Canal Tracing, Tube Visualization, and Safety Proximity Engine (F1).

Provides:
  - Automated centerline + local radius extraction for Left and Right Inferior Alveolar Canals
    from DentalSegmentator class 5 (`Mandibular canal`) or ToothFairy3 (`LEFT_IAN_ID`, `RIGHT_IAN_ID`).
  - Manual canal trace tool and endodontic root-canal trace tool using Slicer Markups Curve nodes.
  - 2D slice intersection and 3D tube visualization across Panoramic (`Green`), Cross-Sections
    (`Yellow`, `YellowMinus`, `YellowPlus`), Axial (`Red`), and 3D views.
  - Configurable safety margin (default `2.0 mm`) and OnDemand3D-style proximity warning evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

from .core import ensure_optional_package
from .core.arch_detector import project_point_to_polyline_arc_mm
from .core.labels import (
    DENTALSEGMENTATOR_CANAL_ID,
    FDI_TO_PULP_ID,
    LEFT_IAN_ID,
    RIGHT_IAN_ID,
)
from .core.mask_centerline import _centerline_from_binary
from .core.volume_loader import CBCTVolume


@dataclass
class CanalTraceData:
    """Extracted or manually traced canal curve with per-control-point radii in mm."""

    side: str  # "Right", "Left", or "Root_<FDI>"
    control_points_ras: np.ndarray
    radii_mm: np.ndarray
    mean_radius_mm: float
    source: str  # "segmentation" or "manual"


@dataclass
class CanalProximityResult:
    """F1: Result of evaluating proximity between a target point/implant and a mandibular canal."""

    side: str
    min_centerline_distance_mm: float
    min_surface_distance_mm: float
    safety_margin_mm: float
    is_violation: bool
    closest_canal_point_ras: np.ndarray
    warning_message: str


class CanalTracerLogic:
    """Manages mandibular canal (IAN) and root-canal extraction, manual tracing, and proximity alerts."""

    DEFAULT_SAFETY_MARGIN_MM: float = 2.0
    DEFAULT_CANAL_RADIUS_MM: float = 1.25

    def __init__(self) -> None:
        self.canals_by_side: Dict[str, CanalTraceData] = {}
        self.root_canals_by_fdi: Dict[str, CanalTraceData] = {}

    @classmethod
    def extract_canals_from_mask(
        cls,
        volume: CBCTVolume,
        mask: np.ndarray,
        is_5class: bool = False,
    ) -> Dict[str, CanalTraceData]:
        """F1: Extract Right (+X in RAS) and Left (-X in RAS) mandibular canal centerlines and local radii."""
        results: Dict[str, CanalTraceData] = {}
        scipy_nd = ensure_optional_package("scipy.ndimage", "scipy", "canal radius distance transform")

        if is_5class:
            canal_vox = mask == DENTALSEGMENTATOR_CANAL_ID
            if not canal_vox.any():
                return results
            # Split into Right (X_RAS >= 0) and Left (X_RAS < 0)
            z_idx, y_idx, x_idx = np.where(canal_vox)
            vox_pts = np.column_stack([x_idx, y_idx, z_idx])
            world_pts = volume.voxel_to_world(vox_pts)
            right_mask = np.zeros_like(canal_vox, dtype=bool)
            left_mask = np.zeros_like(canal_vox, dtype=bool)
            right_sel = world_pts[:, 0] >= 0.0
            left_sel = ~right_sel
            if np.any(right_sel):
                right_mask[z_idx[right_sel], y_idx[right_sel], x_idx[right_sel]] = True
            if np.any(left_sel):
                left_mask[z_idx[left_sel], y_idx[left_sel], x_idx[left_sel]] = True
            side_masks = [("Right", right_mask), ("Left", left_mask)]
        else:
            right_mask = mask == RIGHT_IAN_ID
            left_mask = mask == LEFT_IAN_ID
            if not right_mask.any() and not left_mask.any() and (mask == DENTALSEGMENTATOR_CANAL_ID).any():
                return cls.extract_canals_from_mask(volume, mask, is_5class=True)
            side_masks = [("Right", right_mask), ("Left", left_mask)]

        sz, sy, sx = volume.spacing
        for side_name, bin_mask in side_masks:
            if int(np.count_nonzero(bin_mask)) < 8:
                continue
            # Compute internal EDT inside canal mask to estimate local canal radius
            dt_inside = scipy_nd.distance_transform_edt(bin_mask, sampling=(sz, sy, sx))
            ctrl_pts: Optional[np.ndarray] = None
            try:
                cl_res = _centerline_from_binary(bin_mask, volume.affine, max_ctrl=30)
                if cl_res is not None:
                    ctrl_pts = cl_res[0].control_points
            except Exception:
                ctrl_pts = None

            if ctrl_pts is None or len(ctrl_pts) < 4:
                # Fallback: order canal voxels along principal axis (posterior -> anterior)
                bz, by, bx = np.where(bin_mask)
                vox_c = np.column_stack([bx, by, bz])
                w_pts = volume.voxel_to_world(vox_c)
                order = np.argsort(w_pts[:, 1])  # Posterior (-Y) to Anterior (+Y)
                w_sorted = w_pts[order]
                n_bins = min(16, max(4, len(w_sorted) // 4))
                chunks = np.array_split(w_sorted, n_bins)
                ctrl_pts = np.array([np.median(ch, axis=0) for ch in chunks if len(ch) > 0], dtype=np.float64)

            if len(ctrl_pts) < 2:
                continue

            vox_ctrl = volume.world_to_voxel(ctrl_pts)
            radii: List[float] = []
            for vx, vy, vz in vox_ctrl:
                ix, iy, iz = int(round(vx)), int(round(vy)), int(round(vz))
                if (
                    0 <= iz < dt_inside.shape[0]
                    and 0 <= iy < dt_inside.shape[1]
                    and 0 <= ix < dt_inside.shape[2]
                ):
                    r_val = float(dt_inside[iz, iy, ix])
                    radii.append(max(0.6, r_val if r_val > 0.2 else cls.DEFAULT_CANAL_RADIUS_MM))
                else:
                    radii.append(cls.DEFAULT_CANAL_RADIUS_MM)

            radii_arr = np.array(radii, dtype=np.float64)
            results[side_name] = CanalTraceData(
                side=side_name,
                control_points_ras=ctrl_pts,
                radii_mm=radii_arr,
                mean_radius_mm=round(float(np.mean(radii_arr)), 2),
                source="segmentation",
            )
        return results

    @classmethod
    def extract_root_canal_for_tooth(
        cls,
        fdi: int,
        volume: CBCTVolume,
        mask: np.ndarray,
    ) -> Optional[CanalTraceData]:
        """F1: Extract root-canal centerline from segmented pulp mask (`FDI_TO_PULP_ID`) when present."""
        fdi_str = str(fdi)
        pulp_id = FDI_TO_PULP_ID.get(fdi_str)
        if pulp_id is None:
            return None
        bin_pulp = mask == pulp_id
        if int(np.count_nonzero(bin_pulp)) < 4:
            return None
        pz, py, px = np.where(bin_pulp)
        w_pts = volume.voxel_to_world(np.column_stack([px, py, pz]))
        # Order along Superior-Inferior axis (Z) from crown to apex
        is_lower = int(fdi) >= 31
        order = np.argsort(w_pts[:, 2])[::-1] if is_lower else np.argsort(w_pts[:, 2])
        w_sorted = w_pts[order]
        n_bins = min(8, max(4, len(w_sorted) // 2))
        chunks = np.array_split(w_sorted, n_bins)
        ctrl = np.array([np.mean(ch, axis=0) for ch in chunks if len(ch) > 0], dtype=np.float64)
        radii = np.full(len(ctrl), 0.5, dtype=np.float64)
        return CanalTraceData(
            side=f"Root_{fdi_str}",
            control_points_ras=ctrl,
            radii_mm=radii,
            mean_radius_mm=0.5,
            source="segmentation",
        )

    @staticmethod
    def evaluate_canal_proximity(
        query_points_ras: np.ndarray,
        canal: CanalTraceData,
        safety_margin_mm: float = 2.0,
        query_radius_mm: float = 0.0,
    ) -> CanalProximityResult:
        """F1: Compute shortest centerline and surface-to-surface distance to a canal and check safety margin."""
        q_pts = np.atleast_2d(np.asarray(query_points_ras, dtype=np.float64))
        c_pts = np.asarray(canal.control_points_ras, dtype=np.float64)
        diffs = np.linalg.norm(np.diff(c_pts, axis=0), axis=1)
        cum_s = np.concatenate([[0.0], np.cumsum(diffs)])

        best_center_dist = float("inf")
        best_surf_dist = float("inf")
        best_canal_pt = c_pts[0].copy()

        for q_pt in q_pts:
            arc_s, d_center, closest_pt = project_point_to_polyline_arc_mm(c_pts, cum_s, q_pt)
            local_r = float(np.interp(arc_s, cum_s, canal.radii_mm)) if len(canal.radii_mm) == len(c_pts) else canal.mean_radius_mm
            d_surf = max(0.0, d_center - local_r - float(query_radius_mm))
            if d_surf < best_surf_dist:
                best_surf_dist = d_surf
                best_center_dist = d_center
                best_canal_pt = closest_pt

        margin = float(safety_margin_mm)
        violation = best_surf_dist < margin
        if violation:
            msg = (
                f"WARNING: {canal.side} Mandibular Canal clearance is {best_surf_dist:.2f} mm "
                f"(< {margin:.1f} mm safety margin)!"
            )
        else:
            msg = f"{canal.side} Mandibular Canal clearance OK ({best_surf_dist:.2f} mm >= {margin:.1f} mm)."

        return CanalProximityResult(
            side=canal.side,
            min_centerline_distance_mm=round(best_center_dist, 2),
            min_surface_distance_mm=round(best_surf_dist, 2),
            safety_margin_mm=round(margin, 2),
            is_violation=violation,
            closest_canal_point_ras=best_canal_pt,
            warning_message=msg,
        )

    def create_or_update_canal_curve_node(
        self,
        volumeNode: Any,
        canal: CanalTraceData,
        parameterNode: Any = None,
        visible: bool = True,
    ) -> Any:
        """F1: Create or update a MRML curve node with tube diameter (`2 * mean_radius_mm`) in 2D and 3D views."""
        if slicer is None:
            return None
        role = f"{canal.side}CanalCurve"
        curveNode = None
        if parameterNode is not None and hasattr(parameterNode, "GetNodeReference"):
            curveNode = parameterNode.GetNodeReference(role)
        if curveNode is None:
            vol_name = volumeNode.GetName() if volumeNode else "CBCT"
            name = f"{vol_name}_{canal.side}_MandibularCanal"
            curveNode = slicer.mrmlScene.GetFirstNodeByName(name)
            if not curveNode:
                curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
                curveNode.CreateDefaultDisplayNodes()
            if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
                parameterNode.SetNodeReferenceID(role, curveNode.GetID())

        if (
            volumeNode is not None
            and hasattr(volumeNode, "GetTransformNodeID")
            and hasattr(curveNode, "SetAndObserveTransformNodeID")
        ):
            tf_id = volumeNode.GetTransformNodeID()
            if tf_id:
                curveNode.SetAndObserveTransformNodeID(tf_id)

        was_mod = curveNode.StartModify() if hasattr(curveNode, "StartModify") else None
        try:
            curveNode.SetAttribute("CBCTPlanner.CanalSide", canal.side)
            curveNode.SetAttribute("CBCTPlanner.MeanRadiusMm", f"{canal.mean_radius_mm:.2f}")
            curveNode.RemoveAllControlPoints()
            for pt in canal.control_points_ras:
                curveNode.AddControlPoint(float(pt[0]), float(pt[1]), float(pt[2]))
        finally:
            if was_mod is not None and hasattr(curveNode, "EndModify"):
                curveNode.EndModify(was_mod)

        dNode = curveNode.GetDisplayNode()
        if dNode:
            d_mod = dNode.StartModify() if hasattr(dNode, "StartModify") else None
            try:
                is_root = canal.side.startswith("Root_")
                color = (0.95, 0.45, 0.15) if not is_root else (0.95, 0.20, 0.65)
                dNode.SetColor(*color)
                dNode.SetSelectedColor(*color)
                if hasattr(slicer.vtkMRMLMarkupsDisplayNode, "UseLineDiameter"):
                    dNode.SetCurveLineSizeMode(slicer.vtkMRMLMarkupsDisplayNode.UseLineDiameter)
                dNode.SetLineDiameter(max(1.0, float(2.0 * canal.mean_radius_mm)))
                dNode.SetVisibility(bool(visible))
                dNode.SetVisibility2D(bool(visible))
                dNode.SetVisibility3D(bool(visible))
                if hasattr(dNode, "SetSliceProjection"):
                    dNode.SetSliceProjection(True)
                if hasattr(dNode, "SetPointLabelsVisibility"):
                    dNode.SetPointLabelsVisibility(False)
            finally:
                if d_mod is not None and hasattr(dNode, "EndModify"):
                    dNode.EndModify(d_mod)
        return curveNode

    def start_manual_canal_trace(
        self,
        volumeNode: Any,
        side: str = "Right",
        parameterNode: Any = None,
    ) -> Any:
        """F1: Enter interactive Slicer Markups Place Mode to trace a Mandibular Canal or Root Canal."""
        if slicer is None or not volumeNode:
            return None
        role = f"{side}CanalCurve"
        curveNode = None
        if parameterNode is not None and hasattr(parameterNode, "GetNodeReference"):
            curveNode = parameterNode.GetNodeReference(role)
        if curveNode is None:
            name = f"{volumeNode.GetName()}_{side}_MandibularCanal"
            curveNode = slicer.mrmlScene.GetFirstNodeByName(name)
            if not curveNode:
                curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
                curveNode.CreateDefaultDisplayNodes()
            if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
                parameterNode.SetNodeReferenceID(role, curveNode.GetID())

        dNode = curveNode.GetDisplayNode()
        if dNode:
            dNode.SetColor(0.95, 0.35, 0.15)
            dNode.SetVisibility(True)
            dNode.SetVisibility2D(True)
            dNode.SetVisibility3D(True)

        if getattr(slicer, "app", None) is not None:
            selNode = slicer.app.applicationLogic().GetSelectionNode()
            selNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsCurveNode")
            selNode.SetActivePlaceNodeID(curveNode.GetID())
            intNode = slicer.app.applicationLogic().GetInteractionNode()
            intNode.SetPlaceModePersistence(1)
            intNode.SetCurrentInteractionMode(intNode.Place)
        return curveNode

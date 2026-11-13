"""Cross-Sectional Slice Navigator, Dynamic Multi-Slice Lightbox, & +-30° Tilt Control.

Fixes & Upgrades:
  - B9: Bounded LRU polydata cache (`max_entries=8`).
  - B11: Updates Axial (`Red`) slice offset ONLY when `sync_axial_slice=True` (on tooth jump),
    so the user can freely scroll the axial slice while the cursor is active.
  - B12: Explicit `slab_enabled: bool` and `slab_mode: str` (`Mean` default, `MIP`, `MinIP`) setting
    `SetSlabReconstructionEnabled` and `SetSlabReconstructionType` on slice nodes.
  - B15 & O5: Resolves cursor line nodes and tick model via ParameterNode reference IDs (never
    orphaned when volume is renamed).
  - O2 & O7: Coalesces rapid `PointModifiedEvent` notifications via a ~30 ms single-shot timer,
    evaluates the central frame once, updates only dirty nodes inside `StartModify`/`EndModify`,
    keeps persistent observers, and skips rewriting the actively dragged line unless snap delta
    exceeds `0.05 mm`.
  - O4: `CursorSettings` dataclass replacing repeated multi-argument cursor calls.
  - F5: Dynamic $N$-slice Lightbox offsets and line intensity/density profile (`compute_density_profile_along_line`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import qt  # type: ignore
    import slicer  # type: ignore
except ImportError:
    qt = None
    slicer = None

try:
    import vtk  # type: ignore
except ImportError:
    vtk = None

from .PanoramicCurveLogic import PanoramicCurveLogic
from .Utils import BoundedLRUCache
from .core.volume_loader import CBCTVolume

MAX_TILT_DEG = 30.0
SNAP_EPSILON_MM = 0.05

# Colors matching the custom Lightbox slice view headers:
COLOR_CS_MINUS = (0.96, 0.62, 0.04)  # Amber (#F59E0B) for -step_mm
COLOR_CS_ZERO = (0.00, 0.90, 1.00)   # Cyan (#06B6D4) for 0.0 mm
COLOR_CS_PLUS = (0.06, 0.73, 0.51)   # Emerald (#10B981) for +step_mm


@dataclass
class CursorSettings:
    """O4: Consolidated cursor, slab, tilt, and lightbox settings."""

    width_mm: float = 24.0
    tick_spacing_mm: float = 2.0
    show_all_ticks: bool = True
    slab_enabled: bool = False
    slab_mode: str = "Mean"  # "Mean", "MIP", "MinIP"
    slab_thickness_mm: float = 0.5
    tilt_deg: float = 0.0
    lightbox_enabled: bool = True
    lightbox_step_mm: float = 2.0
    lightbox_count: int = 3
    sync_axial_slice: bool = False
    site_label: Optional[str] = None


class CrossSectionNavigator:
    """Manages interactive cross-sectional reslicing, dynamic lightbox, and +-30° tilt."""

    def __init__(
        self,
        panoramic_logic: Optional[PanoramicCurveLogic] = None,
        max_cache_entries: int = 8,
    ) -> None:
        self.panoramic = panoramic_logic or PanoramicCurveLogic(max_cache_entries=max_cache_entries)
        self._is_updating: bool = False
        # B9: Bounded LRU cache for tick fan polydata
        self._poly_cache: BoundedLRUCache[
            Tuple[str, int, float, float, Tuple[float, float, float]], Any
        ] = BoundedLRUCache(max_entries=max_cache_entries)
        # O2: Persistent observer registry keyed by node ID
        self._line_observers: List[Tuple[Any, int]] = []
        self._observed_node_ids: Dict[str, Any] = {}
        # O5: Node ID cache by role when no ParameterNode is supplied
        self._node_id_by_role: Dict[str, str] = {}
        # O2: Dirty-endpoint cache by role -> (pt0_tuple, pt1_tuple, label0, label1, visible_2d, visible_3d)
        self._last_line_state: Dict[str, Tuple[Any, ...]] = {}
        self._active_drag_node_id: Optional[str] = None
        self._pending_drag_arc_mm: Optional[float] = None
        self._drag_coalesce_timer: Any = None
        self._cursor_changed_callback: Optional[Callable[[float], None]] = None
        self._active_curve_node: Any = None
        self._active_occlusal_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)

        if qt is not None:
            try:
                self._drag_coalesce_timer = qt.QTimer()
                self._drag_coalesce_timer.setSingleShot(True)
                self._drag_coalesce_timer.setInterval(30)
                self._drag_coalesce_timer.timeout.connect(self._flush_coalesced_drag)
            except Exception:
                self._drag_coalesce_timer = None

    def set_cursor_changed_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        """Register callback invoked when the user drags the Axial or Panoramic cursor line."""
        self._cursor_changed_callback = callback

    def _flush_coalesced_drag(self) -> None:
        arc_val = self._pending_drag_arc_mm
        self._pending_drag_arc_mm = None
        if arc_val is not None and self._cursor_changed_callback is not None:
            try:
                self._cursor_changed_callback(arc_val)
            finally:
                self._active_drag_node_id = None

    def _emit_cursor_drag(self, new_arc_mm: float, source_node_id: Optional[str] = None) -> None:
        """O2: Coalesce rapid drag events via 30 ms single-shot timer when running in Qt event loop."""
        self._active_drag_node_id = source_node_id
        self._pending_drag_arc_mm = float(new_arc_mm)
        if (
            self._drag_coalesce_timer is not None
            and slicer is not None
            and getattr(slicer, "app", None) is not None
        ):
            self._drag_coalesce_timer.start()
        else:
            self._flush_coalesced_drag()

    def remove_line_observers(self) -> None:
        """Detach VTK observers from interactive cursor lines."""
        if self._drag_coalesce_timer is not None:
            try:
                self._drag_coalesce_timer.stop()
            except Exception:
                pass
        for node, tag in self._line_observers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._line_observers.clear()
        self._observed_node_ids.clear()

    @staticmethod
    def clamp_tilt_deg(tilt_deg: float) -> float:
        """Clamp cross-section tilt angle to [-30.0, +30.0] degrees."""
        return float(np.clip(float(tilt_deg), -MAX_TILT_DEG, MAX_TILT_DEG))

    @classmethod
    def apply_cross_section_tilt(
        cls,
        T: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        tilt_deg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate (T, B) by `tilt_deg` (clamped to [-30°, +30°]) around Buccal-Lingual normal `N`."""
        clamped = cls.clamp_tilt_deg(tilt_deg)
        t_unit = np.asarray(T, dtype=np.float64) / max(1e-8, float(np.linalg.norm(T)))
        n_unit = np.asarray(N, dtype=np.float64) / max(1e-8, float(np.linalg.norm(N)))
        b_unit = np.asarray(B, dtype=np.float64) / max(1e-8, float(np.linalg.norm(B)))
        if abs(clamped) < 1e-6:
            return t_unit, n_unit, b_unit

        theta = math.radians(clamped)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        t_tilted = cos_t * t_unit - sin_t * b_unit
        b_tilted = cos_t * b_unit + sin_t * t_unit
        t_tilted /= max(1e-8, float(np.linalg.norm(t_tilted)))
        b_tilted /= max(1e-8, float(np.linalg.norm(b_tilted)))
        return t_tilted, n_unit, b_tilted

    @classmethod
    def compute_pano_cursor_endpoints(
        cls,
        arc_mm: float,
        pano_height_mm: float = 50.0,
        tilt_deg: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute bottom and top 3D RAS endpoints for a panoramic cursor line at `arc_mm` tilted by `tilt_deg`."""
        clamped_tilt = cls.clamp_tilt_deg(tilt_deg)
        theta = math.radians(clamped_tilt)
        x_center = PanoramicCurveLogic.pano_x(arc_mm)
        half_h = 0.5 * float(pano_height_mm)
        dx = half_h * math.tan(theta)
        pt_bottom = (float(x_center + dx), 0.0, 0.0)
        pt_top = (float(x_center - dx), float(pano_height_mm), 0.0)
        return pt_bottom, pt_top

    @staticmethod
    def compute_lightbox_offsets(
        center_arc_mm: float,
        total_len_mm: float,
        step_mm: float = 2.0,
    ) -> Tuple[float, float, float]:
        """Return clamped `(arc_minus, arc_zero, arc_plus)` in `[0.0, total_len_mm]` for `(-step_mm, 0, +step_mm)`."""
        step = max(0.1, float(step_mm))
        s0 = float(np.clip(center_arc_mm, 0.0, total_len_mm))
        s_minus = float(np.clip(s0 - step, 0.0, total_len_mm))
        s_plus = float(np.clip(s0 + step, 0.0, total_len_mm))
        return s_minus, s0, s_plus

    @staticmethod
    def compute_dynamic_lightbox_offsets(
        center_arc_mm: float,
        total_len_mm: float,
        step_mm: float = 2.0,
        count: int = 3,
    ) -> List[float]:
        """F5: Compute $N$ consecutive clamped cross-section arc positions centered at `center_arc_mm`."""
        n = max(1, int(count))
        step = max(0.1, float(step_mm))
        s0 = float(np.clip(center_arc_mm, 0.0, total_len_mm))
        half = n // 2
        offsets = [
            round(float(np.clip(s0 + (i - half) * step, 0.0, total_len_mm)), 3)
            for i in range(n)
        ]
        return offsets

    @staticmethod
    def compute_density_profile_along_line(
        volume: CBCTVolume,
        pt0_ras: Sequence[float],
        pt1_ras: Sequence[float],
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """F5: Sample CBCT intensity (approximate gray/HU) along a 3D line segment `[pt0_ras, pt1_ras]`."""
        p0 = np.asarray(pt0_ras, dtype=np.float64).reshape(3)
        p1 = np.asarray(pt1_ras, dtype=np.float64).reshape(3)
        n = max(2, int(num_samples))
        length_mm = float(np.linalg.norm(p1 - p0))
        t_vals = np.linspace(0.0, 1.0, n)
        dists_mm = t_vals * length_mm
        pts = p0[None, :] + t_vals[:, None] * (p1 - p0)[None, :]
        intensities = volume.sample_world_points(pts, order=1, cval=-1000.0)
        return dists_mm, intensities

    @staticmethod
    def _get_view_node_id(slice_or_view_name: str) -> Optional[str]:
        if slicer is None or not getattr(slicer, "app", None):
            return None
        lm = slicer.app.layoutManager()
        if not lm:
            return None
        if slice_or_view_name in ("Red", "Yellow", "YellowMinus", "YellowPlus", "Green"):
            sw = lm.sliceWidget(slice_or_view_name)
            if sw and sw.mrmlSliceNode():
                return sw.mrmlSliceNode().GetID()
        elif slice_or_view_name == "3D":
            if getattr(lm, "threeDViewCount", 0) > 0:
                tw = lm.threeDWidget(0)
                if tw and tw.mrmlViewNode():
                    return tw.mrmlViewNode().GetID()
        return None

    @staticmethod
    def is_panoramic_shown_in_green() -> bool:
        """Return True if the Green slice view is currently displaying a generated Panoramic volume."""
        if slicer is None or not getattr(slicer, "app", None):
            return False
        lm = slicer.app.layoutManager()
        if not lm:
            return False
        green = lm.sliceWidget("Green")
        if not green:
            return False
        comp = green.mrmlSliceCompositeNode()
        bg_id = comp.GetBackgroundVolumeID() if comp else None
        if not bg_id:
            return False
        bg_node = slicer.mrmlScene.GetNodeByID(bg_id)
        if not bg_node:
            return False
        return (
            bg_node.GetAttribute("CBCTPlanner.IsPanoramic") == "1"
            or (bg_node.GetName() or "").endswith("_Panoramic")
        )

    @staticmethod
    def _build_tick_lines_polydata(
        pts: np.ndarray,
        normals: np.ndarray,
        width_mm: float,
    ) -> Any:
        """Build vtkPolyData containing perpendicular cross-section tick segments."""
        if vtk is None:
            return None
        vtk_pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        half_w = width_mm / 2.0

        for p, n in zip(pts, normals):
            p1 = p - n * half_w
            p2 = p + n * half_w
            id1 = vtk_pts.InsertNextPoint(float(p1[0]), float(p1[1]), float(p1[2]))
            id2 = vtk_pts.InsertNextPoint(float(p2[0]), float(p2[1]), float(p2[2]))
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, id1)
            line.GetPointIds().SetId(1, id2)
            lines.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetLines(lines)
        return poly

    def _resolve_or_create_node_by_role(
        self,
        role: str,
        class_name: str,
        default_name: str,
        parameterNode: Any = None,
    ) -> Any:
        """B15 & O5: Resolve helper node by ParameterNode reference ID or cached ID before name fallback."""
        if slicer is None or getattr(slicer, "mrmlScene", None) is None:
            return None
        if parameterNode is not None and hasattr(parameterNode, "GetNodeReference"):
            node = parameterNode.GetNodeReference(role)
            if node is not None:
                self._node_id_by_role[role] = node.GetID()
                return node
        cached_id = self._node_id_by_role.get(role)
        if cached_id and hasattr(slicer.mrmlScene, "GetNodeByID"):
            node = slicer.mrmlScene.GetNodeByID(cached_id)
            if node is not None:
                if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
                    parameterNode.SetNodeReferenceID(role, node.GetID())
                return node
        node = slicer.mrmlScene.GetFirstNodeByName(default_name)
        if not node:
            node = slicer.mrmlScene.AddNewNodeByClass(class_name, default_name)
            if hasattr(node, "CreateDefaultDisplayNodes"):
                node.CreateDefaultDisplayNodes()
        if node is not None:
            self._node_id_by_role[role] = node.GetID()
            if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
                parameterNode.SetNodeReferenceID(role, node.GetID())
        return node

    def _ensure_interactive_observers(
        self,
        activeAxialNode: Any,
        panoCursorNode: Optional[Any],
        curveNode: Any,
        occlusal_normal: Sequence[float],
    ) -> None:
        """O2: Attach persistent interaction observers once per cursor line node ID."""
        if vtk is None:
            return
        self._active_curve_node = curveNode
        self._active_occlusal_normal = (
            float(occlusal_normal[0]),
            float(occlusal_normal[1]),
            float(occlusal_normal[2]),
        )

        def _on_axial_line_dragged(caller: Any, _event: Any) -> None:
            if self._is_updating or not self._active_curve_node:
                return
            if caller.GetNumberOfControlPoints() < 2:
                return
            p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(0, p0)
            caller.GetNthControlPointPositionWorld(1, p1)
            mid = 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))
            new_arc_mm = self.panoramic.project_point_to_curve_arc_mm(
                self._active_curve_node, mid, occlusal_normal=self._active_occlusal_normal
            )
            c_id = caller.GetID() if hasattr(caller, "GetID") else None
            self._emit_cursor_drag(new_arc_mm, source_node_id=c_id)

        def _on_pano_line_dragged(caller: Any, _event: Any) -> None:
            if self._is_updating or not self._active_curve_node:
                return
            if caller.GetNumberOfControlPoints() < 2:
                return
            p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(0, p0)
            caller.GetNthControlPointPositionWorld(1, p1)
            mean_x = 0.5 * (float(p0[0]) + float(p1[0]))
            _, arc_s, _, _, _ = self.panoramic.get_curve_points_and_frames(
                self._active_curve_node,
                num_samples=50,
                occlusal_normal=self._active_occlusal_normal,
            )
            total_len = float(arc_s[-1])
            new_arc_mm = PanoramicCurveLogic.arc_mm(mean_x, max_arc_mm=total_len)
            c_id = caller.GetID() if hasattr(caller, "GetID") else None
            self._emit_cursor_drag(new_arc_mm, source_node_id=c_id)

        for node, handler in (
            (activeAxialNode, _on_axial_line_dragged),
            (panoCursorNode, _on_pano_line_dragged),
        ):
            if node is None:
                continue
            nid = node.GetID() if hasattr(node, "GetID") else str(id(node))
            if nid in self._observed_node_ids:
                continue
            point_mod_evt = getattr(getattr(slicer, "vtkMRMLMarkupsNode", None), "PointModifiedEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33))
            end_inter_evt = getattr(vtk.vtkCommand, "EndInteractionEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33))
            for evt in (point_mod_evt, end_inter_evt):
                tag = node.AddObserver(evt, handler)
                self._line_observers.append((node, tag))
            self._observed_node_ids[nid] = node

    def _update_or_create_markups_line(
        self,
        role: str,
        node_name: str,
        pt0: Sequence[float],
        pt1: Sequence[float],
        color: Tuple[float, float, float],
        label0: str,
        label1: str,
        line_diameter: float,
        visible_2d: bool,
        visible_3d: bool,
        allowed_view_ids: List[str],
        locked: bool = False,
        parameterNode: Any = None,
    ) -> Any:
        """O2, O5, O7, B15: Resolve line by role ID, skip no-op updates, and batch with StartModify/EndModify."""
        if slicer is None:
            return None
        lineNode = self._resolve_or_create_node_by_role(
            role=role,
            class_name="vtkMRMLMarkupsLineNode",
            default_name=node_name,
            parameterNode=parameterNode,
        )
        if not lineNode:
            return None

        p0_arr = np.asarray(pt0, dtype=np.float64)
        p1_arr = np.asarray(pt1, dtype=np.float64)

        # O2: If this line is currently being dragged by the user, do not rewrite its endpoints
        # unless the snap delta exceeds SNAP_EPSILON_MM.
        nid = lineNode.GetID() if hasattr(lineNode, "GetID") else str(id(lineNode))
        if (
            self._active_drag_node_id is not None
            and nid == self._active_drag_node_id
            and lineNode.GetNumberOfControlPoints() == 2
        ):
            cur0, cur1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            lineNode.GetNthControlPointPositionWorld(0, cur0)
            lineNode.GetNthControlPointPositionWorld(1, cur1)
            delta = max(
                float(np.linalg.norm(np.asarray(cur0) - p0_arr)),
                float(np.linalg.norm(np.asarray(cur1) - p1_arr)),
            )
            if delta < SNAP_EPSILON_MM:
                return lineNode

        # O2: Dirty check — skip modifying node if endpoints, labels, and visibility are unchanged
        state_sig = (
            tuple(np.round(p0_arr, 4)),
            tuple(np.round(p1_arr, 4)),
            label0,
            label1,
            bool(visible_2d),
            bool(visible_3d),
            tuple(allowed_view_ids),
            bool(locked),
        )
        if self._last_line_state.get(role) == state_sig and lineNode.GetNumberOfControlPoints() == 2:
            return lineNode
        self._last_line_state[role] = state_sig

        was_mod = lineNode.StartModify() if hasattr(lineNode, "StartModify") else None
        try:
            if hasattr(lineNode, "SetLocked"):
                lineNode.SetLocked(bool(locked))

            if lineNode.GetNumberOfControlPoints() != 2:
                lineNode.RemoveAllControlPoints()
                lineNode.AddControlPoint(float(p0_arr[0]), float(p0_arr[1]), float(p0_arr[2]))
                lineNode.AddControlPoint(float(p1_arr[0]), float(p1_arr[1]), float(p1_arr[2]))
            else:
                lineNode.SetNthControlPointPositionWorld(
                    0, float(p0_arr[0]), float(p0_arr[1]), float(p0_arr[2])
                )
                lineNode.SetNthControlPointPositionWorld(
                    1, float(p1_arr[0]), float(p1_arr[1]), float(p1_arr[2])
                )

            if hasattr(lineNode, "SetNthControlPointLabel"):
                lineNode.SetNthControlPointLabel(0, label0)
                lineNode.SetNthControlPointLabel(1, label1)
        finally:
            if was_mod is not None and hasattr(lineNode, "EndModify"):
                lineNode.EndModify(was_mod)

        disp = lineNode.GetDisplayNode()
        if disp:
            d_mod = disp.StartModify() if hasattr(disp, "StartModify") else None
            try:
                disp.SetColor(*color)
                disp.SetSelectedColor(*color)
                if hasattr(slicer.vtkMRMLMarkupsDisplayNode, "UseLineDiameter"):
                    disp.SetCurveLineSizeMode(slicer.vtkMRMLMarkupsDisplayNode.UseLineDiameter)
                disp.SetLineDiameter(float(line_diameter))
                disp.SetVisibility(bool(visible_2d or visible_3d))
                disp.SetVisibility2D(bool(visible_2d))
                disp.SetVisibility3D(bool(visible_3d))
                if hasattr(disp, "SetPropertiesLabelVisibility"):
                    disp.SetPropertiesLabelVisibility(False)
                if hasattr(disp, "SetPointLabelsVisibility"):
                    disp.SetPointLabelsVisibility(bool(label0 or label1))
                if allowed_view_ids and hasattr(disp, "SetViewNodeIDs"):
                    disp.SetViewNodeIDs(allowed_view_ids)
            finally:
                if d_mod is not None and hasattr(disp, "EndModify"):
                    disp.EndModify(d_mod)
        return lineNode

    def update_cursor_at_arc_mm(
        self,
        volumeNode: Any,
        curveNode: Any,
        arc_mm: float,
        width_mm: float = 24.0,
        tick_spacing_mm: float = 2.0,
        show_all_ticks: bool = True,
        slab_thickness_mm: float = 0.5,
        tilt_deg: float = 0.0,
        lightbox_enabled: bool = True,
        lightbox_step_mm: float = 2.0,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
        site_label: Optional[str] = None,
        panoNode: Optional[Any] = None,
        slab_enabled: Optional[bool] = None,
        slab_mode: str = "Mean",
        sync_axial_slice: bool = False,
        settings: Optional[CursorSettings] = None,
        parameterNode: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """O2, O4, B11, B12, B15: Synchronize Axial lines, Panoramic lines, and 3-slice Cross-Section Lightbox."""
        if settings is not None:
            width_mm = settings.width_mm
            tick_spacing_mm = settings.tick_spacing_mm
            show_all_ticks = settings.show_all_ticks
            slab_enabled = settings.slab_enabled
            slab_mode = settings.slab_mode
            slab_thickness_mm = settings.slab_thickness_mm
            tilt_deg = settings.tilt_deg
            lightbox_enabled = settings.lightbox_enabled
            lightbox_step_mm = settings.lightbox_step_mm
            sync_axial_slice = settings.sync_axial_slice
            if settings.site_label is not None:
                site_label = settings.site_label

        # B12: Explicit slab enable flag (defaults to False unless explicitly enabled)
        effective_slab_enabled = bool(slab_enabled) if slab_enabled is not None else False

        # O2: Evaluate curve frames array ONCE and interpolate central + flanking frames from it
        pts, arc_s, tangents, normals, binormals = self.panoramic.get_curve_points_and_frames(
            curveNode, num_samples=250, occlusal_normal=occlusal_normal
        )
        total_len = float(arc_s[-1])
        s_clamped = float(np.clip(arc_mm, 0.0, total_len))

        def _interp_frame(s_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            p_v = np.array([np.interp(s_val, arc_s, pts[:, d]) for d in range(3)], dtype=np.float64)
            t_v = np.array([np.interp(s_val, arc_s, tangents[:, d]) for d in range(3)], dtype=np.float64)
            n_v = np.array([np.interp(s_val, arc_s, normals[:, d]) for d in range(3)], dtype=np.float64)
            b_v = np.array([np.interp(s_val, arc_s, binormals[:, d]) for d in range(3)], dtype=np.float64)
            t_v /= max(1e-8, float(np.linalg.norm(t_v)))
            n_v /= max(1e-8, float(np.linalg.norm(n_v)))
            b_v /= max(1e-8, float(np.linalg.norm(b_v)))
            return p_v, t_v, n_v, b_v

        P0, T0, N0, B0 = _interp_frame(s_clamped)
        T_tilt, N_tilt, B_tilt = self.apply_cross_section_tilt(T0, N0, B0, tilt_deg=tilt_deg)

        if slicer is None or self._is_updating:
            return P0, T_tilt, N_tilt, B_tilt, s_clamped, total_len

        self._is_updating = True
        try:
            vol_name = volumeNode.GetName() if volumeNode else "CBCT"
            red_id = self._get_view_node_id("Red")
            green_id = self._get_view_node_id("Green")
            threed_id = self._get_view_node_id("3D")

            # 1. Optional faint tick fan in Red (Axial) view only (cached in bounded LRU cache by MTime)
            ticksNode = self._resolve_or_create_node_by_role(
                role="CrossSectionTicks",
                class_name="vtkMRMLModelNode",
                default_name=f"{vol_name}_CrossSectionTicks",
                parameterNode=parameterNode,
            )
            if ticksNode is not None:
                if vtk is not None and show_all_ticks:
                    c_id = curveNode.GetID() if hasattr(curveNode, "GetID") else str(id(curveNode))
                    c_mtime = int(curveNode.GetMTime()) if hasattr(curveNode, "GetMTime") else 0
                    occ_tuple = (
                        round(float(occlusal_normal[0]), 6),
                        round(float(occlusal_normal[1]), 6),
                        round(float(occlusal_normal[2]), 6),
                    )
                    poly_key = (c_id, c_mtime, float(tick_spacing_mm), float(width_mm), occ_tuple)
                    if poly_key not in self._poly_cache:
                        n_ticks = max(3, int(round(total_len / max(0.5, tick_spacing_mm))) + 1)
                        t_pts, _, _, t_normals, _ = self.panoramic.get_curve_points_and_frames(
                            curveNode, num_samples=n_ticks, occlusal_normal=occlusal_normal
                        )
                        self._poly_cache[poly_key] = self._build_tick_lines_polydata(
                            t_pts, t_normals, width_mm=width_mm * 0.75
                        )
                    ticksNode.SetAndObservePolyData(self._poly_cache[poly_key])

                ticksDisp = ticksNode.GetDisplayNode()
                if ticksDisp:
                    t_mod = ticksDisp.StartModify() if hasattr(ticksDisp, "StartModify") else None
                    try:
                        ticksDisp.SetColor(0.95, 0.75, 0.15)
                        ticksDisp.SetOpacity(0.45)
                        ticksDisp.SetLineWidth(1.2)
                        ticksDisp.SetVisibility(bool(show_all_ticks))
                        ticksDisp.SetVisibility2D(bool(show_all_ticks))
                        ticksDisp.SetVisibility3D(False)
                        if hasattr(ticksDisp, "SetSliceDisplayModeToProjection"):
                            ticksDisp.SetSliceDisplayModeToProjection()
                        if red_id and hasattr(ticksDisp, "SetViewNodeIDs"):
                            ticksDisp.SetViewNodeIDs([red_id])
                    finally:
                        if t_mod is not None and hasattr(ticksDisp, "EndModify"):
                            ticksDisp.EndModify(t_mod)

            # 2. Interpolate flanking frames (-step_mm, 0, +step_mm) from the already-fetched curve arrays
            s_minus, s_zero, s_plus = self.compute_lightbox_offsets(
                s_clamped, total_len, step_mm=lightbox_step_mm
            )
            P_m, T_m, N_m, B_m = _interp_frame(s_minus)
            T_m_tilt, N_m_tilt, _ = self.apply_cross_section_tilt(T_m, N_m, B_m, tilt_deg=tilt_deg)

            P_p, T_p, N_p, B_p = _interp_frame(s_plus)
            T_p_tilt, N_p_tilt, _ = self.apply_cross_section_tilt(T_p, N_p, B_p, tilt_deg=tilt_deg)

            half_w = width_mm / 2.0
            axial_views = [vid for vid in (red_id, threed_id) if vid]
            pano_views = [green_id] if green_id else []
            show_pano_line = self.is_panoramic_shown_in_green() or (panoNode is not None)

            pano_height_mm = 50.0
            if panoNode and panoNode.GetAttribute("CBCTPlanner.HeightMm"):
                try:
                    pano_height_mm = float(panoNode.GetAttribute("CBCTPlanner.HeightMm"))
                except ValueError:
                    pano_height_mm = 50.0

            tilt_tag = f" | {self.clamp_tilt_deg(tilt_deg):+.0f}°" if abs(tilt_deg) >= 0.5 else ""
            label_text = f"{s_zero:.1f} mm{tilt_tag}"
            if site_label:
                label_text += f" ({site_label})"

            activeNode = self._update_or_create_markups_line(
                role="ActiveCrossSectionLine",
                node_name=f"{vol_name}_ActiveCrossSectionLine",
                pt0=P0 + N_tilt * half_w,
                pt1=P0 - N_tilt * half_w,
                color=COLOR_CS_ZERO,
                label0=f"B | {label_text}",
                label1="L",
                line_diameter=2.0,
                visible_2d=True,
                visible_3d=True,
                allowed_view_ids=axial_views,
                locked=False,
                parameterNode=parameterNode,
            )

            pano_b0, pano_t0 = self.compute_pano_cursor_endpoints(
                s_zero, pano_height_mm=pano_height_mm, tilt_deg=tilt_deg
            )
            panoCursorNode = self._update_or_create_markups_line(
                role="PanoCursorLine",
                node_name=f"{vol_name}_PanoCursorLine",
                pt0=pano_b0,
                pt1=pano_t0,
                color=COLOR_CS_ZERO,
                label0="",
                label1=f"CS0 ({s_zero:.1f} mm{tilt_tag})",
                line_diameter=1.6,
                visible_2d=show_pano_line,
                visible_3d=False,
                allowed_view_ids=pano_views,
                locked=False,
                parameterNode=parameterNode,
            )

            step_fmt = f"{lightbox_step_mm:.1f}".rstrip("0").rstrip(".")
            self._update_or_create_markups_line(
                role="MinusCrossSectionLine",
                node_name=f"{vol_name}_MinusCrossSectionLine",
                pt0=P_m + N_m_tilt * (half_w * 0.85),
                pt1=P_m - N_m_tilt * (half_w * 0.85),
                color=COLOR_CS_MINUS,
                label0=f"CS- (-{step_fmt}mm)",
                label1="",
                line_diameter=1.3,
                visible_2d=bool(lightbox_enabled),
                visible_3d=bool(lightbox_enabled),
                allowed_view_ids=axial_views,
                locked=True,
                parameterNode=parameterNode,
            )
            self._update_or_create_markups_line(
                role="PlusCrossSectionLine",
                node_name=f"{vol_name}_PlusCrossSectionLine",
                pt0=P_p + N_p_tilt * (half_w * 0.85),
                pt1=P_p - N_p_tilt * (half_w * 0.85),
                color=COLOR_CS_PLUS,
                label0=f"CS+ (+{step_fmt}mm)",
                label1="",
                line_diameter=1.3,
                visible_2d=bool(lightbox_enabled),
                visible_3d=bool(lightbox_enabled),
                allowed_view_ids=axial_views,
                locked=True,
                parameterNode=parameterNode,
            )

            pano_bm, pano_tm = self.compute_pano_cursor_endpoints(
                s_minus, pano_height_mm=pano_height_mm, tilt_deg=tilt_deg
            )
            self._update_or_create_markups_line(
                role="PanoMinusLine",
                node_name=f"{vol_name}_PanoMinusLine",
                pt0=pano_bm,
                pt1=pano_tm,
                color=COLOR_CS_MINUS,
                label0="",
                label1=f"-{step_fmt}mm",
                line_diameter=1.1,
                visible_2d=bool(show_pano_line and lightbox_enabled),
                visible_3d=False,
                allowed_view_ids=pano_views,
                locked=True,
                parameterNode=parameterNode,
            )

            pano_bp, pano_tp = self.compute_pano_cursor_endpoints(
                s_plus, pano_height_mm=pano_height_mm, tilt_deg=tilt_deg
            )
            self._update_or_create_markups_line(
                role="PanoPlusLine",
                node_name=f"{vol_name}_PanoPlusLine",
                pt0=pano_bp,
                pt1=pano_tp,
                color=COLOR_CS_PLUS,
                label0="",
                label1=f"+{step_fmt}mm",
                line_diameter=1.1,
                visible_2d=bool(show_pano_line and lightbox_enabled),
                visible_3d=False,
                allowed_view_ids=pano_views,
                locked=True,
                parameterNode=parameterNode,
            )

            # 3. Update Yellow (CS0), YellowMinus (CS-), YellowPlus (CS+), and optionally Red (Axial)
            self.jump_to_frame(
                P0,
                T_tilt,
                N_tilt,
                B_tilt,
                slab_thickness_mm=slab_thickness_mm,
                slab_enabled=effective_slab_enabled,
                slab_mode=slab_mode,
                update_green_tangential=not self.is_panoramic_shown_in_green(),
                sync_axial_slice=sync_axial_slice,
                volumeNode=volumeNode,
            )
            if lightbox_enabled:
                self._update_single_cross_section_slice_widget(
                    "YellowMinus",
                    P_m,
                    T_m_tilt,
                    N_m_tilt,
                    slab_thickness_mm=slab_thickness_mm,
                    slab_enabled=effective_slab_enabled,
                    slab_mode=slab_mode,
                    volumeNode=volumeNode,
                )
                self._update_single_cross_section_slice_widget(
                    "YellowPlus",
                    P_p,
                    T_p_tilt,
                    N_p_tilt,
                    slab_thickness_mm=slab_thickness_mm,
                    slab_enabled=effective_slab_enabled,
                    slab_mode=slab_mode,
                    volumeNode=volumeNode,
                )

            self._ensure_interactive_observers(
                activeAxialNode=activeNode,
                panoCursorNode=panoCursorNode,
                curveNode=curveNode,
                occlusal_normal=occlusal_normal,
            )
        finally:
            self._is_updating = False

        return P0, T_tilt, N_tilt, B_tilt, s_clamped, total_len

    @staticmethod
    def resolve_vtk_slab_type(slab_mode: str) -> int:
        """B12: Map slab mode string ('Mean', 'MIP', 'MinIP') to VTK_IMAGE_SLAB_* constant."""
        mode_up = (slab_mode or "Mean").strip().upper()
        if vtk is not None:
            if mode_up == "MIP" and hasattr(vtk, "VTK_IMAGE_SLAB_MAX"):
                return int(vtk.VTK_IMAGE_SLAB_MAX)
            if mode_up == "MINIP" and hasattr(vtk, "VTK_IMAGE_SLAB_MIN"):
                return int(vtk.VTK_IMAGE_SLAB_MIN)
            if hasattr(vtk, "VTK_IMAGE_SLAB_MEAN"):
                return int(vtk.VTK_IMAGE_SLAB_MEAN)
        # Standard VTK enum values: MIN=0, MAX=1, MEAN=2, SUM=3
        if mode_up == "MIP":
            return 1
        if mode_up == "MINIP":
            return 0
        return 2

    @classmethod
    def configure_slice_node_slab(
        cls,
        sliceNode: Any,
        slab_enabled: bool = False,
        slab_thickness_mm: float = 0.5,
        slab_mode: str = "Mean",
    ) -> None:
        """B12: Explicitly set slab reconstruction enable flag, type (Mean default, MIP/MinIP), and thickness."""
        if sliceNode is None:
            return
        if hasattr(sliceNode, "SetSlabReconstructionEnabled"):
            sliceNode.SetSlabReconstructionEnabled(bool(slab_enabled))
        if hasattr(sliceNode, "SetSlabReconstructionType"):
            sliceNode.SetSlabReconstructionType(cls.resolve_vtk_slab_type(slab_mode))
        if bool(slab_enabled) and hasattr(sliceNode, "SetSlabReconstructionThickness"):
            sliceNode.SetSlabReconstructionThickness(max(0.1, float(slab_thickness_mm)))

    @classmethod
    def _update_single_cross_section_slice_widget(
        cls,
        slice_tag: str,
        P: np.ndarray,
        T_tilted: np.ndarray,
        N: np.ndarray,
        slab_thickness_mm: float = 0.5,
        slab_enabled: bool = False,
        slab_mode: str = "Mean",
        volumeNode: Any = None,
    ) -> None:
        if slicer is None or not getattr(slicer, "app", None):
            return
        lm = slicer.app.layoutManager()
        if not lm:
            return
        sw = lm.sliceWidget(slice_tag)
        if not sw:
            return
        if volumeNode is not None and sw.mrmlSliceCompositeNode():
            if not sw.mrmlSliceCompositeNode().GetBackgroundVolumeID():
                sw.mrmlSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())
        sNode = sw.sliceLogic().GetSliceNode()
        was_mod = sNode.StartModify() if hasattr(sNode, "StartModify") else None
        try:
            sNode.SetSliceToRASByNTP(
                float(T_tilted[0]), float(T_tilted[1]), float(T_tilted[2]),
                float(-N[0]), float(-N[1]), float(-N[2]),
                float(P[0]), float(P[1]), float(P[2]),
                0,
            )
            cls.configure_slice_node_slab(
                sNode,
                slab_enabled=slab_enabled,
                slab_thickness_mm=slab_thickness_mm,
                slab_mode=slab_mode,
            )
        finally:
            if was_mod is not None and hasattr(sNode, "EndModify"):
                sNode.EndModify(was_mod)

    @staticmethod
    def compute_yellow_slice_axes(
        T: np.ndarray, N: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (screen_right, screen_up, slice_normal) for Yellow Cross-Section view."""
        screen_right = -np.asarray(N, dtype=np.float64)
        slice_normal = np.asarray(T, dtype=np.float64)
        screen_up = np.cross(slice_normal, screen_right)
        if np.dot(screen_up, B) < 0:
            screen_up = -screen_up
        return screen_right, screen_up, slice_normal

    def jump_to_frame(
        self,
        P: np.ndarray,
        T: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        slab_thickness_mm: float = 0.5,
        slab_enabled: bool = False,
        slab_mode: str = "Mean",
        update_green_tangential: bool = False,
        sync_axial_slice: bool = False,
        volumeNode: Any = None,
    ) -> None:
        """B11 & B12: Align Yellow Cross-Section view; only reset Red Axial offset when `sync_axial_slice=True`."""
        if slicer is None or not getattr(slicer, "app", None):
            return
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return

        self._update_single_cross_section_slice_widget(
            "Yellow",
            P,
            T,
            N,
            slab_thickness_mm=slab_thickness_mm,
            slab_enabled=slab_enabled,
            slab_mode=slab_mode,
            volumeNode=volumeNode,
        )

        if update_green_tangential and not self.is_panoramic_shown_in_green():
            green = layoutManager.sliceWidget("Green")
            if green:
                sNode = green.sliceLogic().GetSliceNode()
                was_mod = sNode.StartModify() if hasattr(sNode, "StartModify") else None
                try:
                    sNode.SetSliceToRASByNTP(
                        float(N[0]), float(N[1]), float(N[2]),
                        float(T[0]), float(T[1]), float(T[2]),
                        float(P[0]), float(P[1]), float(P[2]),
                        0,
                    )
                finally:
                    if was_mod is not None and hasattr(sNode, "EndModify"):
                        sNode.EndModify(was_mod)

        # B11: Only force Red to Axial & P[2] on an explicit tooth jump (`sync_axial_slice=True`),
        # so the user can freely scroll the Axial slice while moving the cross-section cursor!
        if sync_axial_slice:
            red = layoutManager.sliceWidget("Red")
            if red:
                sNode = red.sliceLogic().GetSliceNode()
                was_mod = sNode.StartModify() if hasattr(sNode, "StartModify") else None
                try:
                    sNode.SetOrientationToAxial()
                    sNode.SetSliceOffset(float(P[2]))
                finally:
                    if was_mod is not None and hasattr(sNode, "EndModify"):
                        sNode.EndModify(was_mod)

"""Cross-Sectional Slice Navigator, Multi-Slice Lightbox (-2, 0, +2 mm), & +-30° Tilt Control.

Synchronizes via a re-entrancy-guarded cursor in arc-length millimeters (`arc_mm`):
  - Multi-Slice Cross-Sectional Lightbox:
      * YellowMinus (`CS-`, amber `#F59E0B`): Cross-Section at `arc_mm - step_mm` (default `-2.0 mm`)
      * Yellow (`CS0`, cyan `#06B6D4`): Active Cross-Section at `arc_mm` (`0.0 mm`)
      * YellowPlus (`CS+`, emerald `#10B981`): Cross-Section at `arc_mm + step_mm` (default `+2.0 mm`)
    Each cross-section shows Buccal on Screen Left, Lingual on Screen Right, and tilted Superior Up.
  - +-30° Cross-Section Tilt Control (`tilt_deg` clamped to `[-30.0, +30.0]`):
      * Rotates the slice vertical axis `B_tilted = cos(theta)*B + sin(theta)*T` and slice normal
        `T_tilted = cos(theta)*T - sin(theta)*B` around the Buccal-Lingual axis `N` (`det = +1`),
        and tilts the Panoramic cursor line(s) on the Green view accordingly.
  - Red View (Axial): Draggable active cross-section line (`<Vol>_ActiveCrossSectionLine`),
    optional `CS-`/`CS+` flanking lines when lightbox is enabled, and optional Red-only tick fan.
  - Green View (Panoramic): Draggable vertical/tilted cursor line (`<Vol>_PanoCursorLine`) and
    optional `CS-`/`CS+` flanking lines restricted to the Green view via `SetViewNodeIDs`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

try:
    import vtk  # type: ignore
except ImportError:
    vtk = None

from .PanoramicCurveLogic import PanoramicCurveLogic

MAX_TILT_DEG = 30.0

# Colors matching the custom Lightbox slice view headers:
COLOR_CS_MINUS = (0.96, 0.62, 0.04)  # Amber (#F59E0B) for -step_mm
COLOR_CS_ZERO = (0.00, 0.90, 1.00)   # Cyan (#06B6D4) for 0.0 mm
COLOR_CS_PLUS = (0.06, 0.73, 0.51)   # Emerald (#10B981) for +step_mm


class CrossSectionNavigator:
    """Manages interactive cross-sectional reslicing, 3-slice lightbox, and +-30° tilt."""

    def __init__(self, panoramic_logic: Optional[PanoramicCurveLogic] = None) -> None:
        self.panoramic = panoramic_logic or PanoramicCurveLogic()
        self._is_updating: bool = False
        self._poly_cache: Dict[
            Tuple[str, int, float, float, Tuple[float, float, float]], Any
        ] = {}
        self._line_observers: List[Tuple[Any, int]] = []
        self._cursor_changed_callback: Optional[Callable[[float], None]] = None

    def set_cursor_changed_callback(self, callback: Optional[Callable[[float], None]]) -> None:
        """Register callback invoked when the user drags the Axial or Panoramic cursor line."""
        self._cursor_changed_callback = callback

    def remove_line_observers(self) -> None:
        """Detach VTK observers from interactive cursor lines."""
        for node, tag in self._line_observers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._line_observers.clear()

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
        """Rotate (T, B) by `tilt_deg` (clamped to [-30°, +30°]) around Buccal-Lingual normal `N`.

        At `tilt_deg == 0`: returns `(T, N, B)` unchanged.
        At `tilt_deg != 0`:
          - `N_tilted = N` (horizontal Buccal-Lingual axis is invariant)
          - `B_tilted = cos(theta) * B + sin(theta) * T` (in-plane vertical leans toward T)
          - `T_tilted = cos(theta) * T - sin(theta) * B` (slice plane normal)
        Preserves orthonormal right-handedness (`det([-N, B_tilted, T_tilted]) = +1.0`).
        """
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
        """Compute bottom and top 3D RAS endpoints for a panoramic cursor line at `arc_mm` tilted by `tilt_deg`.

        Pivots around the vertical midpoint `(pano_x(arc_mm), pano_height_mm / 2, 0)`.
        Because Panoramic RAS X is `-arc_mm` (Patient Right at X=0 on Screen Left, Patient Left at X=-L on Screen Right),
        leaning the top of the slice toward `+T` (increasing `arc_mm`) shifts the top endpoint by `-0.5 * H * tan(theta)`
        along RAS X and the bottom endpoint by `+0.5 * H * tan(theta)`.
        """
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

    def _ensure_interactive_observers(
        self,
        activeAxialNode: Any,
        panoCursorNode: Optional[Any],
        curveNode: Any,
        occlusal_normal: Sequence[float],
    ) -> None:
        """Attach interaction observers to Axial and Panoramic cursor lines if not already attached."""
        if vtk is None:
            return
        self.remove_line_observers()

        def _on_axial_line_dragged(caller: Any, _event: Any) -> None:
            if self._is_updating or not curveNode:
                return
            if caller.GetNumberOfControlPoints() < 2:
                return
            p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(0, p0)
            caller.GetNthControlPointPositionWorld(1, p1)
            mid = 0.5 * (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64))
            new_arc_mm = self.panoramic.project_point_to_curve_arc_mm(
                curveNode, mid, occlusal_normal=occlusal_normal
            )
            if self._cursor_changed_callback is not None:
                self._cursor_changed_callback(new_arc_mm)

        def _on_pano_line_dragged(caller: Any, _event: Any) -> None:
            if self._is_updating or not curveNode:
                return
            if caller.GetNumberOfControlPoints() < 2:
                return
            p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            caller.GetNthControlPointPositionWorld(0, p0)
            caller.GetNthControlPointPositionWorld(1, p1)
            mean_x = 0.5 * (float(p0[0]) + float(p1[0]))
            _, arc_s, _, _, _ = self.panoramic.get_curve_points_and_frames(
                curveNode, num_samples=50, occlusal_normal=occlusal_normal
            )
            total_len = float(arc_s[-1])
            new_arc_mm = PanoramicCurveLogic.arc_mm(mean_x, max_arc_mm=total_len)
            if self._cursor_changed_callback is not None:
                self._cursor_changed_callback(new_arc_mm)

        for evt in (vtk.vtkCommand.PointModifiedEvent, vtk.vtkCommand.EndInteractionEvent):
            if activeAxialNode is not None:
                tag = activeAxialNode.AddObserver(evt, _on_axial_line_dragged)
                self._line_observers.append((activeAxialNode, tag))
            if panoCursorNode is not None:
                tag = panoCursorNode.AddObserver(evt, _on_pano_line_dragged)
                self._line_observers.append((panoCursorNode, tag))

    @staticmethod
    def _update_or_create_markups_line(
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
    ) -> Any:
        if slicer is None:
            return None
        lineNode = slicer.mrmlScene.GetFirstNodeByName(node_name)
        if not lineNode:
            lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", node_name)
            lineNode.CreateDefaultDisplayNodes()

        if hasattr(lineNode, "SetLocked"):
            lineNode.SetLocked(bool(locked))

        if lineNode.GetNumberOfControlPoints() != 2:
            lineNode.RemoveAllControlPoints()
            lineNode.AddControlPoint(float(pt0[0]), float(pt0[1]), float(pt0[2]))
            lineNode.AddControlPoint(float(pt1[0]), float(pt1[1]), float(pt1[2]))
        else:
            lineNode.SetNthControlPointPositionWorld(0, float(pt0[0]), float(pt0[1]), float(pt0[2]))
            lineNode.SetNthControlPointPositionWorld(1, float(pt1[0]), float(pt1[1]), float(pt1[2]))

        if hasattr(lineNode, "SetNthControlPointLabel"):
            lineNode.SetNthControlPointLabel(0, label0)
            lineNode.SetNthControlPointLabel(1, label1)

        disp = lineNode.GetDisplayNode()
        if disp:
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Synchronize Axial lines, Panoramic lines, and 3-slice Cross-Section Lightbox at `arc_mm` & `tilt_deg`.

        Returns tilted `(P, T_tilted, N, B_tilted, clamped_arc_mm, total_len_mm)` for the central slice.
        """
        P0, T0, N0, B0, s_clamped, total_len = self.panoramic.get_frame_at_arc_mm(
            curveNode, arc_mm=arc_mm, occlusal_normal=occlusal_normal
        )
        T_tilt, N_tilt, B_tilt = self.apply_cross_section_tilt(T0, N0, B0, tilt_deg=tilt_deg)

        if slicer is None:
            return P0, T_tilt, N_tilt, B_tilt, s_clamped, total_len

        if self._is_updating:
            return P0, T_tilt, N_tilt, B_tilt, s_clamped, total_len

        self._is_updating = True
        try:
            vol_name = volumeNode.GetName() if volumeNode else "CBCT"
            red_id = self._get_view_node_id("Red")
            green_id = self._get_view_node_id("Green")
            threed_id = self._get_view_node_id("3D")

            # 1. Optional faint tick fan in Red (Axial) view only (cached by curve MTime)
            ticks_name = f"{vol_name}_CrossSectionTicks"
            ticksNode = slicer.mrmlScene.GetFirstNodeByName(ticks_name)
            if not ticksNode:
                ticksNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", ticks_name)
                ticksNode.CreateDefaultDisplayNodes()

            if vtk is not None and show_all_ticks:
                c_id = curveNode.GetID() if hasattr(curveNode, "GetID") else str(id(curveNode))
                c_mtime = int(curveNode.GetMTime()) if hasattr(curveNode, "GetMTime") else 0
                occ_tuple = (float(occlusal_normal[0]), float(occlusal_normal[1]), float(occlusal_normal[2]))
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

            # 2. Compute frames for (-step_mm, 0 mm, +step_mm)
            s_minus, s_zero, s_plus = self.compute_lightbox_offsets(
                s_clamped, total_len, step_mm=lightbox_step_mm
            )
            P_m, T_m, N_m, B_m, _, _ = self.panoramic.get_frame_at_arc_mm(
                curveNode, arc_mm=s_minus, occlusal_normal=occlusal_normal
            )
            T_m_tilt, N_m_tilt, B_m_tilt = self.apply_cross_section_tilt(T_m, N_m, B_m, tilt_deg=tilt_deg)

            P_p, T_p, N_p, B_p, _, _ = self.panoramic.get_frame_at_arc_mm(
                curveNode, arc_mm=s_plus, occlusal_normal=occlusal_normal
            )
            T_p_tilt, N_p_tilt, B_p_tilt = self.apply_cross_section_tilt(T_p, N_p, B_p, tilt_deg=tilt_deg)

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

            # 2a. Central Active Cross-Section Line (draggable in Red & Green)
            tilt_tag = f" | {self.clamp_tilt_deg(tilt_deg):+.0f}°" if abs(tilt_deg) >= 0.5 else ""
            label_text = f"{s_zero:.1f} mm{tilt_tag}"
            if site_label:
                label_text += f" ({site_label})"

            activeNode = self._update_or_create_markups_line(
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
            )

            pano_b0, pano_t0 = self.compute_pano_cursor_endpoints(
                s_zero, pano_height_mm=pano_height_mm, tilt_deg=tilt_deg
            )
            panoCursorNode = self._update_or_create_markups_line(
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
            )

            # 2b. Flanking Lightbox Lines (`CS-` at -step_mm in Amber, `CS+` at +step_mm in Emerald)
            self._update_or_create_markups_line(
                node_name=f"{vol_name}_MinusCrossSectionLine",
                pt0=P_m + N_m_tilt * (half_w * 0.85),
                pt1=P_m - N_m_tilt * (half_w * 0.85),
                color=COLOR_CS_MINUS,
                label0=f"CS- (-{lightbox_step_mm:.0f}mm)",
                label1="",
                line_diameter=1.3,
                visible_2d=bool(lightbox_enabled),
                visible_3d=bool(lightbox_enabled),
                allowed_view_ids=axial_views,
                locked=True,
            )
            self._update_or_create_markups_line(
                node_name=f"{vol_name}_PlusCrossSectionLine",
                pt0=P_p + N_p_tilt * (half_w * 0.85),
                pt1=P_p - N_p_tilt * (half_w * 0.85),
                color=COLOR_CS_PLUS,
                label0=f"CS+ (+{lightbox_step_mm:.0f}mm)",
                label1="",
                line_diameter=1.3,
                visible_2d=bool(lightbox_enabled),
                visible_3d=bool(lightbox_enabled),
                allowed_view_ids=axial_views,
                locked=True,
            )

            pano_bm, pano_tm = self.compute_pano_cursor_endpoints(
                s_minus, pano_height_mm=pano_height_mm, tilt_deg=tilt_deg
            )
            self._update_or_create_markups_line(
                node_name=f"{vol_name}_PanoMinusLine",
                pt0=pano_bm,
                pt1=pano_tm,
                color=COLOR_CS_MINUS,
                label0="",
                label1=f"-{lightbox_step_mm:.0f}mm",
                line_diameter=1.1,
                visible_2d=bool(show_pano_line and lightbox_enabled),
                visible_3d=False,
                allowed_view_ids=pano_views,
                locked=True,
            )

            pano_bp, pano_tp = self.compute_pano_cursor_endpoints(
                s_plus, pano_height_mm=pano_height_mm, tilt_deg=tilt_deg
            )
            self._update_or_create_markups_line(
                node_name=f"{vol_name}_PanoPlusLine",
                pt0=pano_bp,
                pt1=pano_tp,
                color=COLOR_CS_PLUS,
                label0="",
                label1=f"+{lightbox_step_mm:.0f}mm",
                line_diameter=1.1,
                visible_2d=bool(show_pano_line and lightbox_enabled),
                visible_3d=False,
                allowed_view_ids=pano_views,
                locked=True,
            )

            # 3. Update Yellow (CS0), YellowMinus (CS-), YellowPlus (CS+), and Red (Axial)
            self.jump_to_frame(
                P0,
                T_tilt,
                N_tilt,
                B_tilt,
                slab_thickness_mm=slab_thickness_mm,
                update_green_tangential=not self.is_panoramic_shown_in_green(),
            )
            if lightbox_enabled:
                self._update_single_cross_section_slice_widget(
                    "YellowMinus", P_m, T_m_tilt, N_m_tilt, slab_thickness_mm, volumeNode=volumeNode
                )
                self._update_single_cross_section_slice_widget(
                    "YellowPlus", P_p, T_p_tilt, N_p_tilt, slab_thickness_mm, volumeNode=volumeNode
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
    def _update_single_cross_section_slice_widget(
        slice_tag: str,
        P: np.ndarray,
        T_tilted: np.ndarray,
        N: np.ndarray,
        slab_thickness_mm: float = 0.5,
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
        sNode.SetSliceToRASByNTP(
            float(T_tilted[0]), float(T_tilted[1]), float(T_tilted[2]),
            float(-N[0]), float(-N[1]), float(-N[2]),
            float(P[0]), float(P[1]), float(P[2]),
            0,
        )
        if hasattr(sNode, "SetSlabReconstructionEnabled"):
            enable_slab = float(slab_thickness_mm) > 0.6
            sNode.SetSlabReconstructionEnabled(enable_slab)
            if enable_slab and hasattr(sNode, "SetSlabReconstructionThickness"):
                sNode.SetSlabReconstructionThickness(float(slab_thickness_mm))
        sNode.Modified()

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
        update_green_tangential: bool = False,
    ) -> None:
        """Align Yellow (Cross-Section) and Red (Axial) views to frame without clobbering Panoramic Green view."""
        if slicer is None or not getattr(slicer, "app", None):
            return
        layoutManager = slicer.app.layoutManager()
        if not layoutManager:
            return

        self._update_single_cross_section_slice_widget("Yellow", P, T, N, slab_thickness_mm)

        if update_green_tangential and not self.is_panoramic_shown_in_green():
            green = layoutManager.sliceWidget("Green")
            if green:
                sNode = green.sliceLogic().GetSliceNode()
                sNode.SetSliceToRASByNTP(
                    float(N[0]), float(N[1]), float(N[2]),
                    float(T[0]), float(T[1]), float(T[2]),
                    float(P[0]), float(P[1]), float(P[2]),
                    0,
                )
                sNode.Modified()

        red = layoutManager.sliceWidget("Red")
        if red:
            sNode = red.sliceLogic().GetSliceNode()
            sNode.SetOrientationToAxial()
            sNode.SetSliceOffset(float(P[2]))
            sNode.Modified()

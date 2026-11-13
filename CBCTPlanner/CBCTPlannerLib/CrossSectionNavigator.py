"""Cross-Sectional Slice Navigator & Interactive Axial / Panoramic Cursor Lines.

Synchronizes via a re-entrancy-guarded cursor in arc-length millimeters (`arc_mm`):
  - Yellow View: Upright Cross-Section (Normal = T, Transverse = -N -> Buccal Left, Lingual Right, Up = +B)
    with optional slab thickness reconstruction.
  - Red View (Axial): Draggable active cross-section line (`<Vol>_ActiveCrossSectionLine`) centered
    on the arch curve and snapped perpendicular to the curve, plus an optional faint tick fan
    (`<Vol>_CrossSectionTicks`) restricted to the Red view via `SetViewNodeIDs`.
  - Green View (Panoramic): Draggable vertical cursor line (`<Vol>_PanoCursorLine`) at
    `pano_x(arc_mm)` spanning the panoramic height, restricted to the Green view via `SetViewNodeIDs`.
"""

from __future__ import annotations

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


class CrossSectionNavigator:
    """Manages interactive cross-sectional reslicing and synchronized Axial/Panoramic lines."""

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
    def _get_view_node_id(slice_or_view_name: str) -> Optional[str]:
        if slicer is None or not getattr(slicer, "app", None):
            return None
        lm = slicer.app.layoutManager()
        if not lm:
            return None
        if slice_or_view_name in ("Red", "Yellow", "Green"):
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

    def update_cursor_at_arc_mm(
        self,
        volumeNode: Any,
        curveNode: Any,
        arc_mm: float,
        width_mm: float = 24.0,
        tick_spacing_mm: float = 2.0,
        show_all_ticks: bool = True,
        slab_thickness_mm: float = 0.5,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
        site_label: Optional[str] = None,
        panoNode: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Synchronize Axial line, Panoramic vertical line, and Yellow Cross-Section view at `arc_mm`.

        Returns (P, T, N, B, clamped_arc_mm, total_len_mm).
        """
        P, T, N, B, s_clamped, total_len = self.panoramic.get_frame_at_arc_mm(
            curveNode, arc_mm=arc_mm, occlusal_normal=occlusal_normal
        )
        if slicer is None:
            return P, T, N, B, s_clamped, total_len

        if self._is_updating:
            return P, T, N, B, s_clamped, total_len

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

            # 2. Active draggable Cross-Section Line on Axial (Red) & 3D views
            active_name = f"{vol_name}_ActiveCrossSectionLine"
            activeNode = slicer.mrmlScene.GetFirstNodeByName(active_name)
            if not activeNode:
                activeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", active_name)
                activeNode.CreateDefaultDisplayNodes()

            half_w = width_mm / 2.0
            p_buccal = P + N * half_w
            p_lingual = P - N * half_w
            if activeNode.GetNumberOfControlPoints() != 2:
                activeNode.RemoveAllControlPoints()
                activeNode.AddControlPoint(float(p_buccal[0]), float(p_buccal[1]), float(p_buccal[2]))
                activeNode.AddControlPoint(float(p_lingual[0]), float(p_lingual[1]), float(p_lingual[2]))
            else:
                activeNode.SetNthControlPointPositionWorld(0, float(p_buccal[0]), float(p_buccal[1]), float(p_buccal[2]))
                activeNode.SetNthControlPointPositionWorld(1, float(p_lingual[0]), float(p_lingual[1]), float(p_lingual[2]))

            label_text = f"{s_clamped:.1f} mm"
            if site_label:
                label_text += f" ({site_label})"
            if hasattr(activeNode, "SetNthControlPointLabel"):
                activeNode.SetNthControlPointLabel(0, f"B | {label_text}")
                activeNode.SetNthControlPointLabel(1, "L")

            activeDisp = activeNode.GetDisplayNode()
            if activeDisp:
                activeDisp.SetColor(0.0, 0.90, 1.0)
                activeDisp.SetSelectedColor(0.0, 0.90, 1.0)
                if hasattr(slicer.vtkMRMLMarkupsDisplayNode, "UseLineDiameter"):
                    activeDisp.SetCurveLineSizeMode(slicer.vtkMRMLMarkupsDisplayNode.UseLineDiameter)
                activeDisp.SetLineDiameter(2.0)
                activeDisp.SetVisibility(True)
                activeDisp.SetVisibility2D(True)
                activeDisp.SetVisibility3D(True)
                if hasattr(activeDisp, "SetPropertiesLabelVisibility"):
                    activeDisp.SetPropertiesLabelVisibility(False)
                if hasattr(activeDisp, "SetPointLabelsVisibility"):
                    activeDisp.SetPointLabelsVisibility(True)
                allowed_views = [vid for vid in (red_id, threed_id) if vid]
                if allowed_views and hasattr(activeDisp, "SetViewNodeIDs"):
                    activeDisp.SetViewNodeIDs(allowed_views)

            # 3. Vertical Draggable Cursor Line in Panoramic (Green) view only
            pano_cursor_name = f"{vol_name}_PanoCursorLine"
            panoCursorNode = slicer.mrmlScene.GetFirstNodeByName(pano_cursor_name)
            if not panoCursorNode:
                panoCursorNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsLineNode", pano_cursor_name
                )
                panoCursorNode.CreateDefaultDisplayNodes()

            pano_height_mm = 50.0
            if panoNode and panoNode.GetAttribute("CBCTPlanner.HeightMm"):
                try:
                    pano_height_mm = float(panoNode.GetAttribute("CBCTPlanner.HeightMm"))
                except ValueError:
                    pano_height_mm = 50.0

            x_pano = PanoramicCurveLogic.pano_x(s_clamped)
            if panoCursorNode.GetNumberOfControlPoints() != 2:
                panoCursorNode.RemoveAllControlPoints()
                panoCursorNode.AddControlPoint(x_pano, 0.0, 0.0)
                panoCursorNode.AddControlPoint(x_pano, pano_height_mm, 0.0)
            else:
                panoCursorNode.SetNthControlPointPositionWorld(0, x_pano, 0.0, 0.0)
                panoCursorNode.SetNthControlPointPositionWorld(1, x_pano, pano_height_mm, 0.0)

            if hasattr(panoCursorNode, "SetNthControlPointLabel"):
                panoCursorNode.SetNthControlPointLabel(0, "")
                panoCursorNode.SetNthControlPointLabel(1, f"{s_clamped:.1f} mm")

            panoDisp = panoCursorNode.GetDisplayNode()
            if panoDisp:
                panoDisp.SetColor(0.0, 0.90, 1.0)
                panoDisp.SetSelectedColor(0.0, 0.90, 1.0)
                if hasattr(slicer.vtkMRMLMarkupsDisplayNode, "UseLineDiameter"):
                    panoDisp.SetCurveLineSizeMode(slicer.vtkMRMLMarkupsDisplayNode.UseLineDiameter)
                panoDisp.SetLineDiameter(1.5)
                show_pano_line = self.is_panoramic_shown_in_green() or (panoNode is not None)
                panoDisp.SetVisibility(bool(show_pano_line))
                panoDisp.SetVisibility2D(bool(show_pano_line))
                panoDisp.SetVisibility3D(False)
                if hasattr(panoDisp, "SetPropertiesLabelVisibility"):
                    panoDisp.SetPropertiesLabelVisibility(False)
                if hasattr(panoDisp, "SetPointLabelsVisibility"):
                    panoDisp.SetPointLabelsVisibility(True)
                if green_id and hasattr(panoDisp, "SetViewNodeIDs"):
                    panoDisp.SetViewNodeIDs([green_id])

            # 4. Update Yellow (Cross-Section) and Red (Axial) views; leave Green untouched if Pano is shown
            self.jump_to_frame(
                P,
                T,
                N,
                B,
                slab_thickness_mm=slab_thickness_mm,
                update_green_tangential=not self.is_panoramic_shown_in_green(),
            )

            self._ensure_interactive_observers(
                activeAxialNode=activeNode,
                panoCursorNode=panoCursorNode,
                curveNode=curveNode,
                occlusal_normal=occlusal_normal,
            )
        finally:
            self._is_updating = False

        return P, T, N, B, s_clamped, total_len

    @staticmethod
    def compute_yellow_slice_axes(
        T: np.ndarray, N: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (screen_right, screen_up, slice_normal) for Yellow Cross-Section view.

        In Slicer's SetSliceToRASByNTP(Nx, Ny, Nz, Tx, Ty, Tz, ...):
          - Slice Normal = T (tangent along arch)
          - Transverse (Screen Right) = -N (Lingual on Right -> Buccal on Left)
          - Screen Up = SliceNormal x Transverse = T x (-N) = +B (Superior Up)
        """
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

        # Yellow -> Upright Cross-Section (Buccal Left, Lingual Right, Superior Up)
        yellow = layoutManager.sliceWidget("Yellow")
        if yellow:
            sNode = yellow.sliceLogic().GetSliceNode()
            sNode.SetSliceToRASByNTP(
                float(T[0]), float(T[1]), float(T[2]),
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

        # Green -> Only set tangential if Panoramic Radiograph is NOT currently shown in Green
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

        # Red -> Axial centered at P[2]
        red = layoutManager.sliceWidget("Red")
        if red:
            sNode = red.sliceLogic().GetSliceNode()
            sNode.SetOrientationToAxial()
            sNode.SetSliceOffset(float(P[2]))
            sNode.Modified()

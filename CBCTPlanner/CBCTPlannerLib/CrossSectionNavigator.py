"""Cross-Sectional Slice Navigator & Axial Slice Intersection Overlay.

Synchronizes:
  - Yellow View: Upright Cross-Section (Normal = T, Transverse = -N, Up = +B)
  - Green View: Upright Tangential / Panoramic (Normal = N, Transverse = T, Up = +B)
  - Red View: Axial plane with real-time Cross-Section tick grid & active slice line
    showing exactly which cross-section slice is currently displayed on the axial image.
"""

from typing import Optional, Tuple
import numpy as np
import slicer

from .PanoramicCurveLogic import PanoramicCurveLogic

try:
    import vtk
except ImportError:
    vtk = None


class CrossSectionNavigator:
    """Manages cross-sectional reslicing and axial slice intersection indicators."""

    @staticmethod
    def enable_native_slice_intersections(visible: bool = True) -> None:
        """Toggle 3D Slicer's built-in slice intersection lines across Red/Yellow/Green."""
        for compNode in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            if hasattr(compNode, "SetSliceIntersectionVisibility"):
                compNode.SetSliceIntersectionVisibility(bool(visible))
        for dispNode in slicer.util.getNodesByClass("vtkMRMLSliceDisplayNode"):
            if hasattr(dispNode, "SetIntersectingSlicesVisible"):
                dispNode.SetIntersectingSlicesVisible(bool(visible))

    @staticmethod
    def _build_tick_lines_polydata(
        pts: np.ndarray,
        normals: np.ndarray,
        width_mm: float,
    ):
        """Build vtkPolyData containing perpendicular cross-section tick segments."""
        if vtk is None:
            return None
        vtk_pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        half_w = width_mm / 2.0

        for idx, (p, n) in enumerate(zip(pts, normals)):
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

    def update_axial_cross_section_indicators(
        self,
        volumeNode,
        curveNode,
        active_index: int,
        num_slices: int = 25,
        width_mm: float = 24.0,
        show_all_ticks: bool = True,
        site_label: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Render cross-section slice lines on the Axial (Red) image and return active frame.

        Draws:
          1. `<Vol>_CrossSectionTicks` (vtkMRMLModelNode, 2D projection): all N cross-section
             cut lines along the arch so the clinician sees the entire cross-section fan.
          2. `<Vol>_ActiveCrossSectionLine` (vtkMRMLMarkupsLineNode): bright highlighted
             active cut line on the Axial image with label `CS #<idx> (<arc_mm> mm)`.

        Returns (P, T, N, B, arc_length_mm) for the active cross-section slice.
        """
        pts, arc_s, tangents, normals, binormals = PanoramicCurveLogic.get_curve_points_and_frames(
            curveNode, num_samples=max(2, int(num_slices))
        )
        clamped_idx = max(0, min(len(pts) - 1, int(active_index)))
        P = pts[clamped_idx]
        T = tangents[clamped_idx]
        N = normals[clamped_idx]
        B = binormals[clamped_idx]
        arc_mm = float(arc_s[clamped_idx])

        vol_name = volumeNode.GetName() if volumeNode else "CBCT"

        # 1. Multi-slice tick fan model (visible in 2D Axial + 3D)
        ticks_name = f"{vol_name}_CrossSectionTicks"
        ticksNode = slicer.mrmlScene.GetFirstNodeByName(ticks_name)
        if not ticksNode:
            ticksNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", ticks_name)
            ticksNode.CreateDefaultDisplayNodes()

        if vtk is not None and show_all_ticks:
            poly = self._build_tick_lines_polydata(pts, normals, width_mm=width_mm * 0.75)
            ticksNode.SetAndObservePolyData(poly)

        ticksDisp = ticksNode.GetDisplayNode()
        if ticksDisp:
            ticksDisp.SetColor(0.95, 0.75, 0.15)  # Amber-gold ticks
            ticksDisp.SetOpacity(0.65)
            ticksDisp.SetLineWidth(1.5)
            ticksDisp.SetVisibility(bool(show_all_ticks))
            ticksDisp.SetVisibility2D(bool(show_all_ticks))
            ticksDisp.SetVisibility3D(bool(show_all_ticks))
            if hasattr(ticksDisp, "SetSliceDisplayModeToProjection"):
                ticksDisp.SetSliceDisplayModeToProjection()

        # 2. Active Cross-Section Highlight Line (with slice index label on Axial image)
        active_name = f"{vol_name}_ActiveCrossSectionLine"
        activeNode = slicer.mrmlScene.GetFirstNodeByName(active_name)
        if not activeNode:
            activeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", active_name)
            activeNode.CreateDefaultDisplayNodes()

        activeNode.RemoveAllControlPoints()
        half_w = width_mm / 2.0
        p_buccal = P + N * half_w
        p_lingual = P - N * half_w
        activeNode.AddControlPoint(float(p_buccal[0]), float(p_buccal[1]), float(p_buccal[2]))
        activeNode.AddControlPoint(float(p_lingual[0]), float(p_lingual[1]), float(p_lingual[2]))

        label_text = f"CS #{clamped_idx + 1}/{len(pts)} ({arc_mm:.1f} mm)"
        if site_label:
            label_text += f" | {site_label}"
        activeNode.SetName(active_name)
        if hasattr(activeNode, "SetNthControlPointLabel"):
            activeNode.SetNthControlPointLabel(0, f"B | {label_text}")
            activeNode.SetNthControlPointLabel(1, "L")

        activeDisp = activeNode.GetDisplayNode()
        if activeDisp:
            activeDisp.SetColor(0.0, 0.90, 1.0)  # Bright cyan highlight
            activeDisp.SetSelectedColor(0.0, 0.90, 1.0)
            if hasattr(slicer.vtkMRMLMarkupsDisplayNode, "UseLineDiameter"):
                activeDisp.SetCurveLineSizeMode(slicer.vtkMRMLMarkupsDisplayNode.UseLineDiameter)
            activeDisp.SetLineDiameter(2.4)
            activeDisp.SetVisibility(True)
            activeDisp.SetVisibility2D(True)
            activeDisp.SetVisibility3D(True)
            if hasattr(activeDisp, "SetPropertiesLabelVisibility"):
                activeDisp.SetPropertiesLabelVisibility(True)
            if hasattr(activeDisp, "SetPointLabelsVisibility"):
                activeDisp.SetPointLabelsVisibility(True)

        return P, T, N, B, arc_mm

    def jump_to_frame(
        self,
        P: np.ndarray,
        T: np.ndarray,
        N: np.ndarray,
        B: np.ndarray,
        update_green_tangential: bool = True,
    ) -> None:
        """Align Yellow (Cross-Section), Green (Tangential), and Red (Axial) views to frame."""
        layoutManager = slicer.app.layoutManager() if getattr(slicer, "app", None) else None
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
            sNode.Modified()

        # Green -> Upright Tangential (Mesiodistal horizontal, Superior Up)
        if update_green_tangential:
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

        self.enable_native_slice_intersections(True)

    def jump_to_cross_section_index(
        self,
        volumeNode,
        curveNode,
        index: int,
        num_slices: int = 25,
        width_mm: float = 24.0,
        show_all_ticks: bool = True,
        update_green_tangential: bool = True,
        site_label: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Update Axial cross-section indicators and jump slice views to slice `index`."""
        P, T, N, B, arc_mm = self.update_axial_cross_section_indicators(
            volumeNode=volumeNode,
            curveNode=curveNode,
            active_index=index,
            num_slices=num_slices,
            width_mm=width_mm,
            show_all_ticks=show_all_ticks,
            site_label=site_label,
        )
        self.jump_to_frame(P, T, N, B, update_green_tangential=update_green_tangential)
        return P, T, N, B, arc_mm

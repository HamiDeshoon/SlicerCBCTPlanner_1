"""Volume Rendering (VR) and DentalSegmentator 3D visualization module.

Ports the 5-class anatomical surface presentation from gaudot/SlicerDentalSegmentator
and provides interactive CBCT Volume Rendering (VR) presets, non-cumulative HU shift
adjustment (with 4-tuple vtkPiecewiseFunction and 6-tuple vtkColorTransferFunction buffers),
and ROI cropping using 3D Slicer's VolumeRendering module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

from .Utils import (
    set3DViewBackgroundColors,
    setBoxAndTextVisibilityOnThreeDViews,
    setConventionalWideScreenView,
    toRGB,
)

# Exact 5-class anatomical presets from gaudot/SlicerDentalSegmentator
# (Segment Name, Hex Color, 3D Opacity)
DENTAL_SEGMENT_PRESETS: Dict[int, Tuple[str, str, float]] = {
    1: ("Maxilla & Upper Skull", "#E3DD90", 0.45),
    2: ("Mandible", "#D4A1E6", 0.45),
    3: ("Upper Teeth", "#DC9565", 1.0),
    4: ("Lower Teeth", "#EBDFB4", 1.0),
    5: ("Mandibular canal", "#D8654F", 1.0),
}


class VRVisualizationLogic:
    """Manages DentalSegmentator 3D surface styling and CBCT Volume Rendering."""

    SUPPORTED_VR_PRESETS = (
        "CT-Dental-CBCT",
        "CT-Bone",
        "CT-AAA",
        "CT-MIP",
        "CT-Cardiac",
    )

    def __init__(self) -> None:
        # Cache baseline transfer function control points keyed by (prop_node_id, preset_name)
        self._baseline_tf_cache: Dict[Tuple[str, str], Tuple[List[List[float]], List[List[float]]]] = {}

    @staticmethod
    def _get_segment_ids(segmentation: Any) -> List[str]:
        try:
            import vtk  # type: ignore

            ids = vtk.vtkStringArray()
            segmentation.GetSegmentIDs(ids)
            return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]
        except Exception:
            return []

    @staticmethod
    def _match_preset_by_name(segment_name: str) -> Optional[Tuple[str, str, float]]:
        lowered = (segment_name or "").lower()
        for key, hex_color, opacity in (
            ("maxilla", "#E3DD90", 0.45),
            ("skull", "#E3DD90", 0.45),
            ("upper jaw", "#E3DD90", 0.45),
            ("mandible", "#D4A1E6", 0.45),
            ("lower jaw", "#D4A1E6", 0.45),
            ("upper teeth", "#DC9565", 1.0),
            ("lower teeth", "#EBDFB4", 1.0),
            ("canal", "#D8654F", 1.0),
        ):
            if key in lowered:
                return segment_name, hex_color, opacity
        return None

    def configure_3d_view(
        self,
        white_background: bool = True,
        hide_box_and_labels: bool = True,
        switch_to_widescreen: bool = False,
        frontal_view: bool = True,
    ) -> None:
        """Configure 3D viewport presentation matching SlicerDentalSegmentator."""
        if slicer is None:
            return
        if switch_to_widescreen:
            setConventionalWideScreenView()
        if white_background:
            set3DViewBackgroundColors((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        else:
            set3DViewBackgroundColors((0.1, 0.12, 0.16), (0.04, 0.05, 0.08))
        if hide_box_and_labels:
            setBoxAndTextVisibilityOnThreeDViews(False)

        layoutManager = slicer.app.layoutManager() if getattr(slicer, "app", None) else None
        if layoutManager and frontal_view:
            for i in range(getattr(layoutManager, "threeDViewCount", 1)):
                threeD = layoutManager.threeDWidget(i)
                if i == 0 and threeD and hasattr(threeD, "threeDView"):
                    threeD.threeDView().rotateToViewAxis(3)
        slicer.util.resetThreeDViews()

    def apply_dental_segmentator_styling(
        self,
        segmentationNode: Any,
        smoothing_factor: float = 0.5,
        group_values: Optional[ list ] = None,
        configure_view: bool = True,
    ) -> None:
        """Apply DentalSegmentator colors, 3D opacities, and closed-surface smoothing."""
        if not segmentationNode:
            return
        segmentation = segmentationNode.GetSegmentation()
        if hasattr(segmentation, "SetConversionParameter"):
            segmentation.SetConversionParameter("Smoothing factor", str(smoothing_factor))
        segmentationNode.CreateClosedSurfaceRepresentation()

        if not segmentationNode.GetDisplayNode():
            segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        segmentationNode.SetDisplayVisibility(True)

        seg_ids = self._get_segment_ids(segmentation)
        if not group_values or len(group_values) != len(seg_ids):
            group_values = []

        for index, seg_id in enumerate(seg_ids):
            segment = segmentation.GetSegment(seg_id)
            if segment is None:
                continue
            spec = DENTAL_SEGMENT_PRESETS.get(group_values[index]) if group_values else None
            if spec is None:
                spec = self._match_preset_by_name(segment.GetName() or "")
            if spec is None and len(seg_ids) == len(DENTAL_SEGMENT_PRESETS):
                spec = DENTAL_SEGMENT_PRESETS.get(index + 1)
            if spec is None:
                continue

            name, hex_color, opacity = spec
            segment.SetName(name)
            segment.SetColor(*toRGB(hex_color))
            if displayNode:
                displayNode.SetSegmentVisibility(seg_id, True)
                if hasattr(displayNode, "SetSegmentVisibility3D"):
                    displayNode.SetSegmentVisibility3D(seg_id, True)
                displayNode.SetSegmentOpacity3D(seg_id, opacity)

        if displayNode:
            displayNode.SetVisibility(True)
            if hasattr(displayNode, "SetAllSegmentsVisibility"):
                displayNode.SetAllSegmentsVisibility(True)

        if configure_view:
            self.configure_3d_view()

    @staticmethod
    def _apply_custom_dental_cbct_transfer_function(volumePropertyNode: Any) -> None:
        """Configure a CBCT-optimized bone + enamel transfer function (HU/gray-value)."""
        if not volumePropertyNode:
            return
        opacityTF = volumePropertyNode.GetScalarOpacity()
        colorTF = volumePropertyNode.GetColor()
        gradientTF = volumePropertyNode.GetGradientOpacity()

        if opacityTF:
            opacityTF.RemoveAllPoints()
            opacityTF.AddPoint(-1000.0, 0.0)
            opacityTF.AddPoint(200.0, 0.0)
            opacityTF.AddPoint(450.0, 0.18)
            opacityTF.AddPoint(900.0, 0.55)
            opacityTF.AddPoint(1600.0, 0.88)
            opacityTF.AddPoint(3000.0, 1.0)

        if colorTF:
            colorTF.RemoveAllPoints()
            colorTF.AddRGBPoint(-1000.0, 0.0, 0.0, 0.0)
            colorTF.AddRGBPoint(200.0, 0.55, 0.25, 0.18)
            colorTF.AddRGBPoint(600.0, 0.88, 0.78, 0.58)
            colorTF.AddRGBPoint(1300.0, 0.96, 0.94, 0.86)
            colorTF.AddRGBPoint(2500.0, 1.0, 1.0, 1.0)

        if gradientTF:
            gradientTF.RemoveAllPoints()
            gradientTF.AddPoint(0.0, 0.0)
            gradientTF.AddPoint(60.0, 0.8)
            gradientTF.AddPoint(200.0, 1.0)

        volumePropertyNode.SetShade(True)
        volumePropertyNode.SetAmbient(0.2)
        volumePropertyNode.SetDiffuse(0.85)
        volumePropertyNode.SetSpecular(0.25)

    def get_or_create_vr_display_node(self, volumeNode: Any) -> Tuple[Any, bool]:
        """Return (vtkMRMLVolumeRenderingDisplayNode, created_new)."""
        if slicer is None or not volumeNode:
            return None, False
        volRenLogic = slicer.modules.volumerendering.logic()
        displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if not displayNode:
            displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
            return displayNode, True
        return displayNode, False

    @staticmethod
    def _extract_tf_points(
        volumePropertyNode: Any,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Read 4-element scalar opacity points and 6-element color TF points."""
        opacity_pts: List[List[float]] = []
        color_pts: List[List[float]] = []
        if not volumePropertyNode:
            return opacity_pts, color_pts

        opacity_tf = volumePropertyNode.GetScalarOpacity()
        if opacity_tf:
            n_op = opacity_tf.GetSize()
            buf4 = [0.0, 0.0, 0.5, 0.0]
            for i in range(n_op):
                opacity_tf.GetNodeValue(i, buf4)
                opacity_pts.append(list(buf4))

        color_tf = volumePropertyNode.GetColor()
        if color_tf:
            n_col = color_tf.GetSize()
            buf6 = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
            for i in range(n_col):
                color_tf.GetNodeValue(i, buf6)
                color_pts.append(list(buf6))

        return opacity_pts, color_pts

    @classmethod
    def shift_transfer_function(
        cls,
        volumePropertyNode: Any,
        shift_hu: float,
        base_opacity_pts: Optional[List[List[float]]] = None,
        base_color_pts: Optional[List[List[float]]] = None,
    ) -> None:
        """Shift scalar opacity (4-value) and color transfer function (6-value) points by shift_hu.

        vtkPiecewiseFunction.GetNodeValue / SetNodeValue uses 4 values: [x, opacity, midpoint, sharpness].
        vtkColorTransferFunction.GetNodeValue / SetNodeValue uses 6 values: [x, r, g, b, midpoint, sharpness].
        """
        if not volumePropertyNode:
            return
        if base_opacity_pts is None or base_color_pts is None:
            curr_op, curr_col = cls._extract_tf_points(volumePropertyNode)
            if base_opacity_pts is None:
                base_opacity_pts = curr_op
            if base_color_pts is None:
                base_color_pts = curr_col

        opacity_tf = volumePropertyNode.GetScalarOpacity()
        if opacity_tf and base_opacity_pts:
            for i, pt4 in enumerate(base_opacity_pts):
                shifted4 = [float(pt4[0]) + float(shift_hu), float(pt4[1]), float(pt4[2]), float(pt4[3])]
                opacity_tf.SetNodeValue(i, shifted4)

        color_tf = volumePropertyNode.GetColor()
        if color_tf and base_color_pts:
            for i, pt6 in enumerate(base_color_pts):
                shifted6 = [
                    float(pt6[0]) + float(shift_hu),
                    float(pt6[1]),
                    float(pt6[2]),
                    float(pt6[3]),
                    float(pt6[4]),
                    float(pt6[5]),
                ]
                color_tf.SetNodeValue(i, shifted6)

    def set_volume_rendering(
        self,
        volumeNode: Any,
        visible: bool = True,
        preset_name: str = "CT-Dental-CBCT",
        shift_hu: float = 0.0,
        crop_enabled: bool = False,
    ) -> Any:
        """Enable/disable Volume Rendering without resetting 3D view or ROI on slider ticks."""
        if slicer is None or not volumeNode:
            return None
        volRenLogic = slicer.modules.volumerendering.logic()
        displayNode, created_new = self.get_or_create_vr_display_node(volumeNode)
        if not displayNode:
            return None

        was_visible = bool(displayNode.GetVisibility()) if not created_new else False
        was_cropping = bool(displayNode.GetCroppingEnabled()) if not created_new else False

        propNode = displayNode.GetVolumePropertyNode()
        if propNode:
            prop_id = propNode.GetID() if hasattr(propNode, "GetID") else str(id(propNode))
            cache_key = (prop_id, preset_name)
            if cache_key not in self._baseline_tf_cache:
                if preset_name == "CT-Dental-CBCT":
                    self._apply_custom_dental_cbct_transfer_function(propNode)
                else:
                    preset = volRenLogic.GetPresetByName(preset_name)
                    if preset:
                        propNode.Copy(preset)
                    else:
                        self._apply_custom_dental_cbct_transfer_function(propNode)
                self._baseline_tf_cache[cache_key] = self._extract_tf_points(propNode)

            base_op, base_col = self._baseline_tf_cache[cache_key]
            self.shift_transfer_function(
                propNode,
                shift_hu=shift_hu,
                base_opacity_pts=base_op,
                base_color_pts=base_col,
            )

        displayNode.SetCroppingEnabled(bool(crop_enabled))
        roiNode = displayNode.GetROINode()
        if crop_enabled and not was_cropping:
            volRenLogic.FitROIToVolume(displayNode)
            roiNode = displayNode.GetROINode()
        if roiNode and roiNode.GetDisplayNode():
            roiNode.GetDisplayNode().SetVisibility(bool(crop_enabled and visible))

        displayNode.SetVisibility(bool(visible))
        if visible and not was_visible:
            slicer.util.resetThreeDViews()
        return displayNode

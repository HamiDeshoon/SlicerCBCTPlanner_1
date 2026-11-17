"""Volume Rendering (VR) and SlicerDentalSegmentator Vivid Teeth & Roots 3D Visualization Module.

Upgrades & Guarantees:
  - Vivid SlicerDentalSegmentator 3D surface styling (`https://github.com/gaudot/SlicerDentalSegmentator`):
      * Vivid teeth & roots (`1.0` 3D opacity, high-contrast pearlescent/enamel colors, Phong specular
        highlighting so root apices, furcations, and crown cusps are razor-sharp).
      * Translucent glass-bone Maxilla & Mandible (`0.25` default 3D opacity, adjustable `0.0 - 1.0`
        via live slider) so tooth roots and the inferior alveolar nerve canal are vividly visible
        through the alveolar bone.
      * Multi-class ToothFairy3 (`77` classes) AND 5-class DentalSegmentator styling support.
      * One-click 3D Style Presets:
          1. `"DentalSegmentator Vivid (Teeth & Roots)"` (Default — translucent bone 0.25, vivid roots 1.0, glowing red canal 1.0)
          2. `"Roots & Nerve X-Ray (Hide Bone)"` (Bone 0.05, vivid teeth & roots 1.0, canal 1.0)
          3. `"Endodontic Root-Canal View (Translucent Teeth)"` (Bone 0.08, teeth 0.30, pulp/root canals 1.0, IAN canal 1.0)
          4. `"Surgical Bone & Teeth (Opaque Bone)"` (Bone 0.85, teeth 1.0, canal 1.0)
  - Fixes B8 & O7:
      * Always re-applies the selected VR preset when switching presets (`A -> B -> A`) so transfer
        functions with different point counts are never corrupted by index-based overwrite.
      * Caches only the current active preset's unshifted baseline points.
      * Batches transfer function updates with `DisableModifiedEventOn/Off` / `StartModify/EndModify`.
      * Detects non-HU CBCT intensity ranges (e.g. `min > -500` or `max > 10000`) and derives custom
        CBCT bone/enamel transfer function points from volume intensity percentiles instead of HU constants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

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
from .core.labels import (
    LABELS_BY_ID,
    match_segment_to_label_and_precedence,
    safe_get_segment_tag,
)

# Vivid 5-class anatomical presets from gaudot/SlicerDentalSegmentator
# (Segment Name, Hex Color, Default 3D Opacity — translucent bone makes tooth roots vivid!)
DENTAL_SEGMENT_PRESETS: Dict[int, Tuple[str, str, float]] = {
    1: ("Maxilla & Upper Skull", "#E8DF88", 0.25),
    2: ("Mandible", "#D8A4EC", 0.25),
    3: ("Upper Teeth", "#FFAB66", 1.0),
    4: ("Lower Teeth", "#FFF3C4", 1.0),
    5: ("Mandibular canal", "#FF3B30", 1.0),
}

# 3D Surface Style Presets -> (jaw_opacity, tooth_opacity, canal_opacity, pulp_opacity, sinus_opacity)
VIVID_3D_STYLE_PRESETS: Dict[str, Tuple[float, float, float, float, float]] = {
    "DentalSegmentator Vivid (Teeth & Roots)": (0.25, 1.0, 1.0, 1.0, 0.25),
    "Roots & Nerve X-Ray (Hide Bone)": (0.05, 1.0, 1.0, 1.0, 0.12),
    "Endodontic Root-Canal View (Translucent Teeth)": (0.08, 0.30, 1.0, 1.0, 0.15),
    "Surgical Bone & Teeth (Opaque Bone)": (0.85, 1.0, 1.0, 1.0, 0.35),
}


class VRVisualizationLogic:
    """Manages SlicerDentalSegmentator vivid 3D surface styling and CBCT Volume Rendering."""

    SUPPORTED_VR_PRESETS = (
        "DentalSegmentator (Teeth & Bone)",
        "DentalSegmentator Vivid Roots (VR)",
        "CT-Dental-CBCT",
        "CT-Bone",
        "CT-AAA",
        "CT-MIP",
        "CT-Cardiac",
    )

    SUPPORTED_3D_SURFACE_PRESETS = tuple(VIVID_3D_STYLE_PRESETS.keys())

    def __init__(self) -> None:
        # B8: Track only the currently active preset and its unshifted baseline per property node ID
        self._active_preset_by_prop: Dict[str, str] = {}
        self._active_baseline_by_prop: Dict[str, Tuple[List[List[float]], List[List[float]]]] = {}

    @staticmethod
    def _get_segment_ids(segmentation: Any) -> List[str]:
        try:
            import vtk  # type: ignore

            ids = vtk.vtkStringArray()
            segmentation.GetSegmentIDs(ids)
            return [ids.GetValue(i) for i in range(ids.GetNumberOfValues())]
        except Exception:
            if hasattr(segmentation, "GetNumberOfSegments") and hasattr(segmentation, "GetNthSegmentID"):
                return [
                    segmentation.GetNthSegmentID(i) for i in range(segmentation.GetNumberOfSegments())
                ]
            return []

    @staticmethod
    def _match_preset_by_name(segment_name: str) -> Optional[Tuple[str, str, float]]:
        lowered = (segment_name or "").lower()
        for key, hex_color, opacity in (
            ("maxilla", "#E8DF88", 0.25),
            ("skull", "#E8DF88", 0.25),
            ("upper jaw", "#E8DF88", 0.25),
            ("mandible", "#D8A4EC", 0.25),
            ("lower jaw", "#D8A4EC", 0.25),
            ("upper teeth", "#FFAB66", 1.0),
            ("lower teeth", "#FFF3C4", 1.0),
            ("canal", "#FF3B30", 1.0),
            ("nerve", "#FF3B30", 1.0),
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
            set3DViewBackgroundColors((0.97, 0.98, 1.0), (0.88, 0.91, 0.96))
        else:
            set3DViewBackgroundColors((0.08, 0.11, 0.16), (0.03, 0.04, 0.07))
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
        smoothing_factor: float = 0.40,
        group_values: Optional[List[int]] = None,
        configure_view: bool = True,
        style_preset: str = "DentalSegmentator Vivid (Teeth & Roots)",
        jaw_opacity_override: Optional[float] = None,
    ) -> None:
        """Apply vivid SlicerDentalSegmentator colors, translucent bone for root visibility, and Phong shading."""
        if not segmentationNode:
            return
        segmentation = segmentationNode.GetSegmentation()
        if hasattr(segmentation, "SetConversionParameter"):
            segmentation.SetConversionParameter("Smoothing factor", str(float(smoothing_factor)))
        if hasattr(segmentationNode, "CreateClosedSurfaceRepresentation"):
            segmentationNode.CreateClosedSurfaceRepresentation()

        if not segmentationNode.GetDisplayNode() and hasattr(segmentationNode, "CreateDefaultDisplayNodes"):
            segmentationNode.CreateDefaultDisplayNodes()
        displayNode = segmentationNode.GetDisplayNode()
        if hasattr(segmentationNode, "SetDisplayVisibility"):
            segmentationNode.SetDisplayVisibility(True)

        jaw_op, tooth_op, canal_op, pulp_op, sinus_op = VIVID_3D_STYLE_PRESETS.get(
            style_preset, (0.25, 1.0, 1.0, 1.0, 0.25)
        )
        if jaw_opacity_override is not None:
            jaw_op = float(np.clip(jaw_opacity_override, 0.0, 1.0))

        seg_ids = self._get_segment_ids(segmentation)
        if not group_values or len(group_values) != len(seg_ids):
            group_values = []

        was_modifying = (
            displayNode.StartModify() if (displayNode and hasattr(displayNode, "StartModify")) else None
        )
        try:
            for index, seg_id in enumerate(seg_ids):
                segment = segmentation.GetSegment(seg_id)
                if segment is None:
                    continue

                seg_name = segment.GetName() or ""
                fdi_attr = safe_get_segment_tag(segment, "CBCTPlanner.FDI")
                label_id, category, _is_5c, _prec = match_segment_to_label_and_precedence(
                    seg_id, seg_name, fdi_attribute=fdi_attr
                )

                spec = DENTAL_SEGMENT_PRESETS.get(group_values[index]) if group_values else None
                if spec is None and label_id in LABELS_BY_ID and label_id >= 3 and len(seg_ids) > 5:
                    info = LABELS_BY_ID[label_id]
                    rgb = info.color
                    if category == "tooth":
                        opacity = tooth_op
                    elif category == "pulp":
                        opacity = pulp_op
                    elif category == "canal":
                        opacity = canal_op
                    elif category == "sinus":
                        opacity = sinus_op
                    elif category == "jaw":
                        opacity = jaw_op
                    else:
                        opacity = 0.85
                    segment.SetName(info.name)
                    segment.SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))
                else:
                    if spec is None:
                        spec = self._match_preset_by_name(seg_name)
                    if spec is None and len(seg_ids) == len(DENTAL_SEGMENT_PRESETS):
                        spec = DENTAL_SEGMENT_PRESETS.get(index + 1)
                    if spec is not None:
                        name, hex_color, default_op = spec
                        segment.SetName(name)
                        segment.SetColor(*toRGB(hex_color))
                        name_l = name.lower()
                        if "teeth" in name_l or "tooth" in name_l:
                            opacity = tooth_op
                        elif "canal" in name_l or "nerve" in name_l:
                            opacity = canal_op
                        elif "maxilla" in name_l or "mandible" in name_l or "jaw" in name_l or "skull" in name_l:
                            opacity = jaw_op
                        else:
                            opacity = default_op
                    elif label_id in LABELS_BY_ID:
                        info = LABELS_BY_ID[label_id]
                        segment.SetName(info.name)
                        segment.SetColor(*info.color)
                        opacity = jaw_op if info.category == "jaw" else tooth_op
                    else:
                        continue

                if displayNode:
                    displayNode.SetSegmentVisibility(seg_id, True)
                    if hasattr(displayNode, "SetSegmentVisibility3D"):
                        displayNode.SetSegmentVisibility3D(seg_id, opacity > 0.01)
                    if hasattr(displayNode, "SetSegmentOpacity3D"):
                        displayNode.SetSegmentOpacity3D(seg_id, float(opacity))
                    if hasattr(displayNode, "SetSegmentOpacity2DFill"):
                        displayNode.SetSegmentOpacity2DFill(
                            seg_id, 0.18 if opacity < 0.5 else 0.30
                        )
                    if hasattr(displayNode, "SetSegmentOpacity2DOutline"):
                        displayNode.SetSegmentOpacity2DOutline(seg_id, 0.95)

            if displayNode:
                displayNode.SetVisibility(True)
                if hasattr(displayNode, "SetVisibility3D"):
                    displayNode.SetVisibility3D(True)
                if hasattr(displayNode, "SetVisibility2DOutline"):
                    displayNode.SetVisibility2DOutline(True)
                if hasattr(displayNode, "SetSliceIntersectionThickness"):
                    displayNode.SetSliceIntersectionThickness(2)
                if hasattr(displayNode, "SetAllSegmentsVisibility"):
                    displayNode.SetAllSegmentsVisibility(True)
        finally:
            if displayNode and was_modifying is not None and hasattr(displayNode, "EndModify"):
                displayNode.EndModify(was_modifying)

        if configure_view:
            self.configure_3d_view()

    def set_jawbone_3d_opacity(
        self,
        segmentationNode: Any,
        jaw_opacity: float = 0.25,
        tooth_opacity: float = 1.0,
    ) -> None:
        """Dynamically adjust 3D opacity of jawbone vs teeth/roots so tooth roots remain vivid."""
        if not segmentationNode:
            return
        segmentation = segmentationNode.GetSegmentation()
        displayNode = segmentationNode.GetDisplayNode()
        if not segmentation or not displayNode:
            return
        j_op = float(np.clip(jaw_opacity, 0.0, 1.0))
        t_op = float(np.clip(tooth_opacity, 0.05, 1.0))
        seg_ids = self._get_segment_ids(segmentation)
        was_mod = displayNode.StartModify() if hasattr(displayNode, "StartModify") else None
        try:
            for sid in seg_ids:
                seg = segmentation.GetSegment(sid)
                if not seg:
                    continue
                name_l = (seg.GetName() or "").lower()
                fdi_attr = safe_get_segment_tag(seg, "CBCTPlanner.FDI")
                _lbl, cat, _is5, _pr = match_segment_to_label_and_precedence(
                    sid, seg.GetName() or "", fdi_attribute=fdi_attr
                )
                if cat == "jaw" or any(k in name_l for k in ("maxilla", "mandible", "jaw", "skull")):
                    if hasattr(displayNode, "SetSegmentOpacity3D"):
                        displayNode.SetSegmentOpacity3D(sid, j_op)
                    if hasattr(displayNode, "SetSegmentVisibility3D"):
                        displayNode.SetSegmentVisibility3D(sid, j_op > 0.01)
                elif cat == "tooth" or "teeth" in name_l or "tooth" in name_l:
                    if hasattr(displayNode, "SetSegmentOpacity3D"):
                        displayNode.SetSegmentOpacity3D(sid, t_op)
                    if hasattr(displayNode, "SetSegmentVisibility3D"):
                        displayNode.SetSegmentVisibility3D(sid, True)
        finally:
            if was_mod is not None and hasattr(displayNode, "EndModify"):
                displayNode.EndModify(was_mod)

    @staticmethod
    def is_volume_hu_calibrated(volume_array: Optional[np.ndarray]) -> bool:
        """B8: Return False when volume intensities are not HU-like (e.g. min > -500 or max > 10000)."""
        if volume_array is None or volume_array.size == 0:
            return True
        arr = np.asarray(volume_array)
        step = max(1, arr.size // 50000)
        sample = arr.reshape(-1)[::step]
        vmin = float(np.min(sample))
        vmax = float(np.max(sample))
        if vmin > -500.0 or vmax > 10000.0 or vmax <= 300.0:
            return False
        return True

    @classmethod
    def compute_cbct_transfer_control_points(
        cls, volume_array: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
        """B8: Compute opacity `(x, alpha)` and color `(x, r, g, b)` control points.

        Uses standard HU thresholds when `volume_array` is HU-like, or derives thresholds
        from intensity percentiles (`p10, p55, p75, p90, p98, p99.5`) when `volume_array`
        has uncalibrated gray values (e.g. `min > -500` or `max > 10000`).
        """
        if cls.is_volume_hu_calibrated(volume_array):
            op_pts = [
                (-1000.0, 0.0),
                (200.0, 0.0),
                (450.0, 0.18),
                (900.0, 0.55),
                (1600.0, 0.88),
                (3000.0, 1.0),
            ]
            col_pts = [
                (-1000.0, 0.0, 0.0, 0.0),
                (200.0, 0.55, 0.25, 0.18),
                (600.0, 0.88, 0.78, 0.58),
                (1300.0, 0.96, 0.94, 0.86),
                (2500.0, 1.0, 1.0, 1.0),
            ]
            return op_pts, col_pts

        assert volume_array is not None
        arr = np.asarray(volume_array)
        step = max(1, arr.size // 50000)
        sample = arr.reshape(-1)[::step].astype(np.float64)
        p10, p55, p75, p90, p98, p995 = np.percentile(sample, [10.0, 55.0, 75.0, 90.0, 98.0, 99.5])
        # Ensure strictly increasing x values
        xs = np.maximum.accumulate([p10, p55, p75, p90, p98, p995])
        for i in range(1, len(xs)):
            if xs[i] <= xs[i - 1]:
                xs[i] = xs[i - 1] + 1.0

        op_pts = [
            (float(xs[0]), 0.0),
            (float(xs[1]), 0.0),
            (float(xs[2]), 0.18),
            (float(xs[3]), 0.55),
            (float(xs[4]), 0.88),
            (float(xs[5]), 1.0),
        ]
        col_pts = [
            (float(xs[0]), 0.0, 0.0, 0.0),
            (float(xs[1]), 0.55, 0.25, 0.18),
            (float(0.5 * (xs[2] + xs[3])), 0.88, 0.78, 0.58),
            (float(xs[4]), 0.96, 0.94, 0.86),
            (float(xs[5]), 1.0, 1.0, 1.0),
        ]
        return op_pts, col_pts

    @classmethod
    def _apply_custom_dental_cbct_transfer_function(
        cls,
        volumePropertyNode: Any,
        volume_array: Optional[np.ndarray] = None,
    ) -> None:
        """Configure a CBCT-optimized bone + enamel transfer function (HU or percentile-adaptive)."""
        if not volumePropertyNode:
            return
        was_mod = volumePropertyNode.StartModify() if hasattr(volumePropertyNode, "StartModify") else None
        if hasattr(volumePropertyNode, "DisableModifiedEventOn"):
            volumePropertyNode.DisableModifiedEventOn()
        try:
            op_spec, col_spec = cls.compute_cbct_transfer_control_points(volume_array)
            opacityTF = volumePropertyNode.GetScalarOpacity()
            colorTF = volumePropertyNode.GetColor()
            gradientTF = volumePropertyNode.GetGradientOpacity()

            if opacityTF:
                opacityTF.RemoveAllPoints()
                for x_val, alpha in op_pts_iter(op_spec):
                    opacityTF.AddPoint(float(x_val), float(alpha))

            if colorTF:
                colorTF.RemoveAllPoints()
                for x_val, r, g, b in col_spec:
                    colorTF.AddRGBPoint(float(x_val), float(r), float(g), float(b))

            if gradientTF:
                gradientTF.RemoveAllPoints()
                gradientTF.AddPoint(0.0, 0.0)
                gradientTF.AddPoint(60.0, 0.8)
                gradientTF.AddPoint(200.0, 1.0)

            if hasattr(volumePropertyNode, "SetShade"):
                volumePropertyNode.SetShade(True)
            if hasattr(volumePropertyNode, "SetAmbient"):
                volumePropertyNode.SetAmbient(0.2)
            if hasattr(volumePropertyNode, "SetDiffuse"):
                volumePropertyNode.SetDiffuse(0.85)
            if hasattr(volumePropertyNode, "SetSpecular"):
                volumePropertyNode.SetSpecular(0.25)
        finally:
            if hasattr(volumePropertyNode, "DisableModifiedEventOff"):
                volumePropertyNode.DisableModifiedEventOff()
            if was_mod is not None and hasattr(volumePropertyNode, "EndModify"):
                volumePropertyNode.EndModify(was_mod)

    @classmethod
    def _apply_dental_segmentator_transfer_function(
        cls,
        volumePropertyNode: Any,
        volume_array: Optional[np.ndarray] = None,
        vivid_roots_mode: bool = False,
    ) -> None:
        """DentalSegmentator-style VR: enamel and tooth roots vivid and bright, alveolar bone translucent."""
        if not volumePropertyNode:
            return
        if volume_array is not None and np.asarray(volume_array).size:
            sample = np.asarray(volume_array).reshape(-1)
            step = max(1, sample.size // 50000)
            values = sample[::step].astype(np.float64)
            p50, p75, p86, p93, p97, p995 = np.percentile(values, [50.0, 75.0, 86.0, 93.0, 97.0, 99.5])
            xs = np.maximum.accumulate([p50, p75, p86, p93, p97, p995])
            for index in range(1, len(xs)):
                if xs[index] <= xs[index - 1]:
                    xs[index] = xs[index - 1] + 1.0
            opacities = (
                [0.0, 0.0, 0.025, 0.14, 0.78, 1.0]
                if vivid_roots_mode
                else [0.0, 0.0, 0.045, 0.28, 0.62, 1.0]
            )
        else:
            xs = [-400.0, 80.0, 320.0, 650.0, 1200.0, 2600.0]
            opacities = (
                [0.0, 0.0, 0.025, 0.14, 0.80, 1.0]
                if vivid_roots_mode
                else [0.0, 0.0, 0.05, 0.32, 0.7, 1.0]
            )
        colors = [
            (0.0, 0.0, 0.0),
            (0.45, 0.18, 0.12),
            (0.82, 0.58, 0.38),
            (0.95, 0.83, 0.58),
            (1.0, 0.95, 0.82),
            (1.0, 1.0, 0.98),
        ]

        was_mod = volumePropertyNode.StartModify() if hasattr(volumePropertyNode, "StartModify") else None
        if hasattr(volumePropertyNode, "DisableModifiedEventOn"):
            volumePropertyNode.DisableModifiedEventOn()
        try:
            opacity_tf = volumePropertyNode.GetScalarOpacity()
            color_tf = volumePropertyNode.GetColor()
            gradient_tf = volumePropertyNode.GetGradientOpacity()
            if opacity_tf:
                opacity_tf.RemoveAllPoints()
                for x_val, alpha in zip(xs, opacities):
                    opacity_tf.AddPoint(float(x_val), float(alpha))
            if color_tf:
                color_tf.RemoveAllPoints()
                for x_val, (r, g, b) in zip(xs, colors):
                    color_tf.AddRGBPoint(float(x_val), float(r), float(g), float(b))
            if gradient_tf:
                gradient_tf.RemoveAllPoints()
                gradient_tf.AddPoint(0.0, 0.0)
                gradient_tf.AddPoint(70.0, 0.65)
                gradient_tf.AddPoint(260.0, 1.0)
            if hasattr(volumePropertyNode, "SetShade"):
                volumePropertyNode.SetShade(True)
            if hasattr(volumePropertyNode, "SetAmbient"):
                volumePropertyNode.SetAmbient(0.25)
            if hasattr(volumePropertyNode, "SetDiffuse"):
                volumePropertyNode.SetDiffuse(0.88)
            if hasattr(volumePropertyNode, "SetSpecular"):
                volumePropertyNode.SetSpecular(0.45 if vivid_roots_mode else 0.35)
            if hasattr(volumePropertyNode, "SetSpecularPower"):
                volumePropertyNode.SetSpecularPower(26.0 if vivid_roots_mode else 18.0)
        finally:
            if hasattr(volumePropertyNode, "DisableModifiedEventOff"):
                volumePropertyNode.DisableModifiedEventOff()
            if was_mod is not None and hasattr(volumePropertyNode, "EndModify"):
                volumePropertyNode.EndModify(was_mod)

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
        """B8 & O7: Shift scalar opacity (4-value) and color TF (6-value) points with modified events disabled."""
        if not volumePropertyNode:
            return
        if base_opacity_pts is None or base_color_pts is None:
            curr_op, curr_col = cls._extract_tf_points(volumePropertyNode)
            if base_opacity_pts is None:
                base_opacity_pts = curr_op
            if base_color_pts is None:
                base_color_pts = curr_col

        was_mod = volumePropertyNode.StartModify() if hasattr(volumePropertyNode, "StartModify") else None
        if hasattr(volumePropertyNode, "DisableModifiedEventOn"):
            volumePropertyNode.DisableModifiedEventOn()
        try:
            opacity_tf = volumePropertyNode.GetScalarOpacity()
            if opacity_tf and base_opacity_pts:
                if opacity_tf.GetSize() != len(base_opacity_pts) and hasattr(opacity_tf, "RemoveAllPoints"):
                    opacity_tf.RemoveAllPoints()
                    for pt4 in base_opacity_pts:
                        opacity_tf.AddPoint(
                            float(pt4[0]) + float(shift_hu),
                            float(pt4[1]),
                            float(pt4[2]),
                            float(pt4[3]),
                        )
                else:
                    for i, pt4 in enumerate(base_opacity_pts):
                        shifted4 = [
                            float(pt4[0]) + float(shift_hu),
                            float(pt4[1]),
                            float(pt4[2]),
                            float(pt4[3]),
                        ]
                        opacity_tf.SetNodeValue(i, shifted4)

            color_tf = volumePropertyNode.GetColor()
            if color_tf and base_color_pts:
                if color_tf.GetSize() != len(base_color_pts) and hasattr(color_tf, "RemoveAllPoints"):
                    color_tf.RemoveAllPoints()
                    for pt6 in base_color_pts:
                        color_tf.AddRGBPoint(
                            float(pt6[0]) + float(shift_hu),
                            float(pt6[1]),
                            float(pt6[2]),
                            float(pt6[3]),
                            float(pt6[4]),
                            float(pt6[5]),
                        )
                else:
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
        finally:
            if hasattr(volumePropertyNode, "DisableModifiedEventOff"):
                volumePropertyNode.DisableModifiedEventOff()
            if was_mod is not None and hasattr(volumePropertyNode, "EndModify"):
                volumePropertyNode.EndModify(was_mod)

    def apply_preset_and_shift(
        self,
        volumePropertyNode: Any,
        preset_name: str = "CT-Dental-CBCT",
        shift_hu: float = 0.0,
        preset_provider: Optional[Any] = None,
        volume_array: Optional[np.ndarray] = None,
    ) -> None:
        """B8: Always re-apply preset when `preset_name` changes (`A -> B -> A`), then apply `shift_hu`."""
        if not volumePropertyNode:
            return
        prop_id = (
            volumePropertyNode.GetID()
            if hasattr(volumePropertyNode, "GetID")
            else str(id(volumePropertyNode))
        )
        current_active = self._active_preset_by_prop.get(prop_id)
        if current_active != preset_name or prop_id not in self._active_baseline_by_prop:
            if preset_name == "CT-Dental-CBCT":
                self._apply_custom_dental_cbct_transfer_function(
                    volumePropertyNode, volume_array=volume_array
                )
            elif preset_name == "DentalSegmentator (Teeth & Bone)":
                self._apply_dental_segmentator_transfer_function(
                    volumePropertyNode, volume_array=volume_array, vivid_roots_mode=False
                )
            elif preset_name == "DentalSegmentator Vivid Roots (VR)":
                self._apply_dental_segmentator_transfer_function(
                    volumePropertyNode, volume_array=volume_array, vivid_roots_mode=True
                )
            else:
                preset = (
                    preset_provider.GetPresetByName(preset_name)
                    if (preset_provider and hasattr(preset_provider, "GetPresetByName"))
                    else None
                )
                if preset is not None and hasattr(volumePropertyNode, "Copy"):
                    volumePropertyNode.Copy(preset)
                else:
                    self._apply_custom_dental_cbct_transfer_function(
                        volumePropertyNode, volume_array=volume_array
                    )
            self._active_preset_by_prop[prop_id] = preset_name
            self._active_baseline_by_prop[prop_id] = self._extract_tf_points(volumePropertyNode)

        base_op, base_col = self._active_baseline_by_prop[prop_id]
        self.shift_transfer_function(
            volumePropertyNode,
            shift_hu=shift_hu,
            base_opacity_pts=base_op,
            base_color_pts=base_col,
        )

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
            try:
                vol_arr = slicer.util.arrayFromVolume(volumeNode)
            except Exception:
                vol_arr = None
            self.apply_preset_and_shift(
                volumePropertyNode=propNode,
                preset_name=preset_name,
                shift_hu=shift_hu,
                preset_provider=volRenLogic,
                volume_array=vol_arr,
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


def op_pts_iter(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return pts

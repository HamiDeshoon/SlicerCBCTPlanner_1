"""CBCTPlannerLib: Modular library for the SlicerCBCTPlanner 3D Slicer extension.

Provides:
- Utils: UI helpers, 3D viewport styling, and color conversion utilities adapted from SlicerDentalSegmentator.
- VRVisualization: DentalSegmentator 3D segmentation presets (colors, 3D opacities, surface smoothing) and CBCT Volume Rendering (VR) transfer functions.
- PanoramicCurveLogic: Dental arch curve generation, parallel transport frame calculation, curved planar reformation (CPR), and panoramic projection.
- CrossSectionNavigator: Synchronized orthogonal cross-sectional slice reslicing along the dental arch curve with real-time axial cross-section tick and plane indicators.
"""

from .Utils import (
    addInCollapsibleLayout,
    createButton,
    icon,
    iconPath,
    set3DViewBackgroundColors,
    setBoxAndTextVisibilityOnThreeDViews,
    setConventionalWideScreenView,
    setDentalFourUpLayout,
    toRGB,
)
from .VRVisualization import (
    DENTAL_SEGMENT_PRESETS,
    VRVisualizationLogic,
)
from .PanoramicCurveLogic import PanoramicCurveLogic
from .CrossSectionNavigator import CrossSectionNavigator

__all__ = [
    "DENTAL_SEGMENT_PRESETS",
    "VRVisualizationLogic",
    "PanoramicCurveLogic",
    "CrossSectionNavigator",
    "addInCollapsibleLayout",
    "createButton",
    "icon",
    "iconPath",
    "set3DViewBackgroundColors",
    "setBoxAndTextVisibilityOnThreeDViews",
    "setConventionalWideScreenView",
    "setDentalFourUpLayout",
    "toRGB",
]

"""UI, layout, and 3D viewport utilities adapted from SlicerDentalSegmentator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

try:
    import ctk  # type: ignore
    import qt  # type: ignore
    import slicer  # type: ignore
except ImportError:
    ctk = None
    qt = None
    slicer = None

CUSTOM_DENTAL_LAYOUT_ID = 502
CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID = 503

CUSTOM_DENTAL_LAYOUT_XML = """
<layout type="vertical" split="true">
  <item splitSize="450">
    <view class="vtkMRMLSliceNode" singletontag="Green">
      <property name="orientation" action="default">Axial</property>
      <property name="viewlabel" action="default">G</property>
      <property name="viewcolor" action="default">#6EB04B</property>
    </view>
  </item>
  <item splitSize="550">
    <layout type="horizontal" split="true">
      <item splitSize="333">
        <view class="vtkMRMLSliceNode" singletontag="Red">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">R</property>
          <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      <item splitSize="333">
        <view class="vtkMRMLSliceNode" singletontag="Yellow">
          <property name="orientation" action="default">Sagittal</property>
          <property name="viewlabel" action="default">Y</property>
          <property name="viewcolor" action="default">#EDD54C</property>
        </view>
      </item>
      <item splitSize="334">
        <view class="vtkMRMLViewNode" singletontag="1">
          <property name="viewlabel" action="default">1</property>
        </view>
      </item>
    </layout>
  </item>
</layout>
"""

# OnDemand3D / Romexis-style Multi-Slice Cross-Sectional Lightbox Layout:
# Top row: Panoramic (Green), Axial (Red), 3D View
# Bottom row: 3 consecutive Cross-Sectional slices (YellowMinus [-2mm], Yellow [0mm], YellowPlus [+2mm])
CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML = """
<layout type="vertical" split="true">
  <item splitSize="480">
    <layout type="horizontal" split="true">
      <item splitSize="420">
        <view class="vtkMRMLSliceNode" singletontag="Green">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">Pano</property>
          <property name="viewcolor" action="default">#6EB04B</property>
        </view>
      </item>
      <item splitSize="300">
        <view class="vtkMRMLSliceNode" singletontag="Red">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">Axial</property>
          <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      <item splitSize="280">
        <view class="vtkMRMLViewNode" singletontag="1">
          <property name="viewlabel" action="default">3D</property>
        </view>
      </item>
    </layout>
  </item>
  <item splitSize="520">
    <layout type="horizontal" split="true">
      <item splitSize="333">
        <view class="vtkMRMLSliceNode" singletontag="YellowMinus">
          <property name="orientation" action="default">Sagittal</property>
          <property name="viewlabel" action="default">CS-</property>
          <property name="viewcolor" action="default">#F59E0B</property>
        </view>
      </item>
      <item splitSize="334">
        <view class="vtkMRMLSliceNode" singletontag="Yellow">
          <property name="orientation" action="default">Sagittal</property>
          <property name="viewlabel" action="default">CS0</property>
          <property name="viewcolor" action="default">#06B6D4</property>
        </view>
      </item>
      <item splitSize="333">
        <view class="vtkMRMLSliceNode" singletontag="YellowPlus">
          <property name="orientation" action="default">Sagittal</property>
          <property name="viewlabel" action="default">CS+</property>
          <property name="viewcolor" action="default">#10B981</property>
        </view>
      </item>
    </layout>
  </item>
</layout>
"""


def toRGB(hex_color: str) -> Tuple[float, float, float]:
    """Convert a hex color string (e.g. '#E3DD90') to normalized RGB floats in [0.0, 1.0]."""
    clean_hex = hex_color.lstrip("#")
    if len(clean_hex) != 6:
        raise ValueError(f"Expected 6-character hex color, got: {hex_color}")
    return (
        int(clean_hex[0:2], 16) / 255.0,
        int(clean_hex[2:4], 16) / 255.0,
        int(clean_hex[4:6], 16) / 255.0,
    )


def iconPath(icon_name: str) -> str:
    """Return the absolute path to an icon inside Resources/Icons."""
    base_dir = Path(__file__).resolve().parent.parent
    return str(base_dir / "Resources" / "Icons" / icon_name)


def icon(icon_name: str) -> Any:
    """Return a QIcon for the specified icon filename if it exists."""
    if qt is None:
        return None
    path = iconPath(icon_name)
    return qt.QIcon(path) if Path(path).exists() else qt.QIcon()


def createButton(
    text: str,
    callback: Optional[Callable] = None,
    isCheckable: bool = False,
    icon: Any = None,
    toolTip: str = "",
    parent: Any = None,
) -> Any:
    """Create a configured QPushButton following SlicerDentalSegmentator conventions."""
    if qt is None:
        return None
    btn = qt.QPushButton(text, parent)
    btn.setCheckable(isCheckable)
    if icon is not None and hasattr(icon, "isNull") and not icon.isNull():
        btn.setIcon(icon)
    if toolTip:
        btn.setToolTip(toolTip)
    if callback is not None:
        btn.clicked.connect(callback)
    return btn


def addInCollapsibleLayout(
    childWidget: Any,
    parentLayout: Any,
    text: str,
    isCollapsed: bool = False,
) -> Any:
    """Wrap a widget inside a ctkCollapsibleButton and add it to parentLayout."""
    if ctk is None or qt is None:
        return None
    collapsible = ctk.ctkCollapsibleButton()
    collapsible.text = text
    collapsible.collapsed = isCollapsed
    vbox = qt.QVBoxLayout(collapsible)
    vbox.addWidget(childWidget)
    parentLayout.addWidget(collapsible)
    return collapsible


def setConventionalWideScreenView() -> None:
    """Switch 3D Slicer layout to Conventional Widescreen view."""
    if slicer is None or not getattr(slicer, "app", None):
        return
    layoutManager = slicer.app.layoutManager()
    if layoutManager:
        layoutManager.setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView
        )


def registerAndSetDentalLayout() -> None:
    """Register and activate custom Dental layout: Wide Panoramic top; Axial, Cross-Section, 3D bottom."""
    if slicer is None or not getattr(slicer, "app", None):
        return
    layoutManager = slicer.app.layoutManager()
    if not layoutManager:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if layoutNode:
        if not layoutNode.GetLayoutDescription(CUSTOM_DENTAL_LAYOUT_ID):
            layoutNode.AddLayoutDescription(
                CUSTOM_DENTAL_LAYOUT_ID, CUSTOM_DENTAL_LAYOUT_XML
            )
        else:
            layoutNode.SetLayoutDescription(
                CUSTOM_DENTAL_LAYOUT_ID, CUSTOM_DENTAL_LAYOUT_XML
            )
        layoutManager.setLayout(CUSTOM_DENTAL_LAYOUT_ID)


def registerAndSetDentalLightboxLayout(volumeNode: Any = None) -> None:
    """Register and activate the 3-slice Cross-Sectional Lightbox layout (YellowMinus, Yellow, YellowPlus)."""
    if slicer is None or not getattr(slicer, "app", None):
        return
    layoutManager = slicer.app.layoutManager()
    if not layoutManager:
        return
    layoutNode = layoutManager.layoutLogic().GetLayoutNode()
    if layoutNode:
        if not layoutNode.GetLayoutDescription(CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID):
            layoutNode.AddLayoutDescription(
                CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID, CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML
            )
        else:
            layoutNode.SetLayoutDescription(
                CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID, CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML
            )
        layoutManager.setLayout(CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID)

    if volumeNode is not None:
        vol_id = volumeNode.GetID()
        for tag in ("YellowMinus", "Yellow", "YellowPlus", "Red"):
            sw = layoutManager.sliceWidget(tag)
            if sw and sw.mrmlSliceCompositeNode():
                sw.mrmlSliceCompositeNode().SetBackgroundVolumeID(vol_id)


def setDentalFourUpLayout() -> None:
    """Activate custom dental layout (or Four-Up fallback)."""
    if slicer is None or not getattr(slicer, "app", None):
        return
    try:
        registerAndSetDentalLayout()
    except Exception:
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)


def set3DViewBackgroundColors(
    topColor: Sequence[float] = (1.0, 1.0, 1.0),
    bottomColor: Sequence[float] = (1.0, 1.0, 1.0),
) -> None:
    """Set top and bottom background colors across all 3D viewports."""
    if slicer is None:
        return
    viewNodes = slicer.util.getNodesByClass("vtkMRMLViewNode")
    for viewNode in viewNodes:
        viewNode.SetBackgroundColor(*topColor)
        viewNode.SetBackgroundColor2(*bottomColor)


def setBoxAndTextVisibilityOnThreeDViews(isVisible: bool = False) -> None:
    """Show or hide the 3D bounding box and axis text labels across all 3D views."""
    if slicer is None:
        return
    viewNodes = slicer.util.getNodesByClass("vtkMRMLViewNode")
    for viewNode in viewNodes:
        viewNode.SetBoxVisible(isVisible)
        viewNode.SetAxisLabelsVisible(isVisible)

"""UI, layout, and 3D viewport utilities adapted from SlicerDentalSegmentator."""

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
import ctk
import qt
import slicer


def toRGB(hex_color: str) -> Tuple[float, float, float]:
    """Convert a hex color string (e.g. '#E3DD90') to normalized RGB floats in [0.0, 1.0]."""
    clean_hex = hex_color.lstrip("#")
    if len(clean_hex) != 6:
        raise ValueError(f"Expected 6-character hex color, got: {hex_color}")
    return tuple(int(clean_hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def iconPath(icon_name: str) -> str:
    """Return the absolute path to an icon inside Resources/Icons."""
    base_dir = Path(__file__).resolve().parent.parent
    return str(base_dir / "Resources" / "Icons" / icon_name)


def icon(icon_name: str) -> qt.QIcon:
    """Return a QIcon for the specified icon filename if it exists."""
    path = iconPath(icon_name)
    return qt.QIcon(path) if Path(path).exists() else qt.QIcon()


def createButton(
    text: str,
    callback: Optional[Callable] = None,
    isCheckable: bool = False,
    icon: Optional[qt.QIcon] = None,
    toolTip: str = "",
    parent: Optional[qt.QWidget] = None,
) -> qt.QPushButton:
    """Create a configured QPushButton following SlicerDentalSegmentator conventions."""
    btn = qt.QPushButton(text, parent)
    btn.setCheckable(isCheckable)
    if icon is not None and not icon.isNull():
        btn.setIcon(icon)
    if toolTip:
        btn.setToolTip(toolTip)
    if callback is not None:
        btn.connect("clicked(bool)", callback)
    return btn


def addInCollapsibleLayout(
    childWidget: qt.QWidget,
    parentLayout: qt.QLayout,
    text: str,
    isCollapsed: bool = False,
) -> ctk.ctkCollapsibleButton:
    """Wrap a widget inside a ctkCollapsibleButton and add it to parentLayout."""
    collapsible = ctk.ctkCollapsibleButton()
    collapsible.text = text
    collapsible.collapsed = isCollapsed
    vbox = qt.QVBoxLayout(collapsible)
    vbox.addWidget(childWidget)
    parentLayout.addWidget(collapsible)
    return collapsible


def setConventionalWideScreenView() -> None:
    """Switch 3D Slicer layout to Conventional Widescreen view."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager:
        layoutManager.setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalWidescreenView
        )


def setDentalFourUpLayout() -> None:
    """Switch 3D Slicer layout to Four-Up view (Axial/Red, Panoramic/Green, Cross-Section/Yellow, 3D VR)."""
    layoutManager = slicer.app.layoutManager()
    if layoutManager:
        layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)


def set3DViewBackgroundColors(
    topColor: Sequence[float] = (1.0, 1.0, 1.0),
    bottomColor: Sequence[float] = (1.0, 1.0, 1.0),
) -> None:
    """Set top and bottom background colors across all 3D viewports (DentalSegmentator white preset)."""
    viewNodes = slicer.util.getNodesByClass("vtkMRMLViewNode")
    for viewNode in viewNodes:
        viewNode.SetBackgroundColor(*topColor)
        viewNode.SetBackgroundColor2(*bottomColor)


def setBoxAndTextVisibilityOnThreeDViews(isVisible: bool = False) -> None:
    """Show or hide the 3D bounding box and axis text labels across all 3D views."""
    viewNodes = slicer.util.getNodesByClass("vtkMRMLViewNode")
    for viewNode in viewNodes:
        viewNode.SetBoxVisible(isVisible)
        viewNode.SetAxisLabelsVisible(isVisible)

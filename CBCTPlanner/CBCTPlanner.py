"""
CBCT Dental Planner - 3D Slicer Scripted Loadable Extension Module.

Implements a 5-step clinical workflow (Load -> Arch -> Panoramic -> Cross-sections -> Plan)
backed by a single-source-of-truth MRML ParameterNode and VTKObservationMixin:
  - Step 1 (Load): CBCT volume & optional segmentation selection, Custom Dental Layouts
    (Single Cross-Section vs 3-Slice Cross-Sectional Lightbox [-2, 0, +2 mm]), 3-point occlusal
    plane alignment, and DentalSegmentator 3D / debounced Volume Rendering (VR) controls.
  - Step 2 (Arch): RAS Right-to-Left (+X -> -X) dental arch curve auto-fitting from segmentation
    (with user-edit protection and confirmation) or interactive Markups curve placement when
    unsegmented (never invents control points).
  - Step 3 (Panoramic): Right-handed (det = +1) Curved Planar Reformation (CPR) panoramic
    radiograph with Patient Right on screen left, non-blocking CLI execution with progress/cancel,
    and debounced curve-edit regeneration (low-res preview while dragging, full-res on release).
  - Step 4 (Cross-sections): Interactive draggable cross-section line on the Axial (Red) view
    and vertical/tilted cursor line on the Panoramic (Green) view, Multi-Slice Cross-Sectional
    Lightbox (`YellowMinus` [-2 mm], `Yellow` [0 mm], `YellowPlus` [+2 mm]) with color-coded
    Axial & Panoramic triplet lines, +-30° Cross-Section Tilt control, continuous 1 mm arc-length
    slider (Left/Right 1 mm, PgUp/PgDn 5 mm, Lock-to-Tooth FDI), optional Red-only tick fan,
    and cross-section slab thickness reconstruction.
  - Step 5 (Plan): Clickable 32-tooth Odontogram (click jumps to tooth, Ctrl+Click toggles
    multi-select with incremental single-tooth analysis), honest measurements from segmentation
    or manual Markups rulers (shows "not measured" when unsegmented—never fabricates numbers),
    and HTML/CSV/JSON report export with embedded slice screenshots and HTML escaping.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import ctk  # type: ignore
    import qt  # type: ignore
    import slicer  # type: ignore
    import vtk  # type: ignore
    from slicer.ScriptedLoadableModule import (
        ScriptedLoadableModule,
        ScriptedLoadableModuleLogic,
        ScriptedLoadableModuleTest,
        ScriptedLoadableModuleWidget,
    )
    from slicer.util import VTKObservationMixin  # type: ignore
except ImportError:
    ctk = None
    qt = None
    slicer = None
    vtk = None

    class ScriptedLoadableModule:  # type: ignore[no-redef]
        def __init__(self, parent: Any = None) -> None:
            self.parent = parent

    class ScriptedLoadableModuleWidget:  # type: ignore[no-redef]
        def __init__(self, parent: Any = None) -> None:
            self.parent = parent
            self.layout = None

        def setup(self) -> None:
            pass

    class ScriptedLoadableModuleLogic:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self._parameterNode = None

        def getParameterNode(self) -> Any:
            return self._parameterNode

    class ScriptedLoadableModuleTest:  # type: ignore[no-redef]
        pass

    class VTKObservationMixin:  # type: ignore[no-redef]
        def __init__(self) -> None:
            pass

        def addObserver(self, *args: Any, **kwargs: Any) -> None:
            pass

        def removeObservers(self, *args: Any, **kwargs: Any) -> None:
            pass

from CBCTPlannerLib import (
    DENTAL_SEGMENT_PRESETS,
    CrossSectionNavigator,
    PanoramicCurveLogic,
    VRVisualizationLogic,
    registerAndSetDentalLayout,
    registerAndSetDentalLightboxLayout,
)
from CBCTPlannerLib.core.arch_detector import (
    JAW_TOOTH_ORDER,
    DentalArch,
    fit_dental_arch_from_points,
)
from CBCTPlannerLib.core.labels import (
    FDI_TO_TOOTH_ID,
    LEFT_IAN_ID,
    LEFT_INCISIVE_CANAL_ID,
    LINGUAL_CANAL_ID,
    LOWER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    RIGHT_IAN_ID,
    RIGHT_INCISIVE_CANAL_ID,
    UPPER_JAW_ID,
)
from CBCTPlannerLib.core.measurements import MeasurementEngine
from CBCTPlannerLib.core.volume_loader import CBCTVolume

logger = logging.getLogger("CBCTPlanner")

Q1_TEETH = [18, 17, 16, 15, 14, 13, 12, 11]
Q2_TEETH = [21, 22, 23, 24, 25, 26, 27, 28]
Q3_TEETH = [31, 32, 33, 34, 35, 36, 37, 38]
Q4_TEETH = [48, 47, 46, 45, 44, 43, 42, 41]

NOT_MEASURED = "not measured"


def sort_teeth_anatomically(teeth_list: Sequence[int]) -> List[int]:
    """Sort FDI teeth from Patient Right to Patient Left: Upper (Q1, Q2) then Lower (Q4, Q3)."""
    teeth_set = set(int(t) for t in teeth_list)
    q1 = [t for t in Q1_TEETH if t in teeth_set]
    q2 = [t for t in Q2_TEETH if t in teeth_set]
    q4 = [t for t in Q4_TEETH if t in teeth_set]
    q3 = [t for t in Q3_TEETH if t in teeth_set]
    return q1 + q2 + q4 + q3


def format_metric(val: Optional[float], unit: str = "", decimals: int = 1) -> str:
    """Format an optional numerical metric or return 'not measured' when None."""
    if val is None:
        return NOT_MEASURED
    suffix = f" {unit}" if unit else ""
    return f"{float(val):.{decimals}f}{suffix}"


class CBCTPlanner(ScriptedLoadableModule):
    """3D Slicer ScriptedLoadableModule definition for CBCT Dental Planner."""

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        if self.parent is not None:
            self.parent.title = "CBCT Dental Planner"
            self.parent.categories = ["Dental", "Segmentation", "Planning"]
            self.parent.dependencies = ["VolumeRendering", "Segmentations", "Markups"]
            self.parent.contributors = ["Seyyed Hamid Aghlmand Sarmi"]
            self.parent.helpText = (
                "Dental CBCT Planner: 5-step workflow (Load -> Arch -> Panoramic -> "
                "Cross-sections -> Plan) with DentalSegmentator 3D/VR styling, RAS Right-to-Left "
                "arch curves, Curved Planar Reformation (CPR) panoramic view, 3-slice Cross-Sectional "
                "Lightbox (-2, 0, +2 mm), +-30° Cross-Section Tilt control, draggable Axial and "
                "Panoramic cross-section cursor lines, and honest segmentation/ruler morphometry."
            )
            self.parent.acknowledgementText = (
                "Uses 3D Slicer, DentalSegmentator visualization presets, and PerkLab CPR conventions."
            )


class CBCTPlannerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """5-Step Workflow UI for CBCT Dental Planner with VTKObservationMixin and QSettings."""

    SETTINGS_PREFIX = "CBCTPlanner/"

    def __init__(self, parent: Any = None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: Optional[CBCTPlannerLogic] = None
        self._parameterNode: Any = None
        self._updatingGUIFromParameterNode: bool = False
        self.toothButtons: Dict[int, Any] = {}
        self.selectedTeeth: Set[int] = set()
        self.activeToothFDI: Optional[int] = None
        self.treatmentPlanByFDI: Dict[int, Dict[str, Any]] = {}
        self._vrDebounceTimer: Any = None
        self._panoPreviewTimer: Any = None
        self._shortcuts: List[Any] = []

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = CBCTPlannerLogic()
        self.logic.navigator.set_cursor_changed_callback(self._onInteractiveLineMovedCursor)

        if qt is not None:
            self._vrDebounceTimer = qt.QTimer()
            self._vrDebounceTimer.setSingleShot(True)
            self._vrDebounceTimer.setInterval(150)
            self._vrDebounceTimer.timeout.connect(self._applyDebouncedVr)

            self._panoPreviewTimer = qt.QTimer()
            self._panoPreviewTimer.setSingleShot(True)
            self._panoPreviewTimer.setInterval(200)
            self._panoPreviewTimer.timeout.connect(self._onDebouncedCurvePreviewPano)

        self._buildWorkflowUI()
        self._loadQSettings()
        self._installKeyboardShortcuts()

        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onNodeRemoved)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Disconnect observers, stop timers, and release resources on module exit."""
        self._saveQSettings()
        if self.logic is not None:
            self.logic.cleanup()
        if self._vrDebounceTimer is not None:
            self._vrDebounceTimer.stop()
        if self._panoPreviewTimer is not None:
            self._panoPreviewTimer.stop()
        self.removeObservers()

    # ------------------------------------------------------------------
    # UI Construction (5-Step Progressive Workflow)
    # ------------------------------------------------------------------
    def _buildWorkflowUI(self) -> None:
        if qt is None or ctk is None or slicer is None:
            return

        # ==================== STEP 1: LOAD ====================
        self.step1Box = ctk.ctkCollapsibleButton()
        self.step1Box.text = "Step 1: Load CBCT, Segmentation & 3D / VR View"
        self.step1Box.collapsed = False
        self.layout.addWidget(self.step1Box)
        step1Layout = qt.QFormLayout(self.step1Box)

        self.step1HintLabel = qt.QLabel("Select an input CBCT volume to unlock downstream steps.")
        self.step1HintLabel.setWordWrap(True)
        step1Layout.addRow(self.step1HintLabel)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Select the input dental CBCT scalar volume.")
        step1Layout.addRow("Input CBCT Volume:", self.volumeSelector)

        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.selectNodeUponCreation = False
        self.segmentationSelector.addEnabled = False
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        self.segmentationSelector.setToolTip(
            "Optional: Select a DentalSegmentator or ToothFairy3 segmentation node for automated measurements and 3D styling."
        )
        step1Layout.addRow("Segmentation (Optional):", self.segmentationSelector)

        layoutBtnRow = qt.QHBoxLayout()
        self.btnDentalLightbox = qt.QPushButton("3-Slice Lightbox (-2 / 0 / +2 mm)")
        self.btnDentalLightbox.setToolTip(
            "Switch to OnDemand3D/Romexis-style 3-slice Cross-Sectional Lightbox (CS-, CS0, CS+ on bottom; Pano, Axial, 3D on top)."
        )
        self.btnDentalLayout = qt.QPushButton("Single CS Layout")
        self.btnDentalLayout.setToolTip("Switch to single Cross-Section Dental layout (Wide Pano top; Axial, CS, 3D bottom).")
        self.btnStandardMpr = qt.QPushButton("Standard MPR")
        self.btnStandardMpr.setToolTip("Reset Axial (Red) and Sagittal (Yellow) views without clobbering Panoramic.")
        layoutBtnRow.addWidget(self.btnDentalLightbox)
        layoutBtnRow.addWidget(self.btnDentalLayout)
        layoutBtnRow.addWidget(self.btnStandardMpr)
        step1Layout.addRow("Viewport Layout:", layoutBtnRow)

        # 3-Point Occlusal Plane Tool
        occRow = qt.QHBoxLayout()
        self.btnPlaceOcclusalPts = qt.QPushButton("Place 3 Occlusal Points")
        self.btnPlaceOcclusalPts.setToolTip(
            "Place 3 fiducial points (e.g. right molar, incisors, left molar) to define the true occlusal plane normal."
        )
        self.btnApplyOcclusalPlane = qt.QPushButton("Apply Occlusal Plane")
        self.btnApplyOcclusalPlane.setToolTip(
            "Compute occlusal plane normal from the 3 fiducials and optionally harden alignment transform."
        )
        self.btnResetOcclusalPlane = qt.QPushButton("Reset to World Z")
        self.btnResetOcclusalPlane.setToolTip("Reset occlusal plane normal to default Superior (+Z).")
        occRow.addWidget(self.btnPlaceOcclusalPts)
        occRow.addWidget(self.btnApplyOcclusalPlane)
        occRow.addWidget(self.btnResetOcclusalPlane)
        step1Layout.addRow("Occlusal Plane:", occRow)

        # VR & DentalSegmentator 3D controls
        self.vrPresetCombo = qt.QComboBox()
        self.vrPresetCombo.addItems(list(VRVisualizationLogic.SUPPORTED_VR_PRESETS))
        self.vrPresetCombo.setToolTip("Select Volume Rendering transfer function preset.")
        step1Layout.addRow("VR Preset:", self.vrPresetCombo)

        self.vrShiftSlider = ctk.ctkSliderWidget()
        self.vrShiftSlider.minimum = -1000.0
        self.vrShiftSlider.maximum = 1500.0
        self.vrShiftSlider.singleStep = 25.0
        self.vrShiftSlider.value = 0.0
        self.vrShiftSlider.suffix = " HU"
        self.vrShiftSlider.setToolTip("Shift Volume Rendering scalar opacity and color transfer functions (debounced 150 ms).")
        step1Layout.addRow("VR Threshold Shift:", self.vrShiftSlider)

        vrToggleRow = qt.QHBoxLayout()
        self.showVrCheckBox = qt.QCheckBox("Show 3D Volume Rendering")
        self.showVrCheckBox.setToolTip("Toggle 3D Volume Rendering on the input CBCT volume.")
        self.cropVrCheckBox = qt.QCheckBox("ROI Crop")
        self.cropVrCheckBox.setToolTip("Enable 3D ROI cropping box for Volume Rendering.")
        self.btnApplyDental3D = qt.QPushButton("Style 3D Segmentation")
        self.btnApplyDental3D.setToolTip("Apply SlicerDentalSegmentator colors and 3D opacities to the selected segmentation.")
        vrToggleRow.addWidget(self.showVrCheckBox)
        vrToggleRow.addWidget(self.cropVrCheckBox)
        vrToggleRow.addWidget(self.btnApplyDental3D)
        step1Layout.addRow(vrToggleRow)

        # ==================== STEP 2: ARCH ====================
        self.step2Box = ctk.ctkCollapsibleButton()
        self.step2Box.text = "Step 2: Dental Arch Curve (Right -> Left in RAS)"
        self.step2Box.collapsed = True
        self.layout.addWidget(self.step2Box)
        step2Layout = qt.QFormLayout(self.step2Box)

        self.step2HintLabel = qt.QLabel("Define or auto-fit the dental arch curve from Patient Right (+X) to Patient Left (-X).")
        self.step2HintLabel.setWordWrap(True)
        step2Layout.addRow(self.step2HintLabel)

        self.activeJawCombo = qt.QComboBox()
        self.activeJawCombo.addItems(["Lower", "Upper"])
        self.activeJawCombo.setToolTip("Select which jaw arch curve (Lower Mandible or Upper Maxilla) is active.")
        step2Layout.addRow("Active Jaw Arch:", self.activeJawCombo)

        archBtnRow = qt.QHBoxLayout()
        self.btnAutoFitArch = qt.QPushButton("Auto-Fit Arch from Segmentation")
        self.btnAutoFitArch.setToolTip(
            "Fit arch curve to segmented teeth/jawbone. Asks for confirmation if curve was manually edited."
        )
        self.btnDrawArchManual = qt.QPushButton("Place / Draw Arch Points on Axial")
        self.btnDrawArchManual.setToolTip(
            "Activate Slicer Markups Place Mode to click control points from Patient Right to Patient Left on the Axial slice."
        )
        self.showCurvesCheckBox = qt.QCheckBox("Show Curve Handles")
        self.showCurvesCheckBox.setChecked(True)
        self.showCurvesCheckBox.setToolTip("Show or hide dental arch curve control points in 2D and 3D views.")
        archBtnRow.addWidget(self.btnAutoFitArch)
        archBtnRow.addWidget(self.btnDrawArchManual)
        archBtnRow.addWidget(self.showCurvesCheckBox)
        step2Layout.addRow(archBtnRow)

        # ==================== STEP 3: PANORAMIC ====================
        self.step3Box = ctk.ctkCollapsibleButton()
        self.step3Box.text = "Step 3: Panoramic Visualization (Curved Planar Reformat)"
        self.step3Box.collapsed = True
        self.layout.addWidget(self.step3Box)
        step3Layout = qt.QFormLayout(self.step3Box)

        self.step3HintLabel = qt.QLabel("Requires an active dental arch curve with at least 4 control points.")
        self.step3HintLabel.setWordWrap(True)
        step3Layout.addRow(self.step3HintLabel)

        self.panoThicknessSpin = qt.QDoubleSpinBox()
        self.panoThicknessSpin.minimum = 2.0
        self.panoThicknessSpin.maximum = 45.0
        self.panoThicknessSpin.singleStep = 1.0
        self.panoThicknessSpin.value = 18.0
        self.panoThicknessSpin.suffix = " mm"
        self.panoThicknessSpin.setToolTip("Buccolingual slab thickness for panoramic projection.")
        step3Layout.addRow("Panoramic Slab Thickness:", self.panoThicknessSpin)

        self.panoProjCombo = qt.QComboBox()
        self.panoProjCombo.addItems(["Mean", "MIP", "MinIP"])
        self.panoProjCombo.setToolTip("Intensity projection mode across the straightened slab.")
        step3Layout.addRow("Projection Mode:", self.panoProjCombo)

        self.autoUpdatePanoCheckBox = qt.QCheckBox("Auto-update Panoramic when Curve Changes (Preview / Full-Res)")
        self.autoUpdatePanoCheckBox.setChecked(True)
        self.autoUpdatePanoCheckBox.setToolTip(
            "Automatically regenerate a fast preview panoramic while dragging curve points and full-resolution on release."
        )
        step3Layout.addRow(self.autoUpdatePanoCheckBox)

        panoRunRow = qt.QHBoxLayout()
        self.btnGeneratePanoramic = qt.QPushButton("Generate Panoramic View (Green)")
        self.btnGeneratePanoramic.setToolTip("Run Curved Planar Reformation and display Panoramic radiograph in Green view.")
        self.btnCancelPanoramic = qt.QPushButton("Cancel")
        self.btnCancelPanoramic.setEnabled(False)
        self.btnCancelPanoramic.setToolTip("Cancel an in-progress panoramic resampling job.")
        panoRunRow.addWidget(self.btnGeneratePanoramic)
        panoRunRow.addWidget(self.btnCancelPanoramic)
        step3Layout.addRow(panoRunRow)

        self.panoProgressBar = qt.QProgressBar()
        self.panoProgressBar.setRange(0, 100)
        self.panoProgressBar.setValue(0)
        self.panoProgressBar.setVisible(False)
        step3Layout.addRow(self.panoProgressBar)

        # ==================== STEP 4: CROSS-SECTIONS, LIGHTBOX & TILT ====================
        self.step4Box = ctk.ctkCollapsibleButton()
        self.step4Box.text = "Step 4: Multi-Slice Cross-Sectional Lightbox (-2/0/+2 mm) & +-30° Tilt"
        self.step4Box.collapsed = True
        self.layout.addWidget(self.step4Box)
        step4Layout = qt.QFormLayout(self.step4Box)

        self.step4HintLabel = qt.QLabel(
            "Drag the cyan line in Axial (Red) or Panoramic (Green), adjust the +-30° mesiodistal tilt, "
            "or inspect 3 simultaneous cross-sections (CS- Amber, CS0 Cyan, CS+ Emerald)."
        )
        self.step4HintLabel.setWordWrap(True)
        step4Layout.addRow(self.step4HintLabel)

        arcNavRow = qt.QHBoxLayout()
        self.btnPrevArcMm = qt.QPushButton("-1 mm")
        self.btnPrevArcMm.setToolTip("Step cursor 1 mm toward Patient Right (Left Arrow).")
        self.arcMmSlider = ctk.ctkSliderWidget()
        self.arcMmSlider.minimum = 0.0
        self.arcMmSlider.maximum = 100.0
        self.arcMmSlider.singleStep = 1.0
        self.arcMmSlider.pageStep = 5.0
        self.arcMmSlider.value = 50.0
        self.arcMmSlider.suffix = " mm"
        self.arcMmSlider.setToolTip("Continuous arc-length position along the dental arch curve in mm.")
        self.btnNextArcMm = qt.QPushButton("+1 mm")
        self.btnNextArcMm.setToolTip("Step cursor 1 mm toward Patient Left (Right Arrow).")
        arcNavRow.addWidget(self.btnPrevArcMm)
        arcNavRow.addWidget(self.arcMmSlider)
        arcNavRow.addWidget(self.btnNextArcMm)
        step4Layout.addRow("Cursor Position (Arc mm):", arcNavRow)

        # +-30° Cross-Section Tilt Control
        tiltRow = qt.QHBoxLayout()
        self.csTiltSlider = ctk.ctkSliderWidget()
        self.csTiltSlider.minimum = -30.0
        self.csTiltSlider.maximum = 30.0
        self.csTiltSlider.singleStep = 1.0
        self.csTiltSlider.pageStep = 5.0
        self.csTiltSlider.value = 0.0
        self.csTiltSlider.suffix = "°"
        self.csTiltSlider.setToolTip(
            "Lean the cross-section plane mesiodistally between -30° and +30° around the Buccal-Lingual axis."
        )
        self.btnResetTilt = qt.QPushButton("Reset 0°")
        self.btnResetTilt.setToolTip("Reset cross-section tilt to 0° (perpendicular to occlusal plane).")
        tiltRow.addWidget(self.csTiltSlider)
        tiltRow.addWidget(self.btnResetTilt)
        step4Layout.addRow("Cross-Section Tilt (+-30°):", tiltRow)

        # Multi-Slice Lightbox (-2 mm, 0 mm, +2 mm) Controls
        lightboxRow = qt.QHBoxLayout()
        self.enableLightboxCheckBox = qt.QCheckBox("Enable 3-Slice Lightbox (CS- / CS0 / CS+)")
        self.enableLightboxCheckBox.setChecked(True)
        self.enableLightboxCheckBox.setToolTip(
            "Synchronize YellowMinus (-step), Yellow (0), and YellowPlus (+step) cross-sections and show triplet lines on Axial/Pano."
        )
        self.lightboxStepSpin = qt.QDoubleSpinBox()
        self.lightboxStepSpin.minimum = 0.5
        self.lightboxStepSpin.maximum = 10.0
        self.lightboxStepSpin.singleStep = 0.5
        self.lightboxStepSpin.value = 2.0
        self.lightboxStepSpin.suffix = " mm"
        self.lightboxStepSpin.setToolTip("Arc-length spacing between CS-, CS0, and CS+ slices (default 2.0 mm -> -2, 0, +2 mm).")
        self.btnActivateLightboxLayout = qt.QPushButton("Open 3-Slice Lightbox View")
        self.btnActivateLightboxLayout.setToolTip("Switch viewport layout to show CS-, CS0, and CS+ side by side.")
        lightboxRow.addWidget(self.enableLightboxCheckBox)
        lightboxRow.addWidget(qt.QLabel("Step:"))
        lightboxRow.addWidget(self.lightboxStepSpin)
        lightboxRow.addWidget(self.btnActivateLightboxLayout)
        step4Layout.addRow("Multi-Slice Lightbox:", lightboxRow)

        self.lockFdiCombo = qt.QComboBox()
        self.lockFdiCombo.addItem("Free Cursor (Unlocked)", "")
        for fdi in Q1_TEETH + Q2_TEETH + Q4_TEETH + Q3_TEETH:
            self.lockFdiCombo.addItem(f"Tooth #{fdi}", str(fdi))
        self.lockFdiCombo.setToolTip("Jump and lock the cross-section cursor to a specific FDI tooth site.")
        step4Layout.addRow("Lock to Tooth FDI:", self.lockFdiCombo)

        csParamsRow = qt.QHBoxLayout()
        self.csWidthSpin = qt.QDoubleSpinBox()
        self.csWidthSpin.minimum = 8.0
        self.csWidthSpin.maximum = 50.0
        self.csWidthSpin.value = 24.0
        self.csWidthSpin.suffix = " mm"
        self.csWidthSpin.setToolTip("Buccolingual width of the active cross-section indicator line on the Axial view.")

        self.csSlabSpin = qt.QDoubleSpinBox()
        self.csSlabSpin.minimum = 0.2
        self.csSlabSpin.maximum = 20.0
        self.csSlabSpin.singleStep = 0.5
        self.csSlabSpin.value = 0.5
        self.csSlabSpin.suffix = " mm"
        self.csSlabSpin.setToolTip("Slab thickness (MIP/Mean) of the Cross-Section slice views.")

        self.tickSpacingSpin = qt.QDoubleSpinBox()
        self.tickSpacingSpin.minimum = 1.0
        self.tickSpacingSpin.maximum = 20.0
        self.tickSpacingSpin.singleStep = 1.0
        self.tickSpacingSpin.value = 2.0
        self.tickSpacingSpin.suffix = " mm"
        self.tickSpacingSpin.setToolTip("Spacing in mm between adjacent ticks in the Axial cross-section tick fan.")

        csParamsRow.addWidget(qt.QLabel("Width:"))
        csParamsRow.addWidget(self.csWidthSpin)
        csParamsRow.addWidget(qt.QLabel("Slab:"))
        csParamsRow.addWidget(self.csSlabSpin)
        csParamsRow.addWidget(qt.QLabel("Tick Step:"))
        csParamsRow.addWidget(self.tickSpacingSpin)
        step4Layout.addRow("Slice & Tick Settings:", csParamsRow)

        self.showAxialTicksCheckBox = qt.QCheckBox("Show Cross-Section Tick Fan on Axial (Red) View")
        self.showAxialTicksCheckBox.setChecked(True)
        self.showAxialTicksCheckBox.setToolTip("Display faint perpendicular tick lines along the arch in the Red slice view only.")
        step4Layout.addRow(self.showAxialTicksCheckBox)

        self.csStatusLabel = qt.QLabel("Cursor: 0.0 mm | Tilt: 0° | Lightbox: -2.0 / 0.0 / +2.0 mm")
        step4Layout.addRow(self.csStatusLabel)

        # ==================== STEP 5: PLAN ====================
        self.step5Box = ctk.ctkCollapsibleButton()
        self.step5Box.text = "Step 5: Odontogram, Site Morphometry & Clinical Report"
        self.step5Box.collapsed = True
        self.layout.addWidget(self.step5Box)
        step5Layout = qt.QVBoxLayout(self.step5Box)

        self.step5HintLabel = qt.QLabel(
            "Click any tooth in the Odontogram to jump the Cross-Section view to that site; "
            "Ctrl+Click to toggle inclusion in the treatment plan table."
        )
        self.step5HintLabel.setWordWrap(True)
        step5Layout.addWidget(self.step5HintLabel)

        odontogramWidget = qt.QWidget()
        odontogramGrid = qt.QGridLayout(odontogramWidget)
        odontogramGrid.setContentsMargins(2, 2, 2, 2)
        odontogramGrid.setSpacing(2)

        q1Label = qt.QLabel("Q1 (Upper Right)")
        q2Label = qt.QLabel("Q2 (Upper Left)")
        q2Label.setAlignment(qt.Qt.AlignRight)
        odontogramGrid.addWidget(q1Label, 0, 0, 1, 8)
        odontogramGrid.addWidget(q2Label, 0, 9, 1, 8)

        for col, fdi in enumerate(Q1_TEETH):
            btn = self._createOdontogramButton(fdi)
            odontogramGrid.addWidget(btn, 1, col)
        sepTop = qt.QFrame()
        sepTop.setFrameShape(qt.QFrame.VLine)
        odontogramGrid.addWidget(sepTop, 1, 8)
        for idx, fdi in enumerate(Q2_TEETH):
            btn = self._createOdontogramButton(fdi)
            odontogramGrid.addWidget(btn, 1, 9 + idx)

        for col, fdi in enumerate(Q4_TEETH):
            btn = self._createOdontogramButton(fdi)
            odontogramGrid.addWidget(btn, 2, col)
        sepBot = qt.QFrame()
        sepBot.setFrameShape(qt.QFrame.VLine)
        odontogramGrid.addWidget(sepBot, 2, 8)
        for idx, fdi in enumerate(Q3_TEETH):
            btn = self._createOdontogramButton(fdi)
            odontogramGrid.addWidget(btn, 2, 9 + idx)

        q4Label = qt.QLabel("Q4 (Lower Right)")
        q3Label = qt.QLabel("Q3 (Lower Left)")
        q3Label.setAlignment(qt.Qt.AlignRight)
        odontogramGrid.addWidget(q4Label, 3, 0, 1, 8)
        odontogramGrid.addWidget(q3Label, 3, 9, 1, 8)
        step5Layout.addWidget(odontogramWidget)

        rulerRow = qt.QHBoxLayout()
        self.btnCreateManualRulers = qt.QPushButton("Create / Refresh Manual Rulers for Active Tooth")
        self.btnCreateManualRulers.setToolTip(
            "Create draggable Ridge Height, Crest Width, and Nerve Distance ruler lines on the current Cross-Section slice."
        )
        self.btnRecomputeSelected = qt.QPushButton("Refresh Measurements")
        self.btnRecomputeSelected.setToolTip("Re-read segmentation or manual ruler lengths for planned teeth.")
        self.btnClearPlan = qt.QPushButton("Clear Plan")
        self.btnClearPlan.setToolTip("Clear all selected teeth from the treatment plan table.")
        rulerRow.addWidget(self.btnCreateManualRulers)
        rulerRow.addWidget(self.btnRecomputeSelected)
        rulerRow.addWidget(self.btnClearPlan)
        step5Layout.addLayout(rulerRow)

        self.planTable = qt.QTableWidget()
        self.planTable.setColumnCount(9)
        self.planTable.setHorizontalHeaderLabels(
            [
                "FDI Site",
                "Jaw",
                "Ridge Height",
                "Crest Width",
                "Width @ 3mm",
                "Width @ 5mm",
                "Nerve Distance",
                "Misch (HU)",
                "Suggested Implant",
            ]
        )
        self.planTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.planTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.planTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.planTable.setMinimumHeight(150)
        step5Layout.addWidget(self.planTable)

        exportRow = qt.QHBoxLayout()
        self.btnGenerateReport = qt.QPushButton("Generate Printable HTML Report (with Screenshots)")
        self.btnGenerateReport.setToolTip("Export clinical report with embedded Panoramic, Axial, and Cross-Section Lightbox screenshots.")
        self.btnExportPlan = qt.QPushButton("Export CSV / JSON")
        self.btnExportPlan.setToolTip("Export treatment plan table to CSV (with flattened RAS coordinates) and JSON.")
        exportRow.addWidget(self.btnGenerateReport)
        exportRow.addWidget(self.btnExportPlan)
        step5Layout.addLayout(exportRow)

        self.layout.addStretch()

        # Connect signals consistently
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeNodeChanged)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.btnDentalLightbox.clicked.connect(self.onSetDentalLightboxLayout)
        self.btnDentalLayout.clicked.connect(self.onSetDentalLayout)
        self.btnStandardMpr.clicked.connect(self.onSetStandardMpr)
        self.btnPlaceOcclusalPts.clicked.connect(self.onPlaceOcclusalPoints)
        self.btnApplyOcclusalPlane.clicked.connect(self.onApplyOcclusalPlane)
        self.btnResetOcclusalPlane.clicked.connect(self.onResetOcclusalPlane)

        self.vrPresetCombo.currentTextChanged.connect(self.onVrControlChanged)
        self.vrShiftSlider.valueChanged.connect(self.onVrSliderChanged)
        self.showVrCheckBox.toggled.connect(self.onVrControlChanged)
        self.cropVrCheckBox.toggled.connect(self.onVrControlChanged)
        self.btnApplyDental3D.clicked.connect(self.onApplyDental3DStyling)

        self.activeJawCombo.currentTextChanged.connect(self.onActiveJawChanged)
        self.btnAutoFitArch.clicked.connect(self.onAutoFitArchClicked)
        self.btnDrawArchManual.clicked.connect(self.onDrawArchManualClicked)
        self.showCurvesCheckBox.toggled.connect(self.onToggleCurvesVisibility)

        self.panoThicknessSpin.valueChanged.connect(self.onPanoParamChanged)
        self.panoProjCombo.currentTextChanged.connect(self.onPanoParamChanged)
        self.btnGeneratePanoramic.clicked.connect(self.onGeneratePanoramicClicked)
        self.btnCancelPanoramic.clicked.connect(self.onCancelPanoramicClicked)

        self.btnPrevArcMm.clicked.connect(lambda: self.nudgeCursorMm(-1.0))
        self.btnNextArcMm.clicked.connect(lambda: self.nudgeCursorMm(1.0))
        self.arcMmSlider.valueChanged.connect(self.onArcSliderChanged)
        self.csTiltSlider.valueChanged.connect(self.onTiltSliderChanged)
        self.btnResetTilt.clicked.connect(lambda: self.csTiltSlider.setValue(0.0))
        self.enableLightboxCheckBox.toggled.connect(self.onCsDisplayParamChanged)
        self.lightboxStepSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.btnActivateLightboxLayout.clicked.connect(self.onSetDentalLightboxLayout)
        self.lockFdiCombo.currentIndexChanged.connect(self.onLockFdiChanged)
        self.csWidthSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.csSlabSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.tickSpacingSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.showAxialTicksCheckBox.toggled.connect(self.onCsDisplayParamChanged)

        self.btnCreateManualRulers.clicked.connect(self.onCreateManualRulersClicked)
        self.btnRecomputeSelected.clicked.connect(self.onRefreshAllSelectedTeeth)
        self.btnClearPlan.clicked.connect(self.onClearPlanClicked)
        self.planTable.cellClicked.connect(self.onTableRowClicked)
        self.btnGenerateReport.clicked.connect(self.onGeneratePrintableReport)
        self.btnExportPlan.clicked.connect(self.onExportTreatmentPlan)

    def _createOdontogramButton(self, fdi: int) -> Any:
        btn = qt.QToolButton()
        btn.setText(str(fdi))
        btn.setCheckable(True)
        btn.setMinimumWidth(28)
        btn.setToolTip(
            f"Tooth #{fdi}: Click to jump Cross-Section to #{fdi}; Ctrl+Click to toggle in treatment plan."
        )
        btn.clicked.connect(lambda checked, t=fdi: self.onOdontogramToothClicked(t, checked))
        self.toothButtons[fdi] = btn
        return btn

    def _installKeyboardShortcuts(self) -> None:
        if qt is None or self.parent is None:
            return
        for key_seq, step in (
            (qt.Qt.Key_Left, -1.0),
            (qt.Qt.Key_Right, 1.0),
            (qt.Qt.Key_PageDown, -5.0),
            (qt.Qt.Key_PageUp, 5.0),
        ):
            sc = qt.QShortcut(qt.QKeySequence(key_seq), self.parent)
            sc.activated.connect(lambda d=step: self.nudgeCursorMm(d))
            self._shortcuts.append(sc)

    def _loadQSettings(self) -> None:
        if qt is None:
            return
        settings = qt.QSettings()
        pano_th = settings.value(self.SETTINGS_PREFIX + "panoThicknessMm", 18.0)
        cs_w = settings.value(self.SETTINGS_PREFIX + "csWidthMm", 24.0)
        cs_slab = settings.value(self.SETTINGS_PREFIX + "csSlabMm", 0.5)
        tick_sp = settings.value(self.SETTINGS_PREFIX + "tickSpacingMm", 2.0)
        lb_step = settings.value(self.SETTINGS_PREFIX + "lightboxStepMm", 2.0)
        cs_tilt = settings.value(self.SETTINGS_PREFIX + "csTiltDeg", 0.0)
        vr_preset = settings.value(self.SETTINGS_PREFIX + "vrPreset", "CT-Dental-CBCT")
        try:
            self.panoThicknessSpin.setValue(float(pano_th))
            self.csWidthSpin.setValue(float(cs_w))
            self.csSlabSpin.setValue(float(cs_slab))
            self.tickSpacingSpin.setValue(float(tick_sp))
            self.lightboxStepSpin.setValue(float(lb_step))
            self.csTiltSlider.setValue(float(cs_tilt))
            if vr_preset:
                self.vrPresetCombo.setCurrentText(str(vr_preset))
        except Exception:
            pass

    def _saveQSettings(self) -> None:
        if qt is None:
            return
        try:
            settings = qt.QSettings()
            settings.setValue(self.SETTINGS_PREFIX + "panoThicknessMm", self.panoThicknessSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "csWidthMm", self.csWidthSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "csSlabMm", self.csSlabSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "tickSpacingMm", self.tickSpacingSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "lightboxStepMm", self.lightboxStepSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "csTiltDeg", self.csTiltSlider.value)
            settings.setValue(self.SETTINGS_PREFIX + "vrPreset", self.vrPresetCombo.currentText)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ParameterNode & Scene Lifecycle
    # ------------------------------------------------------------------
    def initializeParameterNode(self) -> None:
        if self.logic is None:
            return
        self.setParameterNode(self.logic.getParameterNode())
        if slicer is not None and self._parameterNode is not None:
            if not self._parameterNode.GetNodeReference("InputVolume"):
                firstVol = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                if firstVol:
                    self._parameterNode.SetNodeReferenceID("InputVolume", firstVol.GetID())

    def setParameterNode(self, inputParameterNode: Any) -> None:
        if self._parameterNode is not None and vtk is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None and vtk is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self.updateGUIFromParameterNode()

    def onSceneStartClose(self, _caller: Any, _event: Any) -> None:
        self.selectedTeeth.clear()
        self.treatmentPlanByFDI.clear()
        self.setParameterNode(None)

    def onSceneEndClose(self, _caller: Any, _event: Any) -> None:
        if self.parent and self.parent.isEntered:
            self.initializeParameterNode()

    def onNodeRemoved(self, _caller: Any, _event: Any) -> None:
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, _caller: Any = None, _event: Any = None) -> None:
        if self._parameterNode is None or self._updatingGUIFromParameterNode or qt is None:
            return
        self._updatingGUIFromParameterNode = True
        try:
            volNode = self._parameterNode.GetNodeReference("InputVolume")
            segNode = self._parameterNode.GetNodeReference("Segmentation")
            self.volumeSelector.setCurrentNode(volNode)
            self.segmentationSelector.setCurrentNode(segNode)

            has_vol = volNode is not None
            jaw = self._parameterNode.GetParameter("ActiveJaw") or "Lower"
            self.activeJawCombo.setCurrentText(jaw)

            curveNode = self.logic.get_curve_node_for_jaw(volNode, jaw) if (has_vol and self.logic) else None
            has_curve = curveNode is not None and curveNode.GetNumberOfControlPoints() >= 4

            self.step2Box.setEnabled(has_vol)
            self.step3Box.setEnabled(has_vol and has_curve)
            self.step4Box.setEnabled(has_vol and has_curve)
            self.step5Box.setEnabled(has_vol and has_curve)

            if not has_vol:
                self.step1HintLabel.setText("Status: Waiting for an input CBCT volume.")
            else:
                seg_status = f"Segmentation: {segNode.GetName()}" if segNode else "Unsegmented (Manual Ruler mode)"
                self.step1HintLabel.setText(f"Loaded: {volNode.GetName()} | {seg_status}")
                if self.step2Box.collapsed and not has_curve:
                    self.step2Box.collapsed = False

            if has_curve and curveNode is not None:
                user_edited = curveNode.GetAttribute("CBCTPlanner.UserEdited") == "1"
                total_len = float(curveNode.GetCurveLengthWorld()) if hasattr(curveNode, "GetCurveLengthWorld") else 100.0
                edit_tag = "User-edited (locked against silent overwrite)" if user_edited else "Auto-fitted"
                self.step2HintLabel.setText(
                    f"{jaw} Arch Curve: {curveNode.GetNumberOfControlPoints()} points, {total_len:.1f} mm ({edit_tag})."
                )
                self.arcMmSlider.maximum = max(1.0, round(total_len, 1))
                cursor_mm = float(self._parameterNode.GetParameter("CursorArcMm") or 0.0)
                self.arcMmSlider.value = min(self.arcMmSlider.maximum, max(0.0, cursor_mm))
            else:
                self.step2HintLabel.setText(
                    f"No {jaw} arch curve yet. Click 'Auto-Fit Arch from Segmentation' or 'Place / Draw Arch Points on Axial'."
                )
        finally:
            self._updatingGUIFromParameterNode = False

    # ------------------------------------------------------------------
    # Step 1 Callbacks: Volume, Layout, Occlusal Plane & VR
    # ------------------------------------------------------------------
    def onVolumeNodeChanged(self, node: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetNodeReferenceID("InputVolume", node.GetID() if node else "")
            self.updateGUIFromParameterNode()

    def onSegmentationNodeChanged(self, node: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetNodeReferenceID("Segmentation", node.GetID() if node else "")
            self.updateGUIFromParameterNode()
            if self.selectedTeeth:
                self.onRefreshAllSelectedTeeth()

    def onSetDentalLightboxLayout(self) -> None:
        volNode = self.volumeSelector.currentNode()
        registerAndSetDentalLightboxLayout(volumeNode=volNode)
        self.enableLightboxCheckBox.setChecked(True)
        if volNode and self.logic:
            self.logic.standardize_to_lpi(volNode)
            self._syncCursorToViews()

    def onSetDentalLayout(self) -> None:
        registerAndSetDentalLayout()
        volNode = self.volumeSelector.currentNode()
        if volNode and self.logic:
            self.logic.standardize_to_lpi(volNode)
            self._syncCursorToViews()

    def onSetStandardMpr(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        self.logic.standardize_to_lpi(volNode)

    def onPlaceOcclusalPoints(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        self.logic.start_place_occlusal_points(volNode)

    def onApplyOcclusalPlane(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        normal = self.logic.apply_occlusal_plane_from_points(volNode, ask_harden_transform=True)
        if normal is not None:
            self.step1HintLabel.setText(
                f"Applied Occlusal Normal: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            )
            self._syncCursorToViews()

    def onResetOcclusalPlane(self) -> None:
        if self.logic:
            self.logic.set_occlusal_normal((0.0, 0.0, 1.0))
            self._syncCursorToViews()

    def onVrSliderChanged(self, _val: float) -> None:
        if self._vrDebounceTimer is not None:
            self._vrDebounceTimer.start()
        else:
            self._applyDebouncedVr()

    def onVrControlChanged(self, *_args: Any) -> None:
        self._applyDebouncedVr()

    def _applyDebouncedVr(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        self.logic.vr.set_volume_rendering(
            volumeNode=volNode,
            visible=self.showVrCheckBox.isChecked(),
            preset_name=self.vrPresetCombo.currentText,
            shift_hu=self.vrShiftSlider.value,
            crop_enabled=self.cropVrCheckBox.isChecked(),
        )

    def onApplyDental3DStyling(self) -> None:
        segNode = self.segmentationSelector.currentNode()
        if not segNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Please select a Segmentation node first.")
            return
        self.logic.vr.apply_dental_segmentator_styling(segNode)

    # ------------------------------------------------------------------
    # Step 2 Callbacks: Dental Arch Curve (Auto-Fit vs Interactive Place)
    # ------------------------------------------------------------------
    def onActiveJawChanged(self, jaw: str) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("ActiveJaw", jaw)
        self.updateGUIFromParameterNode()
        self._syncCursorToViews()

    def onAutoFitArchClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic:
            return
        jaw = self.activeJawCombo.currentText
        curveNode = self.logic.auto_fit_arch_for_jaw(
            volumeNode=volNode,
            segmentationNode=segNode,
            jaw=jaw,
            visible=self.showCurvesCheckBox.isChecked(),
            confirm_overwrite=True,
        )
        if curveNode is not None and curveNode.GetNumberOfControlPoints() >= 4:
            self._observeCurveEdits(curveNode)
            self.step3Box.collapsed = False
            self.step4Box.collapsed = False
            self.step5Box.collapsed = False
            self.updateGUIFromParameterNode()
            self._syncCursorToViews()

    def onDrawArchManualClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        jaw = self.activeJawCombo.currentText
        curveNode = self.logic.start_interactive_arch_placement(volNode, jaw=jaw)
        if curveNode is not None:
            self._observeCurveEdits(curveNode)
            self.updateGUIFromParameterNode()

    def _observeCurveEdits(self, curveNode: Any) -> None:
        if vtk is None or curveNode is None:
            return
        self.addObserver(curveNode, vtk.vtkCommand.PointModifiedEvent, self._onCurvePointDragged)
        self.addObserver(curveNode, vtk.vtkCommand.EndInteractionEvent, self._onCurveInteractionEnded)

    def _onCurvePointDragged(self, caller: Any, _event: Any) -> None:
        if self.logic and not self.logic._is_programmatic_curve_update:
            caller.SetAttribute("CBCTPlanner.UserEdited", "1")
            if self.autoUpdatePanoCheckBox.isChecked() and self._panoPreviewTimer is not None:
                self._panoPreviewTimer.start()
            self._syncCursorToViews()

    def _onCurveInteractionEnded(self, caller: Any, _event: Any) -> None:
        if self.logic and not self.logic._is_programmatic_curve_update:
            caller.SetAttribute("CBCTPlanner.UserEdited", "1")
            self.updateGUIFromParameterNode()
            if self.autoUpdatePanoCheckBox.isChecked():
                self._runPanoramicGeneration(preview=False)

    def onToggleCurvesVisibility(self, checked: bool) -> None:
        volNode = self.volumeSelector.currentNode()
        if volNode and self.logic:
            self.logic.set_curves_visibility(volNode, visible=checked)

    # ------------------------------------------------------------------
    # Step 3 Callbacks: Panoramic CPR Generation (Debounced & Non-Blocking)
    # ------------------------------------------------------------------
    def onPanoParamChanged(self, *_args: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("PanoThicknessMm", str(self.panoThicknessSpin.value))
            self._parameterNode.SetParameter("PanoProjectionMode", self.panoProjCombo.currentText)

    def _onDebouncedCurvePreviewPano(self) -> None:
        if self.autoUpdatePanoCheckBox.isChecked():
            self._runPanoramicGeneration(preview=True)

    def onGeneratePanoramicClicked(self) -> None:
        self._runPanoramicGeneration(preview=False)

    def onCancelPanoramicClicked(self) -> None:
        if self.logic:
            self.logic.panoramic.cancel_active_panoramic_cli()
        self.panoProgressBar.setVisible(False)
        self.btnCancelPanoramic.setEnabled(False)

    def _runPanoramicGeneration(self, preview: bool = False) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        jaw = self.activeJawCombo.currentText
        curveNode = self.logic.get_curve_node_for_jaw(volNode, jaw)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 4:
            if not preview and slicer:
                slicer.util.warningDisplay(f"Please create or fit a {jaw} dental arch curve (>= 4 points) first.")
            return

        spacing = 1.5 if preview else 0.5
        self.panoProgressBar.setVisible(True)
        self.panoProgressBar.setValue(30 if preview else 15)
        self.btnCancelPanoramic.setEnabled(True)

        def _on_done(panoNode: Any, _straightNode: Any) -> None:
            self.panoProgressBar.setValue(100)
            self.panoProgressBar.setVisible(False)
            self.btnCancelPanoramic.setEnabled(False)
            if self._parameterNode is not None and panoNode is not None:
                self._parameterNode.SetNodeReferenceID("PanoramicVolume", panoNode.GetID())
            self._syncCursorToViews()

        try:
            self.logic.panoramic.generate_panoramic_volume(
                volumeNode=volNode,
                curveNode=curveNode,
                slab_thickness_mm=self.panoThicknessSpin.value,
                spacing_mm=spacing,
                projection_mode=self.panoProjCombo.currentText,
                show_in_green_view=True,
                occlusal_normal=self.logic.get_occlusal_normal(),
                async_cli=True,
                on_complete=_on_done,
            )
        except Exception as exc:
            self.panoProgressBar.setVisible(False)
            self.btnCancelPanoramic.setEnabled(False)
            logger.error("Panoramic generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4 Callbacks: Cross-Section Cursor, Lightbox (-2/0/+2) & +-30° Tilt
    # ------------------------------------------------------------------
    def nudgeCursorMm(self, delta_mm: float) -> None:
        new_val = max(
            self.arcMmSlider.minimum,
            min(self.arcMmSlider.maximum, float(self.arcMmSlider.value) + float(delta_mm)),
        )
        self.arcMmSlider.setValue(new_val)

    def _onInteractiveLineMovedCursor(self, new_arc_mm: float) -> None:
        """Called when the user drags the Axial or Panoramic line in a slice view."""
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CursorArcMm", f"{new_arc_mm:.2f}")
        self.arcMmSlider.blockSignals(True)
        self.arcMmSlider.setValue(new_arc_mm)
        self.arcMmSlider.blockSignals(False)
        self._syncCursorToViews(arc_mm=new_arc_mm)

    def onArcSliderChanged(self, arc_mm: float) -> None:
        if self._updatingGUIFromParameterNode:
            return
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CursorArcMm", f"{arc_mm:.2f}")
        self._syncCursorToViews(arc_mm=arc_mm)

    def onTiltSliderChanged(self, tilt_deg: float) -> None:
        if self._updatingGUIFromParameterNode:
            return
        clamped = CrossSectionNavigator.clamp_tilt_deg(tilt_deg)
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CrossSectionTiltDeg", f"{clamped:.1f}")
        self._syncCursorToViews()

    def onLockFdiChanged(self, _idx: int) -> None:
        fdi_str = self.lockFdiCombo.currentData
        if not fdi_str:
            return
        fdi = int(fdi_str)
        volNode = self.volumeSelector.currentNode()
        if volNode and self.logic:
            self.activeToothFDI = fdi
            target_jaw = "Lower" if fdi >= 31 else "Upper"
            if self.activeJawCombo.currentText != target_jaw:
                self.activeJawCombo.setCurrentText(target_jaw)
            arc_mm = self.logic.jump_to_tooth_site(
                fdi,
                volNode,
                width_mm=self.csWidthSpin.value,
                tick_spacing_mm=self.tickSpacingSpin.value,
                show_all_ticks=self.showAxialTicksCheckBox.isChecked(),
                slab_thickness_mm=self.csSlabSpin.value,
                tilt_deg=self.csTiltSlider.value,
                lightbox_enabled=self.enableLightboxCheckBox.isChecked(),
                lightbox_step_mm=self.lightboxStepSpin.value,
            )
            if arc_mm is not None:
                self.arcMmSlider.blockSignals(True)
                self.arcMmSlider.setValue(arc_mm)
                self.arcMmSlider.blockSignals(False)

    def onCsDisplayParamChanged(self, *_args: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter(
                "LightboxEnabled", "true" if self.enableLightboxCheckBox.isChecked() else "false"
            )
            self._parameterNode.SetParameter("LightboxStepMm", f"{self.lightboxStepSpin.value:.1f}")
        self._syncCursorToViews(arc_mm=self.arcMmSlider.value)

    def _syncCursorToViews(self, arc_mm: Optional[float] = None) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        jaw = self.activeJawCombo.currentText
        curveNode = self.logic.get_curve_node_for_jaw(volNode, jaw)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 4:
            return
        target_arc = float(self.arcMmSlider.value if arc_mm is None else arc_mm)
        tilt_deg = float(self.csTiltSlider.value)
        lb_enabled = bool(self.enableLightboxCheckBox.isChecked())
        lb_step = float(self.lightboxStepSpin.value)
        panoNode = self._parameterNode.GetNodeReference("PanoramicVolume") if self._parameterNode else None
        site_lbl = f"#{self.activeToothFDI}" if self.activeToothFDI else None
        _, _, _, _, s_clamped, total_len = self.logic.navigator.update_cursor_at_arc_mm(
            volumeNode=volNode,
            curveNode=curveNode,
            arc_mm=target_arc,
            width_mm=self.csWidthSpin.value,
            tick_spacing_mm=self.tickSpacingSpin.value,
            show_all_ticks=self.showAxialTicksCheckBox.isChecked(),
            slab_thickness_mm=self.csSlabSpin.value,
            tilt_deg=tilt_deg,
            lightbox_enabled=lb_enabled,
            lightbox_step_mm=lb_step,
            occlusal_normal=self.logic.get_occlusal_normal(),
            site_label=site_lbl,
            panoNode=panoNode,
        )
        s_m, s_0, s_p = CrossSectionNavigator.compute_lightbox_offsets(
            s_clamped, total_len, step_mm=lb_step
        )
        lb_text = f" | Lightbox: [{s_m:.1f}, {s_0:.1f}, {s_p:.1f}] mm" if lb_enabled else ""
        self.csStatusLabel.setText(
            f"Cursor: {s_clamped:.1f} / {total_len:.1f} mm ({jaw}) | Tilt: {tilt_deg:+.0f}°{lb_text}"
            + (f" | Site {site_lbl}" if site_lbl else "")
        )

    # ------------------------------------------------------------------
    # Step 5 Callbacks: Odontogram, Manual Rulers, Plan Table & Export
    # ------------------------------------------------------------------
    def onOdontogramToothClicked(self, fdi: int, is_checked: bool) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic:
            return

        modifiers = qt.QApplication.keyboardModifiers() if qt else 0
        ctrl_held = bool(modifiers & qt.Qt.ControlModifier) if qt else False

        self.activeToothFDI = fdi
        target_jaw = "Lower" if fdi >= 31 else "Upper"
        if self.activeJawCombo.currentText != target_jaw:
            self.activeJawCombo.setCurrentText(target_jaw)

        arc_mm = self.logic.jump_to_tooth_site(
            fdi,
            volNode,
            width_mm=self.csWidthSpin.value,
            tick_spacing_mm=self.tickSpacingSpin.value,
            show_all_ticks=self.showAxialTicksCheckBox.isChecked(),
            slab_thickness_mm=self.csSlabSpin.value,
            tilt_deg=self.csTiltSlider.value,
            lightbox_enabled=self.enableLightboxCheckBox.isChecked(),
            lightbox_step_mm=self.lightboxStepSpin.value,
        )
        if arc_mm is not None:
            self.arcMmSlider.blockSignals(True)
            self.arcMmSlider.setValue(arc_mm)
            self.arcMmSlider.blockSignals(False)

        if ctrl_held:
            if is_checked:
                self.selectedTeeth.add(fdi)
                self.treatmentPlanByFDI[fdi] = self.logic.analyze_single_tooth(
                    fdi, volNode, segmentationNode=segNode
                )
            else:
                self.selectedTeeth.discard(fdi)
                self.treatmentPlanByFDI.pop(fdi, None)
        else:
            self.selectedTeeth.add(fdi)
            btn = self.toothButtons.get(fdi)
            if btn:
                btn.blockSignals(True)
                btn.setChecked(True)
                btn.blockSignals(False)
            if fdi not in self.treatmentPlanByFDI:
                self.treatmentPlanByFDI[fdi] = self.logic.analyze_single_tooth(
                    fdi, volNode, segmentationNode=segNode
                )

        self._refreshPlanTableFromCache()

    def onCreateManualRulersClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        fdi = self.activeToothFDI or (next(iter(self.selectedTeeth)) if self.selectedTeeth else 46)
        self.logic.create_or_focus_manual_rulers_for_tooth(fdi, volNode)
        self.treatmentPlanByFDI[fdi] = self.logic.analyze_single_tooth(
            fdi, volNode, segmentationNode=self.segmentationSelector.currentNode()
        )
        self.selectedTeeth.add(fdi)
        self._refreshPlanTableFromCache()

    def onRefreshAllSelectedTeeth(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic or not self.selectedTeeth:
            return
        ordered = sort_teeth_anatomically(list(self.selectedTeeth))
        rows = self.logic.analyze_selected_teeth(ordered, volNode, segmentationNode=segNode)
        self.treatmentPlanByFDI = {r["fdi"]: r for r in rows}
        self._refreshPlanTableFromCache()

    def onClearPlanClicked(self) -> None:
        self.selectedTeeth.clear()
        self.treatmentPlanByFDI.clear()
        for btn in self.toothButtons.values():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self._refreshPlanTableFromCache()

    def _getOrderedPlanRows(self) -> List[Dict[str, Any]]:
        ordered_fdis = sort_teeth_anatomically(list(self.selectedTeeth))
        return [self.treatmentPlanByFDI[fdi] for fdi in ordered_fdis if fdi in self.treatmentPlanByFDI]

    def _refreshPlanTableFromCache(self) -> None:
        rows = self._getOrderedPlanRows()
        self.planTable.setRowCount(len(rows))
        for row_idx, item in enumerate(rows):
            misch_str = (
                f"{item['misch']} ({item['hu']:.0f} HU)"
                if (item.get("misch") and item.get("hu") is not None)
                else NOT_MEASURED
            )
            imp_str = (
                f"D{item['rec_diameter']:.1f} x L{item['rec_length']:.1f} mm"
                if (item.get("rec_diameter") is not None and item.get("rec_length") is not None)
                else NOT_MEASURED
            )
            cells = [
                str(item["fdi"]),
                str(item["jaw"]).capitalize(),
                format_metric(item.get("height_mm"), "mm"),
                format_metric(item.get("w_crest_mm"), "mm"),
                format_metric(item.get("w_3mm"), "mm"),
                format_metric(item.get("w_5mm"), "mm"),
                format_metric(item.get("nerve_clearance_mm"), "mm"),
                misch_str,
                imp_str,
            ]
            for col_idx, text in enumerate(cells):
                cell = qt.QTableWidgetItem(text)
                cell.setTextAlignment(qt.Qt.AlignCenter)
                self.planTable.setItem(row_idx, col_idx, cell)

    def onTableRowClicked(self, row: int, _col: int) -> None:
        rows = self._getOrderedPlanRows()
        if 0 <= row < len(rows):
            fdi = int(rows[row]["fdi"])
            self.activeToothFDI = fdi
            volNode = self.volumeSelector.currentNode()
            if volNode and self.logic:
                self.logic.jump_to_tooth_site(
                    fdi,
                    volNode,
                    width_mm=self.csWidthSpin.value,
                    tick_spacing_mm=self.tickSpacingSpin.value,
                    show_all_ticks=self.showAxialTicksCheckBox.isChecked(),
                    slab_thickness_mm=self.csSlabSpin.value,
                    tilt_deg=self.csTiltSlider.value,
                    lightbox_enabled=self.enableLightboxCheckBox.isChecked(),
                    lightbox_step_mm=self.lightboxStepSpin.value,
                )

    def onGeneratePrintableReport(self) -> None:
        rows = self._getOrderedPlanRows()
        if not rows or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select at least one tooth site in the Odontogram first.")
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(self.parent, "Select Report Output Directory")
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        report_path = self.logic.generate_printable_report(
            rows, self.volumeSelector.currentNode(), out_dir, capture_screenshots=True
        )
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(report_path)))

    def onExportTreatmentPlan(self) -> None:
        rows = self._getOrderedPlanRows()
        if not rows or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select at least one tooth site in the Odontogram first.")
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(self.parent, "Select Export Directory")
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        out_json, out_csv = self.logic.export_treatment_plan_files(rows, out_dir)
        if slicer:
            slicer.util.infoDisplay(f"Exported treatment plan:\n{out_json}\n{out_csv}")


class CBCTPlannerLogic(ScriptedLoadableModuleLogic):
    """Core logic orchestrating ParameterNode state, VR, Arch Curves, Panoramic CPR, Lightbox & Tilt, and Morphometry."""

    def __init__(self) -> None:
        super().__init__()
        self.vr = VRVisualizationLogic()
        self.panoramic = PanoramicCurveLogic()
        self.navigator = CrossSectionNavigator(panoramic_logic=self.panoramic)
        self._is_programmatic_curve_update: bool = False
        self._occlusal_normal: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def cleanup(self) -> None:
        self.panoramic.cancel_active_panoramic_cli()
        self.navigator.remove_line_observers()

    def getParameterNode(self) -> Any:
        node = super().getParameterNode()
        if node is not None:
            self._setDefaultParameters(node)
        return node

    @staticmethod
    def _setDefaultParameters(parameterNode: Any) -> None:
        defaults = {
            "ActiveJaw": "Lower",
            "CursorArcMm": "0.0",
            "CrossSectionWidthMm": "24.0",
            "SlabThicknessMm": "0.5",
            "TickSpacingMm": "2.0",
            "CrossSectionTiltDeg": "0.0",
            "LightboxEnabled": "true",
            "LightboxStepMm": "2.0",
            "PanoThicknessMm": "18.0",
            "PanoProjectionMode": "Mean",
            "ShowAxialTicks": "true",
            "OcclusalNormalX": "0.0",
            "OcclusalNormalY": "0.0",
            "OcclusalNormalZ": "1.0",
        }
        for key, val in defaults.items():
            if not parameterNode.GetParameter(key):
                parameterNode.SetParameter(key, val)

    # ------------------------------------------------------------------
    # Occlusal Plane Normal & Standard MPR (Preserving Panoramic View)
    # ------------------------------------------------------------------
    def get_occlusal_normal(self) -> np.ndarray:
        pNode = self.getParameterNode() if slicer is not None else None
        if pNode is not None:
            try:
                nx = float(pNode.GetParameter("OcclusalNormalX") or 0.0)
                ny = float(pNode.GetParameter("OcclusalNormalY") or 0.0)
                nz = float(pNode.GetParameter("OcclusalNormalZ") or 1.0)
                vec = np.array([nx, ny, nz], dtype=np.float64)
                n = float(np.linalg.norm(vec))
                if n > 1e-8:
                    return vec / n
            except ValueError:
                pass
        return self._occlusal_normal.copy()

    def set_occlusal_normal(self, normal_vec: Sequence[float]) -> np.ndarray:
        vec = np.asarray(normal_vec, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vec))
        vec = vec / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if vec[2] < 0:
            vec = -vec
        self._occlusal_normal = vec
        pNode = self.getParameterNode() if slicer is not None else None
        if pNode is not None:
            pNode.SetParameter("OcclusalNormalX", f"{vec[0]:.6f}")
            pNode.SetParameter("OcclusalNormalY", f"{vec[1]:.6f}")
            pNode.SetParameter("OcclusalNormalZ", f"{vec[2]:.6f}")
        return vec

    @staticmethod
    def compute_occlusal_normal_from_3_points(pts_3x3: np.ndarray) -> np.ndarray:
        """Compute superior-pointing unit normal vector from 3 non-collinear occlusal points."""
        pts = np.asarray(pts_3x3, dtype=np.float64).reshape((3, 3))
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        n = float(np.linalg.norm(normal))
        if n < 1e-8:
            raise ValueError("Occlusal plane points are collinear.")
        normal /= n
        if normal[2] < 0:
            normal = -normal
        return normal

    def start_place_occlusal_points(self, volumeNode: Any) -> Any:
        if slicer is None or not volumeNode:
            return None
        name = f"{volumeNode.GetName()}_OcclusalPlanePoints"
        fidNode = slicer.mrmlScene.GetFirstNodeByName(name)
        if not fidNode:
            fidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
            fidNode.CreateDefaultDisplayNodes()
        fidNode.RemoveAllControlPoints()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(fidNode.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(1)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        return fidNode

    def apply_occlusal_plane_from_points(
        self, volumeNode: Any, ask_harden_transform: bool = True
    ) -> Optional[np.ndarray]:
        if slicer is None or not volumeNode:
            return None
        name = f"{volumeNode.GetName()}_OcclusalPlanePoints"
        fidNode = slicer.mrmlScene.GetFirstNodeByName(name)
        if not fidNode or fidNode.GetNumberOfControlPoints() < 3:
            slicer.util.warningDisplay("Place at least 3 occlusal points first.")
            return None
        pts = np.zeros((3, 3), dtype=np.float64)
        buf = [0.0, 0.0, 0.0]
        for i in range(3):
            fidNode.GetNthControlPointPositionWorld(i, buf)
            pts[i] = buf
        normal = self.compute_occlusal_normal_from_3_points(pts)
        self.set_occlusal_normal(normal)

        if ask_harden_transform and vtk is not None and getattr(slicer, "app", None) is not None:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            angle_rad = float(np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0)))
            if angle_rad > 1e-3 and slicer.util.confirmOkCancelDisplay(
                f"Occlusal tilt is {np.degrees(angle_rad):.1f} deg from world Z.\n"
                "Apply and harden an alignment transform to level the volume to the occlusal plane?",
                windowTitle="Occlusal Plane Alignment",
            ):
                rot_axis = np.cross(normal, z_axis)
                rot_axis /= max(1e-8, float(np.linalg.norm(rot_axis)))
                vtk_tf = vtk.vtkTransform()
                center = np.mean(pts, axis=0)
                vtk_tf.Translate(float(center[0]), float(center[1]), float(center[2]))
                vtk_tf.RotateWXYZ(
                    float(np.degrees(angle_rad)),
                    float(rot_axis[0]),
                    float(rot_axis[1]),
                    float(rot_axis[2]),
                )
                vtk_tf.Translate(float(-center[0]), float(-center[1]), float(-center[2]))
                tfNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLinearTransformNode", f"{volumeNode.GetName()}_OcclusalTransform"
                )
                tfNode.SetMatrixTransformToParent(vtk_tf.GetMatrix())
                volumeNode.SetAndObserveTransformNodeID(tfNode.GetID())
                slicer.vtkSlicerTransformLogic().hardenTransform(volumeNode)
                slicer.mrmlScene.RemoveNode(tfNode)
                self.set_occlusal_normal((0.0, 0.0, 1.0))
        return self.get_occlusal_normal()

    def standardize_to_lpi(self, volumeNode: Any) -> None:
        """Reset Axial (Red) and Cross-Section (Yellow/YellowMinus/YellowPlus) views without clobbering Panoramic."""
        if slicer is None or not volumeNode:
            return
        layoutManager = slicer.app.layoutManager() if getattr(slicer, "app", None) else None
        if not layoutManager:
            return

        pano_shown = self.navigator.is_panoramic_shown_in_green()
        for slice_name, orient in (
            ("Red", "Axial"),
            ("Yellow", "Sagittal"),
            ("YellowMinus", "Sagittal"),
            ("YellowPlus", "Sagittal"),
            ("Green", "Coronal"),
        ):
            if slice_name == "Green" and pano_shown:
                continue
            sw = layoutManager.sliceWidget(slice_name)
            if sw:
                comp = sw.mrmlSliceCompositeNode()
                if comp:
                    comp.SetBackgroundVolumeID(volumeNode.GetID())
                sNode = sw.sliceLogic().GetSliceNode()
                getattr(sNode, f"SetOrientationTo{orient}")()
                sw.sliceLogic().FitSliceToAll()

    # ------------------------------------------------------------------
    # Arch Curve Management (Segmentation Auto-Fit vs Interactive Place)
    # ------------------------------------------------------------------
    @staticmethod
    def get_curve_node_for_jaw(volumeNode: Any, jaw: str) -> Any:
        if slicer is None or not volumeNode:
            return None
        name = f"{volumeNode.GetName()}_{jaw.capitalize()}_DentalArch"
        return slicer.mrmlScene.GetFirstNodeByName(name)

    def ensure_arch_curve(self, volumeNode: Any, jaw: str) -> Any:
        """Return existing curve node for the requested jaw ONLY (never touches the opposite jaw)."""
        if slicer is None or not volumeNode:
            return None
        curveNode = self.get_curve_node_for_jaw(volumeNode, jaw)
        if curveNode is None:
            curveNode = self.panoramic.create_or_update_arch_curve(
                volumeNode, jaw=jaw, points_ras=None, visible=True
            )
        return curveNode

    def start_interactive_arch_placement(self, volumeNode: Any, jaw: str) -> Any:
        """Create/select the curve node for `jaw` and enter Slicer Markups Place Mode (no invented points)."""
        if slicer is None or not volumeNode:
            return None
        curveNode = self.ensure_arch_curve(volumeNode, jaw)
        if curveNode and curveNode.GetDisplayNode():
            curveNode.GetDisplayNode().SetVisibility(True)
            curveNode.GetDisplayNode().SetVisibility2D(True)
            curveNode.GetDisplayNode().SetVisibility3D(True)
        if getattr(slicer, "app", None) is not None:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsCurveNode")
            selectionNode.SetActivePlaceNodeID(curveNode.GetID())
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        return curveNode

    def extract_arch_points_from_segmentation(
        self, segmentationNode: Any, volumeNode: Any, jaw: str
    ) -> Optional[np.ndarray]:
        """Extract 3D tooth/jawbone coordinates in RAS from a segmentation node."""
        if not segmentationNode:
            return None
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return None

        try:
            from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
        except ImportError:
            vtk_to_numpy = None

        target_name = "upper teeth" if jaw.lower() == "upper" else "lower teeth"
        fallback_jaw_name = "maxilla" if jaw.lower() == "upper" else "mandible"
        seg_ids = self.vr._get_segment_ids(segmentation)

        chosen_segment = None
        for sid in seg_ids:
            seg = segmentation.GetSegment(sid)
            if seg and target_name in (seg.GetName() or "").lower():
                chosen_segment = seg
                break
        if chosen_segment is None:
            for sid in seg_ids:
                seg = segmentation.GetSegment(sid)
                if seg and fallback_jaw_name in (seg.GetName() or "").lower():
                    chosen_segment = seg
                    break
        if chosen_segment is None:
            pos_id = "Segment_3" if jaw.lower() == "upper" else "Segment_4"
            chosen_segment = segmentation.GetSegment(pos_id)

        if chosen_segment is not None and slicer is not None:
            rep_name = slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
            poly = chosen_segment.GetRepresentation(rep_name)
            if poly and poly.GetPoints() and poly.GetNumberOfPoints() >= 6:
                if vtk_to_numpy is not None:
                    pts = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
                else:
                    pts = np.array(
                        [poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())],
                        dtype=np.float64,
                    )
                if len(pts) > 1500:
                    step = max(1, len(pts) // 1500)
                    pts = pts[::step]
                return pts

        if volumeNode is not None:
            mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
            if mask is not None:
                vol = CBCTVolume.from_slicer_volume_node(volumeNode)
                if is_5class:
                    target_val = 3 if jaw.lower() == "upper" else 4
                    z_idx, y_idx, x_idx = np.where(mask == target_val)
                    if len(x_idx) < 6:
                        target_val = 1 if jaw.lower() == "upper" else 2
                        z_idx, y_idx, x_idx = np.where(mask == target_val)
                    if len(x_idx) >= 6:
                        step = max(1, len(x_idx) // 1500)
                        vox = np.column_stack([x_idx[::step], y_idx[::step], z_idx[::step]])
                        homog = np.hstack([vox, np.ones((len(vox), 1), dtype=np.float64)])
                        return (vol.affine @ homog.T).T[:, :3]
                else:
                    tooth_ids = MAXILLARY_TEETH_IDS if jaw.lower() == "upper" else MANDIBULAR_TEETH_IDS
                    centroids = []
                    for tid in tooth_ids:
                        tz, ty, tx = np.where(mask == tid)
                        if len(tx) > 0:
                            centroids.append(vol.voxel_to_world([tx.mean(), ty.mean(), tz.mean()]))
                    if len(centroids) >= 4:
                        return np.array(centroids, dtype=np.float64)
        return None

    def auto_fit_arch_for_jaw(
        self,
        volumeNode: Any,
        segmentationNode: Any,
        jaw: str,
        visible: bool = True,
        confirm_overwrite: bool = True,
    ) -> Any:
        """Fit dental arch curve for `jaw` from segmentation; if no segmentation, enter interactive place mode."""
        if slicer is None or not volumeNode:
            return None
        jaw_cap = jaw.capitalize()
        existing = self.get_curve_node_for_jaw(volumeNode, jaw_cap)
        if (
            existing is not None
            and existing.GetNumberOfControlPoints() >= 4
            and existing.GetAttribute("CBCTPlanner.UserEdited") == "1"
            and confirm_overwrite
            and getattr(slicer, "app", None) is not None
        ):
            if not slicer.util.confirmOkCancelDisplay(
                f"The {jaw_cap} dental arch curve has been manually edited.\n"
                "Do you want to overwrite your manual edits with an auto-fitted curve?",
                windowTitle="Overwrite Edited Arch Curve?",
            ):
                return existing

        pts_cloud = self.extract_arch_points_from_segmentation(segmentationNode, volumeNode, jaw_cap)
        if pts_cloud is None or len(pts_cloud) < 4:
            curveNode = self.start_interactive_arch_placement(volumeNode, jaw=jaw_cap)
            if getattr(slicer, "app", None) is not None:
                slicer.util.showStatusMessage(
                    f"No segmentation found for {jaw_cap} jaw — click points along the arch from Patient Right to Patient Left.",
                    5000,
                )
            return curveNode

        if len(pts_cloud) >= 6:
            fitted_pts = fit_dental_arch_from_points(
                pts_cloud, jaw_type=jaw_cap.lower(), num_control_points=11, ras_convention=True
            )
        else:
            fitted_pts = pts_cloud[np.argsort(pts_cloud[:, 0])[::-1]]

        self._is_programmatic_curve_update = True
        try:
            curveNode = self.panoramic.create_or_update_arch_curve(
                volumeNode,
                jaw=jaw_cap,
                points_ras=fitted_pts,
                visible=visible,
            )
            if curveNode:
                curveNode.SetAttribute("CBCTPlanner.UserEdited", "0")
        finally:
            self._is_programmatic_curve_update = False
        return curveNode

    def set_curves_visibility(self, volumeNode: Any, visible: bool) -> None:
        if slicer is None or not volumeNode:
            return
        for jaw in ("Lower", "Upper"):
            node = self.get_curve_node_for_jaw(volumeNode, jaw)
            if node and node.GetDisplayNode():
                dNode = node.GetDisplayNode()
                dNode.SetVisibility(bool(visible))
                dNode.SetVisibility2D(bool(visible))
                dNode.SetVisibility3D(bool(visible))

    # ------------------------------------------------------------------
    # Tooth Frame Lookup & Navigation
    # ------------------------------------------------------------------
    @staticmethod
    def fdi_to_normalized_arch_t(fdi: int) -> float:
        """Map FDI tooth number to normalized arch parameter t in [0.0 (Patient Right), 1.0 (Patient Left)]."""
        fdi_str = str(fdi)
        order = JAW_TOOTH_ORDER["lower"] if fdi >= 31 else JAW_TOOTH_ORDER["upper"]
        if fdi_str not in order:
            raise ValueError(f"Invalid FDI tooth number: {fdi}")
        idx = order.index(fdi_str)
        return idx / float(len(order) - 1)

    def get_tooth_arc_mm_and_frame(
        self, fdi: int, volumeNode: Any
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]:
        jaw = "Lower" if fdi >= 31 else "Upper"
        curveNode = self.get_curve_node_for_jaw(volumeNode, jaw)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return None
        _, arc_s, _, _, _ = self.panoramic.get_curve_points_and_frames(
            curveNode, num_samples=100, occlusal_normal=self.get_occlusal_normal()
        )
        total_len = float(arc_s[-1])
        t_norm = self.fdi_to_normalized_arch_t(fdi)
        target_arc_mm = t_norm * total_len
        return self.panoramic.get_frame_at_arc_mm(
            curveNode, arc_mm=target_arc_mm, occlusal_normal=self.get_occlusal_normal()
        )

    def jump_to_tooth_site(
        self,
        fdi: int,
        volumeNode: Any,
        width_mm: float = 24.0,
        tick_spacing_mm: float = 2.0,
        show_all_ticks: bool = True,
        slab_thickness_mm: float = 0.5,
        tilt_deg: float = 0.0,
        lightbox_enabled: bool = True,
        lightbox_step_mm: float = 2.0,
    ) -> Optional[float]:
        """Jump Cross-Section Lightbox (`YellowMinus`, `Yellow`, `YellowPlus`), Axial lines, and Pano lines to `fdi`."""
        jaw = "Lower" if fdi >= 31 else "Upper"
        curveNode = self.get_curve_node_for_jaw(volumeNode, jaw)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return None
        frame_info = self.get_tooth_arc_mm_and_frame(fdi, volumeNode)
        if frame_info is None:
            return None
        _, _, _, _, arc_mm, _ = frame_info
        pNode = self.getParameterNode() if slicer is not None else None
        panoNode = pNode.GetNodeReference("PanoramicVolume") if pNode else None
        if pNode is not None:
            pNode.SetParameter("CursorArcMm", f"{arc_mm:.2f}")
        self.navigator.update_cursor_at_arc_mm(
            volumeNode=volumeNode,
            curveNode=curveNode,
            arc_mm=arc_mm,
            width_mm=width_mm,
            tick_spacing_mm=tick_spacing_mm,
            show_all_ticks=show_all_ticks,
            slab_thickness_mm=slab_thickness_mm,
            tilt_deg=tilt_deg,
            lightbox_enabled=lightbox_enabled,
            lightbox_step_mm=lightbox_step_mm,
            occlusal_normal=self.get_occlusal_normal(),
            site_label=f"Tooth #{fdi}",
            panoNode=panoNode,
        )
        return arc_mm

    # ------------------------------------------------------------------
    # Honest Clinical Measurements (Segmentation Engine + Manual Rulers)
    # ------------------------------------------------------------------
    def _extract_combined_label_mask(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Tuple[Optional[np.ndarray], bool]:
        if slicer is None or not volumeNode or not segmentationNode:
            return None, False
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return None, False
        seg_ids = self.vr._get_segment_ids(segmentation)
        if not seg_ids:
            return None, False

        vol_arr = slicer.util.arrayFromVolume(volumeNode)
        combined = np.zeros(vol_arr.shape, dtype=np.uint16)
        is_5class = False

        for idx, sid in enumerate(seg_ids):
            seg = segmentation.GetSegment(sid)
            name_low = (seg.GetName() if seg else "").lower()
            label_val = idx + 1
            if "maxilla" in name_low or "skull" in name_low:
                label_val = 1
                is_5class = True
            elif "mandible" in name_low:
                label_val = 2
                is_5class = True
            elif "upper teeth" in name_low:
                label_val = 3
                is_5class = True
            elif "lower teeth" in name_low:
                label_val = 4
                is_5class = True
            elif "canal" in name_low:
                label_val = 5
                is_5class = True
            else:
                for fdi_str, tid in FDI_TO_TOOTH_ID.items():
                    if fdi_str in name_low:
                        label_val = tid
                        break

            try:
                seg_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                    segmentationNode, sid, volumeNode
                )
                if seg_arr is not None and seg_arr.shape == combined.shape:
                    combined[seg_arr > 0] = label_val
            except Exception:
                pass

        if not np.any(combined):
            return None, False
        return combined, is_5class

    @staticmethod
    def _read_manual_ruler_length(ruler_name: str) -> Optional[float]:
        if slicer is None or getattr(slicer, "mrmlScene", None) is None:
            return None
        node = slicer.mrmlScene.GetFirstNodeByName(ruler_name)
        if node and hasattr(node, "GetNumberOfControlPoints") and node.GetNumberOfControlPoints() >= 2:
            if hasattr(node, "GetLineLengthWorld"):
                length = float(node.GetLineLengthWorld())
                return round(length, 2) if length > 0.0 else None
            p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(0, p0)
            node.GetNthControlPointPositionWorld(1, p1)
            length = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
            return round(length, 2) if length > 0.0 else None
        return None

    def create_or_focus_manual_rulers_for_tooth(self, fdi: int, volumeNode: Any) -> None:
        if slicer is None or not volumeNode:
            return
        frame_info = self.get_tooth_arc_mm_and_frame(fdi, volumeNode)
        if frame_info is None:
            P = np.zeros(3, dtype=np.float64)
            N = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            B = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            P, _, N, B, _, _ = frame_info

        is_lower = fdi >= 31
        apical_vec = -B if is_lower else B
        yellow_id = self.navigator._get_view_node_id("Yellow")

        rulers = [
            (f"Tooth_{fdi}_HeightRuler", P, P + apical_vec * 10.0, (0.1, 0.85, 0.2)),
            (f"Tooth_{fdi}_WidthRuler", P - N * 3.0, P + N * 3.0, (0.95, 0.85, 0.1)),
            (f"Tooth_{fdi}_NerveRuler", P + apical_vec * 8.0, P + apical_vec * 12.0, (0.95, 0.25, 0.25)),
        ]
        for r_name, pt0, pt1, color in rulers:
            rNode = slicer.mrmlScene.GetFirstNodeByName(r_name)
            if not rNode:
                rNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", r_name)
                rNode.CreateDefaultDisplayNodes()
                rNode.AddControlPoint(float(pt0[0]), float(pt0[1]), float(pt0[2]))
                rNode.AddControlPoint(float(pt1[0]), float(pt1[1]), float(pt1[2]))
            dNode = rNode.GetDisplayNode()
            if dNode:
                dNode.SetColor(*color)
                dNode.SetVisibility(True)
                dNode.SetVisibility2D(True)
                if hasattr(dNode, "SetPropertiesLabelVisibility"):
                    dNode.SetPropertiesLabelVisibility(True)
                if yellow_id and hasattr(dNode, "SetViewNodeIDs"):
                    dNode.SetViewNodeIDs([yellow_id])

    def analyze_single_tooth(
        self,
        fdi: int,
        volumeNode: Any,
        segmentationNode: Any = None,
        engine: Optional[MeasurementEngine] = None,
    ) -> Dict[str, Any]:
        is_lower = int(fdi) >= 31
        jaw = "lower" if is_lower else "upper"
        frame_info = self.get_tooth_arc_mm_and_frame(int(fdi), volumeNode) if volumeNode else None
        P = frame_info[0] if frame_info is not None else None

        height_mm: Optional[float] = None
        w_crest_mm: Optional[float] = None
        w_3mm: Optional[float] = None
        w_5mm: Optional[float] = None
        nerve_clearance_mm: Optional[float] = None
        mean_hu: Optional[float] = None
        misch: Optional[str] = None
        rec_dia: Optional[float] = None
        rec_len: Optional[float] = None
        source = NOT_MEASURED

        if engine is None and volumeNode is not None and segmentationNode is not None:
            engine = self._build_measurement_engine(volumeNode, segmentationNode)

        if engine is not None:
            meas = engine.measure_implant_site(str(fdi))
            if meas is not None:
                height_mm = meas.bone_height_mm
                w_crest_mm = meas.crest_width_mm
                w_3mm = meas.bone_width_at_depths_mm.get(3.0)
                w_5mm = meas.bone_width_at_depths_mm.get(5.0)
                nerve_clearance_mm = meas.distance_to_vital_structure_mm
                mean_hu = meas.mean_density_hu
                misch = meas.misch_density_class
                rec_dia = meas.suggested_diameter_mm
                rec_len = meas.suggested_length_mm
                if any(v is not None for v in (height_mm, w_crest_mm, nerve_clearance_mm, mean_hu)):
                    source = "segmentation"

        manual_h = self._read_manual_ruler_length(f"Tooth_{fdi}_HeightRuler")
        manual_w = self._read_manual_ruler_length(f"Tooth_{fdi}_WidthRuler")
        manual_n = self._read_manual_ruler_length(f"Tooth_{fdi}_NerveRuler")
        if manual_h is not None:
            height_mm = manual_h
            source = "manual_ruler"
        if manual_w is not None:
            w_crest_mm = manual_w
            source = "manual_ruler"
        if manual_n is not None:
            nerve_clearance_mm = manual_n
            source = "manual_ruler"

        pos_ras: Optional[List[float]] = (
            [round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2)]
            if P is not None
            else None
        )
        return {
            "fdi": int(fdi),
            "jaw": jaw,
            "source": source,
            "height_mm": height_mm,
            "w_crest_mm": w_crest_mm,
            "w_3mm": w_3mm,
            "w_5mm": w_5mm,
            "nerve_clearance_mm": nerve_clearance_mm,
            "hu": mean_hu,
            "misch": misch,
            "rec_diameter": rec_dia,
            "rec_length": rec_len,
            "pos_ras": pos_ras,
        }

    def _build_measurement_engine(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Optional[MeasurementEngine]:
        if slicer is None or volumeNode is None or segmentationNode is None:
            return None
        mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            return None
        vol = CBCTVolume.from_slicer_volume_node(volumeNode)
        arches: Dict[str, DentalArch] = {}
        for jaw in ("lower", "upper"):
            cNode = self.get_curve_node_for_jaw(volumeNode, jaw)
            if cNode and cNode.GetNumberOfControlPoints() >= 4:
                raw_pts = self.panoramic.extract_world_points_from_curve_node(cNode)
                if len(raw_pts) >= 4:
                    arches[jaw] = DentalArch(
                        raw_pts,
                        jaw_type=jaw,
                        occlusal_normal=self.get_occlusal_normal(),
                        ras_convention=True,
                    )
        primary_arch = arches.get("lower") or arches.get("upper")
        return MeasurementEngine(
            vol,
            mask,
            arch=primary_arch,
            arches=arches,
            is_dental_segmentator_5class=is_5class,
        )

    def analyze_selected_teeth(
        self,
        sorted_teeth: Sequence[int],
        volumeNode: Any,
        segmentationNode: Any = None,
    ) -> List[Dict[str, Any]]:
        engine = (
            self._build_measurement_engine(volumeNode, segmentationNode)
            if (volumeNode is not None and segmentationNode is not None)
            else None
        )
        return [
            self.analyze_single_tooth(int(fdi), volumeNode, segmentationNode=segmentationNode, engine=engine)
            for fdi in sorted_teeth
        ]

    # ------------------------------------------------------------------
    # Report & CSV/JSON Export (HTML Escaping, Flattened CSV, Screenshots)
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_slice_screenshot_base64(slice_name: str) -> Optional[str]:
        if slicer is None or qt is None or not getattr(slicer, "app", None):
            return None
        try:
            lm = slicer.app.layoutManager()
            sw = lm.sliceWidget(slice_name) if lm else None
            if not sw:
                return None
            pixmap = sw.sliceView().grab()
            ba = qt.QByteArray()
            buf = qt.QBuffer(ba)
            buf.open(qt.QIODevice.WriteOnly)
            pixmap.save(buf, "PNG")
            b64 = base64.b64encode(bytes(ba.data())).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    @staticmethod
    def export_treatment_plan_files(
        plan_data: Sequence[Dict[str, Any]], out_dir: Path
    ) -> Tuple[Path, Path]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_json = out_path / "treatment_plan.json"
        out_csv = out_path / "treatment_plan.csv"

        out_json.write_text(json.dumps(list(plan_data), indent=2), encoding="utf-8")

        fieldnames = [
            "fdi",
            "jaw",
            "source",
            "height_mm",
            "w_crest_mm",
            "w_3mm",
            "w_5mm",
            "nerve_clearance_mm",
            "hu",
            "misch",
            "rec_diameter",
            "rec_length",
            "pos_r_mm",
            "pos_a_mm",
            "pos_s_mm",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in plan_data:
                pos = item.get("pos_ras")
                row_dict = {
                    "fdi": item.get("fdi"),
                    "jaw": item.get("jaw", ""),
                    "source": item.get("source", NOT_MEASURED),
                    "height_mm": item.get("height_mm") if item.get("height_mm") is not None else NOT_MEASURED,
                    "w_crest_mm": item.get("w_crest_mm") if item.get("w_crest_mm") is not None else NOT_MEASURED,
                    "w_3mm": item.get("w_3mm") if item.get("w_3mm") is not None else NOT_MEASURED,
                    "w_5mm": item.get("w_5mm") if item.get("w_5mm") is not None else NOT_MEASURED,
                    "nerve_clearance_mm": (
                        item.get("nerve_clearance_mm")
                        if item.get("nerve_clearance_mm") is not None
                        else NOT_MEASURED
                    ),
                    "hu": item.get("hu") if item.get("hu") is not None else NOT_MEASURED,
                    "misch": item.get("misch") if item.get("misch") is not None else NOT_MEASURED,
                    "rec_diameter": (
                        item.get("rec_diameter") if item.get("rec_diameter") is not None else NOT_MEASURED
                    ),
                    "rec_length": (
                        item.get("rec_length") if item.get("rec_length") is not None else NOT_MEASURED
                    ),
                    "pos_r_mm": pos[0] if (pos and len(pos) == 3) else NOT_MEASURED,
                    "pos_a_mm": pos[1] if (pos and len(pos) == 3) else NOT_MEASURED,
                    "pos_s_mm": pos[2] if (pos and len(pos) == 3) else NOT_MEASURED,
                }
                writer.writerow(row_dict)
        return out_json, out_csv

    def generate_printable_report(
        self,
        plan_data: Sequence[Dict[str, Any]],
        volumeNode: Any,
        out_dir: Path,
        capture_screenshots: bool = True,
    ) -> Path:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        raw_vol_name = volumeNode.GetName() if (volumeNode and hasattr(volumeNode, "GetName")) else "CBCT_Scan"
        safe_stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_vol_name).strip("_") or "CBCT_Scan"
        escaped_vol_name = html.escape(raw_vol_name)
        report_file = out_path / f"{safe_stem}_Treatment_Plan_Report.html"

        rows_html: List[str] = []
        for item in plan_data:
            fdi_esc = html.escape(str(item.get("fdi", "")))
            jaw_esc = html.escape(str(item.get("jaw", "")).capitalize())
            src_esc = html.escape(str(item.get("source", NOT_MEASURED)))
            ht_str = html.escape(format_metric(item.get("height_mm"), "mm"))
            w0_str = html.escape(format_metric(item.get("w_crest_mm"), "mm"))
            w3_str = html.escape(format_metric(item.get("w_3mm"), "mm"))
            w5_str = html.escape(format_metric(item.get("w_5mm"), "mm"))
            nerve_str = html.escape(format_metric(item.get("nerve_clearance_mm"), "mm"))

            if item.get("misch") and item.get("hu") is not None:
                misch_str = html.escape(f"{item['misch']} ({float(item['hu']):.0f} HU)")
            else:
                misch_str = NOT_MEASURED

            if item.get("rec_diameter") is not None and item.get("rec_length") is not None:
                imp_str = html.escape(
                    f"D{float(item['rec_diameter']):.1f} x L{float(item['rec_length']):.1f} mm"
                )
            else:
                imp_str = NOT_MEASURED

            rows_html.append(
                f"<tr>"
                f"<td>#{fdi_esc}</td>"
                f"<td>{jaw_esc}</td>"
                f"<td>{src_esc}</td>"
                f"<td>{ht_str}</td>"
                f"<td>{w0_str}</td>"
                f"<td>{w3_str}</td>"
                f"<td>{w5_str}</td>"
                f"<td>{nerve_str}</td>"
                f"<td>{misch_str}</td>"
                f"<td>{imp_str}</td>"
                f"</tr>"
            )

        screenshots_html = ""
        if capture_screenshots:
            cards = []
            for label, s_name in (
                ("Panoramic View (Green)", "Green"),
                ("Axial View (Red)", "Red"),
                ("Cross-Section CS- (-2 mm)", "YellowMinus"),
                ("Cross-Section CS0 (0 mm)", "Yellow"),
                ("Cross-Section CS+ (+2 mm)", "YellowPlus"),
            ):
                uri = self._capture_slice_screenshot_base64(s_name)
                if uri:
                    cards.append(
                        f"<div class='shot-card'><div><strong>{html.escape(label)}</strong></div>"
                        f"<img src='{uri}' alt='{html.escape(label)}'/></div>"
                    )
            if cards:
                screenshots_html = f"<div class='shots'>{''.join(cards)}</div>"

        html_doc = (
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>CBCT Implant Planning Report — {escaped_vol_name}</title>"
            "<style>"
            "body{font-family:sans-serif;padding:24px;color:#1e293b;}"
            "table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;}"
            "th,td{border:1px solid #cbd5e1;padding:8px;text-align:center;}"
            "th{background:#0f172a;color:#fff;}"
            ".shots{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;}"
            ".shot-card{flex:1;min-width:180px;border:1px solid #cbd5e1;padding:8px;text-align:center;}"
            ".shot-card img{max-width:100%;height:auto;margin-top:6px;}"
            "</style></head><body>"
            f"<h1>CBCT Implant Planning Report — {escaped_vol_name}</h1>"
            "<p><em>Note: Unsegmented sites without manual rulers are explicitly reported as "
            "\"not measured\".</em></p>"
            f"{screenshots_html}"
            "<table><thead><tr>"
            "<th>FDI</th><th>Jaw</th><th>Source</th><th>Ridge Height</th><th>Crest Width</th>"
            "<th>Width @ 3mm</th><th>Width @ 5mm</th><th>Nerve Distance</th>"
            "<th>Bone Density</th><th>Suggested Implant</th>"
            f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
            "</body></html>"
        )
        report_file.write_text(html_doc, encoding="utf-8")
        return report_file


class CBCTPlannerTest(ScriptedLoadableModuleTest):
    """3D Slicer module self-test."""

    def setUp(self) -> None:
        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            slicer.mrmlScene.Clear(0)

    def runTest(self) -> None:
        self.setUp()
        self.test_ras_arch_fitting_and_honest_measurements()

    def test_ras_arch_fitting_and_honest_measurements(self) -> None:
        pts = np.array(
            [
                [30.0, -20.0, 0.0],
                [20.0, 0.0, 0.0],
                [10.0, 15.0, 0.0],
                [0.0, 20.0, 0.0],
                [-10.0, 15.0, 0.0],
                [-20.0, 0.0, 0.0],
                [-30.0, -20.0, 0.0],
            ],
            dtype=np.float64,
        )
        ctrl = fit_dental_arch_from_points(pts, num_control_points=11, ras_convention=True)
        assert ctrl[0, 0] > ctrl[-1, 0], "Expected RAS Right (+X) -> Left (-X) ordering"
        logic = CBCTPlannerLogic()
        rows = logic.analyze_selected_teeth([46], volumeNode=None, segmentationNode=None)
        assert rows[0]["height_mm"] is None

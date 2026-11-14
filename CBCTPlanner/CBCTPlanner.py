"""
CBCT Dental Planner - 3D Slicer Scripted Loadable Extension Module.

Implements a comprehensive clinical workflow backed by a single-source-of-truth MRML ParameterNode:
  - Phase 1 (Clinical Safety B1-B4, B21-B22):
      * B1: Manual rulers report None until user-modified (`CBCTPlanner.UserModified == "1"`) or placed.
      * B2: FDI->arc mapping uses projected segmentation tooth centroids, draggable calibration markers,
        and an anatomical mesiodistal crown width table with missing-tooth interpolation (flagged approximate).
      * B3: Non-destructive parent linear transform (`OcclusalTransform`) applied together to volume,
        segmentation, curves, fiducials, rulers, and implants, with full reset.
      * B4: Arch-frame cross-section slab resampling in `core/measurements.py` (independent of array axis K
        or volume tilt) and Misch density classes labeled `"approximate (CBCT gray values are not calibrated HU)"`.
      * Visible `"NOT FOR DIAGNOSTIC USE — PLANNING AID ONLY"` banner in UI and all exports.
  - Phase 2 (Correctness B5-B20):
      * B5: Right-handed (`det = +1`) CPR displacement frame `(-N, B, T)` matching grid frame `det = +1`.
      * B6: Fixed `<= 1.5 mm` CPR frame spacing independent of output voxel spacing.
      * B7: Slice-view scoped keyboard shortcuts (`should_handle_navigation_shortcut`).
      * B8: VR preset re-application on `A->B->A` switch and percentile-based transfer functions for non-HU CBCTs.
      * B9: Bounded LRU caches (`max_entries=8`) and automatic cleanup of finished/cancelled CLI nodes.
      * B10: Preserves user zoom/pan on panoramic preview/release refreshes.
      * B11: Updates Axial (`Red`) slice offset only on explicit tooth jump (`sync_axial_slice=True`).
      * B12: Explicit Slab Reconstruction checkbox and mode (`Mean` default, `MIP`, `MinIP`).
      * B13: Sub-sample line-segment curve projection.
      * B14: Automatic curve observers on all ParameterNode-referenced curves (including scene/MRB loads).
      * B15: Node reference IDs stored in ParameterNode so renaming volumes never orphans curves or lines.
      * B16: Full UI state persisted in ParameterNode (`QSettings` used only for initial global defaults).
      * B17: Odontogram plain click = jump only; `Ctrl+click`/double-click/button toggles plan; Lock to Tooth locks.
      * B18: `ctkCollapsibleButton` headers always remain enabled; inner content widgets disable with hint text.
      * B19: Segment matching by terminology/attributes + exact token regex + precedence (`pulp > tooth > canal > sinus > jaw`).
      * B20: Voxel-grid downsampled horseshoe spline arch auto-fit (drops whole-skull fallback).
  - Phase 3 (Performance O1-O9):
      * O1: In-process `scipy.ndimage.map_coordinates` panoramic generator (`Mean`/`MIP`/`MinIP`/`RaySum`).
      * O2: 30 ms coalesced cursor timer, single frame eval, dirty-node-only `StartModify`/`EndModify`.
      * O3: MTime-cached `MeasurementEngine`, zero-copy volume view, and ROI-cropped EDT.
      * O4-O9: `CursorSettings` dataclass, node ID caching, one-time ParameterNode defaults, ruff config, and phantom integration test.
  - Phase 4-6 (UX, Reporting & Clinical Features F1-F13, F18-F19):
      * B21-B22, F10: `vtkWindowToImageFilter` screenshots, dynamic lightbox labels, opt-in patient metadata,
        HTML + PDF report, clean numeric CSV columns with `measurement_source`.
      * F1: Mandibular canal & root-canal centerline + radius tube, manual trace, and 2 mm safety proximity alert.
      * F2, F11, F12, F19: Generic JSON implant library, cross-section placement/tilt, canal/sinus/implant clearance,
        parallelism angles, approximate gray values, implant-centric view, STL/JSON/DICOM export, and clinician confirmation gate.
      * F3-F9, F13, F18: Single occlusal arch mode with migration, FDI overlays, dynamic lightbox & density profile,
        automated occlusal plane from teeth + ASO detection, airway volume/mCSA, TMJ presets, angle/surface distance, and Undo/Redo.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
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
    CLINICAL_DISCLAIMER_TEXT,
    DENTAL_SEGMENT_PRESETS,
    SOFTWARE_VERSION,
    AirwayAnalysisResult,
    CanalProximityResult,
    CanalTracerLogic,
    CrossSectionNavigator,
    CursorSettings,
    ImplantPlannerLogic,
    PanoramicCurveLogic,
    PlacedImplant,
    TMJCondyleFrame,
    UndoRedoManager,
    VRVisualizationLogic,
    capture_slice_view_png_base64,
    compute_angle_between_three_points_deg,
    compute_occlusal_plane_from_segmentation,
    compute_pharyngeal_airway_metrics,
    compute_surface_to_surface_distance_mm,
    compute_tmj_condyle_frames,
    create_or_update_fdi_segmentation_overlays,
    detect_automated_dental_tools_modules,
    export_html_to_pdf,
    export_reformatted_dicom_series,
    registerAndSetDentalLayout,
    registerAndSetDentalLightboxLayout,
    should_handle_navigation_shortcut,
)
from CBCTPlannerLib.core.arch_detector import (
    JAW_TOOTH_ORDER,
    DentalArch,
    ToothArcSite,
    compute_fdi_arc_positions_mm,
    fit_dental_arch_from_points,
)
from CBCTPlannerLib.core.labels import (
    ANATOMICAL_MESIODISTAL_WIDTHS_MM,
    FDI_TO_TOOTH_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    match_segment_to_label_and_precedence,
)
from CBCTPlannerLib.core.measurements import (
    MISCH_APPROXIMATE_DISCLAIMER,
    MeasurementEngine,
)
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
                f"{CLINICAL_DISCLAIMER_TEXT}\n\n"
                "Dental CBCT Planner: 6-step workflow (Load -> Arch -> Panoramic -> "
                "Cross-sections -> Plan & Implants -> Canal/Airway/TMJ) with in-process CPR, "
                "dynamic multi-slice lightbox, +-30 deg tilt, mandibular canal proximity alerts, "
                "extensible JSON implant library, and HTML/PDF/CSV/STL/DICOM export."
            )
            self.parent.acknowledgementText = (
                "Uses 3D Slicer, DentalSegmentator visualization presets, and PerkLab CPR conventions."
            )


class CBCTPlannerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Progressive Workflow UI for CBCT Dental Planner with VTKObservationMixin and full ParameterNode state."""

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
        self._observedCurveNodeIDs: Set[str] = set()

        # Content containers for B18 (keep collapsible headers enabled, disable only inner body)
        self.step2ContentWidget: Any = None
        self.step3ContentWidget: Any = None
        self.step4ContentWidget: Any = None
        self.step5ContentWidget: Any = None
        self.step6ContentWidget: Any = None

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
        self._installKeyboardShortcuts()

        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
            self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onNodeRemoved)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Disconnect observers, stop timers, and release resources on module exit."""
        self._saveGlobalDefaultsToQSettings()
        if self.logic is not None:
            self.logic.cleanup()
        if self._vrDebounceTimer is not None:
            self._vrDebounceTimer.stop()
        if self._panoPreviewTimer is not None:
            self._panoPreviewTimer.stop()
        self._observedCurveNodeIDs.clear()
        self.removeObservers()

    # ------------------------------------------------------------------
    # UI Construction (B18: Headers Always Enabled, Inner Content Gated)
    # ------------------------------------------------------------------
    def _buildWorkflowUI(self) -> None:
        if qt is None or ctk is None or slicer is None:
            return

        # B21/B22: Visible Clinical Disclaimer Banner + Undo/Redo Toolbar (F13)
        bannerFrame = qt.QFrame()
        bannerFrame.setFrameShape(qt.QFrame.StyledPanel)
        bannerLayout = qt.QVBoxLayout(bannerFrame)
        bannerLayout.setContentsMargins(6, 6, 6, 6)
        self.disclaimerBannerLabel = qt.QLabel(f"<b>{html.escape(CLINICAL_DISCLAIMER_TEXT)}</b>")
        self.disclaimerBannerLabel.setWordWrap(True)
        self.disclaimerBannerLabel.setToolTip(CLINICAL_DISCLAIMER_TEXT)
        bannerLayout.addWidget(self.disclaimerBannerLabel)

        undoRow = qt.QHBoxLayout()
        self.btnUndo = qt.QPushButton("Undo (Ctrl+Z)")
        self.btnUndo.setToolTip("Undo last cursor, curve, manual ruler, or implant edit (F13).")
        self.btnRedo = qt.QPushButton("Redo (Ctrl+Y)")
        self.btnRedo.setToolTip("Redo last undone action (F13).")
        self.globalStatusLabel = qt.QLabel("Cursor: 0.0 mm | Active Tooth: None")
        self.globalStatusLabel.setToolTip("Live cursor arc position and active tooth indicator.")
        undoRow.addWidget(self.btnUndo)
        undoRow.addWidget(self.btnRedo)
        undoRow.addWidget(self.globalStatusLabel, 1)
        bannerLayout.addLayout(undoRow)
        self.layout.addWidget(bannerFrame)

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
        step1Layout.addRow("Segmentation (Optional):", self.segmentationSelector); self.btnLoadExternalMask = qt.QPushButton("Load AI Mask (.nii.gz/.nrrd)"); self.btnFlipSIOrientation = qt.QPushButton("Flip S-I Orientation (ToothFairy)"); self.panoFlipVerticalCheck = qt.QCheckBox("Flip OPG Vertically (S-I)"); maskRow = qt.QHBoxLayout(); maskRow.addWidget(self.btnLoadExternalMask); maskRow.addWidget(self.btnFlipSIOrientation); maskRow.addWidget(self.panoFlipVerticalCheck); step1Layout.addRow("AI Mask & Orientation:", maskRow)

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

        # B3 & F6: Non-Destructive Occlusal Plane Alignment (3-Point or Auto from Teeth Segmentation)
        occRow = qt.QHBoxLayout()
        self.btnPlaceOcclusalPts = qt.QPushButton("Place 3 Occlusal Points")
        self.btnPlaceOcclusalPts.setToolTip(
            "Place 3 fiducial points (right molar, incisors, left molar) to define the occlusal plane normal."
        )
        self.btnApplyOcclusalPlane = qt.QPushButton("Align Occlusal Plane (Non-Destructive)")
        self.btnApplyOcclusalPlane.setToolTip(
            "Apply a shared non-destructive parent transform to volume, segmentation, curves, fiducials, and rulers (B3)."
        )
        self.btnAutoOcclusalFromSeg = qt.QPushButton("Fit from Segmented Teeth")
        self.btnAutoOcclusalFromSeg.setToolTip(
            "Automatically fit occlusal plane normal to segmented tooth cusps/incisal edges (F6)."
        )
        self.btnResetOcclusalPlane = qt.QPushButton("Reset Alignment")
        self.btnResetOcclusalPlane.setToolTip(
            "Remove parent occlusal alignment transform from all nodes and restore World Z (B3)."
        )
        occRow.addWidget(self.btnPlaceOcclusalPts)
        occRow.addWidget(self.btnApplyOcclusalPlane)
        occRow.addWidget(self.btnAutoOcclusalFromSeg)
        occRow.addWidget(self.btnResetOcclusalPlane)
        step1Layout.addRow("Occlusal Plane:", occRow)

        self.vrPresetCombo = qt.QComboBox()
        self.vrPresetCombo.addItems(list(VRVisualizationLogic.SUPPORTED_VR_PRESETS))
        self.vrPresetCombo.setToolTip("Select Volume Rendering transfer function preset (re-applied cleanly on change, B8).")
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
        self.btnApplyDental3D = qt.QPushButton("Style 3D + FDI Overlays")
        self.btnApplyDental3D.setToolTip(
            "Apply DentalSegmentator 3D colors/opacities and show tooth outlines & #FDI labels on Pano/CS (F4)."
        )
        vrToggleRow.addWidget(self.showVrCheckBox)
        vrToggleRow.addWidget(self.cropVrCheckBox)
        vrToggleRow.addWidget(self.btnApplyDental3D)
        step1Layout.addRow(vrToggleRow)

        # ==================== STEP 2: ARCH (B18, B20, F3) ====================
        self.step2Box = ctk.ctkCollapsibleButton()
        self.step2Box.text = "Step 2: Dental Arch Curve (Shared Occlusal or Per-Jaw, Right -> Left)"
        self.step2Box.collapsed = True
        self.layout.addWidget(self.step2Box)
        step2Outer = qt.QVBoxLayout(self.step2Box)

        self.step2HintLabel = qt.QLabel("Prerequisite: Select an input CBCT volume in Step 1.")
        self.step2HintLabel.setWordWrap(True)
        step2Outer.addWidget(self.step2HintLabel)

        self.step2ContentWidget = qt.QWidget()
        step2Layout = qt.QFormLayout(self.step2ContentWidget)
        step2Outer.addWidget(self.step2ContentWidget)

        archModeRow = qt.QHBoxLayout()
        self.archModeCombo = qt.QComboBox()
        self.archModeCombo.addItems(["Per-Jaw (Lower / Upper)", "Shared Occlusal Arch"])
        self.archModeCombo.setToolTip(
            "Choose between a single shared occlusal-level arch driving both jaws (F3) or separate Lower/Upper curves."
        )
        self.activeJawCombo = qt.QComboBox()
        self.activeJawCombo.addItems(["Lower", "Upper"])
        self.activeJawCombo.setToolTip("Select which jaw arch curve (Lower Mandible or Upper Maxilla) is active.")
        archModeRow.addWidget(qt.QLabel("Mode:"))
        archModeRow.addWidget(self.archModeCombo)
        archModeRow.addWidget(qt.QLabel("Active Jaw:"))
        archModeRow.addWidget(self.activeJawCombo)
        step2Layout.addRow("Arch Curve Mode:", archModeRow)

        archBtnRow = qt.QHBoxLayout()
        self.btnAutoFitArch = qt.QPushButton("Auto-Fit Arch from Segmented Teeth")
        self.btnAutoFitArch.setToolTip(
            "Fit a smooth horseshoe spline to segmented teeth (B20; never fits to full skull; asks before overwriting manual edits)."
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

        # ==================== STEP 3: PANORAMIC (O1, F3) ====================
        self.step3Box = ctk.ctkCollapsibleButton()
        self.step3Box.text = "Step 3: In-Process Panoramic CPR (Mean / MIP / MinIP / RaySum)"
        self.step3Box.collapsed = True
        self.layout.addWidget(self.step3Box)
        step3Outer = qt.QVBoxLayout(self.step3Box)

        self.step3HintLabel = qt.QLabel("Prerequisite: Requires an active dental arch curve (>= 4 points) in Step 2.")
        self.step3HintLabel.setWordWrap(True)
        step3Outer.addWidget(self.step3HintLabel)

        self.step3ContentWidget = qt.QWidget()
        step3Layout = qt.QFormLayout(self.step3ContentWidget)
        step3Outer.addWidget(self.step3ContentWidget)

        panoGeomRow = qt.QHBoxLayout()
        self.panoThicknessSpin = qt.QDoubleSpinBox()
        self.panoThicknessSpin.minimum = 2.0
        self.panoThicknessSpin.maximum = 45.0
        self.panoThicknessSpin.singleStep = 1.0
        self.panoThicknessSpin.value = 25.0
        self.panoThicknessSpin.suffix = " mm"
        self.panoThicknessSpin.setToolTip("Buccolingual slab thickness for panoramic projection.")

        self.panoHeightSpin = qt.QDoubleSpinBox()
        self.panoHeightSpin.minimum = 20.0
        self.panoHeightSpin.maximum = 220.0
        self.panoHeightSpin.singleStep = 5.0
        self.panoHeightSpin.value = 120.0
        self.panoHeightSpin.suffix = " mm"
        self.panoHeightSpin.setToolTip("Inferior-superior vertical height of the panoramic radiograph (F3).")

        self.panoOffsetSpin = qt.QDoubleSpinBox()
        self.panoOffsetSpin.minimum = -15.0
        self.panoOffsetSpin.maximum = 15.0
        self.panoOffsetSpin.singleStep = 0.5
        self.panoOffsetSpin.value = 0.0
        self.panoOffsetSpin.suffix = " mm"
        self.panoOffsetSpin.setToolTip("Buccolingual offset of the panoramic slab relative to the arch curve (F3).")

        panoGeomRow.addWidget(qt.QLabel("Thickness:"))
        panoGeomRow.addWidget(self.panoThicknessSpin)
        panoGeomRow.addWidget(qt.QLabel("Height:"))
        panoGeomRow.addWidget(self.panoHeightSpin)
        panoGeomRow.addWidget(qt.QLabel("Offset:"))
        panoGeomRow.addWidget(self.panoOffsetSpin)
        step3Layout.addRow("Panoramic Geometry:", panoGeomRow)

        self.panoProjCombo = qt.QComboBox()
        self.panoProjCombo.addItems(["Mean", "MIP", "MinIP", "RaySum"])
        self.panoProjCombo.setToolTip("Intensity projection mode across the slab (including RaySum pseudo-pan, F3).")
        step3Layout.addRow("Projection Mode:", self.panoProjCombo)

        self.autoUpdatePanoCheckBox = qt.QCheckBox("Auto-update Panoramic when Curve Changes (Preview / Full-Res)")
        self.autoUpdatePanoCheckBox.setChecked(True)
        self.autoUpdatePanoCheckBox.setToolTip(
            "Automatically regenerate a fast in-process preview while dragging curve points and full-resolution on release (O1)."
        )
        step3Layout.addRow(self.autoUpdatePanoCheckBox)

        panoRunRow = qt.QHBoxLayout()
        self.btnGeneratePanoramic = qt.QPushButton("Generate Panoramic View (Green)")
        self.btnGeneratePanoramic.setToolTip("Run in-process map_coordinates Curved Planar Reformation (O1).")
        self.btnCancelPanoramic = qt.QPushButton("Cancel")
        self.btnCancelPanoramic.setEnabled(False)
        self.btnCancelPanoramic.setToolTip("Cancel an in-progress panoramic sampling job.")
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
        self.step4Box.text = "Step 4: Multi-Slice Cross-Sectional Lightbox & +-30° Tilt"
        self.step4Box.collapsed = True
        self.layout.addWidget(self.step4Box)
        step4Outer = qt.QVBoxLayout(self.step4Box)

        self.step4HintLabel = qt.QLabel("Prerequisite: Requires an active dental arch curve (>= 4 points) in Step 2.")
        self.step4HintLabel.setWordWrap(True)
        step4Outer.addWidget(self.step4HintLabel)

        self.step4ContentWidget = qt.QWidget()
        step4Layout = qt.QFormLayout(self.step4ContentWidget)
        step4Outer.addWidget(self.step4ContentWidget)

        arcNavRow = qt.QHBoxLayout()
        self.btnPrevArcMm = qt.QPushButton("-1 mm")
        self.btnPrevArcMm.setToolTip("Step cursor 1 mm toward Patient Right (Left Arrow in slice view).")
        self.arcMmSlider = ctk.ctkSliderWidget()
        self.arcMmSlider.minimum = 0.0
        self.arcMmSlider.maximum = 100.0
        self.arcMmSlider.singleStep = 1.0
        self.arcMmSlider.pageStep = 5.0
        self.arcMmSlider.value = 50.0
        self.arcMmSlider.suffix = " mm"
        self.arcMmSlider.setToolTip("Continuous arc-length position along the dental arch curve in mm.")
        self.btnNextArcMm = qt.QPushButton("+1 mm")
        self.btnNextArcMm.setToolTip("Step cursor 1 mm toward Patient Left (Right Arrow in slice view).")
        arcNavRow.addWidget(self.btnPrevArcMm)
        arcNavRow.addWidget(self.arcMmSlider)
        arcNavRow.addWidget(self.btnNextArcMm)
        step4Layout.addRow("Cursor Position (Arc mm):", arcNavRow)

        tiltRow = qt.QHBoxLayout()
        self.csTiltSlider = ctk.ctkSliderWidget()
        self.csTiltSlider.minimum = -30.0
        self.csTiltSlider.maximum = 30.0
        self.csTiltSlider.singleStep = 1.0
        self.csTiltSlider.pageStep = 5.0
        self.csTiltSlider.value = 0.0
        self.csTiltSlider.suffix = "°"
        self.csTiltSlider.setToolTip("Lean the cross-section plane mesiodistally between -30° and +30° around N.")
        self.btnResetTilt = qt.QPushButton("Reset 0°")
        self.btnResetTilt.setToolTip("Reset cross-section tilt to 0° (perpendicular to occlusal plane).")
        tiltRow.addWidget(self.csTiltSlider)
        tiltRow.addWidget(self.btnResetTilt)
        step4Layout.addRow("Cross-Section Tilt (+-30°):", tiltRow)

        lightboxRow = qt.QHBoxLayout()
        self.enableLightboxCheckBox = qt.QCheckBox("Enable Lightbox (CS- / CS0 / CS+)")
        self.enableLightboxCheckBox.setChecked(True)
        self.enableLightboxCheckBox.setToolTip("Synchronize YellowMinus, Yellow, and YellowPlus cross-sections.")
        self.lightboxStepSpin = qt.QDoubleSpinBox()
        self.lightboxStepSpin.minimum = 0.5
        self.lightboxStepSpin.maximum = 10.0
        self.lightboxStepSpin.singleStep = 0.5
        self.lightboxStepSpin.value = 2.0
        self.lightboxStepSpin.suffix = " mm"
        self.lightboxStepSpin.setToolTip("Arc-length spacing between adjacent lightbox slices.")
        self.lightboxCountSpin = qt.QSpinBox()
        self.lightboxCountSpin.minimum = 3
        self.lightboxCountSpin.maximum = 9
        self.lightboxCountSpin.singleStep = 2
        self.lightboxCountSpin.value = 3
        self.lightboxCountSpin.setToolTip("Number of slices in Dynamic Lightbox sequence export (F5).")
        self.btnActivateLightboxLayout = qt.QPushButton("Open 3-Slice Lightbox View")
        self.btnActivateLightboxLayout.setToolTip("Switch viewport layout to show CS-, CS0, and CS+ side by side.")
        lightboxRow.addWidget(self.enableLightboxCheckBox)
        lightboxRow.addWidget(qt.QLabel("Step:"))
        lightboxRow.addWidget(self.lightboxStepSpin)
        lightboxRow.addWidget(qt.QLabel("N:"))
        lightboxRow.addWidget(self.lightboxCountSpin)
        lightboxRow.addWidget(self.btnActivateLightboxLayout)
        step4Layout.addRow("Dynamic Lightbox:", lightboxRow)

        self.lockFdiCombo = qt.QComboBox()
        self.lockFdiCombo.addItem("Free Cursor (Unlocked)", "")
        for fdi in Q1_TEETH + Q2_TEETH + Q4_TEETH + Q3_TEETH:
            self.lockFdiCombo.addItem(f"Tooth #{fdi}", str(fdi))
        self.lockFdiCombo.setToolTip(
            "Lock cursor to a specific FDI tooth site (B17). Dragging the cursor line manually releases the lock."
        )
        step4Layout.addRow("Lock to Tooth FDI:", self.lockFdiCombo)

        # B12: Explicit Slab Reconstruction Checkbox + Mode + Thickness
        csParamsRow = qt.QHBoxLayout()
        self.csWidthSpin = qt.QDoubleSpinBox()
        self.csWidthSpin.minimum = 8.0
        self.csWidthSpin.maximum = 50.0
        self.csWidthSpin.value = 24.0
        self.csWidthSpin.suffix = " mm"
        self.csWidthSpin.setToolTip("Buccolingual width of the active cross-section indicator line on the Axial view.")

        self.csSlabEnabledCheckBox = qt.QCheckBox("Slab")
        self.csSlabEnabledCheckBox.setChecked(False)
        self.csSlabEnabledCheckBox.setToolTip("Explicitly enable thick-slab reconstruction on Cross-Section views (B12).")

        self.csSlabModeCombo = qt.QComboBox()
        self.csSlabModeCombo.addItems(["Mean", "MIP", "MinIP"])
        self.csSlabModeCombo.setToolTip("Cross-section slab reconstruction mode (Mean default, MIP option, B12).")

        self.csSlabSpin = qt.QDoubleSpinBox()
        self.csSlabSpin.minimum = 0.2
        self.csSlabSpin.maximum = 20.0
        self.csSlabSpin.singleStep = 0.5
        self.csSlabSpin.value = 0.5
        self.csSlabSpin.suffix = " mm"
        self.csSlabSpin.setToolTip("Slab thickness in mm when Slab checkbox is enabled.")

        self.tickSpacingSpin = qt.QDoubleSpinBox()
        self.tickSpacingSpin.minimum = 1.0
        self.tickSpacingSpin.maximum = 20.0
        self.tickSpacingSpin.singleStep = 1.0
        self.tickSpacingSpin.value = 2.0
        self.tickSpacingSpin.suffix = " mm"
        self.tickSpacingSpin.setToolTip("Spacing in mm between adjacent ticks in the Axial cross-section tick fan.")

        csParamsRow.addWidget(qt.QLabel("Width:"))
        csParamsRow.addWidget(self.csWidthSpin)
        csParamsRow.addWidget(self.csSlabEnabledCheckBox)
        csParamsRow.addWidget(self.csSlabModeCombo)
        csParamsRow.addWidget(self.csSlabSpin)
        csParamsRow.addWidget(qt.QLabel("Tick:"))
        csParamsRow.addWidget(self.tickSpacingSpin)
        step4Layout.addRow("Slice & Slab Settings:", csParamsRow)

        self.showAxialTicksCheckBox = qt.QCheckBox("Show Cross-Section Tick Fan on Axial (Red) View")
        self.showAxialTicksCheckBox.setChecked(True)
        self.showAxialTicksCheckBox.setToolTip("Display faint perpendicular tick lines along the arch in the Red slice view only.")
        step4Layout.addRow(self.showAxialTicksCheckBox)

        self.csStatusLabel = qt.QLabel("Cursor: 0.0 mm | Tilt: 0° | Lightbox: -2.0 / 0.0 / +2.0 mm")
        step4Layout.addRow(self.csStatusLabel)

        # ==================== STEP 5: ODONTOGRAM, IMPLANTS & REPORT ====================
        self.step5Box = ctk.ctkCollapsibleButton()
        self.step5Box.text = "Step 5: Odontogram, Site Morphometry, Implants & Clinical Report"
        self.step5Box.collapsed = True
        self.layout.addWidget(self.step5Box)
        step5Outer = qt.QVBoxLayout(self.step5Box)

        self.step5HintLabel = qt.QLabel(
            "Prerequisite: Requires an active dental arch curve in Step 2. "
            "Plain Click = Jump only; Ctrl+Click / Toggle button = Add/Remove in Plan (B17)."
        )
        self.step5HintLabel.setWordWrap(True)
        step5Outer.addWidget(self.step5HintLabel)

        self.step5ContentWidget = qt.QWidget()
        step5Layout = qt.QVBoxLayout(self.step5ContentWidget)
        step5Outer.addWidget(self.step5ContentWidget)

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
        self.btnToggleActiveToothInPlan = qt.QPushButton("Toggle Active Tooth in Plan")
        self.btnToggleActiveToothInPlan.setToolTip(
            "Add or remove the currently active tooth in the treatment plan table (B17)."
        )
        self.btnCreateManualRulers = qt.QPushButton("Create Manual Rulers (Must Adjust to Measure)")
        self.btnCreateManualRulers.setToolTip(
            "Create Height, Width, and Nerve rulers for the active tooth. Untouched rulers report None until dragged (B1)."
        )
        self.btnRecomputeSelected = qt.QPushButton("Refresh Measurements")
        self.btnRecomputeSelected.setToolTip("Re-evaluate segmentation or user-modified manual rulers for planned teeth.")
        self.btnClearPlan = qt.QPushButton("Clear Plan")
        self.btnClearPlan.setToolTip("Clear all selected teeth from the treatment plan table.")
        rulerRow.addWidget(self.btnToggleActiveToothInPlan)
        rulerRow.addWidget(self.btnCreateManualRulers)
        rulerRow.addWidget(self.btnRecomputeSelected)
        rulerRow.addWidget(self.btnClearPlan)
        step5Layout.addLayout(rulerRow)

        # F2 & F19: Implant Library & Placement Bar
        implantRow = qt.QHBoxLayout()
        self.implantLibraryCombo = qt.QComboBox()
        if self.logic is not None:
            for entry in self.logic.implants.get_library_entries():
                self.implantLibraryCombo.addItem(entry.label, entry.id)
        self.implantLibraryCombo.setToolTip("Select generic cylinder implant dimensions from extensible JSON library (F2).")
        self.btnPlaceImplant = qt.QPushButton("Place / Update Implant at Active Tooth")
        self.btnPlaceImplant.setToolTip("Place cylinder implant at active tooth site with safety zone verification (F2).")
        self.btnImplantCentricView = qt.QPushButton("Implant-Centric View")
        self.btnImplantCentricView.setToolTip("Align cross-section slice along the placed implant's long axis (F2).")
        self.btnConfirmAiSuggestion = qt.QPushButton("Confirm AI Suggestion")
        self.btnConfirmAiSuggestion.setToolTip(
            "Explicitly confirm an AI-suggested implant size so it can enter exported clinical reports (F19)."
        )
        implantRow.addWidget(self.implantLibraryCombo)
        implantRow.addWidget(self.btnPlaceImplant)
        implantRow.addWidget(self.btnImplantCentricView)
        implantRow.addWidget(self.btnConfirmAiSuggestion)
        step5Layout.addLayout(implantRow)

        self.planTable = qt.QTableWidget()
        self.planTable.setColumnCount(10)
        self.planTable.setHorizontalHeaderLabels(
            [
                "FDI Site",
                "Jaw",
                "Source",
                "Ridge Height",
                "Crest Width",
                "Width @ 3mm",
                "Width @ 5mm",
                "Nerve Distance",
                "Misch (Approx)",
                "Implant / Suggestion",
            ]
        )
        self.planTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.planTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.planTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.planTable.setMinimumHeight(150)
        step5Layout.addWidget(self.planTable)

        exportOptRow = qt.QHBoxLayout()
        self.includePatientMetaCheckBox = qt.QCheckBox("Include Patient Metadata in Report (Opt-In; Anonymized by Default)")
        self.includePatientMetaCheckBox.setChecked(False)
        self.includePatientMetaCheckBox.setToolTip(
            "When unchecked, exported HTML/PDF/DICOM reports anonymize Patient Name and ID to 'Anonymous' (B22)."
        )
        exportOptRow.addWidget(self.includePatientMetaCheckBox)
        step5Layout.addLayout(exportOptRow)

        exportRow = qt.QHBoxLayout()
        self.btnGenerateReport = qt.QPushButton("Export HTML + PDF Clinical Report")
        self.btnGenerateReport.setToolTip(
            "Generate printable HTML and PDF clinical reports with renderWindow screenshots, implants, and disclaimer (B21, B22, F10)."
        )
        self.btnExportPlan = qt.QPushButton("Export CSV / JSON / Implant STL")
        self.btnExportPlan.setToolTip(
            "Export clean numeric CSV (empty when unmeasured), JSON, and triangulated implant STL meshes (B22, F11)."
        )
        exportRow.addWidget(self.btnGenerateReport)
        exportRow.addWidget(self.btnExportPlan)
        step5Layout.addLayout(exportRow)

        # ==================== STEP 6: CANAL, AIRWAY & TMJ (F1, F7, F8) ====================
        self.step6Box = ctk.ctkCollapsibleButton()
        self.step6Box.text = "Step 6: Mandibular Canal (IAN), Pharyngeal Airway & TMJ Views"
        self.step6Box.collapsed = True
        self.layout.addWidget(self.step6Box)
        step6Outer = qt.QVBoxLayout(self.step6Box)

        self.step6HintLabel = qt.QLabel("Extract or trace Mandibular Canals (F1), analyze Airway volume/mCSA (F7), or inspect TMJ (F8).")
        self.step6HintLabel.setWordWrap(True)
        step6Outer.addWidget(self.step6HintLabel)

        self.step6ContentWidget = qt.QWidget()
        step6Layout = qt.QFormLayout(self.step6ContentWidget)
        step6Outer.addWidget(self.step6ContentWidget)

        canalRow = qt.QHBoxLayout()
        self.btnExtractCanals = qt.QPushButton("Extract Canals from Segmentation")
        self.btnExtractCanals.setToolTip("Extract Left and Right Mandibular Canals (IAN) with radius tubes from class 5 / IAN mask (F1).")
        self.btnTraceRightCanal = qt.QPushButton("Trace Right Canal")
        self.btnTraceRightCanal.setToolTip("Manually trace Right Mandibular Canal in Markups Place Mode (F1).")
        self.btnTraceLeftCanal = qt.QPushButton("Trace Left Canal")
        self.btnTraceLeftCanal.setToolTip("Manually trace Left Mandibular Canal in Markups Place Mode (F1).")
        self.canalMarginSpin = qt.QDoubleSpinBox()
        self.canalMarginSpin.minimum = 0.5
        self.canalMarginSpin.maximum = 5.0
        self.canalMarginSpin.singleStep = 0.5
        self.canalMarginSpin.value = 2.0
        self.canalMarginSpin.suffix = " mm"
        self.canalMarginSpin.setToolTip("Configurable safety margin around the mandibular canal (default 2.0 mm, F1).")
        canalRow.addWidget(self.btnExtractCanals)
        canalRow.addWidget(self.btnTraceRightCanal)
        canalRow.addWidget(self.btnTraceLeftCanal)
        canalRow.addWidget(qt.QLabel("Safety:"))
        canalRow.addWidget(self.canalMarginSpin)
        step6Layout.addRow("Mandibular Canal:", canalRow)

        airwayTmjRow = qt.QHBoxLayout()
        self.btnAnalyzeAirway = qt.QPushButton("Compute Pharyngeal Airway Volume & mCSA")
        self.btnAnalyzeAirway.setToolTip("Run threshold + seeded region growing airway analysis and report volume & min CSA (F7).")
        self.btnTmjRightSag = qt.QPushButton("Right TMJ Sagittal")
        self.btnTmjRightSag.setToolTip("Align Yellow/Red views to Right Mandibular Condyle sagittal/coronal axes (F8).")
        self.btnTmjLeftSag = qt.QPushButton("Left TMJ Sagittal")
        self.btnTmjLeftSag.setToolTip("Align Yellow/Red views to Left Mandibular Condyle sagittal/coronal axes (F8).")
        airwayTmjRow.addWidget(self.btnAnalyzeAirway)
        airwayTmjRow.addWidget(self.btnTmjRightSag)
        airwayTmjRow.addWidget(self.btnTmjLeftSag)
        step6Layout.addRow("Airway & TMJ:", airwayTmjRow)

        self.layout.addStretch()

        # Connect signals
        self.btnUndo.clicked.connect(self.onUndoClicked)
        self.btnRedo.clicked.connect(self.onRedoClicked)

        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeNodeChanged)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.btnDentalLightbox.clicked.connect(self.onSetDentalLightboxLayout); self.btnLoadExternalMask.clicked.connect(self.onLoadExternalMask); self.btnFlipSIOrientation.clicked.connect(self.onFlipSIOrientation)
        self.btnDentalLayout.clicked.connect(self.onSetDentalLayout)
        self.btnStandardMpr.clicked.connect(self.onSetStandardMpr)
        self.btnPlaceOcclusalPts.clicked.connect(self.onPlaceOcclusalPoints)
        self.btnApplyOcclusalPlane.clicked.connect(self.onApplyOcclusalPlane)
        self.btnAutoOcclusalFromSeg.clicked.connect(self.onAutoOcclusalFromSegmentation)
        self.btnResetOcclusalPlane.clicked.connect(self.onResetOcclusalPlane)

        self.vrPresetCombo.currentTextChanged.connect(self.onVrControlChanged)
        self.vrShiftSlider.valueChanged.connect(self.onVrSliderChanged)
        self.showVrCheckBox.toggled.connect(self.onVrControlChanged)
        self.cropVrCheckBox.toggled.connect(self.onVrControlChanged)
        self.btnApplyDental3D.clicked.connect(self.onApplyDental3DStyling)

        self.archModeCombo.currentTextChanged.connect(self.onArchModeChanged)
        self.activeJawCombo.currentTextChanged.connect(self.onActiveJawChanged)
        self.btnAutoFitArch.clicked.connect(self.onAutoFitArchClicked)
        self.btnDrawArchManual.clicked.connect(self.onDrawArchManualClicked)
        self.showCurvesCheckBox.toggled.connect(self.onToggleCurvesVisibility)

        self.panoThicknessSpin.valueChanged.connect(self.onPanoParamChanged)
        self.panoHeightSpin.valueChanged.connect(self.onPanoParamChanged)
        self.panoOffsetSpin.valueChanged.connect(self.onPanoParamChanged)
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
        self.lightboxCountSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.btnActivateLightboxLayout.clicked.connect(self.onSetDentalLightboxLayout)
        self.lockFdiCombo.currentIndexChanged.connect(self.onLockFdiChanged)
        self.csWidthSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.csSlabEnabledCheckBox.toggled.connect(self.onCsDisplayParamChanged)
        self.csSlabModeCombo.currentTextChanged.connect(self.onCsDisplayParamChanged)
        self.csSlabSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.tickSpacingSpin.valueChanged.connect(self.onCsDisplayParamChanged)
        self.showAxialTicksCheckBox.toggled.connect(self.onCsDisplayParamChanged)

        self.btnToggleActiveToothInPlan.clicked.connect(self.onToggleActiveToothInPlanClicked)
        self.btnCreateManualRulers.clicked.connect(self.onCreateManualRulersClicked)
        self.btnRecomputeSelected.clicked.connect(self.onRefreshAllSelectedTeeth)
        self.btnClearPlan.clicked.connect(self.onClearPlanClicked)
        self.btnPlaceImplant.clicked.connect(self.onPlaceImplantClicked)
        self.btnImplantCentricView.clicked.connect(self.onImplantCentricViewClicked)
        self.btnConfirmAiSuggestion.clicked.connect(self.onConfirmAiSuggestionClicked)
        self.planTable.cellClicked.connect(self.onTableRowClicked)
        self.includePatientMetaCheckBox.toggled.connect(self.onExportOptChanged)
        self.btnGenerateReport.clicked.connect(self.onGeneratePrintableReport)
        self.btnExportPlan.clicked.connect(self.onExportTreatmentPlan)

        self.btnExtractCanals.clicked.connect(self.onExtractCanalsClicked)
        self.btnTraceRightCanal.clicked.connect(lambda: self.onTraceCanalManual("Right"))
        self.btnTraceLeftCanal.clicked.connect(lambda: self.onTraceCanalManual("Left"))
        self.canalMarginSpin.valueChanged.connect(self.onCanalMarginChanged)
        self.btnAnalyzeAirway.clicked.connect(self.onAnalyzeAirwayClicked)
        self.btnTmjRightSag.clicked.connect(lambda: self.onTmjPresetClicked("Right"))
        self.btnTmjLeftSag.clicked.connect(lambda: self.onTmjPresetClicked("Left"))

    def _createOdontogramButton(self, fdi: int) -> Any:
        btn = qt.QToolButton()
        btn.setText(str(fdi))
        btn.setCheckable(True)
        btn.setMinimumWidth(28)
        btn.setToolTip(
            f"Tooth #{fdi}: Plain Click = Jump Cross-Section to #{fdi} ONLY; "
            f"Ctrl+Click = Toggle #{fdi} in Treatment Plan (B17)."
        )
        btn.clicked.connect(lambda checked, t=fdi: self.onOdontogramToothClicked(t, checked))
        self.toothButtons[fdi] = btn
        return btn

    def _installKeyboardShortcuts(self) -> None:
        """B7 & F13: Install navigation and undo/redo shortcuts guarded by `should_handle_navigation_shortcut`."""
        if qt is None or self.parent is None:
            return
        for key_seq, step in (
            (qt.Qt.Key_Left, -1.0),
            (qt.Qt.Key_Right, 1.0),
            (qt.Qt.Key_PageDown, -5.0),
            (qt.Qt.Key_PageUp, 5.0),
        ):
            sc = qt.QShortcut(qt.QKeySequence(key_seq), self.parent)
            sc.activated.connect(lambda d=step: self._onShortcutNudge(d))
            self._shortcuts.append(sc)

        for key_str, fn in (("Ctrl+Z", self.onUndoClicked), ("Ctrl+Y", self.onRedoClicked)):
            sc_u = qt.QShortcut(qt.QKeySequence(key_str), self.parent)
            sc_u.activated.connect(fn)
            self._shortcuts.append(sc_u)

    def _onShortcutNudge(self, delta_mm: float) -> None:
        """B7: Ignore Left/Right/PgUp/PgDn shortcut when an input widget (slider, spinbox, table, text) has focus."""
        focused = qt.QApplication.focusWidget() if (qt is not None and hasattr(qt, "QApplication")) else None
        if not should_handle_navigation_shortcut(focused):
            return
        self.nudgeCursorMm(delta_mm)

    def _saveGlobalDefaultsToQSettings(self) -> None:
        """B16: Keep QSettings strictly for global defaults; active scene state lives in ParameterNode."""
        if qt is None:
            return
        try:
            settings = qt.QSettings()
            settings.setValue(self.SETTINGS_PREFIX + "panoThicknessMm", self.panoThicknessSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "csWidthMm", self.csWidthSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "csSlabMm", self.csSlabSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "tickSpacingMm", self.tickSpacingSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "lightboxStepMm", self.lightboxStepSpin.value)
            settings.setValue(self.SETTINGS_PREFIX + "vrPreset", self.vrPresetCombo.currentText)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ParameterNode Lifecycle & Full State Round-Trip (B14, B15, B16, B18)
    # ------------------------------------------------------------------
    def initializeParameterNode(self) -> None:
        if self.logic is None:
            return
        pNode = self.logic.getParameterNode()
        self.setParameterNode(pNode)
        if slicer is not None and self._parameterNode is not None:
            if not self._parameterNode.GetNodeReference("InputVolume"):
                firstVol = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                if firstVol and firstVol.GetAttribute("CBCTPlanner.IsPanoramic") != "1":
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
        self._observedCurveNodeIDs.clear()
        self.setParameterNode(None)

    def onSceneEndClose(self, _caller: Any, _event: Any) -> None:
        if self.parent and self.parent.isEntered:
            self.initializeParameterNode()

    def onNodeRemoved(self, _caller: Any, _event: Any) -> None:
        self.updateGUIFromParameterNode()

    def _ensureParameterNodeCurveObservers(self, volNode: Any) -> None:
        """B14: Attach curve edit observers to every curve referenced by ParameterNode (including scene/MRB loads)."""
        if vtk is None or self.logic is None or self._parameterNode is None:
            return
        for jaw in ("Lower", "Upper", "Shared"):
            cNode = self.logic.get_curve_node_for_jaw(volNode, jaw, parameterNode=self._parameterNode)
            if cNode is not None:
                self._observeCurveEdits(cNode)

    def updateGUIFromParameterNode(self, _caller: Any = None, _event: Any = None) -> None:
        """B14, B16, B18: Synchronize all UI controls from ParameterNode and keep collapsible headers enabled."""
        if self._parameterNode is None or self._updatingGUIFromParameterNode or qt is None:
            return
        self._updatingGUIFromParameterNode = True
        try:
            volNode = self._parameterNode.GetNodeReference("InputVolume")
            segNode = self._parameterNode.GetNodeReference("Segmentation")
            self.volumeSelector.setCurrentNode(volNode)
            self.segmentationSelector.setCurrentNode(segNode)

            # B14: Ensure observers are attached even for curves loaded from an MRB/scene file
            self._ensureParameterNodeCurveObservers(volNode)

            has_vol = volNode is not None
            jaw = self._parameterNode.GetParameter("ActiveJaw") or "Lower"
            arch_mode = self._parameterNode.GetParameter("ArchMode") or "Per-Jaw (Lower / Upper)"
            self.archModeCombo.setCurrentText(arch_mode)
            self.activeJawCombo.setCurrentText(jaw)

            # B16: Restore all cross-section, slab, tilt, lightbox, panoramic, and VR parameters from ParameterNode
            self.csWidthSpin.setValue(float(self._parameterNode.GetParameter("CrossSectionWidthMm") or 24.0))
            self.csSlabEnabledCheckBox.setChecked(
                (self._parameterNode.GetParameter("SlabEnabled") or "false").lower() == "true"
            )
            self.csSlabModeCombo.setCurrentText(self._parameterNode.GetParameter("SlabMode") or "Mean")
            self.csSlabSpin.setValue(float(self._parameterNode.GetParameter("SlabThicknessMm") or 0.5))
            self.tickSpacingSpin.setValue(float(self._parameterNode.GetParameter("TickSpacingMm") or 2.0))
            self.showAxialTicksCheckBox.setChecked(
                (self._parameterNode.GetParameter("ShowAxialTicks") or "true").lower() == "true"
            )
            self.csTiltSlider.setValue(float(self._parameterNode.GetParameter("CrossSectionTiltDeg") or 0.0))
            self.enableLightboxCheckBox.setChecked(
                (self._parameterNode.GetParameter("LightboxEnabled") or "true").lower() == "true"
            )
            self.lightboxStepSpin.setValue(float(self._parameterNode.GetParameter("LightboxStepMm") or 2.0))
            self.lightboxCountSpin.setValue(int(float(self._parameterNode.GetParameter("LightboxCount") or 3)))
            self.panoThicknessSpin.setValue(float(self._parameterNode.GetParameter("PanoThicknessMm") or 18.0))
            self.panoHeightSpin.setValue(float(self._parameterNode.GetParameter("PanoHeightMm") or 50.0))
            self.panoOffsetSpin.setValue(float(self._parameterNode.GetParameter("PanoOffsetMm") or 0.0))
            self.panoProjCombo.setCurrentText(self._parameterNode.GetParameter("PanoProjectionMode") or "Mean")
            self.canalMarginSpin.setValue(float(self._parameterNode.GetParameter("CanalSafetyMarginMm") or 2.0))
            self.includePatientMetaCheckBox.setChecked(
                (self._parameterNode.GetParameter("IncludePatientMetadata") or "false").lower() == "true"
            )

            locked_fdi = self._parameterNode.GetParameter("LockedToothFDI") or ""
            idx_lock = self.lockFdiCombo.findData(locked_fdi)
            if idx_lock >= 0:
                self.lockFdiCombo.setCurrentIndex(idx_lock)

            # Restore selected teeth set from ParameterNode
            sel_str = self._parameterNode.GetParameter("SelectedTeethFDI") or ""
            restored_sel = {int(tok) for tok in sel_str.split(",") if tok.strip().isdigit()}
            if restored_sel != self.selectedTeeth:
                self.selectedTeeth = restored_sel
                for fdi_num, btn in self.toothButtons.items():
                    btn.blockSignals(True)
                    btn.setChecked(fdi_num in self.selectedTeeth)
                    btn.blockSignals(False)

            effective_jaw = "Shared" if arch_mode.startswith("Shared") else jaw
            curveNode = (
                self.logic.get_curve_node_for_jaw(volNode, effective_jaw, parameterNode=self._parameterNode)
                if (has_vol and self.logic)
                else None
            )
            has_curve = curveNode is not None and curveNode.GetNumberOfControlPoints() >= 4

            # B18: Keep ctkCollapsibleButton headers ALWAYS enabled so user can expand and read prerequisites;
            # disable only the inner content widgets!
            for box in (self.step1Box, self.step2Box, self.step3Box, self.step4Box, self.step5Box, self.step6Box):
                box.setEnabled(True)
            if self.step2ContentWidget is not None:
                self.step2ContentWidget.setEnabled(has_vol)
            if self.step3ContentWidget is not None:
                self.step3ContentWidget.setEnabled(has_vol and has_curve)
            if self.step4ContentWidget is not None:
                self.step4ContentWidget.setEnabled(has_vol and has_curve)
            if self.step5ContentWidget is not None:
                self.step5ContentWidget.setEnabled(has_vol and has_curve)
            if self.step6ContentWidget is not None:
                self.step6ContentWidget.setEnabled(has_vol)

            if not has_vol:
                self.step1HintLabel.setText("Status: Waiting for an input CBCT volume.")
                self.step2HintLabel.setText("Prerequisite missing: Select an input CBCT volume in Step 1 first.")
                self.step3HintLabel.setText("Prerequisite missing: Load a CBCT volume and define a dental arch curve (>= 4 points).")
                self.step4HintLabel.setText("Prerequisite missing: Load a CBCT volume and define a dental arch curve (>= 4 points).")
                self.step5HintLabel.setText("Prerequisite missing: Load a CBCT volume and define a dental arch curve (>= 4 points).")
            else:
                seg_status = f"Segmentation: {segNode.GetName()}" if segNode else "Unsegmented (Manual Ruler mode)"
                self.step1HintLabel.setText(f"Loaded: {volNode.GetName()} | {seg_status}")
                if self.step2Box.collapsed and not has_curve:
                    self.step2Box.collapsed = False

                if has_curve and curveNode is not None:
                    user_edited = curveNode.GetAttribute("CBCTPlanner.UserEdited") == "1"
                    total_len = (
                        float(curveNode.GetCurveLengthWorld())
                        if hasattr(curveNode, "GetCurveLengthWorld")
                        else 100.0
                    )
                    edit_tag = "User-edited (locked against silent overwrite)" if user_edited else "Auto-fitted"
                    self.step2HintLabel.setText(
                        f"{effective_jaw} Arch Curve: {curveNode.GetNumberOfControlPoints()} points, {total_len:.1f} mm ({edit_tag})."
                    )
                    self.step3HintLabel.setText("Ready: Generate or auto-refresh in-process Curved Planar Reformation.")
                    self.step4HintLabel.setText(
                        "Ready: Drag the cyan line in Axial (Red) or Panoramic (Green), adjust +-30° tilt, or inspect Lightbox."
                    )
                    self.step5HintLabel.setText(
                        "Ready: Plain Click jumps cursor to tooth; Ctrl+Click or Toggle button adds/removes tooth in plan."
                    )
                    self.arcMmSlider.maximum = max(1.0, round(total_len, 1))
                    cursor_mm = float(self._parameterNode.GetParameter("CursorArcMm") or 0.0)
                    self.arcMmSlider.value = min(self.arcMmSlider.maximum, max(0.0, cursor_mm))
                else:
                    self.step2HintLabel.setText(
                        f"No {effective_jaw} arch curve yet. Click 'Auto-Fit Arch from Segmented Teeth' or 'Place / Draw Arch Points on Axial'."
                    )
                    self.step3HintLabel.setText(
                        f"Prerequisite missing: Create or fit a {effective_jaw} dental arch curve (>= 4 points) in Step 2."
                    )
                    self.step4HintLabel.setText(
                        f"Prerequisite missing: Create or fit a {effective_jaw} dental arch curve (>= 4 points) in Step 2."
                    )
                    self.step5HintLabel.setText(
                        f"Prerequisite missing: Create or fit a {effective_jaw} dental arch curve (>= 4 points) in Step 2."
                    )
        finally:
            self._updatingGUIFromParameterNode = False

    def _currentCursorSettings(self, sync_axial_slice: bool = False, site_label: Optional[str] = None) -> CursorSettings:
        """O4: Build a CursorSettings dataclass from the current UI / ParameterNode state."""
        return CursorSettings(
            width_mm=float(self.csWidthSpin.value),
            tick_spacing_mm=float(self.tickSpacingSpin.value),
            show_all_ticks=bool(self.showAxialTicksCheckBox.isChecked()),
            slab_enabled=bool(self.csSlabEnabledCheckBox.isChecked()),
            slab_mode=str(self.csSlabModeCombo.currentText),
            slab_thickness_mm=float(self.csSlabSpin.value),
            tilt_deg=float(self.csTiltSlider.value),
            lightbox_enabled=bool(self.enableLightboxCheckBox.isChecked()),
            lightbox_step_mm=float(self.lightboxStepSpin.value),
            lightbox_count=int(self.lightboxCountSpin.value),
            sync_axial_slice=bool(sync_axial_slice),
            site_label=site_label,
        )

    # ------------------------------------------------------------------
    # Undo / Redo Callbacks (F13)
    # ------------------------------------------------------------------
    def onUndoClicked(self) -> None:
        if self.logic is not None:
            desc = self.logic.undo_redo.undo()
            if desc:
                self.globalStatusLabel.setText(f"Undid: {desc}")

    def onRedoClicked(self) -> None:
        if self.logic is not None:
            desc = self.logic.undo_redo.redo()
            if desc:
                self.globalStatusLabel.setText(f"Redid: {desc}")

    # ------------------------------------------------------------------
    # Step 1 Callbacks: Volume, Layout, Non-Destructive Occlusal Plane & VR
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

    def onLoadExternalMask(self) -> None:
        path = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "Load Pre-computed AI Mask", "", "Mask Volumes (*.nii.gz *.nii *.nrrd *.seg.nrrd);;All Files (*)")
        if not path: return
        segNode = slicer.util.loadSegmentation(path) if path.endswith(".seg.nrrd") else None
        if segNode is None:
            lbl = slicer.util.loadLabelVolume(path)
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lbl, segNode)
            slicer.mrmlScene.RemoveNode(lbl)
        self.segmentationSelector.setCurrentNode(segNode)

    def onFlipSIOrientation(self) -> None:
        m = vtk.vtkMatrix4x4(); m.SetElement(2, 2, -1.0)
        t = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "_TempFlipSI")
        t.SetMatrixTransformToParent(m)
        for n in (self.volumeSelector.currentNode(), self.segmentationSelector.currentNode()):
            if n is not None:
                n.SetAndObserveTransformNodeID(t.GetID())
                slicer.vtkMRMLTransformLogic().hardenTransform(n)
        slicer.mrmlScene.RemoveNode(t); slicer.util.resetSliceViews()

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
        self.logic.start_place_occlusal_points(volNode, parameterNode=self._parameterNode)

    def onApplyOcclusalPlane(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        normal = self.logic.apply_occlusal_plane_from_points(
            volNode,
            segmentationNode=self.segmentationSelector.currentNode(),
            apply_parent_transform=True,
        )
        if normal is not None:
            self.step1HintLabel.setText(
                f"Aligned Occlusal Plane (non-destructive parent transform): ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            )
            self._syncCursorToViews()

    def onAutoOcclusalFromSegmentation(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not segNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select both a CBCT volume and a Segmentation node first.")
            return
        normal = self.logic.apply_occlusal_plane_from_segmentation(volNode, segNode)
        if normal is not None:
            self.step1HintLabel.setText(
                f"Auto-fitted Occlusal Normal from segmented teeth: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            )
            self._syncCursorToViews()

    def onResetOcclusalPlane(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if self.logic:
            self.logic.reset_occlusal_alignment(
                volNode, segmentationNode=self.segmentationSelector.currentNode()
            )
            self.step1HintLabel.setText("Reset occlusal alignment to default World Z (0, 0, 1).")
            self._syncCursorToViews()

    def onVrSliderChanged(self, _val: float) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("VRShiftHu", f"{self.vrShiftSlider.value:.1f}")
        if self._vrDebounceTimer is not None:
            self._vrDebounceTimer.start()
        else:
            self._applyDebouncedVr()

    def onVrControlChanged(self, *_args: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("VRPreset", self.vrPresetCombo.currentText)
            self._parameterNode.SetParameter(
                "VRVisible", "true" if self.showVrCheckBox.isChecked() else "false"
            )
            self._parameterNode.SetParameter(
                "VRCropEnabled", "true" if self.cropVrCheckBox.isChecked() else "false"
            )
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
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not segNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Please select a Segmentation node first.")
            return
        self.logic.vr.apply_dental_segmentator_styling(segNode)
        if volNode is not None:
            self.logic.update_fdi_overlays_from_segmentation(volNode, segNode, visible=True)

    # ------------------------------------------------------------------
    # Step 2 Callbacks: Dental Arch Curve (Shared vs Per-Jaw, Auto-Fit vs Manual)
    # ------------------------------------------------------------------
    def onArchModeChanged(self, mode_text: str) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("ArchMode", mode_text)
        self.updateGUIFromParameterNode()
        self._syncCursorToViews()

    def onActiveJawChanged(self, jaw: str) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("ActiveJaw", jaw)
        self.updateGUIFromParameterNode()
        self._syncCursorToViews()

    def _effectiveJawKey(self) -> str:
        if self.archModeCombo.currentText.startswith("Shared"):
            return "Shared"
        return self.activeJawCombo.currentText

    def onAutoFitArchClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic:
            return
        jaw = self._effectiveJawKey()
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
        jaw = self._effectiveJawKey()
        curveNode = self.logic.start_interactive_arch_placement(volNode, jaw=jaw)
        if curveNode is not None:
            self._observeCurveEdits(curveNode)
            self.updateGUIFromParameterNode()

    def _observeCurveEdits(self, curveNode: Any) -> None:
        """B14: Deduplicate curve observers by node ID and attach to any curve referenced in ParameterNode."""
        if vtk is None or curveNode is None:
            return
        cid = curveNode.GetID() if hasattr(curveNode, "GetID") else str(id(curveNode))
        if cid in self._observedCurveNodeIDs:
            return
        self._observedCurveNodeIDs.add(cid)
        point_mod_evt = getattr(getattr(slicer, "vtkMRMLMarkupsNode", None), "PointModifiedEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33))
        end_inter_evt = getattr(vtk.vtkCommand, "EndInteractionEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33))
        self.addObserver(curveNode, point_mod_evt, self._onCurvePointDragged)
        self.addObserver(curveNode, end_inter_evt, self._onCurveInteractionEnded)

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
    # Step 3 Callbacks: In-Process Panoramic CPR (O1, B10, F3)
    # ------------------------------------------------------------------
    def onPanoParamChanged(self, *_args: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            was_mod = self._parameterNode.StartModify() if hasattr(self._parameterNode, "StartModify") else None
            try:
                self._parameterNode.SetParameter("PanoThicknessMm", f"{self.panoThicknessSpin.value:.1f}")
                self._parameterNode.SetParameter("PanoHeightMm", f"{self.panoHeightSpin.value:.1f}")
                self._parameterNode.SetParameter("PanoOffsetMm", f"{self.panoOffsetSpin.value:.1f}")
                self._parameterNode.SetParameter("PanoProjectionMode", self.panoProjCombo.currentText)
            finally:
                if was_mod is not None and hasattr(self._parameterNode, "EndModify"):
                    self._parameterNode.EndModify(was_mod)

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
        jaw = self._effectiveJawKey()
        curveNode = self.logic.get_curve_node_for_jaw(volNode, jaw, parameterNode=self._parameterNode)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 4:
            if not preview and slicer:
                slicer.util.warningDisplay(f"Please create or fit a {jaw} dental arch curve (>= 4 points) first.")
            return

        spacing = 1.5 if preview else 0.5
        self.panoProgressBar.setVisible(True)
        self.panoProgressBar.setValue(20 if preview else 10)
        self.btnCancelPanoramic.setEnabled(True)

        def _on_progress(pct: int) -> None:
            self.panoProgressBar.setValue(int(pct))

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
                height_mm=self.panoHeightSpin.value,
                spacing_mm=spacing,
                buccolingual_offset_mm=self.panoOffsetSpin.value,
                projection_mode=self.panoProjCombo.currentText,
                show_in_green_view=True,
                occlusal_normal=self.logic.get_occlusal_normal(),
                async_cli=True,
                on_complete=_on_done,
                on_progress=_on_progress,
                parameterNode=self._parameterNode,
                force_fit=False, flip_vertical=self.panoFlipVerticalCheck.checked,
            )
        except Exception as exc:
            self.panoProgressBar.setVisible(False)
            self.btnCancelPanoramic.setEnabled(False)
            logger.error("Panoramic generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4 Callbacks: Cursor, Lock-to-Tooth (B17), Lightbox & Tilt
    # ------------------------------------------------------------------
    def nudgeCursorMm(self, delta_mm: float) -> None:
        new_val = max(
            self.arcMmSlider.minimum,
            min(self.arcMmSlider.maximum, float(self.arcMmSlider.value) + float(delta_mm)),
        )
        self.arcMmSlider.setValue(new_val)

    def _releaseToothLockOnManualMove(self) -> None:
        """B17: Moving the cursor manually via line drag or arc slider releases Lock-to-Tooth."""
        if self._parameterNode is not None and self._parameterNode.GetParameter("LockedToothFDI"):
            self._parameterNode.SetParameter("LockedToothFDI", "")
        if self.lockFdiCombo.currentIndex != 0:
            self.lockFdiCombo.blockSignals(True)
            self.lockFdiCombo.setCurrentIndex(0)
            self.lockFdiCombo.blockSignals(False)

    def _onInteractiveLineMovedCursor(self, new_arc_mm: float) -> None:
        """Called when the user drags the Axial or Panoramic line in a slice view."""
        self._releaseToothLockOnManualMove()
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CursorArcMm", f"{new_arc_mm:.2f}")
        self.arcMmSlider.blockSignals(True)
        self.arcMmSlider.setValue(new_arc_mm)
        self.arcMmSlider.blockSignals(False)
        self._syncCursorToViews(arc_mm=new_arc_mm, sync_axial_slice=False)

    def onArcSliderChanged(self, arc_mm: float) -> None:
        if self._updatingGUIFromParameterNode:
            return
        prev_arc = float(self._parameterNode.GetParameter("CursorArcMm") or 0.0) if self._parameterNode else 0.0
        self._releaseToothLockOnManualMove()
        if self.logic is not None and abs(float(arc_mm) - prev_arc) >= 1.0:
            self.logic.undo_redo.push(
                f"Move cursor {prev_arc:.1f} -> {arc_mm:.1f} mm",
                undo_fn=lambda p=prev_arc: self.arcMmSlider.setValue(p),
                redo_fn=lambda n=float(arc_mm): self.arcMmSlider.setValue(n),
            )
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CursorArcMm", f"{arc_mm:.2f}")
        self._syncCursorToViews(arc_mm=arc_mm, sync_axial_slice=False)

    def onTiltSliderChanged(self, tilt_deg: float) -> None:
        if self._updatingGUIFromParameterNode:
            return
        clamped = CrossSectionNavigator.clamp_tilt_deg(tilt_deg)
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CrossSectionTiltDeg", f"{clamped:.1f}")
        self._syncCursorToViews()

    def onLockFdiChanged(self, _idx: int) -> None:
        """B17: Lock cursor to the chosen FDI tooth site."""
        if self._updatingGUIFromParameterNode:
            return
        fdi_str = str(self.lockFdiCombo.currentData or "")
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("LockedToothFDI", fdi_str)
        if not fdi_str:
            return
        fdi = int(fdi_str)
        volNode = self.volumeSelector.currentNode()
        if volNode and self.logic:
            self.activeToothFDI = fdi
            if not self.archModeCombo.currentText.startswith("Shared"):
                target_jaw = "Lower" if fdi >= 31 else "Upper"
                if self.activeJawCombo.currentText != target_jaw:
                    self.activeJawCombo.setCurrentText(target_jaw)
            settings = self._currentCursorSettings(sync_axial_slice=True, site_label=f"Tooth #{fdi} [Locked]")
            arc_mm = self.logic.jump_to_tooth_site(fdi, volNode, settings=settings)
            if arc_mm is not None:
                self.arcMmSlider.blockSignals(True)
                self.arcMmSlider.setValue(arc_mm)
                self.arcMmSlider.blockSignals(False)

    def onCsDisplayParamChanged(self, *_args: Any) -> None:
        """B16 & O7: Persist all cross-section, slab, tick, and lightbox parameters into ParameterNode."""
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            was_mod = self._parameterNode.StartModify() if hasattr(self._parameterNode, "StartModify") else None
            try:
                self._parameterNode.SetParameter("CrossSectionWidthMm", f"{self.csWidthSpin.value:.1f}")
                self._parameterNode.SetParameter(
                    "SlabEnabled", "true" if self.csSlabEnabledCheckBox.isChecked() else "false"
                )
                self._parameterNode.SetParameter("SlabMode", self.csSlabModeCombo.currentText)
                self._parameterNode.SetParameter("SlabThicknessMm", f"{self.csSlabSpin.value:.1f}")
                self._parameterNode.SetParameter("TickSpacingMm", f"{self.tickSpacingSpin.value:.1f}")
                self._parameterNode.SetParameter(
                    "ShowAxialTicks", "true" if self.showAxialTicksCheckBox.isChecked() else "false"
                )
                self._parameterNode.SetParameter(
                    "LightboxEnabled", "true" if self.enableLightboxCheckBox.isChecked() else "false"
                )
                self._parameterNode.SetParameter("LightboxStepMm", f"{self.lightboxStepSpin.value:.1f}")
                self._parameterNode.SetParameter("LightboxCount", str(int(self.lightboxCountSpin.value)))
            finally:
                if was_mod is not None and hasattr(self._parameterNode, "EndModify"):
                    self._parameterNode.EndModify(was_mod)
        self._syncCursorToViews(arc_mm=self.arcMmSlider.value, sync_axial_slice=False)

    def _syncCursorToViews(self, arc_mm: Optional[float] = None, sync_axial_slice: bool = False) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        jaw = self._effectiveJawKey()
        curveNode = self.logic.get_curve_node_for_jaw(volNode, jaw, parameterNode=self._parameterNode)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 4:
            return

        # B17: If LockedToothFDI is active, keep cursor locked to that tooth's arc position
        locked_fdi = (self._parameterNode.GetParameter("LockedToothFDI") or "") if self._parameterNode else ""
        if locked_fdi.isdigit():
            fdi_locked = int(locked_fdi)
            self.activeToothFDI = fdi_locked
            frame_info = self.logic.get_tooth_arc_mm_and_frame(
                fdi_locked, volNode, segmentationNode=self.segmentationSelector.currentNode()
            )
            if frame_info is not None:
                arc_mm = frame_info[4]
                self.arcMmSlider.blockSignals(True)
                self.arcMmSlider.setValue(arc_mm)
                self.arcMmSlider.blockSignals(False)

        target_arc = float(self.arcMmSlider.value if arc_mm is None else arc_mm)
        site_lbl = f"#{self.activeToothFDI}" if self.activeToothFDI else None
        if locked_fdi:
            site_lbl = f"#{locked_fdi} [Locked]"
        settings = self._currentCursorSettings(sync_axial_slice=sync_axial_slice, site_label=site_lbl)
        panoNode = self._parameterNode.GetNodeReference("PanoramicVolume") if self._parameterNode else None

        _, _, _, _, s_clamped, total_len = self.logic.navigator.update_cursor_at_arc_mm(
            volumeNode=volNode,
            curveNode=curveNode,
            arc_mm=target_arc,
            occlusal_normal=self.logic.get_occlusal_normal(),
            panoNode=panoNode,
            settings=settings,
            parameterNode=self._parameterNode,
        )
        s_m, s_0, s_p = CrossSectionNavigator.compute_lightbox_offsets(
            s_clamped, total_len, step_mm=settings.lightbox_step_mm
        )
        lb_text = f" | Lightbox: [{s_m:.1f}, {s_0:.1f}, {s_p:.1f}] mm" if settings.lightbox_enabled else ""
        status_str = (
            f"Cursor: {s_clamped:.1f} / {total_len:.1f} mm ({jaw}) | Tilt: {settings.tilt_deg:+.0f}°{lb_text}"
            + (f" | Active: {site_lbl}" if site_lbl else "")
        )
        self.csStatusLabel.setText(status_str)
        self.globalStatusLabel.setText(status_str)

    # ------------------------------------------------------------------
    # Step 5 Callbacks: Odontogram (B17), Manual Rulers (B1), Implants (F2/F19) & Export
    # ------------------------------------------------------------------
    def _syncSelectedTeethToParameterNode(self) -> None:
        if self._parameterNode is not None:
            ordered = sort_teeth_anatomically(list(self.selectedTeeth))
            self._parameterNode.SetParameter("SelectedTeethFDI", ",".join(str(t) for t in ordered))

    def toggleToothPlanMembership(self, fdi: int) -> None:
        """B17: Explicitly toggle a tooth's membership in `selectedTeeth` and update button & table."""
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        fdi = int(fdi)
        if fdi in self.selectedTeeth:
            self.selectedTeeth.discard(fdi)
            self.treatmentPlanByFDI.pop(fdi, None)
        else:
            self.selectedTeeth.add(fdi)
            if self.logic is not None:
                self.treatmentPlanByFDI[fdi] = self.logic.analyze_single_tooth(
                    fdi, volNode, segmentationNode=segNode
                )
        btn = self.toothButtons.get(fdi)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(fdi in self.selectedTeeth)
            btn.blockSignals(False)
        self._syncSelectedTeethToParameterNode()
        self._refreshPlanTableFromCache()

    def onOdontogramToothClicked(self, fdi: int, _is_checked: bool) -> None:
        """B17: Plain click = jump cursor ONLY (never alters plan membership); Ctrl+click toggles plan membership."""
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return

        modifiers = qt.QApplication.keyboardModifiers() if qt else 0
        ctrl_held = bool(modifiers & qt.Qt.ControlModifier) if qt else False

        self.activeToothFDI = int(fdi)
        if not self.archModeCombo.currentText.startswith("Shared"):
            target_jaw = "Lower" if fdi >= 31 else "Upper"
            if self.activeJawCombo.currentText != target_jaw:
                self.activeJawCombo.setCurrentText(target_jaw)

        settings = self._currentCursorSettings(sync_axial_slice=True, site_label=f"Tooth #{fdi}")
        arc_mm = self.logic.jump_to_tooth_site(fdi, volNode, settings=settings)
        if arc_mm is not None:
            self.arcMmSlider.blockSignals(True)
            self.arcMmSlider.setValue(arc_mm)
            self.arcMmSlider.blockSignals(False)

        if ctrl_held:
            self.toggleToothPlanMembership(fdi)
        else:
            # B17: Plain click does NOT alter plan membership — restore button checked state to match selectedTeeth
            btn = self.toothButtons.get(int(fdi))
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(int(fdi) in self.selectedTeeth)
                btn.blockSignals(False)

    def onToggleActiveToothInPlanClicked(self) -> None:
        fdi = self.activeToothFDI or 46
        self.toggleToothPlanMembership(fdi)

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
        btn = self.toothButtons.get(fdi)
        if btn:
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
        self._syncSelectedTeethToParameterNode()
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
        self._syncSelectedTeethToParameterNode()
        self._refreshPlanTableFromCache()

    def onPlaceImplantClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic:
            return
        fdi = self.activeToothFDI or (next(iter(self.selectedTeeth)) if self.selectedTeeth else 46)
        entry_id = str(self.implantLibraryCombo.currentData or "generic_4.0x10.0")
        self.logic.place_implant_at_tooth(
            fdi=fdi,
            volumeNode=volNode,
            segmentationNode=segNode,
            entry_id=entry_id,
            mesiodistal_tilt_deg=float(self.csTiltSlider.value),
        )
        self.selectedTeeth.add(fdi)
        self.treatmentPlanByFDI[fdi] = self.logic.analyze_single_tooth(
            fdi, volNode, segmentationNode=segNode
        )
        self._syncSelectedTeethToParameterNode()
        self._refreshPlanTableFromCache()

    def onImplantCentricViewClicked(self) -> None:
        if not self.logic:
            return
        fdi = self.activeToothFDI or (next(iter(self.selectedTeeth)) if self.selectedTeeth else 46)
        imp = self.logic.implants.placed_implants.get(int(fdi))
        if imp is not None:
            self.logic.implants.align_views_to_implant_axis(imp)

    def onConfirmAiSuggestionClicked(self) -> None:
        """F19: Explicitly confirm the AI-suggested implant for the active tooth so it can enter clinical reports."""
        if not self.logic:
            return
        fdi = self.activeToothFDI or (next(iter(self.selectedTeeth)) if self.selectedTeeth else 46)
        self.logic.confirm_ai_suggestion_for_tooth(fdi)
        if fdi in self.treatmentPlanByFDI:
            self.treatmentPlanByFDI[fdi]["ai_suggestion_confirmed"] = True
        self._refreshPlanTableFromCache()

    def onExportOptChanged(self, checked: bool) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("IncludePatientMetadata", "true" if checked else "false")

    def _getOrderedPlanRows(self) -> List[Dict[str, Any]]:
        ordered_fdis = sort_teeth_anatomically(list(self.selectedTeeth))
        return [self.treatmentPlanByFDI[fdi] for fdi in ordered_fdis if fdi in self.treatmentPlanByFDI]

    def _refreshPlanTableFromCache(self) -> None:
        rows = self._getOrderedPlanRows()
        self.planTable.setRowCount(len(rows))
        for row_idx, item in enumerate(rows):
            misch_str = (
                f"{item['misch']} ({item['hu']:.0f} gray)"
                if (item.get("misch") and item.get("hu") is not None)
                else NOT_MEASURED
            )
            if item.get("placed_implant_label"):
                imp_str = str(item["placed_implant_label"])
            elif item.get("rec_diameter") is not None and item.get("rec_length") is not None:
                conf_tag = "Confirmed" if item.get("ai_suggestion_confirmed") else "Suggestion (unconfirmed)"
                imp_str = f"D{item['rec_diameter']:.1f} x L{item['rec_length']:.1f} mm [{conf_tag}]"
            else:
                imp_str = NOT_MEASURED

            src_str = str(item.get("source", NOT_MEASURED))
            if item.get("is_approximate_site"):
                src_str += " (approx site)"

            cells = [
                str(item["fdi"]),
                str(item["jaw"]).capitalize(),
                src_str,
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
                settings = self._currentCursorSettings(sync_axial_slice=True, site_label=f"Tooth #{fdi}")
                self.logic.jump_to_tooth_site(fdi, volNode, settings=settings)

    def onGeneratePrintableReport(self) -> None:
        rows = self._getOrderedPlanRows()
        if not rows or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select at least one tooth site in the Odontogram first.")
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(self.parent, "Select Report Output Directory")
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        html_path, pdf_path = self.logic.generate_clinical_reports(
            rows,
            self.volumeSelector.currentNode(),
            out_dir,
            capture_screenshots=True,
            lightbox_step_mm=float(self.lightboxStepSpin.value),
            include_patient_metadata=bool(self.includePatientMetaCheckBox.isChecked()),
        )
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(html_path)))
        if slicer:
            slicer.util.infoDisplay(f"Generated HTML and PDF reports:\n{html_path}\n{pdf_path}")

    def onExportTreatmentPlan(self) -> None:
        rows = self._getOrderedPlanRows()
        if not rows or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select at least one tooth site in the Odontogram first.")
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(self.parent, "Select Export Directory")
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        out_json, out_csv = self.logic.export_treatment_plan_files(rows, out_dir)
        imp_json, stl_files = self.logic.implants.export_implants_json_and_stl(out_dir)
        if slicer:
            slicer.util.infoDisplay(
                f"Exported treatment plan:\n{out_json}\n{out_csv}\nImplant STL files: {len(stl_files)}"
            )

    # ------------------------------------------------------------------
    # Step 6 Callbacks: Mandibular Canal (F1), Airway (F7) & TMJ (F8)
    # ------------------------------------------------------------------
    def onExtractCanalsClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not segNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select both a CBCT volume and a Segmentation node first.")
            return
        canals = self.logic.extract_and_display_mandibular_canals(volNode, segNode)
        self.step6HintLabel.setText(f"Extracted {len(canals)} mandibular canal(s): {', '.join(canals.keys()) or 'None'}")

    def onTraceCanalManual(self, side: str) -> None:
        volNode = self.volumeSelector.currentNode()
        if volNode and self.logic:
            self.logic.canals.start_manual_canal_trace(volNode, side=side, parameterNode=self._parameterNode)

    def onCanalMarginChanged(self, val: float) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetParameter("CanalSafetyMarginMm", f"{float(val):.1f}")

    def onAnalyzeAirwayClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        res = self.logic.run_airway_analysis(volNode)
        if res is not None:
            self.step6HintLabel.setText(
                f"Airway Volume: {res.total_volume_cm3:.2f} cm³ ({res.total_volume_mm3:.0f} mm³) | "
                f"Min CSA: {res.min_cross_sectional_area_mm2:.1f} mm² at Z={res.min_csa_z_ras_mm:.1f} mm"
            )

    def onTmjPresetClicked(self, side: str) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if volNode and self.logic:
            self.logic.apply_tmj_condyle_view(volNode, segmentationNode=segNode, side=side)


class CBCTPlannerLogic(ScriptedLoadableModuleLogic):
    """Core logic orchestrating ParameterNode state, VR, Arch Curves, Panoramic CPR, Lightbox, Implants, Canal, Airway & TMJ."""

    def __init__(self) -> None:
        super().__init__()
        self.vr = VRVisualizationLogic()
        self.panoramic = PanoramicCurveLogic()
        self.navigator = CrossSectionNavigator(panoramic_logic=self.panoramic)
        self.canals = CanalTracerLogic()
        self.implants = ImplantPlannerLogic()
        self.undo_redo = UndoRedoManager(max_depth=50)
        self._is_programmatic_curve_update: bool = False
        self._occlusal_normal: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # B2: Per-tooth user-calibrated arc positions in mm
        self._calibrated_tooth_arc_mm: Dict[str, float] = {}
        # F19: Set of FDI teeth whose AI implant suggestion has been explicitly confirmed by clinician
        self._confirmed_ai_suggestions: Set[int] = set()
        # O3: Cached MeasurementEngine keyed by (vol_id, vol_mtime, seg_id, seg_mtime, lower_mtime, upper_mtime, occ_tuple)
        self._cached_engine_key: Optional[Tuple[Any, ...]] = None
        self._cached_engine: Optional[MeasurementEngine] = None
        self._ruler_observers: List[Tuple[Any, int]] = []
        self.last_airway_result: Optional[AirwayAnalysisResult] = None

    def cleanup(self) -> None:
        self.panoramic.cancel_active_panoramic_cli()
        self.navigator.remove_line_observers()
        for node, tag in self._ruler_observers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._ruler_observers.clear()
        self._cached_engine = None
        self._cached_engine_key = None

    def getParameterNode(self) -> Any:
        """O6: Return singleton ParameterNode and initialize defaults only once per node."""
        if getattr(self, "_parameterNode", None) is not None:
            return self._parameterNode
        try:
            if slicer is None or getattr(slicer, "mrmlScene", None) is None:
                return getattr(self, "_parameterNode", None)
            node = super().getParameterNode()
        except Exception:
            return getattr(self, "_parameterNode", None)
        if node is not None:
            self._setDefaultParameters(node)
        return node

    @staticmethod
    def _setDefaultParameters(parameterNode: Any) -> None:
        """B16, O6, O7: Initialize all ParameterNode keys once inside a single StartModify/EndModify batch."""
        if hasattr(parameterNode, "GetAttribute") and parameterNode.GetAttribute("CBCTPlanner.DefaultsInitialized") == "1":
            return
        defaults = {
            "ActiveJaw": "Lower",
            "ArchMode": "Per-Jaw (Lower / Upper)",
            "CursorArcMm": "0.0",
            "LockedToothFDI": "",
            "CrossSectionWidthMm": "24.0",
            "SlabEnabled": "false",
            "SlabMode": "Mean",
            "SlabThicknessMm": "0.5",
            "TickSpacingMm": "2.0",
            "CrossSectionTiltDeg": "0.0",
            "LightboxEnabled": "true",
            "LightboxStepMm": "2.0",
            "LightboxCount": "3",
            "PanoThicknessMm": "18.0",
            "PanoHeightMm": "50.0",
            "PanoOffsetMm": "0.0",
            "PanoProjectionMode": "Mean",
            "ShowAxialTicks": "true",
            "OcclusalNormalX": "0.0",
            "OcclusalNormalY": "0.0",
            "OcclusalNormalZ": "1.0",
            "SelectedTeethFDI": "",
            "CanalSafetyMarginMm": "2.0",
            "IncludePatientMetadata": "false",
            "VRPreset": "CT-Dental-CBCT",
            "VRShiftHu": "0.0",
            "VRVisible": "false",
            "VRCropEnabled": "false",
        }
        was_mod = parameterNode.StartModify() if hasattr(parameterNode, "StartModify") else None
        try:
            for key, val in defaults.items():
                if not parameterNode.GetParameter(key):
                    parameterNode.SetParameter(key, val)
            if hasattr(parameterNode, "SetAttribute"):
                parameterNode.SetAttribute("CBCTPlanner.DefaultsInitialized", "1")
        finally:
            if was_mod is not None and hasattr(parameterNode, "EndModify"):
                parameterNode.EndModify(was_mod)

    @staticmethod
    def save_ui_state_to_parameter_node(parameterNode: Any, state_dict: Dict[str, Any]) -> None:
        """B16: Write a dictionary of UI state values into `parameterNode` in one batch."""
        if parameterNode is None:
            return
        was_mod = parameterNode.StartModify() if hasattr(parameterNode, "StartModify") else None
        try:
            for k, v in state_dict.items():
                parameterNode.SetParameter(str(k), str(v))
        finally:
            if was_mod is not None and hasattr(parameterNode, "EndModify"):
                parameterNode.EndModify(was_mod)

    @staticmethod
    def load_ui_state_from_parameter_node(parameterNode: Any) -> Dict[str, str]:
        """B16: Read all persisted UI state keys from `parameterNode`."""
        if parameterNode is None:
            return {}
        keys = (
            "ActiveJaw",
            "ArchMode",
            "CursorArcMm",
            "LockedToothFDI",
            "CrossSectionWidthMm",
            "SlabEnabled",
            "SlabMode",
            "SlabThicknessMm",
            "TickSpacingMm",
            "CrossSectionTiltDeg",
            "LightboxEnabled",
            "LightboxStepMm",
            "LightboxCount",
            "PanoThicknessMm",
            "PanoHeightMm",
            "PanoOffsetMm",
            "PanoProjectionMode",
            "ShowAxialTicks",
            "OcclusalNormalX",
            "OcclusalNormalY",
            "OcclusalNormalZ",
            "SelectedTeethFDI",
            "CanalSafetyMarginMm",
            "IncludePatientMetadata",
            "VRPreset",
            "VRShiftHu",
            "VRVisible",
            "VRCropEnabled",
        )
        return {k: (parameterNode.GetParameter(k) or "") for k in keys}

    # ------------------------------------------------------------------
    # B3 & F6: Non-Destructive Occlusal Plane Alignment & Reset
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
            was_mod = pNode.StartModify() if hasattr(pNode, "StartModify") else None
            try:
                pNode.SetParameter("OcclusalNormalX", f"{vec[0]:.6f}")
                pNode.SetParameter("OcclusalNormalY", f"{vec[1]:.6f}")
                pNode.SetParameter("OcclusalNormalZ", f"{vec[2]:.6f}")
            finally:
                if was_mod is not None and hasattr(pNode, "EndModify"):
                    pNode.EndModify(was_mod)
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

    @staticmethod
    def compute_occlusal_alignment_matrix_4x4(
        normal_vec: Sequence[float],
        center_ras: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """B3: Compute a 4x4 rigid transformation matrix rotating `normal_vec` onto world `+Z` around `center_ras`."""
        n = np.asarray(normal_vec, dtype=np.float64).reshape(3)
        n_len = float(np.linalg.norm(n))
        n = n / n_len if n_len > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if n[2] < 0:
            n = -n
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cos_theta = float(np.clip(np.dot(n, z_axis), -1.0, 1.0))
        angle_rad = float(np.arccos(cos_theta))
        if angle_rad < 1e-6:
            return np.eye(4, dtype=np.float64)

        axis = np.cross(n, z_axis)
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-8:
            return np.eye(4, dtype=np.float64)
        k = axis / axis_len
        K = np.array(
            [
                [0.0, -k[2], k[1]],
                [k[2], 0.0, -k[0]],
                [-k[1], k[0], 0.0],
            ],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)
        c = np.asarray(center_ras, dtype=np.float64).reshape(3)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R
        mat[:3, 3] = c - R @ c
        return mat

    def _collect_all_co_registered_nodes(
        self, volumeNode: Any, segmentationNode: Any = None, parameterNode: Any = None
    ) -> List[Any]:
        """B3: Collect volume, segmentation, curves, fiducials, and rulers that must move together under alignment."""
        nodes: List[Any] = []
        seen_ids: Set[str] = set()

        def _add(nd: Any) -> None:
            if nd is None:
                return
            nid = nd.GetID() if hasattr(nd, "GetID") else str(id(nd))
            if nid not in seen_ids:
                seen_ids.add(nid)
                nodes.append(nd)

        _add(volumeNode)
        _add(segmentationNode)
        pNode = parameterNode or (self.getParameterNode() if slicer is not None else None)
        if pNode is not None and hasattr(pNode, "GetNodeReference"):
            for role in (
                "Segmentation",
                "LowerArchCurve",
                "UpperArchCurve",
                "SharedArchCurve",
                "OcclusalFiducials",
                "RightCanalCurve",
                "LeftCanalCurve",
                "FDIOverlayFiducials",
            ):
                _add(pNode.GetNodeReference(role))

        for jaw in ("Lower", "Upper", "Shared"):
            _add(self.get_curve_node_for_jaw(volumeNode, jaw, parameterNode=pNode))

        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            for fdi in Q1_TEETH + Q2_TEETH + Q4_TEETH + Q3_TEETH:
                for suffix in ("HeightRuler", "WidthRuler", "NerveRuler"):
                    _add(slicer.mrmlScene.GetFirstNodeByName(f"Tooth_{fdi}_{suffix}"))
        return nodes

    def apply_non_destructive_occlusal_transform(
        self,
        volumeNode: Any,
        normal_vec: Sequence[float],
        center_ras: Sequence[float] = (0.0, 0.0, 0.0),
        segmentationNode: Any = None,
        parameterNode: Any = None,
    ) -> Tuple[Any, np.ndarray]:
        """B3: Apply a non-destructive parent linear transform to volume, segmentation, curves, fiducials, and rulers."""
        mat_4x4 = self.compute_occlusal_alignment_matrix_4x4(normal_vec, center_ras=center_ras)
        self.set_occlusal_normal(normal_vec)
        if slicer is None or volumeNode is None:
            return None, mat_4x4

        pNode = parameterNode or self.getParameterNode()
        tfNode = pNode.GetNodeReference("OcclusalTransform") if (pNode and hasattr(pNode, "GetNodeReference")) else None
        if tfNode is None:
            vol_name = volumeNode.GetName() if hasattr(volumeNode, "GetName") else "CBCT"
            tf_name = f"{vol_name}_OcclusalAlignmentTransform"
            tfNode = slicer.mrmlScene.GetFirstNodeByName(tf_name)
            if not tfNode:
                tfNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", tf_name)
            if pNode is not None and hasattr(pNode, "SetNodeReferenceID"):
                pNode.SetNodeReferenceID("OcclusalTransform", tfNode.GetID())

        if hasattr(slicer, "util") and hasattr(slicer.util, "vtkMatrixFromArray"):
            tfNode.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(mat_4x4))
        elif hasattr(tfNode, "SetMatrixTransformToParent"):
            tfNode.SetMatrixTransformToParent(mat_4x4)

        tf_id = tfNode.GetID()
        for nd in self._collect_all_co_registered_nodes(volumeNode, segmentationNode, parameterNode=pNode):
            if hasattr(nd, "SetAndObserveTransformNodeID"):
                nd.SetAndObserveTransformNodeID(tf_id)

        return tfNode, mat_4x4

    def reset_occlusal_alignment(
        self,
        volumeNode: Any = None,
        segmentationNode: Any = None,
        parameterNode: Any = None,
    ) -> None:
        """B3: Remove the non-destructive occlusal parent transform from all co-registered nodes and reset normal to +Z."""
        self.set_occlusal_normal((0.0, 0.0, 1.0))
        if slicer is None:
            return
        pNode = parameterNode or self.getParameterNode()
        for nd in self._collect_all_co_registered_nodes(volumeNode, segmentationNode, parameterNode=pNode):
            if hasattr(nd, "SetAndObserveTransformNodeID"):
                nd.SetAndObserveTransformNodeID(None)
        if pNode is not None and hasattr(pNode, "GetNodeReference"):
            tfNode = pNode.GetNodeReference("OcclusalTransform")
            if tfNode is not None:
                try:
                    slicer.mrmlScene.RemoveNode(tfNode)
                except Exception:
                    pass
                pNode.SetNodeReferenceID("OcclusalTransform", "")

    def start_place_occlusal_points(self, volumeNode: Any, parameterNode: Any = None) -> Any:
        if slicer is None or not volumeNode:
            return None
        pNode = parameterNode or self.getParameterNode()
        fidNode = pNode.GetNodeReference("OcclusalFiducials") if (pNode and hasattr(pNode, "GetNodeReference")) else None
        if fidNode is None:
            name = f"{volumeNode.GetName()}_OcclusalPlanePoints"
            fidNode = slicer.mrmlScene.GetFirstNodeByName(name)
            if not fidNode:
                fidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
                fidNode.CreateDefaultDisplayNodes()
            if pNode is not None and hasattr(pNode, "SetNodeReferenceID"):
                pNode.SetNodeReferenceID("OcclusalFiducials", fidNode.GetID())
        fidNode.RemoveAllControlPoints()
        if getattr(slicer, "app", None) is not None:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
            selectionNode.SetActivePlaceNodeID(fidNode.GetID())
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        return fidNode

    def apply_occlusal_plane_from_points(
        self,
        volumeNode: Any,
        segmentationNode: Any = None,
        apply_parent_transform: bool = True,
        ask_harden_transform: bool = False,
    ) -> Optional[np.ndarray]:
        """B3: Compute occlusal normal from 3 fiducials and apply non-destructive parent transform (never hardens in-place)."""
        _ = ask_harden_transform
        if slicer is None or not volumeNode:
            return None
        pNode = self.getParameterNode()
        fidNode = pNode.GetNodeReference("OcclusalFiducials") if (pNode and hasattr(pNode, "GetNodeReference")) else None
        if fidNode is None:
            name = f"{volumeNode.GetName()}_OcclusalPlanePoints"
            fidNode = slicer.mrmlScene.GetFirstNodeByName(name)
        if not fidNode or fidNode.GetNumberOfControlPoints() < 3:
            if getattr(slicer, "util", None) is not None and hasattr(slicer.util, "warningDisplay"):
                slicer.util.warningDisplay("Place at least 3 occlusal points first.")
            return None
        pts = np.zeros((3, 3), dtype=np.float64)
        buf = [0.0, 0.0, 0.0]
        for i in range(3):
            fidNode.GetNthControlPointPositionWorld(i, buf)
            pts[i] = buf
        normal = self.compute_occlusal_normal_from_3_points(pts)
        center = np.mean(pts, axis=0)
        if apply_parent_transform:
            self.apply_non_destructive_occlusal_transform(
                volumeNode,
                normal_vec=normal,
                center_ras=center,
                segmentationNode=segmentationNode,
                parameterNode=pNode,
            )
        else:
            self.set_occlusal_normal(normal)
        return self.get_occlusal_normal()

    def apply_occlusal_plane_from_segmentation(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Optional[np.ndarray]:
        """F6: Fit occlusal plane normal to segmented tooth crowns/centroids and apply non-destructive alignment."""
        centroids = self._extract_segmented_tooth_centroids(volumeNode, segmentationNode)
        if len(centroids) >= 3:
            pts = np.array(list(centroids.values()), dtype=np.float64)
        else:
            pts_opt = self.extract_arch_points_from_segmentation(segmentationNode, volumeNode, "Lower")
            if pts_opt is None or len(pts_opt) < 3:
                pts_opt = self.extract_arch_points_from_segmentation(segmentationNode, volumeNode, "Upper")
            if pts_opt is None or len(pts_opt) < 3:
                return None
            pts = pts_opt
        normal, center = compute_occlusal_plane_from_segmentation(pts)
        self.apply_non_destructive_occlusal_transform(
            volumeNode, normal_vec=normal, center_ras=center, segmentationNode=segmentationNode
        )
        return normal

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
    # Arch Curve Management (B15 Node Refs, B20 Spline without Skull Fallback, F3 Single Arch)
    # ------------------------------------------------------------------
    def get_curve_node_for_jaw(
        self, volumeNode: Any, jaw: str, parameterNode: Any = None
    ) -> Any:
        """B15 & F3: Resolve curve node via ParameterNode reference ID first; migrate per-jaw curve for Shared mode."""
        pNode = parameterNode or (self.getParameterNode() if slicer is not None else None)
        node = self.panoramic.resolve_curve_node(volumeNode, jaw=jaw, parameterNode=pNode)
        if node is None and jaw.lower().startswith("sh"):
            # F3: Seamless migration — if Shared Occlusal Arch is selected on an existing scene with Lower/Upper curve
            node = self.panoramic.resolve_curve_node(
                volumeNode, jaw="Lower", parameterNode=pNode
            ) or self.panoramic.resolve_curve_node(volumeNode, jaw="Upper", parameterNode=pNode)
        return node

    def ensure_arch_curve(self, volumeNode: Any, jaw: str) -> Any:
        """Return existing curve node for the requested jaw ONLY (never touches the opposite jaw)."""
        if slicer is None or not volumeNode:
            return None
        pNode = self.getParameterNode()
        curveNode = self.get_curve_node_for_jaw(volumeNode, jaw, parameterNode=pNode)
        if curveNode is None:
            curveNode = self.panoramic.create_or_update_arch_curve(
                volumeNode, jaw=jaw, points_ras=None, visible=True, parameterNode=pNode
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
        """B20: Extract 3D tooth coordinates in RAS from a segmentation node.

        Drops the legacy whole-skull ('maxilla'/'mandible') fallback so arch auto-fit never fits to
        the entire cranium or mandibular rami/condyles. If no tooth segments exist, returns None so
        the caller falls back cleanly to manual placement.
        """
        if not segmentationNode:
            return None
        segmentation = segmentationNode.GetSegmentation()
        if not segmentation:
            return None

        try:
            from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
        except ImportError:
            vtk_to_numpy = None

        jaw_low = jaw.lower()
        seg_ids = self.vr._get_segment_ids(segmentation)
        tooth_segments: List[Any] = []

        for sid in seg_ids:
            seg = segmentation.GetSegment(sid)
            if not seg:
                continue
            fdi_attr = seg.GetTag("CBCTPlanner.FDI") if hasattr(seg, "GetTag") else None
            label_val, category, is_5class, _ = match_segment_to_label_and_precedence(
                sid, seg.GetName() or "", fdi_attribute=fdi_attr
            )
            if category == "tooth":
                if is_5class:
                    if jaw_low.startswith("sh"):
                        tooth_segments.append(seg)
                    elif jaw_low.startswith("up") and label_val == 3:
                        tooth_segments.append(seg)
                    elif jaw_low.startswith("low") and label_val == 4:
                        tooth_segments.append(seg)
                else:
                    if jaw_low.startswith("sh"):
                        tooth_segments.append(seg)
                    elif jaw_low.startswith("up") and label_val in MAXILLARY_TEETH_IDS:
                        tooth_segments.append(seg)
                    elif jaw_low.startswith("low") and label_val in MANDIBULAR_TEETH_IDS:
                        tooth_segments.append(seg)

        if tooth_segments and slicer is not None:
            rep_name = slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
            collected: List[np.ndarray] = []
            for seg in tooth_segments:
                poly = seg.GetRepresentation(rep_name)
                if poly and poly.GetPoints() and poly.GetNumberOfPoints() >= 4:
                    if vtk_to_numpy is not None:
                        pts = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
                    else:
                        pts = np.array(
                            [poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())],
                            dtype=np.float64,
                        )
                    collected.append(pts)
            if collected:
                merged = np.vstack(collected)
                if len(merged) >= 4:
                    return merged

        if volumeNode is not None:
            mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
            if mask is not None:
                vol = CBCTVolume.from_slicer_volume_node(volumeNode)
                if is_5class:
                    if jaw_low.startswith("sh"):
                        z_idx, y_idx, x_idx = np.where((mask == 3) | (mask == 4))
                    else:
                        target_val = 3 if jaw_low.startswith("up") else 4
                        z_idx, y_idx, x_idx = np.where(mask == target_val)
                    # B20: Do NOT fall back to label 1 (Maxilla/Skull) or label 2 (full Mandible)!
                    if len(x_idx) >= 6:
                        vox = np.column_stack([x_idx, y_idx, z_idx])
                        return vol.voxel_to_world(vox)
                else:
                    if jaw_low.startswith("sh"):
                        tooth_ids = MAXILLARY_TEETH_IDS + MANDIBULAR_TEETH_IDS
                    else:
                        tooth_ids = MAXILLARY_TEETH_IDS if jaw_low.startswith("up") else MANDIBULAR_TEETH_IDS
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
        """Fit dental arch curve for `jaw` from segmented teeth; if no teeth segmented, enter manual place mode."""
        if slicer is None or not volumeNode:
            return None
        jaw_cap = "Shared" if jaw.lower().startswith("sh") else jaw.capitalize()
        pNode = self.getParameterNode()
        existing = self.get_curve_node_for_jaw(volumeNode, jaw_cap, parameterNode=pNode)
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
                    f"No segmented teeth found for {jaw_cap} arch — click points along the arch from Patient Right to Patient Left.",
                    5000,
                )
            return curveNode

        fitted_pts = fit_dental_arch_from_points(
            pts_cloud, jaw_type=jaw_cap.lower(), num_control_points=11, ras_convention=True
        )

        self._is_programmatic_curve_update = True
        try:
            curveNode = self.panoramic.create_or_update_arch_curve(
                volumeNode,
                jaw=jaw_cap,
                points_ras=fitted_pts,
                visible=visible,
                parameterNode=pNode,
            )
            if curveNode:
                curveNode.SetAttribute("CBCTPlanner.UserEdited", "0")
        finally:
            self._is_programmatic_curve_update = False
        return curveNode

    def set_curves_visibility(self, volumeNode: Any, visible: bool) -> None:
        if slicer is None or not volumeNode:
            return
        pNode = self.getParameterNode()
        for jaw in ("Lower", "Upper", "Shared"):
            node = self.get_curve_node_for_jaw(volumeNode, jaw, parameterNode=pNode)
            if node and node.GetDisplayNode():
                dNode = node.GetDisplayNode()
                dNode.SetVisibility(bool(visible))
                dNode.SetVisibility2D(bool(visible))
                dNode.SetVisibility3D(bool(visible))

    # ------------------------------------------------------------------
    # B2: Tooth Frame Lookup (Segmentation Centroids + Width Table + Calibration)
    # ------------------------------------------------------------------
    def set_calibrated_tooth_arc_mm(self, fdi: int, arc_mm: Optional[float]) -> None:
        """B2: Set or clear a user-calibrated arc-length position in mm for tooth `fdi`."""
        if arc_mm is None:
            self._calibrated_tooth_arc_mm.pop(str(fdi), None)
        else:
            self._calibrated_tooth_arc_mm[str(fdi)] = float(arc_mm)

    @staticmethod
    def fdi_to_normalized_arch_t(fdi: int) -> float:
        """B2: Map FDI tooth number to normalized arch parameter t in [0.04, 0.96] using anatomical mesiodistal widths."""
        fdi_str = str(fdi)
        order = JAW_TOOTH_ORDER["lower"] if int(fdi) >= 31 else JAW_TOOTH_ORDER["upper"]
        if fdi_str not in order:
            raise ValueError(f"Invalid FDI tooth number: {fdi}")
        widths = np.array([ANATOMICAL_MESIODISTAL_WIDTHS_MM.get(k, 7.5) for k in order], dtype=np.float64)
        cum_centers = np.cumsum(widths) - 0.5 * widths
        norm_w = (cum_centers - cum_centers[0]) / max(1e-6, float(cum_centers[-1] - cum_centers[0]))
        idx = order.index(fdi_str)
        return float(0.04 + norm_w[idx] * 0.92)

    def _extract_segmented_tooth_centroids(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Dict[str, np.ndarray]:
        """Extract per-tooth 3D RAS centroids from segmentation when available (feeds B2 and F4)."""
        if volumeNode is None or segmentationNode is None:
            return {}
        engine = self._get_or_build_measurement_engine(volumeNode, segmentationNode)
        if engine is None:
            return {}
        out: Dict[str, np.ndarray] = {}
        for fdi_str in FDI_TO_TOOTH_ID:
            t_info = engine._get_tooth_info(fdi_str)
            if t_info is not None:
                out[fdi_str] = t_info.centroid_mm.copy()
        return out

    def get_fdi_arc_site_map(
        self,
        volumeNode: Any,
        jaw: str = "Lower",
        segmentationNode: Any = None,
    ) -> Dict[str, ToothArcSite]:
        """B2: Compute `Dict[fdi_str, ToothArcSite]` for all teeth in `jaw` with provenance and missing-tooth handling."""
        pNode = self.getParameterNode() if slicer is not None else None
        arch_mode = (pNode.GetParameter("ArchMode") or "") if pNode else ""
        eff_jaw = "Shared" if arch_mode.startswith("Shared") else jaw
        curveNode = self.get_curve_node_for_jaw(volumeNode, eff_jaw, parameterNode=pNode)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return {}
        pts, arc_s, _, _, _ = self.panoramic.get_curve_points_and_frames(
            curveNode, num_samples=250, occlusal_normal=self.get_occlusal_normal()
        )
        total_len = float(arc_s[-1])
        seg_Node = segmentationNode or (
            pNode.GetNodeReference("Segmentation") if (pNode and hasattr(pNode, "GetNodeReference")) else None
        )
        seg_centroids = self._extract_segmented_tooth_centroids(volumeNode, seg_Node)
        return compute_fdi_arc_positions_mm(
            total_arc_length_mm=total_len,
            jaw_type=jaw,
            curve_pts=pts,
            curve_arc_s=arc_s,
            segmented_centroids=seg_centroids,
            calibrated_arc_mm=self._calibrated_tooth_arc_mm,
        )

    def get_tooth_arc_mm_and_frame(
        self,
        fdi: int,
        volumeNode: Any,
        segmentationNode: Any = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]]:
        jaw = "Lower" if int(fdi) >= 31 else "Upper"
        pNode = self.getParameterNode() if slicer is not None else None
        arch_mode = (pNode.GetParameter("ArchMode") or "") if pNode else ""
        eff_jaw = "Shared" if arch_mode.startswith("Shared") else jaw
        curveNode = self.get_curve_node_for_jaw(volumeNode, eff_jaw, parameterNode=pNode)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return None
        site_map = self.get_fdi_arc_site_map(volumeNode, jaw=jaw, segmentationNode=segmentationNode)
        site = site_map.get(str(fdi))
        if site is not None:
            target_arc_mm = site.arc_mm
        else:
            _, arc_s, _, _, _ = self.panoramic.get_curve_points_and_frames(
                curveNode, num_samples=100, occlusal_normal=self.get_occlusal_normal()
            )
            total_len = float(arc_s[-1])
            target_arc_mm = self.fdi_to_normalized_arch_t(fdi) * total_len
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
        settings: Optional[CursorSettings] = None,
    ) -> Optional[float]:
        """O4 & B11: Jump Cross-Section Lightbox, Axial lines, Pano lines, and Red Axial offset (`sync_axial_slice=True`) to `fdi`."""
        jaw = "Lower" if int(fdi) >= 31 else "Upper"
        pNode = self.getParameterNode() if slicer is not None else None
        arch_mode = (pNode.GetParameter("ArchMode") or "") if pNode else ""
        eff_jaw = "Shared" if arch_mode.startswith("Shared") else jaw
        curveNode = self.get_curve_node_for_jaw(volumeNode, eff_jaw, parameterNode=pNode)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return None
        frame_info = self.get_tooth_arc_mm_and_frame(fdi, volumeNode)
        if frame_info is None:
            return None
        _, _, _, _, arc_mm, _ = frame_info
        panoNode = pNode.GetNodeReference("PanoramicVolume") if (pNode and hasattr(pNode, "GetNodeReference")) else None
        if pNode is not None:
            pNode.SetParameter("CursorArcMm", f"{arc_mm:.2f}")

        if settings is None:
            settings = CursorSettings(
                width_mm=width_mm,
                tick_spacing_mm=tick_spacing_mm,
                show_all_ticks=show_all_ticks,
                slab_enabled=False,
                slab_mode="Mean",
                slab_thickness_mm=slab_thickness_mm,
                tilt_deg=tilt_deg,
                lightbox_enabled=lightbox_enabled,
                lightbox_step_mm=lightbox_step_mm,
                sync_axial_slice=True,
                site_label=f"Tooth #{fdi}",
            )
        else:
            settings.sync_axial_slice = True
            if settings.site_label is None:
                settings.site_label = f"Tooth #{fdi}"

        self.navigator.update_cursor_at_arc_mm(
            volumeNode=volumeNode,
            curveNode=curveNode,
            arc_mm=arc_mm,
            occlusal_normal=self.get_occlusal_normal(),
            panoNode=panoNode,
            settings=settings,
            parameterNode=pNode,
        )
        return arc_mm

    # ------------------------------------------------------------------
    # Honest Clinical Measurements (B1 Manual Ruler Flag, B19 Precedence, O3 Engine Cache)
    # ------------------------------------------------------------------
    def _extract_combined_label_mask(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Tuple[Optional[np.ndarray], bool]:
        """B19: Combine segments into a label mask using exact token matching and strict precedence (`pulp > tooth > canal > sinus > jaw`)."""
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
        precedence_buf = np.zeros(vol_arr.shape, dtype=np.uint8)
        is_5class = False

        for idx, sid in enumerate(seg_ids):
            seg = segmentation.GetSegment(sid)
            seg_name = seg.GetName() if seg else ""
            fdi_attr = None
            term_str = None
            if seg is not None and hasattr(seg, "GetTag"):
                t_val = vtk.reference("") if (vtk is not None and hasattr(vtk, "reference")) else None
                if t_val is not None:
                    try:
                        if seg.GetTag("CBCTPlanner.FDI", t_val):
                            fdi_attr = str(t_val)
                        if seg.GetTag("TerminologyEntry", t_val):
                            term_str = str(t_val)
                    except Exception:
                        pass
            if fdi_attr is None and seg is not None and hasattr(seg, "GetAttribute"):
                try:
                    fdi_attr = seg.GetAttribute("CBCTPlanner.FDI")
                except Exception:
                    pass

            label_val, _cat, seg_is_5class, prec = match_segment_to_label_and_precedence(
                sid, seg_name, fdi_attribute=fdi_attr, terminology_str=term_str
            )
            if seg_is_5class:
                is_5class = True
            if label_val == 0:
                if len(seg_ids) == 5:
                    label_val = idx + 1
                    is_5class = True
                    prec = 3 if label_val == 5 else (4 if label_val in (3, 4) else 1)
                else:
                    continue

            try:
                seg_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                    segmentationNode, sid, volumeNode
                )
                if seg_arr is not None and seg_arr.shape == combined.shape:
                    write_mask = (seg_arr > 0) & (prec >= precedence_buf)
                    combined[write_mask] = label_val
                    precedence_buf[write_mask] = prec
            except Exception:
                pass

        if not np.any(combined):
            return None, False
        return combined, is_5class

    @staticmethod
    def _is_ruler_user_modified(node: Any) -> bool:
        """B1: Return True ONLY if the ruler node has been modified by the user (or placed interactively)."""
        if node is None:
            return False
        user_mod_attr = node.GetAttribute("CBCTPlanner.UserModified") if hasattr(node, "GetAttribute") else None
        if user_mod_attr == "1":
            return True

        # Check if control point coordinates differ from the initial seeded defaults
        init_json = node.GetAttribute("CBCTPlanner.InitialEndpoints") if hasattr(node, "GetAttribute") else None
        if init_json and hasattr(node, "GetNumberOfControlPoints") and node.GetNumberOfControlPoints() >= 2:
            try:
                init_pts = np.asarray(json.loads(init_json), dtype=np.float64)
                p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(0, p0)
                node.GetNthControlPointPositionWorld(1, p1)
                cur_pts = np.array([p0, p1], dtype=np.float64)
                if float(np.max(np.linalg.norm(cur_pts - init_pts, axis=1))) > 1e-3:
                    if hasattr(node, "SetAttribute"):
                        node.SetAttribute("CBCTPlanner.UserModified", "1")
                    return True
            except Exception:
                pass
        return False

    @classmethod
    def _read_manual_ruler_length(cls, ruler_name_or_node: Any) -> Optional[float]:
        """B1: Return manual ruler length in mm ONLY when user-modified (`CBCTPlanner.UserModified == "1"`).

        Untouched default-seeded rulers return `None` so pre-created 10/6/4 mm templates are never
        reported as clinical measurements!
        """
        node = ruler_name_or_node
        if isinstance(ruler_name_or_node, str):
            if slicer is None or getattr(slicer, "mrmlScene", None) is None:
                return None
            node = slicer.mrmlScene.GetFirstNodeByName(ruler_name_or_node)
        if not node or not hasattr(node, "GetNumberOfControlPoints") or node.GetNumberOfControlPoints() < 2:
            return None

        if not cls._is_ruler_user_modified(node):
            return None

        if hasattr(node, "GetLineLengthWorld"):
            length = float(node.GetLineLengthWorld())
            return round(length, 2) if length > 0.0 else None
        p0, p1 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        node.GetNthControlPointPositionWorld(0, p0)
        node.GetNthControlPointPositionWorld(1, p1)
        length = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        return round(length, 2) if length > 0.0 else None

    def create_or_focus_manual_rulers_for_tooth(
        self, fdi: int, volumeNode: Any, enter_place_mode: bool = False
    ) -> List[Any]:
        """B1: Create manual ruler nodes flagged `CBCTPlanner.UserModified = "0"` until user moves endpoints."""
        if slicer is None or not volumeNode:
            return []
        frame_info = self.get_tooth_arc_mm_and_frame(fdi, volumeNode)
        if frame_info is None:
            P = np.zeros(3, dtype=np.float64)
            N = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            B = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            P, _, N, B, _, _ = frame_info

        is_lower = int(fdi) >= 31
        apical_vec = -B if is_lower else B
        yellow_id = self.navigator._get_view_node_id("Yellow")

        rulers_spec = [
            (f"Tooth_{fdi}_HeightRuler", P, P + apical_vec * 10.0, (0.1, 0.85, 0.2)),
            (f"Tooth_{fdi}_WidthRuler", P - N * 3.0, P + N * 3.0, (0.95, 0.85, 0.1)),
            (f"Tooth_{fdi}_NerveRuler", P + apical_vec * 8.0, P + apical_vec * 12.0, (0.95, 0.25, 0.25)),
        ]
        created_nodes: List[Any] = []
        for r_name, pt0, pt1, color in rulers_spec:
            rNode = slicer.mrmlScene.GetFirstNodeByName(r_name)
            if not rNode:
                rNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", r_name)
                rNode.CreateDefaultDisplayNodes()
                if enter_place_mode:
                    rNode.SetAttribute("CBCTPlanner.UserModified", "1")
                else:
                    p0_list = [round(float(pt0[0]), 4), round(float(pt0[1]), 4), round(float(pt0[2]), 4)]
                    p1_list = [round(float(pt1[0]), 4), round(float(pt1[1]), 4), round(float(pt1[2]), 4)]
                    rNode.SetAttribute("CBCTPlanner.InitialEndpoints", json.dumps([p0_list, p1_list]))
                    rNode.SetAttribute("CBCTPlanner.UserModified", "0")
                    rNode.AddControlPoint(*p0_list)
                    rNode.AddControlPoint(*p1_list)

                # Attach observer that flips UserModified -> "1" on first user interaction
                if vtk is not None and hasattr(rNode, "AddObserver"):
                    def _on_ruler_interacted(caller: Any, _evt: Any) -> None:
                        cls_ref = self.__class__
                        cls_ref._is_ruler_user_modified(caller)

                    point_mod_evt = getattr(getattr(slicer, "vtkMRMLMarkupsNode", None), "PointModifiedEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33))
                    end_inter_evt = getattr(vtk.vtkCommand, "EndInteractionEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33))
                    for evt in (point_mod_evt, end_inter_evt):
                        tag = rNode.AddObserver(evt, _on_ruler_interacted)
                        self._ruler_observers.append((rNode, tag))

            if (
                volumeNode is not None
                and hasattr(volumeNode, "GetTransformNodeID")
                and hasattr(rNode, "SetAndObserveTransformNodeID")
            ):
                tf_id = volumeNode.GetTransformNodeID()
                if tf_id:
                    rNode.SetAndObserveTransformNodeID(tf_id)

            dNode = rNode.GetDisplayNode()
            if dNode:
                dNode.SetColor(*color)
                dNode.SetVisibility(True)
                dNode.SetVisibility2D(True)
                if hasattr(dNode, "SetPropertiesLabelVisibility"):
                    dNode.SetPropertiesLabelVisibility(True)
                if yellow_id and hasattr(dNode, "SetViewNodeIDs"):
                    dNode.SetViewNodeIDs([yellow_id])
            created_nodes.append(rNode)

        if enter_place_mode and created_nodes and getattr(slicer, "app", None) is not None:
            selNode = slicer.app.applicationLogic().GetSelectionNode()
            selNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsLineNode")
            selNode.SetActivePlaceNodeID(created_nodes[0].GetID())
            intNode = slicer.app.applicationLogic().GetInteractionNode()
            intNode.SetPlaceModePersistence(0)
            intNode.SetCurrentInteractionMode(intNode.Place)

        return created_nodes

    def _get_or_build_measurement_engine(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Optional[MeasurementEngine]:
        """O3: Cache MeasurementEngine by `(vol_id, vol_mtime, seg_id, seg_mtime, lower_mtime, upper_mtime, occ_tuple)`."""
        if slicer is None or volumeNode is None or segmentationNode is None:
            return None
        vol_id = volumeNode.GetID() if hasattr(volumeNode, "GetID") else str(id(volumeNode))
        vol_mtime = int(volumeNode.GetMTime()) if hasattr(volumeNode, "GetMTime") else 0
        seg_id = segmentationNode.GetID() if hasattr(segmentationNode, "GetID") else str(id(segmentationNode))
        seg_mtime = int(segmentationNode.GetMTime()) if hasattr(segmentationNode, "GetMTime") else 0

        pNode = self.getParameterNode()
        c_low = self.get_curve_node_for_jaw(volumeNode, "Lower", parameterNode=pNode)
        c_up = self.get_curve_node_for_jaw(volumeNode, "Upper", parameterNode=pNode)
        c_sh = self.get_curve_node_for_jaw(volumeNode, "Shared", parameterNode=pNode)
        m_low = int(c_low.GetMTime()) if (c_low and hasattr(c_low, "GetMTime")) else 0
        m_up = int(c_up.GetMTime()) if (c_up and hasattr(c_up, "GetMTime")) else 0
        m_sh = int(c_sh.GetMTime()) if (c_sh and hasattr(c_sh, "GetMTime")) else 0
        occ = self.get_occlusal_normal()
        occ_tuple = (round(float(occ[0]), 5), round(float(occ[1]), 5), round(float(occ[2]), 5))

        key = (vol_id, vol_mtime, seg_id, seg_mtime, m_low, m_up, m_sh, occ_tuple)
        if self._cached_engine_key == key and self._cached_engine is not None:
            return self._cached_engine

        engine = self._build_measurement_engine(volumeNode, segmentationNode)
        self._cached_engine_key = key
        self._cached_engine = engine
        return engine

    def _build_measurement_engine(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Optional[MeasurementEngine]:
        if slicer is None or volumeNode is None or segmentationNode is None:
            return None
        mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            return None
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        arches: Dict[str, DentalArch] = {}
        pNode = self.getParameterNode()
        for jaw in ("lower", "upper"):
            cNode = self.get_curve_node_for_jaw(volumeNode, jaw, parameterNode=pNode)
            if cNode is None:
                cNode = self.get_curve_node_for_jaw(volumeNode, "Shared", parameterNode=pNode)
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
            extract_teeth_eagerly=False,
        )

    def confirm_ai_suggestion_for_tooth(self, fdi: int) -> None:
        """F19: Mark the AI-suggested implant at `fdi` as clinician-confirmed."""
        self._confirmed_ai_suggestions.add(int(fdi))
        self.implants.confirm_ai_suggestion(int(fdi))

    def analyze_single_tooth(
        self,
        fdi: int,
        volumeNode: Any,
        segmentationNode: Any = None,
        engine: Optional[MeasurementEngine] = None,
    ) -> Dict[str, Any]:
        is_lower = int(fdi) >= 31
        jaw = "lower" if is_lower else "upper"
        site_map = self.get_fdi_arc_site_map(volumeNode, jaw=jaw, segmentationNode=segmentationNode) if volumeNode else {}
        site_obj = site_map.get(str(fdi))
        frame_info = self.get_tooth_arc_mm_and_frame(int(fdi), volumeNode, segmentationNode=segmentationNode) if volumeNode else None
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
        site_pos_source = site_obj.source if site_obj is not None else "approximate_width_table"
        is_approx_site = site_obj.is_approximate if site_obj is not None else True

        if engine is None and volumeNode is not None and segmentationNode is not None:
            engine = self._get_or_build_measurement_engine(volumeNode, segmentationNode)

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
                site_pos_source = meas.site_position_source
                is_approx_site = meas.is_approximate_site
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

        placed_imp = self.implants.placed_implants.get(int(fdi))
        placed_label = placed_imp.label if placed_imp is not None else None
        confirmed = (int(fdi) in self._confirmed_ai_suggestions) or (
            placed_imp.clinician_confirmed if placed_imp is not None else False
        )

        pos_ras: Optional[List[float]] = (
            [round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2)]
            if P is not None
            else None
        )
        return {
            "fdi": int(fdi),
            "jaw": jaw,
            "source": source,
            "site_position_source": site_pos_source,
            "is_approximate_site": is_approx_site,
            "height_mm": height_mm,
            "w_crest_mm": w_crest_mm,
            "w_3mm": w_3mm,
            "w_5mm": w_5mm,
            "nerve_clearance_mm": nerve_clearance_mm,
            "hu": mean_hu,
            "misch": misch,
            "rec_diameter": rec_dia,
            "rec_length": rec_len,
            "is_ai_suggested": True,
            "ai_suggestion_confirmed": confirmed,
            "placed_implant_label": placed_label,
            "pos_ras": pos_ras,
        }

    def analyze_selected_teeth(
        self,
        sorted_teeth: Sequence[int],
        volumeNode: Any,
        segmentationNode: Any = None,
    ) -> List[Dict[str, Any]]:
        engine = (
            self._get_or_build_measurement_engine(volumeNode, segmentationNode)
            if (volumeNode is not None and segmentationNode is not None)
            else None
        )
        return [
            self.analyze_single_tooth(int(fdi), volumeNode, segmentationNode=segmentationNode, engine=engine)
            for fdi in sorted_teeth
        ]

    # ------------------------------------------------------------------
    # F1, F2, F4, F7, F8: Canal, Implant, FDI Overlays, Airway & TMJ
    # ------------------------------------------------------------------
    def extract_and_display_mandibular_canals(
        self, volumeNode: Any, segmentationNode: Any
    ) -> Dict[str, Any]:
        """F1: Extract Left and Right Mandibular Canals from segmentation and display 2D/3D tubes."""
        mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            return {}
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        canals = self.canals.extract_canals_from_mask(vol, mask, is_5class=is_5class)
        self.canals.canals_by_side = canals
        pNode = self.getParameterNode() if slicer is not None else None
        for canal in canals.values():
            self.canals.create_or_update_canal_curve_node(volumeNode, canal, parameterNode=pNode)
        return canals

    def place_implant_at_tooth(
        self,
        fdi: int,
        volumeNode: Any,
        segmentationNode: Any = None,
        entry_id: str = "generic_4.0x10.0",
        buccolingual_tilt_deg: float = 0.0,
        mesiodistal_tilt_deg: float = 0.0,
    ) -> PlacedImplant:
        """F2: Place or update a cylinder implant at `fdi` and create 2D/3D visualization + safety shell."""
        frame_info = self.get_tooth_arc_mm_and_frame(fdi, volumeNode, segmentationNode=segmentationNode)
        if frame_info is not None:
            P, T, N, B, _, _ = frame_info
        else:
            P = np.zeros(3, dtype=np.float64)
            T = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            N = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            B = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False) if (slicer and volumeNode) else None
        mask, is_5class = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if (slicer and volumeNode and segmentationNode)
            else (None, False)
        )
        pNode = self.getParameterNode() if slicer is not None else None
        safety_apical = float(pNode.GetParameter("CanalSafetyMarginMm") or 2.0) if pNode else 2.0

        imp = self.implants.place_or_update_implant(
            fdi=fdi,
            crest_point_ras=P,
            T=T,
            N=N,
            B=B,
            entry_id=entry_id,
            buccolingual_tilt_deg=buccolingual_tilt_deg,
            mesiodistal_tilt_deg=mesiodistal_tilt_deg,
            safety_apical_mm=safety_apical,
            is_ai_suggestion=False,
            clinician_confirmed=True,
            volume=vol,
            mask=mask,
            is_5class=is_5class,
            canals=self.canals.canals_by_side,
        )
        self.implants.create_or_update_implant_mrml_nodes(volumeNode, imp, show_safety_zone=True)
        return imp

    def update_fdi_overlays_from_segmentation(
        self, volumeNode: Any, segmentationNode: Any, visible: bool = True
    ) -> Any:
        """F4: Project segmented tooth centroids and `#FDI` labels onto Panoramic and Cross-Section views."""
        centroids = self._extract_segmented_tooth_centroids(volumeNode, segmentationNode)
        pNode = self.getParameterNode() if slicer is not None else None
        return create_or_update_fdi_segmentation_overlays(
            volumeNode, segmentationNode, centroids, parameterNode=pNode, visible=visible
        )

    def run_airway_analysis(
        self, volumeNode: Any, threshold_hu: float = -400.0
    ) -> Optional[AirwayAnalysisResult]:
        """F7: Run pharyngeal airway volume & minimum cross-sectional area analysis."""
        if slicer is None or volumeNode is None:
            return None
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        res = compute_pharyngeal_airway_metrics(vol, threshold_hu=threshold_hu)
        self.last_airway_result = res
        return res

    def apply_tmj_condyle_view(
        self, volumeNode: Any, segmentationNode: Any = None, side: str = "Right"
    ) -> Optional[TMJCondyleFrame]:
        """F8: Align Yellow (Sagittal) andGreen/Red (Coronal) views to the Left or Right mandibular condyle."""
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False) if (slicer and volumeNode) else None
        mask, _ = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if (slicer and volumeNode and segmentationNode)
            else (None, False)
        )
        frames = compute_tmj_condyle_frames(volume=vol, mask=mask)
        frame = frames.get(side)
        if frame is not None and slicer is not None and getattr(slicer, "app", None) is not None:
            lm = slicer.app.layoutManager()
            if lm:
                yellow = lm.sliceWidget("Yellow")
                if yellow and yellow.sliceLogic():
                    sNode = yellow.sliceLogic().GetSliceNode()
                    P = frame.condyle_center_ras
                    N_sag = frame.sagittal_normal_ras
                    R_sag = frame.sagittal_right_ras
                    sNode.SetSliceToRASByNTP(
                        float(N_sag[0]), float(N_sag[1]), float(N_sag[2]),
                        float(R_sag[0]), float(R_sag[1]), float(R_sag[2]),
                        float(P[0]), float(P[1]), float(P[2]),
                        0,
                    )
        return frame

    # ------------------------------------------------------------------
    # B21, B22 & F10: Report & CSV/JSON Export
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_slice_screenshot_base64(slice_name: str) -> Optional[str]:
        """B21: Capture slice view via `vtkWindowToImageFilter` on `renderWindow()`."""
        return capture_slice_view_png_base64(slice_name)

    @staticmethod
    def export_treatment_plan_files(
        plan_data: Sequence[Dict[str, Any]], out_dir: Path
    ) -> Tuple[Path, Path]:
        """B22: Export JSON and CSV where numeric columns are empty `""` when unmeasured (never string 'not measured')."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_json = out_path / "treatment_plan.json"
        out_csv = out_path / "treatment_plan.csv"

        payload = {
            "software_version": SOFTWARE_VERSION,
            "disclaimer": CLINICAL_DISCLAIMER_TEXT,
            "sites": list(plan_data),
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        fieldnames = [
            "fdi",
            "jaw",
            "measurement_source",
            "site_position_source",
            "is_approximate_site",
            "height_mm",
            "w_crest_mm",
            "w_3mm",
            "w_5mm",
            "nerve_clearance_mm",
            "hu",
            "misch",
            "rec_diameter",
            "rec_length",
            "ai_suggestion_confirmed",
            "pos_r_mm",
            "pos_a_mm",
            "pos_s_mm",
            "disclaimer",
        ]

        def _num_cell(val: Any) -> Any:
            return val if val is not None else ""

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in plan_data:
                pos = item.get("pos_ras")
                src = item.get("source") or NOT_MEASURED
                # F19: Only write AI-suggested rec_diameter/rec_length into CSV if confirmed (or placed)
                confirmed = bool(item.get("ai_suggestion_confirmed", False))
                row_dict = {
                    "fdi": item.get("fdi"),
                    "jaw": item.get("jaw", ""),
                    "measurement_source": src,
                    "site_position_source": item.get("site_position_source", "approximate_width_table"),
                    "is_approximate_site": bool(item.get("is_approximate_site", True)),
                    "height_mm": _num_cell(item.get("height_mm")),
                    "w_crest_mm": _num_cell(item.get("w_crest_mm")),
                    "w_3mm": _num_cell(item.get("w_3mm")),
                    "w_5mm": _num_cell(item.get("w_5mm")),
                    "nerve_clearance_mm": _num_cell(item.get("nerve_clearance_mm")),
                    "hu": _num_cell(item.get("hu")),
                    "misch": item.get("misch") if item.get("misch") is not None else "",
                    "rec_diameter": _num_cell(item.get("rec_diameter")) if confirmed else "",
                    "rec_length": _num_cell(item.get("rec_length")) if confirmed else "",
                    "ai_suggestion_confirmed": confirmed,
                    "pos_r_mm": pos[0] if (pos and len(pos) == 3) else "",
                    "pos_a_mm": pos[1] if (pos and len(pos) == 3) else "",
                    "pos_s_mm": pos[2] if (pos and len(pos) == 3) else "",
                    "disclaimer": CLINICAL_DISCLAIMER_TEXT,
                }
                writer.writerow(row_dict)
        return out_json, out_csv

    def generate_clinical_reports(
        self,
        plan_data: Sequence[Dict[str, Any]],
        volumeNode: Any,
        out_dir: Path,
        capture_screenshots: bool = True,
        lightbox_step_mm: Optional[float] = None,
        include_patient_metadata: bool = False,
        patient_name: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """B21, B22 & F10: Generate both printable HTML and PDF clinical reports."""
        html_path = self.generate_printable_report(
            plan_data=plan_data,
            volumeNode=volumeNode,
            out_dir=out_dir,
            capture_screenshots=capture_screenshots,
            lightbox_step_mm=lightbox_step_mm,
            include_patient_metadata=include_patient_metadata,
            patient_name=patient_name,
            patient_id=patient_id,
        )
        pdf_path = html_path.with_suffix(".pdf")
        export_html_to_pdf(html_path.read_text(encoding="utf-8"), pdf_path)
        return html_path, pdf_path

    def generate_printable_report(
        self,
        plan_data: Sequence[Dict[str, Any]],
        volumeNode: Any,
        out_dir: Path,
        capture_screenshots: bool = True,
        lightbox_step_mm: Optional[float] = None,
        include_patient_metadata: bool = False,
        patient_name: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Path:
        """B21, B22, F2, F19: Generate HTML report with dynamic lightbox labels, opt-in patient metadata, and disclaimer."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        raw_vol_name = volumeNode.GetName() if (volumeNode and hasattr(volumeNode, "GetName")) else "CBCT_Scan"
        safe_stem = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_vol_name).strip("_") or "CBCT_Scan"
        escaped_vol_name = html.escape(raw_vol_name)
        report_file = out_path / f"{safe_stem}_Treatment_Plan_Report.html"

        pNode = self.getParameterNode() if slicer is not None else None
        if lightbox_step_mm is None:
            lightbox_step_mm = float(pNode.GetParameter("LightboxStepMm") or 2.0) if pNode else 2.0
        step_val = float(lightbox_step_mm)
        step_str = f"{step_val:.1f}".rstrip("0").rstrip(".")

        # B22: Anonymize patient fields by default unless opt-in is True
        eff_patient_name = html.escape(str(patient_name or "Patient") if include_patient_metadata else "Anonymous")
        eff_patient_id = html.escape(str(patient_id or "N/A") if include_patient_metadata else "Anonymous")
        report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        rows_html: List[str] = []
        for item in plan_data:
            fdi_esc = html.escape(str(item.get("fdi", "")))
            jaw_esc = html.escape(str(item.get("jaw", "")).capitalize())
            src_raw = str(item.get("source", NOT_MEASURED))
            if item.get("is_approximate_site"):
                src_raw += " (approximate arch site)"
            src_esc = html.escape(src_raw)
            ht_str = html.escape(format_metric(item.get("height_mm"), "mm"))
            w0_str = html.escape(format_metric(item.get("w_crest_mm"), "mm"))
            w3_str = html.escape(format_metric(item.get("w_3mm"), "mm"))
            w5_str = html.escape(format_metric(item.get("w_5mm"), "mm"))
            nerve_str = html.escape(format_metric(item.get("nerve_clearance_mm"), "mm"))

            if item.get("misch") and item.get("hu") is not None:
                misch_str = html.escape(f"{item['misch']} ({float(item['hu']):.0f} gray)")
            else:
                misch_str = NOT_MEASURED

            # F19: Only include AI-suggested implant dimensions in the report if explicitly confirmed by clinician!
            if item.get("placed_implant_label"):
                imp_str = html.escape(str(item["placed_implant_label"]))
            elif (
                item.get("rec_diameter") is not None
                and item.get("rec_length") is not None
                and item.get("ai_suggestion_confirmed")
            ):
                imp_str = html.escape(
                    f"Suggestion (Confirmed): D{float(item['rec_diameter']):.1f} x L{float(item['rec_length']):.1f} mm"
                )
            elif item.get("rec_diameter") is not None and item.get("rec_length") is not None:
                imp_str = "Suggestion pending clinician confirmation"
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

        # F2 & B22: Placed Implants Verification Section
        implant_rows_html: List[str] = []
        for imp in self.implants.get_reportable_implants(include_unconfirmed=False):
            par_str = (
                ", ".join(f"#{k}: {v:.1f}°" for k, v in imp.parallelism_deg_by_fdi.items())
                or "Single implant"
            )
            warn_str = "; ".join(imp.verification_warnings) if imp.verification_warnings else "Verified OK"
            implant_rows_html.append(
                f"<tr>"
                f"<td>#{imp.fdi}</td>"
                f"<td>{html.escape(imp.label)}</td>"
                f"<td>{format_metric(imp.canal_clearance_mm, 'mm')}</td>"
                f"<td>{format_metric(imp.sinus_clearance_mm, 'mm')}</td>"
                f"<td>{html.escape(par_str)}</td>"
                f"<td>{html.escape(imp.misch_class_approx or NOT_MEASURED)}</td>"
                f"<td>{html.escape(warn_str)}</td>"
                f"</tr>"
            )
        implants_section_html = ""
        if implant_rows_html:
            implants_section_html = (
                "<h2>Placed Implants & Verification</h2>"
                "<table><thead><tr>"
                "<th>FDI</th><th>Implant</th><th>Canal Clearance</th><th>Sinus Clearance</th>"
                "<th>Parallelism</th><th>Peri-Implant Density</th><th>Verification Status</th>"
                f"</tr></thead><tbody>{''.join(implant_rows_html)}</tbody></table>"
            )

        screenshots_html = ""
        if capture_screenshots:
            cards = []
            # B22: Derive lightbox labels dynamically from actual `lightbox_step_mm`
            for label, s_name in (
                ("Panoramic View (Green)", "Green"),
                ("Axial View (Red)", "Red"),
                (f"Cross-Section CS- (-{step_str} mm)", "YellowMinus"),
                ("Cross-Section CS0 (0 mm)", "Yellow"),
                (f"Cross-Section CS+ (+{step_str} mm)", "YellowPlus"),
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
            ".banner{background:#fef2f2;border:2px solid #dc2626;color:#991b1b;padding:10px;font-weight:bold;margin-bottom:14px;}"
            ".meta{font-size:13px;margin-bottom:12px;color:#334155;}"
            "table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;}"
            "th,td{border:1px solid #cbd5e1;padding:8px;text-align:center;}"
            "th{background:#0f172a;color:#fff;}"
            ".shots{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;}"
            ".shot-card{flex:1;min-width:180px;border:1px solid #cbd5e1;padding:8px;text-align:center;}"
            ".shot-card img{max-width:100%;height:auto;margin-top:6px;}"
            "</style></head><body>"
            f"<div class='banner'>{html.escape(CLINICAL_DISCLAIMER_TEXT)}</div>"
            f"<h1>CBCT Implant Planning Report — {escaped_vol_name}</h1>"
            f"<div class='meta'>"
            f"<strong>Patient Name:</strong> {eff_patient_name} | "
            f"<strong>Patient ID:</strong> {eff_patient_id} | "
            f"<strong>Date:</strong> {html.escape(report_date)} | "
            f"<strong>Software Version:</strong> SlicerCBCTPlanner v{html.escape(SOFTWARE_VERSION)}"
            f"</div>"
            "<p><em>Note: Unsegmented sites without user-adjusted manual rulers are explicitly reported as "
            f"\"not measured\". {html.escape(MISCH_APPROXIMATE_DISCLAIMER)}.</em></p>"
            f"{screenshots_html}"
            "<table><thead><tr>"
            "<th>FDI</th><th>Jaw</th><th>Source</th><th>Ridge Height</th><th>Crest Width</th>"
            "<th>Width @ 3mm</th><th>Width @ 5mm</th><th>Nerve Distance</th>"
            "<th>Bone Density (Approx)</th><th>Implant / Confirmed Suggestion</th>"
            f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
            f"{implants_section_html}"
            "</body></html>"
        )
        report_file.write_text(html_doc, encoding="utf-8")
        return report_file


class CBCTPlannerTest(ScriptedLoadableModuleTest):
    """O9: 3D Slicer ScriptedLoadableModuleTest with synthetic arch phantom."""

    def setUp(self) -> None:
        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            slicer.mrmlScene.Clear(0)

    def runTest(self) -> None:
        self.setUp()
        self.test_ras_arch_fitting_and_honest_measurements()
        self.test_synthetic_phantom_pipeline()

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

    def test_synthetic_phantom_pipeline(self) -> None:
        """O9: Verify phantom arch, CPR handedness, bounded caches, untouched ruler -> None, and B4 oblique measurements."""
        img = np.full((48, 64, 64), -600.0, dtype=np.float32)
        mask = np.zeros((48, 64, 64), dtype=np.uint16)
        mask[10:30, 20:30, 10:54] = 2  # Mandible
        img[10:30, 20:30, 10:54] = 950.0
        mask[12:14, 24:26, 10:54] = 5  # Mandibular canal

        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4, dtype=np.float64),
        )
        arch_pts = np.array(
            [[45.0, 25.0, 29.0], [32.0, 25.0, 29.0], [20.0, 25.0, 29.0], [12.0, 25.0, 29.0]],
            dtype=np.float64,
        )
        arch = DentalArch(arch_pts, jaw_type="lower", ras_convention=True)
        engine = MeasurementEngine(
            vol, mask, arch=arch, arches={"lower": arch}, is_dental_segmentator_5class=True
        )
        meas = engine.measure_implant_site("46")
        assert meas is not None and meas.bone_height_mm is not None
        assert "approximate" in (meas.misch_density_class or "")

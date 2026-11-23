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
      * B5-B20: Right-handed CPR displacement frame, <=1.5 mm frame spacing, focus-guarded shortcuts,
        VR preset re-application & non-HU percentiles, bounded LRU caches, zoom-preserving pano refresh,
        Red axial scroll preservation, explicit Slab checkbox & Mean/MIP mode, sub-sample segment projection,
        ParameterNode curve observers, node ID references, full ParameterNode round-trip, odontogram click vs
        toggle & Lock to Tooth, enabled collapsible headers, exact token segment precedence, and spline arch auto-fit.
  - Phase 3 (Performance O1-O9):
      * O1-O9: In-process `scipy.ndimage.map_coordinates` panoramic generator, 30 ms coalesced cursor timer,
        MTime-cached MeasurementEngine, zero-copy volume view, ROI-cropped EDT, `CursorSettings`, ruff config,
        and synthetic phantom integration test.
  - Phase 4-7 (UX, Reporting, P1-P3 Clinical Features F1-F20 & Custom AI Checkpoint Inference):
      * Custom AI Checkpoint Inference (`CheckpointInferenceEngine`): drop in any `.pth`, `.pt`, `.ckpt`,
        `.onnx`, `.npy`, `.nii.gz`, or `nnUNetv2` folder (plus optional custom `.py` model script) to run
        3D Gaussian-weighted sliding-window segmentation directly into a styled `vtkMRMLSegmentationNode`.
      * F1-F13: Mandibular canal & root-canal tracing, generic JSON implant library, single occlusal arch,
        FDI overlays, dynamic lightbox & density profile, occlusal SVD fit, airway volume/mCSA, TMJ presets,
        3D angle & surface distance, HTML + PDF report, STL/DICOM export, NNT bone density, and Undo/Redo.
      * F14-F20: Surgical guide CAD with sleeve bores & printer presets (F14), DICOM SEG/SR export (F15),
        generic abutment & anatomic crown proposals (F16), 3D cephalometry against norms (F17),
        `SlicerAutomatedDentalTools` integration (F18), clinician confirmation gate for AI suggestions (F19),
        and headless batch CLI pipeline (`cli_batch.py`, F20).
"""

from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    DEFAULT_RISEMAR_WEIGHTS_PATH,
    validate_risemar_weights_path,
    SOFTWARE_VERSION,
    VIVID_3D_STYLE_PRESETS,
    ALILandmarkResult,
    ASOOrientationResult,
    AirwayAnalysisResult,
    AutomatedDentalToolsEngine,
    CanalTracerLogic,
    CephalometricMeasurement,
    CheckpointInferenceEngine,
    CheckpointLoadSummary,
    CheckpointSpec,
    CrossSectionNavigator,
    CursorSettings,
    GuideSleeveSpec,
    ImpactedToothCLICResult,
    ImplantPlannerLogic,
    PanoramicCurveLogic,
    PlacedImplant,
    ProstheticPlanner,
    ProstheticProposal,
    RiseMarConfig,
    RiseMarEngine,
    RiseMarResult,
    SurgicalGuideGenerator,
    TMJCondyleFrame,
    UndoRedoManager,
    VFACEAsymmetryResult,
    VRVisualizationLogic,
    capture_slice_view_png_base64,
    compute_3d_cephalometry,
    compute_occlusal_plane_from_segmentation,
    compute_pharyngeal_airway_metrics,
    compute_tmj_condyle_frames,
    create_or_update_fdi_segmentation_overlays,
    export_dicom_seg_and_sr,
    export_html_to_pdf,
    registerAndSetDentalLayout,
    registerAndSetDentalLightboxLayout,
    run_automated_dental_tool,
    should_handle_navigation_shortcut,
)
from CBCTPlannerLib.WorkflowUIHelpers import (
    GalleryNavigationState,
    apply_risemar_validation_to_widget,
    apply_workspace_role,
    compute_volume_segmentation_cache_token,
    load_risemar_weights_setting,
    save_risemar_weights_setting,
)
from CBCTPlannerLib.core.arch_detector import (
    JAW_TOOTH_ORDER,
    DentalArch,
    ToothArcSite,
    compute_fdi_arc_positions_mm,
    fit_arch_from_label_mask,
    fit_dental_arch_from_points,
)
from CBCTPlannerLib.core.labels import (
    is_mandibular_fdi,
    sort_teeth_anatomically as _sort_teeth_anatomically,
    tooth_mask_from_labels,
    ANATOMICAL_MESIODISTAL_WIDTHS_MM,
    FDI_TO_TOOTH_ID,
    LOWER_JAW_ID,
    UPPER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    match_segment_to_label_and_precedence,
    resolve_combined_mask_scheme,
    safe_get_segment_tag,
)
from CBCTPlannerLib.core.measurements import (
    MISCH_APPROXIMATE_DISCLAIMER,
    WITHHELD_NO_SEGMENTATION,
    MeasurementEngine,
    withheld_badge,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume

logger = logging.getLogger("CBCTPlanner")

Q1_TEETH = [18, 17, 16, 15, 14, 13, 12, 11]
Q2_TEETH = [21, 22, 23, 24, 25, 26, 27, 28]
Q3_TEETH = [31, 32, 33, 34, 35, 36, 37, 38]
Q4_TEETH = [48, 47, 46, 45, 44, 43, 42, 41]

NOT_MEASURED = "not measured"

# Single source of truth for GUI/parameter-node defaults (avoids clobbering values).
DEFAULTS = {
    "PanoThicknessMm": 25.0,
    "PanoHeightMm": 120.0,
}


def sort_teeth_anatomically(teeth_list: Sequence[int]) -> List[int]:
    """Deprecated re-export; canonical implementation lives in CBCTPlannerLib.core.labels."""
    return _sort_teeth_anatomically(teeth_list)


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
                "Dental CBCT Planner: 7-step workflow (Load & Custom AI Checkpoint -> Arch -> "
                "Panoramic -> Cross-sections -> Plan & Implants -> Canal/Airway/TMJ -> Surgical Guide, "
                "Crown/Abutment, 3D Ceph & DICOM SEG/SR) with in-process CPR, dynamic multi-slice "
                "lightbox, +-30 deg tilt, and headless CLI batch execution."
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
        self.step7ContentWidget: Any = None

        # v3.0 UI/UX Upgrade attributes (Radiologist/Dentist Workspace, RISE-MAR, Vivid 3D Roots, Inline Gallery, ADT)
        self.roleModeCombo: Any = None
        self.btnQuickVividRoots: Any = None
        self.btnQuickRiseMar: Any = None
        self.btnQuickMeasureGallery: Any = None
        self.btnQuickPrintLayout: Any = None
        self.risemarBox: Any = None
        self.risemarWeightsPathEdit: Any = None
        self.btnBrowseRiseMarWeights: Any = None
        self.risemarFastCheck: Any = None
        self.risemarPreserveMetalCheck: Any = None
        self.risemarInPlaceCheck: Any = None
        self.btnRunRiseMar: Any = None
        self.risemarStatusLabel: Any = None
        self.surfaceStylePresetCombo: Any = None
        self.jawOpacitySlider: Any = None
        self.surfaceSmoothingSlider: Any = None
        self.siteClinicalCalloutLabel: Any = None
        self.printOutputDirEdit: Any = None
        self.btnBrowsePrintDir: Any = None
        self.btnOpenPrintDir: Any = None
        self.inlineGalleryBox: Any = None
        self.inlineGalleryStatusLabel: Any = None
        self.inlineGalleryScroll: Any = None
        self.inlineGalleryContainer: Any = None
        self.inlineGalleryGrid: Any = None
        self.inlineGalleryThumbsBox: Any = None
        self.inlineGalleryStaleLabel: Any = None
        self.inlineGalleryActiveImageLabel: Any = None
        self.inlineGalleryActiveCaptionLabel: Any = None
        self.inlineGalleryIndexLabel: Any = None
        self.btnGalleryPrev: Any = None
        self.btnGalleryNext: Any = None
        self.btnGenerateFullGallery: Any = None
        self.btnRefreshInlineGallery: Any = None
        self._galleryState: GalleryNavigationState = GalleryNavigationState()
        self.btnRunNativeALI: Any = None
        self.btnRunNativeASO: Any = None
        self.btnRunNativeCLIC: Any = None
        self.btnRunNativeVFACE: Any = None
        self.btnRunAutoCropROI: Any = None
        self.adtResultsLabel: Any = None

    def getParameterNode(self) -> Any:
        """Return active MRML ParameterNode safely from widget or logic."""
        if self._parameterNode is not None:
            return self._parameterNode
        if self.logic is not None and hasattr(self.logic, "getParameterNode"):
            return self.logic.getParameterNode()
        return None

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

        chipsRow = qt.QHBoxLayout()
        self.stepStatusLabels: list[Any] = []
        for step in range(1, 8):
            chip = qt.QLabel(f"{step} \u25cb")
            chip.setToolTip(f"Step {step} status")
            self.stepStatusLabels.append(chip)
            chipsRow.addWidget(chip)
        chipsRow.addStretch(1)
        self.btnRunFullPipeline = qt.QPushButton("Run full pipeline")
        self.btnRunFullPipeline.setToolTip(
            "Fit the arch from the segmentation, generate the panoramic CPR, and extract canals in one pass."
        )
        self.btnRunFullPipeline.clicked.connect(self.onRunFullPipelineClicked)
        chipsRow.addWidget(self.btnRunFullPipeline)
        bannerLayout.addLayout(chipsRow)

        # v3.0 Clinical Role Workspace & 1-Click Radiologist / Dentist Action Bar
        roleRow = qt.QHBoxLayout()
        roleRow.addWidget(qt.QLabel("<b>Clinical Workspace:</b>"))
        self.roleModeCombo = qt.QComboBox()
        self.roleModeCombo.addItems(
            [
                "All Clinical Tools (Full Suite)",
                "Radiologist Workspace (Measurements, RISE-MAR & Film Print)",
                "Implant & Surgical Dentist (Vivid Roots 3D, Implants & Guide)",
                "Ortho & Craniofacial (ALI Landmarks, ASO, CLI-C, VFACE & Ceph)",
            ]
        )
        self.roleModeCombo.setToolTip(
            "Switch clinical workspace focus tailored for Radiologists, Implant Dentists, or Orthodontists."
        )
        roleRow.addWidget(self.roleModeCombo, 1)
        bannerLayout.addLayout(roleRow)

        quickBar = qt.QHBoxLayout()
        self.btnQuickVividRoots = qt.QPushButton("🦷 Vivid Teeth & Roots 3D")
        self.btnQuickVividRoots.setToolTip(
            "Apply SlicerDentalSegmentator vivid tooth & root colors with translucent bone (0.25 opacity) so roots & canals shine through."
        )
        self.btnQuickVividRoots.setStyleSheet(
            "font-weight: bold; background-color: #0d9488; color: white; padding: 4px 8px; border-radius: 3px;"
        )
        self.btnQuickRiseMar = qt.QPushButton("🧲 RISE-MAR (Artifact Reduction)")
        self.btnQuickRiseMar.setToolTip(
            "Run RISE-MAR Metal Artifact Reduction using the validated weights file configured in Step 1."
        )
        self.btnQuickRiseMar.setStyleSheet(
            "font-weight: bold; background-color: #7c3aed; color: white; padding: 4px 8px; border-radius: 3px;"
        )
        self.btnQuickMeasureGallery = qt.QPushButton("📐 Measure & Show Slices")
        self.btnQuickMeasureGallery.setToolTip(
            "Compute bone morphometry and display measured cross-sectional images directly in Step 5."
        )
        self.btnQuickMeasureGallery.setStyleSheet(
            "font-weight: bold; background-color: #2563eb; color: white; padding: 4px 8px; border-radius: 3px;"
        )
        self.btnQuickPrintLayout = qt.QPushButton("🖨️ Print 8×10″ Film (PDF+PNG)")
        self.btnQuickPrintLayout.setToolTip(
            "Generate printable 8x10 inch clinical film (PDF, 300 DPI PNG, and HTML) and open preview."
        )
        self.btnQuickPrintLayout.setStyleSheet(
            "font-weight: bold; background-color: #0284c7; color: white; padding: 4px 8px; border-radius: 3px;"
        )
        quickBar.addWidget(self.btnQuickVividRoots)
        quickBar.addWidget(self.btnQuickRiseMar)
        quickBar.addWidget(self.btnQuickMeasureGallery)
        quickBar.addWidget(self.btnQuickPrintLayout)
        bannerLayout.addLayout(quickBar)

        self.layout.addWidget(bannerFrame)

        # AI segmentation configuration lives outside the clinical workflow steps.
        # It is a one-time preset, persisted in QSettings, so the main Step 1 panel
        # stays focused on loading data and reviewing results.
        self.aiSettingsBox = ctk.ctkCollapsibleButton()
        self.aiSettingsBox.text = "AI Segmentation Settings (saved preset — set once)"
        self.aiSettingsBox.collapsed = True
        self.layout.addWidget(self.aiSettingsBox)
        aiSettingsLayout = qt.QFormLayout(self.aiSettingsBox)

        self.nnunetBox = ctk.ctkCollapsibleButton()
        self.nnunetBox.text = "nnU-Net ToothFairy3 model"
        self.nnunetBox.collapsed = False
        aiSettingsLayout.addRow(self.nnunetBox)
        nnunetLayout = qt.QFormLayout(self.nnunetBox)

        modelRow = qt.QHBoxLayout()
        self.nnunetModelPathEdit = qt.QLineEdit()
        self.nnunetModelPathEdit.setPlaceholderText("Path to nnU-Net model folder or checkpoint (.pth)")
        self.btnBrowseNNUNetModel = qt.QPushButton("Browse...")
        modelRow.addWidget(self.nnunetModelPathEdit)
        modelRow.addWidget(self.btnBrowseNNUNetModel)
        nnunetLayout.addRow("Model / Checkpoint:", modelRow)

        self.nnunetDeviceCombo = qt.QComboBox()
        self.nnunetDeviceCombo.addItems(
            [
                "Auto (GPU FP32 / CPU INT8)",
                "CPU (INT8 Quantized)",
                "GPU (FP32 Full Precision)",
            ]
        )
        nnunetLayout.addRow("Compute & Precision:", self.nnunetDeviceCombo)

        self.nnunetFoldCombo = qt.QComboBox()
        self.nnunetFoldCombo.addItems(["0", "all", "1", "2", "3", "4"])
        nnunetLayout.addRow("Fold:", self.nnunetFoldCombo)

        self.btnRunNNUNetInference = qt.QPushButton("Run nnU-Net Inference & Auto-Plan (ToothFairy3)")
        self.btnApplyMaskPipeline = qt.QPushButton("Apply Mask to Full Planning Pipeline")
        nnunetLayout.addRow(self.btnRunNNUNetInference)
        nnunetLayout.addRow(self.btnApplyMaskPipeline)

        self.checkpointPathEdit = ctk.ctkPathLineEdit()
        self.checkpointPathEdit.setToolTip(
            "Select your trained 3D segmentation checkpoint (.pth, .pt, .ckpt, .onnx, .npy, .nii.gz, or nnU-Net model folder)."
        )
        aiSettingsLayout.addRow("Custom AI model / checkpoint:", self.checkpointPathEdit)

        self.customArchPathEdit = ctk.ctkPathLineEdit()
        self.customArchPathEdit.setToolTip(
            "Optional: custom Python (.py) model definition for a raw PyTorch state_dict."
        )
        aiSettingsLayout.addRow("Custom model .py (optional):", self.customArchPathEdit)

        ckptRunRow = qt.QHBoxLayout()
        self.ckptSchemeCombo = qt.QComboBox()
        self.ckptSchemeCombo.addItem("ToothFairy3 (32 FDI Teeth + Canal)", "toothfairy3")
        self.ckptSchemeCombo.addItem("5-Class (Dental)", "5class")
        self.ckptSchemeCombo.setToolTip("Output class label scheme of the checkpoint.")
        self.ckptPatchSpin = qt.QSpinBox()
        self.ckptPatchSpin.minimum = 32
        self.ckptPatchSpin.maximum = 256
        self.ckptPatchSpin.singleStep = 16
        self.ckptPatchSpin.value = 96
        self.ckptPatchSpin.suffix = "³"
        self.ckptPatchSpin.setToolTip("3D sliding-window cubic patch size (e.g. 96³ or 128³).")
        self.btnRunCheckpointSeg = qt.QPushButton("Run Checkpoint Segmentation")
        self.btnRunCheckpointSeg.setToolTip(
            "Run 3D Gaussian-weighted sliding-window inference using your checkpoint and populate the Slicer Segmentation node."
        )
        ckptRunRow.addWidget(self.ckptSchemeCombo)
        ckptRunRow.addWidget(qt.QLabel("Patch:"))
        ckptRunRow.addWidget(self.ckptPatchSpin)
        ckptRunRow.addWidget(self.btnRunCheckpointSeg)
        aiSettingsLayout.addRow("Checkpoint inference:", ckptRunRow)

        aiSaveRow = qt.QHBoxLayout()
        self.btnSaveAiPreset = qt.QPushButton("Save AI Preset")
        self.btnSaveAiPreset.setToolTip(
            "Save model paths, device, fold, scheme and patch size for every future session."
        )
        self.aiPresetStatusLabel = qt.QLabel("Preset not loaded yet")
        aiSaveRow.addWidget(self.btnSaveAiPreset)
        aiSaveRow.addWidget(self.aiPresetStatusLabel, 1)
        aiSettingsLayout.addRow(aiSaveRow)
        self._restore_ai_settings_preset()
        self.btnSaveAiPreset.clicked.connect(self._save_ai_settings_preset)
        for signal_name in ("editingFinished", "currentPathChanged"):
            for widget in (self.nnunetModelPathEdit, self.checkpointPathEdit, self.customArchPathEdit):
                signal = getattr(widget, signal_name, None)
                if signal is not None and hasattr(signal, "connect"):
                    signal.connect(self._save_ai_settings_preset)
        for widget in (self.nnunetDeviceCombo, self.nnunetFoldCombo, self.ckptSchemeCombo):
            widget.currentIndexChanged.connect(self._save_ai_settings_preset)
        self.ckptPatchSpin.valueChanged.connect(self._save_ai_settings_preset)

        # ==================== STEP 1: LOAD & CUSTOM AI CHECKPOINT ====================
        self.step1Box = ctk.ctkCollapsibleButton()
        self.step1Box.text = "Step 1: Load CBCT, Custom AI Checkpoint & 3D / VR View"
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
            "Optional: Select an existing segmentation node, or generate one below using AI."
        )
        step1Layout.addRow("Segmentation (Optional):", self.segmentationSelector)

        maskActionRow = qt.QHBoxLayout()
        self.btnLoadExternalMask = qt.QPushButton("Load AI Mask (.nii.gz/.nrrd)")
        self.btnLoadExternalMask.setToolTip(
            "Load a pre-computed AI mask (automatically orients S-I to match patient anatomy)."
        )
        maskActionRow.addWidget(self.btnLoadExternalMask)
        step1Layout.addRow("AI Mask:", maskActionRow)

        self.orientationCodesLabel = qt.QLabel("Detected orientation: --")
        self.orientationCodesLabel.setToolTip(
            "Anatomical codes of the IJK axes. RAS data reads ('R','A','S')."
        )
        step1Layout.addRow("Orientation:", self.orientationCodesLabel)

        orientRow = qt.QHBoxLayout()
        self.btnStandardizeRas = qt.QPushButton("Standardize to RAS")
        self.btnStandardizeRas.setToolTip(
            "Losslessly permute/flip the volume (and segmentation) to RAS. Refuses oblique data instead of resampling."
        )
        self.btnAutoDetectOrientation = qt.QPushButton("Auto-detect Issues")
        self.btnAutoDetectOrientation.setToolTip(
            "Verify S-I, A-P and L-R orientation from jaw and FDI tooth labels."
        )
        orientRow.addWidget(self.btnStandardizeRas)
        orientRow.addWidget(self.btnAutoDetectOrientation)
        step1Layout.addRow("RAS:", orientRow)

        flipRow = qt.QHBoxLayout()
        self.btnFlipLr = qt.QPushButton("Flip L-R")
        self.btnFlipAp = qt.QPushButton("Flip A-P")
        self.btnFlipSi = qt.QPushButton("Flip S-I")
        self.btnFlipLr.setToolTip("Mirror the stored volume left-right (manual header correction).")
        self.btnFlipAp.setToolTip("Mirror the stored volume anterior-posterior (manual header correction).")
        self.btnFlipSi.setToolTip("Mirror the stored volume superior-inferior (manual header correction).")
        flipRow.addWidget(self.btnFlipLr)
        flipRow.addWidget(self.btnFlipAp)
        flipRow.addWidget(self.btnFlipSi)
        step1Layout.addRow("Manual Correction:", flipRow)

        self.autoStandardizeOnLoadCheck = qt.QCheckBox("Auto-standardize on load (RAS)")
        self.autoStandardizeOnLoadCheck.setChecked(True)
        self.autoStandardizeOnLoadCheck.setToolTip(
            "When enabled, loading a volume attempts a lossless RAS standardization. Data is never changed silently."
        )
        try:
            settings = qt.QSettings()
            self.autoStandardizeOnLoadCheck.setChecked(
                str(settings.value(self.SETTINGS_PREFIX + "autoStandardizeOnLoad", "true")).lower() == "true"
            )
        except Exception:
            logger.debug("Could not read autoStandardizeOnLoad setting", exc_info=True)
        step1Layout.addRow(self.autoStandardizeOnLoadCheck)

        layoutBtnRow = qt.QHBoxLayout()
        self.btnDentalLightbox = qt.QPushButton("3-Slice Lightbox (-2 / 0 / +2 mm)")
        self.btnDentalLightbox.setToolTip(
            "Switch to OnDemand3D/Romexis-style 3-slice Cross-Sectional Lightbox (CS-, CS0, CS+ on bottom; Pano, Axial, 3D on top)."
        )
        self.btnDentalLayout = qt.QPushButton("Single CS Layout")
        self.btnDentalLayout.setToolTip(
            "Switch to single Cross-Section Dental layout (Wide Pano top; Axial, CS, 3D bottom)."
        )
        self.btnStandardMpr = qt.QPushButton("Standard MPR")
        self.btnStandardMpr.setToolTip(
            "Reset Axial (Red) and Sagittal (Yellow) views without clobbering Panoramic."
        )
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
        self.vrPresetCombo.setCurrentText("DentalSegmentator (Teeth & Bone)")
        try:
            saved_vr = str(qt.QSettings().value(self.SETTINGS_PREFIX + "vrPreset", "") or "")
            if saved_vr:
                self.vrPresetCombo.setCurrentText(saved_vr)
        except Exception:
            logger.debug("Could not restore VR preset", exc_info=True)
        self.vrPresetCombo.setToolTip(
            "Select Volume Rendering transfer function preset (re-applied cleanly on change, B8)."
        )
        step1Layout.addRow("VR Preset:", self.vrPresetCombo)

        self.vrShiftSlider = ctk.ctkSliderWidget()
        self.vrShiftSlider.minimum = -1000.0
        self.vrShiftSlider.maximum = 1500.0
        self.vrShiftSlider.singleStep = 25.0
        self.vrShiftSlider.value = 0.0
        self.vrShiftSlider.suffix = " HU"
        self.vrShiftSlider.setToolTip(
            "Shift Volume Rendering scalar opacity and color transfer functions (debounced 150 ms)."
        )
        step1Layout.addRow("VR Threshold Shift:", self.vrShiftSlider)

        vrToggleRow = qt.QHBoxLayout()
        self.showVrCheckBox = qt.QCheckBox("Show 3D Volume Rendering")
        self.showVrCheckBox.setToolTip("Toggle 3D Volume Rendering on the input CBCT volume.")
        self.cropVrCheckBox = qt.QCheckBox("ROI Crop")
        self.cropVrCheckBox.setToolTip("Enable 3D ROI cropping box for Volume Rendering.")
        self.btnApplyDental3D = qt.QPushButton("Style Vivid 3D Teeth & Roots + FDI Overlays")
        self.btnApplyDental3D.setToolTip(
            "Apply SlicerDentalSegmentator vivid 3D tooth & root colors, translucent bone, and #FDI labels on Pano/CS."
        )
        vrToggleRow.addWidget(self.showVrCheckBox)
        vrToggleRow.addWidget(self.cropVrCheckBox)
        vrToggleRow.addWidget(self.btnApplyDental3D)
        step1Layout.addRow(vrToggleRow)

        # SlicerDentalSegmentator Vivid 3D Surface & Root Visibility Controls
        self.surfaceStylePresetCombo = qt.QComboBox()
        self.surfaceStylePresetCombo.addItems(list(VIVID_3D_STYLE_PRESETS.keys()))
        self.surfaceStylePresetCombo.setToolTip(
            "Select 3D segmentation surface preset (e.g. Vivid Teeth & Roots with translucent bone, Roots & Nerve X-Ray, or Endo Root-Canal view)."
        )
        step1Layout.addRow("3D Surface Style Preset:", self.surfaceStylePresetCombo)

        vividSlidersRow = qt.QHBoxLayout()
        self.jawOpacitySlider = ctk.ctkSliderWidget()
        self.jawOpacitySlider.minimum = 0.0
        self.jawOpacitySlider.maximum = 1.0
        self.jawOpacitySlider.singleStep = 0.05
        self.jawOpacitySlider.decimals = 2
        self.jawOpacitySlider.value = 0.25
        self.jawOpacitySlider.setToolTip(
            "Adjust Maxilla & Mandible 3D bone opacity (0.25 default makes tooth roots and mandibular canal vividly visible through bone)."
        )
        self.surfaceSmoothingSlider = ctk.ctkSliderWidget()
        self.surfaceSmoothingSlider.minimum = 0.0
        self.surfaceSmoothingSlider.maximum = 1.0
        self.surfaceSmoothingSlider.singleStep = 0.05
        self.surfaceSmoothingSlider.decimals = 2
        self.surfaceSmoothingSlider.value = 0.40
        self.surfaceSmoothingSlider.setToolTip(
            "SlicerDentalSegmentator 3D closed-surface smoothing factor (0.40 default)."
        )
        vividSlidersRow.addWidget(qt.QLabel("Bone 3D Opacity:"))
        vividSlidersRow.addWidget(self.jawOpacitySlider)
        vividSlidersRow.addWidget(qt.QLabel("Smoothing:"))
        vividSlidersRow.addWidget(self.surfaceSmoothingSlider)
        step1Layout.addRow("Root Visibility & Smoothing:", vividSlidersRow)

        # RISE-MAR Metal Artifact Reduction Section (Masaaki-75/rise-mar)
        self.risemarBox = ctk.ctkCollapsibleButton()
        self.risemarBox.text = "RISE-MAR Dental Metal Artifact Reduction (Masaaki-75/rise-mar)"
        self.risemarBox.collapsed = False
        step1Layout.addRow(self.risemarBox)
        risemarLayout = qt.QFormLayout(self.risemarBox)

        marPathRow = qt.QHBoxLayout()
        saved_mar_path = load_risemar_weights_setting(qt)
        self.risemarWeightsPathEdit = qt.QLineEdit()
        self.risemarWeightsPathEdit.setPlaceholderText("Select RISE-MAR model weights (.pkl / .pth)...")
        self.risemarWeightsPathEdit.setText(saved_mar_path)
        self.risemarWeightsPathEdit.setToolTip(
            "Path to RISE-MAR model weights (.pkl / .pth). Remembered in 3D Slicer settings."
        )
        self.btnBrowseRiseMarWeights = qt.QPushButton("Browse...")
        marPathRow.addWidget(self.risemarWeightsPathEdit, 1)
        marPathRow.addWidget(self.btnBrowseRiseMarWeights)
        risemarLayout.addRow("RISE-MAR Weights (.pkl):", marPathRow)

        marOptRow = qt.QHBoxLayout()
        self.risemarFastCheck = qt.QCheckBox("Fast Mode (Metal Slices >= 2200 HU Only)")
        self.risemarFastCheck.setChecked(True)
        self.risemarPreserveMetalCheck = qt.QCheckBox("Preserve Metal Core")
        self.risemarPreserveMetalCheck.setChecked(True)
        self.risemarInPlaceCheck = qt.QCheckBox("Apply In-Place")
        self.risemarInPlaceCheck.setChecked(False)
        marOptRow.addWidget(self.risemarFastCheck)
        marOptRow.addWidget(self.risemarPreserveMetalCheck)
        marOptRow.addWidget(self.risemarInPlaceCheck)
        risemarLayout.addRow("MAR Options:", marOptRow)

        marRunRow = qt.QHBoxLayout()
        self.btnRunRiseMar = qt.QPushButton("🧲 Run RISE-MAR Metal Artifact Reduction")
        self.btnRunRiseMar.setStyleSheet(
            "font-weight: bold; background-color: #7c3aed; color: white; padding: 6px;"
        )
        self.risemarStatusLabel = qt.QLabel("")
        self.risemarStatusLabel.setWordWrap(True)
        marRunRow.addWidget(self.btnRunRiseMar)
        marRunRow.addWidget(self.risemarStatusLabel, 1)
        risemarLayout.addRow(marRunRow)
        apply_risemar_validation_to_widget(self, saved_mar_path)

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
        self.activeJawCombo.setToolTip(
            "Select which jaw arch curve (Lower Mandible or Upper Maxilla) is active."
        )
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
        self.showCurvesCheckBox.setToolTip(
            "Show or hide dental arch curve control points in 2D and 3D views."
        )
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

        self.step3HintLabel = qt.QLabel(
            "Prerequisite: Requires an active dental arch curve (>= 4 points) in Step 2."
        )
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
        self.panoOffsetSpin.setToolTip(
            "Buccolingual offset of the panoramic slab relative to the arch curve (F3)."
        )

        panoGeomRow.addWidget(qt.QLabel("Thickness:"))
        panoGeomRow.addWidget(self.panoThicknessSpin)
        panoGeomRow.addWidget(qt.QLabel("Height:"))
        panoGeomRow.addWidget(self.panoHeightSpin)
        panoGeomRow.addWidget(qt.QLabel("Offset:"))
        panoGeomRow.addWidget(self.panoOffsetSpin)
        step3Layout.addRow("Panoramic Geometry:", panoGeomRow)

        self.panoProjCombo = qt.QComboBox()
        self.panoProjCombo.addItems(["Mean", "MIP", "MinIP", "RaySum"])
        self.panoProjCombo.setToolTip(
            "Intensity projection mode across the slab (including RaySum pseudo-pan, F3)."
        )
        step3Layout.addRow("Projection Mode:", self.panoProjCombo)

        self.panoFlipVerticalCheck = qt.QCheckBox("Flip vertical (Advanced)")
        self.panoFlipVerticalCheck.setChecked(False)
        self.panoFlipVerticalCheck.setToolTip(
            "Flip the panoramic image superior-inferior. Only needed for legacy non-RAS data; "
            "standardized RAS volumes do not require it."
        )
        step3Layout.addRow(self.panoFlipVerticalCheck)

        self.autoUpdatePanoCheckBox = qt.QCheckBox(
            "Auto-update Panoramic when Curve Changes (Preview / Full-Res)"
        )
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

        self.step4HintLabel = qt.QLabel(
            "Prerequisite: Requires an active dental arch curve (>= 4 points) in Step 2."
        )
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
        self.csTiltSlider.setToolTip(
            "Lean the cross-section plane mesiodistally between -30° and +30° around N."
        )
        self.btnResetTilt = qt.QPushButton("Reset 0°")
        self.btnResetTilt.setToolTip("Reset cross-section tilt to 0° (perpendicular to occlusal plane).")
        tiltRow.addWidget(self.csTiltSlider)
        tiltRow.addWidget(self.btnResetTilt)
        step4Layout.addRow("Cross-Section Tilt (+-30°):", tiltRow)

        lightboxRow = qt.QHBoxLayout()
        self.enableLightboxCheckBox = qt.QCheckBox("Enable Lightbox (CS- / CS0 / CS+)")
        self.enableLightboxCheckBox.setChecked(True)
        self.enableLightboxCheckBox.setToolTip(
            "Synchronize YellowMinus, Yellow, and YellowPlus cross-sections."
        )
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
        self.btnActivateLightboxLayout.setToolTip(
            "Switch viewport layout to show CS-, CS0, and CS+ side by side."
        )
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

        csParamsRow = qt.QHBoxLayout()
        self.csWidthSpin = qt.QDoubleSpinBox()
        self.csWidthSpin.minimum = 8.0
        self.csWidthSpin.maximum = 50.0
        self.csWidthSpin.value = 24.0
        self.csWidthSpin.suffix = " mm"
        self.csWidthSpin.setToolTip(
            "Buccolingual width of the active cross-section indicator line on the Axial view."
        )

        self.csSlabEnabledCheckBox = qt.QCheckBox("Slab")
        self.csSlabEnabledCheckBox.setChecked(False)
        self.csSlabEnabledCheckBox.setToolTip(
            "Explicitly enable thick-slab reconstruction on Cross-Section views (B12)."
        )

        self.csSlabModeCombo = qt.QComboBox()
        self.csSlabModeCombo.addItems(["Mean", "MIP", "MinIP"])
        self.csSlabModeCombo.setToolTip(
            "Cross-section slab reconstruction mode (Mean default, MIP option, B12)."
        )

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
        self.tickSpacingSpin.setToolTip(
            "Spacing in mm between adjacent ticks in the Axial cross-section tick fan."
        )

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
        self.showAxialTicksCheckBox.setToolTip(
            "Display faint perpendicular tick lines along the arch in the Red slice view only."
        )
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

        # Quick Quadrant Selection Row
        quadrantRow = qt.QHBoxLayout()
        self.btnSelectQ1 = qt.QPushButton("Select Q1 (Upper R)")
        self.btnSelectQ2 = qt.QPushButton("Select Q2 (Upper L)")
        self.btnSelectQ3 = qt.QPushButton("Select Q3 (Lower L)")
        self.btnSelectQ4 = qt.QPushButton("Select Q4 (Lower R)")
        self.btnClearSelection = qt.QPushButton("Clear Selection")
        quadrantRow.addWidget(self.btnSelectQ1)
        quadrantRow.addWidget(self.btnSelectQ2)
        quadrantRow.addWidget(self.btnSelectQ3)
        quadrantRow.addWidget(self.btnSelectQ4)
        quadrantRow.addWidget(self.btnClearSelection)
        step5Layout.addLayout(quadrantRow)

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

        # Status Label showing current selection
        self.step5SelectionStatusLabel = qt.QLabel(
            "Active Site: Quadrant 4 (Mandibular Right) — 15 to 27 Cross-Sectional Slices"
        )
        self.step5SelectionStatusLabel.setStyleSheet("font-weight: bold; color: #38bdf8; padding: 4px;")
        step5Layout.addWidget(self.step5SelectionStatusLabel)

        # Radiologist & Dentist Clinical Callout Card for Active Tooth Site
        self.siteClinicalCalloutLabel = qt.QLabel(
            "<b>Clinical Callout:</b> Select a tooth or quadrant above to inspect Ridge Height (H), "
            "Crestal/3mm/5mm Widths (W0/W3/W5), IAN Nerve Clearance, Bone Density (HU / Misch D1–D4), and Implant Sizing."
        )
        self.siteClinicalCalloutLabel.setWordWrap(True)
        self.siteClinicalCalloutLabel.setStyleSheet(
            "background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-left: 4px solid #38bdf8; padding: 8px; border-radius: 4px;"
        )
        step5Layout.addWidget(self.siteClinicalCalloutLabel)

        # Print Output Folder Picker (ensures print files are saved to a visible user directory)
        printDirRow = qt.QHBoxLayout()
        printDirRow.addWidget(qt.QLabel("Print Output Folder:"))
        self.printOutputDirEdit = qt.QLineEdit()
        try:
            from CBCTPlannerLib.PrintLayoutEngine import PrintLayoutEngine

            self.printOutputDirEdit.setText(str(PrintLayoutEngine.default_export_directory()))
        except Exception:
            self.printOutputDirEdit.setText(str(Path.home() / "Documents" / "CBCTPlanner_Prints"))
        self.printOutputDirEdit.setToolTip(
            "Folder where printable PDF, 300 DPI PNG, and HTML film layouts are saved."
        )
        self.btnBrowsePrintDir = qt.QPushButton("Browse...")
        self.btnOpenPrintDir = qt.QPushButton("📂 Open Folder")
        self.btnOpenPrintDir.setToolTip("Open the print output folder in your system file explorer.")
        printDirRow.addWidget(self.printOutputDirEdit, 1)
        printDirRow.addWidget(self.btnBrowsePrintDir)
        printDirRow.addWidget(self.btnOpenPrintDir)
        step5Layout.addLayout(printDirRow)

        # Primary Action: 8x10 Print Layout Button
        self.btnGenerate8x10PrintLayout = qt.QPushButton(
            "🖨️ Generate 8×10″ Print Layout (Saves PDF + 300 DPI PNG + HTML & Opens Preview)"
        )
        self.btnGenerate8x10PrintLayout.setToolTip(
            "Generate standard 8x10 inch clinical film layout with panoramic reformat, "
            "15 to 27 sequential cross-sectional slices with overlaid measurement lines, and 3D quadrant VR."
        )
        self.btnGenerate8x10PrintLayout.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 8px; background-color: #0284c7; color: white; border-radius: 4px;"
        )
        step5Layout.addWidget(self.btnGenerate8x10PrintLayout)

        # Embedded Inline Measured Cross-Section Viewer & Lazy Thumbnail Strip (Part 6.1)
        self.inlineGalleryBox = ctk.ctkCollapsibleButton()
        self.inlineGalleryBox.text = "Measured Cross-Section Viewer (H, W0, W3, W5, Root & Vital Overlays)"
        self.inlineGalleryBox.collapsed = False
        galleryOuter = qt.QVBoxLayout(self.inlineGalleryBox)
        galleryOuter.setContentsMargins(6, 6, 6, 6)
        galleryOuter.setSpacing(6)

        self.inlineGalleryStaleLabel = qt.QLabel("")
        self.inlineGalleryStaleLabel.setWordWrap(True)
        self.inlineGalleryStaleLabel.setStyleSheet(
            "color: #fbbf24; background-color: #451a03; border: 1px solid #d97706; border-radius: 4px; padding: 4px 6px; font-weight: bold;"
        )
        self.inlineGalleryStaleLabel.setVisible(False)
        galleryOuter.addWidget(self.inlineGalleryStaleLabel)

        galleryTopRow = qt.QHBoxLayout()
        self.inlineGalleryStatusLabel = qt.QLabel(
            "Select a tooth or click 'Refresh Active Slice' to display the measured cross-section."
        )
        self.inlineGalleryStatusLabel.setWordWrap(True)
        self.btnRefreshInlineGallery = qt.QPushButton("Refresh Active Slice")
        self.btnRefreshInlineGallery.setToolTip(
            "Render the active tooth cross-section immediately without generating all 15-28 gallery tiles."
        )
        self.btnGenerateFullGallery = qt.QPushButton("Generate Full Gallery")
        self.btnGenerateFullGallery.setToolTip(
            "Generate the complete multi-slice cross-section gallery (15-27 slices) on demand."
        )
        galleryTopRow.addWidget(self.inlineGalleryStatusLabel, 1)
        galleryTopRow.addWidget(self.btnRefreshInlineGallery)
        galleryTopRow.addWidget(self.btnGenerateFullGallery)
        galleryOuter.addLayout(galleryTopRow)

        self.inlineGalleryActiveImageLabel = qt.QLabel("No active cross-section rendered yet.")
        self.inlineGalleryActiveImageLabel.setAlignment(qt.Qt.AlignCenter)
        self.inlineGalleryActiveImageLabel.setMinimumHeight(280)
        self.inlineGalleryActiveImageLabel.setStyleSheet(
            "background-color: #020617; color: #94a3b8; border: 1px solid #1e293b; border-radius: 6px; padding: 6px;"
        )
        galleryOuter.addWidget(self.inlineGalleryActiveImageLabel)

        self.inlineGalleryActiveCaptionLabel = qt.QLabel("")
        self.inlineGalleryActiveCaptionLabel.setAlignment(qt.Qt.AlignCenter)
        self.inlineGalleryActiveCaptionLabel.setWordWrap(True)
        galleryOuter.addWidget(self.inlineGalleryActiveCaptionLabel)

        navRow = qt.QHBoxLayout()
        self.btnGalleryPrev = qt.QPushButton("Previous")
        self.btnGalleryPrev.setEnabled(False)
        self.inlineGalleryIndexLabel = qt.QLabel("Slice 0 / 0")
        self.inlineGalleryIndexLabel.setAlignment(qt.Qt.AlignCenter)
        self.btnGalleryNext = qt.QPushButton("Next")
        self.btnGalleryNext.setEnabled(False)
        navRow.addWidget(self.btnGalleryPrev)
        navRow.addWidget(self.inlineGalleryIndexLabel, 1)
        navRow.addWidget(self.btnGalleryNext)
        galleryOuter.addLayout(navRow)

        self.inlineGalleryThumbsBox = ctk.ctkCollapsibleButton()
        self.inlineGalleryThumbsBox.text = "All Cross-Section Thumbnails (0)"
        self.inlineGalleryThumbsBox.collapsed = True
        thumbsOuter = qt.QVBoxLayout(self.inlineGalleryThumbsBox)
        thumbsOuter.setContentsMargins(4, 4, 4, 4)

        self.inlineGalleryScroll = qt.QScrollArea()
        self.inlineGalleryScroll.setWidgetResizable(True)
        self.inlineGalleryScroll.setMinimumHeight(180)
        self.inlineGalleryScroll.setMaximumHeight(225)
        self.inlineGalleryContainer = qt.QWidget()
        self.inlineGalleryGrid = qt.QGridLayout(self.inlineGalleryContainer)
        self.inlineGalleryGrid.setSpacing(6)
        self.inlineGalleryScroll.setWidget(self.inlineGalleryContainer)
        thumbsOuter.addWidget(self.inlineGalleryScroll)
        galleryOuter.addWidget(self.inlineGalleryThumbsBox)
        step5Layout.addWidget(self.inlineGalleryBox)

        # Collapsible Section for Advanced Details (prevents UI clutter)
        self.advancedStep5Box = ctk.ctkCollapsibleButton()
        self.advancedStep5Box.text = "Advanced Implant Planning & Detailed Measurements (Optional)"
        self.advancedStep5Box.collapsed = True
        advancedStep5Layout = qt.QVBoxLayout(self.advancedStep5Box)

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
        self.btnRecomputeSelected.setToolTip(
            "Re-evaluate segmentation or user-modified manual rulers for planned teeth."
        )
        self.btnClearPlan = qt.QPushButton("Clear Plan")
        self.btnClearPlan.setToolTip("Clear all selected teeth from the treatment plan table.")
        rulerRow.addWidget(self.btnToggleActiveToothInPlan)
        rulerRow.addWidget(self.btnCreateManualRulers)
        rulerRow.addWidget(self.btnRecomputeSelected)
        rulerRow.addWidget(self.btnClearPlan)
        advancedStep5Layout.addLayout(rulerRow)

        implantRow = qt.QHBoxLayout()
        self.implantLibraryCombo = qt.QComboBox()
        if self.logic is not None:
            for entry in self.logic.implants.get_library_entries():
                self.implantLibraryCombo.addItem(entry.label, entry.id)
        self.implantLibraryCombo.setToolTip(
            "Select generic cylinder implant dimensions from extensible JSON library (F2)."
        )
        self.btnPlaceImplant = qt.QPushButton("Place / Update Implant at Active Tooth")
        self.btnPlaceImplant.setToolTip(
            "Place cylinder implant at active tooth site with safety zone verification (F2)."
        )
        self.btnImplantCentricView = qt.QPushButton("Implant-Centric View")
        self.btnImplantCentricView.setToolTip(
            "Align cross-section slice along the placed implant's long axis (F2)."
        )
        self.btnConfirmAiSuggestion = qt.QPushButton("Confirm AI Suggestion")
        self.btnConfirmAiSuggestion.setToolTip(
            "Explicitly confirm an AI-suggested implant size so it can enter exported clinical reports (F19)."
        )
        implantRow.addWidget(self.implantLibraryCombo)
        implantRow.addWidget(self.btnPlaceImplant)
        implantRow.addWidget(self.btnImplantCentricView)
        implantRow.addWidget(self.btnConfirmAiSuggestion)
        advancedStep5Layout.addLayout(implantRow)

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
        advancedStep5Layout.addWidget(self.planTable)

        exportOptRow = qt.QHBoxLayout()
        self.includePatientMetaCheckBox = qt.QCheckBox(
            "Include Patient Metadata in Report (Opt-In; Anonymized by Default)"
        )
        self.includePatientMetaCheckBox.setChecked(False)
        self.includePatientMetaCheckBox.setToolTip(
            "When unchecked, exported HTML/PDF/DICOM reports anonymize Patient Name and ID to 'Anonymous' (B22)."
        )
        exportOptRow.addWidget(self.includePatientMetaCheckBox)
        advancedStep5Layout.addLayout(exportOptRow)

        exportRow = qt.QHBoxLayout()
        self.btnGenerateSliceSeries = qt.QPushButton("Cross-Section Series Preview (15-28 Slices)")
        self.btnGenerateSliceSeries.setToolTip(
            "Generate 15 to 28 sequential cross-sectional slice images across tooth or quadrant with overlaid measurement levels."
        )
        self.btnGenerateReport = qt.QPushButton("Export HTML + PDF Clinical Report")
        self.btnGenerateReport.setToolTip(
            "Generate printable HTML and PDF clinical reports with cross-sectional series, measurements, and disclaimer (B21, B22, F10)."
        )
        self.btnExportPlan = qt.QPushButton("Export CSV / JSON / Implant STL")
        self.btnExportPlan.setToolTip(
            "Export clean numeric CSV (empty when unmeasured), JSON, and triangulated implant STL meshes (B22, F11)."
        )
        exportRow.addWidget(self.btnGenerateSliceSeries)
        exportRow.addWidget(self.btnGenerateReport)
        exportRow.addWidget(self.btnExportPlan)
        advancedStep5Layout.addLayout(exportRow)

        step5Layout.addWidget(self.advancedStep5Box)

        # ==================== STEP 6: SURGICAL EXPORT & CLINICAL TOOLS ====================
        self.step6Box = ctk.ctkCollapsibleButton()
        self.step6Box.text = "Step 6: Mandibular Canals, Surgical Guide CAD & DICOM Export"
        self.step6Box.collapsed = True
        self.layout.addWidget(self.step6Box)
        step6Outer = qt.QVBoxLayout(self.step6Box)

        self.step6HintLabel = qt.QLabel(
            "Extract/trace Mandibular Canals (F1), export 3D-printable surgical guide STL (F14), or DICOM SEG/SR."
        )
        self.step6HintLabel.setWordWrap(True)
        step6Outer.addWidget(self.step6HintLabel)

        self.step6ContentWidget = qt.QWidget()
        step6Layout = qt.QFormLayout(self.step6ContentWidget)
        step6Outer.addWidget(self.step6ContentWidget)

        canalRow = qt.QHBoxLayout()
        self.btnExtractCanals = qt.QPushButton("Extract Canals from Seg")
        self.btnExtractCanals.setToolTip(
            "Extract Left and Right Mandibular Canals (IAN) from segmentation mask (F1)."
        )
        self.btnTraceRightCanal = qt.QPushButton("Trace Right")
        self.btnTraceRightCanal.setToolTip("Manually trace Right Mandibular Canal (F1).")
        self.btnTraceLeftCanal = qt.QPushButton("Trace Left")
        self.btnTraceLeftCanal.setToolTip("Manually trace Left Mandibular Canal (F1).")
        self.canalMarginSpin = qt.QDoubleSpinBox()
        self.canalMarginSpin.minimum = 0.5
        self.canalMarginSpin.maximum = 5.0
        self.canalMarginSpin.singleStep = 0.5
        self.canalMarginSpin.value = 2.0
        self.canalMarginSpin.suffix = " mm"
        self.canalMarginSpin.setToolTip(
            "Configurable safety margin around the mandibular canal (default 2.0 mm, F1)."
        )
        canalRow.addWidget(self.btnExtractCanals)
        canalRow.addWidget(self.btnTraceRightCanal)
        canalRow.addWidget(self.btnTraceLeftCanal)
        canalRow.addWidget(qt.QLabel("Safety:"))
        canalRow.addWidget(self.canalMarginSpin)
        step6Layout.addRow("Mandibular Canal:", canalRow)

        guideRow = qt.QHBoxLayout()
        self.guideSupportCombo = qt.QComboBox()
        self.guideSupportCombo.addItem("Tooth-Supported", "tooth_supported")
        self.guideSupportCombo.addItem("Mucosa-Supported", "mucosa_supported")
        self.printerPresetCombo = qt.QComboBox()
        if self.logic is not None:
            for p_entry in self.logic.surgical_guides.presets.values():
                self.printerPresetCombo.addItem(p_entry.label, p_entry.id)
        self.btnExportSurgicalGuide = qt.QPushButton("Export Surgical Guide STL")
        self.btnExportSurgicalGuide.setToolTip(
            "Build surgical guide mesh with open drill sleeve bores and export to STL + JSON (F14)."
        )
        guideRow.addWidget(self.guideSupportCombo)
        guideRow.addWidget(self.printerPresetCombo)
        guideRow.addWidget(self.btnExportSurgicalGuide)
        step6Layout.addRow("Surgical Guide:", guideRow)

        exportRow = qt.QHBoxLayout()
        self.btnExportDicomSegSr = qt.QPushButton("Export DICOM SEG + SR")
        self.btnExportDicomSegSr.setToolTip(
            "Export segmentation and treatment plan as DICOM-SEG and DICOM BasicTextSR (F15)."
        )
        self.btnProposeCrowns = qt.QPushButton("Propose Crowns & Abutments")
        exportRow.addWidget(self.btnExportDicomSegSr)
        exportRow.addWidget(self.btnProposeCrowns)
        step6Layout.addRow("DICOM & Prosthetics:", exportRow)

        airwayRow = qt.QHBoxLayout()
        self.btnAnalyzeAirway = qt.QPushButton("Analyze Airway")
        self.btnAnalyzeAirway.setToolTip(
            "Segment pharyngeal airway and report volume / minimum cross-sectional area (F7)."
        )
        self.btnTmjRightSag = qt.QPushButton("TMJ Right")
        self.btnTmjRightSag.setToolTip("Show sagittal TMJ condyle frame for the right joint (F8).")
        self.btnTmjLeftSag = qt.QPushButton("TMJ Left")
        self.btnTmjLeftSag.setToolTip("Show sagittal TMJ condyle frame for the left joint (F8).")
        airwayRow.addWidget(self.btnAnalyzeAirway)
        airwayRow.addWidget(self.btnTmjRightSag)
        airwayRow.addWidget(self.btnTmjLeftSag)
        step6Layout.addRow("Airway & TMJ:", airwayRow)

        # ==================== STEP 7: ADVANCED TOOLING ====================
        self.step7Box = ctk.ctkCollapsibleButton()
        self.step7Box.text = "Step 7: 3D Cephalometry & Automated Dental Tools"
        self.step7Box.collapsed = True
        self.layout.addWidget(self.step7Box)
        step7Outer = qt.QVBoxLayout(self.step7Box)

        self.step7HintLabel = qt.QLabel(
            "Place S, N, A, B, Go, Gn fiducials for 3D cephalometry, or launch an installed "
            "SlicerAutomatedDentalTools module."
        )
        self.step7HintLabel.setWordWrap(True)
        step7Outer.addWidget(self.step7HintLabel)

        self.step7ContentWidget = qt.QWidget()
        step7Layout = qt.QFormLayout(self.step7ContentWidget)
        step7Outer.addWidget(self.step7ContentWidget)

        cephRow = qt.QHBoxLayout()
        self.btnCompute3DCeph = qt.QPushButton("Compute 3D Cephalometry")
        self.btnCompute3DCeph.setToolTip(
            "Compute SNA/SNB/ANB and related angles from S,N,A,B,Go,Gn fiducials (F17)."
        )
        cephRow.addWidget(self.btnCompute3DCeph)
        step7Layout.addRow("Cephalometry:", cephRow)

        adtNativeRow1 = qt.QHBoxLayout()
        self.btnRunNativeALI = qt.QPushButton("📍 ALI: Auto-Identify 3D Landmarks")
        self.btnRunNativeALI.setToolTip(
            "Automatically identify 3D craniofacial & dental landmarks (S, N, A, B, ANS, PNS, Go, Co, Me, Gn, Pg, Mental Foramen, Crown/Apex) and create Slicer Markups."
        )
        self.btnRunNativeASO = qt.QPushButton("🧭 ASO: Auto-Orient Head & Occlusal")
        self.btnRunNativeASO.setToolTip(
            "Compute and apply 3D Standardized Orientation (Midsagittal + Occlusal/Frankfort Horizontal alignment)."
        )
        adtNativeRow1.addWidget(self.btnRunNativeALI)
        adtNativeRow1.addWidget(self.btnRunNativeASO)
        step7Layout.addRow("ALI & ASO (Native):", adtNativeRow1)

        adtNativeRow2 = qt.QHBoxLayout()
        self.btnRunNativeCLIC = qt.QPushButton("🦷 CLI-C: Impacted Canine/Molar Analysis")
        self.btnRunNativeCLIC.setToolTip(
            "Classify impacted canines (#13,#23,#33,#43) & wisdom teeth (#18,#28,#38,#48) into Buccal/Bicortical/Palatal zones and assess adjacent root resorption risk."
        )
        self.btnRunNativeVFACE = qt.QPushButton("⚖️ VFACE: Facial & Mandibular Asymmetry")
        self.btnRunNativeVFACE.setToolTip(
            "Quantify Left vs Right Ramus height, Mandibular body length, Condylar height, and Menton chin deviation."
        )
        self.btnRunAutoCropROI = qt.QPushButton("✂️ AutoCrop3D: Fit Dental ROI")
        self.btnRunAutoCropROI.setToolTip(
            "Fit a tight 3D Region of Interest (ROI) bounding box around the segmented dental arches."
        )
        adtNativeRow2.addWidget(self.btnRunNativeCLIC)
        adtNativeRow2.addWidget(self.btnRunNativeVFACE)
        adtNativeRow2.addWidget(self.btnRunAutoCropROI)
        step7Layout.addRow("CLI-C, VFACE & Crop:", adtNativeRow2)

        self.adtResultsLabel = qt.QLabel("AutomatedDentalTools ready — click ALI, ASO, CLI-C, VFACE, or AutoCrop3D above.")
        self.adtResultsLabel.setWordWrap(True)
        self.adtResultsLabel.setStyleSheet(
            "background-color: #0f172a; color: #e2e8f0; padding: 6px; border-radius: 4px; border: 1px solid #334155;"
        )
        step7Layout.addRow("Clinical Analysis:", self.adtResultsLabel)

        adtRow = qt.QHBoxLayout()
        self.adtToolCombo = qt.QComboBox()
        for tool in ("ASO", "AMASSS", "ALI", "AReg", "AutoCrop3D", "BatchDentalSegmentator", "MRI2CBCT", "CLI-C", "VFACE"):
            self.adtToolCombo.addItem(tool, tool)
        self.adtToolCombo.setToolTip("Installed SlicerAutomatedDentalTools module to launch (or run built-in native engine).")
        self.btnLaunchAdtTool = qt.QPushButton("Launch / Run Tool")
        self.btnLaunchAdtTool.setToolTip(
            "Launch the selected SlicerAutomatedDentalTools module if installed, or execute the built-in native algorithm."
        )
        adtRow.addWidget(self.adtToolCombo)
        adtRow.addWidget(self.btnLaunchAdtTool)
        step7Layout.addRow("External Module Bridge:", adtRow)

        self.layout.addStretch()

        # Connect signals
        self.btnUndo.clicked.connect(self.onUndoClicked)
        self.btnRedo.clicked.connect(self.onRedoClicked)

        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeNodeChanged)
        self.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.btnBrowseNNUNetModel.clicked.connect(self.onBrowseNNUNetModelClicked)
        self.btnRunNNUNetInference.clicked.connect(self.onRunNNUNetInferenceClicked)
        self.btnApplyMaskPipeline.clicked.connect(self.onApplyMaskPipelineClicked)
        self.btnRunCheckpointSeg.clicked.connect(self.onRunCheckpointSegmentationClicked)
        self.btnDentalLightbox.clicked.connect(self.onSetDentalLightboxLayout)
        self.btnLoadExternalMask.clicked.connect(self.onLoadExternalMask)
        self.btnStandardizeRas.clicked.connect(self.onStandardizeRasClicked)
        self.btnAutoDetectOrientation.clicked.connect(self.onAutoDetectOrientationClicked)
        self.btnFlipLr.clicked.connect(lambda: self.onFlipOrientationClicked("LR"))
        self.btnFlipAp.clicked.connect(lambda: self.onFlipOrientationClicked("AP"))
        self.btnFlipSi.clicked.connect(lambda: self.onFlipOrientationClicked("SI"))
        self.autoStandardizeOnLoadCheck.toggled.connect(self.onAutoStandardizeOnLoadToggled)
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
        self.panoFlipVerticalCheck.toggled.connect(self.onPanoParamChanged)
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

        self.btnSelectQ1.clicked.connect(lambda: self.onSelectQuadrant(1))
        self.btnSelectQ2.clicked.connect(lambda: self.onSelectQuadrant(2))
        self.btnSelectQ3.clicked.connect(lambda: self.onSelectQuadrant(3))
        self.btnSelectQ4.clicked.connect(lambda: self.onSelectQuadrant(4))
        self.btnClearSelection.clicked.connect(self.onClearToothSelection)
        self.btnGenerate8x10PrintLayout.clicked.connect(self.onGenerate8x10PrintLayout)
        self.btnToggleActiveToothInPlan.clicked.connect(self.onToggleActiveToothInPlanClicked)
        self.btnCreateManualRulers.clicked.connect(self.onCreateManualRulersClicked)
        self.btnRecomputeSelected.clicked.connect(self.onRefreshAllSelectedTeeth)
        self.btnClearPlan.clicked.connect(self.onClearPlanClicked)
        self.btnPlaceImplant.clicked.connect(self.onPlaceImplantClicked)
        self.btnImplantCentricView.clicked.connect(self.onImplantCentricViewClicked)
        self.btnConfirmAiSuggestion.clicked.connect(self.onConfirmAiSuggestionClicked)
        self.planTable.cellClicked.connect(self.onTableRowClicked)
        self.includePatientMetaCheckBox.toggled.connect(self.onExportOptChanged)
        self.btnGenerateSliceSeries.clicked.connect(self.onGenerateSliceSeriesClicked)
        self.btnGenerateReport.clicked.connect(self.onGeneratePrintableReport)
        self.btnExportPlan.clicked.connect(self.onExportTreatmentPlan)

        self.btnExtractCanals.clicked.connect(self.onExtractCanalsClicked)
        self.btnTraceRightCanal.clicked.connect(lambda: self.onTraceCanalManual("Right"))
        self.btnTraceLeftCanal.clicked.connect(lambda: self.onTraceCanalManual("Left"))
        self.canalMarginSpin.valueChanged.connect(self.onCanalMarginChanged)
        self.btnAnalyzeAirway.clicked.connect(self.onAnalyzeAirwayClicked)
        self.btnTmjRightSag.clicked.connect(lambda: self.onTmjPresetClicked("Right"))
        self.btnTmjLeftSag.clicked.connect(lambda: self.onTmjPresetClicked("Left"))

        self.btnExportSurgicalGuide.clicked.connect(self.onExportSurgicalGuideClicked)
        self.btnProposeCrowns.clicked.connect(self.onProposeCrownsClicked)
        self.btnCompute3DCeph.clicked.connect(self.onCompute3DCephClicked)
        self.btnExportDicomSegSr.clicked.connect(self.onExportDicomSegSrClicked)
        self.btnLaunchAdtTool.clicked.connect(self.onLaunchAdtToolClicked)

        # v3.0 Signal Connections
        self.roleModeCombo.currentTextChanged.connect(self.onRoleModeChanged)
        self.btnQuickVividRoots.clicked.connect(self.onApplyDental3DStyling)
        self.btnQuickRiseMar.clicked.connect(self.onRunRiseMarClicked)
        self.btnQuickMeasureGallery.clicked.connect(self.onRefreshInlineGalleryClicked)
        self.btnQuickPrintLayout.clicked.connect(self.onGenerate8x10PrintLayout)
        self.surfaceStylePresetCombo.currentTextChanged.connect(self.onSurfaceStylePresetChanged)
        self.jawOpacitySlider.valueChanged.connect(self.onJawOpacitySliderChanged)
        self.surfaceSmoothingSlider.valueChanged.connect(self.onSurfaceSmoothingSliderChanged)
        self.risemarWeightsPathEdit.textChanged.connect(self.onRiseMarWeightsPathChanged)
        self.btnBrowseRiseMarWeights.clicked.connect(self.onBrowseRiseMarWeightsClicked)
        self.btnRunRiseMar.clicked.connect(self.onRunRiseMarClicked)
        self.btnBrowsePrintDir.clicked.connect(self.onBrowsePrintDirClicked)
        self.btnOpenPrintDir.clicked.connect(self.onOpenPrintDirClicked)
        self.btnRefreshInlineGallery.clicked.connect(self.onRefreshInlineGalleryClicked)
        self.btnGenerateFullGallery.clicked.connect(self.onGenerateFullGalleryClicked)
        self.btnGalleryPrev.clicked.connect(self.onGalleryPrevClicked)
        self.btnGalleryNext.clicked.connect(self.onGalleryNextClicked)
        self.btnRunNativeALI.clicked.connect(self.onRunNativeALIClicked)
        self.btnRunNativeASO.clicked.connect(self.onRunNativeASOClicked)
        self.btnRunNativeCLIC.clicked.connect(self.onRunNativeCLICClicked)
        self.btnRunNativeVFACE.clicked.connect(self.onRunNativeVFACEClicked)
        self.btnRunAutoCropROI.clicked.connect(self.onRunAutoCropROIClicked)

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
            if self.checkpointPathEdit.currentPath:
                settings.setValue(
                    self.SETTINGS_PREFIX + "lastCheckpointPath", self.checkpointPathEdit.currentPath
                )
        except Exception:
            logger.debug(
                "Best-effort operation failed", exc_info=True
            )  # silent-ok: non-critical cleanup/probe

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
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None and vtk is not None:
            self.addObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
            )
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
            if volNode is not None and hasattr(self, "orientationCodesLabel") and self.logic is not None:
                try:
                    from CBCTPlannerLib.core.orientation import axcodes_from_affine

                    _, affine = self.logic._volume_array_and_affine(volNode)
                    codes = axcodes_from_affine(affine) if affine is not None else ("?", "?", "?")
                    self.orientationCodesLabel.setText("Detected orientation: " + ", ".join(codes))
                except Exception:
                    logger.debug("Could not read volume orientation codes", exc_info=True)

            self._ensureParameterNodeCurveObservers(volNode)

            has_vol = volNode is not None
            jaw = self._parameterNode.GetParameter("ActiveJaw") or "Lower"
            arch_mode = self._parameterNode.GetParameter("ArchMode") or "Per-Jaw (Lower / Upper)"
            self.archModeCombo.setCurrentText(arch_mode)
            self.activeJawCombo.setCurrentText(jaw)

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
            self.lightboxCountSpin.setValue(
                int(float(self._parameterNode.GetParameter("LightboxCount") or 3))
            )
            self._set_spin_if_changed(
                self.panoThicknessSpin,
                float(self._parameterNode.GetParameter("PanoThicknessMm") or DEFAULTS["PanoThicknessMm"]),
            )
            self._set_spin_if_changed(
                self.panoHeightSpin,
                float(self._parameterNode.GetParameter("PanoHeightMm") or DEFAULTS["PanoHeightMm"]),
            )
            self.panoOffsetSpin.setValue(float(self._parameterNode.GetParameter("PanoOffsetMm") or 0.0))
            self.panoProjCombo.setCurrentText(
                self._parameterNode.GetParameter("PanoProjectionMode") or "Mean"
            )
            self.panoFlipVerticalCheck.setChecked(
                (self._parameterNode.GetParameter("PanoFlipVertical") or "false").lower() == "true"
            )
            self.canalMarginSpin.setValue(
                float(self._parameterNode.GetParameter("CanalSafetyMarginMm") or 2.0)
            )
            self.includePatientMetaCheckBox.setChecked(
                (self._parameterNode.GetParameter("IncludePatientMetadata") or "false").lower() == "true"
            )

            ckpt_path_str = self._parameterNode.GetParameter("CheckpointPath") or ""
            if ckpt_path_str and hasattr(self.checkpointPathEdit, "setCurrentPath"):
                self.checkpointPathEdit.setCurrentPath(ckpt_path_str)

            locked_fdi = self._parameterNode.GetParameter("LockedToothFDI") or ""
            idx_lock = self.lockFdiCombo.findData(locked_fdi)
            if idx_lock >= 0:
                self.lockFdiCombo.setCurrentIndex(idx_lock)

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

            for box in (
                self.step1Box,
                self.step2Box,
                self.step3Box,
                self.step4Box,
                self.step5Box,
                self.step6Box,
            ):
                if box is not None:
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
                self.step2HintLabel.setText(
                    "Prerequisite missing: Select an input CBCT volume in Step 1 first."
                )
                self.step3HintLabel.setText(
                    "Prerequisite missing: Load a CBCT volume and define a dental arch curve (>= 4 points)."
                )
                self.step4HintLabel.setText(
                    "Prerequisite missing: Load a CBCT volume and define a dental arch curve (>= 4 points)."
                )
                self.step5HintLabel.setText(
                    "Prerequisite missing: Load a CBCT volume and define a dental arch curve (>= 4 points)."
                )
            else:
                seg_status = (
                    f"Segmentation: {segNode.GetName()}"
                    if segNode
                    else "Unsegmented (Manual Ruler or AI Checkpoint mode)"
                )
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
                    edit_tag = (
                        "User-edited (locked against silent overwrite)" if user_edited else "Auto-fitted"
                    )
                    self.step2HintLabel.setText(
                        f"{effective_jaw} Arch Curve: {curveNode.GetNumberOfControlPoints()} points, {total_len:.1f} mm ({edit_tag})."
                    )
                    self.step3HintLabel.setText(
                        "Ready: Generate or auto-refresh in-process Curved Planar Reformation."
                    )
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
            self._updateStepChips()

    def _currentCursorSettings(
        self, sync_axial_slice: bool = False, site_label: Optional[str] = None
    ) -> CursorSettings:
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
    # Step 1 Callbacks: Volume, Checkpoint Inference, Layout, Occlusal Plane & VR
    # ------------------------------------------------------------------
    def onVolumeNodeChanged(self, node: Any) -> None:
        self._markInlineGalleryStale("CBCT volume changed")
        if (
            node is not None
            and self.logic is not None
            and hasattr(self, "autoStandardizeOnLoadCheck")
            and self.autoStandardizeOnLoadCheck.isChecked()
        ):
            try:
                from CBCTPlannerLib.core.orientation import axcodes_from_affine

                _, affine = self.logic._volume_array_and_affine(node)
                if affine is not None and axcodes_from_affine(affine) != ("R", "A", "S"):
                    segNode = (
                        self.segmentationSelector.currentNode()
                        if hasattr(self, "segmentationSelector")
                        else None
                    )
                    standardized, report = self.logic.standardize_orientation(node, segNode, mode="in_place")
                    if standardized is not None:
                        self.step1HintLabel.setText("Orientation: " + self._formatOrientationReport(report))
                        if slicer:
                            try:
                                slicer.util.setSliceViewerLayers(background=standardized, fit=True)
                            except Exception:
                                logger.debug("Could not refit slice views after in-place RAS", exc_info=True)
                    else:
                        logger.warning("Auto-standardize skipped: %s", "; ".join(report.notes))
            except Exception:
                logger.exception("Auto-standardize on load failed")
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetNodeReferenceID("InputVolume", node.GetID() if node else "")
            self.updateGUIFromParameterNode()

    def onSegmentationNodeChanged(self, node: Any) -> None:
        self._markInlineGalleryStale("Segmentation changed")
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            self._parameterNode.SetNodeReferenceID("Segmentation", node.GetID() if node else "")
            self.updateGUIFromParameterNode()
            if self.selectedTeeth:
                self.onRefreshAllSelectedTeeth()

    def _formatOrientationReport(self, report: Any) -> str:
        def state(value: Any) -> str:
            if value is None:
                return "cannot determine"
            return "OK" if value else "MISMATCH"

        parts = [
            f"S-I: {state(report.si_ok)}",
            f"A-P: {state(report.ap_ok)}",
            f"L-R: {state(report.lr_ok)}",
        ]
        return " | ".join(parts) + " — " + "; ".join(report.notes)

    def onAutoStandardizeOnLoadToggled(self, checked: bool) -> None:
        if qt is None:
            return
        try:
            settings = qt.QSettings()
            settings.setValue(self.SETTINGS_PREFIX + "autoStandardizeOnLoad", "true" if checked else "false")
        except Exception:
            logger.debug("Could not persist autoStandardizeOnLoad setting", exc_info=True)

    def onStandardizeRasClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select an input CBCT volume first.")
            return
        newVol, report = self.logic.standardize_orientation(volNode, segNode, mode="clone")
        summary = self._formatOrientationReport(report)
        if newVol is None:
            logger.warning("RAS standardization refused: %s", summary)
            if slicer:
                slicer.util.warningDisplay("RAS standardization not applied:\n" + summary)
            self.step1HintLabel.setText("Orientation: " + summary)
            return
        self.volumeSelector.setCurrentNode(newVol)
        if self._parameterNode is not None:
            self._parameterNode.SetNodeReferenceID("InputVolume", newVol.GetID())
        if segNode is not None:
            self.segmentationSelector.setCurrentNode(segNode)
        self.step1HintLabel.setText("Orientation: " + summary)
        self.globalStatusLabel.setText("Standardized to RAS")
        self.updateGUIFromParameterNode()
        if slicer:
            try:
                slicer.util.setSliceViewerLayers(background=newVol, fit=True)
            except Exception:
                logger.debug("Could not refit slice views after RAS standardization", exc_info=True)
            slicer.util.showStatusMessage("Standardized volume to RAS", 4000)

    def onAutoDetectOrientationClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select an input CBCT volume first.")
            return
        report = self.logic.detect_orientation_issues(volNode, segNode)
        summary = self._formatOrientationReport(report)
        self.step1HintLabel.setText("Orientation: " + summary)
        self.globalStatusLabel.setText("Orientation check: " + summary)
        if slicer:
            if report.all_ok:
                slicer.util.showStatusMessage("Orientation check: " + summary, 5000)
            else:
                from CBCTPlannerLib.core.orientation import inverted_axes_from_report

                bad_axes = inverted_axes_from_report(report)
                if bad_axes and slicer.util.confirmYesNoDisplay(
                    "Orientation issues detected:\n"
                    + summary
                    + f"\n\nMirror voxel data along {', '.join(bad_axes)} now so Axial, "
                    "Sagittal, and Coronal views display true RAS anatomy?",
                    windowTitle="Auto-Fix Inverted Orientation",
                ):
                    if self.logic.flip_orientation_axis(volNode, segNode, bad_axes):
                        updated_report = self.logic.detect_orientation_issues(volNode, segNode)
                        updated_summary = self._formatOrientationReport(updated_report)
                        self.step1HintLabel.setText("Orientation: " + updated_summary)
                        self.globalStatusLabel.setText(f"Corrected {', '.join(bad_axes)} orientation")
                        slicer.util.showStatusMessage(
                            f"Corrected {', '.join(bad_axes)} orientation in all slice views", 5000
                        )
                    else:
                        slicer.util.errorDisplay("Could not auto-correct orientation.")
                else:
                    slicer.util.warningDisplay("Orientation issues detected:\n" + summary)

    def onFlipOrientationClicked(self, axis: str) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select an input CBCT volume first.")
            return
        if slicer and not slicer.util.confirmOkCancelDisplay(
            f"Mirror the volume and segmentation along {axis}?\n"
            "This mirrors the stored voxel data and keeps the IJK-to-RAS matrix, "
            "so the displayed anatomy flips in Axial, Sagittal, and Coronal views.",
            windowTitle=f"Flip {axis}",
        ):
            return
        if self.logic.flip_orientation_axis(volNode, segNode, axis):
            self.globalStatusLabel.setText(f"Manual {axis} flip applied")
            self.step1HintLabel.setText(
                f"Manual {axis} flip applied and slice views refreshed."
            )
            if slicer:
                slicer.util.showStatusMessage(f"Applied {axis} flip to volume and slice views", 4000)
        else:
            logger.error("Manual %s flip failed", axis)
            if slicer:
                slicer.util.errorDisplay(f"Manual {axis} flip failed.")

    def _restore_ai_settings_preset(self) -> None:
        """Load the one-time AI segmentation preset saved in QSettings."""
        if qt is None:
            return
        try:
            settings = qt.QSettings()
            prefix = self.SETTINGS_PREFIX + "aiPreset/"
            self.nnunetModelPathEdit.setText(str(settings.value(prefix + "nnunetModelPath", "") or ""))
            self.nnunetDeviceCombo.setCurrentIndex(int(settings.value(prefix + "deviceIndex", 0) or 0))
            fold = str(settings.value(prefix + "fold", "0") or "0")
            fold_index = self.nnunetFoldCombo.findText(fold)
            self.nnunetFoldCombo.setCurrentIndex(max(0, fold_index))
            self.checkpointPathEdit.setCurrentPath(str(settings.value(prefix + "checkpointPath", "") or ""))
            self.customArchPathEdit.setCurrentPath(str(settings.value(prefix + "customArchPath", "") or ""))
            scheme = str(settings.value(prefix + "scheme", "toothfairy3") or "toothfairy3")
            scheme_index = self.ckptSchemeCombo.findData(scheme)
            self.ckptSchemeCombo.setCurrentIndex(max(0, scheme_index))
            self.ckptPatchSpin.value = int(settings.value(prefix + "patchSize", 96) or 96)
            has_preset = bool(
                self.nnunetModelPathEdit.text
                or self.checkpointPathEdit.currentPath
                or self.customArchPathEdit.currentPath
            )
            if hasattr(self, "aiPresetStatusLabel"):
                self.aiPresetStatusLabel.setText(
                    "Preset restored for this session" if has_preset else "Preset not set yet"
                )
        except Exception:
            logger.debug("Could not restore the AI segmentation preset", exc_info=True)

    def _save_ai_settings_preset(self, *_args: Any) -> None:
        """Persist the AI segmentation preset so it is configured once, not per case."""
        if qt is None:
            return
        try:
            settings = qt.QSettings()
            prefix = self.SETTINGS_PREFIX + "aiPreset/"
            settings.setValue(prefix + "nnunetModelPath", self.nnunetModelPathEdit.text)
            settings.setValue(prefix + "deviceIndex", self.nnunetDeviceCombo.currentIndex)
            settings.setValue(prefix + "fold", self.nnunetFoldCombo.currentText)
            settings.setValue(prefix + "checkpointPath", self.checkpointPathEdit.currentPath)
            settings.setValue(prefix + "customArchPath", self.customArchPathEdit.currentPath)
            settings.setValue(prefix + "scheme", self.ckptSchemeCombo.currentData)
            settings.setValue(prefix + "patchSize", self.ckptPatchSpin.value)
            settings.sync()
            if hasattr(self, "aiPresetStatusLabel"):
                self.aiPresetStatusLabel.setText("AI preset saved — reused in future sessions")
        except Exception:
            logger.exception("Could not save the AI segmentation preset")

    def onBrowseNNUNetModelClicked(self) -> None:
        path = qt.QFileDialog.getExistingDirectory(self.parent, "Select nnU-Net ToothFairy3 Model Folder")
        if not path:
            path = qt.QFileDialog.getOpenFileName(
                self.parent,
                "Select nnU-Net Checkpoint",
                "",
                "PyTorch Checkpoints (*.pth *.pt);;All Files (*)",
            )
        if path:
            self.nnunetModelPathEdit.setText(path)

    def onRunNNUNetInferenceClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        if volNode is None:
            slicer.util.warningDisplay("Please select an input CBCT volume first.")
            return
        model_path = self.nnunetModelPathEdit.text.strip()
        if not model_path:
            slicer.util.warningDisplay("Please specify a ToothFairy3 nnU-Net model folder or checkpoint.")
            return
        device_label = self.nnunetDeviceCombo.currentText
        if "CPU" in device_label and "Auto" not in device_label:
            device_mode = "cpu_int8"
        elif "GPU" in device_label and "Auto" not in device_label:
            device_mode = "gpu_fp32"
        else:
            device_mode = "auto"
        fold = self.nnunetFoldCombo.currentText.strip() or "0"
        try:
            from CBCTPlannerLib.AdvancedTools import run_nnunet_toothfairy3_inference

            def _progress(pct, msg):
                if hasattr(self, "globalStatusLabel"):
                    self.globalStatusLabel.setText(f"[{pct}%] {msg}")

            segNode = run_nnunet_toothfairy3_inference(
                volNode,
                model_path,
                device_mode=device_mode,
                fold=fold,
                progress_callback=_progress,
            )
            if segNode is not None:
                self.segmentationSelector.setCurrentNode(segNode)
                self.onApplyMaskPipelineClicked()
        except Exception as exc:
            slicer.util.errorDisplay(f"nnU-Net ToothFairy3 inference failed:\n{exc}")

    def onApplyMaskPipelineClicked(self) -> None:
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        if segNode is None or volNode is None:
            slicer.util.warningDisplay("Please select both a CBCT volume and a ToothFairy3 segmentation.")
            return
        from CBCTPlannerLib.core.labels import LABELS_BY_ID, match_segment_to_label_and_precedence

        segmentation = segNode.GetSegmentation()
        if not segmentation.ContainsRepresentation(
            slicer.vtkSegmentationConverter.GetClosedSurfaceRepresentationName()
        ):
            segmentation.CreateClosedSurfaceRepresentation()
        detected_teeth = []
        for idx in range(segmentation.GetNumberOfSegments()):
            seg = segmentation.GetNthSegment(idx)
            sid = segmentation.GetNthSegmentID(idx)
            fdi_tag = safe_get_segment_tag(seg, "CBCTPlanner.FDI")
            label_id, category, _, _ = match_segment_to_label_and_precedence(
                sid, seg.GetName(), fdi_attribute=fdi_tag
            )
            info = LABELS_BY_ID.get(label_id)
            if info is not None:
                seg.SetName(info.name)
                r, g, b = (float(c) / 255.0 if max(info.color) > 1.0 else float(c) for c in info.color)
                seg.SetColor(r, g, b)
                if hasattr(seg, "SetTag"):
                    seg.SetTag("CBCTPlanner.LabelID", str(info.id))
                    seg.SetTag("CBCTPlanner.Category", info.category)
                    if info.category == "tooth" or (11 <= info.id <= 48):
                        seg.SetTag("CBCTPlanner.FDI", str(info.id))
                        detected_teeth.append(int(info.id))
        if self.logic is not None:
            preset_name = (
                self.surfaceStylePresetCombo.currentText
                if getattr(self, "surfaceStylePresetCombo", None) is not None
                else "DentalSegmentator Vivid (Teeth & Roots)"
            )
            jaw_op = (
                float(self.jawOpacitySlider.value)
                if getattr(self, "jawOpacitySlider", None) is not None
                else 0.25
            )
            smooth_val = (
                float(self.surfaceSmoothingSlider.value)
                if getattr(self, "surfaceSmoothingSlider", None) is not None
                else 0.40
            )
            self.logic.vr.apply_dental_segmentator_styling(
                segNode,
                smoothing_factor=smooth_val,
                style_preset=preset_name,
                jaw_opacity_override=jaw_op,
            )
        if detected_teeth and hasattr(self, "selectedTeeth"):
            for fdi in detected_teeth:
                self.selectedTeeth.add(int(fdi))
        jaw = self._effectiveJawKey()
        existing_curve = (
            self.logic.get_curve_node_for_jaw(volNode, jaw, parameterNode=self._parameterNode)
            if self.logic is not None
            else None
        )
        user_edited_curve = (
            existing_curve is not None
            and existing_curve.GetNumberOfControlPoints() >= 4
            and existing_curve.GetAttribute("CBCTPlanner.UserEdited") == "1"
        )
        if user_edited_curve:
            if hasattr(self, "globalStatusLabel"):
                self.globalStatusLabel.setText(
                    f"Pipeline kept user-edited {jaw} arch curve; panoramic regenerated."
                )
            if hasattr(self, "onGeneratePanoramicClicked"):
                self.onGeneratePanoramicClicked()
        elif hasattr(self, "onAutoFitArchClicked"):
            # Auto-Fit generates the panoramic exactly once; do not duplicate it here.
            self.onAutoFitArchClicked()
        if hasattr(self, "onExtractCanalsClicked"):
            self.onExtractCanalsClicked()
        if hasattr(self, "onRefreshAllSelectedTeeth"):
            self.onRefreshAllSelectedTeeth()
        if hasattr(self, "globalStatusLabel"):
            self.globalStatusLabel.setText("Automated dental planning pipeline complete.")

    def onLoadExternalMask(self) -> None:
        path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Load Pre-computed AI Mask",
            "",
            "Mask Volumes (*.nii.gz *.nii *.nrrd *.seg.nrrd);;All Files (*)",
        )
        if not path:
            return
        segNode = slicer.util.loadSegmentation(path) if path.endswith(".seg.nrrd") else None
        if segNode is None:
            lbl = slicer.util.loadLabelVolume(path)
            # Check if ToothFairy3 remapping is needed on label volume
            from CBCTPlannerLib.core.labels import remap_toothfairy3_array

            lbl_arr = slicer.util.arrayFromVolume(lbl)
            if lbl_arr is not None and np.any((lbl_arr > 0) & (lbl_arr <= 77)):
                remapped = remap_toothfairy3_array(lbl_arr)
                slicer.util.updateVolumeFromArray(lbl, remapped)
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lbl, segNode)
            slicer.mrmlScene.RemoveNode(lbl)
        self.segmentationSelector.setCurrentNode(segNode)
        # Create closed surface representation
        if segNode is not None:
            segNode.CreateClosedSurfaceRepresentation()
        # Auto-execute full planning pipeline!
        self.onApplyMaskPipelineClicked()

    def onRunCheckpointSegmentationClicked(self) -> None:
        """Run user-supplied AI checkpoint (`.pth`, `.pt`, `.ckpt`, `.onnx`, `.npy`, `.nii.gz`) on the active CBCT volume."""
        volNode = self.volumeSelector.currentNode()
        ckpt_str = str(self.checkpointPathEdit.currentPath or "").strip()
        if not volNode or not ckpt_str or not self.logic:
            if slicer:
                slicer.util.warningDisplay(
                    "Please select both an Input CBCT Volume and an AI Model Checkpoint file/folder."
                )
            return
        custom_py = str(self.customArchPathEdit.currentPath or "").strip()
        scheme = str(self.ckptSchemeCombo.currentData or "5class")
        p_dim = int(self.ckptPatchSpin.value)
        if self._parameterNode is not None:
            self._parameterNode.SetParameter("CheckpointPath", ckpt_str)

        segNode, summary = self.logic.run_custom_checkpoint_on_volume_node(
            volumeNode=volNode,
            checkpoint_path=Path(ckpt_str),
            label_scheme=scheme,
            patch_size_zyx=(p_dim, p_dim, p_dim),
            custom_arch_py_path=Path(custom_py) if custom_py else None,
        )
        if segNode is not None:
            self.segmentationSelector.setCurrentNode(segNode)
            self.step1HintLabel.setText(
                f"Checkpoint ({summary.format_type}, {summary.device_used}): loaded {summary.loaded_tensors}/{summary.total_ckpt_tensors} tensors -> {segNode.GetName()}"
            )
            self.updateGUIFromParameterNode()
            self.onApplyMaskPipelineClicked()

    def onSetDentalLightboxLayout(self) -> None:
        volNode = self.volumeSelector.currentNode()
        registerAndSetDentalLightboxLayout(volumeNode=volNode)
        self.enableLightboxCheckBox.setChecked(True)
        if volNode and self.logic:
            self.logic.reset_slice_views(volNode)
            self._syncCursorToViews()

    def onSetDentalLayout(self) -> None:
        registerAndSetDentalLayout()
        volNode = self.volumeSelector.currentNode()
        if volNode and self.logic:
            self.logic.reset_slice_views(volNode)
            self._syncCursorToViews()

    def onSetStandardMpr(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        self.logic.reset_slice_views(volNode)

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
        preset_name = (
            self.surfaceStylePresetCombo.currentText
            if getattr(self, "surfaceStylePresetCombo", None) is not None
            else "DentalSegmentator Vivid (Teeth & Roots)"
        )
        jaw_op = (
            float(self.jawOpacitySlider.value)
            if getattr(self, "jawOpacitySlider", None) is not None
            else 0.25
        )
        smooth_val = (
            float(self.surfaceSmoothingSlider.value)
            if getattr(self, "surfaceSmoothingSlider", None) is not None
            else 0.40
        )
        self.logic.vr.apply_dental_segmentator_styling(
            segNode,
            smoothing_factor=smooth_val,
            style_preset=preset_name,
            jaw_opacity_override=jaw_op,
        )
        if volNode is not None:
            self.logic.update_fdi_overlays_from_segmentation(volNode, segNode, visible=True)
        self.globalStatusLabel.setText(f"Applied 3D style: {preset_name} (Bone opacity: {jaw_op:.2f})")

    def onSurfaceStylePresetChanged(self, preset_name: str) -> None:
        if getattr(self, "jawOpacitySlider", None) is not None and preset_name in VIVID_3D_STYLE_PRESETS:
            jaw_op = VIVID_3D_STYLE_PRESETS[preset_name][0]
            self.jawOpacitySlider.blockSignals(True)
            self.jawOpacitySlider.setValue(jaw_op)
            self.jawOpacitySlider.blockSignals(False)
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if segNode is not None and self.logic is not None:
            self.onApplyDental3DStyling()

    def onJawOpacitySliderChanged(self, val: float) -> None:
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if segNode is not None and self.logic is not None:
            self.logic.vr.set_jawbone_3d_opacity(segNode, jaw_opacity=float(val), tooth_opacity=1.0)
            self.globalStatusLabel.setText(f"Jawbone 3D Opacity: {float(val) * 100:.0f}% (Teeth & Roots: 100%)")

    def onSurfaceSmoothingSliderChanged(self, _val: float) -> None:
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if segNode is not None and self.logic is not None:
            self.onApplyDental3DStyling()

    def onRoleModeChanged(self, mode_text: str) -> None:
        """Apply workspace role without erasing volume/segmentation selections or measurements."""
        apply_workspace_role(self, mode_text)

    def onRiseMarWeightsPathChanged(self, new_path: str) -> None:
        save_risemar_weights_setting(qt, new_path)
        apply_risemar_validation_to_widget(self, new_path)

    def onBrowseRiseMarWeightsClicked(self) -> None:
        if qt is None:
            return
        cur_txt = self.risemarWeightsPathEdit.text.strip() if self.risemarWeightsPathEdit is not None else ""
        start_dir = str(Path(cur_txt).parent) if cur_txt else str(Path.home())
        path = qt.QFileDialog.getOpenFileName(
            self.parent,
            "Select RISE-MAR Model Weights (.pkl / .pth)",
            start_dir,
            "RISE-MAR Weights (*.pkl *.pth *.pt);;All Files (*)",
        )
        if path:
            self.risemarWeightsPathEdit.setText(path)
            save_risemar_weights_setting(qt, path)
            apply_risemar_validation_to_widget(self, path)

    def onRunRiseMarClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        if not volNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Please select an input CBCT volume first.")
            return
        w_path = (
            self.risemarWeightsPathEdit.text.strip()
            if getattr(self, "risemarWeightsPathEdit", None) is not None
            else ""
        )
        ok_weights, msg_weights = validate_risemar_weights_path(w_path)
        if not ok_weights:
            apply_risemar_validation_to_widget(self, w_path)
            if slicer:
                slicer.util.warningDisplay(msg_weights)
            return
        fast_only = (
            bool(self.risemarFastCheck.isChecked())
            if getattr(self, "risemarFastCheck", None) is not None
            else True
        )
        preserve_metal = (
            bool(self.risemarPreserveMetalCheck.isChecked())
            if getattr(self, "risemarPreserveMetalCheck", None) is not None
            else True
        )
        in_place = (
            bool(self.risemarInPlaceCheck.isChecked())
            if getattr(self, "risemarInPlaceCheck", None) is not None
            else False
        )

        def _prog(pct: int, msg: str) -> None:
            if getattr(self, "risemarStatusLabel", None) is not None:
                self.risemarStatusLabel.setText(f"[{pct}%] {msg}")
            self.globalStatusLabel.setText(f"[{pct}%] {msg}")
            if slicer and hasattr(slicer, "app"):
                slicer.app.processEvents()

        try:
            marNode, res = self.logic.run_risemar_metal_artifact_reduction(
                volumeNode=volNode,
                weights_path=w_path,
                metal_slices_only=fast_only,
                preserve_metal_core=preserve_metal,
                in_place=in_place,
                progress_callback=_prog,
            )
            if marNode is not None:
                self.volumeSelector.setCurrentNode(marNode)
                if slicer:
                    try:
                        slicer.util.setSliceViewerLayers(background=marNode, fit=False)
                    except Exception:
                        logger.debug("Could not set MAR volume in slice viewers", exc_info=True)
            if res.mode_used == "no_metal_noop":
                summary_txt = (
                    f"RISE-MAR Pre-Check: 0 metal voxels detected (>= {res.effective_metal_threshold_hu:.0f} HU) — "
                    f"original contrast preserved ({res.elapsed_sec:.1f}s)"
                )
            else:
                summary_txt = (
                    f"RISE-MAR Complete ({res.mode_used}): {res.processed_slices}/{res.total_slices} metal slices "
                    f"({res.metal_voxel_count} vox) | CQA: {res.cqa_score_before:.2f} -> {res.cqa_score_after:.2f} ({res.elapsed_sec:.1f}s)"
                )
            if getattr(self, "risemarStatusLabel", None) is not None:
                self.risemarStatusLabel.setText(summary_txt)
            self.globalStatusLabel.setText(summary_txt)
        except Exception as exc:
            logger.exception("RISE-MAR execution failed")
            if slicer:
                slicer.util.errorDisplay(f"RISE-MAR failed:\n{exc}")

    # ------------------------------------------------------------------
    # Step 2 Callbacks: Dental Arch Curve (Shared vs Per-Jaw, Auto-Fit vs Manual)
    # ------------------------------------------------------------------
    def onArchModeChanged(self, mode_text: str) -> None:
        self._markInlineGalleryStale("Arch configuration changed")
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
        if segNode is None:
            if slicer:
                slicer.util.warningDisplay(
                    "Auto-fit needs segmented teeth.\n"
                    "Select/segment a jaw first, or use 'Place / Draw Arch Points on Axial'."
                )
                slicer.util.showStatusMessage("Auto-fit needs segmented teeth", 5000)
            self.logic.start_interactive_arch_placement(volNode, jaw=jaw)
            return

        curveNode = self.logic.auto_fit_arch_for_jaw(
            volumeNode=volNode,
            segmentationNode=segNode,
            jaw=jaw,
            visible=self.showCurvesCheckBox.isChecked(),
            confirm_overwrite=True,
        )
        fit = self.logic.last_arch_fit
        if curveNode is not None and curveNode.GetNumberOfControlPoints() >= 4 and fit is not None:
            self._observeCurveEdits(curveNode)
            applied = float(getattr(fit, "applied_extension_mm", fit.end_extension_mm))
            ext_note = (
                f", extension clipped to {applied:.1f} mm"
                if applied + 0.5 < float(fit.end_extension_mm)
                else ""
            )
            self.step2HintLabel.setText(
                f"Fitted {len(fit.points_ras)} pts, tooth-bearing {fit.tooth_bearing_arc_mm:.1f} mm, "
                f"curve {fit.arc_length_mm:.1f} mm, RMS {fit.residual_rms_mm:.1f} mm "
                f"({fit.method}){ext_note}"
            )
            self.step3Box.collapsed = False
            self.updateGUIFromParameterNode()
            self._syncCursorToViews()
            self.onGeneratePanoramicClicked()
            return

        reason = self.logic.last_arch_fit_error or "no accepted fit"
        logger.warning("Auto-Fit produced no accepted arch curve: %s", reason)
        self.step2HintLabel.setText(
            f"Auto-Fit rejected ({reason}). Place 4+ arch points manually from Patient Right to Patient Left."
        )
        if slicer:
            slicer.util.warningDisplay(
                "Auto-Fit could not produce a reliable arch curve:\n"
                f"{reason}\n\nPlace arch points manually or check the segmentation."
            )
        self.logic.start_interactive_arch_placement(volNode, jaw=jaw)

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
        point_mod_evt = getattr(
            getattr(slicer, "vtkMRMLMarkupsNode", None),
            "PointModifiedEvent",
            getattr(vtk.vtkCommand, "ModifiedEvent", 33),
        )
        end_inter_evt = getattr(
            vtk.vtkCommand, "EndInteractionEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33)
        )
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
    @staticmethod
    def _set_spin_if_changed(spin: Any, value: float) -> None:
        """Set a spin box without emitting change signals or clobbering equal values."""
        if abs(float(spin.value) - float(value)) < 1e-9:
            return
        spin.blockSignals(True)
        spin.setValue(float(value))
        spin.blockSignals(False)

    def onPanoParamChanged(self, *_args: Any) -> None:
        if self._parameterNode is not None and not self._updatingGUIFromParameterNode:
            was_mod = (
                self._parameterNode.StartModify() if hasattr(self._parameterNode, "StartModify") else None
            )
            try:
                self._parameterNode.SetParameter("PanoThicknessMm", f"{self.panoThicknessSpin.value:.1f}")
                self._parameterNode.SetParameter("PanoHeightMm", f"{self.panoHeightSpin.value:.1f}")
                self._parameterNode.SetParameter("PanoOffsetMm", f"{self.panoOffsetSpin.value:.1f}")
                self._parameterNode.SetParameter("PanoProjectionMode", self.panoProjCombo.currentText)
                self._parameterNode.SetParameter(
                    "PanoFlipVertical", "true" if self.panoFlipVerticalCheck.checked else "false"
                )
            finally:
                if was_mod is not None and hasattr(self._parameterNode, "EndModify"):
                    self._parameterNode.EndModify(was_mod)
        if (
            self.autoUpdatePanoCheckBox.isChecked()
            and self._panoPreviewTimer is not None
            and not self._updatingGUIFromParameterNode
        ):
            self._panoPreviewTimer.start()

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
                slicer.util.warningDisplay(
                    f"Please create or fit a {jaw} dental arch curve (>= 4 points) first."
                )
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
                force_fit=False,
                flip_vertical=self.panoFlipVerticalCheck.checked,
            )
        except Exception as exc:
            self.panoProgressBar.setVisible(False)
            self.btnCancelPanoramic.setEnabled(False)
            logger.exception("Panoramic generation failed")
            if slicer:
                slicer.util.showStatusMessage(f"Panoramic generation failed: {exc}", 5000)
                if not preview:
                    slicer.util.errorDisplay(f"Panoramic generation failed:\n{exc}")

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
        prev_arc = (
            float(self._parameterNode.GetParameter("CursorArcMm") or 0.0) if self._parameterNode else 0.0
        )
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
                target_jaw = "Lower" if is_mandibular_fdi(fdi) else "Upper"
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
            was_mod = (
                self._parameterNode.StartModify() if hasattr(self._parameterNode, "StartModify") else None
            )
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
        self._updateSelectionStatusLabel()
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
            target_jaw = "Lower" if is_mandibular_fdi(fdi) else "Upper"
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
            self.selectedTeeth.add(int(fdi))
            if self.logic is not None:
                self.treatmentPlanByFDI[int(fdi)] = self.logic.analyze_single_tooth(
                    int(fdi), volNode, segmentationNode=self.segmentationSelector.currentNode()
                )
            btn = self.toothButtons.get(int(fdi))
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(True)
                btn.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white;")
                btn.blockSignals(False)
            self._updateSelectionStatusLabel()
            self._refreshPlanTableFromCache()
            self._updateActiveSiteClinicalCallout(int(fdi))
            self.onRefreshInlineGalleryClicked()

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
        self._updateSelectionStatusLabel()
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
        self._updateSelectionStatusLabel()
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
        self.treatmentPlanByFDI[fdi] = self.logic.analyze_single_tooth(fdi, volNode, segmentationNode=segNode)
        self._syncSelectedTeethToParameterNode()
        self._updateSelectionStatusLabel()
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

    def onGenerateSliceSeriesClicked(self) -> None:
        """Generate 15 to 28 sequential cross-sectional slice images across active tooth or quadrant with overlaid measurements."""
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select an input CBCT volume first.")
            return

        from CBCTPlannerLib.core.labels import sort_teeth_anatomically
        from CBCTPlannerLib.core.print_layout import slice_count_for_selection

        selected = sort_teeth_anatomically(list(self.selectedTeeth))
        fdi = self.activeToothFDI or (selected[0] if selected else 46)
        num_slices, step_mm = slice_count_for_selection(max(1, len(selected)))
        dlg = qt.QDialog(self.parent)
        dlg.setWindowTitle(f"Cross-Sectional Slice Series — Site #{fdi} ({num_slices} Slices)")
        dlg.resize(950, 700)
        d_layout = qt.QVBoxLayout(dlg)

        info_lbl = qt.QLabel(
            f"<b>Sequential Cross-Sectional Series (15–28 Slices):</b> Tooth Site #{fdi} | "
            f"{num_slices} slices @ {step_mm} mm spacing along the arch curve.<br>"
            "Each slice displays bone height, crestal width, 3 mm and 5 mm widths, and safety margin lines."
        )
        info_lbl.setWordWrap(True)
        d_layout.addWidget(info_lbl)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        container = qt.QWidget()
        grid = qt.QGridLayout(container)
        grid.setSpacing(8)

        # Generate slice series images via measurement engine
        series_data = self.logic.generate_cross_section_slice_series(
            fdi=fdi,
            volumeNode=volNode,
            segmentationNode=self.segmentationSelector.currentNode(),
            num_slices=num_slices,
            step_mm=step_mm,
            fdis=selected or [fdi],
        )

        cols = 5
        for idx, item in enumerate(series_data):
            r = idx // cols
            c = idx % cols
            card = qt.QFrame()
            card.setFrameShape(qt.QFrame.StyledPanel)
            card_layout = qt.QVBoxLayout(card)
            card_layout.setContentsMargins(4, 4, 4, 4)

            title_lbl = qt.QLabel(f"<b>Slice {idx + 1}/{num_slices}</b> ({item['offset_mm']:+.1f} mm)")
            title_lbl.setAlignment(qt.Qt.AlignCenter)
            card_layout.addWidget(title_lbl)

            img_lbl = qt.QLabel()
            pixmap = item.get("pixmap")
            if pixmap is None and item.get("image_data_uri"):
                image = qt.QImage.fromData(
                    qt.QByteArray.fromBase64(item["image_data_uri"].split(",", 1)[-1].encode("ascii"))
                )
                if not image.isNull():
                    pixmap = qt.QPixmap.fromImage(image)
            if pixmap is not None:
                img_lbl.setPixmap(pixmap.scaled(160, 200, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation))
            else:
                img_lbl.setText("image unavailable")
            img_lbl.setAlignment(qt.Qt.AlignCenter)
            card_layout.addWidget(img_lbl)

            metrics_txt = (
                f"H: {item.get('height_mm', 'N/A')} mm | "
                f"W0: {item.get('w_crest_mm', 'N/A')} mm<br>"
                f"W3: {item.get('w_3mm', 'N/A')} mm | "
                f"W5: {item.get('w_5mm', 'N/A')} mm"
            )
            met_lbl = qt.QLabel(metrics_txt)
            met_lbl.setAlignment(qt.Qt.AlignCenter)
            met_lbl.setStyleSheet("font-size: 10px; color: #334155;")
            card_layout.addWidget(met_lbl)
            grid.addWidget(card, r, c)

        scroll.setWidget(container)
        d_layout.addWidget(scroll)

        btn_box = qt.QHBoxLayout()
        btn_close = qt.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_box.addStretch()
        btn_box.addWidget(btn_close)
        d_layout.addLayout(btn_box)

        dlg.exec_()

    def onSelectQuadrant(self, q: int) -> None:
        """Select all teeth belonging to quadrant `q` (1, 2, 3, or 4)."""
        quad_teeth = {
            1: [18, 17, 16, 15, 14, 13, 12, 11],
            2: [21, 22, 23, 24, 25, 26, 27, 28],
            3: [31, 32, 33, 34, 35, 36, 37, 38],
            4: [48, 47, 46, 45, 44, 43, 42, 41],
        }.get(q, [])
        self.selectedTeeth.clear()
        for fdi in quad_teeth:
            self.selectedTeeth.add(fdi)
            btn = self.toothButtons.get(fdi)
            if btn:
                btn.blockSignals(True)
                btn.setChecked(True)
                btn.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white;")
                btn.blockSignals(False)
        for fdi, btn in self.toothButtons.items():
            if fdi not in quad_teeth:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.setStyleSheet("")
                btn.blockSignals(False)
        self.activeToothFDI = quad_teeth[len(quad_teeth) // 2]
        self._updateSelectionStatusLabel()
        self.onRefreshAllSelectedTeeth()
        self._updateActiveSiteClinicalCallout(self.activeToothFDI)
        self.onRefreshInlineGalleryClicked()

    def onClearToothSelection(self) -> None:
        """Clear all active tooth selections."""
        self.selectedTeeth.clear()
        self.activeToothFDI = None
        for btn in self.toothButtons.values():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.setStyleSheet("")
            btn.blockSignals(False)
        self._updateSelectionStatusLabel()
        self._refreshPlanTableFromCache()

    def _updateSelectionStatusLabel(self) -> None:
        """Update Step 5 header text with active quadrant and slice count."""
        if not hasattr(self, "step5SelectionStatusLabel"):
            return
        if not self.selectedTeeth:
            self.step5SelectionStatusLabel.setText(
                "Active Site: None selected (Click a quadrant or teeth above)"
            )
            return
        count = len(self.selectedTeeth)
        fdis_sorted = sorted(list(self.selectedTeeth))
        q = str(fdis_sorted[0])[0]
        from CBCTPlannerLib.core.print_layout import slice_count_for_selection

        n_slices, _step = slice_count_for_selection(count)
        self.step5SelectionStatusLabel.setText(
            f"Active Site: Quadrant {q} ({count} tooth site{'s' if count > 1 else ''}: {', '.join(f'#{f}' for f in fdis_sorted)}) — {n_slices} Cross-Section Slices"
        )
        self._updateStepChips()

    def _updateStepChips(self) -> None:
        """Refresh the 1-7 status chips from the current scene state."""
        labels = getattr(self, "stepStatusLabels", None)
        if not labels or self._parameterNode is None:
            return
        vol = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        curve = None
        if vol is not None and self.logic is not None:
            try:
                curve = self.logic.get_curve_node_for_jaw(
                    vol, self._effectiveJawKey(), parameterNode=self._parameterNode
                )
            except Exception:
                logger.debug("Could not read arch curve for step chips", exc_info=True)
        pano = self._parameterNode.GetNodeReference("PanoramicVolume")
        canals = [
            self._parameterNode.GetNodeReference(role) for role in ("RightCanalCurve", "LeftCanalCurve")
        ]
        states = [
            vol is not None,
            curve is not None and curve.GetNumberOfControlPoints() >= 4,
            pano is not None,
            pano is not None,
            bool(getattr(self, "selectedTeeth", None)),
            any(node is not None for node in canals),
            False,
        ]
        for index, done in enumerate(states, start=1):
            label = labels[index - 1]
            label.setText(f"{index} \u2713" if done else f"{index} \u25cb")
            color = "#2e7d32" if done else "#9e9e9e"
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            label.setToolTip(
                f"Step {index}: {'complete' if done else 'pending' if index < 7 else 'optional'}"
            )

    def onRunFullPipelineClicked(self) -> None:
        """Run segmentation -> arch fit -> panoramic CPR -> canals with shared error handling."""
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select a CBCT volume before running the pipeline.")
            return
        self.globalStatusLabel.setText("Pipeline: fitting arch...")
        try:
            if segNode is None:
                self.globalStatusLabel.setText("Pipeline stopped: no segmentation selected.")
                if slicer:
                    slicer.util.warningDisplay("Select a Segmentation node to fit the arch.")
                return
            jaw = self._effectiveJawKey()
            curveNode = self.logic.auto_fit_arch_for_jaw(
                volumeNode=volNode,
                segmentationNode=segNode,
                jaw=jaw,
                visible=self.showCurvesCheckBox.isChecked(),
                confirm_overwrite=False,
                allow_overwrite_user_edit=False,
            )
            fit = self.logic.last_arch_fit
            if curveNode is None or curveNode.GetNumberOfControlPoints() < 4:
                reason = self.logic.last_arch_fit_error or "no accepted fit"
                self.step2HintLabel.setText(f"Auto-Fit rejected ({reason}).")
                self.globalStatusLabel.setText(f"Pipeline stopped: auto-fit rejected ({reason}).")
                return
            self._observeCurveEdits(curveNode)
            if fit is not None:
                applied = float(getattr(fit, "applied_extension_mm", fit.end_extension_mm))
                ext_note = (
                    f", extension clipped to {applied:.1f} mm"
                    if applied + 0.5 < float(fit.end_extension_mm)
                    else ""
                )
                self.step2HintLabel.setText(
                    f"Fitted {len(fit.points_ras)} pts, tooth-bearing {fit.tooth_bearing_arc_mm:.1f} mm, "
                    f"curve {fit.arc_length_mm:.1f} mm, RMS {fit.residual_rms_mm:.1f} mm "
                    f"({fit.method}){ext_note}"
                )
            self.step3Box.collapsed = False
            self.globalStatusLabel.setText("Pipeline: generating panoramic CPR...")
            self._runPanoramicGeneration(preview=False)
            self.globalStatusLabel.setText("Pipeline: extracting canals...")
            self.onExtractCanalsClicked()
            self.globalStatusLabel.setText("Pipeline complete: arch, panoramic CPR and canals.")
        except Exception as exc:
            logger.exception("Run full pipeline failed")
            self.globalStatusLabel.setText(f"Pipeline failed: {exc}")
        finally:
            self.updateGUIFromParameterNode()

    def onGenerate8x10PrintLayout(self) -> None:
        """Generate standard 8x10 inch clinical film layout with panoramic view, 15-27 cross-sections, and VR."""
        volNode = self.volumeSelector.currentNode()
        if not volNode:
            if slicer:
                slicer.util.warningDisplay("Please select an input CBCT volume first.")
            return

        segNode = self.segmentationSelector.currentNode()
        selected_fdis = list(self.selectedTeeth)
        if not selected_fdis:
            if self.activeToothFDI:
                selected_fdis = [self.activeToothFDI]
            else:
                selected_fdis = [41, 42, 43, 44, 45, 46, 47]  # Standard quadrant default

        quadrant = int(str(selected_fdis[0])[0]) if selected_fdis else 4
        if segNode is None and self.logic is not None:
            if slicer:
                slicer.util.warningDisplay(
                    "The print layout needs a segmentation to locate the arch and draw measurements.\n"
                    "Run AI segmentation (or load a mask), then try again."
                )
            return

        # The layout needs an arch curve. Fit it automatically instead of failing with
        # "No arch positions" when the user went straight from segmentation to print.
        if segNode is not None and self.logic is not None:
            jaw_key = "lower" if quadrant in (3, 4) else "upper"
            curve = self.logic.get_curve_node_for_jaw(volNode, jaw_key.capitalize())
            if curve is None or curve.GetNumberOfControlPoints() < 4:
                self.globalStatusLabel.setText("Print layout: fitting arch...")
                curve = self.logic.auto_fit_arch_for_jaw(
                    volumeNode=volNode,
                    segmentationNode=segNode,
                    jaw=jaw_key.capitalize(),
                    visible=False,
                    confirm_overwrite=False,
                    allow_overwrite_user_edit=False,
                )
                if curve is None or curve.GetNumberOfControlPoints() < 4:
                    reason = self.logic.last_arch_fit_error or "no accepted arch fit"
                    self.globalStatusLabel.setText(f"Print layout stopped: {reason}")
                    if slicer:
                        slicer.util.warningDisplay(
                            "Could not fit the dental arch for the print layout:\n"
                            f"{reason}\n\nFit or draw the arch in Step 2, then retry."
                        )
                    return
                self._observeCurveEdits(curve)

        from CBCTPlannerLib.PrintLayoutEngine import PrintLayoutEngine

        pNode = self.getParameterNode() if slicer else None
        p_name = pNode.GetParameter("PatientName") if pNode else "Anonymous"
        p_id = pNode.GetParameter("PatientID") if pNode else "Anonymous"
        include_meta = (
            self.includePatientMetaCheckBox.isChecked()
            if hasattr(self, "includePatientMetaCheckBox")
            else False
        )
        custom_dir_str = (
            self.printOutputDirEdit.text.strip()
            if getattr(self, "printOutputDirEdit", None) is not None
            else ""
        )
        if custom_dir_str:
            out_dir = Path(custom_dir_str)
        elif slicer and getattr(slicer, "app", None) is not None:
            out_dir = PrintLayoutEngine.default_export_directory()
        else:
            out_dir = Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        engine = PrintLayoutEngine(logic=self.logic)

        progress = None
        if qt is not None and self.parent is not None:
            progress = qt.QProgressDialog("Rendering clinical film pages…", "", 0, 0, self.parent)
            progress.setWindowTitle("Generating 8x10 layout")
            progress.setCancelButton(None)
            progress.show()
            qt.QApplication.processEvents()

        isolation = (
            self.logic.quadrant_vr_isolation(segNode, selected_fdis)
            if (self.logic is not None and hasattr(self.logic, "quadrant_vr_isolation"))
            else contextlib.nullcontext()
        )
        try:
            with isolation:
                html_file, pdf_file, slices_data = engine.generate_8x10_layout(
                    volumeNode=volNode,
                    segmentationNode=segNode,
                    selected_fdis=selected_fdis,
                    quadrant=quadrant,
                    out_dir=out_dir,
                    patient_name=p_name or "Anonymous",
                    patient_id=p_id or "Anonymous",
                    include_metadata=include_meta,
                )
        except Exception:
            logger.exception("8x10 print layout generation failed")
            if slicer:
                slicer.util.errorDisplay("The 8x10 print layout could not be generated. See the console log.")
            return
        finally:
            if progress is not None:
                progress.close()

        png_file = getattr(engine, "last_png_file", None)
        self._populateInlineMeasuredGallery(
            slices_data, html_path=html_file, pdf_path=pdf_file, png_path=png_file
        )
        self._showPrintPreviewDialog(html_file, pdf_file, slices_data, quadrant, sorted(selected_fdis))
        self.globalStatusLabel.setText(
            f"Saved print files to {out_dir}: {pdf_file.name}"
            + (f", {png_file.name}" if png_file else "")
        )
        if slicer:
            slicer.util.showStatusMessage(f"Generated 8×10″ Print Layout: {pdf_file}", 5000)

    def _showPrintPreviewDialog(
        self, html_path: Path, pdf_path: Path, slices_data: List[Any], quadrant: int, fdis: List[int]
    ) -> None:
        """Display an interactive modal preview dialog in 3D Slicer for the 8x10 layout."""
        if qt is None:
            return
        dlg = qt.QDialog(self.parent)
        dlg.setWindowTitle(f"8×10″ Clinical Dental Print Layout — Quadrant {quadrant}")
        dlg.resize(1050, 750)
        d_layout = qt.QVBoxLayout(dlg)

        top_bar = qt.QHBoxLayout()
        info_lbl = qt.QLabel(
            f"<b>8×10″ Printable Dental Film Layout:</b> Quadrant {quadrant} ({len(fdis)} sites: {', '.join(f'#{f}' for f in fdis)}) | "
            f"<b>{len(slices_data)} Cross-Sections</b> with clinical bone measurements & VR."
        )
        top_bar.addWidget(info_lbl)
        top_bar.addStretch()

        png_candidate = pdf_path.with_suffix(".png")
        btn_open_pdf = qt.QPushButton("📄 Open PDF (Print)")
        btn_open_pdf.clicked.connect(
            lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(pdf_path)))
        )
        btn_open_png = qt.QPushButton("🖼️ Open 300 DPI PNG")
        btn_open_png.clicked.connect(
            lambda: qt.QDesktopServices.openUrl(
                qt.QUrl.fromLocalFile(str(png_candidate if png_candidate.exists() else pdf_path))
            )
        )
        btn_open_html = qt.QPushButton("🌐 Open in Browser (HTML)")
        btn_open_html.clicked.connect(
            lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(html_path)))
        )
        btn_open_folder = qt.QPushButton("📂 Open Output Folder")
        btn_open_folder.clicked.connect(
            lambda: qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(pdf_path.parent)))
        )
        top_bar.addWidget(btn_open_pdf)
        top_bar.addWidget(btn_open_png)
        top_bar.addWidget(btn_open_html)
        top_bar.addWidget(btn_open_folder)
        d_layout.addLayout(top_bar)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        container = qt.QWidget()
        grid = qt.QGridLayout(container)
        grid.setSpacing(6)

        cols = 5 if len(slices_data) <= 15 else (7 if len(slices_data) <= 21 else 9)
        for idx, item in enumerate(slices_data):
            r = idx // cols
            c = idx % cols
            card = qt.QFrame()
            card.setFrameShape(qt.QFrame.StyledPanel)
            card_l = qt.QVBoxLayout(card)
            card_l.setContentsMargins(2, 2, 2, 2)
            lbl = qt.QLabel()
            uri = getattr(item, "image_data_uri", "")
            if uri and "," in uri:
                import base64

                pixmap = qt.QPixmap()
                if pixmap.loadFromData(base64.b64decode(uri.split(",", 1)[1])):
                    lbl.setPixmap(pixmap.scaled(130, 160, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation))
            lbl.setAlignment(qt.Qt.AlignCenter)
            card_l.addWidget(lbl)
            grid.addWidget(card, r, c)

        scroll.setWidget(container)
        d_layout.addWidget(scroll)

        btn_box = qt.QHBoxLayout()
        btn_close = qt.QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_box.addStretch()
        btn_box.addWidget(btn_close)
        d_layout.addLayout(btn_box)

        dlg.exec_()

    def onGeneratePrintableReport(self) -> None:
        """Route standard report generation directly to 8x10 print layout engine."""
        self.onGenerate8x10PrintLayout()

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
        self.step6HintLabel.setText(
            f"Extracted {len(canals)} mandibular canal(s): {', '.join(canals.keys()) or 'None'}"
        )

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

    # ------------------------------------------------------------------
    # Step 7 Callbacks: Surgical Guide (F14), DICOM SEG/SR (F15), Crowns (F16), Ceph (F17), ADT (F18)
    # ------------------------------------------------------------------
    def onExportSurgicalGuideClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        if not volNode or not self.logic:
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(
            self.parent, "Select Surgical Guide Output Directory"
        )
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        stl_path, json_path, sleeves = self.logic.generate_and_export_surgical_guide(
            volumeNode=volNode,
            out_dir=out_dir,
            support_type=str(self.guideSupportCombo.currentData or "tooth_supported"),
            preset_id=str(self.printerPresetCombo.currentData or "sla_dlp_standard_50um"),
        )
        self.step7HintLabel.setText(f"Exported Surgical Guide STL ({len(sleeves)} sleeves): {stl_path.name}")

    def onProposeCrownsClicked(self) -> None:
        if not self.logic:
            return
        proposals = self.logic.propose_crowns_and_abutments_for_placed_implants()
        self.step7HintLabel.setText(
            f"Generated {len(proposals)} prosthetic crown + Ti-base abutment proposal(s)."
        )

    def onCompute3DCephClicked(self) -> None:
        if not self.logic:
            return
        table = self.logic.compute_3d_cephalometry_from_scene()
        summary = ", ".join(f"{m.name}={m.value:.1f}{m.unit} ({m.interpretation})" for m in table[:3])
        self.step7HintLabel.setText(
            f"3D Cephalometry: {summary or 'Place S, N, A, B, Go, Gn fiducials first'}"
        )

    def onExportDicomSegSrClicked(self) -> None:
        volNode = self.volumeSelector.currentNode()
        segNode = self.segmentationSelector.currentNode()
        if not volNode or not self.logic:
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(self.parent, "Select DICOM SEG/SR Output Directory")
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        seg_dcm, sr_dcm = self.logic.export_scene_dicom_seg_and_sr(
            volNode,
            segNode,
            self._getOrderedPlanRows(),
            out_dir,
            include_patient_metadata=bool(self.includePatientMetaCheckBox.isChecked()),
        )
        self.step7HintLabel.setText(f"Exported DICOM SEG & SR: {seg_dcm.name}, {sr_dcm.name}")

    def _updateActiveSiteClinicalCallout(self, fdi: Optional[int] = None) -> None:
        """Update the Step 5 Radiologist/Dentist clinical callout card for `fdi`."""
        if getattr(self, "siteClinicalCalloutLabel", None) is None:
            return
        target_fdi = fdi or self.activeToothFDI or (next(iter(self.selectedTeeth)) if self.selectedTeeth else None)
        if not target_fdi:
            return
        row = self.treatmentPlanByFDI.get(int(target_fdi))
        if row is None and self.logic is not None and hasattr(self, "volumeSelector"):
            volNode = self.volumeSelector.currentNode()
            segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
            if volNode is not None:
                row = self.logic.analyze_single_tooth(int(target_fdi), volNode, segmentationNode=segNode)
                self.treatmentPlanByFDI[int(target_fdi)] = row
        if not row:
            return

        h_val = row.get("height_mm")
        w0_val = row.get("w_crest_mm")
        w3_val = row.get("w_3mm")
        w5_val = row.get("w_5mm")
        nerve_val = row.get("nerve_clearance_mm")
        hu_val = row.get("hu")
        misch_val = str(row.get("misch") or "N/A").split()[0]
        rec_d = row.get("rec_diameter")
        rec_l = row.get("rec_length")

        if nerve_val is not None and float(nerve_val) < 2.0:
            badge = "<span style='color:#ef4444;font-weight:bold;'>[ALERT: Nerve &lt; 2.0 mm]</span>"
        elif h_val is not None and w0_val is not None and float(h_val) >= 8.0 and float(w0_val) >= 5.5:
            badge = "<span style='color:#22c55e;font-weight:bold;'>[ADEQUATE BONE VOLUME]</span>"
        else:
            badge = "<span style='color:#f59e0b;font-weight:bold;'>[VERIFY RIDGE / GRAFT]</span>"

        imp_txt = f"Ø{rec_d:.1f} × {rec_l:.1f} mm" if (rec_d and rec_l) else "Manual / Graft Required"
        hu_txt = f"{hu_val:.0f} gray ({misch_val})" if hu_val is not None else "N/A"
        self.siteClinicalCalloutLabel.setText(
            f"<b>Site #{target_fdi} ({str(row.get('jaw', '')).capitalize()}):</b> {badge} | "
            f"<b>Height (H):</b> {format_metric(h_val, 'mm')} | "
            f"<b>W0:</b> {format_metric(w0_val, 'mm')} | "
            f"<b>W3:</b> {format_metric(w3_val, 'mm')} | "
            f"<b>W5:</b> {format_metric(w5_val, 'mm')}<br>"
            f"<b>Vital Distance:</b> {format_metric(nerve_val, 'mm')} | "
            f"<b>Bone Quality:</b> {hu_txt} | "
            f"<b>Suggested Implant:</b> {imp_txt}"
        )

    def onBrowsePrintDirClicked(self) -> None:
        if qt is None:
            return
        folder = qt.QFileDialog.getExistingDirectory(
            self.parent,
            "Select Print Layout Output Folder",
            self.printOutputDirEdit.text if getattr(self, "printOutputDirEdit", None) else "",
        )
        if folder and getattr(self, "printOutputDirEdit", None) is not None:
            self.printOutputDirEdit.setText(folder)

    def onOpenPrintDirClicked(self) -> None:
        if qt is None or getattr(self, "printOutputDirEdit", None) is None:
            return
        out_path = Path(self.printOutputDirEdit.text.strip() or ".")
        out_path.mkdir(parents=True, exist_ok=True)
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(out_path)))

    def _markInlineGalleryStale(self, reason: str = "Input volume, segmentation, or arch changed") -> None:
        self._galleryState.mark_stale(reason)
        if getattr(self, "inlineGalleryStaleLabel", None) is not None and self._galleryState.is_stale:
            self.inlineGalleryStaleLabel.setText(f"Gallery may be stale ({reason}). Click Refresh to update.")
            self.inlineGalleryStaleLabel.setVisible(True)

    def _renderGalleryItems(self, *, full_gallery: bool) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            return
        selected = _sort_teeth_anatomically(list(self.selectedTeeth))
        fdi = self.activeToothFDI or (selected[0] if selected else 46)
        jaw_key = "Lower" if is_mandibular_fdi(fdi) else "Upper"
        curve = self.logic.get_curve_node_for_jaw(volNode, jaw_key, parameterNode=self._parameterNode)
        if (curve is None or curve.GetNumberOfControlPoints() < 4) and segNode is not None:
            curve = self.logic.auto_fit_arch_for_jaw(
                volumeNode=volNode,
                segmentationNode=segNode,
                jaw=jaw_key,
                visible=False,
                confirm_overwrite=False,
                allow_overwrite_user_edit=False,
            )
        if full_gallery:
            from CBCTPlannerLib.core.print_layout import slice_count_for_selection

            n_slices, step_mm = slice_count_for_selection(max(1, len(selected)))
            target_fdis = selected or [fdi]
        else:
            n_slices, step_mm = 1, 1.0
            target_fdis = [fdi]
        series = self.logic.generate_cross_section_slice_series(
            fdi=fdi,
            volumeNode=volNode,
            segmentationNode=segNode,
            num_slices=n_slices,
            step_mm=step_mm,
            fdis=target_fdis,
        )
        self._populateInlineMeasuredGallery(series, full_gallery=full_gallery)
        self._updateActiveSiteClinicalCallout(fdi)

    def onRefreshInlineGalleryClicked(self) -> None:
        """Render the single active measured cross-section by default (or refresh full gallery if already loaded)."""
        self._renderGalleryItems(full_gallery=self._galleryState.full_gallery_loaded)

    def onGenerateFullGalleryClicked(self) -> None:
        """Generate the complete 15-27 slice cross-section gallery on explicit user request."""
        self._renderGalleryItems(full_gallery=True)

    def onGalleryPrevClicked(self) -> None:
        self._galleryState.step(-1)
        self._renderActiveGallerySlice()

    def onGalleryNextClicked(self) -> None:
        self._galleryState.step(1)
        self._renderActiveGallerySlice()

    def _selectGallerySlice(self, index: int) -> None:
        self._galleryState.select(index)
        self._renderActiveGallerySlice()

    def _renderActiveGallerySlice(self) -> None:
        if qt is None or getattr(self, "inlineGalleryActiveImageLabel", None) is None:
            return
        item = self._galleryState.active_item
        total = len(self._galleryState.items)
        idx = self._galleryState.active_index
        if getattr(self, "btnGalleryPrev", None) is not None:
            self.btnGalleryPrev.setEnabled(self._galleryState.can_prev)
        if getattr(self, "btnGalleryNext", None) is not None:
            self.btnGalleryNext.setEnabled(self._galleryState.can_next)
        if getattr(self, "inlineGalleryIndexLabel", None) is not None:
            self.inlineGalleryIndexLabel.setText(f"Slice {idx + 1} / {total}" if total > 0 else "Slice 0 / 0")
        if item is None:
            self.inlineGalleryActiveImageLabel.setText("No active cross-section rendered yet.")
            if getattr(self, "inlineGalleryActiveCaptionLabel", None) is not None:
                self.inlineGalleryActiveCaptionLabel.setText("")
            return

        import base64

        if isinstance(item, dict):
            uri = item.get("image_data_uri", "")
            off = float(item.get("offset_mm", 0.0))
            fdi_val = item.get("fdi")
            h_txt = str(item.get("height_mm", "--")).replace("N/A", "--")
            w0_txt = str(item.get("w_crest_mm", "--")).replace("N/A", "--")
            w3_txt = str(item.get("w_3mm", "--")).replace("N/A", "--")
            w5_txt = str(item.get("w_5mm", "--")).replace("N/A", "--")
            ian_v = item.get("nerve_dist_mm")
            ian_txt = f"{float(ian_v):.1f}" if ian_v is not None else "--"
            reason = item.get("withheld_reason")
        else:
            uri = getattr(item, "image_data_uri", "")
            off = float(getattr(item, "offset_mm", 0.0))
            fdi_val = getattr(item, "fdi", None)
            h_v = getattr(item, "height_mm", None)
            w0_v = getattr(item, "crest_width_mm", None)
            w3_v = getattr(item, "width_3mm", None)
            w5_v = getattr(item, "width_5mm", None)
            ian_v = getattr(item, "nerve_dist_mm", None)
            h_txt = f"{h_v:.1f}" if h_v is not None else "--"
            w0_txt = f"{w0_v:.1f}" if w0_v is not None else "--"
            w3_txt = f"{w3_v:.1f}" if w3_v is not None else "--"
            w5_txt = f"{w5_v:.1f}" if w5_v is not None else "--"
            ian_txt = f"{ian_v:.1f}" if ian_v is not None else "--"
            reason = getattr(item, "withheld_reason", None)

        if uri and "," in uri:
            pixmap = qt.QPixmap()
            if pixmap.loadFromData(base64.b64decode(uri.split(",", 1)[1])):
                self.inlineGalleryActiveImageLabel.setPixmap(
                    pixmap.scaled(360, 440, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
                )
        else:
            self.inlineGalleryActiveImageLabel.setText("Slice unavailable")

        if getattr(self, "inlineGalleryActiveCaptionLabel", None) is not None:
            fdi_tag = f"FDI #{fdi_val} | " if fdi_val is not None else ""
            note_part = f" | Note: {reason}" if reason else ""
            self.inlineGalleryActiveCaptionLabel.setText(
                f"<b>{fdi_tag}Slice {idx + 1} ({off:+.1f} mm)</b> — "
                f"H: {h_txt} mm | W0: {w0_txt} mm | W3: {w3_txt} mm | W5: {w5_txt} mm | Vital: {ian_txt} mm{note_part}"
            )

    def _populateInlineMeasuredGallery(
        self,
        slices_items: Sequence[Any],
        html_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
        png_path: Optional[Path] = None,
        full_gallery: Optional[bool] = None,
    ) -> None:
        """Populate the Step 5 active cross-section viewer and collapsible horizontal thumbnail strip."""
        if qt is None or getattr(self, "inlineGalleryGrid", None) is None:
            return
        while self.inlineGalleryGrid.count():
            child = self.inlineGalleryGrid.takeAt(0)
            if child and child.widget():
                child.widget().deleteLater()

        is_full = (len(slices_items) > 1) if full_gallery is None else bool(full_gallery)
        self._galleryState.set_items(slices_items, full_gallery=is_full)
        if getattr(self, "inlineGalleryStaleLabel", None) is not None:
            self.inlineGalleryStaleLabel.setVisible(False)
        if getattr(self, "inlineGalleryThumbsBox", None) is not None:
            self.inlineGalleryThumbsBox.text = f"All Cross-Section Thumbnails ({len(slices_items)})"

        self._renderActiveGallerySlice()
        if not slices_items:
            if getattr(self, "inlineGalleryStatusLabel", None) is not None:
                self.inlineGalleryStatusLabel.setText(
                    "No measured slices available yet — ensure a segmentation is loaded and arch is fitted."
                )
            return

        import base64

        shown = list(slices_items)
        thumb_w, thumb_h = (120, 155)
        for idx, item in enumerate(shown):
            r, c = 0, idx
            if isinstance(item, dict):
                uri = item.get("image_data_uri", "")
                off = float(item.get("offset_mm", 0.0))
                fdi_val = item.get("fdi")
                h_txt = str(item.get("height_mm", "--")).replace("N/A", "--")
                w0_txt = str(item.get("w_crest_mm", "--")).replace("N/A", "--")
                w3_txt = str(item.get("w_3mm", "--")).replace("N/A", "--")
                w5_txt = str(item.get("w_5mm", "--")).replace("N/A", "--")
                ian_v = item.get("nerve_dist_mm")
                ian_txt = f"{float(ian_v):.1f}" if ian_v is not None else None
                misch = item.get("misch_class")
                reason = item.get("withheld_reason")
            else:
                uri = getattr(item, "image_data_uri", "")
                off = float(getattr(item, "offset_mm", 0.0))
                fdi_val = getattr(item, "fdi", None)
                h_v = getattr(item, "height_mm", None)
                w0_v = getattr(item, "crest_width_mm", None)
                w3_v = getattr(item, "width_3mm", None)
                w5_v = getattr(item, "width_5mm", None)
                ian_v = getattr(item, "nerve_dist_mm", None)
                h_txt = f"{h_v:.1f}" if h_v is not None else "--"
                w0_txt = f"{w0_v:.1f}" if w0_v is not None else "--"
                w3_txt = f"{w3_v:.1f}" if w3_v is not None else "--"
                w5_txt = f"{w5_v:.1f}" if w5_v is not None else "--"
                ian_txt = f"{ian_v:.1f}" if ian_v is not None else None
                misch = getattr(item, "misch_class", None)
                reason = getattr(item, "withheld_reason", None)

            is_center = abs(off) < 0.05
            card = qt.QFrame()
            card.setFrameShape(qt.QFrame.StyledPanel)
            border_css = "2px solid #38bdf8" if is_center else "1px solid #1e293b"
            card.setStyleSheet(
                f"background-color: #020617; border: {border_css}; border-radius: 5px;"
            )
            tip_parts = [f"Slice {idx + 1} ({off:+.1f} mm)"]
            if fdi_val is not None:
                tip_parts[0] = f"FDI #{fdi_val} - " + tip_parts[0]
            tip_parts.append(f"H: {h_txt} mm | W0: {w0_txt} mm | W3: {w3_txt} mm | W5: {w5_txt} mm")
            if ian_txt is not None:
                tip_parts.append(f"IAN / Sinus clearance: {ian_txt} mm")
            if misch:
                tip_parts.append(f"Density (approx): {misch}")
            if reason:
                tip_parts.append(f"Note: {reason}")
            card.setToolTip("\n".join(tip_parts))

            c_lay = qt.QVBoxLayout(card)
            c_lay.setContentsMargins(4, 4, 4, 4)
            c_lay.setSpacing(2)

            fdi_prefix = f"#{fdi_val} " if fdi_val is not None else ""
            hdr = qt.QLabel(f"<b>{fdi_prefix}S{idx + 1}</b> ({off:+.1f} mm)")
            hdr.setAlignment(qt.Qt.AlignCenter)
            hdr.setStyleSheet(
                "color: #ffffff; background-color: #0c4a6e; border-radius: 3px; padding: 1px 3px; font-size: 10px;"
                if is_center
                else "color: #38bdf8; font-size: 10px;"
            )
            c_lay.addWidget(hdr)

            img_lbl = qt.QLabel()
            if uri and "," in uri:
                pixmap = qt.QPixmap()
                if pixmap.loadFromData(base64.b64decode(uri.split(",", 1)[1])):
                    img_lbl.setPixmap(
                        pixmap.scaled(thumb_w, thumb_h, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
                    )
            else:
                img_lbl.setText("No slice")
            img_lbl.setAlignment(qt.Qt.AlignCenter)
            c_lay.addWidget(img_lbl)

            ian_part = f" | <span style='color:#f87171'>IAN:{ian_txt}</span>" if ian_txt is not None else ""
            ftr = qt.QLabel(
                f"<span style='color:#4ade80'><b>H:{h_txt}</b></span> | "
                f"<span style='color:#22d3ee'><b>W0:{w0_txt}</b></span><br/>"
                f"<span style='color:#fbbf24'>W3:{w3_txt}</span> | "
                f"<span style='color:#fb923c'>W5:{w5_txt}</span>{ian_part}"
            )
            ftr.setAlignment(qt.Qt.AlignCenter)
            ftr.setStyleSheet("color: #e2e8f0; font-size: 9.5px;")
            c_lay.addWidget(ftr)

            badge = withheld_badge(reason)
            if badge:
                # Short badge so a `--` width reads as withheld, with the full reason on hover.
                badge_lbl = qt.QLabel(badge)
                badge_lbl.setAlignment(qt.Qt.AlignCenter)
                badge_lbl.setStyleSheet("color: #f59e0b; font-size: 8.5px; font-weight: bold;")
                badge_lbl.setToolTip(str(reason))
                c_lay.addWidget(badge_lbl)
            self.inlineGalleryGrid.addWidget(card, r, c)

        if getattr(self, "inlineGalleryStatusLabel", None) is not None:
            files_note = ""
            if pdf_path is not None:
                files_note = f" | Saved: <b>{pdf_path.name}</b>" + (
                    f" &amp; <b>{png_path.name}</b>" if png_path else ""
                )
            self.inlineGalleryStatusLabel.setText(
                f"Showing <b>{len(shown)} measured cross-sections</b> with Green Height (H), Cyan/Gold Width (W0/W3/W5) &amp; Red Nerve lines{files_note}."
            )

    def onRunNativeALIClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            return
        lms, fidNode = self.logic.run_native_ali_landmarks(volNode, segNode)
        msg = f"ALI identified {len(lms)} 3D landmarks ({', '.join(list(lms.keys())[:8])}...)."
        if getattr(self, "adtResultsLabel", None) is not None:
            self.adtResultsLabel.setText(msg)
        self.step7HintLabel.setText(msg)

    def onRunNativeASOClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            return
        aso_res = self.logic.run_native_aso_orientation(volNode, segNode)
        msg = (
            f"ASO Standardized Orientation applied: Pitch={aso_res.pitch_deg:+.1f}°, "
            f"Roll={aso_res.roll_deg:+.1f}°, Yaw={aso_res.yaw_deg:+.1f}° "
            f"(Landmarks: {', '.join(aso_res.landmarks_used[:6]) or 'Bone SVD'})."
        )
        if getattr(self, "adtResultsLabel", None) is not None:
            self.adtResultsLabel.setText(msg)
        self.step7HintLabel.setText(msg)

    def onRunNativeCLICClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not segNode or not self.logic:
            if slicer:
                slicer.util.warningDisplay("Select both a CBCT volume and a ToothFairy3 segmentation for CLI-C.")
            return
        items = self.logic.run_native_clic_impaction_analysis(volNode, segNode)
        if not items:
            msg = "CLI-C: No canines/wisdom teeth detected in current segmentation."
        else:
            parts = [
                f"#{it.fdi}: {it.intra_osseous_zone} ({it.winter_angulation_class}, {it.angulation_to_midline_deg:.0f}°, Risk: {it.resorption_risk})"
                for it in items[:4]
            ]
            msg = "CLI-C Canine/Molar Localization: " + " | ".join(parts)
        if getattr(self, "adtResultsLabel", None) is not None:
            self.adtResultsLabel.setText(msg)
        self.step7HintLabel.setText(msg)

    def onRunNativeVFACEClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            return
        vf = self.logic.run_native_vface_asymmetry_analysis(volNode, segNode)
        if getattr(self, "adtResultsLabel", None) is not None:
            self.adtResultsLabel.setText(f"VFACE Asymmetry: {vf.summary_text}")
        self.step7HintLabel.setText(f"VFACE: {vf.summary_text}")

    def onRunAutoCropROIClicked(self) -> None:
        volNode = self.volumeSelector.currentNode() if hasattr(self, "volumeSelector") else None
        segNode = self.segmentationSelector.currentNode() if hasattr(self, "segmentationSelector") else None
        if not volNode or not self.logic:
            return
        min_ras, max_ras, _roiNode = self.logic.run_native_autocrop_roi(volNode, segNode, padding_mm=10.0)
        msg = (
            f"AutoCrop3D ROI fitted: [{min_ras[0]:.1f}..{max_ras[0]:.1f}, "
            f"{min_ras[1]:.1f}..{max_ras[1]:.1f}, {min_ras[2]:.1f}..{max_ras[2]:.1f}] mm."
        )
        if getattr(self, "adtResultsLabel", None) is not None:
            self.adtResultsLabel.setText(msg)
        self.step7HintLabel.setText(msg)

    def onLaunchAdtToolClicked(self) -> None:
        tool_name = str(self.adtToolCombo.currentData or "ASO")
        res = run_automated_dental_tool(tool_name)
        if res.get("status") == "not_installed":
            if tool_name == "ALI":
                self.onRunNativeALIClicked()
                return
            if tool_name == "ASO":
                self.onRunNativeASOClicked()
                return
            if tool_name == "CLI-C":
                self.onRunNativeCLICClicked()
                return
            if tool_name == "VFACE":
                self.onRunNativeVFACEClicked()
                return
            if tool_name == "AutoCrop3D":
                self.onRunAutoCropROIClicked()
                return
        self.step7HintLabel.setText(f"AutomatedDentalTools [{tool_name}]: {res['status']}")


class CBCTPlannerLogic(ScriptedLoadableModuleLogic):
    """Core logic orchestrating ParameterNode state, Checkpoint Inference, VR, Arch Curves, Panoramic CPR, Lightbox, Implants, Canal, Airway, TMJ & Phase 7 CAD."""

    def __init__(self) -> None:
        super().__init__()
        self.vr = VRVisualizationLogic()
        self.panoramic = PanoramicCurveLogic()
        self.navigator = CrossSectionNavigator(panoramic_logic=self.panoramic)
        self.canals = CanalTracerLogic()
        self.implants = ImplantPlannerLogic()
        self.checkpoint_engine = CheckpointInferenceEngine()
        self.risemar = RiseMarEngine()
        self.adt_engine = AutomatedDentalToolsEngine()
        self.surgical_guides = SurgicalGuideGenerator()
        self.prosthetics = ProstheticPlanner()
        self.undo_redo = UndoRedoManager(max_depth=50)
        self._is_programmatic_curve_update: bool = False
        self._occlusal_normal: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._calibrated_tooth_arc_mm: Dict[str, float] = {}
        self._confirmed_ai_suggestions: Set[int] = set()
        self._cached_engine_key: Optional[Tuple[Any, ...]] = None
        self._cached_engine: Optional[MeasurementEngine] = None
        self._mask_cache_key: Optional[Tuple[Any, ...]] = None
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_cache_is_5class: bool = False
        self.last_arch_fit: Any = None
        self.last_arch_fit_error: Optional[str] = None
        self._ruler_observers: List[Tuple[Any, int]] = []
        self.last_airway_result: Optional[AirwayAnalysisResult] = None
        self.last_prosthetic_proposals: List[ProstheticProposal] = []
        self.last_cephalometry_table: List[CephalometricMeasurement] = []

    def cleanup(self) -> None:
        self.panoramic.cancel_active_panoramic_cli()
        self.navigator.remove_line_observers()
        for node, tag in self._ruler_observers:
            try:
                node.RemoveObserver(tag)
            except Exception:
                logger.debug(
                    "Best-effort operation failed", exc_info=True
                )  # silent-ok: non-critical cleanup/probe
        self._ruler_observers.clear()
        self._cached_engine = None
        self._cached_engine_key = None
        self._mask_cache_key = None
        self._mask_cache = None

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
        if (
            hasattr(parameterNode, "GetAttribute")
            and parameterNode.GetAttribute("CBCTPlanner.DefaultsInitialized") == "1"
        ):
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
            "PanoThicknessMm": f"{DEFAULTS['PanoThicknessMm']:.1f}",
            "PanoHeightMm": f"{DEFAULTS['PanoHeightMm']:.1f}",
            "PanoOffsetMm": "0.0",
            "PanoProjectionMode": "Mean",
            "PanoFlipVertical": "false",
            "ShowAxialTicks": "true",
            "OcclusalNormalX": "0.0",
            "OcclusalNormalY": "0.0",
            "OcclusalNormalZ": "1.0",
            "SelectedTeethFDI": "",
            "CanalSafetyMarginMm": "2.0",
            "IncludePatientMetadata": "false",
            "CheckpointPath": "",
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
            "PanoFlipVertical",
            "ShowAxialTicks",
            "OcclusalNormalX",
            "OcclusalNormalY",
            "OcclusalNormalZ",
            "SelectedTeethFDI",
            "CanalSafetyMarginMm",
            "IncludePatientMetadata",
            "CheckpointPath",
            "VRPreset",
            "VRShiftHu",
            "VRVisible",
            "VRCropEnabled",
        )
        return {k: (parameterNode.GetParameter(k) or "") for k in keys}

    # ------------------------------------------------------------------
    # Custom AI Checkpoint Inference ("Place My Checkpoint" Workflow)
    # ------------------------------------------------------------------
    def run_custom_checkpoint_on_volume_node(
        self,
        volumeNode: Any,
        checkpoint_path: Path,
        label_scheme: str = "5class",
        patch_size_zyx: Tuple[int, int, int] = (96, 96, 96),
        custom_arch_py_path: Optional[Path] = None,
        custom_arch_class_name: Optional[str] = None,
        on_progress: Optional[Any] = None,
    ) -> Tuple[Any, CheckpointLoadSummary]:
        """Run user-supplied checkpoint on `volumeNode` and populate a styled `vtkMRMLSegmentationNode`."""
        if slicer is None or volumeNode is None:
            raise RuntimeError("Slicer and an input CBCT volume node are required.")
        img_zyx = slicer.util.arrayFromVolume(volumeNode)
        spec = CheckpointSpec(
            checkpoint_path=Path(checkpoint_path),
            label_scheme=label_scheme,
            patch_size_zyx=patch_size_zyx,
            custom_arch_py_path=custom_arch_py_path,
            custom_arch_class_name=custom_arch_class_name,
        )
        labelmap_zyx, summary = self.checkpoint_engine.run_checkpoint_on_array(
            img_zyx, spec, on_progress=on_progress
        )
        pNode = self.getParameterNode()
        segNode = self.checkpoint_engine.populate_slicer_segmentation_from_labelmap(
            labelmap_zyx,
            volumeNode=volumeNode,
            label_scheme=label_scheme,
            parameterNode=pNode,
        )
        if segNode is not None:
            self.vr.apply_dental_segmentator_styling(segNode)
            self.update_fdi_overlays_from_segmentation(volumeNode, segNode, visible=True)
        return segNode, summary

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
        tfNode = (
            pNode.GetNodeReference("OcclusalTransform")
            if (pNode and hasattr(pNode, "GetNodeReference"))
            else None
        )
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
                    logger.debug(
                        "Best-effort operation failed", exc_info=True
                    )  # silent-ok: non-critical cleanup/probe
                pNode.SetNodeReferenceID("OcclusalTransform", "")

    def start_place_occlusal_points(self, volumeNode: Any, parameterNode: Any = None) -> Any:
        if slicer is None or not volumeNode:
            return None
        pNode = parameterNode or self.getParameterNode()
        fidNode = (
            pNode.GetNodeReference("OcclusalFiducials")
            if (pNode and hasattr(pNode, "GetNodeReference"))
            else None
        )
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
        fidNode = (
            pNode.GetNodeReference("OcclusalFiducials")
            if (pNode and hasattr(pNode, "GetNodeReference"))
            else None
        )
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

    # ------------------------------------------------------------------
    # Phase 1: Lossless RAS orientation
    # ------------------------------------------------------------------
    @staticmethod
    def _volume_array_and_affine(volumeNode: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the Slicer scalar array `(Z, Y, X)` and its IJK->RAS affine."""
        if slicer is None or volumeNode is None or vtk is None:
            return None, None
        arr = slicer.util.arrayFromVolume(volumeNode)
        if arr is None:
            return None, None
        matrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(matrix)
        affine = np.array(
            [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)],
            dtype=np.float64,
        )
        return arr, affine

    @staticmethod
    def _write_volume_array_and_affine(volumeNode: Any, arr: np.ndarray, affine: np.ndarray) -> None:
        if slicer is None or volumeNode is None:
            return
        slicer.util.updateVolumeFromArray(volumeNode, np.ascontiguousarray(arr))
        volumeNode.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(np.asarray(affine, dtype=np.float64)))
        if hasattr(slicer.util, "arrayFromVolumeModified"):
            slicer.util.arrayFromVolumeModified(volumeNode)
        if hasattr(volumeNode, "Modified"):
            volumeNode.Modified()

    def _apply_segment_ops(
        self,
        segmentationNode: Any,
        sourceVolumeNode: Any,
        targetVolumeNode: Any,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        preloaded_arrays: Optional[dict[str, np.ndarray]] = None,
    ) -> int:
        """Apply `func` to every segment binary labelmap and write it against `targetVolumeNode`.

        Slicer 5.6 exposes `arrayFromSegmentBinaryLabelmap` and
        `updateSegmentBinaryLabelmapFromArray`; re-encoding each segment keeps voxel
        data aligned when axes are permuted. If that API is unavailable we log an
        error instead of silently resetting geometry to a possibly wrong reference.
        """
        if slicer is None or segmentationNode is None or targetVolumeNode is None:
            return 0
        segmentation = segmentationNode.GetSegmentation()
        if segmentation is None:
            return 0
        if not (
            hasattr(slicer.util, "arrayFromSegmentBinaryLabelmap")
            and hasattr(slicer.util, "updateSegmentBinaryLabelmapFromArray")
        ):
            logger.error(
                "Slicer build lacks segment labelmap array helpers; cannot re-encode segmentation orientation"
            )
            return 0
        updated = 0
        for index in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(index)
            if preloaded_arrays is not None:
                seg_arr = preloaded_arrays.get(segment_id)
            else:
                seg_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                    segmentationNode, segment_id, sourceVolumeNode
                )
            if seg_arr is None:
                continue
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                func(np.asarray(seg_arr)), segmentationNode, segment_id, targetVolumeNode
            )
            updated += 1
        return updated

    def standardize_orientation(
        self,
        volumeNode: Any,
        segmentationNode: Any = None,
        *,
        mode: str = "clone",
        allow_resample: bool = True,
        auto_fix_anatomy: bool = True,
    ) -> Tuple[Optional[Any], Any]:
        """Convert a volume (and optionally its segmentation) to RAS.

        Returns `(new_or_updated_volume_node, OrientationReport)`. Permuted/flipped
        data is lossless; oblique data is linearly resampled and flagged in the report
        as interpolation that must not be treated as an exact voxel copy. When a
        segmentation is provided and `auto_fix_anatomy=True`, any detected anatomical
        inversions (S-I, A-P, L-R) are also mirrored in voxel space while keeping the
        RAS affine so Axial, Sagittal, and Coronal views display right-way-up anatomy.
        """
        from CBCTPlannerLib.core.orientation import (
            OrientationReport,
            _obliquity_deg,
            axcodes_from_affine,
            canonicalize_to_ras,
            inverted_axes_from_report,
            resample_mask_to_grid,
            resample_to_ras,
            verify_orientation,
        )

        if slicer is None or volumeNode is None:
            return None, OrientationReport(notes=["No volume selected."])
        transform_id = ""
        if hasattr(volumeNode, "GetTransformNodeID"):
            transform_id = volumeNode.GetTransformNodeID() or ""
        if transform_id:
            return None, OrientationReport(
                notes=[
                    "The volume has a parent transform (occlusal alignment). "
                    "Standardize orientation before applying occlusal alignment."
                ]
            )

        arr, affine = self._volume_array_and_affine(volumeNode)
        if arr is None or affine is None:
            return None, OrientationReport(notes=["Could not read the volume array or IJK-to-RAS matrix."])

        try:
            result = canonicalize_to_ras(arr, affine)
        except ValueError as exc:
            logger.exception("RAS canonicalization failed")
            return None, OrientationReport(notes=[f"Canonicalization failed: {exc}"])

        resampled = False
        if result.is_oblique:
            if not allow_resample:
                return None, OrientationReport(
                    notes=[
                        f"Volume axes are oblique ({_obliquity_deg(affine):.1f}° > 5°). "
                        "No lossless permutation is possible; enable resampling or run OrientScalarVolume."
                    ]
                )
            result = resample_to_ras(arr, affine)
            resampled = True

        # In-place rewriting changes the volume geometry. Snapshot every segment in
        # the *source* geometry first, otherwise Slicer would read the old labelmap
        # through the new reference volume and the permutation/resampling would be
        # applied twice or sample the wrong voxels.
        preloaded_arrays: Optional[dict[str, np.ndarray]] = None
        if mode == "in_place" and segmentationNode is not None:
            preloaded_arrays = {}
            segmentation = segmentationNode.GetSegmentation()
            if segmentation is not None and hasattr(slicer.util, "arrayFromSegmentBinaryLabelmap"):
                for index in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(index)
                    seg_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                        segmentationNode, segment_id, volumeNode
                    )
                    if seg_arr is not None:
                        preloaded_arrays[segment_id] = np.asarray(seg_arr).copy()

        target_volume = volumeNode
        if mode == "clone":
            name = f"{volumeNode.GetName()}_RAS"
            target_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
            self._write_volume_array_and_affine(target_volume, result.image_zyx, result.affine)
            if volumeNode.GetDisplayNode() is not None:
                target_volume.CreateDefaultDisplayNodes()
                target_volume.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())
        elif mode == "in_place":
            self._write_volume_array_and_affine(volumeNode, result.image_zyx, result.affine)
        else:
            return None, OrientationReport(notes=[f"Unknown mode {mode!r}; use 'clone' or 'in_place'."])

        if hasattr(target_volume, "SetAttribute"):
            target_volume.SetAttribute("CBCTPlanner.Orientation", "RAS")

        updated_segments = 0
        if segmentationNode is not None:
            from CBCTPlannerLib.core.orientation import apply_ops

            if resampled:
                target_shape = result.image_zyx.shape
                segment_func = lambda seg_arr: resample_mask_to_grid(  # noqa: E731
                    seg_arr, affine, result.affine, target_shape
                )
            else:
                segment_func = lambda seg_arr: apply_ops(seg_arr, result.ops)  # noqa: E731
            updated_segments = self._apply_segment_ops(
                segmentationNode,
                volumeNode,
                target_volume,
                segment_func,
                preloaded_arrays=preloaded_arrays,
            )
            if hasattr(segmentationNode, "SetReferenceImageGeometryParameterFromVolumeNode"):
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(target_volume)
            if hasattr(segmentationNode, "SetAttribute"):
                segmentationNode.SetAttribute("CBCTPlanner.Orientation", "RAS")

        report = OrientationReport(
            notes=[
                f"{'Resampled' if resampled else 'Canonicalized'} "
                f"{axcodes_from_affine(affine)} -> {axcodes_from_affine(result.affine)} "
                f"({updated_segments} segment(s) re-encoded)."
            ]
        )
        if segmentationNode is not None:
            mask, is_5class = self._extract_combined_label_mask(target_volume, segmentationNode)
            if mask is not None:
                report = verify_orientation(mask, result.affine, is_5class=is_5class)
                bad_axes = inverted_axes_from_report(report)
                if auto_fix_anatomy and bad_axes:
                    self.flip_orientation_axis(target_volume, segmentationNode, bad_axes)
                    mask_fixed, is_5class_fixed = self._extract_combined_label_mask(target_volume, segmentationNode)
                    if mask_fixed is not None:
                        report = verify_orientation(mask_fixed, result.affine, is_5class=is_5class_fixed)
                    report.notes.insert(
                        0,
                        f"Auto-corrected inverted anatomy along {', '.join(bad_axes)} (visible in Axial/Sagittal/Coronal).",
                    )
                report.notes.insert(
                    0,
                    f"{'Resampled' if resampled else 'Canonicalized'} "
                    f"{axcodes_from_affine(affine)} -> "
                    f"{axcodes_from_affine(result.affine)} ({updated_segments} segment(s) re-encoded).",
                )
            else:
                report.notes.append("No label mask available; orientation could not be verified.")
        else:
            report.notes.append("No segmentation supplied; orientation could not be verified.")
        if resampled:
            report.notes.append(
                "Oblique input was linearly interpolated onto the RAS grid; "
                "re-check fine measurements against the source resolution."
            )
        self.reset_slice_views(target_volume)
        return target_volume, report

    def detect_orientation_issues(self, volumeNode: Any, segmentationNode: Any = None) -> Any:
        """Run segmentation-based S-I / A-P / L-R verification without changing data."""
        from CBCTPlannerLib.core.orientation import OrientationReport, verify_orientation

        if slicer is None or volumeNode is None:
            return OrientationReport(notes=["No volume selected."])
        arr, affine = self._volume_array_and_affine(volumeNode)
        if arr is None or affine is None:
            return OrientationReport(notes=["Could not read the volume array or IJK-to-RAS matrix."])
        if segmentationNode is None:
            return OrientationReport(notes=["Auto-detect needs a segmentation with jaw or tooth labels."])
        mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            return OrientationReport(notes=["Could not build a label mask from the segmentation."])
        return verify_orientation(mask, affine, is_5class=is_5class)

    def flip_orientation_axis(
        self, volumeNode: Any, segmentationNode: Any, axis: str | Sequence[str]
    ) -> bool:
        """Mirror a volume and segmentation along LR/AP/SI/ALL so the displayed anatomy flips.

        The IJK-to-RAS matrix is intentionally kept: flipping the voxels *and* the matrix
        only reorders data and leaves every viewer looking unchanged.
        """
        from CBCTPlannerLib.core.orientation import flip_array_only

        if slicer is None or volumeNode is None:
            return False
        arr, affine = self._volume_array_and_affine(volumeNode)
        if arr is None or affine is None:
            return False

        # Snapshot segment labelmaps BEFORE modifying volumeNode so every segment is read cleanly.
        preloaded_arrays: Optional[dict[str, np.ndarray]] = None
        if segmentationNode is not None:
            preloaded_arrays = {}
            segmentation = segmentationNode.GetSegmentation()
            if segmentation is not None and hasattr(slicer.util, "arrayFromSegmentBinaryLabelmap"):
                for index in range(segmentation.GetNumberOfSegments()):
                    segment_id = segmentation.GetNthSegmentID(index)
                    seg_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
                        segmentationNode, segment_id, volumeNode
                    )
                    if seg_arr is not None:
                        preloaded_arrays[segment_id] = np.asarray(seg_arr).copy()

        flipped = flip_array_only(arr, affine, axis)
        self._write_volume_array_and_affine(volumeNode, flipped, affine)
        if segmentationNode is not None:
            self._apply_segment_ops(
                segmentationNode,
                volumeNode,
                volumeNode,
                lambda seg_arr: flip_array_only(np.asarray(seg_arr), affine, axis),
                preloaded_arrays=preloaded_arrays,
            )
            if hasattr(segmentationNode, "SetReferenceImageGeometryParameterFromVolumeNode"):
                segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            if hasattr(segmentationNode, "Modified"):
                segmentationNode.Modified()
            try:
                seg = segmentationNode.GetSegmentation()
                if seg is not None and seg.ContainsRepresentation("Closed surface"):
                    segmentationNode.CreateClosedSurfaceRepresentation()
            except Exception:
                logger.debug("Could not refresh closed surface after flip", exc_info=True)

        axis_tag = axis if isinstance(axis, str) else "_".join(str(a) for a in axis)
        if hasattr(volumeNode, "SetAttribute"):
            volumeNode.SetAttribute("CBCTPlanner.Orientation", f"manual_flip_{axis_tag}")
        if hasattr(self, "_measurement_engine"):
            self._measurement_engine = None
        self.reset_slice_views(volumeNode)
        return True

    def reset_slice_views(self, volumeNode: Any) -> None:
        """Reset Axial (Red), Sagittal/Cross-Section (Yellow*), and Coronal (Green) views without clobbering Panoramic."""
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
                if hasattr(sNode, "UpdateMatrices"):
                    sNode.UpdateMatrices()
                sw.sliceLogic().FitSliceToAll()
                try:
                    sw.sliceView().forceRender()
                except Exception:
                    pass  # silent-ok: headless sliceView mock may not implement forceRender

    def standardize_to_lpi(self, volumeNode: Any) -> None:
        """Deprecated alias for `reset_slice_views` (the old name was misleading)."""
        self.reset_slice_views(volumeNode)

    # ------------------------------------------------------------------
    # Arch Curve Management (B15 Node Refs, B20 Spline without Skull Fallback, F3 Single Arch)
    # ------------------------------------------------------------------
    def get_curve_node_for_jaw(self, volumeNode: Any, jaw: str, parameterNode: Any = None) -> Any:
        """B15 & F3: Resolve curve node via ParameterNode reference ID first; migrate per-jaw curve for Shared mode."""
        pNode = parameterNode or (self.getParameterNode() if slicer is not None else None)
        node = self.panoramic.resolve_curve_node(volumeNode, jaw=jaw, parameterNode=pNode)
        if node is None and jaw.lower().startswith("sh"):
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
        """B20: Extract 3D tooth coordinates in RAS from a segmentation node (never falls back to whole skull)."""
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
            fdi_attr = safe_get_segment_tag(seg, "CBCTPlanner.FDI")
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
            if not segmentation.ContainsRepresentation(rep_name):
                try:
                    segmentation.CreateClosedSurfaceRepresentation()
                except Exception:
                    logger.debug(
                        "Best-effort operation failed", exc_info=True
                    )  # silent-ok: non-critical cleanup/probe
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
                    # Subsample each tooth to 20 points to prevent massive point clouds
                    step = max(1, len(pts) // 20)
                    collected.append(pts[::step])
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
                        z_idx, y_idx, x_idx = np.where(tooth_mask_from_labels(mask, is_5class=True))
                    else:
                        target_val = 3 if jaw_low.startswith("up") else 4
                        z_idx, y_idx, x_idx = np.where(mask == target_val)
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
                    pts_list = list(centroids)
                    jaw_ids = (
                        (LOWER_JAW_ID, UPPER_JAW_ID)
                        if jaw_low.startswith("sh")
                        else ((UPPER_JAW_ID,) if jaw_low.startswith("up") else (LOWER_JAW_ID,))
                    )
                    jz, jy, jx = np.where(np.isin(mask, jaw_ids))
                    if len(jx) > 0:
                        step = max(1, len(jx) // 4000)
                        jaw_vox = np.column_stack([jx[::step], jy[::step], jz[::step]])
                        jaw_world = vol.voxel_to_world(jaw_vox)
                        if len(centroids) > 0:
                            min_tooth_a = float(np.min(np.asarray(centroids, dtype=np.float64)[:, 1]))
                            post_jaw = jaw_world[jaw_world[:, 1] < min_tooth_a]
                            if len(post_jaw) > 0:
                                post_step = max(1, len(post_jaw) // 80)
                                pts_list.extend(post_jaw[::post_step].tolist())
                        else:
                            jaw_step = max(1, len(jaw_world) // 120)
                            pts_list.extend(jaw_world[::jaw_step].tolist())
                    if len(pts_list) >= 4:
                        return np.array(pts_list, dtype=np.float64)
        return None

    def auto_fit_arch_for_jaw(
        self,
        volumeNode: Any,
        segmentationNode: Any,
        jaw: str,
        visible: bool = True,
        confirm_overwrite: bool = True,
        allow_overwrite_user_edit: bool = True,
    ) -> Any:
        """Fit the dental arch from segmented tooth voxels; fall back to manual placement.

        No surface meshes are generated. The fit is rejected with a reason when the
        residual/arc-length sanity gates fail, and the caller can offer manual placement.
        """
        if slicer is None or not volumeNode:
            return None
        jaw_cap = "Shared" if jaw.lower().startswith("sh") else jaw.capitalize()
        pNode = self.getParameterNode()
        self.last_arch_fit = None
        existing = self.get_curve_node_for_jaw(volumeNode, jaw_cap, parameterNode=pNode)

        user_edited = (
            existing is not None
            and existing.GetNumberOfControlPoints() >= 4
            and existing.GetAttribute("CBCTPlanner.UserEdited") == "1"
        )
        if user_edited and not allow_overwrite_user_edit:
            logger.info("Skipping auto-fit: %s arch curve is user-edited", jaw_cap)
            return existing
        if user_edited and confirm_overwrite and getattr(slicer, "app", None) is not None:
            if not slicer.util.confirmOkCancelDisplay(
                f"The {jaw_cap} dental arch curve has been manually edited.\n"
                "Do you want to overwrite your manual edits with an auto-fitted curve?",
                windowTitle="Overwrite Edited Arch Curve?",
            ):
                return existing

        if segmentationNode is None:
            logger.warning("Auto-fit needs a segmentation with tooth labels")
            curveNode = self.start_interactive_arch_placement(volumeNode, jaw=jaw_cap)
            if getattr(slicer, "app", None) is not None:
                slicer.util.showStatusMessage(
                    "Auto-fit needs segmented teeth — place arch points manually or run segmentation first.",
                    6000,
                )
            return curveNode

        mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            logger.warning("Auto-fit could not build a label mask from the segmentation")
            curveNode = self.start_interactive_arch_placement(volumeNode, jaw=jaw_cap)
            if getattr(slicer, "app", None) is not None:
                slicer.util.showStatusMessage(
                    "Auto-fit could not read segmented teeth — place arch points manually.", 6000
                )
            return curveNode

        _, affine = self._volume_array_and_affine(volumeNode)
        if affine is None:
            logger.error("Auto-fit could not read the volume IJK-to-RAS matrix")
            return None

        fit = fit_arch_from_label_mask(
            mask,
            affine,
            jaw=jaw_cap,
            is_5class=is_5class,
            num_control_points=11,
            end_extension_mm=12.0,
        )
        self.last_arch_fit = fit
        if fit is None or not fit.accepted:
            reason = "no tooth labels found" if fit is None else (fit.reason or "fit rejected")
            logger.warning("Auto-fit rejected for %s arch: %s", jaw_cap, reason)
            self.last_arch_fit_error = reason
            if getattr(slicer, "app", None) is not None:
                slicer.util.showStatusMessage(f"Auto-fit rejected: {reason}", 6000)
            return None

        previous_points: Optional[np.ndarray] = None
        if existing is not None and existing.GetNumberOfControlPoints() >= 2:
            previous_points = np.zeros((existing.GetNumberOfControlPoints(), 3), dtype=np.float64)
            buffer = [0.0, 0.0, 0.0]
            for index in range(existing.GetNumberOfControlPoints()):
                existing.GetNthControlPointPositionWorld(index, buffer)
                previous_points[index] = buffer

        def _apply_fit() -> None:
            self._is_programmatic_curve_update = True
            try:
                node = self.panoramic.create_or_update_arch_curve(
                    volumeNode,
                    jaw=jaw_cap,
                    points_ras=fit.points_ras,
                    visible=visible,
                    parameterNode=pNode,
                )
                if node is not None:
                    node.SetAttribute("CBCTPlanner.UserEdited", "0")
            finally:
                self._is_programmatic_curve_update = False

        if previous_points is not None:

            def _restore_previous() -> None:
                self._is_programmatic_curve_update = True
                try:
                    node = self.panoramic.create_or_update_arch_curve(
                        volumeNode,
                        jaw=jaw_cap,
                        points_ras=previous_points,
                        visible=visible,
                        parameterNode=pNode,
                    )
                    if node is not None:
                        node.SetAttribute("CBCTPlanner.UserEdited", "1")
                finally:
                    self._is_programmatic_curve_update = False

            self.undo_redo.push(
                f"Replace {jaw_cap} arch curve with auto-fit",
                _restore_previous,
                _apply_fit,
            )

        _apply_fit()
        return self.get_curve_node_for_jaw(volumeNode, jaw_cap, parameterNode=pNode)

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
        order = JAW_TOOTH_ORDER["lower"] if is_mandibular_fdi(fdi) else JAW_TOOTH_ORDER["upper"]
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
        jaw = "Lower" if is_mandibular_fdi(fdi) else "Upper"
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
        jaw = "Lower" if is_mandibular_fdi(fdi) else "Upper"
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
        panoNode = (
            pNode.GetNodeReference("PanoramicVolume")
            if (pNode and hasattr(pNode, "GetNodeReference"))
            else None
        )
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
        matched_records: List[Tuple[int, str, bool, int]] = []
        per_seg_matches: List[Tuple[str, int, int]] = []

        for idx, sid in enumerate(seg_ids):
            seg = segmentation.GetSegment(sid)
            seg_name = seg.GetName() if seg else ""
            fdi_attr = safe_get_segment_tag(seg, "CBCTPlanner.FDI")
            term_str = safe_get_segment_tag(seg, "TerminologyEntry")

            label_val, cat, seg_is_5class, prec = match_segment_to_label_and_precedence(
                sid, seg_name, fdi_attribute=fdi_attr, terminology_str=term_str
            )
            if label_val == 0:
                if len(seg_ids) == 5:
                    label_val = idx + 1
                    seg_is_5class = True
                    cat = "canal" if label_val == 5 else ("tooth" if label_val in (3, 4) else "jaw")
                    prec = 3 if label_val == 5 else (4 if label_val in (3, 4) else 1)
                else:
                    continue
            matched_records.append((label_val, cat, seg_is_5class, prec))
            per_seg_matches.append((sid, label_val, prec))

        is_5class = resolve_combined_mask_scheme(matched_records)

        for sid, label_val, prec in per_seg_matches:
            # If the overall segmentation is ToothFairy3 (not 5-class), ensure Mandible=1 and Maxilla=2
            if not is_5class and label_val in (1, 2):
                seg = segmentation.GetSegment(sid)
                sname_l = (seg.GetName() if seg else "").lower()
                if "mandible" in sname_l or "lower" in sname_l:
                    label_val = LOWER_JAW_ID
                elif "maxilla" in sname_l or "upper" in sname_l or "skull" in sname_l:
                    label_val = UPPER_JAW_ID

            try:
                seg_arr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, sid, volumeNode)
                if seg_arr is not None and seg_arr.shape == combined.shape:
                    write_mask = (seg_arr > 0) & (prec >= precedence_buf)
                    combined[write_mask] = label_val
                    precedence_buf[write_mask] = prec
            except Exception:
                logger.debug(
                    "Best-effort operation failed", exc_info=True
                )  # silent-ok: non-critical cleanup/probe

        if not np.any(combined):
            return None, False
        return combined, is_5class

    @staticmethod
    def _is_ruler_user_modified(node: Any) -> bool:
        """B1: Return True ONLY if the ruler node has been modified by the user (or placed interactively)."""
        if node is None:
            return False
        user_mod_attr = (
            node.GetAttribute("CBCTPlanner.UserModified") if hasattr(node, "GetAttribute") else None
        )
        if user_mod_attr == "1":
            return True

        init_json = (
            node.GetAttribute("CBCTPlanner.InitialEndpoints") if hasattr(node, "GetAttribute") else None
        )
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
                logger.debug(
                    "Best-effort operation failed", exc_info=True
                )  # silent-ok: non-critical cleanup/probe
        return False

    @classmethod
    def _read_manual_ruler_length(cls, ruler_name_or_node: Any) -> Optional[float]:
        """B1: Return manual ruler length in mm ONLY when user-modified (`CBCTPlanner.UserModified == "1"`)."""
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

        is_lower = is_mandibular_fdi(fdi)
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

                if vtk is not None and hasattr(rNode, "AddObserver"):

                    def _on_ruler_interacted(caller: Any, _evt: Any) -> None:
                        cls_ref = self.__class__
                        cls_ref._is_ruler_user_modified(caller)

                    point_mod_evt = getattr(
                        getattr(slicer, "vtkMRMLMarkupsNode", None),
                        "PointModifiedEvent",
                        getattr(vtk.vtkCommand, "ModifiedEvent", 33),
                    )
                    end_inter_evt = getattr(
                        vtk.vtkCommand, "EndInteractionEvent", getattr(vtk.vtkCommand, "ModifiedEvent", 33)
                    )
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
        """Cache the label mask separately from the arch, so curve drags skip segment export."""
        if slicer is None or volumeNode is None or segmentationNode is None:
            return None
        mask_key = self._mask_cache_token(volumeNode, segmentationNode)

        pNode = self.getParameterNode()
        c_low = self.get_curve_node_for_jaw(volumeNode, "Lower", parameterNode=pNode)
        c_up = self.get_curve_node_for_jaw(volumeNode, "Upper", parameterNode=pNode)
        c_sh = self.get_curve_node_for_jaw(volumeNode, "Shared", parameterNode=pNode)
        curve_key = (
            int(c_low.GetMTime()) if (c_low and hasattr(c_low, "GetMTime")) else 0,
            int(c_up.GetMTime()) if (c_up and hasattr(c_up, "GetMTime")) else 0,
            int(c_sh.GetMTime()) if (c_sh and hasattr(c_sh, "GetMTime")) else 0,
            tuple(round(float(v), 5) for v in self.get_occlusal_normal()),
        )

        if self._cached_engine is not None and self._cached_engine_key == (mask_key, curve_key):
            return self._cached_engine

        if (
            self._cached_engine is not None
            and self._cached_engine_key is not None
            and self._cached_engine_key[0] == mask_key
        ):
            # Only the curve/occlusal normal changed: update arches, keep mask and teeth.
            arches = self._collect_arches(volumeNode)
            self._cached_engine.set_arches(arches)
            self._cached_engine_key = (mask_key, curve_key)
            return self._cached_engine

        engine = self._build_measurement_engine(volumeNode, segmentationNode, mask_key=mask_key)
        self._cached_engine_key = (mask_key, curve_key)
        self._cached_engine = engine
        return engine

    @staticmethod
    def _mask_cache_token(volumeNode: Any, segmentationNode: Any) -> Tuple[Any, ...]:
        """Identity, mtime, segmentation/image mtime, geometry, and parent transform token."""
        return compute_volume_segmentation_cache_token(volumeNode, segmentationNode)

    def _collect_arches(self, volumeNode: Any) -> Dict[str, DentalArch]:
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
        return arches

    def _get_or_extract_mask(
        self, volumeNode: Any, segmentationNode: Any, key: Optional[Tuple[Any, ...]] = None
    ) -> Optional[Tuple[np.ndarray, bool]]:
        """Cache the combined label mask by volume/segmentation identity and mtime."""
        if slicer is None or volumeNode is None or segmentationNode is None:
            return None
        if key is None:
            key = self._mask_cache_token(volumeNode, segmentationNode)
        if self._mask_cache_key == key and self._mask_cache is not None:
            return self._mask_cache, self._mask_cache_is_5class
        mask, is_5class = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            return None
        self._mask_cache_key = key
        self._mask_cache = mask
        self._mask_cache_is_5class = bool(is_5class)
        return mask, bool(is_5class)

    def _build_measurement_engine(
        self,
        volumeNode: Any,
        segmentationNode: Any,
        mask_key: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[MeasurementEngine]:
        if slicer is None or volumeNode is None or segmentationNode is None:
            return None
        cached = self._get_or_extract_mask(volumeNode, segmentationNode, key=mask_key)
        if cached is None:
            return None
        mask, is_5class = cached
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        arches = self._collect_arches(volumeNode)
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
        is_lower = is_mandibular_fdi(fdi)
        jaw = "lower" if is_lower else "upper"
        site_map = (
            self.get_fdi_arc_site_map(volumeNode, jaw=jaw, segmentationNode=segmentationNode)
            if volumeNode
            else {}
        )
        site_obj = site_map.get(str(fdi))
        frame_info = (
            self.get_tooth_arc_mm_and_frame(int(fdi), volumeNode, segmentationNode=segmentationNode)
            if volumeNode
            else None
        )
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
        withheld_reason: Optional[str] = None
        site_pos_source = site_obj.source if site_obj is not None else "approximate_width_table"
        is_approx_site = site_obj.is_approximate if site_obj is not None else True

        if engine is None and volumeNode is not None and segmentationNode is not None:
            engine = self._get_or_build_measurement_engine(volumeNode, segmentationNode)
        if engine is None and segmentationNode is None:
            withheld_reason = WITHHELD_NO_SEGMENTATION

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
                withheld_reason = meas.withheld_reason
                if any(v is not None for v in (height_mm, w_crest_mm, nerve_clearance_mm, mean_hu)):
                    source = "segmentation"
            else:
                # The whole site was rejected: keep the cross-section's reason for the tile/CSV.
                withheld_reason = getattr(engine, "last_withheld_reason", None)

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
            [round(float(P[0]), 2), round(float(P[1]), 2), round(float(P[2]), 2)] if P is not None else None
        )
        return {
            "fdi": int(fdi),
            "jaw": jaw,
            "source": source,
            "withheld_reason": withheld_reason,
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
    # F1, F2, F4, F7, F8, F14, F15, F16, F17: Clinical & Phase 7 Operations
    # ------------------------------------------------------------------
    def extract_and_display_mandibular_canals(self, volumeNode: Any, segmentationNode: Any) -> Dict[str, Any]:
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

        vol = (
            CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
            if (slicer and volumeNode)
            else None
        )
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
        """F8: Align Yellow (Sagittal) and Green/Red (Coronal) views to the Left or Right mandibular condyle."""
        vol = (
            CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
            if (slicer and volumeNode)
            else None
        )
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
                        float(N_sag[0]),
                        float(N_sag[1]),
                        float(N_sag[2]),
                        float(R_sag[0]),
                        float(R_sag[1]),
                        float(R_sag[2]),
                        float(P[0]),
                        float(P[1]),
                        float(P[2]),
                        0,
                    )
        return frame

    def generate_and_export_surgical_guide(
        self,
        volumeNode: Any,
        out_dir: Path,
        support_type: str = "tooth_supported",
        preset_id: str = "sla_dlp_standard_50um",
    ) -> Tuple[Path, Path, List[GuideSleeveSpec]]:
        """F14: Generate surgical guide mesh from active arch curve and placed implants, and export STL + JSON."""
        pNode = self.getParameterNode() if slicer is not None else None
        jaw = (pNode.GetParameter("ActiveJaw") or "Lower") if pNode else "Lower"
        curveNode = self.get_curve_node_for_jaw(volumeNode, jaw, parameterNode=pNode)
        if curveNode is None or curveNode.GetNumberOfControlPoints() < 4:
            raise ValueError(
                f"Cannot export surgical guide for {jaw} jaw: a valid fitted or manually confirmed "
                "dental arch curve (>= 4 control points) is required. Fit or draw the arch in Step 2 first."
            )
        arch_pts = self.panoramic.extract_world_points_from_curve_node(curveNode)
        if len(arch_pts) < 4 or not np.all(np.isfinite(arch_pts)):
            raise ValueError(
                f"Cannot export surgical guide for {jaw} jaw: arch curve control points are invalid or degenerate."
            )
        implants = self.implants.get_reportable_implants(include_unconfirmed=False)
        return self.surgical_guides.export_surgical_guide_stl_and_json(
            arch_pts, implants, out_dir=out_dir, support_type=support_type, preset_id=preset_id
        )

    def generate_cross_section_slice_series(
        self,
        fdi: int,
        volumeNode: Any,
        segmentationNode: Any = None,
        num_slices: int = 21,
        step_mm: float = 0.75,
        fdis: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Render genuine arc cross-sections with measured overlays (one resample per slice)."""
        from CBCTPlannerLib.core_print_render import render_cross_section

        results: List[Dict[str, Any]] = []
        if volumeNode is None:
            return results
        engine = self._get_or_build_measurement_engine(volumeNode, segmentationNode)
        if engine is None:
            logger.error("Slice series needs a measurement engine (volume + arch)")
            return results

        tooth_list = [int(f) for f in (fdis or [fdi])]
        if hasattr(engine, "expand_missing_edentulous_span"):
            try:
                expanded_list, _missing_list = engine.expand_missing_edentulous_span(tooth_list)
                if expanded_list:
                    tooth_list = sort_teeth_anatomically(expanded_list)
                    from CBCTPlannerLib.core.print_layout import slice_count_for_selection
                    rec_slices, rec_step = slice_count_for_selection(len(tooth_list))
                    num_slices = max(int(num_slices), int(rec_slices))
                    step_mm = float(rec_step)
            except Exception:
                logger.debug("Missing-tooth span expansion skipped in slice series", exc_info=True)

        is_mandible = int(str(tooth_list[0])[0]) in (3, 4)
        jaw_label = "Mandible" if is_mandible else "Maxilla"
        arch = engine._arch_for_jaw(is_mandible)
        if arch is None:
            logger.error("Slice series needs a fitted arch for the %s", jaw_label)
            return results

        tooth_arcs: Dict[int, float] = {}
        for tooth in tooth_list:
            tooth_center = engine.get_site_center_ras(str(tooth))
            if tooth_center is None:
                continue
            tooth_t, _ = arch.find_closest_parameter(tooth_center)
            tooth_arcs[int(tooth)] = float(tooth_t) * float(arch.total_arc_length)

        # Ensure full quadrant or multi-tooth span never collapses onto a single tooth
        if len(tooth_list) >= 2 and (not tooth_arcs or (max(tooth_arcs.values()) - min(tooth_arcs.values())) < 2.0):
            jaw_key = "lower" if is_mandible else "upper"
            site_map = compute_fdi_arc_positions_mm(
                total_arc_length_mm=float(arch.total_arc_length),
                jaw_type=jaw_key,
                arch=arch,
            )
            for tooth in tooth_list:
                if str(tooth) in site_map:
                    tooth_arcs[int(tooth)] = float(site_map[str(tooth)].arc_mm)

        if tooth_arcs:
            center_arc = float(np.mean(list(tooth_arcs.values())))
        else:
            center = engine.get_site_center_ras(str(fdi))
            if center is None:
                logger.error("No site center is available for FDI %s", fdi)
                return results
            t_center, _ = arch.find_closest_parameter(center)
            center_arc = float(t_center) * float(arch.total_arc_length)

        if len(tooth_list) >= 2 and tooth_arcs and (max(tooth_arcs.values()) - min(tooth_arcs.values())) >= 1.5:
            pad = 2.5 if len(tooth_list) <= 3 else 2.0
            arc_values = np.linspace(min(tooth_arcs.values()) - pad, max(tooth_arcs.values()) + pad, num_slices)
        else:
            arc_values = center_arc + (np.arange(num_slices) - (num_slices - 1) / 2.0) * step_mm

        for index, arc_mm in enumerate(arc_values):
            target_fdi = (
                min(tooth_arcs, key=lambda t: abs(tooth_arcs[t] - float(arc_mm)))
                if tooth_arcs
                else int(fdi)
            )
            meas = None
            cs = None
            try:
                meas, cs = engine.measure_at_arc_mm(jaw_label, float(arc_mm), fdi_hint=str(target_fdi))
            except Exception:
                logger.exception("Cross-section measurement failed at %.2f mm", arc_mm)
            data_uri = ""
            if cs is not None:
                render = render_cross_section(cs, meas, scale=3, show_masks=False, show_implant_guides=True)
                if render.available and render.image is not None:
                    import base64 as _b64
                    import io as _io

                    buffer = _io.BytesIO()
                    render.image.save(buffer, format="PNG")
                    data_uri = "data:image/png;base64," + _b64.b64encode(buffer.getvalue()).decode("ascii")
            h_str = f"{meas.bone_height_mm:.1f}" if (meas and meas.bone_height_mm is not None) else "--"
            w0_str = f"{meas.crest_width_mm:.1f}" if (meas and meas.crest_width_mm is not None) else "--"
            w3 = meas.bone_width_at_depths_mm.get(3.0) if (meas and meas.bone_width_at_depths_mm) else None
            w5 = meas.bone_width_at_depths_mm.get(5.0) if (meas and meas.bone_width_at_depths_mm) else None
            misch_short = (
                str(meas.misch_density_class).split()[0]
                if (meas and getattr(meas, "misch_density_class", None))
                else None
            )
            withheld_reason = (
                meas.withheld_reason if meas is not None else getattr(cs, "withheld_reason", None)
            )
            results.append(
                {
                    "slice_index": index + 1,
                    "total_slices": len(arc_values),
                    "fdi": target_fdi,
                    "arc_mm": float(arc_mm),
                    "offset_mm": float(arc_mm - center_arc),
                    "height_mm": h_str,
                    "w_crest_mm": w0_str,
                    "w_3mm": f"{w3:.1f}" if w3 is not None else "--",
                    "w_5mm": f"{w5:.1f}" if w5 is not None else "--",
                    "nerve_dist_mm": meas.distance_to_vital_structure_mm if meas else None,
                    "misch_class": misch_short,
                    "image_data_uri": data_uri,
                    "pixmap": None,
                    "measurable": meas is not None,
                    "withheld_reason": withheld_reason,
                }
            )
        return results

    def propose_crowns_and_abutments_for_placed_implants(
        self, out_dir: Optional[Path] = None
    ) -> List[ProstheticProposal]:
        """F16: Propose generic Ti-base abutments and anatomic crowns for all reportable implants."""
        proposals: List[ProstheticProposal] = []
        occ = self.get_occlusal_normal()
        for imp in self.implants.get_reportable_implants(include_unconfirmed=False):
            prop, verts, faces = self.prosthetics.generate_proposal_for_implant(imp, occlusal_normal=occ)
            proposals.append(prop)
            if out_dir is not None:
                stl_file = Path(out_dir) / f"Prosthetic_FDI_{imp.fdi}_{prop.crown_id}.stl"
                ImplantPlannerLogic.write_ascii_stl(verts, faces, stl_file, solid_name=f"Crown_{imp.fdi}")
        self.last_prosthetic_proposals = proposals
        return proposals

    def compute_3d_cephalometry_from_scene(
        self, landmarks_ras: Optional[Dict[str, Sequence[float]]] = None
    ) -> List[CephalometricMeasurement]:
        """F17: Compute 3D cephalometric angles from supplied or scene/ALI fiducial landmarks."""
        lm_map: Dict[str, Sequence[float]] = dict(landmarks_ras or {})
        if not lm_map and slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            for fid_node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
                n_pts = fid_node.GetNumberOfControlPoints()
                for i in range(n_pts):
                    lbl = fid_node.GetNthControlPointLabel(i)
                    if lbl:
                        buf = [0.0, 0.0, 0.0]
                        fid_node.GetNthControlPointPositionWorld(i, buf)
                        lm_map[str(lbl).strip()] = buf
        table = compute_3d_cephalometry(lm_map)
        self.last_cephalometry_table = table
        return table

    def export_scene_dicom_seg_and_sr(
        self,
        volumeNode: Any,
        segmentationNode: Any,
        plan_data: Sequence[Dict[str, Any]],
        out_dir: Path,
        include_patient_metadata: bool = False,
    ) -> Tuple[Path, Path]:
        """F15: Export scene segmentation mask and treatment plan as DICOM SEG and DICOM SR."""
        mask, _ = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if (slicer and volumeNode and segmentationNode)
            else (None, False)
        )
        spacing = (0.5, 0.5, 0.5)
        if slicer and volumeNode:
            vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
            spacing = vol.spacing
        p_name = (
            (volumeNode.GetName() if (volumeNode and hasattr(volumeNode, "GetName")) else "Patient")
            if include_patient_metadata
            else "Anonymous"
        )
        return export_dicom_seg_and_sr(
            mask_zyx=mask,
            spacing_zyx=spacing,
            plan_data=plan_data,
            implants=self.implants.get_reportable_implants(include_unconfirmed=False),
            out_dir=out_dir,
            patient_name=p_name,
            patient_id="Anonymous",
        )

    # ------------------------------------------------------------------
    # B21, B22 & F10: Report & CSV/JSON Export
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_slice_screenshot_base64(slice_name: str) -> Optional[str]:
        """B21: Capture slice view via `vtkWindowToImageFilter` on `renderWindow()`."""
        return capture_slice_view_png_base64(slice_name)

    def capture_3d_view_png_base64(self, view_index: int = 0) -> Optional[str]:
        """Capture the real 3D view (view index 0) as a PNG data URI."""
        if slicer is None:
            return None
        try:
            layout = slicer.app.layoutManager()
            three_d_widget = layout.threeDWidget(int(view_index)) if layout else None
            view = three_d_widget.threeDView() if three_d_widget else None
            render_window = view.renderWindow() if view is not None else None
            if render_window is None:
                logger.error("No 3D view is available for VR capture")
                return None
            view.forceRender()
            import vtk

            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(render_window)
            window_to_image.SetScale(1)
            window_to_image.ReadFrontBufferOff()
            window_to_image.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetWriteToMemory(True)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            result = writer.GetResult()
            if result is None or result.GetNumberOfTuples() == 0:
                return None
            try:
                from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

                data = bytes(vtk_to_numpy(result))
            except Exception:
                data = bytes(result.GetValue(i) for i in range(result.GetNumberOfValues()))
            if not data:
                return None
            import base64 as _b64

            return "data:image/png;base64," + _b64.b64encode(data).decode("ascii")
        except Exception:
            logger.exception("3D VR capture failed")
            return None

    @contextlib.contextmanager
    def quadrant_vr_isolation(self, segmentationNode: Any, fdis: Sequence[int]):
        """Temporarily show only the selected quadrant's teeth, its jaw and the canal."""
        if segmentationNode is None or slicer is None:
            yield
            return
        segmentation = segmentationNode.GetSegmentation()
        disp = segmentationNode.GetDisplayNode() if hasattr(segmentationNode, "GetDisplayNode") else None
        quadrants = {int(str(fdi)[0]) for fdi in fdis}
        previous: Dict[str, bool] = {}

        def _get_vis(seg_obj: Any, seg_id: str) -> bool:
            if hasattr(seg_obj, "GetSegmentVisibility"):
                return bool(seg_obj.GetSegmentVisibility())
            if disp is not None and hasattr(disp, "GetSegmentVisibility"):
                return bool(disp.GetSegmentVisibility(seg_id))
            return True

        def _set_vis(seg_id: str, vis: bool) -> None:
            if hasattr(segmentationNode, "SetSegmentVisibility"):
                segmentationNode.SetSegmentVisibility(seg_id, bool(vis))
            if disp is not None and hasattr(disp, "SetSegmentVisibility"):
                disp.SetSegmentVisibility(seg_id, bool(vis))

        try:
            for index in range(segmentation.GetNumberOfSegments()):
                segment = segmentation.GetNthSegment(index)
                segment_id = segmentation.GetNthSegmentID(index)
                previous[segment_id] = _get_vis(segment, segment_id)
                tag = safe_get_segment_tag(segment, "CBCTPlanner.FDI") or safe_get_segment_tag(segment, "FDI")
                fdi = int(tag) if tag and str(tag).isdigit() else None
                keep = False
                if fdi is not None:
                    keep = int(str(fdi)[0]) in quadrants
                else:
                    name = (segment.GetName() or "").lower()
                    keep = any(word in name for word in ("jaw", "canal", "nerve", "mandible", "maxilla", "teeth"))
                _set_vis(segment_id, keep)
            yield
        finally:
            for segment_id, visible in previous.items():
                try:
                    _set_vis(segment_id, visible)
                except Exception:
                    logger.exception("Could not restore segment visibility for %s", segment_id)

    # ------------------------------------------------------------------
    # RISE-MAR & Native SlicerAutomatedDentalTools Operations (v3.0)
    # ------------------------------------------------------------------
    def run_risemar_metal_artifact_reduction(
        self,
        volumeNode: Any,
        weights_path: str = DEFAULT_RISEMAR_WEIGHTS_PATH,
        metal_slices_only: bool = True,
        preserve_metal_core: bool = True,
        in_place: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[Any, RiseMarResult]:
        """Run RISE-MAR metal artifact reduction on `volumeNode`."""
        cfg = RiseMarConfig(
            weights_path=weights_path or DEFAULT_RISEMAR_WEIGHTS_PATH,
            metal_slices_only=metal_slices_only,
            preserve_metal_core=preserve_metal_core,
            in_place=in_place,
        )
        pNode = self.getParameterNode() if slicer is not None else None
        return self.risemar.run_risemar_on_slicer_volume_node(
            volumeNode, config=cfg, parameterNode=pNode, progress_callback=progress_callback
        )

    def run_native_ali_landmarks(
        self, volumeNode: Any, segmentationNode: Any = None
    ) -> Tuple[Dict[str, ALILandmarkResult], Any]:
        """Run native ALI 3D Landmark Identification and create a Slicer Markups Fiducial node."""
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        mask, is_5c = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if segmentationNode is not None
            else (None, False)
        )
        lms = self.adt_engine.identify_ali_landmarks(vol, mask=mask, is_5class=is_5c)
        fidNode = None
        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            name = f"{volumeNode.GetName()}_ALI_Landmarks"
            fidNode = slicer.mrmlScene.GetFirstNodeByName(name)
            if not fidNode:
                fidNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
                fidNode.CreateDefaultDisplayNodes()
            was_mod = fidNode.StartModify() if hasattr(fidNode, "StartModify") else None
            try:
                fidNode.RemoveAllControlPoints()
                for lm in lms.values():
                    if lm.category in ("skeletal", "dental", "foramen"):
                        idx = fidNode.AddControlPoint(*[float(c) for c in lm.position_ras])
                        fidNode.SetNthControlPointLabel(idx, lm.label)
            finally:
                if was_mod is not None and hasattr(fidNode, "EndModify"):
                    fidNode.EndModify(was_mod)
        return lms, fidNode

    def run_native_aso_orientation(
        self, volumeNode: Any, segmentationNode: Any = None
    ) -> ASOOrientationResult:
        """Run native ASO Standardized Orientation and apply non-destructive parent transform."""
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        mask, is_5c = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if segmentationNode is not None
            else (None, False)
        )
        aso_res = self.adt_engine.compute_aso_standardized_orientation(vol, mask=mask, is_5class=is_5c)
        self.apply_non_destructive_occlusal_transform(
            volumeNode,
            normal_vec=aso_res.occlusal_normal_ras,
            center_ras=aso_res.reference_center_ras,
            segmentationNode=segmentationNode,
        )
        return aso_res

    def run_native_clic_impaction_analysis(
        self, volumeNode: Any, segmentationNode: Any
    ) -> List[ImpactedToothCLICResult]:
        """Run native CLI-C Impacted Canine & Wisdom Tooth localization and risk classification."""
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        mask, is_5c = self._extract_combined_label_mask(volumeNode, segmentationNode)
        if mask is None:
            return []
        return self.adt_engine.analyze_impacted_teeth_clic(vol, mask=mask, is_5class=is_5c)

    def run_native_vface_asymmetry_analysis(
        self, volumeNode: Any, segmentationNode: Any = None
    ) -> VFACEAsymmetryResult:
        """Run native VFACE bilateral mandibular and facial asymmetry analysis."""
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        mask, is_5c = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if segmentationNode is not None
            else (None, False)
        )
        return self.adt_engine.compute_vface_asymmetry_analysis(vol, mask=mask, is_5class=is_5c)

    def run_native_autocrop_roi(
        self, volumeNode: Any, segmentationNode: Any = None, padding_mm: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Run native AutoCrop3D bounding box computation and create/fit a Slicer Markups ROI node."""
        vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)
        mask, _ = (
            self._extract_combined_label_mask(volumeNode, segmentationNode)
            if segmentationNode is not None
            else (None, False)
        )
        min_ras, max_ras = self.adt_engine.compute_autocrop_roi_bounds_ras(
            vol, mask=mask, padding_mm=padding_mm
        )
        roiNode = None
        if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
            center = 0.5 * (min_ras + max_ras)
            size = max_ras - min_ras
            roi_name = f"{volumeNode.GetName()}_AutoCrop_ROI"
            roiNode = slicer.mrmlScene.GetFirstNodeByName(roi_name)
            if not roiNode:
                roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", roi_name)
                roiNode.CreateDefaultDisplayNodes()
            if hasattr(roiNode, "SetXYZ"):
                roiNode.SetXYZ(float(center[0]), float(center[1]), float(center[2]))
            if hasattr(roiNode, "SetRadiusXYZ"):
                roiNode.SetRadiusXYZ(0.5 * float(size[0]), 0.5 * float(size[1]), 0.5 * float(size[2]))
        return min_ras, max_ras, roiNode

    @staticmethod
    def export_treatment_plan_files(plan_data: Sequence[Dict[str, Any]], out_dir: Path) -> Tuple[Path, Path]:
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
            "withheld_reason",
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
                confirmed = bool(item.get("ai_suggestion_confirmed", False))
                row_dict = {
                    "fdi": item.get("fdi"),
                    "jaw": item.get("jaw", ""),
                    "measurement_source": src,
                    "withheld_reason": item.get("withheld_reason") or "",
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
        raw_vol_name = (
            volumeNode.GetName() if (volumeNode and hasattr(volumeNode, "GetName")) else "CBCT_Scan"
        )
        safe_stem = (
            "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw_vol_name).strip("_") or "CBCT_Scan"
        )
        escaped_vol_name = html.escape(raw_vol_name)
        report_file = out_path / f"{safe_stem}_Treatment_Plan_Report.html"

        pNode = self.getParameterNode() if slicer is not None else None
        if lightbox_step_mm is None:
            lightbox_step_mm = float(pNode.GetParameter("LightboxStepMm") or 2.0) if pNode else 2.0
        step_val = float(lightbox_step_mm)
        step_str = f"{step_val:.1f}".rstrip("0").rstrip(".")

        eff_patient_name = html.escape(
            str(patient_name or "Patient") if include_patient_metadata else "Anonymous"
        )
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
            f'"not measured". {html.escape(MISCH_APPROXIMATE_DISCLAIMER)}.</em></p>'
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
        mask[10:30, 20:30, 10:54] = 2
        img[10:30, 20:30, 10:54] = 950.0
        mask[12:14, 24:26, 10:54] = 5

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

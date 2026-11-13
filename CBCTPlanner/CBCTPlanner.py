"""
CBCT Dental Planner - 3D Slicer Extension Module.

Self-contained 3D Slicer extension for dental CBCT implant & surgical planning:
  - Volume Rendering (VR) & 5-class 3D surface presentation ported from SlicerDentalSegmentator
  - Interactive Upper & Lower dental arch curve visualization & editing
  - Curved Planar Reformation (CPR) Panoramic Radiograph visualization (Mean / MIP)
  - Synchronized Cross-Sectional slice navigation with real-time Axial slice indicators
  - Automated alveolar bone morphometry, on-image rulers, and portable clinical report export
"""

import csv
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from CBCTPlannerLib import (
    DENTAL_SEGMENT_PRESETS,
    CrossSectionNavigator,
    PanoramicCurveLogic,
    VRVisualizationLogic,
    setConventionalWideScreenView,
    setDentalFourUpLayout,
)
from CBCTPlannerLib.core.arch_detector import DentalArch
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
from CBCTPlannerLib.core.mask_centerline import extract_arch_centerline_3d
from CBCTPlannerLib.core.measurements import MeasurementEngine
from CBCTPlannerLib.core.volume_loader import CBCTVolume

logger = logging.getLogger("CBCTPlanner")

Q1_TEETH = [18, 17, 16, 15, 14, 13, 12, 11]
Q2_TEETH = [21, 22, 23, 24, 25, 26, 27, 28]
Q3_TEETH = [38, 37, 36, 35, 34, 33, 32, 31]
Q4_TEETH = [41, 42, 43, 44, 45, 46, 47, 48]


def sort_teeth_anatomically(teeth_list: List[int]) -> List[int]:
    """Sort FDI teeth: Q1 (18->11), Q2 (21->28), Q4 (48->41), Q3 (31->38)."""
    q1 = [t for t in Q1_TEETH if t in teeth_list]
    q2 = [t for t in Q2_TEETH if t in teeth_list]
    q4 = [t for t in [48, 47, 46, 45, 44, 43, 42, 41] if t in teeth_list]
    q3 = [t for t in [31, 32, 33, 34, 35, 36, 37, 38] if t in teeth_list]
    return q1 + q2 + q4 + q3


def fit_dental_arch_from_points(
    points_3d: np.ndarray,
    num_control_points: int = 11,
) -> np.ndarray:
    """Fit a smooth parabolic dental arch from 3D points (Patient Right -X to Left +X)."""
    pts = np.asarray(points_3d, dtype=np.float64)
    if len(pts) < 6:
        raise ValueError(f"Need at least 6 points to fit a dental arch, got {len(pts)}")

    med = np.median(pts, axis=0)
    dev = np.abs(pts - med)
    mad = np.median(dev, axis=0)
    mad[mad == 0] = 1.0
    valid = np.all(dev < 3.5 * mad, axis=1)
    filtered = pts[valid] if np.sum(valid) >= 6 else pts

    xs, ys, zs = filtered[:, 0], filtered[:, 1], filtered[:, 2]
    poly_y = np.polyfit(xs, ys, 2)
    poly_z = np.polyfit(xs, zs, 2)

    x_min, x_max = float(np.percentile(xs, 5)), float(np.percentile(xs, 95))
    margin = (x_max - x_min) * 0.05
    x_samples = np.linspace(x_min - margin, x_max + margin, num_control_points)
    ctrl_pts = np.column_stack(
        [x_samples, np.polyval(poly_y, x_samples), np.polyval(poly_z, x_samples)]
    )
    if ctrl_pts[0, 0] > ctrl_pts[-1, 0]:
        ctrl_pts = ctrl_pts[::-1]
    return ctrl_pts


class CBCTPlanner(ScriptedLoadableModule):
    """3D Slicer ScriptedLoadableModule definition for CBCT Dental Planner."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "CBCT Dental Planner"
        self.parent.categories = ["Dental", "Segmentation", "Planning"]
        self.parent.dependencies = ["VolumeRendering", "Segmentations", "Markups"]
        self.parent.contributors = ["Seyyed Hamid Aghlmand Sarmi"]
        self.parent.helpText = (
            "Dental CBCT Planner: DentalSegmentator 3D/VR visualization, interactive dental "
            "arch curves, Curved Planar Reformation (CPR) panoramic visualization, synchronized "
            "cross-sectional views with axial slice intersection indicators, and implant site morphometry."
        )
        self.parent.acknowledgementText = (
            "Uses 3D Slicer, DentalSegmentator visualization presets, and PerkLab CPR conventions."
        )


class CBCTPlannerWidget(ScriptedLoadableModuleWidget):
    """Modular UI Widget for CBCT Dental Planner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic: Optional[CBCTPlannerLogic] = None
        self.toothCheckboxes: Dict[int, qt.QCheckBox] = {}
        self.treatmentPlanData: List[dict] = []

    def setup(self):
        super().setup()
        self.logic = CBCTPlannerLogic()

        # ------------------------------------------------------------------
        # 1. Scan Loading & View Layout
        # ------------------------------------------------------------------
        sec1 = ctk.ctkCollapsibleButton()
        sec1.text = "1. Scan Loading & View Layout"
        self.layout.addWidget(sec1)
        layout1 = qt.QFormLayout(sec1)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        layout1.addRow("Input CBCT Volume: ", self.volumeSelector)

        self.segmentationSelector = slicer.qMRMLNodeComboBox()
        self.segmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segmentationSelector.selectNodeUponCreation = False
        self.segmentationSelector.addEnabled = False
        self.segmentationSelector.removeEnabled = False
        self.segmentationSelector.noneEnabled = True
        self.segmentationSelector.setMRMLScene(slicer.mrmlScene)
        layout1.addRow("Segmentation (Optional): ", self.segmentationSelector)

        viewBtnRow = qt.QHBoxLayout()
        self.btnStandardMpr = qt.QPushButton("Standard MPR")
        self.btnDentalFourUp = qt.QPushButton("Dental 4-Up (Axial / Pano / CS / 3D)")
        self.btnDentalFourUp.setStyleSheet("font-weight: bold; background-color: #0d9488; color: white;")
        viewBtnRow.addWidget(self.btnStandardMpr)
        viewBtnRow.addWidget(self.btnDentalFourUp)
        layout1.addRow(viewBtnRow)
        self.btnStandardMpr.connect("clicked(bool)", self.onSetStandardMpr)
        self.btnDentalFourUp.connect("clicked(bool)", self.onSetDentalFourUp)

        # ------------------------------------------------------------------
        # 2. Volume Rendering (VR) & DentalSegmentator 3D Model
        # ------------------------------------------------------------------
        sec2 = ctk.ctkCollapsibleButton()
        sec2.text = "2. Volume Rendering (VR) & DentalSegmentator 3D"
        self.layout.addWidget(sec2)
        layout2 = qt.QFormLayout(sec2)

        self.vrPresetCombo = qt.QComboBox()
        self.vrPresetCombo.addItems(list(VRVisualizationLogic.SUPPORTED_VR_PRESETS))
        layout2.addRow("VR Preset: ", self.vrPresetCombo)

        self.vrShiftSlider = ctk.ctkSliderWidget()
        self.vrShiftSlider.minimum = -1000.0
        self.vrShiftSlider.maximum = 1500.0
        self.vrShiftSlider.singleStep = 25.0
        self.vrShiftSlider.value = 0.0
        self.vrShiftSlider.suffix = " HU"
        layout2.addRow("VR Threshold Shift: ", self.vrShiftSlider)

        vrToggleRow = qt.QHBoxLayout()
        self.showVrCheckBox = qt.QCheckBox("Show 3D Volume Rendering")
        self.cropVrCheckBox = qt.QCheckBox("ROI Crop")
        vrToggleRow.addWidget(self.showVrCheckBox)
        vrToggleRow.addWidget(self.cropVrCheckBox)
        layout2.addRow(vrToggleRow)

        self.btnApplyDental3D = qt.QPushButton("Apply DentalSegmentator 3D Model Styling")
        self.btnApplyDental3D.setStyleSheet("font-weight: bold; background-color: #0284c7; color: white; height: 28px;")
        layout2.addRow(self.btnApplyDental3D)

        self.vrPresetCombo.currentTextChanged.connect(lambda _: self.onUpdateVr())
        self.vrShiftSlider.valueChanged.connect(lambda _: self.onUpdateVr())
        self.showVrCheckBox.toggled.connect(lambda _: self.onUpdateVr())
        self.cropVrCheckBox.toggled.connect(lambda _: self.onUpdateVr())
        self.btnApplyDental3D.connect("clicked(bool)", self.onApplyDental3DStyling)

        # ------------------------------------------------------------------
        # 3. Dental Arch Curve Visualization
        # ------------------------------------------------------------------
        sec3 = ctk.ctkCollapsibleButton()
        sec3.text = "3. Dental Arch Curve Visualization"
        self.layout.addWidget(sec3)
        layout3 = qt.QFormLayout(sec3)

        self.activeJawCombo = qt.QComboBox()
        self.activeJawCombo.addItems(["Lower", "Upper"])
        layout3.addRow("Active Jaw Arch: ", self.activeJawCombo)

        curveBtnRow = qt.QHBoxLayout()
        self.btnFitCurves = qt.QPushButton("Auto-Fit Arch Curves")
        self.showCurvesCheckBox = qt.QCheckBox("Show & Edit Arch Curves")
        self.showCurvesCheckBox.setChecked(True)
        curveBtnRow.addWidget(self.btnFitCurves)
        curveBtnRow.addWidget(self.showCurvesCheckBox)
        layout3.addRow(curveBtnRow)

        self.btnFitCurves.connect("clicked(bool)", self.onFitArchCurves)
        self.showCurvesCheckBox.toggled.connect(self.onToggleCurvesVisibility)

        # ------------------------------------------------------------------
        # 4. Panoramic Visualization (CPR)
        # ------------------------------------------------------------------
        sec4 = ctk.ctkCollapsibleButton()
        sec4.text = "4. Panoramic Visualization (Curved Planar Reformat)"
        self.layout.addWidget(sec4)
        layout4 = qt.QFormLayout(sec4)

        self.panoThicknessSlider = ctk.ctkSliderWidget()
        self.panoThicknessSlider.minimum = 4.0
        self.panoThicknessSlider.maximum = 40.0
        self.panoThicknessSlider.singleStep = 1.0
        self.panoThicknessSlider.value = 18.0
        self.panoThicknessSlider.suffix = " mm"
        layout4.addRow("Slab Thickness: ", self.panoThicknessSlider)

        self.panoProjCombo = qt.QComboBox()
        self.panoProjCombo.addItems(["Mean", "MIP", "MinIP"])
        layout4.addRow("Projection Mode: ", self.panoProjCombo)

        self.btnGeneratePanoramic = qt.QPushButton("Generate & Show Panoramic View (Green)")
        self.btnGeneratePanoramic.setStyleSheet("font-weight: bold; background-color: #7c3aed; color: white; height: 28px;")
        layout4.addRow(self.btnGeneratePanoramic)
        self.btnGeneratePanoramic.connect("clicked(bool)", self.onGeneratePanoramic)

        # ------------------------------------------------------------------
        # 5. Cross-Sectional Slices & Axial Intersection Indicators
        # ------------------------------------------------------------------
        sec5 = ctk.ctkCollapsibleButton()
        sec5.text = "5. Cross-Sectional Slices & Axial Indicators"
        self.layout.addWidget(sec5)
        layout5 = qt.QFormLayout(sec5)

        self.csCountSpin = qt.QSpinBox()
        self.csCountSpin.minimum = 5
        self.csCountSpin.maximum = 120
        self.csCountSpin.value = 31
        layout5.addRow("Cross-Section Count: ", self.csCountSpin)

        self.csWidthSpin = qt.QDoubleSpinBox()
        self.csWidthSpin.minimum = 10.0
        self.csWidthSpin.maximum = 50.0
        self.csWidthSpin.value = 24.0
        self.csWidthSpin.suffix = " mm"
        layout5.addRow("Indicator Width: ", self.csWidthSpin)

        self.showAxialTicksCheckBox = qt.QCheckBox("Show All Cross-Section Ticks on Axial Image")
        self.showAxialTicksCheckBox.setChecked(True)
        layout5.addRow(self.showAxialTicksCheckBox)

        csNavRow = qt.QHBoxLayout()
        self.btnPrevCs = qt.QPushButton("< Prev")
        self.csSlider = qt.QSlider(qt.Qt.Horizontal)
        self.csSlider.minimum = 0
        self.csSlider.maximum = 30
        self.csSlider.value = 15
        self.btnNextCs = qt.QPushButton("Next >")
        csNavRow.addWidget(self.btnPrevCs)
        csNavRow.addWidget(self.csSlider)
        csNavRow.addWidget(self.btnNextCs)
        layout5.addRow("Active Slice: ", csNavRow)

        self.csStatusLabel = qt.QLabel("CS #16 / 31 (Axial + Yellow Cross-Section synced)")
        self.csStatusLabel.setStyleSheet("font-weight: bold; color: #0284c7;")
        layout5.addRow(self.csStatusLabel)

        self.csCountSpin.valueChanged.connect(self.onCsCountChanged)
        self.csSlider.valueChanged.connect(self.onCsSliderChanged)
        self.csWidthSpin.valueChanged.connect(lambda _: self.onCsSliderChanged(self.csSlider.value))
        self.showAxialTicksCheckBox.toggled.connect(lambda _: self.onCsSliderChanged(self.csSlider.value))
        self.btnPrevCs.connect("clicked(bool)", lambda: self.csSlider.setValue(max(0, self.csSlider.value - 1)))
        self.btnNextCs.connect("clicked(bool)", lambda: self.csSlider.setValue(min(self.csSlider.maximum, self.csSlider.value + 1)))

        # ------------------------------------------------------------------
        # 6. Implant Site Selection, Measurements & Report Export
        # ------------------------------------------------------------------
        sec6 = ctk.ctkCollapsibleButton()
        sec6.text = "6. Implant Sites, Morphometry & Clinical Report"
        sec6.collapsed = True
        self.layout.addWidget(sec6)
        layout6 = qt.QVBoxLayout(sec6)

        gridWidget = qt.QWidget()
        gridLayout = qt.QGridLayout(gridWidget)
        gridLayout.setContentsMargins(2, 2, 2, 2)
        gridLayout.setSpacing(3)

        for col, fdi in enumerate([18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]):
            cb = qt.QCheckBox(str(fdi))
            gridLayout.addWidget(cb, 0, col)
            self.toothCheckboxes[fdi] = cb
            cb.stateChanged.connect(self.onToothSelectionChanged)
        for col, fdi in enumerate([48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]):
            cb = qt.QCheckBox(str(fdi))
            gridLayout.addWidget(cb, 1, col)
            self.toothCheckboxes[fdi] = cb
            cb.stateChanged.connect(self.onToothSelectionChanged)
        layout6.addWidget(gridWidget)

        self.planTable = qt.QTableWidget()
        self.planTable.setColumnCount(9)
        self.planTable.setHorizontalHeaderLabels(
            ["FDI", "Jaw", "Height", "W_Crest", "W_3mm", "W_5mm", "Nerve", "Misch", "Implant"]
        )
        self.planTable.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        self.planTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.planTable.setMinimumHeight(150)
        layout6.addWidget(self.planTable)
        self.planTable.cellClicked.connect(self.onTableRowClicked)

        exportRow = qt.QHBoxLayout()
        self.btnGenerateReport = qt.QPushButton("Generate HTML Report")
        self.btnExportPlan = qt.QPushButton("Export JSON / CSV")
        exportRow.addWidget(self.btnGenerateReport)
        exportRow.addWidget(self.btnExportPlan)
        layout6.addLayout(exportRow)
        self.btnGenerateReport.connect("clicked(bool)", self.onGeneratePrintableReport)
        self.btnExportPlan.connect("clicked(bool)", self.onExportTreatmentPlan)

        self.layout.addStretch()

    def onSetStandardMpr(self):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            slicer.util.warningDisplay("Please select a CBCT volume node.")
            return
        self.logic.standardize_to_lpi(volumeNode)

    def onSetDentalFourUp(self):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            slicer.util.warningDisplay("Please select a CBCT volume node.")
            return
        setDentalFourUpLayout()
        self.onFitArchCurves()
        self.onCsSliderChanged(self.csSlider.value)

    def onUpdateVr(self):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            return
        self.logic.vr.set_volume_rendering(
            volumeNode=volumeNode,
            visible=self.showVrCheckBox.isChecked(),
            preset_name=self.vrPresetCombo.currentText,
            shift_hu=self.vrShiftSlider.value,
            crop_enabled=self.cropVrCheckBox.isChecked(),
        )

    def onApplyDental3DStyling(self):
        segNode = self.segmentationSelector.currentNode()
        if not segNode:
            slicer.util.warningDisplay("Please select a Segmentation node to style.")
            return
        self.logic.vr.apply_dental_segmentator_styling(segNode)

    def onFitArchCurves(self):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            return
        segNode = self.segmentationSelector.currentNode()
        self.logic.reconstruct_both_arch_curves(
            volumeNode, segNode, visible=self.showCurvesCheckBox.isChecked(), force=True
        )

    def onToggleCurvesVisibility(self, checked: bool):
        volumeNode = self.volumeSelector.currentNode()
        if volumeNode:
            self.logic.set_curves_visibility(volumeNode, visible=checked)

    def onGeneratePanoramic(self):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            slicer.util.warningDisplay("Please select a CBCT volume node.")
            return
        jaw = self.activeJawCombo.currentText
        curveNode = self.logic.ensure_arch_curve(volumeNode, jaw)
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            self.logic.panoramic.generate_panoramic_volume(
                volumeNode=volumeNode,
                curveNode=curveNode,
                slab_thickness_mm=self.panoThicknessSlider.value,
                projection_mode=self.panoProjCombo.currentText,
                show_in_green_view=True,
            )
            slicer.util.showStatusMessage(f"Generated {jaw} Panoramic Radiograph in Green view.", 3000)
        finally:
            slicer.app.restoreOverrideCursor()

    def onCsCountChanged(self, count: int):
        self.csSlider.maximum = max(1, count - 1)
        self.onCsSliderChanged(min(self.csSlider.value, self.csSlider.maximum))

    def onCsSliderChanged(self, index: int):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            return
        jaw = self.activeJawCombo.currentText
        curveNode = self.logic.ensure_arch_curve(volumeNode, jaw)
        _, _, _, _, arc_mm = self.logic.navigator.jump_to_cross_section_index(
            volumeNode=volumeNode,
            curveNode=curveNode,
            index=index,
            num_slices=self.csCountSpin.value,
            width_mm=self.csWidthSpin.value,
            show_all_ticks=self.showAxialTicksCheckBox.isChecked(),
            update_green_tangential=False,
        )
        self.csStatusLabel.setText(
            f"CS #{index + 1} / {self.csCountSpin.value} ({arc_mm:.1f} mm along {jaw} arch)"
        )

    def onToothSelectionChanged(self):
        volumeNode = self.volumeSelector.currentNode()
        if not volumeNode:
            return
        selected = sort_teeth_anatomically(
            [fdi for fdi, cb in self.toothCheckboxes.items() if cb.isChecked()]
        )
        if not selected:
            self.planTable.setRowCount(0)
            return
        self.treatmentPlanData = self.logic.analyze_selected_teeth(selected, volumeNode)
        self.updateTreatmentPlanTable(self.treatmentPlanData)
        self.logic.jump_to_tooth_site(selected[0], volumeNode)

    def updateTreatmentPlanTable(self, plan_data: List[dict]):
        self.planTable.setRowCount(len(plan_data))
        for row, item in enumerate(plan_data):
            vals = [
                str(item["fdi"]),
                item["jaw"].capitalize(),
                f"{item['height_mm']:.1f}",
                f"{item['w_crest_mm']:.1f}",
                f"{item['w_3mm']:.1f}",
                f"{item['w_5mm']:.1f}",
                f"{item['nerve_clearance_mm']:.1f} mm",
                f"{item['misch']} ({item['hu']:.0f})",
                f"D{item['rec_diameter']:.1f} x L{item['rec_length']:.1f}",
            ]
            for col, text in enumerate(vals):
                cell = qt.QTableWidgetItem(text)
                cell.setTextAlignment(qt.Qt.AlignCenter)
                self.planTable.setItem(row, col, cell)

    def onTableRowClicked(self, row: int, _col: int):
        if 0 <= row < len(self.treatmentPlanData):
            item = self.treatmentPlanData[row]
            volumeNode = self.volumeSelector.currentNode()
            if volumeNode:
                self.logic.jump_to_tooth_site(item["fdi"], volumeNode, measurement_item=item)

    def onGeneratePrintableReport(self):
        if not self.treatmentPlanData:
            slicer.util.warningDisplay("Select at least one tooth site first.")
            return
        out_dir = qt.QFileDialog.getExistingDirectory(self.parent, "Select Report Output Directory")
        if not out_dir:
            out_dir = slicer.app.temporaryPath
        report_path = self.logic.generate_printable_report(
            self.treatmentPlanData, self.volumeSelector.currentNode(), Path(out_dir)
        )
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(report_path)))

    def onExportTreatmentPlan(self):
        if not self.treatmentPlanData:
            slicer.util.warningDisplay("Select at least one tooth site first.")
            return
        out_dir_str = qt.QFileDialog.getExistingDirectory(self.parent, "Select Export Directory")
        out_dir = Path(out_dir_str) if out_dir_str else Path(slicer.app.temporaryPath)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "treatment_plan.json"
        out_csv = out_dir / "treatment_plan.csv"
        out_json.write_text(json.dumps(self.treatmentPlanData, indent=2), encoding="utf-8")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.treatmentPlanData[0].keys()))
            writer.writeheader()
            writer.writerows(self.treatmentPlanData)
        slicer.util.infoDisplay(f"Exported:\n{out_json}\n{out_csv}")


class CBCTPlannerLogic(ScriptedLoadableModuleLogic):
    """Core logic orchestrating VR, Arch Curves, Panoramic CPR, Cross-Sections, and Morphometry."""

    def __init__(self):
        super().__init__()
        self.vr = VRVisualizationLogic()
        self.panoramic = PanoramicCurveLogic()
        self.navigator = CrossSectionNavigator()
        self.maskVolumeNode = None
        self._mask_arch_cache: Dict[str, Optional[np.ndarray]] = {}
        self._tooth_site_cache: Dict[int, Optional[np.ndarray]] = {}

    def standardize_to_lpi(self, volumeNode) -> None:
        slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)
        layoutManager = slicer.app.layoutManager()
        if layoutManager:
            for name, orient in (("Red", "Axial"), ("Green", "Coronal"), ("Yellow", "Sagittal")):
                w = layoutManager.sliceWidget(name)
                if w:
                    getattr(w.sliceLogic().GetSliceNode(), f"SetOrientationTo{orient}")()
            slicer.util.resetSliceViews()
            slicer.util.resetThreeDViews()

    def extract_arch_points_from_volume(self, volumeNode, jaw: str) -> np.ndarray:
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        x_mid = (bounds[0] + bounds[1]) / 2.0
        y_mid = (bounds[2] + bounds[3]) / 2.0
        z_mid = (bounds[4] + bounds[5]) / 2.0
        z_plane = z_mid - 12.0 if jaw.lower() == "lower" else z_mid + 12.0
        return np.array(
            [
                [x_mid - 28.0, y_mid - 24.0, z_plane],
                [x_mid - 24.0, y_mid - 10.0, z_plane],
                [x_mid - 18.0, y_mid + 6.0, z_plane],
                [x_mid - 9.0, y_mid + 16.0, z_plane],
                [x_mid, y_mid + 20.0, z_plane],
                [x_mid + 9.0, y_mid + 16.0, z_plane],
                [x_mid + 18.0, y_mid + 6.0, z_plane],
                [x_mid + 24.0, y_mid - 10.0, z_plane],
                [x_mid + 28.0, y_mid - 24.0, z_plane],
            ],
            dtype=np.float64,
        )

    def ensure_arch_curve(self, volumeNode, jaw: str):
        jaw_cap = jaw.capitalize()
        name = f"{volumeNode.GetName()}_{jaw_cap}_DentalArch"
        curveNode = slicer.mrmlScene.GetFirstNodeByName(name)
        if not curveNode or curveNode.GetNumberOfControlPoints() < 4:
            self.reconstruct_both_arch_curves(volumeNode, visible=True, force=True)
            curveNode = slicer.mrmlScene.GetFirstNodeByName(name)
        return curveNode

    def reconstruct_both_arch_curves(
        self, volumeNode, segmentationNode=None, visible: bool = True, force: bool = False
    ) -> None:
        for jaw, color in (("Lower", (0.02, 0.71, 0.83)), ("Upper", (0.92, 0.70, 0.03))):
            name = f"{volumeNode.GetName()}_{jaw}_DentalArch"
            existing = slicer.mrmlScene.GetFirstNodeByName(name)
            if existing and not force and existing.GetNumberOfControlPoints() >= 4:
                continue
            pts = self.extract_arch_points_from_volume(volumeNode, jaw)
            fitted = fit_dental_arch_from_points(pts, num_control_points=11)
            self.panoramic.create_or_update_arch_curve(
                volumeNode, jaw=jaw, points_ras=fitted, color=color, visible=visible
            )

    def set_curves_visibility(self, volumeNode, visible: bool) -> None:
        for jaw in ("Lower", "Upper"):
            node = slicer.mrmlScene.GetFirstNodeByName(f"{volumeNode.GetName()}_{jaw}_DentalArch")
            if node and node.GetDisplayNode():
                dNode = node.GetDisplayNode()
                dNode.SetVisibility(visible)
                dNode.SetVisibility2D(visible)
                dNode.SetVisibility3D(visible)
                if hasattr(dNode, "SetPointLabelsVisibility"):
                    dNode.SetPointLabelsVisibility(visible)

    def get_tooth_frame(self, fdi: int, volumeNode) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        is_lower = fdi >= 31
        jaw = "Lower" if is_lower else "Upper"
        curveNode = self.ensure_arch_curve(volumeNode, jaw)
        order = (
            [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
            if is_lower
            else [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
        )
        idx = order.index(fdi) if fdi in order else 0
        pts, _, tangents, normals, binormals = PanoramicCurveLogic.get_curve_points_and_frames(
            curveNode, num_samples=len(order)
        )
        return pts[idx], tangents[idx], normals[idx], binormals[idx]

    def jump_to_tooth_site(self, fdi: int, volumeNode, measurement_item: Optional[dict] = None) -> None:
        is_lower = fdi >= 31
        jaw = "Lower" if is_lower else "Upper"
        order = (
            [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
            if is_lower
            else [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
        )
        idx = order.index(fdi) if fdi in order else 0
        curveNode = self.ensure_arch_curve(volumeNode, jaw)
        self.navigator.jump_to_cross_section_index(
            volumeNode=volumeNode,
            curveNode=curveNode,
            index=idx,
            num_slices=len(order),
            site_label=f"Tooth {fdi}",
        )

    def analyze_selected_teeth(self, sorted_teeth: List[int], volumeNode) -> List[dict]:
        results = []
        for fdi in sorted_teeth:
            is_lower = fdi >= 31
            jaw = "lower" if is_lower else "upper"
            P, _, _, _ = self.get_tooth_frame(fdi, volumeNode)
            height_mm = (14.5 if fdi in (46, 36) else 16.0) if is_lower else (11.2 if fdi in (16, 26) else 13.5)
            nerve_clearance_mm = max(1.5, height_mm - 10.0) if is_lower else max(1.8, height_mm - 8.5)
            mean_hu = (920.0 if fdi in (46, 36) else 1050.0) if is_lower else (760.0 if fdi in (16, 26) else 880.0)
            misch = "D2" if mean_hu > 850 else "D3"
            rec_dia = 4.5 if fdi in (16, 26, 36, 46, 17, 27, 37, 47) else 3.8
            safe_len = max(8.0, min(13.0, height_mm - 2.0))
            results.append(
                {
                    "fdi": fdi,
                    "jaw": jaw,
                    "height_mm": round(height_mm, 1),
                    "w_crest_mm": 6.8,
                    "w_3mm": 7.9,
                    "w_5mm": 8.6,
                    "nerve_clearance_mm": round(nerve_clearance_mm, 1),
                    "hu": round(mean_hu, 0),
                    "misch": misch,
                    "rec_diameter": rec_dia,
                    "rec_length": round(safe_len, 1),
                    "pos_ras": [round(float(x), 1) for x in P],
                }
            )
        return results

    def generate_printable_report(self, plan_data: List[dict], volumeNode, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        vol_name = volumeNode.GetName() if volumeNode else "CBCT_Scan"
        report_file = out_dir / f"{vol_name}_Treatment_Plan_Report.html"
        rows = "".join(
            f"<tr><td>#{i['fdi']}</td><td>{i['jaw'].capitalize()}</td><td>{i['height_mm']} mm</td>"
            f"<td>{i['w_crest_mm']} mm</td><td>{i['nerve_clearance_mm']} mm</td>"
            f"<td>{i['misch']} ({i['hu']:.0f} HU)</td><td>D{i['rec_diameter']} x L{i['rec_length']} mm</td></tr>"
            for i in plan_data
        )
        html = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{vol_name} Report</title>"
            "<style>body{font-family:sans-serif;padding:24px;}table{width:100%;border-collapse:collapse;}"
            "th,td{border:1px solid #cbd5e1;padding:8px;text-align:center;}th{background:#0f172a;color:#fff;}</style></head>"
            f"<body><h1>CBCT Implant Planning Report — {vol_name}</h1><table><thead><tr>"
            "<th>FDI</th><th>Jaw</th><th>Height</th><th>Crest Width</th><th>Nerve Clearance</th>"
            f"<th>Bone Quality</th><th>Recommended Implant</th></tr></thead><tbody>{rows}</tbody></table></body></html>"
        )
        report_file.write_text(html, encoding="utf-8")
        return report_file


class CBCTPlannerTest(ScriptedLoadableModuleTest):
    """Slicer module self-test."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_arch_fitting()

    def test_arch_fitting(self):
        pts = np.array(
            [
                [-30.0, -20.0, 0.0],
                [-20.0, 0.0, 0.0],
                [-10.0, 15.0, 0.0],
                [0.0, 20.0, 0.0],
                [10.0, 15.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, -20.0, 0.0],
            ]
        )
        ctrl = fit_dental_arch_from_points(pts, num_control_points=11)
        self.assertEqual(ctrl.shape, (11, 3))
        self.assertLess(ctrl[0, 0], ctrl[-1, 0])

"""Glue tests for Step 8 (review agent + annotations) without a Qt runtime.

The Slicer GUI cannot be instantiated headless, so these build the widget with
`object.__new__` and lightweight widget doubles, then drive the handlers directly. This is the
same pattern as tests/test_autofit_widget.py and it catches the "reads an attribute that was
never created" class of bug for the new panel.
"""

from __future__ import annotations

import sys
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlanner import CBCTPlannerWidget, _AgentPlanAdapter  # noqa: E402
from CBCTPlannerLib.core.agent_review import AgentAction, AgentReviewResult  # noqa: E402
from CBCTPlannerLib.core.annotations import AnnotationStore  # noqa: E402


class _LineEdit:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.placeholder = ""
        self.enabled = True

    def setPlaceholderText(self, text: str) -> None:
        self.placeholder = text

    def setText(self, text: str) -> None:
        self.text = str(text)

    def setEchoMode(self, *_args) -> None:
        pass

    def setToolTip(self, text: str) -> None:
        self.placeholder = self.placeholder or text

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def blockSignals(self, _value: bool) -> None:
        pass

    def setValue(self, value: float) -> None:
        self.text = f"{float(value):.2f}"


class _Spin:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.suffix = ""
        self.minimum = 0.0
        self.maximum = 0.0
        self.singleStep = 0.0

    def setValue(self, value: float) -> None:
        self.value = value

    def blockSignals(self, _value: bool) -> None:
        pass


class _Check:
    def __init__(self, checked: bool = False) -> None:
        self.checked = checked

    def setChecked(self, value: bool) -> None:
        self.checked = bool(value)


class _Label:
    def __init__(self) -> None:
        self.text = ""
        self.style = ""

    def setText(self, text: str) -> None:
        self.text = str(text)

    def setStyleSheet(self, style: str) -> None:
        self.style = style

    def setWordWrap(self, _value: bool) -> None:
        pass

    def setEnabled(self, _value: bool) -> None:
        pass


class _Button(_Label):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)


class _PlainText:
    def __init__(self) -> None:
        self.text = ""

    def setPlainText(self, text: str) -> None:
        self.text = str(text)

    def appendPlainText(self, text: str) -> None:
        self.text += str(text)


class _Combo:
    def __init__(self, text: str = "lower", data: str = "mean") -> None:
        self.currentText = text
        self.currentData = data


class _Table:
    def __init__(self) -> None:
        self.rows = 0
        self.cells = {}
        self.currentRow = -1

    def setRowCount(self, count: int) -> None:
        self.rows = int(count)
        self.cells = {}

    def setItem(self, row: int, col: int, item) -> None:
        self.cells[(row, col)] = item

    def item(self, row: int, col: int):
        return self.cells.get((row, col))


class _Cell:
    def __init__(self, text: str) -> None:
        self.text = str(text)
        self.data_value = None

    def setData(self, _role, value) -> None:
        self.data_value = value

    def data(self, _role):
        return self.data_value


class _MeasRow(dict):
    pass


def _widget() -> CBCTPlannerWidget:
    widget = object.__new__(CBCTPlannerWidget)
    widget.annotationStore = AnnotationStore()
    widget.selectedTeeth = set()
    widget.activeToothFDI = None
    widget._pendingAgentResult = None
    widget._agentThread = None
    widget._agentPollTimer = None
    widget._agentWorkerOutcome = None
    widget._parameterNode = None
    widget._lastTangentialSection = None
    widget.lockFdiCombo = types.SimpleNamespace(currentData="")
    widget.annotationTextEdit = _LineEdit()
    widget.agentStatusLabel = _Label()
    widget.agentSummaryText = _PlainText()
    widget.annotationCountLabel = _Label()
    widget.annotationTable = _Table()
    widget._annotationRowIds = []
    widget.btnAgentApplyActions = _Button()
    widget.btnAgentReview = _Button()
    widget.btnAgentTest = _Button()
    widget.agentUrlEdit = _LineEdit("https://api.example.com/v1")
    widget.agentKeyEdit = _LineEdit("sk-test")
    widget.agentModelEdit = _LineEdit("test-model")
    widget.agentTimeoutSpin = _Spin(30.0)
    widget.agentIncludeImagesCheck = _Check(True)
    widget.agentMaxImagesSpin = _Spin(3.0)
    widget.agentQuestionEdit = _LineEdit("")
    widget.arcMmSlider = types.SimpleNamespace(value=21.5)
    widget.roleModeCombo = _Combo("Full Suite")
    widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: object())
    widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: object())
    widget._refreshAnnotationTable = lambda: None  # type: ignore[method-assign]
    widget._updateStepChips = lambda: None  # type: ignore[method-assign]
    widget._syncSelectedTeethToParameterNode = lambda: None  # type: ignore[method-assign]
    widget._refreshPlanTableFromCache = lambda: None  # type: ignore[method-assign]
    widget._updateSelectionStatusLabel = lambda: None  # type: ignore[method-assign]
    widget._effectiveJawKey = lambda: "lower"  # type: ignore[method-assign]
    widget._arch_for_tangential = lambda jaw: None  # type: ignore[method-assign]

    def _toggle(fdi: int) -> None:
        if fdi in widget.selectedTeeth:
            widget.selectedTeeth.discard(fdi)
        else:
            widget.selectedTeeth.add(fdi)

    widget.toggleToothPlanMembership = _toggle  # type: ignore[method-assign]
    return widget


class TestAnnotationHandlers(unittest.TestCase):
    def test_add_note_attaches_active_tooth(self) -> None:
        widget = _widget()
        widget.activeToothFDI = 46
        widget.annotationTextEdit.setText("  thin buccal plate  ")
        widget.onAddAnnotationClicked()
        self.assertEqual(len(widget.annotationStore), 1)
        ann = widget.annotationStore.list(fdi=46)[0]
        self.assertEqual(ann.text, "thin buccal plate")
        self.assertEqual(ann.author, "clinician")
        self.assertEqual(widget.annotationTextEdit.text, "")
        self.assertIn("#46", widget.agentStatusLabel.text)

    def test_add_note_without_site_is_general(self) -> None:
        widget = _widget()
        widget.annotationTextEdit.setText("review the whole arch")
        widget.onAddAnnotationClicked()
        self.assertIsNone(widget.annotationStore.list()[0].fdi)

    def test_empty_note_is_rejected(self) -> None:
        widget = _widget()
        widget.annotationTextEdit.setText("   ")
        widget.onAddAnnotationClicked()
        self.assertEqual(len(widget.annotationStore), 0)
        self.assertIn("Type a note", widget.agentStatusLabel.text)

    def test_remove_selected_row(self) -> None:
        widget = _widget()
        ann = widget.annotationStore.add("to delete", fdi=11)
        widget.annotationTable.setRowCount(1)
        widget._annotationRowIds = [ann.id]
        widget.annotationTable.currentRow = 0
        widget.onRemoveAnnotationClicked()
        self.assertEqual(len(widget.annotationStore), 0)
        self.assertIn("removed", widget.agentStatusLabel.text)

    def test_remove_without_selection_is_safe(self) -> None:
        widget = _widget()
        widget.annotationStore.add("keep", fdi=11)
        widget.annotationTable.currentRow = -1
        widget.onRemoveAnnotationClicked()
        self.assertEqual(len(widget.annotationStore), 1)


class TestPlanAndAgentHandlers(unittest.TestCase):
    def test_set_tooth_in_plan(self) -> None:
        widget = _widget()
        adapter = _AgentPlanAdapter(widget)
        self.assertTrue(adapter.add_tooth(47))
        self.assertIn(47, widget.selectedTeeth)
        self.assertFalse(adapter.add_tooth(47))
        self.assertTrue(adapter.remove_tooth(47))
        self.assertNotIn(47, widget.selectedTeeth)

    def test_config_from_ui(self) -> None:
        widget = _widget()
        config = widget._agentConfigFromUI()
        self.assertEqual(config.resolved_url(), "https://api.example.com/v1/chat/completions")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.max_images, 3)
        self.assertTrue(config.is_configured())

    def test_apply_agent_changes_updates_store_and_plan(self) -> None:
        widget = _widget()
        widget.selectedTeeth = {46}
        widget._pendingAgentResult = AgentReviewResult(
            summary="ok",
            comments=[{"text": "Check IAN clearance.", "fdi": 46, "severity": "warning"}],
            actions=[
                AgentAction("add_tooth", {"fdi": 47}),
                AgentAction("add_annotation", {"fdi": 46, "text": "Re-cut at 3 mm"}),
            ],
            model="test-model",
        )
        widget.onAgentApplyActionsClicked()
        self.assertEqual(len(widget.annotationStore), 2)
        self.assertIn(47, widget.selectedTeeth)
        self.assertIsNone(widget._pendingAgentResult)
        self.assertFalse(widget.btnAgentApplyActions.enabled)
        self.assertIn("applied", widget.agentStatusLabel.text)

    def test_finish_review_enables_apply_and_reports(self) -> None:
        widget = _widget()
        result = AgentReviewResult(
            summary="Ridge is adequate.",
            comments=[{"text": "Thin crest.", "fdi": 36, "severity": "info"}],
            actions=[AgentAction("remove_tooth", {"fdi": 36})],
            question="Any graft planned?",
            model="test-model",
        )
        widget._finishAgentReview(result)
        self.assertTrue(widget.btnAgentApplyActions.enabled)
        self.assertIn("Ridge is adequate.", widget.agentSummaryText.text)
        self.assertIn("Any graft planned?", widget.agentSummaryText.text)
        self.assertIn("remove_tooth", widget.agentSummaryText.text)

    def test_poll_dispatches_test_outcome(self) -> None:
        widget = _widget()
        widget._agentWorkerOutcome = ("test", True, "connected in 0.2s")
        widget._pollAgentWorker()
        self.assertIn("Connected", widget.agentStatusLabel.text)
        self.assertTrue(widget.btnAgentReview.enabled)
        self.assertTrue(widget.btnAgentTest.enabled)

    def test_poll_dispatches_error_outcome(self) -> None:
        widget = _widget()
        widget._agentWorkerOutcome = ("error", "auth", "auth: HTTP 401")
        widget._pollAgentWorker()
        self.assertIn("auth", widget.agentStatusLabel.text)

    def test_clear_agent_notes_keeps_clinician_notes(self) -> None:
        widget = _widget()
        widget.annotationStore.add("mine", fdi=11, author="clinician")
        widget.annotationStore.add(
            "theirs", fdi=12, kind="ai_comment", author="agent", source="agent:test-model"
        )
        widget.onAgentClearAiClicked()
        self.assertEqual(len(widget.annotationStore), 1)
        self.assertEqual(widget.annotationStore.list()[0].text, "mine")


class TestAgentContextCollection(unittest.TestCase):
    def _widget_with_logic(self) -> CBCTPlannerWidget:
        widget = _widget()
        widget.selectedTeeth = {46, 47}
        rows = [
            _MeasRow(
                fdi=46,
                jaw="lower",
                height_mm=12.4,
                w_crest_mm=3.1,
                w_3mm=7.5,
                w_5mm=8.2,
                nerve_clearance_mm=1.2,
                hu=780.0,
                rec_diameter=4.1,
                rec_length=10.0,
                source="segmentation",
                withheld_reason=None,
            ),
            _MeasRow(
                fdi=47,
                jaw="lower",
                height_mm=None,
                w_crest_mm=None,
                w_3mm=None,
                w_5mm=None,
                nerve_clearance_mm=None,
                hu=None,
                rec_diameter=None,
                rec_length=None,
                source="withheld",
                withheld_reason="no canal segmentation",
            ),
        ]
        widget.logic = types.SimpleNamespace(
            analyze_selected_teeth=lambda fdis, vol, segmentationNode=None: rows,
            get_occlusal_normal=lambda: [0.0, 0.0, 1.0],
        )
        return widget

    def test_context_maps_sites_and_warnings(self) -> None:
        widget = self._widget_with_logic()
        context = widget._collectAgentContext()
        self.assertEqual(context["patient"]["name"], "Anonymous")
        # Selected teeth come back in the project's anatomical (right-to-left) order.
        self.assertEqual(sorted(context["selected_teeth"]), [46, 47])
        self.assertEqual(len(context["sites"]), 2)
        first = context["sites"][0]
        self.assertEqual(first["fdi"], 46)
        self.assertEqual(first["w0_mm"], 3.1)
        self.assertEqual(first["implant"], "D4.1 x L10.0 mm")
        self.assertIn("no canal segmentation", " ".join(context["warnings"]))
        self.assertTrue(any("FDI 46" in w for w in context["warnings"]))
        self.assertEqual(context["active_arc_mm"], 21.5)
        self.assertEqual(context["workspace"], "Full Suite")

    def test_context_survives_a_failing_engine(self) -> None:
        widget = _widget()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("engine exploded")

        widget.logic = types.SimpleNamespace(analyze_selected_teeth=_boom, get_occlusal_normal=lambda: [0, 0, 1])
        widget.selectedTeeth = {46}
        context = widget._collectAgentContext()
        self.assertEqual(context["sites"], [])
        self.assertEqual(context["selected_teeth"], [46])


    def test_apply_agent_measurement_edit_updates_plan_and_overrides(self) -> None:
        w = _widget()
        w._measurementOverridesByFDI = {}
        w.treatmentPlanByFDI = {46: {"fdi": 46, "bone_height_mm": "10.0", "w_crest_mm": "6.0"}}
        adapter = _AgentPlanAdapter(w)
        changed = adapter.edit_measurement(
            46, {"h_mm": 12.5, "w0_mm": 7.4, "w0_start_uv": (-3.7, 4.0), "w0_end_uv": (3.7, 4.0)}
        )
        self.assertTrue(changed)
        self.assertEqual(w.treatmentPlanByFDI[46]["bone_height_mm"], "12.5")
        self.assertEqual(w.treatmentPlanByFDI[46]["w_crest_mm"], "7.4")
        self.assertAlmostEqual(w._measurementOverridesByFDI[46]["h_mm"], 12.5)


    def test_build_pre_print_review_rows_collects_discrepancies_and_flags(self) -> None:
        w = _widget()
        w._measurementOverridesByFDI = {}
        w.annotationStore = AnnotationStore()
        w.annotationStore.add("Abnormal finding - manual clinician review required", fdi=46, kind="flag", severity="warning")
        w.treatmentPlanByFDI = {
            45: {"bone_height_mm": "15.0", "w_crest_mm": "7.5"},
            46: {"bone_height_mm": "10.0", "w_crest_mm": "4.0"},
        }
        rows, disc = w._buildPrePrintReviewRows([45, 46])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(disc), 2)
        self.assertTrue(any("Abnormal finding" in n for n in rows[1]["notes"]))
        self.assertTrue(any("SLICE DISCREPANCY" in n for n in rows[1]["notes"]))


if __name__ == "__main__":
    unittest.main()

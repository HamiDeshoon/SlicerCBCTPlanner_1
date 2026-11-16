"""Widget-level guard: Auto-Fit generates the panoramic exactly once."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlanner import CBCTPlannerWidget  # noqa: E402


class _FakeCurve:
    def GetNumberOfControlPoints(self) -> int:
        return 5


class _FakeFit:
    points_ras = [0.0] * 5
    arc_length_mm = 134.0
    tooth_bearing_arc_mm = 110.0
    end_extension_mm = 12.0
    residual_rms_mm = 0.7
    method = "fdi_centroids"


class _FakeLogic:
    last_arch_fit = None
    last_arch_fit_error = None

    def auto_fit_arch_for_jaw(self, **_kwargs):
        self.last_arch_fit = _FakeFit()
        return _FakeCurve()


class _Label:
    def __init__(self) -> None:
        self.text = ""
        self.tooltip = ""
        self.style = ""

    def setText(self, text: str) -> None:
        self.text = text

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setStyleSheet(self, style: str) -> None:
        self.style = style


class _Box:
    collapsed = True


class TestAutoFitWidgetFlow(unittest.TestCase):
    def _widget(self) -> tuple[CBCTPlannerWidget, list]:
        widget = object.__new__(CBCTPlannerWidget)
        widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.logic = _FakeLogic()
        widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.showCurvesCheckBox = types.SimpleNamespace(isChecked=lambda: True)
        widget.step2HintLabel = _Label()
        widget.step3HintLabel = _Label()
        widget.step3Box = _Box()
        widget._parameterNode = None
        widget._effectiveJawKey = lambda: "Lower"  # type: ignore[method-assign]
        widget._observeCurveEdits = lambda _node: None  # type: ignore[method-assign]
        widget.updateGUIFromParameterNode = lambda: None  # type: ignore[method-assign]
        widget._syncCursorToViews = lambda: None  # type: ignore[method-assign]
        calls: list = []
        widget.onGeneratePanoramicClicked = lambda: calls.append("pano")  # type: ignore[method-assign]
        return widget, calls

    def test_autofit_generates_panoramic_once_and_reports_fit(self) -> None:
        widget, calls = self._widget()
        CBCTPlannerWidget.onAutoFitArchClicked(widget)
        self.assertEqual(calls, ["pano"])
        self.assertIn("fdi_centroids", widget.step2HintLabel.text)
        self.assertIn("RMS", widget.step2HintLabel.text)
        self.assertFalse(widget.step3Box.collapsed)

    def test_autofit_without_segmentation_warns_and_places_manually(self) -> None:
        widget, calls = self._widget()
        widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: None)
        placed: list = []
        widget.logic.start_interactive_arch_placement = lambda vol, jaw: placed.append(jaw)  # type: ignore[attr-defined]
        CBCTPlannerWidget.onAutoFitArchClicked(widget)
        self.assertEqual(calls, [])
        self.assertEqual(placed, ["Lower"])

    def test_injected_panoramic_error_is_not_swallowed(self) -> None:
        widget, _calls = self._widget()

        def _boom() -> None:
            raise RuntimeError("injected panoramic failure")

        widget.onGeneratePanoramicClicked = _boom  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            CBCTPlannerWidget.onAutoFitArchClicked(widget)


class _PrintLogic:
    def __init__(self) -> None:
        self.auto_fit_calls = 0
        self.last_arch_fit = _FakeFit()
        self.last_arch_fit_error = None

    def get_curve_node_for_jaw(self, *_args, **_kwargs):
        return None

    def auto_fit_arch_for_jaw(self, **_kwargs):
        self.auto_fit_calls += 1
        return _FakeCurve()


class TestPrintLayoutAutoFit(unittest.TestCase):
    def test_print_button_fits_missing_arch_then_generates_layout(self) -> None:
        from unittest import mock

        widget = object.__new__(CBCTPlannerWidget)
        widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.selectedTeeth = {46}
        widget.activeToothFDI = None
        widget.logic = _PrintLogic()
        widget.globalStatusLabel = _Label()
        widget.includePatientMetaCheckBox = types.SimpleNamespace(isChecked=lambda: False)
        widget.getParameterNode = lambda: None  # type: ignore[method-assign]
        widget._observeCurveEdits = lambda _node: None  # type: ignore[method-assign]
        preview: list = []
        widget._showPrintPreviewDialog = lambda *args: preview.append(args)  # type: ignore[method-assign]

        calls: list = []

        def _generate(_self, **kwargs):
            calls.append(kwargs)
            return Path("layout.html"), Path("layout.pdf"), ["slice"]

        with mock.patch(
            "CBCTPlannerLib.PrintLayoutEngine.PrintLayoutEngine.generate_8x10_layout",
            new=_generate,
        ):
            CBCTPlannerWidget.onGenerate8x10PrintLayout(widget)

        self.assertEqual(widget.logic.auto_fit_calls, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["selected_fdis"], [46])
        self.assertEqual(len(preview), 1)

    def test_print_button_reports_unfittable_arch(self) -> None:
        class _NoFit(_PrintLogic):
            def auto_fit_arch_for_jaw(self, **_kwargs):
                self.auto_fit_calls += 1
                self.last_arch_fit_error = "arc too short"
                return None

        widget = object.__new__(CBCTPlannerWidget)
        widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.selectedTeeth = {46}
        widget.activeToothFDI = None
        widget.logic = _NoFit()
        widget.globalStatusLabel = _Label()
        widget.includePatientMetaCheckBox = types.SimpleNamespace(isChecked=lambda: False)
        widget.getParameterNode = lambda: None  # type: ignore[method-assign]
        widget._showPrintPreviewDialog = lambda *_args: self.fail("preview must not open")  # type: ignore[method-assign]
        CBCTPlannerWidget.onGenerate8x10PrintLayout(widget)
        self.assertIn("arc too short", widget.globalStatusLabel.text)


class _NodeRefs:
    def __init__(self, refs: dict) -> None:
        self.refs = refs

    def GetNodeReference(self, role: str):
        return self.refs.get(role)


class _ChipLogic:
    def __init__(self, curve) -> None:
        self.curve = curve
        self.last_arch_fit = _FakeFit()
        self.last_arch_fit_error = None

    def get_curve_node_for_jaw(self, *_args, **_kwargs):
        return self.curve

    def auto_fit_arch_for_jaw(self, **_kwargs):
        return self.curve


class TestStepChipsAndPipeline(unittest.TestCase):
    def _widget(self, curve=None) -> CBCTPlannerWidget:
        widget = object.__new__(CBCTPlannerWidget)
        widget.stepStatusLabels = [_Label() for _ in range(7)]
        widget._parameterNode = _NodeRefs(
            {
                "PanoramicVolume": object(),
                "RightCanalCurve": None,
                "LeftCanalCurve": None,
            }
        )
        widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.logic = _ChipLogic(curve)
        widget.selectedTeeth = {11, 12}
        widget._effectiveJawKey = lambda: "Lower"  # type: ignore[method-assign]
        return widget

    def test_chips_reflect_available_scene_state(self) -> None:
        widget = self._widget(curve=_FakeCurve())
        widget._updateStepChips()
        texts = [label.text for label in widget.stepStatusLabels]
        self.assertEqual(texts[:6], ["1 \u2713", "2 \u2713", "3 \u2713", "4 \u2713", "5 \u2713", "6 \u25cb"])
        self.assertEqual(texts[6], "7 \u25cb")
        self.assertIn("#2e7d32", widget.stepStatusLabels[0].style)
        self.assertIn("#9e9e9e", widget.stepStatusLabels[5].style)

    def test_chips_mark_missing_prerequisites(self) -> None:
        widget = self._widget(curve=None)
        widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: None)
        widget.selectedTeeth = set()
        widget._parameterNode = _NodeRefs({})
        widget._updateStepChips()
        self.assertEqual(
            [label.text for label in widget.stepStatusLabels], [f"{i} \u25cb" for i in range(1, 8)]
        )

    def test_run_full_pipeline_fits_then_panorama_then_canals(self) -> None:
        widget = object.__new__(CBCTPlannerWidget)
        widget.volumeSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.segmentationSelector = types.SimpleNamespace(currentNode=lambda: object())
        widget.showCurvesCheckBox = types.SimpleNamespace(isChecked=lambda: True)
        widget.logic = _ChipLogic(_FakeCurve())
        widget.step2HintLabel = _Label()
        widget.step3HintLabel = _Label()
        widget.step3Box = _Box()
        widget.globalStatusLabel = _Label()
        widget._effectiveJawKey = lambda: "Lower"  # type: ignore[method-assign]
        widget._observeCurveEdits = lambda _node: None  # type: ignore[method-assign]
        widget.updateGUIFromParameterNode = lambda: None  # type: ignore[method-assign]
        calls: list = []
        widget._runPanoramicGeneration = lambda preview=False: calls.append("pano")  # type: ignore[method-assign]
        widget.onExtractCanalsClicked = lambda: calls.append("canals")  # type: ignore[method-assign]
        CBCTPlannerWidget.onRunFullPipelineClicked(widget)
        self.assertEqual(calls, ["pano", "canals"])
        self.assertIn("fdi_centroids", widget.step2HintLabel.text)
        self.assertIn("Pipeline complete", widget.globalStatusLabel.text)
        self.assertFalse(widget.step3Box.collapsed)


if __name__ == "__main__":
    unittest.main()

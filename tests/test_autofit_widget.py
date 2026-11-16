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
    arc_length_mm = 110.0
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

    def setText(self, text: str) -> None:
        self.text = text


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


if __name__ == "__main__":
    unittest.main()

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "CBCTPlanner"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBCTPlanner import (  # noqa: E402
    CBCTPlannerWidget,
    Q1_TEETH,
    Q2_TEETH,
    Q3_TEETH,
    Q4_TEETH,
)
from CBCTPlannerLib.core.labels import LOWER_JAW_ID  # noqa: E402
from CBCTPlannerLib.core.measurements import (  # noqa: E402
    MeasurementEngine,
    coronal_offset_along_axis,
    reanchor_h_start_to_w0,
)
from CBCTPlannerLib.core_print_render import render_cross_section  # noqa: E402


class TestInlineLabelsAndCustomOverlays(unittest.TestCase):
    def test_inline_line_labels_and_custom_overlays_render(self) -> None:
        u = np.linspace(-10.0, 10.0, 41)
        v = np.linspace(-15.0, 15.0, 61)
        cs = SimpleNamespace(
            u_coords=u,
            v_coords=v,
            img_cs=np.full((len(v), len(u)), 200.0, dtype=np.float32),
            mask_cs=np.zeros((len(v), len(u)), dtype=np.int16),
            crest_uv=(0.0, 5.0),
            basal_uv=(0.0, -8.0),
            width_endpoints_uv={0.0: ((-3.5, 4.0), (3.5, 4.0))},
            canal_centroid_uv=(0.0, -9.5),
            vital_contour_uv=None,
            sinus_floor_uv=None,
            crest_profile_uv=None,
            ridge_axis_angle_deg=0.0,
            is_mandible=True,
            measurements={},
        )
        meas = SimpleNamespace(
            bone_height_mm=13.0,
            crest_width_mm=7.0,
            bone_width_at_depths_mm={0.0: 7.0},
            misch_density_class="D2",
            provenance="label_derived",
            withheld_reason=None,
        )
        custom = [
            {"kind": "line", "p1_uv": (-4.0, 0.0), "p2_uv": (4.0, 0.0), "text": "Check ridge"},
            {"kind": "circle", "center_uv": (2.0, -3.0), "radius_mm": 2.0, "text": "Abnormal?"},
        ]
        rnd = render_cross_section(cs, meas, scale=3, show_footer=False, inline_line_labels=True, custom_overlays=custom)
        self.assertTrue(rnd.available)
        texts = [t for _x, _y, t in rnd.labels]
        self.assertTrue(any(t.startswith("H:") for t in texts))
        self.assertTrue(any(t.startswith("W0 ") for t in texts))
        self.assertIn("IAN", texts)
        self.assertIn("Check ridge", texts)
        self.assertIn("Abnormal?", texts)

    def test_reanchor_h_start_to_w0_preserves_coronal_offset_and_midpoint(self) -> None:
        h_start = (0.0, 6.5)
        h_end = (0.0, -6.0)
        w0_start = (-3.0, 4.5)
        w0_end = (3.0, 4.5)
        offset = coronal_offset_along_axis(h_start, h_end, w0_start, w0_end)
        self.assertAlmostEqual(offset, 2.0, delta=1e-3)

        new_w0_start = (-1.5, 5.5)
        new_w0_end = (4.5, 5.5)
        new_h_start = reanchor_h_start_to_w0(h_start, h_end, new_w0_start, new_w0_end, coronal_offset_mm=offset)
        new_mid = (1.5, 5.5)
        # H ray still passes through new_mid, and signed coronal distance matches `offset`.
        vec_mid_end = (new_mid[0] - h_end[0], new_mid[1] - h_end[1])
        vec_h_mid = (new_h_start[0] - new_mid[0], new_h_start[1] - new_mid[1])
        cross = vec_mid_end[0] * vec_h_mid[1] - vec_mid_end[1] * vec_h_mid[0]
        self.assertAlmostEqual(cross, 0.0, delta=0.1)
        self.assertAlmostEqual(coronal_offset_along_axis(new_h_start, h_end, new_w0_start, new_w0_end), offset, delta=0.05)

    def test_build_preprint_rows_autopopulates_and_persists_overrides(self) -> None:
        class _FakeParamNode:
            def __init__(self) -> None:
                self.params = {}

            def SetParameter(self, k: str, v: str) -> None:
                self.params[k] = v

            def GetParameter(self, k: str) -> str:
                return self.params.get(k, "")

        stub_site = SimpleNamespace(
            bone_height_mm=12.4,
            crest_width_mm=6.8,
            distance_to_vital_structure_mm=None,
            withheld_reason=None,
        )
        stub_engine = SimpleNamespace(measure_implant_site=lambda _fdi: stub_site)
        widget = object.__new__(CBCTPlannerWidget)
        widget.treatmentPlanByFDI = {}
        widget._measurementOverridesByFDI = {}
        widget.annotationStore = None
        widget._parameterNode = _FakeParamNode()
        widget._resolveMeasurementEngine = lambda: stub_engine

        rows, _discrepancies = CBCTPlannerWidget._buildPrePrintReviewRows(widget, [46])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["h_mm"], 12.4)
        self.assertAlmostEqual(rows[0]["w0_mm"], 6.8)

        widget._measurementOverridesByFDI[46] = {
            "h_mm": 13.2,
            "w0_mm": 7.4,
            "h_start_uv": (0.2, 6.0),
            "h_end_uv": (0.2, -7.2),
            "w0_start_uv": (-3.5, 4.5),
            "w0_end_uv": (3.9, 4.5),
            "custom_overlays": [{"kind": "circle", "center_uv": (1.0, -2.0), "radius_mm": 2.0, "text": "Defect"}],
        }
        widget._saveOverridesToParameterNode()
        widget._measurementOverridesByFDI = {}
        widget._loadOverridesFromParameterNode()
        restored = widget._measurementOverridesByFDI.get(46, {})
        self.assertAlmostEqual(restored["h_mm"], 13.2)
        self.assertAlmostEqual(restored["w0_mm"], 7.4)
        self.assertEqual(restored["h_start_uv"], (0.2, 6.0))
        self.assertEqual(restored["custom_overlays"][0]["text"], "Defect")

        jsonl_txt = CBCTPlannerWidget._serialize_corrected_keypoints_jsonl(widget._measurementOverridesByFDI)
        parsed = json.loads(jsonl_txt.strip())
        self.assertEqual(parsed["fdi"], 46)
        self.assertEqual(parsed["h_mm"], 13.2)
        self.assertEqual(parsed["custom_overlays"][0]["text"], "Defect")

    def test_jaw_wise_odontogram_hides_absent_jaw_and_expands_missing_567_span(self) -> None:
        class _Btn:
            def __init__(self) -> None:
                self.visible = True
                self.checked = False
                self.style = ""
                self.tooltip = ""

            def setVisible(self, v: bool) -> None:
                self.visible = bool(v)

            def setChecked(self, v: bool) -> None:
                self.checked = bool(v)

            def setStyleSheet(self, s: str) -> None:
                self.style = str(s)

            def setToolTip(self, t: str) -> None:
                self.tooltip = str(t)

            def blockSignals(self, _v: bool) -> None:
                pass

        class _Combo:
            def __init__(self, current: str) -> None:
                self.currentText = current

            def setCurrentText(self, val: str) -> None:
                self.currentText = str(val)

        # Build segmentation mask with Mandible only (no Maxilla / upper teeth), and lower 5/6/7 (45,46,47) missing.
        mask = np.zeros((20, 20, 20), dtype=np.int16)
        mask[2:8, 2:18, 2:18] = LOWER_JAW_ID
        for fdi in Q3_TEETH + Q4_TEETH:
            if fdi not in (45, 46, 47):
                mask[10:14, 4:8, 4:8] = int(fdi)  # ensure >= 25 voxels

        # Ensure unique non-overlapping counts for present teeth: assign distinct voxels.
        idx = 0
        for fdi in Q3_TEETH + Q4_TEETH:
            if fdi in (45, 46, 47):
                continue
            row = idx % 16
            col = idx // 16
            mask[12:18, 2 + row, 2 + col * 5 : 7 + col * 5] = int(fdi)
            idx += 1

        eng = object.__new__(MeasurementEngine)
        eng.is_dental_segmentator_5class = False
        eng._get_tooth_info = lambda f: None if int(f) in (45, 46, 47) else SimpleNamespace(fdi=str(f))

        analyzed_calls = []
        widget = object.__new__(CBCTPlannerWidget)
        widget.selectedTeeth = {16, 44}  # #16 should be pruned when upper jaw is absent
        widget.activeToothFDI = 16
        widget.treatmentPlanByFDI = {16: {"fdi": 16}}
        widget._presentFdisInScan = None
        widget._parameterNode = None
        widget.toothButtons = {fdi: _Btn() for fdi in Q1_TEETH + Q2_TEETH + Q3_TEETH + Q4_TEETH}
        q_btns = [_Btn(), _Btn(), _Btn(), _Btn()]
        widget._odontogramUpperWidgets = [q_btns[0], q_btns[1]] + [widget.toothButtons[f] for f in Q1_TEETH + Q2_TEETH]
        widget._odontogramLowerWidgets = [q_btns[2], q_btns[3]] + [widget.toothButtons[f] for f in Q3_TEETH + Q4_TEETH]
        widget.archModeCombo = _Combo("Per-Jaw (Lower / Upper)")
        widget.activeJawCombo = _Combo("Upper")
        widget.arcMmSlider = SimpleNamespace(blockSignals=lambda _v: None, setValue=lambda _v: None)
        widget.volumeSelector = SimpleNamespace(currentNode=lambda: object())
        widget.segmentationSelector = SimpleNamespace(currentNode=lambda: object())
        widget.logic = SimpleNamespace(
            _extract_combined_label_mask=lambda _v, _s: (mask, False),
            jump_to_tooth_site=lambda *_a, **_k: 25.0,
            analyze_selected_teeth=lambda fdis, _v, **_k: (
                analyzed_calls.append(list(fdis)) or [{"fdi": int(x), "bone_height_mm": 12.0} for x in fdis]
            ),
            analyze_single_tooth=lambda fdi, _v, **_k: {"fdi": int(fdi)},
        )
        widget._resolveMeasurementEngine = lambda: eng
        widget._refreshPlanTableFromCache = lambda: None
        widget._updateSelectionStatusLabel = lambda: None
        widget._updateActiveSiteClinicalCallout = lambda _f: None
        gallery_calls = []
        widget._renderGalleryItems = lambda **kw: gallery_calls.append(kw)
        widget._currentCursorSettings = lambda **_k: {}

        widget._refreshOdontogramAvailability()
        self.assertFalse(q_btns[0].visible)
        self.assertFalse(q_btns[1].visible)
        self.assertTrue(q_btns[2].visible)
        self.assertTrue(q_btns[3].visible)
        self.assertTrue(all(not widget.toothButtons[f].visible for f in Q1_TEETH + Q2_TEETH))
        self.assertTrue(all(widget.toothButtons[f].visible for f in Q3_TEETH + Q4_TEETH))
        self.assertNotIn(16, widget.selectedTeeth)
        self.assertEqual(widget.activeJawCombo.currentText, "Lower")
        self.assertIn("dashed", widget.toothButtons[46].style)

        # Clicking lower 6 (#46) when 5, 6, 7 (45, 46, 47) are missing must select & measure ALL 3 (Q4 right->left).
        widget.onOdontogramToothClicked(46, True)
        self.assertEqual(analyzed_calls[-1], [47, 46, 45])
        for fdi in (45, 46, 47):
            self.assertIn(fdi, widget.selectedTeeth)
            self.assertIn(fdi, widget.treatmentPlanByFDI)
            self.assertTrue(widget.toothButtons[fdi].checked)
        self.assertEqual(gallery_calls[-1]["fdis_override"], [47, 46, 45])
        self.assertTrue(gallery_calls[-1]["full_gallery"])


if __name__ == "__main__":
    unittest.main()

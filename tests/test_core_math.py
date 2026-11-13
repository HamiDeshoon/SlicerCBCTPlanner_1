"""
Unit tests for SlicerCBCTPlanner (Phases 0, 1, 2, 3, 4 + Lightbox & +-30° Tilt Upgrade).
Runnable outside 3D Slicer using standard Python + NumPy:
    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import CBCTPlanner
from CBCTPlanner import (
    NOT_MEASURED,
    CBCTPlannerLogic,
    format_metric,
)
from CBCTPlannerLib import (
    CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID,
    CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML,
    CrossSectionNavigator,
    PanoramicCurveLogic,
    VRVisualizationLogic,
)
from CBCTPlannerLib.core.arch_detector import (
    DentalArch,
    fit_dental_arch_from_points,
)
from CBCTPlannerLib.core.measurements import MeasurementEngine
from CBCTPlannerLib.core.volume_loader import CBCTVolume


class _MockPiecewiseFunction:
    """Strictly enforces 4-element [x, y, midpoint, sharpness] buffer like vtkPiecewiseFunction."""

    def __init__(self, pts: list[list[float]]) -> None:
        self._pts = [list(p) for p in pts]

    def GetSize(self) -> int:
        return len(self._pts)

    def GetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 4:
            raise TypeError(f"vtkPiecewiseFunction.GetNodeValue expects 4 floats, got {len(val)}")
        for i in range(4):
            val[i] = self._pts[index][i]
        return 1

    def SetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 4:
            raise TypeError(f"vtkPiecewiseFunction.SetNodeValue expects 4 floats, got {len(val)}")
        self._pts[index] = list(val)
        return 1


class _MockColorTransferFunction:
    """Strictly enforces 6-element [x, r, g, b, midpoint, sharpness] buffer like vtkColorTransferFunction."""

    def __init__(self, pts: list[list[float]]) -> None:
        self._pts = [list(p) for p in pts]

    def GetSize(self) -> int:
        return len(self._pts)

    def GetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 6:
            raise TypeError(f"vtkColorTransferFunction.GetNodeValue expects 6 floats, got {len(val)}")
        for i in range(6):
            val[i] = self._pts[index][i]
        return 1

    def SetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 6:
            raise TypeError(f"vtkColorTransferFunction.SetNodeValue expects 6 floats, got {len(val)}")
        self._pts[index] = list(val)
        return 1


class _MockVolumePropertyNode:
    def __init__(self) -> None:
        self._op = _MockPiecewiseFunction([[-1000.0, 0.0, 0.5, 0.0], [500.0, 0.8, 0.5, 0.0]])
        self._col = _MockColorTransferFunction(
            [[-1000.0, 0.0, 0.0, 0.0, 0.5, 0.0], [500.0, 0.9, 0.8, 0.7, 0.5, 0.0]]
        )

    def GetScalarOpacity(self) -> _MockPiecewiseFunction:
        return self._op

    def GetColor(self) -> _MockColorTransferFunction:
        return self._col


class _MockCurveNode:
    def __init__(self, pts: np.ndarray, mtime: int = 1) -> None:
        self._pts = np.asarray(pts, dtype=np.float64)
        self._mtime = mtime

    def GetID(self) -> str:
        return "vtkMRMLMarkupsCurveNode1"

    def GetMTime(self) -> int:
        return self._mtime

    def GetNumberOfControlPoints(self) -> int:
        return len(self._pts)

    def GetNthControlPointPositionWorld(self, idx: int, buf: list[float]) -> None:
        buf[0], buf[1], buf[2] = float(self._pts[idx, 0]), float(self._pts[idx, 1]), float(self._pts[idx, 2])

    def GetCurveWorld(self) -> None:
        return None


class TestSlicerCBCTPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.ras_arch_pts = np.array(
            [
                [28.0, -22.0, 0.0],
                [24.0, -8.0, 0.0],
                [16.0, 8.0, 0.0],
                [8.0, 16.0, 0.0],
                [0.0, 20.0, 0.0],
                [-8.0, 16.0, 0.0],
                [-16.0, 8.0, 0.0],
                [-24.0, -8.0, 0.0],
                [-28.0, -22.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_phase0_zero_optional_deps_and_single_fit_function(self) -> None:
        """Module imports with zero optional deps and uses a single fit_dental_arch_from_points."""
        self.assertIs(
            CBCTPlanner.fit_dental_arch_from_points,
            fit_dental_arch_from_points,
        )
        with patch.dict(sys.modules, {"skimage": None, "nibabel": None, "pydicom": None, "SimpleITK": None}):
            arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
            self.assertGreater(arch.total_arc_length, 80.0)

    def test_phase1_ras_right_to_left_orientation_and_fdi_side_mapping(self) -> None:
        """In Slicer RAS (+X = Patient Right), arch starts at +X and ends at -X; Q1/Q4 map to +X, Q2/Q3 to -X."""
        shuffled = self.ras_arch_pts[::-1].copy()
        fitted = fit_dental_arch_from_points(shuffled, num_control_points=11, ras_convention=True)
        self.assertGreater(fitted[0, 0], 0.0)
        self.assertLess(fitted[-1, 0], 0.0)

        arch = DentalArch(fitted, jaw_type="lower", ras_convention=True)
        logic = CBCTPlannerLogic()

        for right_fdi in (18, 16, 11, 48, 46, 41):
            t_r = logic.fdi_to_normalized_arch_t(right_fdi)
            pt_r = arch.evaluate_at(t_r)
            self.assertLess(t_r, 0.5)
            self.assertGreater(pt_r[0], 0.0)

        for left_fdi in (21, 26, 28, 31, 36, 38):
            t_l = logic.fdi_to_normalized_arch_t(left_fdi)
            pt_l = arch.evaluate_at(t_l)
            self.assertGreater(t_l, 0.5)
            self.assertLess(pt_l[0], 0.0)

    def test_phase1_frame_normal_points_buccal_and_yellow_slice_orientation(self) -> None:
        """Verify N = T x Up points Buccal everywhere and Yellow slice has Buccal-Left / Lingual-Right / Superior-Up."""
        arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
        f_right = arch.get_frame(0.1)
        self.assertGreater(f_right.normal[0], 0.5)

        f_mid = arch.get_frame(0.5)
        self.assertGreater(f_mid.normal[1], 0.8)

        f_left = arch.get_frame(0.9)
        self.assertLess(f_left.normal[0], -0.5)

        for frame in (f_right, f_mid, f_left):
            screen_right, screen_up, slice_normal = CrossSectionNavigator.compute_yellow_slice_axes(
                frame.tangent, frame.normal, frame.binormal
            )
            np.testing.assert_allclose(screen_right, -frame.normal, atol=1e-6)
            np.testing.assert_allclose(screen_up, frame.binormal, atol=1e-6)
            det = float(np.linalg.det(np.column_stack([screen_right, screen_up, slice_normal])))
            self.assertAlmostEqual(det, 1.0, places=5)

    def test_phase1_occlusal_plane_3_points(self) -> None:
        """Verify 3-point occlusal plane normal computation and frame alignment."""
        p1 = [20.0, -10.0, -1.0]
        p2 = [0.0, 20.0, 2.0]
        p3 = [-20.0, -10.0, -1.0]
        normal = CBCTPlannerLogic.compute_occlusal_normal_from_3_points(np.array([p1, p2, p3]))
        self.assertGreater(normal[2], 0.9)
        self.assertLess(normal[1], 0.0)

        arch = DentalArch(self.ras_arch_pts, occlusal_normal=normal, ras_convention=True)
        frame = arch.get_frame(0.5)
        np.testing.assert_allclose(frame.binormal, normal, atol=1e-5)
        self.assertAlmostEqual(float(np.dot(frame.tangent, normal)), 0.0, places=6)

    def test_phase1_vr_shift_transfer_function_4_and_6_tuple_buffers(self) -> None:
        """Verify vtkPiecewiseFunction uses 4 floats, vtkColorTransferFunction uses 6 floats, non-cumulatively."""
        prop = _MockVolumePropertyNode()
        base_op, base_col = VRVisualizationLogic._extract_tf_points(prop)
        self.assertEqual(len(base_op[0]), 4)
        self.assertEqual(len(base_col[0]), 6)

        VRVisualizationLogic.shift_transfer_function(prop, 200.0, base_op, base_col)
        self.assertAlmostEqual(prop.GetScalarOpacity()._pts[1][0], 700.0)
        self.assertAlmostEqual(prop.GetColor()._pts[1][0], 700.0)

        VRVisualizationLogic.shift_transfer_function(prop, 350.0, base_op, base_col)
        self.assertAlmostEqual(prop.GetScalarOpacity()._pts[1][0], 850.0)
        self.assertAlmostEqual(prop.GetColor()._pts[1][0], 850.0)

    def test_phase1_cpr_matrix_handedness_and_pano_x_helpers(self) -> None:
        """Verify CPR direction matrix has det = +1 and pano_x / arc_mm map Patient Right to Screen Left."""
        dir_mat = PanoramicCurveLogic.compute_cpr_direction_matrix(
            self.ras_arch_pts[0], self.ras_arch_pts[-1], occlusal_normal=(0.0, 0.0, 1.0)
        )
        det = float(np.linalg.det(dir_mat[:3, :3]))
        self.assertAlmostEqual(det, 1.0, places=6)

        self.assertEqual(PanoramicCurveLogic.pano_x(0.0), 0.0)
        self.assertEqual(PanoramicCurveLogic.pano_x(95.0), -95.0)
        self.assertEqual(PanoramicCurveLogic.arc_mm(-95.0, max_arc_mm=100.0), 95.0)
        self.assertEqual(PanoramicCurveLogic.arc_mm(-120.0, max_arc_mm=100.0), 100.0)

    def test_phase1_honest_measurements_and_csv_html_export(self) -> None:
        """Unsegmented sites return None / 'not measured'; segmented sites compute real volume metrics."""
        logic = CBCTPlannerLogic()
        unseg_rows = logic.analyze_selected_teeth([46, 16], volumeNode=None, segmentationNode=None)
        for row in unseg_rows:
            self.assertEqual(row["source"], NOT_MEASURED)
            self.assertIsNone(row["height_mm"])
            self.assertIsNone(row["w_crest_mm"])
            self.assertIsNone(row["nerve_clearance_mm"])
            self.assertIsNone(row["hu"])
            self.assertEqual(format_metric(row["height_mm"], "mm"), NOT_MEASURED)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _, csv_path = logic.export_treatment_plan_files(unseg_rows, tmp_path)
            with open(csv_path, encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[0]["height_mm"], NOT_MEASURED)
            self.assertIn("pos_r_mm", reader[0])
            self.assertNotIn("pos_ras", reader[0])

            class _FakeVol:
                def GetName(self) -> str:
                    return "<script>alert('xss')</script>"

            html_path = logic.generate_printable_report(
                unseg_rows, _FakeVol(), tmp_path, capture_screenshots=False
            )
            content = html_path.read_text(encoding="utf-8")
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertIn(NOT_MEASURED, content)

    def test_phase1_measurement_engine_on_synthetic_segmented_cbct(self) -> None:
        """Verify MeasurementEngine computes real physical metrics from a segmented volume."""
        img = np.full((40, 40, 40), -500.0, dtype=np.float32)
        mask = np.zeros((40, 40, 40), dtype=np.uint16)
        mask[5:25, 15:23, 5:35] = 2
        img[5:25, 15:23, 5:35] = 950.0
        mask[7:9, 18:20, 5:35] = 5

        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4, dtype=np.float64),
        )
        arch_pts = np.array(
            [[32.0, 19.0, 24.0], [25.0, 19.0, 24.0], [15.0, 19.0, 24.0], [8.0, 19.0, 24.0]],
            dtype=np.float64,
        )
        arch = DentalArch(arch_pts, jaw_type="lower", ras_convention=True)
        engine = MeasurementEngine(
            vol, mask, arch=arch, arches={"lower": arch}, is_dental_segmentator_5class=True
        )
        meas = engine.measure_implant_site("46")
        self.assertIsNotNone(meas)
        assert meas is not None
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 16.0, delta=1.0)
        self.assertAlmostEqual(meas.mean_density_hu or 0.0, 950.0, delta=10.0)
        self.assertEqual(meas.misch_density_class, "D2")

    def test_phase2_mtime_caching_and_curve_projection(self) -> None:
        """Verify PanoramicCurveLogic caches frames by MTime and projects dragged points to arc_mm."""
        pano_logic = PanoramicCurveLogic()
        mock_curve = _MockCurveNode(self.ras_arch_pts, mtime=42)
        frames1 = pano_logic.get_curve_points_and_frames(mock_curve, num_samples=50)
        frames2 = pano_logic.get_curve_points_and_frames(mock_curve, num_samples=50)
        self.assertIs(frames1, frames2)

        total_len = float(frames1[1][-1])
        arc_mid = pano_logic.project_point_to_curve_arc_mm(mock_curve, [0.0, 20.0, 0.0])
        self.assertAlmostEqual(arc_mid, total_len * 0.5, delta=3.0)

    def test_lightbox_offsets_and_cross_section_tilt_30_deg(self) -> None:
        """Verify 3-slice lightbox (-2, 0, +2 mm) offsets, layout XML, and +-30° cross-section tilt math."""
        self.assertEqual(CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID, 503)
        self.assertIn('singletontag="YellowMinus"', CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)
        self.assertIn('singletontag="Yellow"', CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)
        self.assertIn('singletontag="YellowPlus"', CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)

        # 1. Lightbox offsets (-2, 0, +2 mm) and boundary clamping
        s_m, s_0, s_p = CrossSectionNavigator.compute_lightbox_offsets(50.0, 100.0, step_mm=2.0)
        self.assertEqual((s_m, s_0, s_p), (48.0, 50.0, 52.0))
        s_m_edge, s_0_edge, s_p_edge = CrossSectionNavigator.compute_lightbox_offsets(1.0, 100.0, step_mm=2.0)
        self.assertEqual((s_m_edge, s_0_edge, s_p_edge), (0.0, 1.0, 3.0))

        # 2. Clamping beyond +-30°
        self.assertEqual(CrossSectionNavigator.clamp_tilt_deg(45.0), 30.0)
        self.assertEqual(CrossSectionNavigator.clamp_tilt_deg(-60.0), -30.0)

        # 3. Frame orthogonality & det = +1 under +-30° tilt
        arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
        f_mid = arch.get_frame(0.5)
        for deg in (-30.0, -15.0, 0.0, 15.0, 30.0):
            t_tilt, n_tilt, b_tilt = CrossSectionNavigator.apply_cross_section_tilt(
                f_mid.tangent, f_mid.normal, f_mid.binormal, tilt_deg=deg
            )
            # N (Buccal-Lingual) must remain invariant
            np.testing.assert_allclose(n_tilt, f_mid.normal, atol=1e-6)
            # Orthonormal check
            self.assertAlmostEqual(float(np.dot(t_tilt, b_tilt)), 0.0, places=6)
            self.assertAlmostEqual(float(np.dot(t_tilt, n_tilt)), 0.0, places=6)
            self.assertAlmostEqual(float(np.dot(b_tilt, n_tilt)), 0.0, places=6)
            # Angle between B_tilt and B must equal |deg|
            cos_angle = float(np.clip(np.dot(b_tilt, f_mid.binormal), -1.0, 1.0))
            self.assertAlmostEqual(math.degrees(math.acos(cos_angle)), abs(deg), places=4)
            # Yellow slice right-handedness (det = +1)
            sr, su, sn = CrossSectionNavigator.compute_yellow_slice_axes(t_tilt, n_tilt, b_tilt)
            det = float(np.linalg.det(np.column_stack([sr, su, sn])))
            self.assertAlmostEqual(det, 1.0, places=5)

        # 4. Panoramic tilted line endpoints pivot around center X = pano_x(arc_mm)
        b_pt, t_pt = CrossSectionNavigator.compute_pano_cursor_endpoints(40.0, pano_height_mm=50.0, tilt_deg=20.0)
        self.assertAlmostEqual(0.5 * (b_pt[0] + t_pt[0]), PanoramicCurveLogic.pano_x(40.0), places=6)
        self.assertNotAlmostEqual(b_pt[0], t_pt[0])


if __name__ == "__main__":
    unittest.main()

"""v3.3 Regression Tests: Clinical Cross-Section Geometry + Complete ADT Package (`DOCShapeAXI`, `CLIC`, `ASO`, `ALI`)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLANNER_DIR = ROOT / "CBCTPlanner"
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))

from CBCTPlannerLib.adt import (  # noqa: E402
    AutomatedDentalToolsEngine,
    extract_jaw_arch_anchor_bounds,
    kabsch_rigid_alignment,
)
from CBCTPlannerLib.core.arch_detector import DentalArch, fit_arch_from_label_mask  # noqa: E402
from CBCTPlannerLib.core.labels import LOWER_JAW_ID, RIGHT_IAN_ID, UPPER_JAW_ID  # noqa: E402
from CBCTPlannerLib.core.measurements import CrossSectionData, MeasurementEngine  # noqa: E402
from CBCTPlannerLib.core.orientation import landmark_guided_aso_frame  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402
from CBCTPlannerLib.core_print_render import render_cross_section, select_periodic_depth_key  # noqa: E402


class TestClinicalGeometryV33(unittest.TestCase):
    """Verify strict T-junction, unclipped full-jaw framing, endosteal H end, W0 smoothness, and connected labels."""

    @staticmethod
    def _vol(img: np.ndarray, spacing: float = 1.0) -> CBCTVolume:
        aff = np.diag([spacing, spacing, spacing, 1.0])
        return CBCTVolume(img, (spacing, spacing, spacing), (0.0, 0.0, 0.0), np.eye(3), aff)

    def test_strict_t_junction_between_h_and_w0_never_crosses(self) -> None:
        shape = (50, 50, 50)
        mask = np.zeros(shape, dtype=np.int32)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask[10:29, 21:29, 10:40] = LOWER_JAW_ID
        mask[14:17, 24:26, 10:40] = RIGHT_IAN_ID
        img[mask > 0] = 900.0
        arch = DentalArch(
            np.array(
                [[36.0, 25.0, 28.0], [28.0, 25.0, 28.0], [20.0, 25.0, 28.0], [14.0, 25.0, 28.0]],
                dtype=np.float64,
            ),
            jaw_type="lower",
            ras_convention=True,
        )
        eng = MeasurementEngine(
            self._vol(img), mask, arch=arch, arches={"lower": arch}, is_dental_segmentator_5class=False
        )
        meas, cs = eng.measure_at_arc_mm("lower", 11.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert cs is not None and cs.crest_uv is not None and cs.basal_uv is not None
        self.assertIn(0.0, cs.width_endpoints_uv)
        w0_l, w0_r = cs.width_endpoints_uv[0.0]
        w0_mid = (0.5 * (w0_l[0] + w0_r[0]), 0.5 * (w0_l[1] + w0_r[1]))
        # T-junction: H starts strictly AT W0's midpoint with zero coronal overhang past W0
        self.assertAlmostEqual(float(cs.crest_uv[0]), float(w0_mid[0]), delta=1e-4)
        self.assertAlmostEqual(float(cs.crest_uv[1]), float(w0_mid[1]), delta=1e-4)
        self.assertLess(float(cs.basal_uv[1]), float(cs.crest_uv[1]))

    def test_tall_mandible_and_maxilla_never_clipped_and_endosteal_h_stops_inside_cortex(self) -> None:
        # Build an 80x80x80 volume with a 36 mm tall mandible extending from Z=10 to Z=46,
        # while tooth/arch plane sits coronally at Z=48 -> basal Z=10 is -38 mm below P!
        shape = (80, 80, 80)
        mask = np.zeros(shape, dtype=np.int32)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask[10:47, 36:45, 20:60] = LOWER_JAW_ID
        # Trabecular marrow core (300 HU) surrounded by 2.5 mm thick dense cortical plates (1250 HU)
        img[mask == LOWER_JAW_ID] = 1250.0
        img[13:44, 38:43, 22:58] = 320.0
        arch = DentalArch(
            np.array(
                [[56.0, 40.0, 48.0], [45.0, 40.0, 48.0], [35.0, 40.0, 48.0], [24.0, 40.0, 48.0]],
                dtype=np.float64,
            ),
            jaw_type="lower",
            ras_convention=True,
        )
        eng = MeasurementEngine(
            self._vol(img), mask, arch=arch, arches={"lower": arch}, is_dental_segmentator_5class=False
        )
        meas, cs = eng.measure_at_arc_mm("lower", 16.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert cs is not None and cs.basal_uv is not None
        # Entire mandible (down to -38 mm relative to P) must fit comfortably inside v_coords (no clipping!)
        self.assertLess(float(cs.v_coords[0]), -39.0)
        # No bone voxels touch bottom row of resampled mask_cs
        self.assertFalse(bool(np.any(cs.mask_cs[0, :] == LOWER_JAW_ID)))
        # Endosteal internal border inset should reflect the ~2.5-3.0 mm cortex (>= 1.5 mm)
        self.assertGreater(float(meas.bone_height_mm or 0.0), 30.0)
        self.assertLess(float(meas.bone_height_mm or 0.0), 36.5)

    def test_maxilla_top_never_clipped(self) -> None:
        shape = (80, 80, 80)
        mask = np.zeros(shape, dtype=np.int32)
        img = np.full(shape, -500.0, dtype=np.float32)
        # Tall maxilla from Z=32 up to Z=68 while arch plane sits at Z=30 (+38 mm above P)
        mask[32:69, 36:45, 20:60] = UPPER_JAW_ID
        img[mask == UPPER_JAW_ID] = 850.0
        arch = DentalArch(
            np.array(
                [[56.0, 40.0, 30.0], [45.0, 40.0, 30.0], [35.0, 40.0, 30.0], [24.0, 40.0, 30.0]],
                dtype=np.float64,
            ),
            jaw_type="upper",
            ras_convention=True,
        )
        eng = MeasurementEngine(
            self._vol(img), mask, arch=arch, arches={"upper": arch}, is_dental_segmentator_5class=False
        )
        meas, cs = eng.measure_at_arc_mm("upper", 16.0, fdi_hint="16")
        self.assertIsNotNone(meas)
        assert cs is not None
        self.assertGreater(float(cs.v_coords[-1]), 39.0)
        self.assertFalse(bool(np.any(cs.mask_cs[-1, :] == UPPER_JAW_ID)))

    def test_stabilize_entry_points_smooths_adjacent_w0_angles_on_similar_slices(self) -> None:
        u = np.linspace(-15.0, 15.0, 121)
        v = np.linspace(-25.0, 25.0, 201)
        img = np.zeros((201, 121), dtype=np.float32)
        mask = np.zeros((201, 121), dtype=np.int16)
        mask[60:140, 40:80] = 1
        img[60:140, 40:80] = 900.0
        frame = (
            np.zeros(3, dtype=np.float64),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        items = []
        # 5 similar jaw slices where slice 2 has a 28-deg angular glitch while neighbors are near 0 deg
        for idx, tilt_deg in enumerate((0.0, 1.0, 28.0, -1.0, 0.0)):
            rad = math.radians(tilt_deg)
            dx, dy = 4.0 * math.cos(rad), 4.0 * math.sin(rad)
            w0_l, w0_r = (-dx, 9.5 - dy), (dx, 9.5 + dy)
            cs = CrossSectionData(
                u_coords=u,
                v_coords=v,
                mask_cs=mask.copy(),
                img_cs=img.copy(),
                frame=frame,
                crest_v=9.5,
                width_segments={0.0: (w0_l[0], w0_r[0])},
                canal_centroid_uv=None,
                crest_uv=(0.0, 9.5),
                basal_uv=(0.0, -8.5),
                width_endpoints_uv={0.0: (w0_l, w0_r)},
                ridge_axis_angle_deg=tilt_deg,
                is_mandible=True,
            )
            items.append((None, cs))

        eng = MeasurementEngine(
            self._vol(np.zeros((20, 20, 20), dtype=np.float32)),
            np.zeros((20, 20, 20), dtype=np.int32),
        )
        eng.stabilize_entry_points(items, max_angle_jump_deg=8.0)

        angles = []
        for _m, cs in items:
            p0, p1 = cs.width_endpoints_uv[0.0]
            angles.append(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))
        for i in range(len(angles) - 1):
            self.assertLessEqual(abs(angles[i + 1] - angles[i]), 8.05)

    def test_periodic_single_w3_w5_and_connected_ian_w0_leaders(self) -> None:
        self.assertEqual(select_periodic_depth_key(0, 15, stride=4), (3.0,))
        self.assertEqual(select_periodic_depth_key(1, 15, is_center=False, stride=4), ())
        self.assertEqual(select_periodic_depth_key(4, 15, is_center=False, stride=4), (5.0,))
        self.assertEqual(select_periodic_depth_key(8, 15, is_center=False, stride=4), (3.0,))
        self.assertEqual(select_periodic_depth_key(7, 15, is_center=True, stride=4), (3.0,))

        u = np.linspace(-15.0, 15.0, 121)
        v = np.linspace(-25.0, 25.0, 201)
        cs = CrossSectionData(
            u_coords=u,
            v_coords=v,
            mask_cs=np.zeros((201, 121), dtype=np.int16),
            img_cs=np.full((201, 121), -400.0, dtype=np.float32),
            frame=None,
            crest_v=6.0,
            width_segments={0.0: (-3.5, 3.5), 3.0: (-4.0, 4.0), 5.0: (-4.5, 4.5)},
            canal_centroid_uv=(0.0, -6.0),
            crest_uv=(0.0, 6.0),
            basal_uv=(0.0, -4.5),
            width_endpoints_uv={
                0.0: ((-3.5, 6.0), (3.5, 6.0)),
                3.0: ((-4.0, 3.0), (4.0, 3.0)),
                5.0: ((-4.5, 1.0), (4.5, 1.0)),
            },
            is_mandible=True,
        )
        rnd = render_cross_section(
            cs,
            None,
            scale=3,
            show_footer=False,
            inline_line_labels=True,
            depth_key_depths=(5.0,),
        )
        self.assertTrue(rnd.available)
        texts = [t for _x, _y, t in rnd.labels]
        self.assertIn("IAN", texts)
        self.assertTrue(any(t.startswith("W0 ") for t in texts))
        self.assertIn("W3/W5 key", texts)


class TestADTCompletePackageV33(unittest.TestCase):
    """Verify upgraded DOCShapeAXI, CLIC, ASO, ALI, and VFACE package algorithms."""

    def test_doc_shapeaxi_airway_condyle_cleft_and_vtk_saliency_export(self) -> None:
        shape = (40, 48, 48)
        hu = np.full(shape, 100.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int16)

        # Mandible with condyle head + Maxilla + constricted internal airway tube
        mask[5:32, 16:32, 12:36] = LOWER_JAW_ID
        mask[22:36, 16:32, 14:34] = UPPER_JAW_ID
        # Airway cavity inside central ROI (constricts sharply at z=18..21)
        mask[:, :, :] = mask
        hu[5:35, 20:28, 20:28] = -850.0
        hu[17:22, 22:26, 22:26] = 100.0  # narrow airway stenosis to 4x4 mm^2
        hu[17:22, 23:25, 23:25] = -850.0
        mask[5:35, 20:28, 20:28] = 0

        aff = np.diag([1.0, 1.0, 1.0, 1.0])
        vol = CBCTVolume(hu, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3), aff)
        eng = AutomatedDentalToolsEngine()

        with tempfile.TemporaryDirectory() as tmp:
            air_vtk = Path(tmp) / "airway_explain.vtk"
            air_res = eng.run_doc_shapeaxi(
                data_type="Airway",
                task="severity",
                volume=vol,
                mask=mask,
                out_vtk_path=air_vtk,
            )
            self.assertEqual(air_res.data_type, "Airway")
            self.assertGreater(air_res.vertex_count, 50)
            self.assertTrue(air_vtk.exists())
            vtk_txt = air_vtk.read_text(encoding="utf-8", errors="ignore")
            self.assertIn("ShapeAXI_Saliency", vtk_txt)

            cond_res = eng.run_doc_shapeaxi(
                data_type="Condyle",
                task="severity",
                volume=vol,
                mask=mask,
                out_vtk_path=Path(tmp) / "condyle_explain.vtk",
            )
            self.assertIn(cond_res.predicted_class, (0, 1, 2, 3))
            self.assertGreater(cond_res.surface_area_mm2, 10.0)

            cleft_res = eng.run_doc_shapeaxi(
                data_type="Cleft",
                task="severity",
                volume=vol,
                mask=mask,
            )
            self.assertIn(cleft_res.predicted_class, (0, 1, 2, 3))

    def test_upgraded_aso_kabsch_outlier_filter_and_clic_labelmap_and_landmark_arch(self) -> None:
        shape = (44, 60, 60)
        mask = np.zeros(shape, dtype=np.int16)
        hu = np.full(shape, 400.0, dtype=np.float32)
        mask[6:22, 14:46, 10:50] = LOWER_JAW_ID
        mask[24:38, 14:46, 12:48] = UPPER_JAW_ID
        mask[20:26, 16:22, 26:30] = 11
        mask[20:26, 16:22, 31:35] = 21
        mask[28:35, 34:41, 20:25] = (
            13  # buccal / high impacted right canine (Buccal shifted anterior/outward)
        )
        mask[20:26, 18:24, 36:40] = 23

        aff = np.diag([0.5, 0.5, 0.5, 1.0])
        aff[:3, 3] = [-15.0, -15.0, -11.0]
        vol = CBCTVolume(hu, (0.5, 0.5, 0.5), (-15.0, -15.0, -11.0), np.eye(3), aff)
        eng = AutomatedDentalToolsEngine()

        lms = eng.identify_ali_landmarks(vol, mask=mask, is_5class=False)
        bounds = extract_jaw_arch_anchor_bounds(lms, "lower")
        self.assertIn("posterior_min_y", bounds)

        # Build rotated reference landmarks (+15 deg yaw) + inject 1 corrupted outlier landmark (`Co_L`)
        theta = math.radians(15.0)
        Rz = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        ref_lms = {}
        for key in ("Me", "Pg", "Go_R", "Go_L", "Co_R", "Co_L", "ANS", "PNS"):
            if key in lms:
                p = np.asarray(lms[key].position_ras, dtype=np.float64)
                rp = Rz @ p + np.array([2.0, -1.0, 0.5])
                if key == "Co_L":
                    rp += np.array([25.0, 30.0, -20.0])  # corrupted outlier!
                ref_lms[key] = rp.tolist()

        aso = eng.compute_aso_standardized_orientation(
            vol,
            mask=mask,
            is_5class=False,
            landmarks=lms,
            reference_landmarks_ras=ref_lms,
            use_icp_refinement=False,
        )
        self.assertIn("Co_L", aso.rejected_landmarks)
        self.assertEqual(aso.matrix_4x4_ras.shape, (4, 4))

        # Check Kabsch directly recovers exact SO(3)
        pts_a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pts_b = (Rz @ pts_a.T).T + np.array([3.0, -2.0, 1.0])
        m_fit = kabsch_rigid_alignment(pts_a, pts_b)
        np.testing.assert_allclose(m_fit[:3, :3], Rz, atol=1e-5)

        # Upgraded CLI-C builds 3D {1, 2, 3} labelmap + cortical clearances
        clic_items = eng.analyze_impacted_teeth_clic(vol, mask, is_5class=False, build_labelmap=True)
        self.assertTrue(any(it.fdi == "13" for it in clic_items))
        self.assertIsNotNone(eng.last_clic_labelmap)
        assert eng.last_clic_labelmap is not None
        self.assertTrue(np.any(eng.last_clic_labelmap > 0))

        # Landmark-guided orientation helper in core/orientation.py
        guide = landmark_guided_aso_frame(mask, aff, is_5class=False)
        self.assertIsNotNone(guide)

        # Verify 5-class mandible arch fitting uses label 2 (mandible) cleanly without maxilla (label 1)
        m5 = np.zeros((30, 50, 50), dtype=np.int16)
        # Horseshoe mandible in label 2 only (no teeth, 5-class)
        zz, yy, xx = np.indices((30, 50, 50))
        rad = np.sqrt((xx - 25.0) ** 2 + (yy - 10.0) ** 2)
        m5[(zz >= 5) & (zz <= 14) & (rad >= 14.0) & (rad <= 20.0) & (yy >= 10.0)] = 2
        m5[(zz >= 20) & (zz <= 27) & (rad >= 13.0) & (rad <= 19.0) & (yy >= 10.0)] = 1  # maxilla
        fit5 = fit_arch_from_label_mask(m5, np.eye(4), jaw="lower", is_5class=True)
        self.assertIsNotNone(fit5)
        assert fit5 is not None
        self.assertEqual(fit5.method, "jawbone_crest")
        self.assertLess(float(np.mean(fit5.points_ras[:, 2])), 16.0)


if __name__ == "__main__":
    unittest.main()

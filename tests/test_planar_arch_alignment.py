"""Unit tests for roll-free sagittal-pitched planar dental arch alignment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.core.arch_detector import (  # noqa: E402
    ArchFitResult,
    DentalArch,
    JAW_TOOTH_ORDER,
    align_arch_to_sagittal_plane,
    fit_arch_from_label_mask,
    fit_dental_arch_from_points,
)
from CBCTPlannerLib.core.labels import (  # noqa: E402
    LOWER_JAW_ID,
    UPPER_JAW_ID,
)


def _max_sagittal_plane_deviation_mm(points_ras: np.ndarray, pitch_deg: float) -> float:
    """Measure max residual |z - (z0 + tan(pitch) * (y - y0))| across points."""
    pts = np.asarray(points_ras, dtype=np.float64)
    y0 = float(np.median(pts[:, 1]))
    slope = math.tan(math.radians(float(pitch_deg)))
    z0 = float(np.median(pts[:, 2] - slope * (pts[:, 1] - y0)))
    pred = z0 + slope * (pts[:, 1] - y0)
    return float(np.max(np.abs(pts[:, 2] - pred)))


def _horseshoe_xy(n: int = 16, rx: float = 30.0, ry: float = 24.0) -> np.ndarray:
    """Right-to-left lower/upper horseshoe polyline centered at (0, 0) in XY."""
    theta = np.linspace(0.0, np.pi, n)
    return np.column_stack([rx * np.cos(theta), ry * np.sin(theta)])


def _stamp_world_sphere(
    mask: np.ndarray,
    inv_affine: np.ndarray,
    center_ras: np.ndarray,
    radius_mm: float,
    label_id: int,
) -> None:
    """Stamp a spherical ROI specified in world RAS mm into a (Z, Y, X) mask."""
    ijk = (inv_affine @ np.append(center_ras, 1.0))[:3]
    cx, cy, cz = float(ijk[0]), float(ijk[1]), float(ijk[2])
    r = float(radius_mm)
    x0, x1 = max(0, int(math.floor(cx - r))), min(mask.shape[2], int(math.ceil(cx + r)) + 1)
    y0, y1 = max(0, int(math.floor(cy - r))), min(mask.shape[1], int(math.ceil(cy + r)) + 1)
    z0, z1 = max(0, int(math.floor(cz - r))), min(mask.shape[0], int(math.ceil(cz + r)) + 1)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r**2
    mask[z0:z1, y0:y1, x0:x1][inside] = int(label_id)


class TestPlanarSagittalArchAlignment(unittest.TestCase):
    def test_zero_lateral_roll_and_exact_planarity(self) -> None:
        xy = _horseshoe_xy(21)
        z_rolled = -10.0 + 0.30 * xy[:, 0] + math.tan(math.radians(7.0)) * xy[:, 1]
        pts = np.column_stack([xy[:, 0], xy[:, 1], z_rolled])

        aligned, pitch_deg, plane = align_arch_to_sagittal_plane(pts)
        self.assertLess(abs(pitch_deg), 15.0)
        self.assertLess(_max_sagittal_plane_deviation_mm(aligned, pitch_deg), 1e-6)
        # Same Y symmetric left/right X points must share identical Z (dz/dx == 0).
        self.assertLess(abs(float(aligned[0, 2] - aligned[-1, 2])), 1e-6)
        self.assertLess(abs(float(aligned[5, 2] - aligned[-6, 2])), 1e-6)
        # Projecting new points onto explicit plane_params must remain strictly exact.
        reproj, rp_deg, rp_plane = align_arch_to_sagittal_plane(pts, plane_params=plane)
        self.assertAlmostEqual(rp_deg, pitch_deg, places=6)
        self.assertEqual(rp_plane, plane)
        np.testing.assert_allclose(reproj, aligned, atol=1e-9)

    def test_theil_sen_recovers_pitch_despite_outlier_and_clamps_extreme_slopes(self) -> None:
        xy = _horseshoe_xy(15)
        slope_10 = math.tan(math.radians(10.0))
        z_10 = 5.0 + slope_10 * (xy[:, 1] - float(np.median(xy[:, 1])))
        # Inject an ectopic/outlier spike on one canine point.
        z_10[4] += 22.0
        pts_10 = np.column_stack([xy[:, 0], xy[:, 1], z_10])

        aligned, pitch_deg, _ = align_arch_to_sagittal_plane(pts_10)
        self.assertAlmostEqual(pitch_deg, 10.0, delta=0.5)
        self.assertLess(_max_sagittal_plane_deviation_mm(aligned, pitch_deg), 1e-6)

        # Extreme 35-deg tilt must clamp cleanly to +/- max_pitch_deg (15 deg default).
        z_steep = math.tan(math.radians(35.0)) * xy[:, 1]
        pts_steep = np.column_stack([xy[:, 0], xy[:, 1], z_steep])
        aligned_s, pitch_s, _ = align_arch_to_sagittal_plane(pts_steep, max_pitch_deg=15.0)
        self.assertAlmostEqual(pitch_s, 15.0, places=4)
        self.assertLess(_max_sagittal_plane_deviation_mm(aligned_s, pitch_s), 1e-6)

        # Small N or narrow Y span falls back to 0.0 deg.
        pts_few = pts_10[:3]
        aligned_f, pitch_f, _ = align_arch_to_sagittal_plane(pts_few)
        self.assertEqual(pitch_f, 0.0)
        self.assertLess(_max_sagittal_plane_deviation_mm(aligned_f, 0.0), 1e-6)

    def test_fit_arch_fdi_centroids_resilient_to_ectopic_canine_43(self) -> None:
        shape = (80, 104, 104)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, 3] = (-shape[2] / 2.0, -shape[1] / 2.0, -shape[0] / 2.0)
        inv_aff = np.linalg.inv(affine)
        mask = np.zeros(shape, dtype=np.int16)

        order = JAW_TOOTH_ORDER["lower"][1:-1]  # 14 teeth: 47..37
        xy = _horseshoe_xy(len(order), rx=30.0, ry=24.0)
        slope = math.tan(math.radians(10.0))
        y_med = float(np.median(xy[:, 1]))

        for idx, fdi in enumerate(order):
            x, y = float(xy[idx, 0]), float(xy[idx, 1])
            z = -4.0 + slope * (y - y_med)
            if fdi == "43":
                # Ectopic/impacted #43 shifted lingually (-12 mm toward origin) and coronally (+15 mm).
                r_xy = max(1e-6, math.hypot(x, y))
                x -= 12.0 * (x / r_xy)
                y -= 12.0 * (y / r_xy)
                z += 15.0
            _stamp_world_sphere(mask, inv_aff, np.array([x, y, z]), 2.8, int(fdi))
            _stamp_world_sphere(mask, inv_aff, np.array([x, y, z - 6.0]), 3.0, LOWER_JAW_ID)

        # Posterior rising ramus bone extensions that must NOT drag extended arch out of plane.
        for side_sign in (-1.0, 1.0):
            for step in np.linspace(1.0, 14.0, 6):
                _stamp_world_sphere(
                    mask,
                    inv_aff,
                    np.array(
                        [side_sign * (31.0 + 0.2 * step), -step, -4.0 + slope * (-step - y_med) + 0.9 * step]
                    ),
                    3.2,
                    LOWER_JAW_ID,
                )

        res = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False, end_extension_mm=12.0)
        self.assertIsNotNone(res)
        assert isinstance(res, ArchFitResult)
        self.assertTrue(res.accepted, res.reason)
        self.assertEqual(res.method, "fdi_centroids")
        self.assertAlmostEqual(res.pitch_deg, 10.0, delta=2.0)
        self.assertLessEqual(abs(res.pitch_deg), 15.0)
        self.assertLess(_max_sagittal_plane_deviation_mm(res.points_ras, res.pitch_deg), 1e-6)
        self.assertLess(abs(float(res.points_ras[0, 2] - res.points_ras[-1, 2])), 0.5)

    def test_fit_arch_principal_curve_jawbone_crest_and_shared_are_planar(self) -> None:
        shape = (80, 104, 104)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, 3] = (-shape[2] / 2.0, -shape[1] / 2.0, -shape[0] / 2.0)
        inv_aff = np.linalg.inv(affine)

        # 1) 5-class principal_curve tilted -7 deg
        mask_5c = np.zeros(shape, dtype=np.int16)
        xy = _horseshoe_xy(24, rx=30.0, ry=24.0)
        y_med = float(np.median(xy[:, 1]))
        slope_neg7 = math.tan(math.radians(-7.0))
        for x, y in xy:
            z = -6.0 + slope_neg7 * (y - y_med)
            _stamp_world_sphere(mask_5c, inv_aff, np.array([x, y, z]), 2.6, 4)
        res_pc = fit_arch_from_label_mask(mask_5c, affine, jaw="lower", is_5class=True, end_extension_mm=10.0)
        self.assertIsNotNone(res_pc)
        assert res_pc is not None
        self.assertTrue(res_pc.accepted, res_pc.reason)
        self.assertEqual(res_pc.method, "principal_curve")
        self.assertAlmostEqual(res_pc.pitch_deg, -7.0, delta=2.5)
        self.assertLess(_max_sagittal_plane_deviation_mm(res_pc.points_ras, res_pc.pitch_deg), 1e-6)

        # 2) Edentulous jawbone_crest with tall posterior ramus jump
        mask_ed = np.zeros(shape, dtype=np.int16)
        for x, y in _horseshoe_xy(28, rx=30.0, ry=24.0):
            z_base = -8.0 + 0.08 * y
            if y < 5.0:
                z_base += 1.2 * (5.0 - y)
            _stamp_world_sphere(mask_ed, inv_aff, np.array([x, y, z_base]), 3.5, LOWER_JAW_ID)
        res_ed = fit_arch_from_label_mask(
            mask_ed, affine, jaw="lower", is_5class=False, end_extension_mm=15.0
        )
        self.assertIsNotNone(res_ed)
        assert res_ed is not None
        self.assertTrue(res_ed.accepted, res_ed.reason)
        self.assertEqual(res_ed.method, "jawbone_crest")
        self.assertLessEqual(abs(res_ed.pitch_deg), 15.0)
        self.assertLess(_max_sagittal_plane_deviation_mm(res_ed.points_ras, res_ed.pitch_deg), 1e-6)

        # 3) Shared dual jaw with independent sagittal pitches
        mask_sh = np.zeros(shape, dtype=np.int16)
        for idx, (x, y) in enumerate(_horseshoe_xy(16, rx=30.0, ry=24.0)):
            z_low = -10.0 + math.tan(math.radians(8.0)) * (y - y_med)
            z_up = 10.0 + math.tan(math.radians(4.0)) * (y - y_med)
            f_low = int(JAW_TOOTH_ORDER["lower"][idx])
            f_up = int(JAW_TOOTH_ORDER["upper"][idx])
            _stamp_world_sphere(mask_sh, inv_aff, np.array([x, y, z_low]), 2.6, f_low)
            _stamp_world_sphere(mask_sh, inv_aff, np.array([x, y, z_up]), 2.6, f_up)
            _stamp_world_sphere(mask_sh, inv_aff, np.array([x, y, z_low - 5.0]), 2.8, LOWER_JAW_ID)
            _stamp_world_sphere(mask_sh, inv_aff, np.array([x, y, z_up + 5.0]), 2.8, UPPER_JAW_ID)
        res_sh = fit_arch_from_label_mask(
            mask_sh, affine, jaw="shared", is_5class=False, end_extension_mm=10.0
        )
        self.assertIsNotNone(res_sh)
        assert res_sh is not None
        self.assertTrue(res_sh.accepted, res_sh.reason)
        self.assertTrue(res_sh.method.startswith("shared("), res_sh.method)
        self.assertLessEqual(abs(res_sh.pitch_deg), 15.0)
        self.assertLess(_max_sagittal_plane_deviation_mm(res_sh.points_ras, res_sh.pitch_deg), 1e-6)

    def test_fit_points_from_teeth_centroids_and_frame_continuity(self) -> None:
        xy = _horseshoe_xy(14, rx=30.0, ry=24.0)
        y_med = float(np.median(xy[:, 1]))
        slope = math.tan(math.radians(10.0))
        z_vals = slope * (xy[:, 1] - y_med) + 0.18 * xy[:, 0]
        pts = np.column_stack([xy[:, 0], xy[:, 1], z_vals])

        ctrl = fit_dental_arch_from_points(pts, jaw_type="lower", num_control_points=25)
        self.assertEqual(ctrl.shape, (25, 3))
        _, p_fit, _ = align_arch_to_sagittal_plane(ctrl)
        self.assertLess(_max_sagittal_plane_deviation_mm(ctrl, p_fit), 1e-6)

        fdis = JAW_TOOTH_ORDER["lower"][1:-1]
        arch = DentalArch.from_teeth_centroids(
            [(fdis[i], pts[i]) for i in range(len(fdis))], jaw_type="lower"
        )
        _, p_arch, _ = align_arch_to_sagittal_plane(arch.control_points)
        self.assertLess(_max_sagittal_plane_deviation_mm(arch.control_points, p_arch), 1e-6)

        frames = arch.sample_frames_uniform(step_mm=2.0)
        self.assertGreater(len(frames), 10)
        for prev_f, next_f in zip(frames[:-1], frames[1:]):
            self.assertGreater(float(np.dot(prev_f.tangent, next_f.tangent)), 0.90)
            self.assertGreater(float(np.dot(prev_f.normal, next_f.normal)), 0.90)


if __name__ == "__main__":
    unittest.main()

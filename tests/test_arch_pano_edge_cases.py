"""Synthetic, anisotropic, tilted, partial-arch, and panoramic edge-clamp tests (Phases 4 & 5)."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.PanoramicCurveLogic import PanoramicCurveLogic  # noqa: E402
from CBCTPlannerLib.core.arch_detector import (  # noqa: E402
    DentalArch,
    compute_parallel_transport_frames,
    fit_arch_from_label_mask,
    fit_dental_arch_from_points,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


class ParallelTransportArchTests(unittest.TestCase):
    def test_parallel_transport_prevents_hemi_arch_and_scurve_frame_flips(self) -> None:
        # Asymmetric right+anterior hemi-arch where chord_mid sits off-center and could flip Frenet normals
        theta = np.linspace(0.05, 0.72 * np.pi, 40)
        pts = np.column_stack(
            [
                28.0 * np.cos(theta),
                32.0 * np.sin(theta),
                12.0 + 3.0 * np.sin(2.0 * theta),  # Curve of Spee / wave in Z
            ]
        )
        arch = DentalArch(pts, jaw_type="lower", smoothing=0.0, ras_convention=True)
        normals = []
        binormals = []
        for s in np.linspace(0.0, arch.total_arc_length, 60):
            f = arch.get_frame_at_mm(float(s))
            self.assertAlmostEqual(float(np.linalg.norm(f.tangent)), 1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(f.normal)), 1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(f.binormal)), 1.0, places=5)
            self.assertLess(abs(float(np.dot(f.tangent, f.normal))), 1e-6)
            self.assertLess(abs(float(np.dot(f.normal, f.binormal))), 1e-6)
            disp_x, disp_y, disp_z = PanoramicCurveLogic.compute_cpr_displacement_frame(
                f.tangent, f.normal, f.binormal
            )
            self.assertGreater(float(np.linalg.det(np.column_stack([disp_x, disp_y, disp_z]))), 0.99)
            normals.append(f.normal)
            binormals.append(f.binormal)

        # Adjacent frames must maintain strict positive directional continuity (no 180 deg flips)
        for i in range(1, len(normals)):
            self.assertGreater(float(np.dot(normals[i], normals[i - 1])), 0.75)
            self.assertGreater(float(np.dot(binormals[i], binormals[i - 1])), 0.75)

    def test_sparse_arch_fdi_outlier_rejection_and_tilted_normal(self) -> None:
        clean_pts = np.array(
            [
                [26.0, 10.0, 15.0],
                [20.0, 24.0, 15.0],
                [8.0, 32.0, 15.0],
                [-8.0, 32.0, 15.0],
                [-20.0, 24.0, 15.0],
                [-26.0, 10.0, 15.0],
            ],
            dtype=np.float64,
        )
        # Inject one extreme ectopic/segmented artifact 45 mm below the occlusal band
        noisy = np.vstack([clean_pts, np.array([[0.0, 25.0, -30.0]])])
        fitted = fit_dental_arch_from_points(noisy, jaw_type="lower", num_control_points=7)
        self.assertGreaterEqual(float(np.min(fitted[:, 2])), 12.0)

        tilted_up = np.array([0.0, 0.25, 0.968], dtype=np.float64)
        t_arr, n_arr, b_arr = compute_parallel_transport_frames(clean_pts, occlusal_normal=tilted_up)
        self.assertGreater(float(np.dot(b_arr[0], tilted_up)), 0.95)
        self.assertLess(abs(float(np.dot(t_arr[0], tilted_up))), 1e-6)
        self.assertLess(abs(float(np.dot(n_arr[0], b_arr[0]))), 1e-6)

    def test_anisotropic_spacing_edentulous_arch_fit(self) -> None:
        # Highly anisotropic voxel spacing: 0.3 mm in-plane, 1.2 mm z-slice thickness
        nz, ny, nx = 40, 220, 220
        sx, sy, sz = 0.3, 0.3, 1.2
        aff = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
        mask = np.zeros((nz, ny, nx), dtype=np.int32)
        zz, yy, xx = np.ogrid[:nz, :ny, :nx]
        x_mm = xx * sx
        y_mm = yy * sy
        z_mm = zz * sz
        u = ((x_mm - 33.0) / 24.0) ** 2 + ((y_mm - 20.0) / 28.0) ** 2
        mandible = (u >= 0.72) & (u <= 1.25) & (y_mm >= 16.0) & (z_mm >= 10.0) & (z_mm <= 26.0)
        mask[mandible] = 1

        fit = fit_arch_from_label_mask(mask, aff, jaw="Mandible", is_5class=False)
        self.assertTrue(fit.accepted, fit.reason)
        self.assertEqual(fit.method, "jawbone_crest")
        self.assertGreater(fit.arc_length_mm, 80.0)


class PanoramicEdgeClampAndPhantomTests(unittest.TestCase):
    def test_edge_slice_half_voxel_clamp_avoids_dark_border_artifact(self) -> None:
        PanoramicCurveLogic.clear_panoramic_caches()
        # Uniform bone slab extending all the way to voxel slice z=0 and z=nz-1
        img = np.full((16, 40, 40), 700.0, dtype=np.float32)
        aff = np.eye(4, dtype=np.float64)
        vol = CBCTVolume(img, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3), aff)
        pts = np.array([[8.0, 20.0, 7.5], [14.0, 20.0, 7.5], [22.0, 20.0, 7.5], [30.0, 20.0, 7.5]])
        t_arr, n_arr, b_arr = compute_parallel_transport_frames(pts)

        # height_mm=15.4 reaches z in [-0.2, 15.2] (inside half-voxel boundary [-0.5, 15.5], outside [0, 15])
        mean_pano = PanoramicCurveLogic.sample_panoramic_slab_array(
            vol, pts, t_arr, n_arr, b_arr, slab_thickness_mm=4.0, height_mm=15.4, spacing_mm=0.5, projection_mode="Mean"
        )
        # Border rows within half-voxel of slice 0 / 15 must stay at bone ~700 HU, NOT washed out toward -1000 air
        self.assertGreater(float(np.min(mean_pano)), 690.0)
        self.assertLess(float(np.max(mean_pano)), 710.0)

        # Samples genuinely far outside the FOV (height_mm=24 -> z up to +19.5) still return -1000 air
        wide_pano = PanoramicCurveLogic.sample_panoramic_slab_array(
            vol, pts, t_arr, n_arr, b_arr, slab_thickness_mm=4.0, height_mm=24.0, spacing_mm=0.5, projection_mode="Mean"
        )
        self.assertLess(float(wide_pano[0, len(wide_pano[0]) // 2]), -900.0)


if __name__ == "__main__":
    unittest.main()

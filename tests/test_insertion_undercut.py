"""Unit tests for mask-space surgical guide insertion & undercut evaluation."""

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

from CBCTPlannerLib.core.insertion import (  # noqa: E402
    compute_blocked_surface,
    suggest_insertion_axis,
)


class TestInsertionUndercut(unittest.TestCase):
    def _eye_affine(self, spacing_mm: float = 1.0) -> np.ndarray:
        aff = np.eye(4, dtype=np.float64)
        aff[0, 0] = spacing_mm
        aff[1, 1] = spacing_mm
        aff[2, 2] = spacing_mm
        return aff

    def test_box_has_zero_undercut_along_draw_axes(self) -> None:
        mask = np.zeros((28, 28, 28), dtype=np.uint8)
        mask[6:18, 6:18, 6:18] = 1
        aff = self._eye_affine(1.0)
        for axis in ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)):
            rep = compute_blocked_surface(mask, aff, axis)
            self.assertEqual(rep.blocked_voxel_count, 0)
            self.assertEqual(rep.blocked_fraction, 0.0)
            self.assertTrue(rep.is_insertable)

    def test_coronal_hemisphere_and_sphere_equator_behavior(self) -> None:
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        zz, yy, xx = np.ogrid[:32, :32, :32]
        # Top hemisphere / dome sitting cleanly on base at z=16 (0 undercut along -Z)
        mask[((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2 <= 8**2) & (zz >= 16)] = 1
        aff = self._eye_affine(1.0)
        rep_dome = compute_blocked_surface(mask, aff, (0.0, 0.0, -1.0))
        self.assertLessEqual(rep_dome.blocked_fraction, 0.05)
        self.assertTrue(rep_dome.is_insertable)

    def test_shelf_over_stem_blocks_vertical_and_clears_open_side(self) -> None:
        mask = np.zeros((28, 28, 28), dtype=np.uint8)
        # Stem + wide overhanging shelf above (in +Z) across all Y
        mask[4:14, 6:22, 11:17] = 1
        mask[14:19, 6:22, 5:23] = 1
        aff = self._eye_affine(1.0)

        rep_vert = compute_blocked_surface(mask, aff, (0.0, 0.0, -1.0))
        self.assertGreater(rep_vert.blocked_voxel_count, 10)
        self.assertFalse(rep_vert.is_insertable)
        self.assertGreater(rep_vert.worst_depth_mm, 1.0)

        # Along -Y, both shelf and stem share identical [6:22] profile -> zero undercut
        rep_side = compute_blocked_surface(mask, aff, (0.0, -1.0, 0.0))
        self.assertEqual(rep_side.blocked_voxel_count, 0)
        self.assertTrue(rep_side.is_insertable)

    def test_oblique_20_deg_cylinder_via_affine_and_axis_recovery(self) -> None:
        # Build voxel cylinder rotated 20 deg in RAS-XZ via rotated volume affine
        mask = np.zeros((44, 40, 40), dtype=np.uint8)
        zz, yy, xx = np.ogrid[:44, :40, :40]
        # Tilted axis cylinder inside mask itself around X-Z slant of 20 deg
        theta = math.radians(20.0)
        dz = zz - 22.0
        x_center = 20.0 - dz * math.tan(theta)
        inside = ((xx - x_center) ** 2 + (yy - 20.0) ** 2 <= 5.5**2) & (zz >= 10) & (zz <= 34)
        mask[inside] = 2

        # Identity affine rotated +15 deg in world? Here use anisotropic 0.5 mm RAS affine
        aff = np.eye(4, dtype=np.float64)
        aff[0, 0] = 0.5
        aff[1, 1] = 0.5
        aff[2, 2] = 0.5

        own_axis = np.array([math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float64)
        rep_own = compute_blocked_surface(mask, aff, own_axis, label_ids=[2])
        rep_z = compute_blocked_surface(mask, aff, (0.0, 0.0, -1.0), label_ids=[2])
        self.assertLess(rep_own.blocked_fraction, 0.02)
        self.assertGreater(rep_z.blocked_fraction, rep_own.blocked_fraction + 0.05)

        best_axis, best_rep, _ = suggest_insertion_axis(
            mask,
            aff,
            label_ids=[2],
            seed_axis_ras=(0.0, 0.0, -1.0),
            max_tilt_deg=25.0,
            n_candidates=25,
        )
        ang_err_deg = float(np.degrees(np.arccos(np.clip(float(np.dot(best_axis, own_axis)), -1.0, 1.0))))
        self.assertLessEqual(ang_err_deg, 5.0)
        self.assertTrue(best_rep.is_insertable)

        # Also test non-diagonal rotated 4x4 affine path (upright voxel cylinder, rotated RAS affine)
        upright = np.zeros((32, 32, 32), dtype=np.uint8)
        zu, yu, xu = np.ogrid[:32, :32, :32]
        upright[((xu - 16) ** 2 + (yu - 16) ** 2 <= 5.0**2) & (zu >= 6) & (zu <= 26)] = 3
        rot = np.eye(4, dtype=np.float64)
        rot[0, 0] = 0.5 * math.cos(theta)
        rot[0, 2] = 0.5 * math.sin(theta)
        rot[2, 0] = -0.5 * math.sin(theta)
        rot[2, 2] = 0.5 * math.cos(theta)
        rot[1, 1] = 0.5
        own_rot_axis = np.array([-math.sin(theta), 0.0, -math.cos(theta)], dtype=np.float64)
        rep_rot = compute_blocked_surface(upright, rot, own_rot_axis, label_ids=[3])
        rep_ras_z = compute_blocked_surface(upright, rot, (0.0, 0.0, -1.0), label_ids=[3])
        self.assertLess(rep_rot.blocked_fraction, 0.02)
        self.assertGreater(rep_ras_z.blocked_fraction, rep_rot.blocked_fraction + 0.05)

    def test_coronal_band_restricts_surface_to_the_seating_collar(self) -> None:
        # Stem under a wide shelf: the stem walls apical to the shelf are the true undercut.
        mask = np.zeros((28, 28, 28), dtype=np.uint8)
        mask[4:14, 6:22, 11:17] = 1
        mask[14:19, 6:22, 5:23] = 1
        aff = self._eye_affine(1.0)

        full = compute_blocked_surface(mask, aff, (0.0, 0.0, -1.0))
        banded = compute_blocked_surface(mask, aff, (0.0, 0.0, -1.0), coronal_band_mm=4.0)

        self.assertGreater(full.blocked_voxel_count, 10)
        self.assertFalse(full.is_insertable)
        self.assertLess(banded.blocked_voxel_count, full.blocked_voxel_count)
        self.assertTrue(banded.is_insertable)

    def test_empty_and_single_voxel_masks_return_cleanly(self) -> None:
        aff = self._eye_affine(0.5)
        empty = np.zeros((10, 10, 10), dtype=np.uint8)
        rep_empty = compute_blocked_surface(empty, aff, (0.0, 0.0, -1.0))
        self.assertEqual(rep_empty.surface_voxel_count, 0)
        self.assertEqual(rep_empty.blocked_voxel_count, 0)
        self.assertTrue(rep_empty.is_insertable)

        single = np.zeros((10, 10, 10), dtype=np.uint8)
        single[5, 5, 5] = 1
        rep_single = compute_blocked_surface(single, aff, (0.0, 0.0, -1.0))
        self.assertEqual(rep_single.surface_voxel_count, 0)
        self.assertTrue(rep_single.is_insertable)


if __name__ == "__main__":
    unittest.main()

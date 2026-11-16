"""Headless tests for lossless RAS canonicalization and orientation verification."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.core.arch_detector import JAW_TOOTH_ORDER  # noqa: E402
from CBCTPlannerLib.core.orientation import (  # noqa: E402
    apply_ops,
    axcodes_from_affine,
    canonicalize_to_ras,
    flip_axis,
    verify_orientation,
)


def _base_affine() -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 0.4
    affine[1, 1] = 0.35
    affine[2, 2] = 0.45
    affine[:3, 3] = (-9.0, -12.0, -14.0)
    return affine


def _stamp_ellipsoid(mask: np.ndarray, center_zyx, radii_zyx, value: int) -> None:
    cz, cy, cx = center_zyx
    rz, ry, rx = radii_zyx
    z0, z1 = max(0, int(cz - rz)), min(mask.shape[0], int(cz + rz) + 1)
    y0, y1 = max(0, int(cy - ry)), min(mask.shape[1], int(cy + ry) + 1)
    x0, x1 = max(0, int(cx - rx)), min(mask.shape[2], int(cx + rx) + 1)
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    inside = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    region = mask[z0:z1, y0:y1, x0:x1]
    region[inside] = value


def _arch_phantom(is_5class: bool) -> np.ndarray:
    """Two jaws plus a horseshoe of 16 FDI-labelled teeth in canonical RAS array order."""
    shape = (28, 56, 56)
    mask = np.zeros(shape, dtype=np.int16)
    if is_5class:
        mandible, maxilla, upper_teeth, lower_teeth = 2, 1, 3, 4
    else:
        from CBCTPlannerLib.core.labels import LOWER_JAW_ID, UPPER_JAW_ID

        mandible, maxilla = LOWER_JAW_ID, UPPER_JAW_ID
        upper_teeth = lower_teeth = None

    mask[6:9, 4:52, 4:52] = mandible
    mask[21:24, 4:52, 4:52] = maxilla

    radius = 24.0
    y_center = 28.0
    lower_order = JAW_TOOTH_ORDER["lower"]
    upper_order = JAW_TOOTH_ORDER["upper"]
    for index in range(16):
        theta = np.pi * (0.12 + 0.76 * index / 15.0)
        x = 28.0 + radius * np.cos(theta)
        y = y_center + radius * np.sin(theta)
        if is_5class:
            lower_label = lower_teeth
            upper_label = upper_teeth
        else:
            lower_label = int(lower_order[index])
            upper_label = int(upper_order[index])
        _stamp_ellipsoid(mask, (11.0, y, x), (1.6, 2.4, 2.4), lower_label)
        _stamp_ellipsoid(mask, (18.0, y, x), (1.6, 2.4, 2.4), upper_label)
    return mask


def _variant(arr: np.ndarray, affine: np.ndarray, perm3, signs):
    """Re-encode `arr` for a new IJK-axis permutation/sign set (old axis = perm3[new axis])."""
    perm_array = tuple(2 - perm3[2 - out_axis] for out_axis in range(3))
    out = np.transpose(arr, perm_array)
    flip_axes = tuple(2 - axis for axis in range(3) if signs[axis] < 0)
    for axis in flip_axes:
        out = np.flip(out, axis=axis)
    out = np.ascontiguousarray(out)

    new_affine = affine.copy()
    translation = affine[:3, 3].copy()
    for new_axis in range(3):
        old_axis = perm3[new_axis]
        new_affine[:3, new_axis] = signs[new_axis] * affine[:3, old_axis]
        if signs[new_axis] < 0:
            old_array_axis = 2 - old_axis
            translation = translation + affine[:3, old_axis] * float(arr.shape[old_array_axis] - 1)
    new_affine[:3, 3] = translation
    return out, new_affine


class TestRasCanonicalization(unittest.TestCase):
    def test_all_48_orientations_become_ras_with_unchanged_world_marker(self) -> None:
        shape = (10, 12, 14)
        arr = np.zeros(shape, dtype=np.float32)
        arr[2, 3, 4] = 1000.0
        arr[7, 9, 11] = 500.0
        affine = _base_affine()
        marker_world = affine @ np.array([4.0, 3.0, 2.0, 1.0])
        second_world = affine @ np.array([11.0, 9.0, 7.0, 1.0])

        for perm3 in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                variant, variant_affine = _variant(arr, affine, perm3, signs)
                result = canonicalize_to_ras(variant, variant_affine)
                self.assertEqual(axcodes_from_affine(result.affine), ("R", "A", "S"))
                for value, expected in ((1000.0, marker_world), (500.0, second_world)):
                    indices = np.argwhere(result.image_zyx == value)[0]
                    z, y, x = indices
                    world = result.affine @ np.array([x, y, z, 1.0])
                    np.testing.assert_allclose(world, expected, atol=1e-9)

    def test_canonicalize_is_noop_on_ras(self) -> None:
        arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        affine = _base_affine()
        result = canonicalize_to_ras(arr, affine)
        np.testing.assert_array_equal(result.image_zyx, arr)
        np.testing.assert_allclose(result.affine, affine)
        self.assertFalse(result.is_oblique)

    def test_oblique_affine_is_flagged_not_resampled(self) -> None:
        affine = _base_affine()
        angle = np.radians(10.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        affine[:3, :3] = rotation @ affine[:3, :3]
        result = canonicalize_to_ras(np.zeros((6, 6, 6)), affine)
        self.assertTrue(result.is_oblique)
        self.assertAlmostEqual(result.max_obliquity_deg, 10.0, places=5)

    def test_ops_can_be_reapplied_to_label_mask(self) -> None:
        mask = np.zeros((6, 7, 8), dtype=np.int16)
        mask[1, 2, 3] = 46
        affine = _base_affine()
        variant, variant_affine = _variant(mask, affine, (2, 0, 1), (1, -1, -1))
        result = canonicalize_to_ras(variant, variant_affine)
        canonical_mask = apply_ops(variant, result.ops)
        np.testing.assert_array_equal(canonical_mask, result.image_zyx)
        self.assertEqual(int(canonical_mask.sum()), 46)


class TestFlipAxis(unittest.TestCase):
    def test_flip_is_idempotent_and_swaps_codes(self) -> None:
        arr = np.zeros((6, 7, 8), dtype=np.float32)
        arr[1, 2, 3] = 9.0
        affine = _base_affine()
        before_codes = axcodes_from_affine(affine)
        flipped, flipped_affine = flip_axis(arr, affine, "LR")
        after_codes = axcodes_from_affine(flipped_affine)
        self.assertNotEqual(before_codes[0], after_codes[0])
        self.assertEqual((before_codes[1], before_codes[2]), (after_codes[1], after_codes[2]))

        restored, restored_affine = flip_axis(flipped, flipped_affine, "LR")
        np.testing.assert_array_equal(restored, arr)
        np.testing.assert_allclose(restored_affine, affine)

    def test_flip_preserves_world_bounding_box(self) -> None:
        arr = np.zeros((6, 7, 8), dtype=np.float32)
        affine = _base_affine()
        _, flipped_affine = flip_axis(arr, affine, "SI")
        corners = list(itertools.product((0, 7), (0, 6), (0, 5)))
        worlds = sorted(
            tuple(np.round((flipped_affine @ np.array([x, y, z, 1.0]))[:3], 9)) for x, y, z in corners
        )
        expected = sorted(tuple(np.round((affine @ np.array([x, y, z, 1.0]))[:3], 9)) for x, y, z in corners)
        self.assertEqual(worlds, expected)


class TestVerifyOrientation(unittest.TestCase):
    def test_toothfairy_phantom_is_ok_in_all_48_orientations(self) -> None:
        mask = _arch_phantom(is_5class=False)
        affine = _base_affine()
        for perm3 in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                variant, variant_affine = _variant(mask, affine, perm3, signs)
                canonical = canonicalize_to_ras(variant, variant_affine)
                canonical_mask = apply_ops(variant, canonical.ops)
                report = verify_orientation(canonical_mask, canonical.affine, is_5class=False)
                self.assertTrue(report.si_ok, (perm3, signs, report.notes))
                self.assertTrue(report.ap_ok, (perm3, signs, report.notes))
                self.assertTrue(report.lr_ok, (perm3, signs, report.notes))

    def test_5class_never_guesses_left_right(self) -> None:
        mask = _arch_phantom(is_5class=True)
        affine = _base_affine()
        report = verify_orientation(mask, affine, is_5class=True)
        self.assertTrue(report.si_ok, report.notes)
        self.assertTrue(report.ap_ok, report.notes)
        self.assertIsNone(report.lr_ok)
        self.assertTrue(any("L-R" in note for note in report.notes))

    def test_wrong_axis_inverts_only_that_flag(self) -> None:
        mask = _arch_phantom(is_5class=False)
        affine = _base_affine()
        self.assertTrue(verify_orientation(mask, affine, is_5class=False).si_ok)
        inverted_si = np.flip(mask, axis=0)
        report_si = verify_orientation(inverted_si, affine, is_5class=False)
        self.assertFalse(report_si.si_ok)

        inverted_ap = np.flip(mask, axis=1)
        report_ap = verify_orientation(inverted_ap, affine, is_5class=False)
        self.assertFalse(report_ap.ap_ok)

        inverted_lr = np.flip(mask, axis=2)
        report_lr = verify_orientation(inverted_lr, affine, is_5class=False)
        self.assertFalse(report_lr.lr_ok)

    def test_oblique_volume_resamples_to_ras_without_mirroring(self) -> None:
        from CBCTPlannerLib.core.orientation import resample_to_ras

        angle = np.deg2rad(11.0)
        rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
        )
        affine = np.eye(4)
        affine[:3, :3] = rotation
        affine[:3, 3] = [-15.0, -15.0, -15.0]
        image = np.zeros((24, 32, 40), dtype=np.int16)
        image[8:14, 12:22, 16:28] = 46
        result = resample_to_ras(image, affine)
        self.assertEqual(axcodes_from_affine(result.affine), ("R", "A", "S"))
        self.assertTrue(result.is_oblique)
        self.assertGreater(int(np.count_nonzero(result.image_zyx == 46)), 300)

        # World centroid must stay in the same anatomical octant and within one voxel.
        def world_centroid(arr, aff):
            z, y, x = np.nonzero(arr == 46)
            return (aff @ np.array([x.mean(), y.mean(), z.mean(), 1.0]))[:3]

        self.assertTrue(
            np.allclose(
                world_centroid(image, affine), world_centroid(result.image_zyx, result.affine), atol=1.5
            )
        )

    def test_oblique_ras_resampling_preserves_unsigned_background_and_full_extent(self) -> None:
        """Unsigned CT must not get a 64536 halo, and no world corner may be cropped."""
        from CBCTPlannerLib.core.orientation import _background_value, resample_to_ras

        angle = np.deg2rad(17.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([0.3, 0.35, 0.5])
        affine[:3, 3] = [-12.3, -10.1, -9.7]
        image = np.zeros((21, 25, 31), dtype=np.uint16)
        image[8:13, 10:15, 12:18] = 1200
        self.assertEqual(_background_value(np.dtype(np.uint16), -1000.0), 0.0)
        self.assertEqual(_background_value(np.dtype(np.uint8), -1000.0), 0.0)
        self.assertEqual(_background_value(np.dtype(np.int16), -1000.0), -1000.0)
        self.assertEqual(_background_value(np.dtype(np.float32), -1000.0), -1000.0)
        result = resample_to_ras(image, affine)
        self.assertEqual(result.image_zyx.dtype, np.uint16)
        self.assertLessEqual(int(result.image_zyx.max()), 1200)

        z, y, x = np.indices(image.shape)
        corners = np.stack([x.ravel(), y.ravel(), z.ravel(), np.ones(image.size)], axis=0)
        world = (affine @ corners)[:3]
        lo, hi = world.min(axis=1), world.max(axis=1)
        spacing = np.linalg.norm(result.affine[:3, :3], axis=0)
        got_lo = result.affine[:3, 3]
        got_hi = got_lo + (np.array(result.image_zyx.shape[::-1]) - 1) * spacing
        self.assertTrue(np.all(got_lo <= lo + 1e-6), (got_lo, lo))
        self.assertTrue(np.all(got_hi >= hi - 1e-6), (got_hi, hi))

    def test_oblique_ras_mask_resampling_preserves_label_identity(self) -> None:
        from CBCTPlannerLib.core.orientation import resample_mask_to_grid, resample_to_ras

        angle = np.deg2rad(13.0)
        rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
        )
        affine = np.eye(4)
        affine[:3, :3] = rotation
        image = np.zeros((28, 36, 44), dtype=np.int16)
        mask = np.zeros_like(image, dtype=np.uint16)
        mask[10:16, 14:24, 18:30] = 46
        mask[12:14, 17:20, 21:24] = 38
        result = resample_to_ras(image, affine)
        out_mask = resample_mask_to_grid(mask, affine, result.affine, result.image_zyx.shape)
        self.assertEqual(out_mask.dtype, np.uint16)
        self.assertEqual(set(np.unique(out_mask)) - {0}, {38, 46})
        for label in (38, 46):
            src_n = int(np.count_nonzero(mask == label))
            dst_n = int(np.count_nonzero(out_mask == label))
            self.assertGreater(dst_n, 0)
            self.assertLess(abs(dst_n - src_n) / src_n, 0.05)


if __name__ == "__main__":
    unittest.main()

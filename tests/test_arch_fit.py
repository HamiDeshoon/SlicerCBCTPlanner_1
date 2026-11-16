"""Headless tests for label-mask dental arch fitting (Phase 2)."""

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

from CBCTPlannerLib.core.arch_detector import (  # noqa: E402
    JAW_TOOTH_ORDER,
    fit_arch_from_label_mask,
)
from CBCTPlannerLib.core.orientation import apply_ops, canonicalize_to_ras  # noqa: E402


def _affine(shape=(40, 96, 96)) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = (-shape[2] / 2.0, -shape[1] / 2.0, -shape[0] / 2.0)
    return affine


def _arch_centerline(theta: np.ndarray) -> np.ndarray:
    radius_x = 32.0
    radius_y = 25.0
    return np.column_stack([radius_x * np.cos(theta), radius_y * np.sin(theta)])


def _stamp(mask: np.ndarray, center_xyz, radius_mm: float, value: int) -> None:
    cx, cy, cz = center_xyz
    x0, x1 = max(0, int(cx - radius_mm)), min(mask.shape[2], int(cx + radius_mm) + 2)
    y0, y1 = max(0, int(cy - radius_mm)), min(mask.shape[1], int(cy + radius_mm) + 2)
    z0, z1 = max(0, int(cz - radius_mm)), min(mask.shape[0], int(cz + radius_mm) + 2)
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius_mm**2
    mask[z0:z1, y0:y1, x0:x1][inside] = value


def _phantom(jaw: str, is_5class: bool, shape=(40, 96, 96)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.int16)
    affine = _affine(shape)
    theta_values = np.linspace(0.0, np.pi, 16)
    centerline = _arch_centerline(theta_values)
    for index, (x_world, y_world) in enumerate(centerline):
        if is_5class:
            label = 4 if jaw in ("lower", "shared") else 3
        else:
            order = JAW_TOOTH_ORDER["lower" if jaw != "upper" else "upper"]
            label = int(order[index])
        z_world = 6.0 if jaw in ("lower", "shared") else 20.0
        center_voxel = np.linalg.inv(affine) @ np.array([x_world, y_world, z_world, 1.0])
        _stamp(mask, center_voxel[:3], 2.6, label)
    return mask


def _variant(arr: np.ndarray, affine: np.ndarray, perm3, signs):
    perm_array = tuple(2 - perm3[2 - out_axis] for out_axis in range(3))
    out = np.transpose(arr, perm_array)
    for axis in range(3):
        if signs[axis] < 0:
            out = np.flip(out, axis=2 - axis)
    out = np.ascontiguousarray(out)
    new_affine = affine.copy()
    translation = affine[:3, 3].copy()
    for new_axis in range(3):
        old_axis = perm3[new_axis]
        new_affine[:3, new_axis] = signs[new_axis] * affine[:3, old_axis]
        if signs[new_axis] < 0:
            translation = translation + affine[:3, old_axis] * float(arr.shape[2 - old_axis] - 1)
    new_affine[:3, 3] = translation
    return out, new_affine


def _distance_to_centerline(points: np.ndarray) -> np.ndarray:
    theta = np.linspace(-np.pi, np.pi, 2000)
    centerline = _arch_centerline(theta)
    dists = np.linalg.norm(points[:, None, :2] - centerline[None, :, :], axis=2)
    return np.min(dists, axis=1)


class TestFitArchFromLabelMask(unittest.TestCase):
    def test_toothfairy_lower_upper_shared_fit_ground_truth(self) -> None:
        for jaw in ("lower", "upper", "shared"):
            mask = _phantom(jaw, is_5class=False)
            affine = _affine(mask.shape)
            result = fit_arch_from_label_mask(mask, affine, jaw=jaw, is_5class=False)
            self.assertIsNotNone(result, jaw)
            assert result is not None
            self.assertTrue(result.accepted, (jaw, result.reason))
            self.assertGreaterEqual(len(result.points_ras), 4)
            self.assertGreater(result.points_ras[0, 0], result.points_ras[-1, 0])
            distances = _distance_to_centerline(result.points_ras)
            self.assertLess(float(np.max(distances)), 1.5, (jaw, distances, result.method))

    def test_5class_fallback_principal_curve(self) -> None:
        for jaw in ("lower", "upper", "shared"):
            mask = _phantom(jaw, is_5class=True)
            affine = _affine(mask.shape)
            result = fit_arch_from_label_mask(mask, affine, jaw=jaw, is_5class=True)
            self.assertIsNotNone(result, jaw)
            assert result is not None
            self.assertTrue(result.accepted, (jaw, result.reason))
            self.assertEqual(result.method, "principal_curve")
            self.assertGreater(result.points_ras[0, 0], result.points_ras[-1, 0])
            distances = _distance_to_centerline(result.points_ras)
            self.assertLess(float(np.max(distances)), 1.5, (jaw, distances))

    def test_all_48_orientations_fit_after_canonicalization(self) -> None:
        mask = _phantom("lower", is_5class=False)
        affine = _affine(mask.shape)
        for perm3 in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                variant, variant_affine = _variant(mask, affine, perm3, signs)
                canonical = canonicalize_to_ras(variant, variant_affine)
                canonical_mask = apply_ops(variant, canonical.ops)
                result = fit_arch_from_label_mask(
                    canonical_mask, canonical.affine, jaw="lower", is_5class=False
                )
                self.assertIsNotNone(result, (perm3, signs))
                assert result is not None
                self.assertTrue(result.accepted, (perm3, signs, result.reason))
                self.assertGreater(result.points_ras[0, 0], result.points_ras[-1, 0])

    def test_end_extension_reaches_past_posterior_teeth(self) -> None:
        mask = _phantom("lower", is_5class=False)
        affine = _affine(mask.shape)
        base = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False)
        extended = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False, end_extension_mm=12.0)
        self.assertIsNotNone(base)
        self.assertIsNotNone(extended)
        assert base is not None and extended is not None
        self.assertGreater(extended.arc_length_mm, base.arc_length_mm + 20.0)
        # The new first/last points lie one extension length beyond the tooth cloud.
        self.assertAlmostEqual(
            float(np.linalg.norm(extended.points_ras[0] - base.points_ras[0])), 12.0, delta=0.5
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(extended.points_ras[-1] - base.points_ras[-1])), 12.0, delta=0.5
        )

    def test_no_teeth_returns_none_without_raising(self) -> None:
        mask = np.zeros((20, 20, 20), dtype=np.int16)
        self.assertIsNone(fit_arch_from_label_mask(mask, _affine(mask.shape), jaw="lower", is_5class=False))

    def test_residual_gate_rejects_garbage_cloud(self) -> None:
        rng = np.random.default_rng(7)
        mask = np.zeros((40, 96, 96), dtype=np.int16)
        x_coords = rng.integers(8, 88, size=800)
        y_coords = rng.integers(8, 88, size=800)
        z_coords = rng.integers(5, 35, size=800)
        mask[z_coords, y_coords, x_coords] = 46
        result = fit_arch_from_label_mask(mask, _affine(mask.shape), jaw="lower", is_5class=False)
        if result is not None:
            self.assertFalse(result.accepted)
            self.assertTrue(result.reason)


if __name__ == "__main__":
    unittest.main()

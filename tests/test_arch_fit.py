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
        # The clinical (tooth-bearing) arc must not be inflated by the cosmetic extension.
        self.assertAlmostEqual(extended.tooth_bearing_arc_mm, base.arc_length_mm, delta=0.5)
        self.assertAlmostEqual(extended.end_extension_mm, 12.0)
        self.assertAlmostEqual(extended.arc_length_mm - extended.tooth_bearing_arc_mm, 24.0, delta=0.5)
        # The new first/last points lie one extension length beyond the tooth cloud.
        self.assertAlmostEqual(
            float(np.linalg.norm(extended.points_ras[0] - base.points_ras[0])), 12.0, delta=0.5
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(extended.points_ras[-1] - base.points_ras[-1])), 12.0, delta=0.5
        )

    def test_extension_is_monotone_bounded_and_seam_continuous(self) -> None:
        """The 12 mm extension must not kink, flip slice normals, teleport, or leave the FOV."""
        from CBCTPlannerLib.core.arch_detector import (
            DentalArch,
            _extend_arch_endpoints,
            project_point_to_polyline_arc_mm,
        )

        points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]])
        bounded = _extend_arch_endpoints(points, 10.0, inverse_affine=np.eye(4), shape_zyx=(1, 1, 11))
        self.assertTrue(np.all(bounded[:, 0] <= 10.0 + 1e-9))
        self.assertAlmostEqual(float(bounded[-1, 0]), 10.0, delta=1e-9)
        steps = np.linalg.norm(np.diff(bounded, axis=0), axis=1)
        self.assertLessEqual(float(steps.max()), 0.2 + 1e-9)

        mask = _phantom("lower", is_5class=False)
        affine = _affine(mask.shape)
        base = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False)
        extended = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False, end_extension_mm=12.0)
        self.assertIsNotNone(base)
        self.assertIsNotNone(extended)
        assert base is not None and extended is not None
        self.assertAlmostEqual(extended.applied_extension_mm, 12.0, delta=0.1)
        arch = DentalArch(extended.points_ras, jaw_type="lower", ras_convention=True)
        dense_steps = np.linalg.norm(np.diff(arch.pts_dense, axis=0), axis=1)
        self.assertTrue(np.all(dense_steps > 1e-9))
        self.assertLess(float(dense_steps.max() / dense_steps.min()), 1.25)
        self.assertTrue(np.all(np.diff(arch.cum_arc_lengths) > 1e-9))

        vox = (
            np.linalg.inv(affine) @ np.hstack([extended.points_ras, np.ones((len(extended.points_ras), 1))]).T
        ).T[:, :3]
        self.assertTrue(np.all(vox >= -1e-6))
        self.assertTrue(np.all(vox <= np.array(mask.shape[::-1]) - 1 + 1e-6))

        for idx, base_idx in ((0, 0), (-1, -1)):
            vec = extended.points_ras[idx] - base.points_ras[base_idx]
            base_tangent = (
                base.points_ras[0] - base.points_ras[1]
                if idx == 0
                else base.points_ras[-1] - base.points_ras[-2]
            )
            self.assertGreater(
                float(np.dot(vec, base_tangent) / (np.linalg.norm(vec) * np.linalg.norm(base_tangent))), 0.999
            )
            seam, dist, _ = project_point_to_polyline_arc_mm(
                arch.pts_dense, arch.cum_arc_lengths, base.points_ras[base_idx]
            )
            self.assertLess(float(dist), 0.05)
            before = arch.get_frame_at_mm(seam - 0.5)
            after = arch.get_frame_at_mm(seam + 0.5)
            tangent_angle = np.degrees(np.arccos(np.clip(np.dot(before.tangent, after.tangent), -1.0, 1.0)))
            normal_angle = np.degrees(np.arccos(np.clip(np.dot(before.normal, after.normal), -1.0, 1.0)))
            self.assertLess(float(tangent_angle), 5.0)
            self.assertLess(float(normal_angle), 5.0)
            self.assertGreater(float(np.dot(before.normal, after.normal)), 0.99)

    def test_short_tooth_row_cannot_pass_gate_by_extending_ends(self) -> None:
        """A 45 mm tooth row is rejected even though a 20 mm extension makes the curve 85 mm."""
        mask = np.zeros((40, 96, 96), dtype=np.int16)
        affine = _affine(mask.shape)
        for index, theta in enumerate(np.linspace(0.0, np.pi, 8)):
            center = np.linalg.inv(affine) @ np.array([16.0 * np.cos(theta), 12.5 * np.sin(theta), 6.0, 1.0])
            _stamp(mask, center[:3], 2.0, int(JAW_TOOTH_ORDER["lower"][index]))
        fit = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False, end_extension_mm=20.0)
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertAlmostEqual(fit.tooth_bearing_arc_mm, 45.1, delta=1.5)
        self.assertGreater(fit.arc_length_mm, 80.0)
        self.assertFalse(fit.accepted)
        self.assertIn("tooth-bearing arc", fit.reason)

    def test_anchored_measurement_sites_are_unchanged_by_extension(self) -> None:
        """With per-tooth anchors, site points are the same physical points after extension."""
        from CBCTPlannerLib.core.arch_detector import (
            DentalArch,
            compute_fdi_arc_positions_mm,
        )

        mask = _phantom("lower", is_5class=False)
        affine = _affine(mask.shape)
        base = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False)
        extended = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=False, end_extension_mm=12.0)
        assert base is not None and extended is not None
        centroids = {}
        for fdi in ("33", "34", "35", "36", "43", "44", "45", "46"):
            z_idx, y_idx, x_idx = np.where(mask == int(fdi))
            centroids[fdi] = np.array(
                [
                    affine[0, 0] * x_idx.mean() + affine[0, 3],
                    affine[1, 1] * y_idx.mean() + affine[1, 3],
                    affine[2, 2] * z_idx.mean() + affine[2, 3],
                ]
            )
        base_arch = DentalArch(base.points_ras, jaw_type="lower", ras_convention=True)
        ext_arch = DentalArch(extended.points_ras, jaw_type="lower", ras_convention=True)
        base_sites = compute_fdi_arc_positions_mm(
            base_arch.total_arc_length, "lower", arch=base_arch, segmented_centroids=centroids
        )
        ext_sites = compute_fdi_arc_positions_mm(
            ext_arch.total_arc_length, "lower", arch=ext_arch, segmented_centroids=centroids
        )
        for fdi in centroids:
            self.assertEqual(base_sites[fdi].source, "segmentation")
            self.assertLess(
                float(np.linalg.norm(base_sites[fdi].point_ras - ext_sites[fdi].point_ras)),
                0.1,
            )

    def test_5class_tooth_span_excludes_retromolar_extension(self) -> None:
        """Approximate FDI sites must not be spread over the cosmetic 12 mm extension."""
        from CBCTPlannerLib.core.arch_detector import DentalArch
        from CBCTPlannerLib.core.measurements import MeasurementEngine
        from CBCTPlannerLib.core.volume_loader import CBCTVolume

        mask = _phantom("lower", is_5class=True)
        affine = _affine(mask.shape)
        fit = fit_arch_from_label_mask(mask, affine, jaw="lower", is_5class=True, end_extension_mm=12.0)
        self.assertIsNotNone(fit)
        assert fit is not None
        arch = DentalArch(fit.points_ras, jaw_type="lower", ras_convention=True)
        volume = CBCTVolume(
            mask.astype(np.float32),
            spacing=(1.0, 1.0, 1.0),
            origin=(-48.0, -48.0, -20.0),
            direction=np.eye(3),
            affine=affine,
        )
        engine = MeasurementEngine(
            volume,
            mask,
            arch=arch,
            arches={"lower": arch},
            is_dental_segmentator_5class=True,
            extract_teeth_eagerly=False,
        )
        span = engine._tooth_bearing_span_mm(arch, "lower")
        self.assertIsNotNone(span)
        assert span is not None
        self.assertGreater(span[0], 5.0)
        self.assertLess(span[1], arch.total_arc_length - 5.0)
        from CBCTPlannerLib.core.arch_detector import project_point_to_polyline_arc_mm

        for fdi in ("48", "47", "46", "45", "44", "43", "42", "41"):
            point, source, approximate = engine._estimate_missing_tooth_site_with_provenance(fdi)
            arc, _dist, _closest = project_point_to_polyline_arc_mm(
                arch.pts_dense, arch.cum_arc_lengths, point
            )
            self.assertEqual(source, "approximate_width_table")
            self.assertTrue(approximate)
            self.assertGreaterEqual(arc, span[0] - 0.5, (fdi, arc, span))
            self.assertLessEqual(arc, span[1] + 0.5, (fdi, arc, span))

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

    def test_edentulous_jaw_falls_back_to_mandible_and_maxilla_bone_labels(self) -> None:
        from CBCTPlannerLib.core.arch_detector import fit_arch_from_label_mask

        # Build an edentulous horseshoe mandible (label 1) and maxilla (label 2) with ZERO teeth.
        mask = np.zeros((60, 120, 120), dtype=np.int16)
        angles = np.linspace(0.2, np.pi - 0.2, 240)
        for ang in angles:
            cx = int(round(60 + 30 * np.cos(ang)))
            cy = int(round(35 + 32 * np.sin(ang)))
            mask[12:26, max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4] = 1  # Mandible
            mask[34:48, max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4] = 2  # Maxilla
        aff = np.eye(4, dtype=np.float64)
        res_mand = fit_arch_from_label_mask(mask, aff, jaw="lower", is_5class=False)
        res_max = fit_arch_from_label_mask(mask, aff, jaw="upper", is_5class=False)
        self.assertIsNotNone(res_mand)
        self.assertIsNotNone(res_max)
        self.assertEqual(res_mand.method, "jawbone_crest")
        self.assertEqual(res_max.method, "jawbone_crest")
        self.assertTrue(res_mand.accepted)
        self.assertTrue(res_max.accepted)

    def test_shared_mode_with_both_jaws_never_zigzags_in_z(self) -> None:
        mask_low = _phantom("lower", is_5class=False)
        mask_up = _phantom("upper", is_5class=False)
        combined = np.where(mask_low > 0, mask_low, mask_up)
        aff = _affine(combined.shape)
        res = fit_arch_from_label_mask(combined, aff, jaw="shared", is_5class=False)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertTrue(res.accepted, res.reason)
        self.assertIn("shared(", res.method)
        # Blended lower+upper curve must stay smooth in Z with zero inter-jaw vertical zigzag.
        self.assertLess(float(np.max(np.abs(np.diff(res.points_ras[:, 2])))), 2.0)

    def test_jawbone_extends_partially_edentulous_posterior_arch(self) -> None:
        # Stamp lower teeth except missing posterior right teeth (#45, #46, #47, #48), plus full mandible bone.
        mask = np.zeros((40, 96, 96), dtype=np.int16)
        aff = _affine(mask.shape)
        angles_bone = np.linspace(0.05, np.pi - 0.05, 180)
        for ang in angles_bone:
            xy = _arch_centerline(np.array([ang]))[0]
            vox = np.linalg.inv(aff) @ np.array([xy[0], xy[1], 6.0, 1.0])
            _stamp(mask, vox[:3], 2.4, 1)  # LOWER_JAW_ID
        order = JAW_TOOTH_ORDER["lower"]
        for index, theta in enumerate(np.linspace(0.0, np.pi, 16)):
            fdi = int(order[index])
            if fdi in (45, 46, 47, 48):
                continue
            xy = _arch_centerline(np.array([theta]))[0]
            vox = np.linalg.inv(aff) @ np.array([xy[0], xy[1], 7.0, 1.0])
            _stamp(mask, vox[:3], 2.5, fdi)
        res = fit_arch_from_label_mask(mask, aff, jaw="lower", is_5class=False, end_extension_mm=12.0)
        self.assertIsNotNone(res)
        assert res is not None
        self.assertTrue(res.accepted, res.reason)
        # Curve must extend posteriorly back along the right jawbone into missing molar region (X > 20, Y < 12).
        self.assertGreater(float(res.points_ras[0, 0]), 20.0)
        self.assertLess(float(res.points_ras[0, 1]), 12.0)


if __name__ == "__main__":
    unittest.main()

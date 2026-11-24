"""Unit tests for anatomical orientation auto-fix and aligned cross-section rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import unittest

import numpy as np

from CBCTPlannerLib.core.labels import LOWER_JAW_ID, UPPER_JAW_ID
from CBCTPlannerLib.core.orientation import (
    auto_fix_anatomical_orientation,
    flip_array_only,
    inverted_axes_from_report,
    verify_orientation,
)
from CBCTPlannerLib.core_print_render import (
    _px,
    measurement_footer_text,
    render_cross_section,
)


def _ras_affine() -> np.ndarray:
    aff = np.eye(4, dtype=np.float64)
    aff[0, 0] = 0.5
    aff[1, 1] = 0.5
    aff[2, 2] = 0.5
    aff[:3, 3] = (-20.0, -20.0, -20.0)
    return aff


def _build_dental_phantom() -> Tuple[np.ndarray, np.ndarray]:
    """Build a synthetic 3D (Z, Y, X) CBCT volume and multi-label FDI mask in true RAS."""
    shape = (40, 48, 48)
    img = np.full(shape, -1000.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.int32)

    # Canonical ToothFairy3 IDs: UPPER_JAW_ID (2) superior (high z), LOWER_JAW_ID (1) inferior (low z)
    mask[24:34, 16:36, 10:38] = UPPER_JAW_ID
    mask[6:16, 16:36, 10:38] = LOWER_JAW_ID
    img[mask > 0] = 800.0

    # Parabolic dental arch convex toward +Y (anterior at high y, molars at lower y)
    # Right side (+X: high x) = FDI 11/16/41/46; Left side (-X: low x) = FDI 21/26/31/36
    tooth_spots = [
        (22, 34, 28, 11),  # upper right central incisor (+X, +Y, +Z)
        (22, 34, 20, 21),  # upper left central incisor (-X, +Y, +Z)
        (22, 20, 36, 16),  # upper right molar (+X, -Y, +Z)
        (22, 20, 12, 26),  # upper left molar (-X, -Y, +Z)
        (16, 34, 28, 41),  # lower right central incisor (+X, +Y, -Z)
        (16, 34, 20, 31),  # lower left central incisor (-X, +Y, -Z)
        (16, 20, 36, 46),  # lower right molar (+X, -Y, -Z)
        (16, 20, 12, 36),  # lower left molar (-X, -Y, -Z)
    ]
    for zc, yc, xc, label in tooth_spots:
        mask[zc - 2 : zc + 2, yc - 2 : yc + 2, xc - 2 : xc + 2] = label
        img[zc - 2 : zc + 2, yc - 2 : yc + 2, xc - 2 : xc + 2] = 1800.0

    return img, mask


@dataclass
class _DummyCS:
    u_coords: np.ndarray
    v_coords: np.ndarray
    img_cs: np.ndarray
    mask_cs: np.ndarray
    crest_v: Optional[float] = 8.0
    basal_v: Optional[float] = -6.0
    height_u: float = 0.0
    is_mandible: bool = True
    width_segments: Dict[float, Tuple[float, float]] = field(
        default_factory=lambda: {
            0.0: (-3.2, 3.2),
            3.0: (-4.1, 4.0),
            5.0: (-4.6, 4.5),
            6.0: (-4.8, 4.7),
        }
    )
    canal_centroid_uv: Optional[Tuple[float, float]] = (0.5, -7.5)


@dataclass
class _DummyMeas:
    bone_height_mm: Optional[float] = 14.0
    crest_width_mm: Optional[float] = 6.4


class TestAnatomicalOrientationAutoFix(unittest.TestCase):
    def test_fixes_all_three_planes_flipped_simultaneously(self) -> None:
        img, mask = _build_dental_phantom()
        affine = _ras_affine()
        self.assertTrue(verify_orientation(mask, affine, is_5class=False).all_ok)

        # Invert all three anatomical axes in voxel data while keeping RAS affine
        inv_img = flip_array_only(img, affine, "ALL")
        inv_mask = flip_array_only(mask, affine, "ALL")
        bad_report = verify_orientation(inv_mask, affine, is_5class=False)
        self.assertFalse(bad_report.si_ok)
        self.assertFalse(bad_report.ap_ok)
        self.assertFalse(bad_report.lr_ok)
        self.assertEqual(set(inverted_axes_from_report(bad_report)), {"SI", "AP", "LR"})

        fixed_img, fixed_mask, flipped_axes, fixed_report = auto_fix_anatomical_orientation(
            inv_img, inv_mask, affine, is_5class=False
        )
        self.assertEqual(set(flipped_axes), {"SI", "AP", "LR"})
        self.assertTrue(fixed_report.all_ok)
        self.assertTrue(np.array_equal(fixed_img, img))
        self.assertTrue(np.array_equal(fixed_mask, mask))

    def test_noop_when_already_oriented(self) -> None:
        img, mask = _build_dental_phantom()
        affine = _ras_affine()
        fixed_img, fixed_mask, flipped_axes, report = auto_fix_anatomical_orientation(
            img, mask, affine, is_5class=False
        )
        self.assertEqual(flipped_axes, ())
        self.assertTrue(report.all_ok)
        self.assertTrue(np.array_equal(fixed_img, img))
        self.assertTrue(np.array_equal(fixed_mask, mask))


class TestCrossSectionVerticalAlignment(unittest.TestCase):
    def test_superior_voxels_render_at_top_matching_px(self) -> None:
        u_coords = np.linspace(-15.0, 15.0, 121)
        v_coords = np.linspace(-25.0, 25.0, 201)
        img_cs = np.full((len(v_coords), len(u_coords)), -1000.0, dtype=np.float32)
        mask_cs = np.zeros_like(img_cs, dtype=np.int32)

        # Place a bright bone block exclusively in the SUPERIOR half (v in [+10, +18] mm)
        sup_rows = (v_coords >= 10.0) & (v_coords <= 18.0)
        mid_cols = np.abs(u_coords) <= 4.0
        img_cs[np.ix_(sup_rows, mid_cols)] = 2500.0
        mask_cs[np.ix_(sup_rows, mid_cols)] = 2

        cs = _DummyCS(
            u_coords=u_coords,
            v_coords=v_coords,
            img_cs=img_cs,
            mask_cs=mask_cs,
            crest_v=None,
            basal_v=None,
            width_segments={},
            canal_centroid_uv=None,
        )
        render = render_cross_section(cs, None, scale=2)
        self.assertTrue(render.available)
        arr = np.asarray(render.image)

        # `_px(0, 14.0)` must land in the top half AND on bright pixels of the rendered image
        px_x, px_y = _px(0.0, 14.0, u_coords, v_coords, 2)
        self.assertLess(px_y, arr.shape[0] / 2.0)
        top_patch = arr[int(px_y) - 3 : int(px_y) + 3, int(px_x) - 3 : int(px_x) + 3].mean()
        # The mirrored inferior location (-14.0 mm) must be dark background
        _, inf_y = _px(0.0, -14.0, u_coords, v_coords, 2)
        bot_patch = arr[int(inf_y) - 3 : int(inf_y) + 3, int(px_x) - 3 : int(px_x) + 3].mean()
        self.assertGreater(top_patch, 150.0)
        self.assertLess(bot_patch, 40.0)

    def test_measurement_footer_never_fabricates_zero(self) -> None:
        self.assertEqual(measurement_footer_text(None), "H --  W0 --")
        self.assertEqual(
            measurement_footer_text(_DummyMeas(bone_height_mm=12.3, crest_width_mm=None)), "H 12.3 mm  W0 --"
        )


if __name__ == "__main__":
    unittest.main()

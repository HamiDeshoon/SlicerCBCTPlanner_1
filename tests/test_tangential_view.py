"""Tests for ridge-parallel (tangential) sections and their renderer."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CBCTPlanner"))

from CBCTPlannerLib.core.arch_detector import DentalArch  # noqa: E402
from CBCTPlannerLib.core.tangential import (  # noqa: E402
    TangentialSection,
    crest_drop_mm,
    render_tangential_section,
    sample_tangential_section,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402

SPACING = 0.5
ORIGIN = -10.0
SHAPE = 40  # 40 * 0.5 = 20 mm per axis, spanning [-10, 9.5] mm


def _volume(z_value) -> CBCTVolume:
    """Volume whose every voxel row k has intensity z_value(world_z)."""
    zs = ORIGIN + np.arange(SHAPE) * SPACING
    rows = [np.full((SHAPE, SHAPE), z_value(float(z)), dtype=np.float32) for z in zs]
    image = np.stack(rows)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = SPACING
    affine[:3, 3] = (ORIGIN, ORIGIN, ORIGIN)
    return CBCTVolume(image, (SPACING,) * 3, (ORIGIN, ORIGIN, ORIGIN), np.eye(3), affine)


def _arch() -> DentalArch:
    pts = np.array([[-8.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    return DentalArch(pts, jaw_type="lower", occlusal_normal=(0.0, 0.0, 1.0), ras_convention=True)


class TestTangentialSampling(unittest.TestCase):
    def test_shape_and_pixel_size(self) -> None:
        section = sample_tangential_section(
            _volume(lambda z: 800.0),
            _arch(),
            20.0,
            width_mm=8.0,
            height_mm=6.0,
            pixel_mm=0.5,
            thickness_mm=1.0,
        )
        self.assertTrue(section.available)
        self.assertEqual(section.shape, (13, 17))
        self.assertAlmostEqual(section.mm_per_px, 0.5)
        self.assertAlmostEqual(section.width_mm, 8.0)
        self.assertEqual(section.thickness_mode, "mean")

    def test_superior_rows_are_on_top(self) -> None:
        section = sample_tangential_section(
            _volume(lambda z: 800.0 if z > 0 else -800.0),
            _arch(),
            20.0,
            width_mm=10.0,
            height_mm=10.0,
            pixel_mm=0.5,
            thickness_mm=0.5,
            apical_bias=0.6,
        )
        self.assertTrue(section.available)
        arr = section.array
        top = float(np.mean(arr[:3, :]))
        bottom = float(np.mean(arr[-3:, :]))
        self.assertGreater(top, bottom)
        # v = 0 sits at (1 - apical_bias) of the height from the top.
        column = arr[:, arr.shape[1] // 2]
        crossing = int(np.argmax(column < 0.0))
        expected = (1.0 - 0.6) * (arr.shape[0] - 1)
        self.assertLessEqual(abs(crossing - expected), 2.0)

    def test_slab_modes_are_ordered(self) -> None:
        # Intensity grows with y (the buccal normal of a straight x-arch).
        def graded(z: float) -> float:
            return 1000.0 + 100.0 * z

        vol = CBCTVolume(
            np.broadcast_to(
                np.linspace(-1000, 1000, SHAPE, dtype=np.float32)[None, :, None], (SHAPE, SHAPE, SHAPE)
            ).copy(),
            (SPACING,) * 3,
            (ORIGIN, ORIGIN, ORIGIN),
            np.eye(3),
            np.array(
                [[SPACING, 0, 0, ORIGIN], [0, SPACING, 0, ORIGIN], [0, 0, SPACING, ORIGIN], [0, 0, 0, 1]],
                dtype=np.float64,
            ),
        )
        means = {}
        for mode in ("mean", "max", "min"):
            section = sample_tangential_section(
                vol, _arch(), 20.0, width_mm=6.0, height_mm=6.0, pixel_mm=0.5, thickness_mm=3.0, mode=mode
            )
            self.assertTrue(section.available)
            means[mode] = float(np.mean(section.array))
        self.assertGreater(means["max"], means["mean"])
        self.assertGreater(means["mean"], means["min"])

    def test_crest_line_follows_masked_bone(self) -> None:
        vol = _volume(lambda z: 900.0 if z > 0 else -900.0)
        zs = ORIGIN + np.arange(SHAPE) * SPACING
        mask = np.zeros((SHAPE, SHAPE, SHAPE), dtype=np.int32)
        for k, z in enumerate(zs):
            if z < 0:  # mandibular bone lies below the crest
                mask[k] = 1
        section = sample_tangential_section(
            vol, _arch(), 20.0, width_mm=10.0, height_mm=10.0, pixel_mm=0.5, mask=mask, is_mandible=True
        )
        self.assertGreater(len(section.crest_line_mm), 10)
        drop = crest_drop_mm(section)
        self.assertIsNotNone(drop)
        self.assertLess(drop, 1.5)

    def test_sloped_crest_reports_drop(self) -> None:
        vol = _volume(lambda z: 900.0)
        zs = ORIGIN + np.arange(SHAPE) * SPACING
        xs = ORIGIN + np.arange(SHAPE) * SPACING
        mask = np.zeros((SHAPE, SHAPE, SHAPE), dtype=np.int32)
        for k, z in enumerate(zs):
            for i, x in enumerate(xs):
                if z < 0.15 * x:  # bone below a crest that rises with +x
                    mask[k, :, i] = 1
        section = sample_tangential_section(
            vol, _arch(), 20.0, width_mm=12.0, height_mm=12.0, pixel_mm=0.5, mask=mask, is_mandible=True
        )
        drop = crest_drop_mm(section)
        self.assertIsNotNone(drop)
        self.assertGreater(drop, 1.0)

    def test_unavailable_paths(self) -> None:
        self.assertFalse(sample_tangential_section(None, _arch(), 10.0).available)
        self.assertFalse(sample_tangential_section(_volume(lambda z: 0.0), None, 10.0).available)
        empty = TangentialSection(array=None)
        self.assertEqual(empty.shape, (0, 0))
        self.assertIsNone(crest_drop_mm(empty))

    def test_arc_is_clamped_and_recorded(self) -> None:
        section = sample_tangential_section(
            _volume(lambda z: 700.0), _arch(), 999.0, width_mm=6.0, height_mm=6.0
        )
        self.assertTrue(section.available)
        self.assertGreaterEqual(section.arc_mm, 0.0)
        meta = section.to_dict()
        self.assertIn("thickness_mm", meta)
        self.assertTrue(meta["available"])
        self.assertEqual(len(meta["point_ras"]), 3)


class TestTangentialRender(unittest.TestCase):
    def _section(self, **kwargs) -> TangentialSection:
        return sample_tangential_section(
            _volume(lambda z: 800.0 if z > 0 else -600.0),
            _arch(),
            20.0,
            width_mm=10.0,
            height_mm=8.0,
            pixel_mm=0.5,
            mask=np.where((ORIGIN + np.arange(SHAPE) * SPACING)[:, None, None] < 0, 1, 0)
            .repeat(SHAPE, axis=1)
            .repeat(SHAPE, axis=2)
            .astype(np.int32),
            **kwargs,
        )

    def test_renders_rgba_with_air_transparency(self) -> None:
        section = self._section()
        render = render_tangential_section(section, scale=2)
        self.assertTrue(render.available, render.reason)
        self.assertEqual(render.image.mode, "RGBA")
        self.assertEqual(render.image.size, (section.shape[1] * 2, section.shape[0] * 2))
        alpha = np.asarray(render.image.split()[3])
        self.assertLess(int(alpha.min()), 255)
        self.assertAlmostEqual(render.mm_per_px, section.mm_per_px / 2.0)

    def test_footer_extends_canvas(self) -> None:
        section = self._section()
        plain = render_tangential_section(section, scale=1)
        with_footer = render_tangential_section(section, scale=1, footer_text="Tangential 20.0 mm")
        self.assertGreater(with_footer.image.size[1], plain.image.size[1])
        self.assertEqual(with_footer.image.size[0], plain.image.size[0])

    def test_monochrome_and_opaque(self) -> None:
        section = self._section()
        render = render_tangential_section(section, scale=1, monochrome=True, transparent_bg=False)
        self.assertTrue(render.available)
        alpha = np.asarray(render.image.split()[3])
        self.assertEqual(int(alpha.min()), 255)

    def test_unavailable_section(self) -> None:
        render = render_tangential_section(TangentialSection(available=False, reason="no arch"))
        self.assertFalse(render.available)
        self.assertIn("no arch", render.reason or "")


if __name__ == "__main__":
    unittest.main()

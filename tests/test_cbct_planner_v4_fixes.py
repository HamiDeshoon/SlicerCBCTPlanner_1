"""Comprehensive tests for SlicerCBCTPlanner_1 MAR, panoramic, full-quadrant,
smart multi-missing-tooth span, and mask-free implant-guide cross-section upgrades.
"""

from __future__ import annotations

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CBCT_DIR = REPO_ROOT / "CBCTPlanner"
if str(CBCT_DIR) not in sys.path:
    sys.path.insert(0, str(CBCT_DIR))

import importlib

mar = importlib.import_module("CBCTPlannerLib.RiseMarEngine")
from CBCTPlannerLib import Utils as utils
from CBCTPlannerLib import core_print_render as cpr
from CBCTPlannerLib.core.measurements import MeasurementEngine


class TestMarFixes(unittest.TestCase):
    def test_fast_mode_no_metal_preserves_volume_exactly(self):
        rng = np.random.default_rng(11)
        vol = rng.normal(loc=450.0, scale=110.0, size=(10, 96, 96)).astype(np.float32)
        vol[:, :10, :] = -950.0
        engine = mar.RiseMarEngine(default_weights_path="/nonexistent/weights.pkl")
        res = engine.run_risemar_on_volume(vol, config=mar.RiseMarConfig(metal_slices_only=True))
        self.assertEqual(res.mode_used, "no_metal_noop")
        self.assertEqual(res.processed_slices, 0)
        np.testing.assert_allclose(res.output_image_zyx, vol, atol=1e-5)

    def test_metal_correction_preserves_contrast_and_pre_mar_categorization(self):
        vol = np.full((8, 128, 128), -900.0, dtype=np.float32)
        vol[:, 24:104, 24:104] = 650.0
        vol[4, 60:68, 60:68] = 3200.0
        vol[4, 62:66, 70:92] = -400.0
        engine = mar.RiseMarEngine(default_weights_path="/nonexistent/weights.pkl")
        res = engine.run_risemar_on_volume(vol, config=mar.RiseMarConfig(metal_slices_only=True))
        self.assertEqual(res.metal_slice_indices, [4])
        self.assertGreater(res.metal_voxel_count, 0)
        self.assertAlmostEqual(res.pre_mar_categorization["tissue_median"], 650.0, delta=5.0)
        np.testing.assert_allclose(res.output_image_zyx[0], vol[0], atol=1e-5)
        self.assertGreater(float(np.mean(res.output_image_zyx[4, 62:66, 70:92])), -350.0)
        far_before = float(np.median(vol[4, 26:40, 26:40]))
        far_after = float(np.median(res.output_image_zyx[4, 26:40, 26:40]))
        self.assertAlmostEqual(far_after, far_before, delta=15.0)

    def test_uncalibrated_cbct_adaptive_metal_detection(self):
        vol = np.full((6, 96, 96), 400.0, dtype=np.float32)
        vol[:, 20:76, 20:76] = 900.0
        vol[3, 44:52, 44:52] = 2050.0
        cat = mar.categorize_volume_pre_mar(vol, metal_threshold_hu=2200.0, adaptive_metal_threshold=True)
        self.assertTrue(cat["adaptive_triggered"])
        self.assertEqual(cat["metal_slice_indices"], [3])


class TestSmartMissingToothExpansion(unittest.TestCase):
    def test_requesting_tooth_5_when_5_and_6_missing_expands_to_both(self):
        # Stub MeasurementEngine with teeth 44 and 47 present, 45 and 46 missing
        eng = object.__new__(MeasurementEngine)
        eng.is_dental_segmentator_5class = False
        present = {"41", "42", "43", "44", "47"}
        eng._get_tooth_info = lambda fdi: SimpleNamespace(fdi=fdi) if str(fdi) in present else None

        expanded, missing = eng.expand_missing_edentulous_span([45])
        self.assertEqual(expanded, [45, 46])
        self.assertEqual(missing, [45, 46])

    def test_requesting_present_tooth_does_not_expand(self):
        eng = object.__new__(MeasurementEngine)
        eng.is_dental_segmentator_5class = False
        present = {"44", "47"}
        eng._get_tooth_info = lambda fdi: SimpleNamespace(fdi=fdi) if str(fdi) in present else None

        expanded, missing = eng.expand_missing_edentulous_span([44])
        self.assertEqual(expanded, [44])
        self.assertEqual(missing, [])


class TestPanoramicAndCrossSectionRender(unittest.TestCase):
    def test_window_panoramic_layout_enlarged(self):
        self.assertIn('splitSize="580"', utils.CUSTOM_DENTAL_LAYOUT_XML)
        self.assertIn('splitSize="560"', utils.CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)

    def test_panoramic_upright_large_with_cross_section_lines(self):
        pano = np.full((120, 400), -900.0, dtype=np.float32)
        pano[:40, :] = 1200.0
        res = cpr.render_panoramic_strip(
            pano,
            column_spacing_mm=0.25,
            tick_arcs_mm=[20.0, 40.0, 60.0, 80.0],
            labels=["#44", "#45", "#46", "#47"],
            highlight_indices=[1, 2],
        )
        self.assertTrue(res.available)
        self.assertGreaterEqual(res.image.height, 340)
        self.assertEqual(len(res.overlay_px), 4)
        arr_rgb = np.asarray(res.image)
        self.assertGreater(
            float(np.mean(arr_rgb[250:290, 10:60])), float(np.mean(arr_rgb[10:40, 10:60])) + 50.0
        )

    def test_cross_section_mask_free_and_draws_vertical_entry_and_guides(self):
        u = np.linspace(-15.0, 15.0, 121)
        v = np.linspace(-20.0, 20.0, 161)
        img = np.full((len(v), len(u)), -900.0, dtype=np.float32)
        v_bone = (v >= -12.0) & (v <= 6.0)
        u_bone = (u >= -4.0) & (u <= 4.0)
        img[np.ix_(v_bone, u_bone)] = 950.0
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[np.ix_(v_bone, u_bone)] = 1

        # Single source of truth: when meas=None and crest_v=None, renderer draws NO fabricated H/W lines
        cs_empty = SimpleNamespace(
            img_cs=img,
            mask_cs=mask,
            u_coords=u,
            v_coords=v,
            crest_v=None,
            basal_v=None,
            height_u=0.0,
            width_segments={},
            is_mandible=True,
            canal_centroid_uv=None,
        )
        r1 = cpr.render_cross_section(cs_empty, meas=None, scale=3)
        self.assertTrue(r1.available)
        self.assertEqual(len(r1.overlay_px), 0)
        self.assertTrue(any("not measurable" in t for _, _, t in r1.labels))

        # Full measured site renders engine-supplied H, W0, Impl suggestion, and Misch density verbatim
        cs_full = SimpleNamespace(
            img_cs=img,
            mask_cs=mask,
            u_coords=u,
            v_coords=v,
            crest_v=6.0,
            basal_v=-8.0,
            height_u=0.0,
            width_segments={0.0: (-3.8, 3.8), 3.0: (-4.1, 4.1), 5.0: (-4.2, 4.2)},
            is_mandible=True,
            canal_centroid_uv=(0.0, -9.5),
        )
        meas = SimpleNamespace(
            bone_height_mm=14.0,
            crest_width_mm=7.6,
            suggested_diameter_mm=4.5,
            suggested_length_mm=11.5,
            misch_density_class="D2 (approximate)",
        )
        r2 = cpr.render_cross_section(cs_full, meas=meas, scale=3)
        texts = [t for _, _, t in r2.labels]
        self.assertTrue(any("H 14.0 mm" in t for t in texts), texts)
        self.assertTrue(any("W0 7.6 mm" in t for t in texts), texts)
        self.assertTrue(any("Impl 4.5x11.5 mm" in t for t in texts), texts)
        self.assertTrue(any("D2" in t for t in texts), texts)


if __name__ == "__main__":
    unittest.main()

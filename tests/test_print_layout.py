"""Print layout rules and panoramic strip geometry."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.PrintLayoutEngine import (  # noqa: E402
    PrintLayoutEngine,
    _data_uri_to_image,
    _image_to_data_uri,
    _paste_contain,
)
from CBCTPlannerLib.core.arch_detector import DentalArch  # noqa: E402
from CBCTPlannerLib.core.measurements import MeasurementEngine  # noqa: E402
from CBCTPlannerLib.core.print_layout import slice_count_for_selection  # noqa: E402
from CBCTPlannerLib.core.labels import sort_teeth_anatomically  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402
from CBCTPlannerLib.core_print_render import render_3d_mip_card, render_panoramic_strip  # noqa: E402


class TestPrintLayoutRules(unittest.TestCase):
    def test_slice_counts(self) -> None:
        self.assertEqual(slice_count_for_selection(1), (15, 0.75))
        self.assertEqual(slice_count_for_selection(2), (21, 0.85))
        self.assertEqual(slice_count_for_selection(3), (21, 0.85))
        self.assertEqual(slice_count_for_selection(4), (27, 1.0))
        self.assertEqual(slice_count_for_selection(8), (27, 1.0))

    def test_selection_is_sorted_right_to_left(self) -> None:
        self.assertEqual(
            sort_teeth_anatomically([48, 11, 21, 36, 46]),
            [11, 21, 48, 46, 36],
        )

    def test_panoramic_ticks_come_from_arc_positions(self) -> None:
        pano = np.tile(np.linspace(-500, 1500, 200, dtype=np.float32), (60, 1))
        render = render_panoramic_strip(
            pano,
            column_spacing_mm=0.5,
            tick_arcs_mm=[0.0, 25.0, 50.0],
            labels=["41", "44", "47"],
            strip_height=120,
            scale=2,
        )
        self.assertTrue(render.available)
        x_positions = [tick[0] for tick in render.labels]
        self.assertEqual(len(x_positions), 3)
        self.assertAlmostEqual(x_positions[1] - x_positions[0], 25.0 / 0.5 * 2, delta=1.0)
        self.assertAlmostEqual(x_positions[2] - x_positions[1], 25.0 / 0.5 * 2, delta=1.0)

    def test_engine_arches_update_without_rebuilding_mask(self) -> None:
        engine = object.__new__(MeasurementEngine)
        engine.arch = object()
        engine.arches = {}
        engine.localizer = None
        lower = object()
        upper = object()
        engine.set_arches({"lower": lower, "upper": upper})
        self.assertIs(engine.arch, lower)
        self.assertIs(engine._arch_for_jaw(True), lower)
        self.assertIs(engine._arch_for_jaw(False), upper)

    def test_invalid_panoramic_is_explicitly_unavailable(self) -> None:
        render = render_panoramic_strip(np.empty((0, 0)), column_spacing_mm=0.5, tick_arcs_mm=[0.0])
        self.assertFalse(render.available)
        self.assertIsNotNone(render.reason)

    def test_render_panoramic_strip_transparent_bg_crop_and_isotropic(self) -> None:
        pano = np.full((160, 260), -1250.0, dtype=np.float32)
        pano[55:115, 20:240] = 650.0
        # Simulate an interior radiolucent sinus/canal pocket inside the jaw
        pano[70:82, 80:110] = -250.0

        render = render_panoramic_strip(
            pano,
            column_spacing_mm=0.5,
            row_spacing_mm=0.5,
            tick_arcs_mm=[20.0, 60.0, 100.0],
            labels=["#46", "#44", "#41"],
            target_size=(2256, 552),
            transparent_bg=True,
            crop_empty_vertical=True,
            preserve_aspect=True,
            monochrome=True,
        )
        self.assertTrue(render.available)
        assert render.image is not None
        self.assertEqual(render.image.mode, "RGBA")
        # Isotropic tight fit: wide arch stays wider than tall without vertical stretch
        self.assertGreater(render.image.width, int(render.image.height * 1.5))
        self.assertLessEqual(render.image.height, 552)
        arr = np.asarray(render.image)
        # Background air and tick-strip empty space stay transparent (alpha=0)
        self.assertEqual(int(arr[0, 0, 3]), 0)
        self.assertEqual(int(arr[-4, render.image.width // 2, 3]), 0)
        # Monochrome preserves R == G == B across pixels
        self.assertTrue(np.array_equal(arr[:, :, 0], arr[:, :, 1]))
        self.assertTrue(np.array_equal(arr[:, :, 1], arr[:, :, 2]))
        # Anatomy pixels remain opaque
        self.assertGreater(int(np.max(arr[:, :, 3])), 200)

    def test_pano_strip_uri_defaults_to_raysum_and_honors_param_node(self) -> None:
        vol_arr = np.full((40, 60, 80), -1000.0, dtype=np.float32)
        vol_arr[15:30, 20:35, 10:70] = 800.0
        vol = CBCTVolume(vol_arr, (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), np.eye(3), np.diag([0.5, 0.5, 0.5, 1.0]))
        arch = DentalArch(
            np.array([[5.0, 14.0, 11.0], [15.0, 14.0, 11.0], [25.0, 14.0, 11.0], [35.0, 14.0, 11.0]]),
            jaw_type="lower",
            smoothing=0.0,
            ras_convention=True,
        )
        engine = SimpleNamespace(volume=vol, _arch_for_jaw=lambda _is_mand: arch)
        seen_modes: list[str] = []

        from CBCTPlannerLib.PanoramicCurveLogic import PanoramicCurveLogic

        orig = PanoramicCurveLogic.sample_panoramic_slab_array.__func__

        def _spy(cls, *args, **kwargs):
            seen_modes.append(str(kwargs.get("projection_mode")))
            return orig(cls, *args, **kwargs)

        with patch.object(PanoramicCurveLogic, "sample_panoramic_slab_array", classmethod(_spy)):
            ple_default = PrintLayoutEngine(logic=None)
            uri = ple_default._pano_strip_uri([5.0, 15.0], ["#46", "#44"], engine=engine, monochrome=True)
            self.assertIsNotNone(uri)
            self.assertEqual(seen_modes[-1], "RaySum")
            decoded = _data_uri_to_image(uri or "")
            self.assertIsNotNone(decoded)
            assert decoded is not None
            self.assertEqual(decoded.mode, "RGBA")
            self.assertGreater(decoded.width, decoded.height)

            pnode = SimpleNamespace(GetParameter=lambda k: "MIP" if k == "PanoProjectionMode" else "")
            ple_param = PrintLayoutEngine(logic=SimpleNamespace(getParameterNode=lambda: pnode))
            ple_param._pano_strip_uri([5.0, 15.0], ["#46", "#44"], engine=engine, monochrome=False)
            self.assertEqual(seen_modes[-1], "MIP")

    def test_render_3d_mip_card_rgba_aspect_and_toothfairy_canals(self) -> None:
        vol = np.full((40, 30, 80), -1000.0, dtype=np.float32)
        vol[10:32, 10:20, 25:55] = 750.0
        mask = np.zeros_like(vol, dtype=np.int32)
        mask[12:24, 10:20, 30:50] = 1
        mask[22:30, 12:18, 36:44] = 46
        mask[14:18, 12:18, 34:46] = 103  # ToothFairy3 canal label

        # Physical extent: Z=22*2.0=44mm tall vs X=30*0.5=15mm wide -> must stay taller than wide
        render = render_3d_mip_card(
            vol,
            mask=mask,
            spacing_zyx=(2.0, 1.0, 0.5),
            quadrant=4,
            target_size=(772, 272),
            transparent_bg=True,
            show_title=True,
        )
        self.assertTrue(render.available)
        assert render.image is not None
        self.assertEqual(render.image.mode, "RGBA")
        self.assertGreater(render.image.height, render.image.width)
        arr = np.asarray(render.image)
        self.assertEqual(int(arr[-1, -1, 3]), 0)
        # ToothFairy3 canal must render vivid red
        self.assertTrue(np.any((arr[:, :, 0] > 200) & (arr[:, :, 1] < 100) & (arr[:, :, 3] == 255)))

    def test_data_uri_and_paste_contain_preserve_rgba_transparency(self) -> None:
        from PIL import Image

        rgba = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        for y in range(12, 28):
            for x in range(12, 28):
                rgba.putpixel((x, y), (255, 0, 0, 255))
        uri = _image_to_data_uri(rgba)
        decoded = _data_uri_to_image(uri)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.mode, "RGBA")

        canvas = Image.new("RGB", (60, 60), (255, 255, 255))
        _paste_contain(canvas, decoded, (0, 0, 60, 60), bg=(255, 255, 255))
        # Transparent corner stays white paper, not black
        self.assertEqual(canvas.getpixel((2, 2)), (255, 255, 255))
        self.assertEqual(canvas.getpixel((30, 30)), (255, 0, 0))
        # Verify base64 round-trip preserved alpha bytes
        raw = base64.b64decode(uri.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as reopened:
            self.assertEqual(reopened.mode, "RGBA")

    def test_html_layout_uses_transparent_pano_and_vr_backgrounds(self) -> None:
        ple = PrintLayoutEngine(logic=None)
        for gs in (False, True):
            doc = ple._build_8x10_html(
                slices=[],
                pano_b64="data:image/png;base64,AA==",
                vr_b64="data:image/png;base64,AA==",
                quadrant=4,
                jaw_label="Mandible",
                fdis=[46],
                patient_name="Anonymous",
                patient_id="Anonymous",
                grayscale=gs,
            )
            self.assertNotIn("max-height: 220px", doc)
            self.assertIn(
                ".pano-img { width: 100%; height: auto; display: block; object-fit: contain; background: transparent;",
                doc,
            )
            self.assertIn(
                ".vr-img { width: 100%; height: auto; max-height: 150px; display: block; object-fit: contain; background: transparent;",
                doc,
            )


if __name__ == "__main__":
    unittest.main()

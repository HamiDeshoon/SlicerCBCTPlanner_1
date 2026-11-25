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
from CBCTPlannerLib.core.annotations import AnnotationStore  # noqa: E402
from CBCTPlannerLib.core.arch_detector import DentalArch  # noqa: E402
from CBCTPlannerLib.core.measurements import CrossSectionData, MeasurementEngine  # noqa: E402
from CBCTPlannerLib.core.print_layout import slice_count_for_selection  # noqa: E402
from CBCTPlannerLib.core.labels import sort_teeth_anatomically  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402
from CBCTPlannerLib.core_print_render import (  # noqa: E402
    render_3d_mip_card,
    render_cross_section,
    render_panoramic_strip,
)


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


    def test_grayscale_layout_includes_vr_and_omits_redundant_card_footer(self) -> None:
        from CBCTPlannerLib.PrintLayoutEngine import CrossSectionSliceSpec

        ple = PrintLayoutEngine(logic=None)
        spec = CrossSectionSliceSpec(
            slice_index=1,
            total_slices=1,
            offset_mm=0.0,
            fdi=46,
            image_data_uri="data:image/png;base64,AA==",
            height_mm=12.0,
            crest_width_mm=7.5,
            width_3mm=8.0,
            width_5mm=8.5,
        )
        html_doc = ple._build_8x10_html(
            slices=[spec],
            pano_b64="data:image/png;base64,AA==",
            vr_b64="data:image/png;base64,BB==",
            quadrant=4,
            jaw_label="Mandible",
            fdis=[46],
            patient_name="Anonymous",
            patient_id="Anonymous",
            grayscale=True,
        )
        self.assertIn("data:image/png;base64,BB==", html_doc)
        self.assertNotIn("<div class='cs-ftr'>", html_doc)


    def test_generate_8x10_layout_applies_measurement_overrides(self) -> None:
        from types import SimpleNamespace
        from CBCTPlannerLib.core.measurements import CrossSectionData
        from CBCTPlannerLib.core.arch_detector import DentalArch

        mask_arr = np.zeros((40, 40, 40), dtype=np.int16)
        img_arr = np.full((40, 40, 40), -500.0, dtype=np.float32)
        arch = DentalArch(
            control_points=np.array([[5.0, 20.0, 20.0], [15.0, 20.0, 20.0], [25.0, 20.0, 20.0], [35.0, 20.0, 20.0]], dtype=np.float64),
            jaw_type="lower",
            ras_convention=True,
        )

        class _StubEngine:
            volume = SimpleNamespace(image=img_arr, spacing=(1.0, 1.0, 1.0))
            mask = mask_arr
            arch_mand = arch
            arch_max = None
            last_withheld_reason = None

            def get_site_center_ras(self, _fdi):
                return np.array([20.0, 20.0, 20.0], dtype=np.float64)

            def _arch_for_fdi(self, _fdi):
                return self.arch_mand

            def measure_at_arc_mm(self, jaw_key, arc_mm, fdi_hint=None):
                u = np.linspace(-10.0, 10.0, 41)
                v = np.linspace(-15.0, 15.0, 61)
                cs = CrossSectionData(
                    u_coords=u,
                    v_coords=v,
                    img_cs=np.zeros((len(v), len(u)), dtype=np.float32),
                    mask_cs=np.zeros((len(v), len(u)), dtype=np.int16),
                    frame=None,
                    crest_v=5.0,
                    width_segments={},
                    canal_centroid_uv=None,
                    crest_uv=(0.0, 5.0),
                    basal_uv=(0.0, -5.0),
                    width_endpoints_uv={0.0: ((-3.0, 5.0), (3.0, 5.0))},
                )
                meas = SimpleNamespace(
                    tooth_fdi=str(fdi_hint or "46"),
                    bone_height_mm=10.0,
                    crest_width_mm=6.0,
                    bone_width_at_depths_mm={0.0: 6.0, 3.0: 7.0, 5.0: 7.5},
                    distance_to_vital_structure_mm=None,
                    misch_density_class="D2",
                    mean_density_hu=700.0,
                    provenance="label_derived",
                    withheld_reason=None,
                )
                return meas, cs

        logic = SimpleNamespace(_get_or_build_measurement_engine=lambda _v, _s: _StubEngine())
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _html, _pdf, specs = PrintLayoutEngine(logic=logic).generate_8x10_layout(
                volumeNode=object(),
                segmentationNode=None,
                selected_fdis=[46],
                out_dir=Path(tmp),
                grayscale=True,
                measurement_overrides={46: {"h_mm": 13.5, "w0_mm": 7.8}},
            )
            center_specs = [s for s in specs if abs(float(s.offset_mm)) < 0.05]
            self.assertTrue(center_specs)
            self.assertAlmostEqual(center_specs[0].height_mm, 13.5)
            self.assertAlmostEqual(center_specs[0].crest_width_mm, 7.8)

    def test_w3_w5_legend_moves_off_overlapping_corner_and_avoids_scale_bar(self) -> None:
        u = np.linspace(-10.0, 10.0, 61)
        v = np.linspace(-15.0, 15.0, 91)
        img_cs = np.full((len(v), len(u)), -500.0, dtype=np.float32)
        mask_cs = np.zeros((len(v), len(u)), dtype=np.int16)
        # Occupy bottom-right corner (`v` lowest indices -> bottom rows after render flip).
        mask_cs[:28, 32:] = 1
        img_cs[:28, 32:] = 800.0
        cs = CrossSectionData(
            u_coords=u,
            v_coords=v,
            img_cs=img_cs,
            mask_cs=mask_cs,
            frame=None,
            crest_v=6.0,
            width_segments={},
            canal_centroid_uv=None,
            crest_uv=(-2.0, 6.0),
            basal_uv=(-2.0, -4.0),
            width_endpoints_uv={
                0.0: ((-5.0, 6.0), (1.0, 6.0)),
                3.0: ((-5.5, 3.0), (1.5, 3.0)),
                5.0: ((-6.0, 1.0), (2.0, 1.0)),
            },
        )
        meas = SimpleNamespace(
            bone_height_mm=10.0,
            crest_width_mm=6.0,
            bone_width_at_depths_mm={0.0: 6.0, 3.0: 7.0, 5.0: 8.0},
            misch_density_class="D2",
            provenance="label_derived",
            withheld_reason=None,
        )
        rnd = render_cross_section(cs, meas, scale=3, show_masks=False, monochrome=True, show_footer=False)
        self.assertTrue(rnd.available)
        key_labels = [(x, y) for x, y, t in rnd.labels if t == "W3/W5 key"]
        self.assertEqual(len(key_labels), 1)
        kx, ky = key_labels[0]
        # Moved off bottom-right (x stays left of midpoint) and sits above bottom scale bar.
        self.assertLess(kx, rnd.image.width * 0.45)
        self.assertGreaterEqual(ky, 24.0)
        self.assertLess(ky, rnd.image.height - 32.0)

        # Now occupy BOTH bottom corners on a standard 30 mm field -> legend relocates to top-left below L badge.
        u2 = np.linspace(-15.0, 15.0, 91)
        img2 = np.full((len(v), len(u2)), -500.0, dtype=np.float32)
        mask_cs2 = np.zeros((len(v), len(u2)), dtype=np.int16)
        mask_cs2[:30, :] = 1
        img2[:30, :] = 800.0
        cs2 = CrossSectionData(
            u_coords=u2,
            v_coords=v,
            img_cs=img2,
            mask_cs=mask_cs2,
            frame=None,
            crest_v=4.0,
            width_segments={},
            canal_centroid_uv=None,
            crest_uv=(0.0, 4.0),
            basal_uv=(0.0, -4.0),
            width_endpoints_uv={
                0.0: ((-3.0, 4.0), (3.0, 4.0)),
                3.0: ((-3.5, 1.0), (3.5, 1.0)),
            },
        )
        rnd2 = render_cross_section(cs2, meas, scale=3, show_masks=False, monochrome=True, show_footer=False)
        kx2, ky2 = [(x, y) for x, y, t in rnd2.labels if t == "W3/W5 key"][0]
        self.assertGreaterEqual(ky2, 24.0)
        self.assertLess(ky2, rnd2.image.height * 0.45)

    def test_bl14_clinical_notes_strip_renders_when_flagged_and_omits_when_empty(self) -> None:
        mask_arr = np.zeros((40, 40, 40), dtype=np.int16)
        img_arr = np.full((40, 40, 40), -500.0, dtype=np.float32)
        arch = DentalArch(
            control_points=np.array([[5.0, 20.0, 20.0], [15.0, 20.0, 20.0], [25.0, 20.0, 20.0], [35.0, 20.0, 20.0]], dtype=np.float64),
            jaw_type="lower",
            ras_convention=True,
        )

        class _MultiStubEngine:
            volume = SimpleNamespace(image=img_arr, spacing=(1.0, 1.0, 1.0))
            mask = mask_arr
            arch_mand = arch
            arch_max = None
            last_withheld_reason = None

            def get_site_center_ras(self, fdi):
                return np.array([15.0 if str(fdi) == "46" else 25.0, 20.0, 20.0], dtype=np.float64)

            def _arch_for_fdi(self, _fdi):
                return self.arch_mand

            def measure_at_arc_mm(self, _jaw_key, _arc_mm, fdi_hint=None):
                u = np.linspace(-10.0, 10.0, 41)
                v = np.linspace(-15.0, 15.0, 61)
                cs = CrossSectionData(
                    u_coords=u,
                    v_coords=v,
                    img_cs=np.zeros((len(v), len(u)), dtype=np.float32),
                    mask_cs=np.zeros((len(v), len(u)), dtype=np.int16),
                    frame=None,
                    crest_v=5.0,
                    width_segments={},
                    canal_centroid_uv=None,
                    crest_uv=(0.0, 5.0),
                    basal_uv=(0.0, -5.0),
                    width_endpoints_uv={0.0: ((-3.0, 5.0), (3.0, 5.0))},
                )
                meas = SimpleNamespace(
                    tooth_fdi=str(fdi_hint or "46"),
                    bone_height_mm=10.0,
                    crest_width_mm=6.0,
                    bone_width_at_depths_mm={0.0: 6.0, 3.0: 7.0},
                    distance_to_vital_structure_mm=None,
                    misch_density_class="D2",
                    mean_density_hu=700.0,
                    provenance="label_derived",
                    withheld_reason=None,
                )
                return meas, cs

        logic = SimpleNamespace(_get_or_build_measurement_engine=lambda _v, _s: _MultiStubEngine())
        store = AnnotationStore()
        store.add("Suspected radiolucency — manual review", fdi=46, kind="flag", severity="warning")
        store.add("Verify cortical plate", fdi=47, kind="ai_comment", severity="info")
        store.add("Routine observation", fdi=46, kind="comment", severity="info")

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            html_path, pdf_path, specs = PrintLayoutEngine(logic=logic).generate_8x10_layout(
                volumeNode=object(),
                segmentationNode=None,
                selected_fdis=[46, 47],
                out_dir=Path(tmp),
                grayscale=True,
                annotation_store=store,
            )
            html_doc = Path(html_path).read_text(encoding="utf-8")
            self.assertIn("Clinical Notes &amp; Review Flags:", html_doc)
            self.assertIn("Suspected radiolucency", html_doc)
            self.assertIn("Verify cortical plate", html_doc)
            self.assertNotIn("Routine observation", html_doc)
            self.assertTrue((Path(tmp) / "CBCT_8x10_Print_Layout.png").exists())
            self.assertTrue(Path(pdf_path).exists())
            self.assertTrue(any(abs(float(s.offset_mm)) < 0.05 for s in specs))

        with tempfile.TemporaryDirectory() as tmp_empty:
            html_empty, _pdf2, _specs2 = PrintLayoutEngine(logic=logic).generate_8x10_layout(
                volumeNode=object(),
                segmentationNode=None,
                selected_fdis=[46, 47],
                out_dir=Path(tmp_empty),
                grayscale=True,
                annotation_store=None,
            )
            html_doc_empty = Path(html_empty).read_text(encoding="utf-8")
            self.assertNotIn("Clinical Notes &amp; Review Flags:", html_doc_empty)


if __name__ == "__main__":
    unittest.main()

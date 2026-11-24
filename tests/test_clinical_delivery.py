"""Clinical delivery regression suite.

Pins the behaviours added for the grayscale film workflow and the follow-up hardening:

* W0 / entry point come from jaw bone, never from the tooth crown (the "tooth as entry" bug)
* the entry point cannot jump between adjacent slices (interpolated from neighbours)
* the height caliper follows the PCA long axis and continues to the jaw end without a nerve
* W3/W5 are semi-transparent, root/implant/2 mm overlays are not drawn, film is grayscale
* RAS auto-standardization is the persisted default
* typed jaw keys + an anti-pattern source scan
* real ToothFairy3 NIfTI fixture through orientation -> arch fit -> measurement
* preflight/site diagnostics, provenance hash, calibration guard, span worst case
* scheme-detection hardening, cache invalidation, shared-arch fallback, randomized invariants
"""

from __future__ import annotations

# ruff: noqa: E402

import gzip
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CBCTPlanner"))

from CBCTPlannerLib.PrintLayoutEngine import PrintLayoutEngine
from CBCTPlannerLib.core.arch_detector import DentalArch, fit_arch_from_label_mask
from CBCTPlannerLib.core.diagnostics import (
    build_preflight_checklist,
    calibration_stamp,
    calibration_status,
    checklist_text,
    measurement_report_text,
    provenance_hash,
    summarize_span_worst_case,
    write_diagnostics_text,
)
from CBCTPlannerLib.core.labels import (
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    infer_mask_is_5class,
    jaw_key_from_text,
    parse_jaw_key,
)
from CBCTPlannerLib.core.measurements import (
    MAX_ENTRY_JUMP_MM,
    MeasurementEngine,
    MeasurementRecord,
)
from CBCTPlannerLib.core.orientation import axcodes_from_affine, canonicalize_to_ras
from CBCTPlannerLib.core.volume_loader import CBCTVolume
from CBCTPlannerLib.core_print_render import (
    WIDTH_ALPHA_W0,
    WIDTH_ALPHA_W3,
    WIDTH_ALPHA_W5,
    _px,
    render_cross_section,
)

FIXTURE = ROOT / "tests" / "data" / "ToothFairy3S_0021_seg.nii.gz"
REAL_IMAGE = Path("C:/Users/hamid/Downloads/inspect_ToothFairy3S_0021/ToothFairy3S_0021_0000.nii.gz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nifti_header(path: Path) -> tuple[tuple[int, ...], np.ndarray, int]:
    """Read just the NIfTI-1 header: shape (x, y, z), RAS affine, datatype code."""
    with gzip.open(str(path), "rb") as fh:
        header = fh.read(352)
    dim = struct.unpack("<8h", header[40:56])
    ndim = int(dim[0])
    shape = tuple(int(v) for v in dim[1 : 1 + ndim])
    datatype = int(struct.unpack("<h", header[70:72])[0])
    srow = [struct.unpack("<4f", header[offset : offset + 16]) for offset in (280, 296, 312)]
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(srow, dtype=np.float64)
    return shape, affine, datatype


def _read_nifti_gz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal dependency-free NIfTI-1 reader for the labels fixture -> (Z, Y, X), affine."""
    with gzip.open(str(path), "rb") as fh:
        blob = fh.read()
    header = blob[:352]
    dim = struct.unpack("<8h", header[40:56])
    ndim = int(dim[0])
    shape_xyz = tuple(int(v) for v in dim[1 : 1 + ndim])
    datatype = int(struct.unpack("<h", header[70:72])[0])
    vox_offset = int(round(struct.unpack("<f", header[108:112])[0]))
    srow = [struct.unpack("<4f", header[offset : offset + 16]) for offset in (280, 296, 312)]
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(srow, dtype=np.float64)
    dtypes = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16}
    data = np.frombuffer(blob[vox_offset:], dtype=dtypes[datatype], count=int(np.prod(shape_xyz))).reshape(
        shape_xyz, order="F"
    )
    return np.transpose(data, (2, 1, 0)).astype(np.int16), affine


def _straight_arch(z_mm: float = 25.0, jaw: str = "lower") -> DentalArch:
    return DentalArch(
        np.array(
            [[40.0, 25.0, z_mm], [30.0, 25.0, z_mm], [20.0, 25.0, z_mm], [10.0, 25.0, z_mm]],
            dtype=np.float64,
        ),
        jaw_type=jaw,
        smoothing=0.0,
        ras_convention=True,
    )


def _volume(mask: np.ndarray, spacing: float = 1.0) -> CBCTVolume:
    img = np.where(mask > 0, 900.0, -500.0).astype(np.float32)
    return CBCTVolume(
        image=img,
        spacing=(spacing, spacing, spacing),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        affine=np.diag([spacing, spacing, spacing, 1.0]),
    )


def _dentate_phantom(with_canal: bool = True) -> tuple[np.ndarray, DentalArch]:
    """Mandible crest at Z=27 with a tooth crown reaching Z=34 (7 mm above the crest)."""
    mask = np.zeros((45, 50, 50), dtype=np.int32)
    mask[10:28, 21:29, 10:40] = LOWER_JAW_ID  # 8 mm wide ridge, crest Z=27
    mask[26:35, 23:27, 23:28] = 46  # tooth crown/root reaching above the crest
    if with_canal:
        mask[12:14, 24:26, 10:40] = RIGHT_IAN_ID
    return mask, _straight_arch()


def _engine(mask: np.ndarray, arch: DentalArch | None = None, jaw: str = "lower") -> MeasurementEngine:
    use_arch = arch if arch is not None else _straight_arch(jaw=jaw)
    return MeasurementEngine(
        _volume(mask),
        mask,
        arch=use_arch,
        arches={jaw: use_arch},
        is_dental_segmentator_5class=False,
        extract_teeth_eagerly=False,
    )


def _arc_of_site(engine: MeasurementEngine, arch: DentalArch, fdi: str) -> float:
    center = engine.get_site_center_ras(fdi)
    assert center is not None
    t_param, _ = arch.find_closest_parameter(center)
    return float(t_param) * float(arch.total_arc_length)


# ---------------------------------------------------------------------------
# Entry point / W0 / height axis
# ---------------------------------------------------------------------------
class TestEntryPointAndW0(unittest.TestCase):
    def test_crest_and_w0_come_from_bone_not_the_tooth_crown(self) -> None:
        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        meas = engine.measure_implant_site("46")
        self.assertIsNotNone(meas)
        assert meas is not None
        meas_arc, cs = engine.measure_at_arc_mm("lower", _arc_of_site(engine, arch, "46"), fdi_hint="46")
        self.assertIsNotNone(meas_arc)
        assert cs is not None and cs.crest_uv is not None

        # The section plane is at arch height Z=25: the bone crest Z=27 must win. The tooth
        # crown reaches Z=34 and used to become the entry point (v~+9).
        frame_p, _tangent, _normal, binormal = cs.frame
        crest_world = np.asarray(frame_p) + float(cs.crest_uv[1]) * np.asarray(binormal)
        self.assertAlmostEqual(float(crest_world[2]), 27.0, delta=1.5)
        # W0 is bone only: the two cortical plates around the socket, never the 5 mm tooth.
        self.assertIsNotNone(meas.crest_width_mm)
        self.assertLess(float(meas.crest_width_mm or 0.0), 3.5)
        self.assertGreater(float(meas.crest_width_mm or 0.0), 0.0)

    def test_height_continues_to_jaw_end_without_a_nerve(self) -> None:
        with_canal, _arch = _dentate_phantom(with_canal=True)
        without_canal, _arch2 = _dentate_phantom(with_canal=False)
        with_meas = _engine(with_canal).measure_implant_site("46")
        without_meas = _engine(without_canal).measure_implant_site("46")
        self.assertIsNotNone(with_meas)
        self.assertIsNotNone(without_meas)
        assert with_meas is not None and without_meas is not None
        # Canal at Z~13 limits the height; without it the caliper runs to the jaw base (Z=10).
        self.assertLess(float(with_meas.bone_height_mm or 0.0), 15.0)
        self.assertGreater(float(without_meas.bone_height_mm or 0.0), 15.0)
        self.assertIsNone(without_meas.distance_to_vital_structure_mm)

    def test_height_axis_follows_the_pca_long_axis(self) -> None:
        mask = np.zeros((50, 50, 50), dtype=np.int32)
        for z in range(10, 29):
            shift = int(round((28 - z) * math.tan(math.radians(15.0))))
            mask[z, 20 + shift : 28 + shift, 10:40] = LOWER_JAW_ID
        engine = _engine(mask)
        meas, cs = engine.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None
        self.assertIsNotNone(cs.ridge_axis_angle_deg)
        self.assertGreater(abs(float(cs.ridge_axis_angle_deg or 0.0)), 5.0)
        record = cs.measurements["available_bone_height"]
        self.assertTrue(record.is_available)
        assert record.start_uv_mm is not None and record.end_uv_mm is not None
        # The caliper follows the tilted long axis, so the height line is not purely vertical.
        self.assertGreater(abs(record.end_uv_mm[0] - record.start_uv_mm[0]), 0.3)

    def test_entry_is_interpolated_across_slices(self) -> None:
        u = np.linspace(-15.0, 15.0, 121)
        v = np.linspace(-25.0, 25.0, 201)
        mask = np.zeros((len(v), len(u)), dtype=np.int32)
        items = []
        for index, crest_v in enumerate((5.0, 14.0, 5.0)):
            cross = SimpleNamespace(
                u_coords=u,
                v_coords=v,
                mask_cs=mask,
                img_cs=np.zeros_like(mask, dtype=np.float32),
                frame=(
                    np.zeros(3),
                    np.array([-1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ),
                crest_v=crest_v,
                width_segments={},
                canal_centroid_uv=None,
                crest_uv=(0.0, crest_v),
                measurements={},
            )
            cross.measurements["available_bone_height"] = MeasurementRecord(
                measurement_type="available_bone_height",
                value_mm=10.0,
                start_world_mm=(0.0, 0.0, crest_v),
                end_world_mm=(0.0, 0.0, -5.0),
                start_uv_mm=(0.0, crest_v),
                end_uv_mm=(0.0, -5.0),
                provenance="label_derived",
                quality="valid",
            )
            items.append((None, cross))
        engine = _engine(np.zeros((20, 20, 20), dtype=np.int32))
        engine.stabilize_entry_points(items)
        middle = items[1][1]
        self.assertAlmostEqual(float(middle.crest_uv[1]), 5.0, delta=0.01)
        record = middle.measurements["available_bone_height"]
        self.assertAlmostEqual(float(record.value_mm or 0.0), 10.0, delta=0.01)

        # A three-slice gallery over the real phantom never steps the entry more than the limit.
        mask3, arch3 = _dentate_phantom()
        eng3 = _engine(mask3, arch3)
        base = _arc_of_site(eng3, arch3, "46")
        series = [eng3.measure_at_arc_mm("lower", base + delta, fdi_hint="46") for delta in (-1.5, 0.0, 1.5)]
        eng3.stabilize_entry_points(series)
        crests = [float(cs.crest_uv[1]) for _m, cs in series if cs is not None and cs.crest_uv is not None]
        self.assertEqual(len(crests), 3)
        self.assertLessEqual(max(crests) - min(crests), MAX_ENTRY_JUMP_MM + 0.01)

    def test_printed_height_equals_drawn_height(self) -> None:
        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        meas, cs = engine.measure_at_arc_mm("lower", _arc_of_site(engine, arch, "46"), fdi_hint="46")
        assert meas is not None and cs is not None
        render = render_cross_section(cs, meas, scale=3)
        record = cs.measurements["available_bone_height"]
        exp0 = _px(record.start_uv_mm[0], record.start_uv_mm[1], cs.u_coords, cs.v_coords, 3)
        exp1 = _px(record.end_uv_mm[0], record.end_uv_mm[1], cs.u_coords, cs.v_coords, 3)
        self.assertTrue(
            any(
                math.hypot(seg[0][0] - exp0[0], seg[0][1] - exp0[1]) < 1e-3
                and math.hypot(seg[1][0] - exp1[0], seg[1][1] - exp1[1]) < 1e-3
                for seg in render.overlay_px
            )
        )
        drawn_mm = math.hypot(exp1[0] - exp0[0], exp1[1] - exp0[1]) * render.mm_per_px
        self.assertAlmostEqual(drawn_mm, float(meas.bone_height_mm or 0.0), delta=0.15)


# ---------------------------------------------------------------------------
# Rendering / grayscale film
# ---------------------------------------------------------------------------
class TestGrayscaleFilm(unittest.TestCase):
    def _shot(self):
        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        meas, cs = engine.measure_at_arc_mm("lower", _arc_of_site(engine, arch, "46"), fdi_hint="46")
        assert meas is not None and cs is not None
        return cs, meas

    def test_width_calipers_are_transparent_and_overlays_are_gone(self) -> None:
        self.assertEqual(WIDTH_ALPHA_W0, 255)
        self.assertTrue(0 < WIDTH_ALPHA_W3 < 255)
        self.assertTrue(0 < WIDTH_ALPHA_W5 < 255)
        cs, meas = self._shot()
        render = render_cross_section(cs, meas, scale=3)
        texts = [text for _x, _y, text in render.labels]
        self.assertFalse(any(text.startswith("Impl ") for text in texts), texts)
        self.assertFalse(any(text.startswith("Root ") for text in texts), texts)
        self.assertFalse(any("2mm SAFE" in text for text in texts), texts)
        self.assertFalse(any("Ang " in text for text in texts), texts)

    def test_monochrome_tile_is_true_grayscale(self) -> None:
        cs, meas = self._shot()
        render = render_cross_section(cs, meas, scale=2, monochrome=True)
        self.assertTrue(render.available)
        pixels = np.asarray(render.image)
        self.assertTrue(np.array_equal(pixels[:, :, 0], pixels[:, :, 1]))
        self.assertTrue(np.array_equal(pixels[:, :, 1], pixels[:, :, 2]))
        color = render_cross_section(cs, meas, scale=2)
        self.assertFalse(np.array_equal(np.asarray(color.image)[:, :, 0], np.asarray(color.image)[:, :, 1]))

    def test_grayscale_print_layout_and_span_summary(self) -> None:
        from CBCTPlanner import CBCTPlannerLogic

        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        logic = object.__new__(CBCTPlannerLogic)
        logic._get_or_build_measurement_engine = lambda _v, _s: engine  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp:
            html_file, _pdf, specs = PrintLayoutEngine(logic=logic).generate_8x10_layout(
                volumeNode=object(),
                segmentationNode=None,
                selected_fdis=[46],
                out_dir=Path(tmp),
                grayscale=True,
            )
            html_text = Path(html_file).read_text(encoding="utf-8")
            self.assertIn("bottom-section { display: none", html_text)
            self.assertIn("filter: grayscale(1)", html_text)
            self.assertTrue(any(spec.height_mm is not None for spec in specs))
            from PIL import Image

            pngs = list(Path(tmp).glob("*.png"))
            self.assertTrue(pngs)
            pixels = np.asarray(Image.open(pngs[0]).convert("RGB"))
            self.assertTrue(np.array_equal(pixels[:, :, 0], pixels[:, :, 1]))
            self.assertTrue(np.array_equal(pixels[:, :, 1], pixels[:, :, 2]))
        span = summarize_span_worst_case([engine.measure_implant_site("46")])
        self.assertEqual(span["sites"], 1)
        self.assertIsNotNone(span["min_height_mm"])
        self.assertEqual(span["limiting_height_site"], "46")


# ---------------------------------------------------------------------------
# Typed jaw keys + anti-pattern scan + orientation default
# ---------------------------------------------------------------------------
class TestTypedJawKeys(unittest.TestCase):
    def test_parse_jaw_key_is_strict_when_asked(self) -> None:
        self.assertEqual(parse_jaw_key("Mandible"), "lower")
        self.assertEqual(parse_jaw_key("Maxilla & Upper Skull"), "upper")
        self.assertIsNone(parse_jaw_key("Shared"))
        with self.assertRaises(ValueError):
            parse_jaw_key("Shared", strict=True)
        self.assertEqual(jaw_key_from_text("Lower"), "lower")

    def test_no_engine_call_receives_a_display_name(self) -> None:
        pattern = re.compile(
            r"(?:measure_at_arc_mm|fit_arch_from_label_mask|compute_fdi_arc_positions_mm)"
            r"\([^)\n]*(?:jaw[^=,)]*=\s*)?[\"'](Mandible|Maxilla|Lower|Upper|Shared)[\"']"
        )
        offenders = []
        for path in (ROOT / "CBCTPlanner").rglob("*.py"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{line_number}: {line.strip()}")
        self.assertEqual(offenders, [], "display jaw names passed to canonical-key APIs")

    def test_ras_auto_standardization_is_the_persisted_default(self) -> None:
        source = (ROOT / "CBCTPlanner" / "CBCTPlanner.py").read_text(encoding="utf-8")
        self.assertIn('"autoStandardizeOnLoadRas", "true"', source)
        self.assertNotIn('"autoStandardizeOnLoad", "false"', source)


# ---------------------------------------------------------------------------
# Real ToothFairy3 fixture (dependency-free NIfTI read)
# ---------------------------------------------------------------------------
class TestRealToothFairyFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE.exists():
            raise unittest.SkipTest(f"missing fixture {FIXTURE}")
        cls.mask_raw, cls.affine_raw = _read_nifti_gz(FIXTURE)

    def test_fixture_is_a_canonical_toothfairy_labelmap(self) -> None:
        shape_xyz, affine, datatype = _nifti_header(FIXTURE)
        self.assertEqual(datatype, 2)
        self.assertEqual(tuple(reversed(shape_xyz)), self.mask_raw.shape)
        self.assertAlmostEqual(float(abs(affine[0, 0])), 0.3, places=3)
        self.assertFalse(infer_mask_is_5class(self.mask_raw))
        values = set(int(v) for v in np.unique(self.mask_raw))
        self.assertIn(46, values, "expected zipped ToothFairy3 FDI labels")
        self.assertIn(103, values, "expected zipped ToothFairy3 canal labels")

    def test_orientation_arch_fit_and_measurement_on_real_labels(self) -> None:
        canonical = canonicalize_to_ras(self.mask_raw, self.affine_raw)
        self.assertEqual(axcodes_from_affine(canonical.affine), ("R", "A", "S"))
        mask = canonical.image_zyx.astype(np.int32)

        fit = fit_arch_from_label_mask(mask, canonical.affine, jaw="Mandible", is_5class=False)
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertTrue(fit.accepted, fit.reason)
        arch = DentalArch(fit.points_ras, jaw_type="lower", ras_convention=True)
        spacing = float(np.linalg.norm(canonical.affine[:3, 2]))
        origin = tuple(float(v) for v in canonical.affine[:3, 3])
        direction = canonical.affine[:3, :3] / np.maximum(
            np.linalg.norm(canonical.affine[:3, :3], axis=0), 1e-8
        )
        image = np.where(mask > 0, 900.0, -500.0).astype(np.float32)
        vol = CBCTVolume(image, (spacing, spacing, spacing), origin, direction, canonical.affine)
        engine = MeasurementEngine(
            vol,
            mask,
            arch=arch,
            arches={"lower": arch},
            is_dental_segmentator_5class=False,
            extract_teeth_eagerly=False,
        )
        present = {int(v) for v in np.unique(mask)}
        fdi = 46 if 46 in present else next(v for v in (36, 47, 45, 35, 44) if v in present)
        arc = _arc_of_site(engine, arch, str(fdi))
        meas, cs = engine.measure_at_arc_mm("lower", arc, fdi_hint=str(fdi))
        self.assertIsNotNone(cs, "a real mandibular cross-section must resample")
        assert cs is not None
        if meas is not None:
            self.assertIsNotNone(meas.bone_height_mm)
            self.assertGreater(float(meas.bone_height_mm or 0.0), 0.0)
            render = render_cross_section(cs, meas, scale=2, monochrome=True)
            self.assertTrue(render.available)
        self.assertTrue(mask.shape[0] > 0)

    def test_real_volume_file_matches_the_label_grid(self) -> None:
        if not REAL_IMAGE.exists():
            self.skipTest("real ToothFairy3 image not available on this machine")
        image_shape, image_affine, _datatype = _nifti_header(REAL_IMAGE)
        label_shape, label_affine, _ = _nifti_header(FIXTURE)
        self.assertEqual(image_shape, label_shape)
        self.assertTrue(np.allclose(image_affine, label_affine, atol=1e-5))


# ---------------------------------------------------------------------------
# Diagnostics, calibration, provenance, span
# ---------------------------------------------------------------------------
class TestClinicalDiagnostics(unittest.TestCase):
    def test_preflight_checklist_and_text_report(self) -> None:
        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        checks = build_preflight_checklist(engine, [46])
        names = {check["check"] for check in checks}
        self.assertTrue({"calibration", "segmentation", "lower_arch", "site_46"} <= names)
        self.assertTrue(next(c for c in checks if c["check"] == "calibration")["ok"])
        self.assertTrue(next(c for c in checks if c["check"] == "site_46")["ok"])
        text = checklist_text(checks)
        self.assertIn("preflight:", text)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_diagnostics_text(engine, Path(tmp) / "diag.txt", [46])
            content = out.read_text(encoding="utf-8")
            self.assertIn("site: #46", content)
            self.assertIn("provenance_hash:", content)

    def test_calibration_guard_flags_implausible_spacing(self) -> None:
        good = SimpleNamespace(spacing=(0.3, 0.3, 0.3))
        bad = SimpleNamespace(spacing=(5.0, 5.0, 5.0))
        missing = SimpleNamespace(spacing=(0.0, 0.3, 0.3))
        self.assertEqual(calibration_status(good)[0], "ok")
        self.assertEqual(calibration_status(bad)[0], "unverified")
        self.assertEqual(calibration_status(missing)[0], "unverified")
        self.assertIn("UNVERIFIED", calibration_stamp(bad))
        self.assertIn("SCALE OK", calibration_stamp(good))

    def test_provenance_hash_is_stable_and_value_sensitive(self) -> None:
        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        meas, cs = engine.measure_at_arc_mm("lower", _arc_of_site(engine, arch, "46"), fdi_hint="46")
        assert meas is not None and cs is not None
        first = provenance_hash(meas)
        self.assertEqual(first, provenance_hash(meas))
        record = cs.measurements["available_bone_height"]
        record.value_mm = float(record.value_mm or 0.0) + 0.5
        self.assertNotEqual(first, provenance_hash(meas))
        report = measurement_report_text(meas)
        self.assertIn("height_mm:", report)
        self.assertIn("crest_width_mm:", report)
        self.assertIn("withheld: none", report)

    def test_span_worst_case_uses_the_limiting_site(self) -> None:
        rows = [
            SimpleNamespace(
                tooth_fdi="46",
                bone_height_mm=16.0,
                crest_width_mm=7.0,
                suggested_diameter_mm=4.0,
                suggested_length_mm=10.0,
                withheld_reason=None,
            ),
            SimpleNamespace(
                tooth_fdi="47",
                bone_height_mm=11.0,
                crest_width_mm=5.0,
                suggested_diameter_mm=3.0,
                suggested_length_mm=8.0,
                withheld_reason=None,
            ),
            SimpleNamespace(
                tooth_fdi="48",
                bone_height_mm=None,
                crest_width_mm=None,
                suggested_diameter_mm=None,
                suggested_length_mm=None,
                withheld_reason="no bone",
            ),
        ]
        span = summarize_span_worst_case(rows)
        self.assertEqual(span["sites"], 3)
        self.assertEqual(span["min_height_mm"], 11.0)
        self.assertEqual(span["limiting_height_site"], "47")
        self.assertEqual(span["max_span_diameter_mm"], 3.0)
        self.assertEqual(span["withheld"], [("48", "no bone")])

    def test_one_click_resample_method_is_wired(self) -> None:
        from CBCTPlanner import CBCTPlannerLogic

        self.assertTrue(hasattr(CBCTPlannerLogic, "resample_segmentation_to_volume"))
        logic = CBCTPlannerLogic()
        self.assertEqual(logic.resample_segmentation_to_volume(None, None), 0)


# ---------------------------------------------------------------------------
# Scheme detection, caches, shared arch, randomized invariants
# ---------------------------------------------------------------------------
class TestHardening(unittest.TestCase):
    def test_scheme_detection_uses_canonical_evidence(self) -> None:
        base = np.zeros((3, 3, 3), dtype=np.int32)
        five = base.copy()
        five[0, 0, 0] = 1
        five[0, 0, 1] = 2
        five[0, 0, 2] = 3
        five[1, 0, 0] = 4
        five[1, 0, 1] = 5
        self.assertTrue(infer_mask_is_5class(five))
        with_sinus = five.copy()
        with_sinus[1, 0, 2] = 6
        self.assertFalse(infer_mask_is_5class(with_sinus))
        with_tooth = five.copy()
        with_tooth[1, 1, 0] = 46
        self.assertFalse(infer_mask_is_5class(with_tooth))
        with_canal = five.copy()
        with_canal[1, 1, 1] = 103
        self.assertFalse(infer_mask_is_5class(with_canal))

    def test_arch_change_invalidates_the_arc_cache(self) -> None:
        mask, arch = _dentate_phantom()
        engine = _engine(mask, arch)
        engine.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertTrue(engine._arc_cache)
        engine.set_arches({"lower": arch})
        self.assertFalse(engine._arc_cache)
        self.assertIs(engine._arch_for_jaw(True), arch)

    def test_shared_single_arch_falls_back_for_both_jaws(self) -> None:
        mask = np.zeros((30, 50, 50), dtype=np.int32)
        mask[10:26, 21:29, 10:40] = LOWER_JAW_ID
        arch = _straight_arch()
        engine = MeasurementEngine(
            _volume(mask),
            mask,
            arch=arch,
            arches={},
            is_dental_segmentator_5class=False,
            extract_teeth_eagerly=False,
        )
        self.assertIs(engine._arch_for_jaw(True), arch)
        self.assertIs(engine._arch_for_jaw(False), arch)
        meas, cs = engine.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(cs)
        if meas is not None:
            self.assertIsNotNone(meas.bone_height_mm)

    def test_randomized_ridge_invariants_hold_across_views(self) -> None:
        rng = np.random.default_rng(20261123)
        for _ in range(4):
            height_mm = int(rng.integers(12, 20))
            width_mm = int(rng.integers(5, 11))
            z0 = 6
            y0 = 25 - width_mm // 2
            mask = np.zeros((45, 50, 50), dtype=np.int32)
            mask[z0 : z0 + height_mm, y0 : y0 + width_mm, 10:40] = LOWER_JAW_ID
            engine = _engine(mask)
            meas, cs = engine.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
            self.assertIsNotNone(cs)
            if meas is None or cs is None or cs.crest_uv is None:
                continue
            record = cs.measurements["available_bone_height"]
            if not record.is_available:
                continue
            self.assertAlmostEqual(float(record.value_mm or 0.0), float(meas.bone_height_mm or 0.0), places=6)
            render = render_cross_section(cs, meas, scale=2)
            exp0 = _px(record.start_uv_mm[0], record.start_uv_mm[1], cs.u_coords, cs.v_coords, 2)
            exp1 = _px(record.end_uv_mm[0], record.end_uv_mm[1], cs.u_coords, cs.v_coords, 2)
            self.assertTrue(
                any(
                    math.hypot(seg[0][0] - exp0[0], seg[0][1] - exp0[1]) < 1e-3
                    and math.hypot(seg[1][0] - exp1[0], seg[1][1] - exp1[1]) < 1e-3
                    for seg in render.overlay_px
                )
            )
            if meas.crest_width_mm is not None:
                self.assertGreater(float(meas.crest_width_mm), 0.0)
                self.assertLess(float(meas.crest_width_mm), float(width_mm) + 2.0)


if __name__ == "__main__":
    unittest.main()

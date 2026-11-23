"""Comprehensive anatomy-aware measurement, rendering, UI helper, and safety test suite.

Covers all 18 validation scenarios in Part 9 (headless, no live 3D Slicer required):
  1. Sloped mandibular ridge
  2. Curved alveolar ridge
  3. Oblique jaw orientation
  4. Non-isotropic voxel spacing
  5. Isolated stray bone voxels
  6. Disconnected bone islands
  7. Missing jaw segmentation (no intensity fallback)
  8. Missing canal segmentation
  9. Unrelated or contralateral canal elsewhere in the volume
 10. Individual FDI tooth labels
 11. Different segmentation insertion orders
 12. Root apex distinct from available-bone endpoint (drawn length == displayed value)
 13. Tilted or asymmetric buccolingual cortical plates
 14. Correct line orientation after image flipping
 15. Consistency between GUI, gallery, and print measurements
 16. Consistency of implant-size suggestions across views
 17. Surgical-guide export rejection without a valid arch
 18. Cache invalidation after segmentation modifications
Plus WorkflowUIHelpers tests (workspace roles, single-active-slice gallery, RISE-MAR validation).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CBCTPlanner"))

from CBCTPlannerLib.WorkflowUIHelpers import (
    GalleryNavigationState,
    apply_risemar_validation_to_widget,
    apply_workspace_role,
    compute_volume_segmentation_cache_token,
)
from CBCTPlannerLib.RiseMarEngine import DEFAULT_RISEMAR_WEIGHTS_PATH, validate_risemar_weights_path
from CBCTPlannerLib.core.arch_detector import DentalArch, validate_arch_frame
from CBCTPlannerLib.core.labels import (
    LEFT_IAN_ID,
    LEFT_SINUS_ID,
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    RIGHT_SINUS_ID,
    UPPER_JAW_ID,
    combine_labeled_segment_masks,
    get_ipsilateral_canal_ids,
    get_ipsilateral_sinus_ids,
    is_mandibular_fdi,
    resolve_fdi_number,
)
from CBCTPlannerLib.core.measurements import (
    IMPLANT_APICAL_SAFETY_MARGIN_MM,
    MeasurementEngine,
    PROVENANCE_LABEL_DERIVED,
    WITHHELD_NO_CANAL_SEG,
    WITHHELD_NO_MANDIBLE_SEG,
    suggest_implant_dimensions,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume
from CBCTPlannerLib.core_print_render import _px, render_cross_section


def _make_straight_arch(y_mm: float = 25.0, z_mm: float = 25.0, jaw: str = "lower") -> DentalArch:
    pts = np.array(
        [[40.0, y_mm, z_mm], [30.0, y_mm, z_mm], [20.0, y_mm, z_mm], [10.0, y_mm, z_mm]],
        dtype=np.float64,
    )
    return DentalArch(pts, jaw_type=jaw, smoothing=0.0, ras_convention=True)


def _make_curved_arch(jaw: str = "lower", z_mm: float = 25.0) -> DentalArch:
    angles = np.linspace(0.15 * np.pi, 0.85 * np.pi, 11)
    cx, cy, radius = 30.0, 12.0, 18.0
    pts = np.column_stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles), np.full_like(angles, z_mm)]
    )
    return DentalArch(pts, jaw_type=jaw, smoothing=0.0, ras_convention=True)


def _verify_rendered_measurement_sync(
    test: unittest.TestCase,
    cs,
    meas,
    scale: int = 3,
) -> None:
    """Verify every valid measurement's physical distance matches its value and rendered line."""
    render = render_cross_section(cs, meas, scale=scale)
    test.assertTrue(render.available)
    u_coords = np.asarray(cs.u_coords, dtype=np.float64)
    v_coords = np.asarray(cs.v_coords, dtype=np.float64)

    h_rec = cs.measurements.get("available_bone_height")
    if h_rec is not None and h_rec.is_available:
        p0_w = np.asarray(h_rec.start_world_mm, dtype=np.float64)
        p1_w = np.asarray(h_rec.end_world_mm, dtype=np.float64)
        phys_dist = float(np.linalg.norm(p1_w - p0_w))
        test.assertAlmostEqual(phys_dist, float(h_rec.value_mm), places=1)
        test.assertAlmostEqual(phys_dist, float(meas.bone_height_mm), places=1)

        # Rendered height line must connect the exact pixel projection of start_uv_mm -> end_uv_mm
        exp_p0 = _px(h_rec.start_uv_mm[0], h_rec.start_uv_mm[1], u_coords, v_coords, scale)
        exp_p1 = _px(h_rec.end_uv_mm[0], h_rec.end_uv_mm[1], u_coords, v_coords, scale)
        found = any(
            math.hypot(seg[0][0] - exp_p0[0], seg[0][1] - exp_p0[1]) < 1e-3
            and math.hypot(seg[1][0] - exp_p1[0], seg[1][1] - exp_p1[1]) < 1e-3
            for seg in render.overlay_px
        )
        test.assertTrue(found, f"Height line {(exp_p0, exp_p1)} missing from overlay_px={render.overlay_px}")
        # Displayed numerical value matches the physical & rendered length
        rendered_px_len = math.hypot(exp_p1[0] - exp_p0[0], exp_p1[1] - exp_p0[1])
        rendered_mm_len = rendered_px_len * render.mm_per_px
        test.assertAlmostEqual(rendered_mm_len, float(meas.bone_height_mm), delta=0.15)


class TestAnatomyAwareMeasurements(unittest.TestCase):
    def _build_engine(
        self,
        mask: np.ndarray,
        img: np.ndarray,
        *,
        affine: np.ndarray | None = None,
        arch: DentalArch | None = None,
        jaw: str = "lower",
        is_5class: bool = False,
    ) -> MeasurementEngine:
        aff = np.eye(4, dtype=np.float64) if affine is None else affine
        col_norms = np.linalg.norm(aff[:3, :3], axis=0)
        spacing = (float(col_norms[2]), float(col_norms[1]), float(col_norms[0]))
        origin = tuple(float(v) for v in aff[:3, 3])
        direction = aff[:3, :3] / np.where(col_norms > 1e-9, col_norms, 1.0)[None, :]
        vol = CBCTVolume(image=img, spacing=spacing, origin=origin, direction=direction, affine=aff)
        use_arch = arch if arch is not None else _make_straight_arch(jaw=jaw)
        return MeasurementEngine(
            vol,
            mask,
            arch=use_arch,
            arches={jaw: use_arch},
            is_dental_segmentator_5class=is_5class,
            extract_teeth_eagerly=False,
        )

    # 1. Sloped mandibular ridge
    def test_01_sloped_mandibular_ridge(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        # Mandibular ridge whose crest slopes upward from lingual (Y=21) to buccal (Y=28)
        for y in range(21, 29):
            z_top = 24 + int(round(0.6 * (y - 21)))  # Z=24..28 sloped crest
            mask[10 : z_top + 1, y, 10:40] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 900.0

        eng = self._build_engine(mask, img)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        self.assertIsNotNone(cs)
        assert meas is not None and cs is not None
        self.assertIsNotNone(cs.crest_profile_uv)
        self.assertGreaterEqual(len(cs.crest_profile_uv or []), 4)
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 16.0, delta=2.5)
        # At the sloped peak W0 is narrower than W5 (full ridge width 5 mm apically)
        self.assertIsNotNone(meas.crest_width_mm)
        w5 = meas.bone_width_at_depths_mm.get(5.0)
        self.assertIsNotNone(w5)
        self.assertLess(float(meas.crest_width_mm), float(w5))
        self.assertAlmostEqual(float(w5), 7.5, delta=1.5)
        _verify_rendered_measurement_sync(self, cs, meas)

    # 2. Curved alveolar ridge
    def test_02_curved_alveolar_ridge(self) -> None:
        shape = (50, 60, 60)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        arch = _make_curved_arch(jaw="lower", z_mm=22.0)
        zz, yy, xx = np.indices(shape, dtype=np.float64)
        radial = np.sqrt((xx - 30.0) ** 2 + (yy - 12.0) ** 2)
        in_ring = (np.abs(radial - 18.0) <= 3.5) & (yy >= 12.0) & (zz >= 12.0) & (zz <= 28.0)
        mask[in_ring] = LOWER_JAW_ID
        img[in_ring] = 880.0

        eng = self._build_engine(mask, img, arch=arch)
        frame = arch.get_frame(0.5)
        ok, issues = validate_arch_frame(frame)
        self.assertTrue(ok, issues)

        meas, cs = eng.measure_at_arc_mm("lower", 0.5 * arch.total_arc_length, fdi_hint="41")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 16.0, delta=1.5)
        self.assertAlmostEqual(meas.crest_width_mm or 0.0, 7.0, delta=1.5)
        _verify_rendered_measurement_sync(self, cs, meas)

    # 3. Oblique jaw orientation
    def test_03_oblique_jaw_orientation(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        mask[12:30, 21:29, 10:40] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 900.0

        # 18-degree rotation around X axis in affine
        theta = math.radians(18.0)
        c, s = math.cos(theta), math.sin(theta)
        aff = np.eye(4, dtype=np.float64)
        aff[1, 1], aff[1, 2] = c, -s
        aff[2, 1], aff[2, 2] = s, c

        # Arch control points transformed by the same oblique affine
        vox_pts = np.array([[40, 25, 25, 1], [30, 25, 25, 1], [20, 25, 25, 1], [10, 25, 25, 1]], dtype=np.float64)
        world_pts = (aff @ vox_pts.T).T[:, :3]
        occlusal_n = aff[:3, 2]
        arch = DentalArch(world_pts, jaw_type="lower", occlusal_normal=occlusal_n, ras_convention=True)
        ok, issues = validate_arch_frame(arch.get_frame(0.5))
        self.assertTrue(ok, issues)

        eng = self._build_engine(mask, img, affine=aff, arch=arch)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 17.5, delta=1.5)
        self.assertAlmostEqual(meas.crest_width_mm or 0.0, 7.5, delta=1.5)
        _verify_rendered_measurement_sync(self, cs, meas)

    # 4. Non-isotropic voxel spacing
    def test_04_non_isotropic_voxel_spacing(self) -> None:
        shape = (40, 80, 80)  # Z=0.8mm, Y=0.4mm, X=0.4mm
        aff = np.diag([0.4, 0.4, 0.8, 1.0])
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        # Physical Z: 8..28 => 16.0 mm tall; Physical Y: 12..20 => 8.0 mm wide
        mask[10:30, 30:50, 15:65] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 920.0

        arch = _make_straight_arch(y_mm=16.0, z_mm=16.0, jaw="lower")
        # Keep arch inside X=[6..26]
        arch = DentalArch(
            np.array([[24.0, 16.0, 16.0], [18.0, 16.0, 16.0], [12.0, 16.0, 16.0], [8.0, 16.0, 16.0]]),
            jaw_type="lower",
            ras_convention=True,
        )
        eng = self._build_engine(mask, img, affine=aff, arch=arch)
        meas, cs = eng.measure_at_arc_mm("lower", 8.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 15.5, delta=1.2)
        self.assertAlmostEqual(meas.crest_width_mm or 0.0, 7.8, delta=1.0)
        _verify_rendered_measurement_sync(self, cs, meas)

    # 5 & 6. Isolated stray bone voxels & disconnected bone islands
    def test_05_06_stray_voxels_and_disconnected_islands_rejected(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        # Clean ridge: Z=12..28 (16 mm tall), Y=21..29 (8 mm wide)
        mask[12:29, 21:29, 10:40] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 900.0

        eng_clean = self._build_engine(mask.copy(), img)
        meas_clean, _ = eng_clean.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        assert meas_clean is not None

        # Inject isolated stray voxels above crest AND a disconnected lateral bone island
        noisy = mask.copy()
        noisy[38, 25, 24:27] = LOWER_JAW_ID           # isolated stray voxels 10 mm above crest
        noisy[15:25, 11:15, 10:40] = LOWER_JAW_ID     # disconnected lingual island 6 mm away
        noisy[15:25, 34:37, 10:40] = LOWER_JAW_ID     # disconnected buccal island 5 mm away

        eng_noisy = self._build_engine(noisy, img)
        meas_noisy, cs_noisy = eng_noisy.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas_noisy)
        assert meas_noisy is not None and cs_noisy is not None
        # Neither stray voxels nor disconnected islands may change height or width!
        self.assertAlmostEqual(meas_noisy.bone_height_mm or 0.0, meas_clean.bone_height_mm or 0.0, delta=0.5)
        self.assertAlmostEqual(meas_noisy.crest_width_mm or 0.0, meas_clean.crest_width_mm or 0.0, delta=0.5)
        _verify_rendered_measurement_sync(self, cs_noisy, meas_noisy)

    # 7. Missing jaw segmentation (must withhold, never substitute intensity threshold)
    def test_07_missing_jaw_segmentation_withholds_no_intensity_fallback(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        img[10:30, 21:29, 10:40] = 1100.0  # bright bone intensities, but NO jaw label
        mask = np.zeros(shape, dtype=np.int32)

        eng = self._build_engine(mask, img)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNone(meas)
        self.assertIsNotNone(cs)
        assert cs is not None
        self.assertEqual(cs.withheld_reason, WITHHELD_NO_MANDIBLE_SEG)
        render = render_cross_section(cs, None, scale=2)
        self.assertEqual(render.overlay_px, [])

    # 8. Missing canal segmentation
    def test_08_missing_canal_segmentation_marks_vital_unavailable(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        mask[10:28, 21:29, 10:40] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 900.0

        eng = self._build_engine(mask, img)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None
        self.assertIsNone(meas.distance_to_vital_structure_mm)
        self.assertEqual(cs.vital_withheld_reason, WITHHELD_NO_CANAL_SEG)
        self.assertIsNone(cs.canal_centroid_uv)
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 17.5, delta=1.0)

    # 9. Unrelated or contralateral canal elsewhere in the volume
    def test_09_contralateral_or_distant_canal_rejected(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        mask[10:28, 21:29, 10:40] = LOWER_JAW_ID
        # Place ONLY LEFT_IAN_ID (quadrant 3) while measuring FDI #46 (quadrant 4 = RIGHT)!
        mask[15:18, 24:27, 10:40] = LEFT_IAN_ID
        img[mask > 0] = 900.0

        eng = self._build_engine(mask, img)
        meas_right, cs_right = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas_right)
        assert meas_right is not None and cs_right is not None
        self.assertIsNone(meas_right.distance_to_vital_structure_mm)
        self.assertIsNone(cs_right.canal_centroid_uv)
        # Now measure FDI #36 (quadrant 3 = LEFT) — ipsilateral LEFT_IAN_ID IS selected!
        meas_left, cs_left = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="36")
        self.assertIsNotNone(meas_left)
        assert meas_left is not None and cs_left is not None
        self.assertIsNotNone(meas_left.distance_to_vital_structure_mm)
        self.assertIsNotNone(cs_left.canal_centroid_uv)
        self.assertLess(meas_left.bone_height_mm or 99.0, meas_right.bone_height_mm or 0.0)

    # 10 & 11. Individual FDI tooth labels & different segmentation insertion orders
    def test_10_11_fdi_registry_and_insertion_order_independence(self) -> None:
        self.assertEqual(resolve_fdi_number("#46"), "46")
        self.assertEqual(resolve_fdi_number(146), "46")
        self.assertTrue(is_mandibular_fdi("36"))
        self.assertFalse(is_mandibular_fdi("16"))
        self.assertEqual(get_ipsilateral_canal_ids("46"), [RIGHT_IAN_ID])
        self.assertEqual(get_ipsilateral_canal_ids("36"), [LEFT_IAN_ID])
        self.assertEqual(get_ipsilateral_canal_ids("16"), [])
        self.assertEqual(get_ipsilateral_sinus_ids("16"), [RIGHT_SINUS_ID])
        self.assertEqual(get_ipsilateral_sinus_ids("26"), [LEFT_SINUS_ID])
        self.assertEqual(get_ipsilateral_sinus_ids("46"), [])

        # Combine overlapping jaw, canal, tooth, and pulp in two reversed insertion orders
        shape = (10, 10, 10)
        jaw_b = np.ones(shape, dtype=bool)
        canal_b = np.zeros(shape, dtype=bool)
        canal_b[2:4, 4:6, :] = True
        tooth_b = np.zeros(shape, dtype=bool)
        tooth_b[4:8, 3:7, 3:7] = True
        pulp_b = np.zeros(shape, dtype=bool)
        pulp_b[5:7, 4:6, 4:6] = True

        segs_a = [
            ("s1", "Mandible", jaw_b, None, None),
            ("s2", "Right Inferior Alveolar Canal", canal_b, None, None),
            ("s3", "Tooth 46", tooth_b, "46", None),
            ("s4", "Pulp 46", pulp_b, "46", None),
        ]
        segs_b = list(reversed(segs_a))
        comb_a, is5_a = combine_labeled_segment_masks(segs_a)
        comb_b, is5_b = combine_labeled_segment_masks(segs_b)
        self.assertFalse(is5_a)
        self.assertFalse(is5_b)
        self.assertTrue(np.array_equal(comb_a, comb_b))
        # Verify precedence: pulp (146) > tooth (46) > canal (4) > jaw (1)
        self.assertEqual(int(comb_a[5, 4, 4]), 146)
        self.assertEqual(int(comb_a[4, 3, 3]), 46)
        self.assertEqual(int(comb_a[2, 4, 4]), RIGHT_IAN_ID)
        self.assertEqual(int(comb_a[0, 0, 0]), LOWER_JAW_ID)

    # 12. Root apex distinct from available-bone endpoint (regression test!)
    def test_12_root_apex_distinct_from_available_bone_height(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        # Mandible Z=8..28 (20 mm tall), tooth #46 root apex stops at Z=18 (10 mm below crest)
        mask[8:29, 21:29, 10:40] = LOWER_JAW_ID
        mask[18:34, 23:27, 22:28] = 46
        # Right IAN canal at Z=11..13 (crest-to-canal ~ 14.5 mm, distinct from root apex ~ 10 mm!)
        mask[11:14, 24:26, 10:40] = RIGHT_IAN_ID
        img[mask > 0] = 900.0

        eng = self._build_engine(mask, img)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None

        h_rec = cs.measurements["available_bone_height"]
        r_rec = cs.measurements["root_extent"]
        self.assertTrue(h_rec.is_available)
        self.assertTrue(r_rec.is_available)
        # Available bone height (~14.5 mm to canal) MUST NOT be replaced by root extent (~10 mm)!
        self.assertGreater(float(h_rec.value_mm), float(r_rec.value_mm) + 2.5)
        self.assertNotAlmostEqual(float(h_rec.end_uv_mm[1]), float(r_rec.end_uv_mm[1]), delta=2.0)
        _verify_rendered_measurement_sync(self, cs, meas)

    # 13. Tilted or asymmetric buccolingual cortical plates
    def test_13_tilted_buccolingual_cortical_plates(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        # Ridge tilted 15 degrees buccally as it descends apically
        for z in range(10, 29):
            shift = int(round((28 - z) * math.tan(math.radians(15.0))))
            y0 = 20 + shift
            y1 = 27 + shift
            mask[z, y0:y1, 10:40] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 900.0

        eng = self._build_engine(mask, img)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        self.assertIsNotNone(meas)
        assert meas is not None and cs is not None
        self.assertIsNotNone(cs.ridge_axis_angle_deg)
        self.assertGreater(abs(float(cs.ridge_axis_angle_deg)), 5.0)
        # Width endpoints must adapt to the tilted ridge (non-zero delta-v across caliper!)
        self.assertIn(3.0, cs.width_endpoints_uv)
        p_l, p_r = cs.width_endpoints_uv[3.0]
        self.assertGreater(abs(p_r[1] - p_l[1]), 0.3)
        _verify_rendered_measurement_sync(self, cs, meas)

    # 14. Correct line orientation after image flipping
    def test_14_line_orientation_after_v_flip(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        mask[12:29, 21:29, 10:40] = LOWER_JAW_ID
        img[mask == LOWER_JAW_ID] = 950.0

        eng = self._build_engine(mask, img)
        meas, cs = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        assert meas is not None and cs is not None
        r_asc = render_cross_section(cs, meas, scale=2)

        # Flip v_coords + img_cs + mask_cs together — rendered pixels & overlay_px must match!
        cs_flip = SimpleNamespace(**cs.__dict__)
        cs_flip.v_coords = cs.v_coords[::-1].copy()
        cs_flip.img_cs = cs.img_cs[::-1, :].copy()
        cs_flip.mask_cs = cs.mask_cs[::-1, :].copy()
        r_desc = render_cross_section(cs_flip, meas, scale=2)
        self.assertEqual(r_asc.overlay_px, r_desc.overlay_px)

    # 15 & 16. Consistency across GUI, gallery, print, and implant-size suggestions
    def test_15_16_consistency_across_views_and_implant_suggestions(self) -> None:
        shape = (50, 50, 50)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        mask[10:28, 21:29, 10:40] = LOWER_JAW_ID
        mask[22:32, 23:27, 23:27] = 46
        img[mask > 0] = 950.0

        eng = self._build_engine(mask, img)
        meas_gui = eng.measure_implant_site("46")
        self.assertIsNotNone(meas_gui)
        assert meas_gui is not None
        center = eng.get_site_center_ras("46")
        assert center is not None
        t_param, _ = eng.arch.find_closest_parameter(center)
        arc_mm = t_param * eng.arch.total_arc_length
        meas_gal, cs_gal = eng.measure_at_arc_mm("lower", arc_mm, fdi_hint="46")
        assert meas_gal is not None and cs_gal is not None

        # GUI and gallery measurements agree closely
        self.assertAlmostEqual(meas_gui.bone_height_mm or 0.0, meas_gal.bone_height_mm or 0.0, delta=0.75)
        self.assertAlmostEqual(meas_gui.crest_width_mm or 0.0, meas_gal.crest_width_mm or 0.0, delta=0.75)

        # Implant suggestion in engine == suggest_implant_dimensions == rendered overlay text!
        exp_d, exp_l, _ = suggest_implant_dimensions(
            meas_gal.crest_width_mm, meas_gal.mid_root_width_mm, meas_gal.bone_height_mm, meas_gal.mesiodistal_space_mm
        )
        self.assertEqual(meas_gal.suggested_diameter_mm, exp_d)
        self.assertEqual(meas_gal.suggested_length_mm, exp_l)
        self.assertEqual(cs_gal.suggested_implant_diameter_mm, exp_d)
        self.assertEqual(cs_gal.suggested_implant_length_mm, exp_l)
        if exp_d is not None and exp_l is not None:
            r = render_cross_section(cs_gal, meas_gal, scale=3)
            self.assertTrue(
                any(f"Impl {exp_d:.1f}x{exp_l:g} mm" in txt for _, _, txt in r.labels),
                r.labels,
            )

    # 17. Surgical-guide export rejection without a valid arch
    def test_17_surgical_guide_export_rejects_missing_arch(self) -> None:
        from CBCTPlanner import CBCTPlannerLogic

        logic = CBCTPlannerLogic()
        logic.get_curve_node_for_jaw = lambda *a, **kw: None  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                logic.generate_and_export_surgical_guide(volumeNode=object(), out_dir=Path(td))
            self.assertIn("dental arch curve", str(ctx.exception).lower())

    # 18. Cache invalidation after segmentation modifications
    def test_18_cache_invalidation_after_segmentation_edit(self) -> None:
        class _MockObj:
            def __init__(self, mtime: int):
                self._mtime = mtime
            def GetMTime(self) -> int:
                return self._mtime

        class _MockSeg(_MockObj):
            def __init__(self, mtime: int, n: int):
                super().__init__(mtime)
                self._n = n
            def GetNumberOfSegments(self) -> int:
                return self._n

        class _MockNode:
            def __init__(self, nid: str, mtime: int, seg: _MockSeg | None = None):
                self._id = nid
                self._mtime = mtime
                self._seg = seg
                self._t = None
            def GetID(self) -> str:
                return self._id
            def GetMTime(self) -> int:
                return self._mtime
            def GetSegmentation(self):
                return self._seg
            def GetParentTransformNode(self):
                return self._t

        seg_obj = _MockSeg(10, 5)
        vol = _MockNode("vol1", 100)
        seg_node = _MockNode("seg1", 200, seg=seg_obj)
        tok1 = compute_volume_segmentation_cache_token(vol, seg_node)

        # Edit underlying vtkSegmentation without bumping outer node mtime
        seg_obj._mtime = 11
        tok2 = compute_volume_segmentation_cache_token(vol, seg_node)
        self.assertNotEqual(tok1, tok2)

        # Apply parent transform
        seg_node._t = _MockNode("xfm1", 50)
        tok3 = compute_volume_segmentation_cache_token(vol, seg_node)
        self.assertNotEqual(tok2, tok3)

        # Verify MeasurementEngine.invalidate_caches clears cached site measurements
        shape = (40, 40, 40)
        img = np.full(shape, -500.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        mask[10:25, 16:24, 10:30] = LOWER_JAW_ID
        img[mask > 0] = 900.0
        arch = _make_straight_arch(y_mm=20.0, z_mm=20.0)
        eng = self._build_engine(mask, img, arch=arch)
        m1, _ = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        assert m1 is not None
        # Mutate mask in place and invalidate cache
        mask[25:30, 16:24, 10:30] = LOWER_JAW_ID
        eng.invalidate_caches(mask_changed=True)
        m2, _ = eng.measure_at_arc_mm("lower", 15.0, fdi_hint="46")
        assert m2 is not None
        self.assertGreater(m2.bone_height_mm or 0.0, (m1.bone_height_mm or 0.0) + 3.0)


class TestWorkflowUIHelpers(unittest.TestCase):
    def test_workspace_roles_preserve_selections_and_full_suite_restores_all(self) -> None:
        class _Box:
            def __init__(self):
                self.collapsed = True
                self.visible = False
            def setVisible(self, v: bool):
                self.visible = v

        class _Lbl:
            def __init__(self):
                self.text = ""
            def setText(self, t: str):
                self.text = t

        widget = SimpleNamespace(
            volumeSelector="VOL_NODE_KEEP",
            segmentationSelector="SEG_NODE_KEEP",
            step1Box=_Box(),
            step2Box=_Box(),
            step3Box=_Box(),
            step4Box=_Box(),
            step5Box=_Box(),
            step6Box=_Box(),
            step7Box=_Box(),
            risemarBox=_Box(),
            advancedStep5Box=_Box(),
            globalStatusLabel=_Lbl(),
        )
        for role in ("Radiologist Workspace", "Implant & Surgical Dentist", "Ortho & Craniofacial", "All Clinical Tools (Full Suite)"):
            apply_workspace_role(widget, role)
            self.assertEqual(widget.volumeSelector, "VOL_NODE_KEEP")
            self.assertEqual(widget.segmentationSelector, "SEG_NODE_KEEP")
        # Full Suite restores all workflow sections visible
        for attr in ("step1Box", "step2Box", "step3Box", "step4Box", "step5Box", "step6Box", "step7Box"):
            self.assertTrue(getattr(widget, attr).visible)

    def test_gallery_navigation_and_stale_tracking(self) -> None:
        state = GalleryNavigationState()
        state.set_items([{"offset_mm": -2.0}, {"offset_mm": 0.0}, {"offset_mm": 2.0}], full_gallery=True)
        self.assertEqual(state.active_index, 1)
        self.assertTrue(state.can_prev)
        self.assertTrue(state.can_next)
        state.step(-1)
        self.assertEqual(state.active_index, 0)
        self.assertFalse(state.can_prev)
        state.mark_stale("Segmentation edited")
        self.assertTrue(state.is_stale)

    def test_risemar_no_personal_path_and_validation(self) -> None:
        self.assertEqual(DEFAULT_RISEMAR_WEIGHTS_PATH, "")
        ok, msg = validate_risemar_weights_path("")
        self.assertFalse(ok)
        self.assertIn("No RISE-MAR weights", msg)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tf:
            tf.write(b"dummy")
            tf.flush()
            ok2, msg2 = validate_risemar_weights_path(tf.name)
            self.assertTrue(ok2, msg2)


if __name__ == "__main__":
    unittest.main()

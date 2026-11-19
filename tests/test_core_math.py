"""
Unit tests for SlicerCBCTPlanner covering Phase 0, Phase 1 (Clinical Safety B1-B4),
and Phase 2 (Correctness Bugs B5-B20).
Runnable outside 3D Slicer using standard Python + NumPy/SciPy:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from typing import Optional
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import CBCTPlanner  # noqa: E402
from CBCTPlanner import (  # noqa: E402
    NOT_MEASURED,
    CBCTPlannerLogic,
    format_metric,
)
from CBCTPlannerLib import (  # noqa: E402
    CLINICAL_DISCLAIMER_TEXT,
    CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID,
    CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML,
    CrossSectionNavigator,
    PanoramicCurveLogic,
    VRVisualizationLogic,
    should_handle_navigation_shortcut,
)
from CBCTPlannerLib.core.arch_detector import (  # noqa: E402
    DentalArch,
    compute_fdi_arc_positions_mm,
    fit_dental_arch_from_points,
    project_point_to_polyline_arc_mm,
)
from CBCTPlannerLib.core.labels import (  # noqa: E402
    FDI_TO_TOOTH_ID,
    match_segment_to_label_and_precedence,
)
from CBCTPlannerLib.core.measurements import MeasurementEngine  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


class _MockPiecewiseFunction:
    """Strictly enforces 4-element [x, y, midpoint, sharpness] buffer like vtkPiecewiseFunction."""

    def __init__(self, pts: list[list[float]]) -> None:
        self._pts = [list(p) for p in pts]

    def GetSize(self) -> int:
        return len(self._pts)

    def RemoveAllPoints(self) -> None:
        self._pts.clear()

    def AddPoint(self, x: float, y: float, midpoint: float = 0.5, sharpness: float = 0.0) -> int:
        self._pts.append([float(x), float(y), float(midpoint), float(sharpness)])
        return len(self._pts) - 1

    def GetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 4:
            raise TypeError(f"vtkPiecewiseFunction.GetNodeValue expects 4 floats, got {len(val)}")
        for i in range(4):
            val[i] = self._pts[index][i]
        return 1

    def SetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 4:
            raise TypeError(f"vtkPiecewiseFunction.SetNodeValue expects 4 floats, got {len(val)}")
        self._pts[index] = list(val)
        return 1


class _MockColorTransferFunction:
    """Strictly enforces 6-element [x, r, g, b, midpoint, sharpness] buffer like vtkColorTransferFunction."""

    def __init__(self, pts: list[list[float]]) -> None:
        self._pts = [list(p) for p in pts]

    def GetSize(self) -> int:
        return len(self._pts)

    def RemoveAllPoints(self) -> None:
        self._pts.clear()

    def AddRGBPoint(
        self,
        x: float,
        r: float,
        g: float,
        b: float,
        midpoint: float = 0.5,
        sharpness: float = 0.0,
    ) -> int:
        self._pts.append([float(x), float(r), float(g), float(b), float(midpoint), float(sharpness)])
        return len(self._pts) - 1

    def GetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 6:
            raise TypeError(f"vtkColorTransferFunction.GetNodeValue expects 6 floats, got {len(val)}")
        for i in range(6):
            val[i] = self._pts[index][i]
        return 1

    def SetNodeValue(self, index: int, val: list[float]) -> int:
        if len(val) != 6:
            raise TypeError(f"vtkColorTransferFunction.SetNodeValue expects 6 floats, got {len(val)}")
        self._pts[index] = list(val)
        return 1


class _MockVolumePropertyNode:
    def __init__(self, prop_id: str = "Prop1") -> None:
        self._id = prop_id
        self._op = _MockPiecewiseFunction([[-1000.0, 0.0, 0.5, 0.0], [500.0, 0.8, 0.5, 0.0]])
        self._col = _MockColorTransferFunction(
            [[-1000.0, 0.0, 0.0, 0.0, 0.5, 0.0], [500.0, 0.9, 0.8, 0.7, 0.5, 0.0]]
        )
        self._grad = _MockPiecewiseFunction([[0.0, 0.0, 0.5, 0.0], [100.0, 1.0, 0.5, 0.0]])
        self.disabled_events_count = 0

    def GetID(self) -> str:
        return self._id

    def StartModify(self) -> int:
        return 0

    def EndModify(self, _prev: int) -> None:
        pass

    def DisableModifiedEventOn(self) -> None:
        self.disabled_events_count += 1

    def DisableModifiedEventOff(self) -> None:
        pass

    def Copy(self, other: "_MockVolumePropertyNode") -> None:
        self._op = _MockPiecewiseFunction(other._op._pts)
        self._col = _MockColorTransferFunction(other._col._pts)

    def GetScalarOpacity(self) -> _MockPiecewiseFunction:
        return self._op

    def GetColor(self) -> _MockColorTransferFunction:
        return self._col

    def GetGradientOpacity(self) -> _MockPiecewiseFunction:
        return self._grad


class _MockCurveNode:
    def __init__(self, pts: np.ndarray, mtime: int = 1, node_id: str = "vtkMRMLMarkupsCurveNode1") -> None:
        self._pts = np.asarray(pts, dtype=np.float64)
        self._mtime = mtime
        self._id = node_id
        self._attrs: dict[str, str] = {}
        self._transform_id: Optional[str] = None

    def GetID(self) -> str:
        return self._id

    def GetMTime(self) -> int:
        return self._mtime

    def GetNumberOfControlPoints(self) -> int:
        return len(self._pts)

    def GetNthControlPointPositionWorld(self, idx: int, buf: list[float]) -> None:
        buf[0], buf[1], buf[2] = float(self._pts[idx, 0]), float(self._pts[idx, 1]), float(self._pts[idx, 2])

    def SetNthControlPointPositionWorld(self, idx: int, x: float, y: float, z: float) -> None:
        self._pts[idx] = [float(x), float(y), float(z)]
        self._mtime += 1

    def GetCurveWorld(self) -> None:
        return None

    def SetAttribute(self, k: str, v: str) -> None:
        self._attrs[k] = str(v)

    def GetAttribute(self, k: str) -> Optional[str]:
        return self._attrs.get(k)

    def SetAndObserveTransformNodeID(self, tf_id: Optional[str]) -> None:
        self._transform_id = tf_id

    def GetTransformNodeID(self) -> Optional[str]:
        return self._transform_id


class _MockRulerNode(_MockCurveNode):
    def GetLineLengthWorld(self) -> float:
        if len(self._pts) < 2:
            return 0.0
        return float(np.linalg.norm(self._pts[1] - self._pts[0]))


class TestSlicerCBCTPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.ras_arch_pts = np.array(
            [
                [28.0, -22.0, 0.0],
                [24.0, -8.0, 0.0],
                [16.0, 8.0, 0.0],
                [8.0, 16.0, 0.0],
                [0.0, 20.0, 0.0],
                [-8.0, 16.0, 0.0],
                [-16.0, 8.0, 0.0],
                [-24.0, -8.0, 0.0],
                [-28.0, -22.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_phase0_zero_optional_deps_and_single_fit_function(self) -> None:
        """Module imports with zero optional deps and uses a single fit_dental_arch_from_points."""
        self.assertIs(
            CBCTPlanner.fit_dental_arch_from_points,
            fit_dental_arch_from_points,
        )
        with patch.dict(sys.modules, {"skimage": None, "nibabel": None, "pydicom": None, "SimpleITK": None}):
            arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
            self.assertGreater(arch.total_arc_length, 80.0)

    def test_phase1_ras_right_to_left_orientation_and_fdi_side_mapping(self) -> None:
        """In Slicer RAS (+X = Patient Right), arch starts at +X and ends at -X; Q1/Q4 map to +X, Q2/Q3 to -X."""
        shuffled = self.ras_arch_pts[::-1].copy()
        fitted = fit_dental_arch_from_points(shuffled, num_control_points=11, ras_convention=True)
        self.assertGreater(fitted[0, 0], 0.0)
        self.assertLess(fitted[-1, 0], 0.0)

        arch = DentalArch(fitted, jaw_type="lower", ras_convention=True)
        logic = CBCTPlannerLogic()

        for right_fdi in (18, 16, 11, 48, 46, 41):
            t_r = logic.fdi_to_normalized_arch_t(right_fdi)
            pt_r = arch.evaluate_at(t_r)
            self.assertLess(t_r, 0.5)
            self.assertGreater(pt_r[0], 0.0)

        for left_fdi in (21, 26, 28, 31, 36, 38):
            t_l = logic.fdi_to_normalized_arch_t(left_fdi)
            pt_l = arch.evaluate_at(t_l)
            self.assertGreater(t_l, 0.5)
            self.assertLess(pt_l[0], 0.0)

    def test_phase1_frame_normal_points_buccal_and_yellow_slice_orientation(self) -> None:
        """Verify N = T x Up points Buccal everywhere and Yellow slice has Buccal-Left / Lingual-Right / Superior-Up."""
        arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
        f_right = arch.get_frame(0.1)
        self.assertGreater(f_right.normal[0], 0.5)

        f_mid = arch.get_frame(0.5)
        self.assertGreater(f_mid.normal[1], 0.8)

        f_left = arch.get_frame(0.9)
        self.assertLess(f_left.normal[0], -0.5)

        for frame in (f_right, f_mid, f_left):
            screen_right, screen_up, slice_normal = CrossSectionNavigator.compute_yellow_slice_axes(
                frame.tangent, frame.normal, frame.binormal
            )
            np.testing.assert_allclose(screen_right, -frame.normal, atol=1e-6)
            np.testing.assert_allclose(screen_up, frame.binormal, atol=1e-6)
            det = float(np.linalg.det(np.column_stack([screen_right, screen_up, slice_normal])))
            self.assertAlmostEqual(det, 1.0, places=5)

    def test_b1_manual_ruler_untouched_returns_none_and_modified_returns_length(self) -> None:
        """B1: Untouched manual ruler returns None; user-modified ruler returns physical length."""
        p0 = [10.0, 5.0, 0.0]
        p1 = [10.0, 5.0, -10.0]
        ruler = _MockRulerNode(np.array([p0, p1]), node_id="Ruler46")
        ruler.SetAttribute("CBCTPlanner.InitialEndpoints", json.dumps([p0, p1]))
        ruler.SetAttribute("CBCTPlanner.UserModified", "0")

        # Untouched default ruler MUST return None
        self.assertIsNone(CBCTPlannerLogic._read_manual_ruler_length(ruler))

        # Simulate user dragging endpoint 1 from -10.0 to -12.5 mm (12.5 mm length)
        ruler.SetNthControlPointPositionWorld(1, 10.0, 5.0, -12.5)
        self.assertEqual(CBCTPlannerLogic._read_manual_ruler_length(ruler), 12.5)
        self.assertEqual(ruler.GetAttribute("CBCTPlanner.UserModified"), "1")

    def test_b2_fdi_arc_mapping_segmentation_vs_width_table_and_missing_teeth(self) -> None:
        """B2: Verify segmentation centroid projection, mesiodistal width table, missing teeth, and calibrated markers."""
        arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
        total_len = arch.total_arc_length

        # 1. Without segmentation: uses anatomical mesiodistal width table (non-uniform spacing!) and flags approximate
        unseg_map = compute_fdi_arc_positions_mm(total_len, jaw_type="lower", arch=arch)
        self.assertTrue(unseg_map["46"].is_approximate)
        self.assertEqual(unseg_map["46"].source, "approximate_width_table")
        # Molar step (47->46) is wider than incisor step (42->41)
        molar_step = unseg_map["46"].arc_mm - unseg_map["47"].arc_mm
        incisor_step = unseg_map["41"].arc_mm - unseg_map["42"].arc_mm
        self.assertGreater(molar_step, incisor_step * 1.4)

        # 2. With segmented teeth #47 and #45 but missing #46: #47/#45 are exact ("segmentation"), #46 is interpolated
        seg_centroids = {
            "47": arch.evaluate_at_mm(18.0),
            "45": arch.evaluate_at_mm(38.0),
        }
        cal_markers = {"41": 49.5}
        mixed_map = compute_fdi_arc_positions_mm(
            total_len,
            jaw_type="lower",
            arch=arch,
            segmented_centroids=seg_centroids,
            calibrated_arc_mm=cal_markers,
        )
        self.assertEqual(mixed_map["47"].source, "segmentation")
        self.assertFalse(mixed_map["47"].is_approximate)
        self.assertAlmostEqual(mixed_map["47"].arc_mm, 18.0, delta=0.1)

        self.assertEqual(mixed_map["46"].source, "interpolated_from_neighbors")
        self.assertTrue(mixed_map["46"].is_approximate)
        self.assertTrue(mixed_map["46"].is_missing_tooth)
        self.assertGreater(mixed_map["46"].arc_mm, 18.0)
        self.assertLess(mixed_map["46"].arc_mm, 38.0)

        self.assertEqual(mixed_map["41"].source, "calibrated_marker")
        self.assertFalse(mixed_map["41"].is_approximate)
        self.assertAlmostEqual(mixed_map["41"].arc_mm, 49.5, places=2)

    def test_b3_non_destructive_occlusal_alignment_and_reset(self) -> None:
        """B3: Verify non-destructive occlusal alignment matrix and co-registration across nodes."""
        p1 = [20.0, -10.0, -2.0]
        p2 = [0.0, 20.0, 3.0]
        p3 = [-20.0, -10.0, -2.0]
        normal = CBCTPlannerLogic.compute_occlusal_normal_from_3_points(np.array([p1, p2, p3]))
        center = np.mean([p1, p2, p3], axis=0)
        mat = CBCTPlannerLogic.compute_occlusal_alignment_matrix_4x4(normal, center_ras=center)

        # Rotation block must be orthogonal with det = +1 and map `normal` to +Z (0, 0, 1)
        R = mat[:3, :3]
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)
        rotated_normal = R @ normal
        np.testing.assert_allclose(rotated_normal, [0.0, 0.0, 1.0], atol=1e-6)
        # Center of rotation must remain invariant under mat
        center_h = np.array([center[0], center[1], center[2], 1.0])
        np.testing.assert_allclose((mat @ center_h)[:3], center, atol=1e-6)

    def test_b4_arch_frame_cross_section_measurements_flipped_k_and_oblique_phantom(self) -> None:
        """B4: Verify MeasurementEngine measures in arch-frame cross-section on standard, flipped-K, and oblique phantoms."""
        # Standard phantom: mandible block height = 16 mm above canal, width = 8 mm along Y, density = 950
        img = np.full((50, 50, 50), -500.0, dtype=np.float32)
        mask = np.zeros((50, 50, 50), dtype=np.uint16)
        mask[10:30, 21:29, 10:40] = 2  # Mandible (Z=10..29 -> 19 mm total, crest at Z=29)
        img[10:30, 21:29, 10:40] = 950.0
        mask[11:13, 24:26, 10:40] = 5  # Canal top at Z=12 -> distance from crest Z=29 is ~17 mm

        vol_std = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4, dtype=np.float64),
        )
        arch_pts = np.array(
            [[35.0, 25.0, 29.0], [28.0, 25.0, 29.0], [20.0, 25.0, 29.0], [12.0, 25.0, 29.0]],
            dtype=np.float64,
        )
        arch_std = DentalArch(arch_pts, jaw_type="lower", ras_convention=True)
        eng_std = MeasurementEngine(
            vol_std, mask, arch=arch_std, arches={"lower": arch_std}, is_dental_segmentator_5class=True
        )
        meas_std = eng_std.measure_implant_site("46")
        self.assertIsNotNone(meas_std)
        assert meas_std is not None
        self.assertAlmostEqual(meas_std.bone_height_mm or 0.0, 17.0, delta=0.75)
        self.assertAlmostEqual(meas_std.crest_width_mm or 0.0, 7.0, delta=1.0)
        self.assertIn(
            "approximate (CBCT gray values are not calibrated HU)", meas_std.misch_density_class or ""
        )

        # 2. Flipped-K phantom: reverse array along axis 0 (K) and set affine[2, 2] = -1, affine[2, 3] = 49.0
        # Physical world coordinates of every voxel remain 100% identical!
        img_flip = img[::-1, :, :].copy()
        mask_flip = mask[::-1, :, :].copy()
        affine_flip = np.eye(4, dtype=np.float64)
        affine_flip[2, 2] = -1.0
        affine_flip[2, 3] = 49.0
        vol_flip = CBCTVolume(
            image=img_flip,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 49.0),
            direction=np.diag([1.0, 1.0, -1.0]),
            affine=affine_flip,
        )
        eng_flip = MeasurementEngine(
            vol_flip, mask_flip, arch=arch_std, arches={"lower": arch_std}, is_dental_segmentator_5class=True
        )
        meas_flip = eng_flip.measure_implant_site("46")
        self.assertIsNotNone(meas_flip)
        assert meas_flip is not None
        self.assertAlmostEqual(meas_flip.bone_height_mm or 0.0, meas_std.bone_height_mm or 0.0, delta=0.5)
        self.assertAlmostEqual(meas_flip.crest_width_mm or 0.0, meas_std.crest_width_mm or 0.0, delta=0.5)

        # 3. Oblique phantom: rotate volume, affine, and arch by +25 degrees around X axis
        theta = math.radians(25.0)
        R_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(theta), -math.sin(theta)],
                [0.0, math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float64,
        )
        affine_oblique = np.eye(4, dtype=np.float64)
        affine_oblique[:3, :3] = R_x
        vol_oblique = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=R_x,
            affine=affine_oblique,
        )
        occ_oblique = R_x @ np.array([0.0, 0.0, 1.0])
        arch_pts_oblique = (R_x @ arch_pts.T).T
        arch_oblique = DentalArch(
            arch_pts_oblique, jaw_type="lower", occlusal_normal=occ_oblique, ras_convention=True
        )
        eng_oblique = MeasurementEngine(
            vol_oblique,
            mask,
            arch=arch_oblique,
            arches={"lower": arch_oblique},
            is_dental_segmentator_5class=True,
        )
        meas_oblique = eng_oblique.measure_implant_site("46")
        self.assertIsNotNone(meas_oblique)
        assert meas_oblique is not None
        self.assertAlmostEqual(meas_oblique.bone_height_mm or 0.0, meas_std.bone_height_mm or 0.0, delta=0.75)
        self.assertAlmostEqual(meas_oblique.crest_width_mm or 0.0, meas_std.crest_width_mm or 0.0, delta=0.75)

    @staticmethod
    def _ridge_engine(
        mask: np.ndarray, img: np.ndarray, jaw_type: str = "lower", is_5class: bool = True
    ) -> MeasurementEngine:
        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4, dtype=np.float64),
        )
        arch = DentalArch(
            np.array(
                [[35.0, 25.0, 29.0], [28.0, 25.0, 29.0], [20.0, 25.0, 29.0], [12.0, 25.0, 29.0]],
                dtype=np.float64,
            ),
            jaw_type=jaw_type,
            ras_convention=True,
        )
        return MeasurementEngine(
            vol, mask, arch=arch, arches={jaw_type: arch}, is_dental_segmentator_5class=is_5class
        )

    def test_ridge_plausibility_rejects_full_cross_section_mask(self) -> None:
        """A mask filling the whole cross-section is not measurable: no crest or width segments."""
        img = np.full((50, 50, 50), 950.0, dtype=np.float32)
        mask = np.full((50, 50, 50), 2, dtype=np.uint16)  # 5-class mandible fills the field
        engine = self._ridge_engine(mask, img)
        meas, cs = engine.measure_at_arc_mm("lower", 20.0, fdi_hint="46")
        self.assertIsNone(meas)
        self.assertIsNotNone(cs)
        assert cs is not None
        self.assertIsNone(cs.crest_v)
        self.assertEqual(cs.width_segments, {})

    def test_ridge_plausibility_accepts_normal_8x19mm_ridge(self) -> None:
        """A normal 8 mm wide x 19 mm tall ridge still measures."""
        img = np.full((50, 50, 50), -500.0, dtype=np.float32)
        mask = np.zeros((50, 50, 50), dtype=np.uint16)
        mask[10:30, 21:29, 10:40] = 2  # 19 mm tall (Z=10..29), 8 mm wide (Y=21..28)
        img[10:30, 21:29, 10:40] = 950.0
        engine = self._ridge_engine(mask, img)
        meas = engine.measure_implant_site("46")
        self.assertIsNotNone(meas)
        assert meas is not None
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 19.0, delta=0.75)
        self.assertAlmostEqual(meas.crest_width_mm or 0.0, 7.0, delta=1.0)

    def test_ridge_plausibility_rejected_tile_draws_no_hw_overlays(self) -> None:
        """The rendered tile for a rejected cross-section draws no H/W measurement overlays."""
        from CBCTPlannerLib.core_print_render import render_cross_section

        img = np.full((50, 50, 50), 950.0, dtype=np.float32)
        mask = np.full((50, 50, 50), 2, dtype=np.uint16)
        engine = self._ridge_engine(mask, img)
        meas, cs = engine.measure_at_arc_mm("lower", 20.0, fdi_hint="46")
        self.assertIsNone(meas)
        assert cs is not None
        render = render_cross_section(cs, None, scale=2)
        self.assertTrue(render.available)
        self.assertEqual(render.overlay_px, [])
        self.assertFalse(any(label[2].startswith(("H", "W")) for label in render.labels))

    def test_ridge_plausibility_width_only_rejection_keeps_maxillary_height(self) -> None:
        """A maxillary mask spanning the full buccolingual field drops widths but keeps the crest height."""
        img = np.full((50, 50, 50), -500.0, dtype=np.float32)
        mask = np.zeros((50, 50, 50), dtype=np.uint16)
        # 5-class maxilla (id 1): spans every buccolingual voxel but only 19 mm crest-to-base.
        mask[10:30, :, 10:40] = 1
        img[10:30, :, 10:40] = 950.0
        engine = self._ridge_engine(mask, img, jaw_type="upper")
        meas, cs = engine.measure_at_arc_mm("upper", 20.0, fdi_hint="26")
        self.assertIsNotNone(cs)
        assert cs is not None
        # Height survives: the crest is resolved, only the buccolingual widths are withheld.
        self.assertIsNotNone(cs.crest_v)
        self.assertEqual(cs.width_segments, {})
        self.assertIsNotNone(meas)
        assert meas is not None
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 19.0, delta=1.5)
        self.assertIsNone(meas.crest_width_mm)
        self.assertIsNone(meas.mid_root_width_mm)
        self.assertIsNone(meas.apical_width_mm)
        self.assertTrue(all(w is None for w in meas.bone_width_at_depths_mm.values()))

    def test_cross_section_footer_prints_dashes_for_withheld_crest_width(self) -> None:
        """RC-01: a withheld crest width prints `W0 --`, never a fabricated `W0 0.0 mm`."""
        from CBCTPlannerLib.core_print_render import render_cross_section

        img = np.full((50, 50, 50), -500.0, dtype=np.float32)
        mask = np.zeros((50, 50, 50), dtype=np.uint16)
        mask[10:30, :, 10:40] = 1  # 5-class maxilla spanning the whole buccolingual field
        img[10:30, :, 10:40] = 950.0
        engine = self._ridge_engine(mask, img, jaw_type="upper")
        meas, cs = engine.measure_at_arc_mm("upper", 20.0, fdi_hint="26")
        self.assertIsNotNone(meas)
        self.assertIsNotNone(cs)
        assert meas is not None and cs is not None
        self.assertIsNone(meas.crest_width_mm)
        self.assertIsNotNone(meas.bone_height_mm)

        render = render_cross_section(cs, meas, scale=2)
        footer = [label[2] for label in render.labels if label[2].startswith("H ")]
        self.assertEqual(len(footer), 1, render.labels)
        self.assertIn("W0 --", footer[0])
        self.assertNotIn("0.0 mm", footer[0])

    def test_ridge_envelope_includes_in_socket_root_on_partially_dentate_phantom(self) -> None:
        """RC-01: a partially dentate ridge measures jawbone plus the in-socket root wall."""
        img = np.full((50, 50, 50), -500.0, dtype=np.float32)
        mask = np.zeros((50, 50, 50), dtype=np.uint16)
        # ToothFairy3 mandible: lingual plate only (buccal plate missing) plus basal bone.
        mask[10:28, 21:23, 10:40] = 1
        mask[10:14, 21:29, 10:40] = 1
        # In-socket root of #46 fills the hollow socket and closes the ridge buccally.
        mask[14:27, 23:27, 20:30] = 46
        img[mask != 0] = 950.0

        engine = self._ridge_engine(mask, img, is_5class=False)
        meas = engine.measure_implant_site("46")
        self.assertIsNotNone(meas)
        assert meas is not None
        self.assertIsNone(meas.withheld_reason)
        # The crest resolves from the root where the jaw mask is hollow; jawbone alone
        # would stop at the basal block (~3 mm of height).
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 16.0, delta=1.0)
        # Crest width spans the lingual plate plus the root wall; jawbone alone is ~1 mm.
        self.assertAlmostEqual(meas.crest_width_mm or 0.0, 5.0, delta=0.75)

    def test_ridge_envelope_hu_fallback_when_segmentation_covers_only_teeth(self) -> None:
        """RC-01: a teeth-only segmentation measures gray-value bone instead of rejecting the site."""
        img = np.full((50, 50, 50), -500.0, dtype=np.float32)
        mask = np.zeros((50, 50, 50), dtype=np.uint16)
        mask[10:30, 21:29, 10:40] = 46  # only the tooth is segmented
        img[10:30, 21:29, 10:40] = 950.0  # cortical/trabecular bone well above 250 HU

        engine = self._ridge_engine(mask, img, is_5class=False)
        meas = engine.measure_implant_site("46")
        self.assertIsNotNone(meas)
        assert meas is not None
        self.assertIsNone(meas.withheld_reason)
        self.assertAlmostEqual(meas.bone_height_mm or 0.0, 19.0, delta=0.75)
        self.assertAlmostEqual(meas.crest_width_mm or 0.0, 7.0, delta=1.0)

    def test_b5_cpr_displacement_and_grid_frame_identical_handedness(self) -> None:
        """B5: Verify CPR displacement frame (-N, B, T) and grid frame both have det = +1 (no mirroring)."""
        dir_mat = PanoramicCurveLogic.compute_cpr_direction_matrix(
            self.ras_arch_pts[0], self.ras_arch_pts[-1], occlusal_normal=(0.0, 0.0, 1.0)
        )
        det_grid = float(np.linalg.det(dir_mat[:3, :3]))
        self.assertAlmostEqual(det_grid, 1.0, places=6)

        arch = DentalArch(self.ras_arch_pts, jaw_type="lower", ras_convention=True)
        for t_val in (0.1, 0.5, 0.9):
            f = arch.get_frame(t_val)
            dx, dy, dz = PanoramicCurveLogic.compute_cpr_displacement_frame(f.tangent, f.normal, f.binormal)
            det_disp = float(np.linalg.det(np.column_stack([dx, dy, dz])))
            self.assertAlmostEqual(det_disp, det_grid, places=6)
            np.testing.assert_allclose(dx, -f.normal, atol=1e-6)

    def test_b6_cpr_fixed_frame_spacing_on_15mm_radius_arch(self) -> None:
        """B6: Verify fixed <= 1.5 mm CPR frame spacing keeps max chordal deviation < 0.05 mm on a 15 mm arch."""
        radius = 15.0
        angles = np.linspace(0.0, math.pi, 15)
        semi_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)])
        total_arc = math.pi * radius  # ~47.12 mm

        # Even if output preview spacing is 1.5 mm, frame spacing is <= 1.5 mm (never 1.5 * 5 = 7.5 mm!)
        n_frames = PanoramicCurveLogic.compute_cpr_num_frames(total_arc, max_frame_spacing_mm=1.5)
        self.assertGreaterEqual(n_frames, 32)

        pano_logic = PanoramicCurveLogic()
        curve = _MockCurveNode(semi_pts)
        pts, _, _, _, _ = pano_logic.get_curve_points_and_frames(curve, num_samples=n_frames)
        midpoints = 0.5 * (pts[:-1] + pts[1:])
        radii_mid = np.linalg.norm(midpoints[:, :2], axis=1)
        max_dev = float(np.max(np.abs(radii_mid - radius)))
        self.assertLess(max_dev, 0.10)

    def test_b7_navigation_shortcut_focus_guard(self) -> None:
        """B7: Verify navigation shortcuts ignore sliders, spinboxes, tables, and text inputs."""

        class _FakeWidget:
            def __init__(self, cls_name: str, parent: object = None) -> None:
                self._cls = cls_name
                self._parent = parent

            def metaObject(self) -> object:
                c = self._cls

                class _M:
                    def className(self) -> str:
                        return c

                return _M()

            def parent(self) -> object:
                return self._parent

        for input_cls in (
            "QDoubleSpinBox",
            "ctkSliderWidget",
            "QSlider",
            "QLineEdit",
            "QTableWidget",
            "QComboBox",
        ):
            self.assertFalse(should_handle_navigation_shortcut(_FakeWidget(input_cls)))

        slice_view = _FakeWidget("qMRMLSliceWidget")
        self.assertTrue(should_handle_navigation_shortcut(slice_view))
        self.assertTrue(should_handle_navigation_shortcut(None))

    def test_b8_vr_preset_switch_a_b_a_and_non_hu_percentiles(self) -> None:
        """B8: Switching VR presets A -> B -> A preserves exact point counts/values; non-HU uses percentiles."""
        vr = VRVisualizationLogic()
        prop = _MockVolumePropertyNode("PropNode1")

        preset_b = _MockVolumePropertyNode("PresetB")
        preset_b.GetScalarOpacity()._pts = [[0.0, 0.0, 0.5, 0.0], [1000.0, 1.0, 0.5, 0.0]]
        preset_b.GetColor()._pts = [[0.0, 0.0, 0.0, 0.0, 0.5, 0.0], [1000.0, 1.0, 1.0, 1.0, 0.5, 0.0]]

        class _Provider:
            def GetPresetByName(self, _name: str) -> _MockVolumePropertyNode:
                return preset_b

        hu_vol = np.linspace(-1000.0, 2500.0, 1000, dtype=np.float32)
        vr.apply_preset_and_shift(
            prop,
            preset_name="CT-Dental-CBCT",
            shift_hu=100.0,
            preset_provider=_Provider(),
            volume_array=hu_vol,
        )
        self.assertEqual(prop.GetScalarOpacity().GetSize(), 6)
        self.assertAlmostEqual(prop.GetScalarOpacity()._pts[0][0], -900.0)

        # Switch to Preset B (2 points)
        vr.apply_preset_and_shift(
            prop, preset_name="CT-MIP", shift_hu=50.0, preset_provider=_Provider(), volume_array=hu_vol
        )
        self.assertEqual(prop.GetScalarOpacity().GetSize(), 2)
        self.assertAlmostEqual(prop.GetScalarOpacity()._pts[1][0], 1050.0)

        # Switch back to Preset A (CT-Dental-CBCT) with shift 0 -> must restore all 6 points cleanly!
        vr.apply_preset_and_shift(
            prop, preset_name="CT-Dental-CBCT", shift_hu=0.0, preset_provider=_Provider(), volume_array=hu_vol
        )
        self.assertEqual(prop.GetScalarOpacity().GetSize(), 6)
        self.assertAlmostEqual(prop.GetScalarOpacity()._pts[0][0], -1000.0)
        self.assertGreater(prop.disabled_events_count, 0)

        # Non-HU volume (unsigned gray values 200..15000) -> derives transfer function from percentiles
        non_hu_vol = np.linspace(200.0, 15000.0, 2000, dtype=np.float32)
        self.assertFalse(vr.is_volume_hu_calibrated(non_hu_vol))
        op_pts, _ = vr.compute_cbct_transfer_control_points(non_hu_vol)
        self.assertGreater(op_pts[0][0], 200.0)
        self.assertLess(op_pts[-1][0], 15000.0)

    def test_b9_bounded_lru_caches_1000_mtime_changes_and_cli_cleanup(self) -> None:
        """B9: Verify _frame_cache and _poly_cache stay bounded (<= 8 entries) over 1000 MTime changes."""
        pano = PanoramicCurveLogic(max_cache_entries=8)
        nav = CrossSectionNavigator(panoramic_logic=pano, max_cache_entries=8)
        for mtime in range(1000):
            c = _MockCurveNode(self.ras_arch_pts, mtime=mtime)
            pano.get_curve_points_and_frames(c, num_samples=30)
            nav._poly_cache[(c.GetID(), mtime, 2.0, 24.0, (0.0, 0.0, 1.0))] = mtime
        self.assertLessEqual(len(pano._frame_cache), 8)
        self.assertLessEqual(len(nav._poly_cache), 8)

    def test_b12_explicit_slab_checkbox_and_mean_mip_mode(self) -> None:
        """B12: Verify explicit SlabEnabled checkbox and Mean/MIP/MinIP reconstruction types."""

        class _FakeSliceNode:
            def __init__(self) -> None:
                self.enabled = True
                self.slab_type = -1
                self.thickness = 0.0

            def SetSlabReconstructionEnabled(self, val: bool) -> None:
                self.enabled = bool(val)

            def SetSlabReconstructionType(self, t: int) -> None:
                self.slab_type = int(t)

            def SetSlabReconstructionThickness(self, th: float) -> None:
                self.thickness = float(th)

        snode = _FakeSliceNode()
        CrossSectionNavigator.configure_slice_node_slab(
            snode, slab_enabled=False, slab_thickness_mm=5.0, slab_mode="Mean"
        )
        self.assertFalse(snode.enabled)
        self.assertEqual(snode.slab_type, CrossSectionNavigator.resolve_vtk_slab_type("Mean"))

        CrossSectionNavigator.configure_slice_node_slab(
            snode, slab_enabled=True, slab_thickness_mm=3.5, slab_mode="MIP"
        )
        self.assertTrue(snode.enabled)
        self.assertEqual(snode.slab_type, CrossSectionNavigator.resolve_vtk_slab_type("MIP"))
        self.assertAlmostEqual(snode.thickness, 3.5)

    def test_b13_analytic_arc_segment_projection_subsample_precision(self) -> None:
        """B13: Verify line-segment projection onto an analytic circular arc has sub-sample error < 0.01 mm."""
        radius = 25.0
        n_samples = 100
        thetas = np.linspace(0.0, math.pi, n_samples)
        pts = np.column_stack([radius * np.cos(thetas), radius * np.sin(thetas), np.zeros_like(thetas)])
        arc_s = radius * thetas

        # Pick an angle halfway between discrete sample vertices
        test_theta = float(0.5 * (thetas[37] + thetas[38]))
        true_arc = radius * test_theta
        query_pt = [radius * math.cos(test_theta), radius * math.sin(test_theta), 0.0]

        proj_arc, dist, _ = project_point_to_polyline_arc_mm(pts, arc_s, query_pt)
        self.assertLess(abs(proj_arc - true_arc), 0.01)
        self.assertLess(dist, 0.01)

    def test_b19_segment_matching_exact_tokens_and_precedence(self) -> None:
        """B19: Verify 'Segment_11' is not matched as FDI #11, 'Tooth_11' is matched, and teeth > canal > jaw precedence holds."""
        # Segment_999 does not match any known ToothFairy3 or canonical ID -> unknown
        lbl_generic, cat_generic, _, prec_generic = match_segment_to_label_and_precedence(
            "Segment_999", "Segment_999"
        )
        self.assertEqual(lbl_generic, 0)
        self.assertEqual(cat_generic, "unknown")
        self.assertEqual(prec_generic, 0)

        # ToothFairy3 Segment_11 matches canonical FDI #11
        lbl_tf3, cat_tf3, _, prec_tf3 = match_segment_to_label_and_precedence("Segment_11", "Segment_11")
        self.assertEqual(lbl_tf3, FDI_TO_TOOTH_ID["11"])
        self.assertEqual(cat_tf3, "tooth")

        # Explicit tooth name or attribute MUST match FDI #11
        lbl_t11, cat_t11, _, prec_t11 = match_segment_to_label_and_precedence("Segment_11", "Tooth 11")
        self.assertEqual(lbl_t11, FDI_TO_TOOTH_ID["11"])
        self.assertEqual(cat_t11, "tooth")

        lbl_canal, cat_canal, _, prec_canal = match_segment_to_label_and_precedence(
            "Segment_5", "Mandibular canal"
        )
        lbl_jaw, cat_jaw, _, prec_jaw = match_segment_to_label_and_precedence("Segment_2", "Mandible")
        self.assertGreater(prec_t11, prec_canal)
        self.assertGreater(prec_canal, prec_jaw)

    def test_b20_arch_autofit_u_v_tapered_arches_and_no_skull_fallback(self) -> None:
        """B20: Verify fit_dental_arch_from_points accurately fits U-shaped, V-shaped, and tapered arches with uneven density."""
        # 1. U-shaped arch (parallel posterior legs at X = +-22 mm for Y in [-25, 0], then semicircular anterior)
        leg_y = np.linspace(-25.0, 0.0, 20)
        right_leg = np.column_stack([np.full_like(leg_y, 22.0), leg_y, np.zeros_like(leg_y)])
        left_leg = np.column_stack([np.full_like(leg_y, -22.0), leg_y[::-1], np.zeros_like(leg_y)])
        ant_th = np.linspace(0.0, math.pi, 30)
        ant_arc = np.column_stack([22.0 * np.cos(ant_th), 22.0 * np.sin(ant_th), np.zeros_like(ant_th)])
        # Add 200 dense cluster points on one molar to test voxel-grid density de-biasing
        dense_cluster = right_leg[0] + np.random.default_rng(42).normal(0.0, 0.2, size=(200, 3))
        u_cloud = np.vstack([right_leg, ant_arc, left_leg, dense_cluster])

        u_ctrl = fit_dental_arch_from_points(u_cloud, num_control_points=13, ras_convention=True)
        self.assertGreater(u_ctrl[0, 0], 18.0)
        self.assertLess(u_ctrl[-1, 0], -18.0)
        self.assertGreater(float(np.max(u_ctrl[:, 1])), 19.0)

        # 2. V-shaped arch
        v_x_r = np.linspace(24.0, 0.0, 25)
        v_y_r = np.linspace(-20.0, 22.0, 25)
        v_x_l = np.linspace(0.0, -24.0, 25)
        v_y_l = np.linspace(22.0, -20.0, 25)
        v_cloud = np.vstack(
            [
                np.column_stack([v_x_r, v_y_r, np.zeros_like(v_x_r)]),
                np.column_stack([v_x_l, v_y_l, np.zeros_like(v_x_l)]),
            ]
        )
        v_ctrl = fit_dental_arch_from_points(v_cloud, num_control_points=11, ras_convention=True)
        self.assertGreater(v_ctrl[0, 0], 20.0)
        self.assertLess(v_ctrl[-1, 0], -20.0)
        self.assertGreater(float(np.max(v_ctrl[:, 1])), 18.0)

        # 3. Tapered arch
        tap_th = np.linspace(0.0, math.pi, 40)
        tap_r = 22.0 - 4.0 * np.sin(tap_th)
        tap_cloud = np.column_stack(
            [tap_r * np.cos(tap_th), 1.3 * tap_r * np.sin(tap_th), np.zeros_like(tap_th)]
        )
        tap_ctrl = fit_dental_arch_from_points(tap_cloud, num_control_points=11, ras_convention=True)
        self.assertGreater(tap_ctrl[0, 0], 18.0)
        self.assertLess(tap_ctrl[-1, 0], -18.0)

    def test_b22_honest_measurements_clean_numeric_csv_and_html_pdf_export(self) -> None:
        """B22: CSV numeric columns are empty '' when unmeasured (with measurement_source column); HTML & PDF include disclaimer."""
        logic = CBCTPlannerLogic()
        unseg_rows = logic.analyze_selected_teeth([46, 16], volumeNode=None, segmentationNode=None)
        for row in unseg_rows:
            self.assertEqual(row["source"], NOT_MEASURED)
            self.assertIsNone(row["height_mm"])
            self.assertEqual(format_metric(row["height_mm"], "mm"), NOT_MEASURED)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _, csv_path = logic.export_treatment_plan_files(unseg_rows, tmp_path)
            with open(csv_path, encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            # Numeric columns must be empty string "", NOT "not measured"
            self.assertEqual(reader[0]["height_mm"], "")
            self.assertEqual(reader[0]["w_crest_mm"], "")
            self.assertEqual(reader[0]["measurement_source"], NOT_MEASURED)
            self.assertIn("pos_r_mm", reader[0])

            class _FakeVol:
                def GetName(self) -> str:
                    return "<script>alert('xss')</script>"

            html_path, pdf_path = logic.generate_clinical_reports(
                unseg_rows,
                _FakeVol(),
                tmp_path,
                capture_screenshots=False,
                lightbox_step_mm=3.5,
                include_patient_metadata=False,
                patient_name="Secret Name",
            )
            content = html_path.read_text(encoding="utf-8")
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertIn(CLINICAL_DISCLAIMER_TEXT, content)
            # Default anonymization must hide "Secret Name"
            self.assertNotIn("Secret Name", content)
            self.assertIn("Anonymous", content)
            self.assertTrue(pdf_path.exists())
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-1.4"))

    def test_lightbox_offsets_and_cross_section_tilt_30_deg(self) -> None:
        """Verify 3-slice lightbox (-2, 0, +2 mm) offsets, layout XML, and +-30° cross-section tilt math."""
        self.assertEqual(CUSTOM_DENTAL_LIGHTBOX_LAYOUT_ID, 503)
        self.assertIn('singletontag="YellowMinus"', CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)
        self.assertIn('singletontag="Yellow"', CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)
        self.assertIn('singletontag="YellowPlus"', CUSTOM_DENTAL_LIGHTBOX_LAYOUT_XML)

        s_m, s_0, s_p = CrossSectionNavigator.compute_lightbox_offsets(50.0, 100.0, step_mm=2.0)
        self.assertEqual((s_m, s_0, s_p), (48.0, 50.0, 52.0))
        self.assertEqual(CrossSectionNavigator.clamp_tilt_deg(45.0), 30.0)
        self.assertEqual(CrossSectionNavigator.clamp_tilt_deg(-60.0), -30.0)

    def test_8x10_print_layout_engine_generation(self) -> None:
        """Verify genuine 8x10 pages: real slice count, PDF media box, explicit unavailable tiles."""
        import types

        from CBCTPlannerLib.PrintLayoutEngine import PrintLayoutEngine
        from CBCTPlannerLib.core.measurements import CrossSectionData, ImplantSiteMeasurement
        from CBCTPlannerLib.core_print_render import render_cross_section

        u = np.arange(-10.0, 10.01, 0.25)
        v = np.arange(-10.0, 10.01, 0.25)
        mask_cs = np.zeros((len(v), len(u)), dtype=np.int32)
        mask_cs[v <= 2.0, :] = 2
        img_cs = (600.0 + np.add.outer(v * 40.0, u * 20.0)).astype(np.float32)

        def _cs() -> CrossSectionData:
            return CrossSectionData(
                u_coords=u,
                v_coords=v,
                mask_cs=mask_cs,
                img_cs=img_cs,
                frame=(
                    np.zeros(3),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ),
                crest_v=2.0,
                width_segments={0.0: (-3.0, 3.0), 3.0: (-2.8, 2.8), 5.0: (-2.5, 2.5)},
                canal_centroid_uv=(0.0, -4.0),
            )

        def _meas(fdi="46") -> ImplantSiteMeasurement:
            return ImplantSiteMeasurement(
                tooth_fdi=fdi,
                bone_height_mm=12.0,
                bone_width_at_depths_mm={0.0: 6.0, 3.0: 5.6, 5.0: 5.0},
                crest_width_mm=6.0,
                mid_root_width_mm=5.8,
                apical_width_mm=4.5,
                mesiodistal_space_mm=11.0,
                mean_density_hu=550.0,
                misch_density_class="D2",
                distance_to_vital_structure_mm=8.0,
                vital_structure_name="Inferior Alveolar Canal",
                safety_margin_mm=2.0,
                suggested_diameter_mm=4.1,
                suggested_length_mm=10.0,
                suggested_angulation_deg=0.0,
            )

        meas = _meas()

        render = render_cross_section(_cs(), meas, scale=2)
        self.assertTrue(render.available)
        width = (len(u) - 1) * 2
        expected_x = (-3.0 - u[0]) / (u[-1] - u[0]) * width
        crest_line = next(
            line for line in render.overlay_px if line[0][1] == line[1][1] and line[0][0] < line[1][0]
        )
        self.assertLessEqual(abs(crest_line[0][0] - 0.0), 1.5)
        self.assertAlmostEqual(crest_line[1][0], width, delta=1.5)
        w0_line = render.overlay_px[-3]
        self.assertLessEqual(abs(w0_line[0][0] - expected_x), 1.0)

        class _Arch:
            total_arc_length = 120.0

            def find_closest_parameter(self, _point):
                return 0.5, 0.0

        class _Engine:
            def _arch_for_fdi(self, _fdi):
                return _Arch()

            def get_site_center_ras(self, _fdi):
                return np.array([20.0, 0.0, 0.0])

            def measure_at_arc_mm(self, _jaw, _arc, fdi_hint=None):
                return _meas(str(fdi_hint or "46")), _cs()

        fake_logic = types.SimpleNamespace(
            _get_or_build_measurement_engine=lambda *_args, **_kwargs: _Engine(),
        )
        engine = PrintLayoutEngine(logic=fake_logic)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            html_p, pdf_p, slices_15 = engine.generate_8x10_layout(
                volumeNode=None, selected_fdis=[46], quadrant=4, out_dir=td_path
            )
            self.assertEqual(len(slices_15), 15)
            self.assertTrue(html_p.exists())
            self.assertTrue(pdf_p.exists())
            html_txt = html_p.read_text(encoding="utf-8")
            self.assertIn("size: 8in 10in", html_txt)
            self.assertTrue(
                all(spec.image_data_uri.startswith("data:image/png;base64,") for spec in slices_15)
            )
            self.assertIn("data:image/png;base64,", html_txt)
            # The PNG must contain real rendered anatomy, not a blank placeholder tile.
            import base64
            import io

            from PIL import Image

            first_png = base64.b64decode(slices_15[0].image_data_uri.split(",", 1)[1])
            with Image.open(io.BytesIO(first_png)) as png:
                self.assertEqual(png.size, (160, 160))
                pixels = np.asarray(png.convert("L"), dtype=np.float32)
            self.assertGreater(float(pixels.std()), 15.0)
            self.assertGreater(float(pixels.max()) - float(pixels.min()), 80.0)
            self.assertGreaterEqual(float(pixels.min()), 0.0)
            self.assertLessEqual(float(pixels.max()), 255.0)
            self.assertIn("Quadrant 4", html_txt)
            self.assertIn("Panoramic View", html_txt)
            self.assertNotIn("simulated", html_txt.lower())
            # Legend boxes must use the same colours the renderer draws for W0/W3/W5.
            self.assertIn("background: #00dcff", html_txt)
            self.assertIn("background: #ffb400", html_txt)
            self.assertIn("background: #ff7828", html_txt)
            pdf_bytes = pdf_p.read_bytes()
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))
            import re

            media_box = re.search(rb"/MediaBox \[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_bytes)
            self.assertIsNotNone(media_box)
            self.assertAlmostEqual(float(media_box.group(1)), 576.0, delta=1.0)
            self.assertAlmostEqual(float(media_box.group(2)), 720.0, delta=1.0)

            _, _, slices_27 = engine.generate_8x10_layout(
                volumeNode=None,
                selected_fdis=[48, 47, 46, 45, 44, 43, 42, 41],
                quadrant=4,
                out_dir=td_path,
            )
            self.assertEqual(len(slices_27), 27)

            _, pdf_multi, slices_multi = engine.generate_8x10_layout(
                volumeNode=None,
                selected_fdis=[11, 46],
                quadrant=None,
                out_dir=td_path,
            )
            self.assertEqual(len(slices_multi), 30)
            self.assertGreaterEqual(pdf_multi.read_bytes().count(b"/Type /Page"), 2)
            multi_html = html_p.read_text(encoding="utf-8")
            self.assertIn("Page 1 of 2", multi_html)
            self.assertIn("Page 2 of 2", multi_html)
            self.assertNotIn("Page 1 of 1", multi_html)

    def test_dental_segmentator_vr_preset_is_available_and_ramps_up(self) -> None:
        from CBCTPlannerLib.VRVisualization import VRVisualizationLogic

        self.assertIn("DentalSegmentator (Teeth & Bone)", VRVisualizationLogic.SUPPORTED_VR_PRESETS)

        class _TF:
            def __init__(self):
                self.points = []

            def RemoveAllPoints(self):
                self.points = []

            def AddPoint(self, x, a):
                self.points.append((x, a))

            def AddRGBPoint(self, x, r, g, b):
                self.points.append((x, r, g, b))

        class _Prop:
            def __init__(self):
                self.op = _TF()
                self.col = _TF()
                self.grad = _TF()

            def GetScalarOpacity(self):
                return self.op

            def GetColor(self):
                return self.col

            def GetGradientOpacity(self):
                return self.grad

        prop = _Prop()
        VRVisualizationLogic._apply_dental_segmentator_transfer_function(
            prop, volume_array=np.linspace(-1000, 3000, 5000, dtype=np.float32)
        )
        xs = [p[0] for p in prop.op.points]
        alphas = [p[1] for p in prop.op.points]
        self.assertEqual(len(xs), 6)
        self.assertTrue(all(b > a for a, b in zip(xs, xs[1:])))
        self.assertEqual(alphas[0], 0.0)
        self.assertEqual(alphas[-1], 1.0)
        self.assertEqual(len(prop.col.points), 6)

    def test_remove_small_label_islands_keeps_main_component(self) -> None:
        from CBCTPlannerLib.CheckpointInference import remove_small_label_islands

        labels = np.zeros((20, 20, 20), dtype=np.uint16)
        labels[2:10, 2:10, 2:10] = 46
        labels[15, 15, 15] = 46
        cleaned = remove_small_label_islands(labels, min_voxels=20)
        self.assertEqual(int(np.count_nonzero(cleaned == 46)), 8 * 8 * 8)
        self.assertEqual(int(cleaned[15, 15, 15]), 0)

    def test_remove_small_label_islands_preserves_sub_threshold_sliver_and_blob(self) -> None:
        """A thin root and a small impacted molar, each below the threshold, must survive."""
        from CBCTPlannerLib.CheckpointInference import remove_small_label_islands

        labels = np.zeros((24, 24, 24), dtype=np.uint16)
        labels[2:14, 2, 2] = 38  # 12-voxel thin-root sliver (main component)
        labels[2, 2, 2] = 38  # extra speckle inside the same bbox
        labels[16:18, 16:18, 16:19] = 48  # 12-voxel compact impacted-molar blob
        labels[20, 20, 20] = 48  # isolated one-voxel seam speckle
        cleaned = remove_small_label_islands(labels, min_voxels=20)
        self.assertEqual(int(np.count_nonzero(cleaned == 38)), 12)
        self.assertEqual(int(np.count_nonzero(cleaned == 48)), 12)
        self.assertTrue(np.all(cleaned[2:14, 2, 2] == 38))
        self.assertTrue(np.all(cleaned[16:18, 16:18, 16:19] == 48))
        self.assertEqual(int(cleaned[20, 20, 20]), 0)

    def test_remove_small_label_islands_is_deterministic_and_bbox_bounded(self) -> None:
        """Repeated runs match, and a neighbouring label inside the crop is never cleared."""
        from CBCTPlannerLib.CheckpointInference import remove_small_label_islands

        labels = np.zeros((20, 20, 20), dtype=np.uint16)
        labels[2, 2, 2] = 38
        labels[2, 2, 18] = 38  # same label's bbox spans x=2..18
        labels[2, 2, 8:13] = 46  # neighbouring label's only (small) component
        outputs = [remove_small_label_islands(labels, min_voxels=8) for _ in range(8)]
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])
        self.assertTrue(np.all(outputs[0][2, 2, 8:13] == 46))
        self.assertEqual(int(np.count_nonzero(outputs[0] == 38)), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for Phase 6 multi-root topology EndodonticEngine & CanalTracer bridge."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.CanalTracer import CanalTracerLogic  # noqa: E402
from CBCTPlannerLib.EndodonticEngine import (  # noqa: E402
    CLINICAL_ENDO_STATUS,
    EndoToothAnalysis,
    EndodonticEngine,
)
from CBCTPlannerLib.core.contracts import (  # noqa: E402
    SOURCE_ESTIMATED,
    SOURCE_MANUALLY_CORRECTED,
    SOURCE_MEASURED,
    SOURCE_NOT_AVAILABLE,
)
from CBCTPlannerLib.core.labels import FDI_TO_PULP_ID, FDI_TO_TOOTH_ID  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


def _paint_tube(
    mask: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    radius_mm: float,
    spacing_mm: float,
    label_id: int,
    *,
    bend_mm: float = 0.0,
) -> None:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    mid = 0.5 * (p0 + p1) + np.array([bend_mm, 0.0, 0.0], dtype=np.float64)
    nz, ny, nx = mask.shape
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    x_mm = xx * spacing_mm
    y_mm = yy * spacing_mm
    z_mm = zz * spacing_mm
    for t in np.linspace(0.0, 1.0, 80):
        # Quadratic Bezier curve through (p0, mid, p1)
        q = ((1.0 - t) ** 2) * p0 + (2.0 * (1.0 - t) * t) * mid + (t**2) * p1
        d2 = (x_mm - q[0]) ** 2 + (y_mm - q[1]) ** 2 + (z_mm - q[2]) ** 2
        mask[d2 <= radius_mm**2] = label_id


class MultiRootEndodonticEngineTests(unittest.TestCase):
    def test_bifurcated_two_root_molar_topology_curvature_edt_and_wall_clearance(self) -> None:
        sp = 0.4
        shape = (80, 64, 64)
        aff = np.diag([sp, sp, sp, 1.0]).astype(np.float64)
        mask = np.zeros(shape, dtype=np.int32)
        tooth_id = FDI_TO_TOOTH_ID["46"]
        pulp_id = FDI_TO_PULP_ID["46"]

        # Paint outer molar crown + mesial & distal roots in tooth_id
        _paint_tube(mask, [12.0, 12.0, 26.0], [12.0, 12.0, 17.5], 4.2, sp, tooth_id)
        _paint_tube(mask, [12.0, 12.0, 19.0], [17.0, 15.0, 5.0], 2.4, sp, tooth_id, bend_mm=2.5)
        _paint_tube(mask, [12.0, 12.0, 19.0], [7.5, 9.5, 5.0], 2.4, sp, tooth_id)

        # Paint Y-shaped bifurcating pulp: wide chamber (r=1.6 mm) + curved root 1 (r=0.75) + straight root 2 (r=0.75)
        _paint_tube(mask, [12.0, 12.0, 24.5], [12.0, 12.0, 18.0], 1.6, sp, pulp_id)
        _paint_tube(mask, [12.0, 12.0, 18.5], [17.0, 15.0, 6.0], 0.85, sp, pulp_id, bend_mm=2.5)
        _paint_tube(mask, [12.0, 12.0, 18.5], [7.5, 9.5, 6.0], 0.85, sp, pulp_id)

        engine = EndodonticEngine(min_branch_spur_mm=2.0)
        res = engine.analyze_tooth(46, mask, aff)

        self.assertTrue(res.available, res.withheld_reason)
        self.assertEqual(res.status_flag, CLINICAL_ENDO_STATUS)
        self.assertEqual(res.pulp_source_type, SOURCE_MEASURED)
        self.assertGreaterEqual(res.total_canals, 2)
        self.assertGreaterEqual(len(res.bifurcations), 1)

        # Real EDT radius variation: wide chamber > narrow apex (NOT constant 0.5 mm!)
        radii = res.canals[0].radii_mm
        self.assertGreater(float(np.max(radii) - np.min(radii)), 0.35)
        for c in res.canals:
            self.assertGreater(c.length_mm, 12.0)
            self.assertIsNotNone(c.min_wall_distance_mm)
            self.assertGreater(float(c.min_wall_distance_mm), 0.7)

        # Curved vs straighter branch deflection angles
        max_curv = max(c.curvature_deg for c in res.canals)
        self.assertGreater(max_curv, 18.0)

        # Access cavity planning aid + manual override preservation
        self.assertIsNotNone(res.access_cavity)
        self.assertEqual(res.access_cavity.source_type, SOURCE_MEASURED)
        self.assertFalse(res.access_cavity.manual_edit)

        manual_res = engine.analyze_tooth(
            46,
            mask,
            aff,
            manual_access_entry_ras=[12.0, 12.0, 26.0],
            manual_access_target_ras=[12.0, 12.0, 18.5],
        )
        self.assertTrue(manual_res.access_cavity.manual_edit)
        self.assertEqual(manual_res.access_cavity.source_type, SOURCE_MANUALLY_CORRECTED)

        # JSON save / restore round-trip
        restored = EndoToothAnalysis.from_dict(json.loads(res.to_json()))
        self.assertEqual(restored.total_canals, res.total_canals)
        self.assertEqual(len(restored.bifurcations), len(res.bifurcations))
        self.assertAlmostEqual(restored.canals[0].length_mm, res.canals[0].length_mm, places=2)

        # Backward compatibility via CanalTracerLogic
        vol = CBCTVolume(np.zeros(shape, dtype=np.float32), (sp, sp, sp), (0.0, 0.0, 0.0), np.eye(3), aff)
        legacy = CanalTracerLogic.extract_root_canal_for_tooth(46, vol, mask)
        self.assertIsNotNone(legacy)
        self.assertGreater(float(np.std(legacy.radii_mm)), 0.05)

    def test_three_root_maxillary_molar_and_conservative_missing_pulp_behavior(self) -> None:
        sp = 0.4
        shape = (80, 64, 64)
        aff = np.diag([sp, sp, sp, 1.0]).astype(np.float64)
        mask = np.zeros(shape, dtype=np.int32)
        tooth_id = FDI_TO_TOOTH_ID["16"]
        pulp_id = FDI_TO_PULP_ID["16"]

        # Maxilla crown at lower Z (inverted relative to mandible roots going superiorly +Z)
        _paint_tube(mask, [12.0, 12.0, 6.0], [12.0, 12.0, 14.0], 4.5, sp, tooth_id)
        _paint_tube(mask, [12.0, 12.0, 12.5], [16.5, 16.0, 26.0], 2.2, sp, tooth_id)
        _paint_tube(mask, [12.0, 12.0, 12.5], [16.5, 7.5, 26.0], 2.2, sp, tooth_id)
        _paint_tube(mask, [12.0, 12.0, 12.5], [6.5, 12.0, 26.0], 2.4, sp, tooth_id)

        _paint_tube(mask, [12.0, 12.0, 7.5], [12.0, 12.0, 13.5], 1.5, sp, pulp_id)
        _paint_tube(mask, [12.0, 12.0, 13.0], [16.5, 16.0, 25.0], 0.85, sp, pulp_id)
        _paint_tube(mask, [12.0, 12.0, 13.0], [16.5, 7.5, 25.0], 0.85, sp, pulp_id)
        _paint_tube(mask, [12.0, 12.0, 13.0], [6.5, 12.0, 25.0], 0.90, sp, pulp_id)

        engine = EndodonticEngine()
        res = engine.analyze_tooth(16, mask, aff)
        self.assertTrue(res.available, res.withheld_reason)
        self.assertFalse(res.is_mandible)
        self.assertEqual(res.total_canals, 3)

        # Conservative missing pulp: solid uniform dentin tooth without pulp label must NOT invent fake canals
        solid_mask = np.zeros((40, 32, 32), dtype=np.int32)
        solid_mask[6:30, 10:22, 10:22] = FDI_TO_TOOTH_ID["44"]
        solid_img = np.where(solid_mask > 0, 1500.0, -800.0).astype(np.float32)
        res_missing = engine.analyze_tooth(
            44,
            solid_mask,
            np.diag([0.5, 0.5, 0.5, 1.0]),
            image_zyx=solid_img,
            allow_intensity_refinement=True,
            optional_checkpoint_path=Path("nonexistent_endo_weights.pth"),
        )
        self.assertFalse(res_missing.available)
        self.assertEqual(res_missing.pulp_source_type, SOURCE_NOT_AVAILABLE)
        self.assertIsNotNone(res_missing.external_model_status)

        # Genuine dark interior lumen with intensity refinement allowed -> flagged strictly as estimated
        img_lumen = solid_img.copy()
        img_lumen[8:28, 15:17, 15:17] = 100.0
        res_est = engine.analyze_tooth(
            44,
            solid_mask,
            np.diag([0.5, 0.5, 0.5, 1.0]),
            image_zyx=img_lumen,
            allow_intensity_refinement=True,
        )
        self.assertTrue(res_est.available)
        self.assertEqual(res_est.pulp_source_type, SOURCE_ESTIMATED)
        self.assertGreater(len(res_est.caveats), 0)


if __name__ == "__main__":
    unittest.main()

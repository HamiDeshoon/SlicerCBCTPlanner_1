"""Tests for Phase 7 exact capsule implant geometry, multi-structure collision, and uncertainty."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.ImplantPlanner import ImplantPlannerLogic  # noqa: E402
from CBCTPlannerLib.core.labels import (  # noqa: E402
    FDI_TO_TOOTH_ID,
    LOWER_JAW_ID,
    RIGHT_IAN_ID,
    RIGHT_SINUS_ID,
    UPPER_JAW_ID,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


class CapsuleImplantGeometryAndUncertaintyTests(unittest.TestCase):
    def test_closed_form_capsule_inter_implant_distance_and_collision(self) -> None:
        logic = ImplantPlannerLogic()
        t_dir = np.array([-1.0, 0.0, 0.0])
        n_dir = np.array([0.0, 1.0, 0.0])
        b_dir = np.array([0.0, 0.0, 1.0])

        # Two vertical 4.0 x 10.0 mm implants placed 6.5 mm apart center-to-center:
        # exact surface-to-surface clearance = 6.5 - (2.0 + 2.0) = 2.5 mm (< 3.0 mm warning)
        i1 = logic.place_or_update_implant(
            fdi=46,
            crest_point_ras=[10.0, 10.0, 20.0],
            T=t_dir,
            N=n_dir,
            B=b_dir,
            diameter_mm=4.0,
            length_mm=10.0,
        )
        i2 = logic.place_or_update_implant(
            fdi=45,
            crest_point_ras=[16.5, 10.0, 20.0],
            T=t_dir,
            N=n_dir,
            B=b_dir,
            diameter_mm=4.0,
            length_mm=10.0,
        )
        self.assertIsNotNone(i1.min_neighbor_implant_distance_mm)
        self.assertAlmostEqual(i1.min_neighbor_implant_distance_mm, 2.5, places=1)
        self.assertAlmostEqual(i2.min_neighbor_implant_distance_mm, 2.5, places=1)
        self.assertFalse(i1.is_verified_safe)
        clr = i1.clearances["neighbor_implant"]
        self.assertEqual(clr["method"], "closed_form_capsule_segment")
        self.assertIsNone(clr["estimated_uncertainty_mm"])
        self.assertIn("No calibrated uncertainty", clr["limitation_note"])

    def test_multi_structure_sdf_adjacent_tooth_canal_sinus_cortical_and_uncertainty_priors(self) -> None:
        sp = 0.5
        shape = (64, 64, 64)
        aff = np.diag([sp, sp, sp, 1.0]).astype(np.float64)
        mask = np.zeros(shape, dtype=np.int32)
        img = np.full(shape, -800.0, dtype=np.float32)

        # Mandibular bone block at x in [6..24], y in [10..22], z in [4..22] mm
        zz, yy, xx = np.ogrid[:64, :64, :64]
        x_mm, y_mm, z_mm = xx * sp, yy * sp, zz * sp
        jaw_box = (
            (x_mm >= 6.0) & (x_mm <= 24.0) & (y_mm >= 10.0) & (y_mm <= 22.0) & (z_mm >= 4.0) & (z_mm <= 22.0)
        )
        mask[jaw_box] = LOWER_JAW_ID
        img[jaw_box] = 650.0

        # IAN canal right underneath future implant #46 (centered around x=15, y=16, z=7.5 mm)
        ian_box = ((x_mm - 15.0) ** 2 + (y_mm - 16.0) ** 2 + (z_mm - 7.5) ** 2) <= 1.2**2
        mask[ian_box] = RIGHT_IAN_ID

        # Adjacent premolar tooth #45 near implant margin (x in [8..11], y in [13..19], z in [11..24] mm)
        tooth_45 = (
            (x_mm >= 8.0) & (x_mm <= 11.0) & (y_mm >= 13.0) & (y_mm <= 19.0) & (z_mm >= 11.0) & (z_mm <= 24.0)
        )
        mask[tooth_45] = FDI_TO_TOOTH_ID["45"]

        vol = CBCTVolume(img, (sp, sp, sp), (0.0, 0.0, 0.0), np.eye(3), aff)
        logic = ImplantPlannerLogic()
        t_dir = np.array([-1.0, 0.0, 0.0])
        n_dir = np.array([0.0, 1.0, 0.0])
        b_dir = np.array([0.0, 0.0, 1.0])

        # Place implant at x=15, y=16, crest z=21.5 -> apex z=11.5 (r=2.0 -> x reaches 13.0 -> 2.0 mm from #45; apex 11.5 -> ~2.8 mm from canal z=8.7)
        imp = logic.place_or_update_implant(
            fdi=46,
            crest_point_ras=[15.0, 16.0, 21.5],
            T=t_dir,
            N=n_dir,
            B=b_dir,
            diameter_mm=4.0,
            length_mm=10.0,
            volume=vol,
            mask=mask,
        )
        self.assertIsNotNone(imp.adjacent_tooth_clearance_mm)
        self.assertEqual(imp.nearest_tooth_fdi, "45")
        self.assertIsNotNone(imp.canal_clearance_mm)
        self.assertFalse(imp.cortical_perforation)
        self.assertIn("adjacent_tooth", imp.clearances)
        self.assertIn("inferior_alveolar_canal", imp.clearances)
        self.assertIn("quantization", imp.clearances["inferior_alveolar_canal"]["limitation_note"])

        # With explicit calibrated uncertainty prior (e.g. 1.5 mm canal segmentation bound),
        # effective clearance drops strictly below nominal and triggers safety warning
        logic.recompute_all_verifications(
            volume=vol,
            mask=mask,
            uncertainty_priors_mm={"inferior_alveolar_canal": 1.5, "adjacent_tooth": 0.6},
        )
        c_dict = imp.clearances["inferior_alveolar_canal"]
        self.assertAlmostEqual(c_dict["estimated_uncertainty_mm"], 1.5, places=3)
        self.assertAlmostEqual(
            c_dict["effective_clearance_mm"],
            c_dict["nominal_clearance_mm"] - 1.5,
            places=2,
        )
        self.assertFalse(imp.is_verified_safe)

        # Cortical perforation test: shift implant outside jaw box
        imp_perf = logic.place_or_update_implant(
            fdi=47,
            crest_point_ras=[15.0, 23.0, 21.5],
            T=t_dir,
            N=n_dir,
            B=b_dir,
            diameter_mm=4.0,
            length_mm=10.0,
            volume=vol,
            mask=mask,
        )
        self.assertTrue(imp_perf.cortical_perforation)
        self.assertFalse(imp_perf.is_verified_safe)

        # Maxillary sinus test
        upper_mask = np.zeros(shape, dtype=np.int32)
        upper_mask[jaw_box] = UPPER_JAW_ID
        sinus_box = (
            (x_mm >= 12.0)
            & (x_mm <= 18.0)
            & (y_mm >= 13.0)
            & (y_mm <= 19.0)
            & (z_mm >= 18.0)
            & (z_mm <= 22.0)
        )
        upper_mask[sinus_box] = RIGHT_SINUS_ID
        imp_u = ImplantPlannerLogic().place_or_update_implant(
            fdi=16,
            crest_point_ras=[15.0, 16.0, 6.0],
            T=t_dir,
            N=n_dir,
            B=b_dir,
            diameter_mm=4.0,
            length_mm=11.0,
            volume=vol,
            mask=upper_mask,
        )
        self.assertIsNotNone(imp_u.sinus_clearance_mm)
        self.assertIn("maxillary_sinus", imp_u.clearances)


if __name__ == "__main__":
    unittest.main()

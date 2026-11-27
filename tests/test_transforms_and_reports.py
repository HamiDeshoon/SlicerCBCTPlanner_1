"""Tests for Phase 9 structured JSON reports, provenance taxonomy, and coordinate transforms."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import CBCTPlanner as cbct_mod  # noqa: E402
from CBCTPlannerLib.ImplantPlanner import ImplantPlannerLogic  # noqa: E402
from CBCTPlannerLib.PrintLayoutEngine import PrintLayoutEngine  # noqa: E402
from CBCTPlannerLib.Utils import SOFTWARE_VERSION  # noqa: E402
from CBCTPlannerLib.core.arch_detector import DentalArch, fit_arch_from_label_mask  # noqa: E402
from CBCTPlannerLib.core.contracts import (  # noqa: E402
    SOURCE_ESTIMATED,
    SOURCE_MANUALLY_CORRECTED,
    SOURCE_MEASURED,
    SOURCE_NOT_AVAILABLE,
    VALID_SOURCE_TYPES,
)
from CBCTPlannerLib.core.measurements import MeasurementEngine  # noqa: E402
from CBCTPlannerLib.core.orientation import canonicalize_to_ras  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


class StructuredReportAndTransformTests(unittest.TestCase):
    def test_lps_and_rotated_oblique_scan_canonicalize_and_measure_consistently(self) -> None:
        # Simulate tilted LPS volume (`diag([-0.5, -0.5, 0.5])` rotated gently in XY)
        theta = np.radians(10.0)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        aff_lps = np.eye(4, dtype=np.float64)
        aff_lps[:3, :3] = rot @ np.diag([-0.5, -0.5, 0.5])
        aff_lps[:3, 3] = [20.0, 20.0, 0.0]

        shape = (48, 80, 80)
        img = np.full(shape, -800.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        img[10:34, 24:52, 18:62] = 650.0
        mask[10:34, 24:52, 18:62] = 1

        canon = canonicalize_to_ras(img, aff_lps)
        canon_m = canonicalize_to_ras(mask.astype(np.float32), aff_lps)
        self.assertGreater(float(np.linalg.det(canon.affine[:3, :3])), 0.0)
        self.assertGreater(float(canon.affine[0, 0]), 0.0)
        self.assertGreater(float(canon.affine[1, 1]), 0.0)
        self.assertGreater(float(canon.affine[2, 2]), 0.0)
        self.assertEqual(canon.image_zyx.shape, shape)
        self.assertEqual(canon_m.image_zyx.shape, shape)

    def test_structured_json_export_and_8x10_report_provenance(self) -> None:
        shape = (48, 80, 80)
        sp = 0.5
        aff = np.diag([sp, sp, sp, 1.0]).astype(np.float64)
        img = np.full(shape, -850.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.int32)
        zz, yy, xx = np.ogrid[:48, :80, :80]
        x_mm, y_mm, z_mm = xx * sp, yy * sp, zz * sp
        u = ((x_mm - 20.0) / 14.0) ** 2 + ((y_mm - 12.0) / 16.0) ** 2
        mandible = (u >= 0.65) & (u <= 1.35) & (y_mm >= 10.0) & (z_mm >= 6.0) & (z_mm <= 18.0)
        mask[mandible] = 1
        img[mandible] = 700.0

        vol = CBCTVolume(img, (sp, sp, sp), (0.0, 0.0, 0.0), np.eye(3), aff)
        fit = fit_arch_from_label_mask(mask, aff, jaw="Mandible", is_5class=False)
        self.assertTrue(fit.accepted, fit.reason)
        arch = DentalArch(fit.points_ras, jaw_type="lower", ras_convention=True)
        eng = MeasurementEngine(
            vol, mask, arch=arch, arches={"lower": arch}, is_dental_segmentator_5class=False
        )

        imp_logic = ImplantPlannerLogic()
        fr = arch.get_frame(0.3)
        imp = imp_logic.place_or_update_implant(
            fdi=46,
            crest_point_ras=fr.point,
            T=fr.tangent,
            N=fr.normal,
            B=fr.binormal,
            diameter_mm=3.5,
            length_mm=8.0,
            volume=vol,
            mask=mask,
        )

        class _HeadlessPrintLogic:
            def __init__(self, e: MeasurementEngine) -> None:
                self._e = e

            def _get_or_build_measurement_engine(self, *_args, **_kwargs) -> MeasurementEngine:
                return self._e

        with tempfile.TemporaryDirectory(prefix="cbct_report_json_") as tmp:
            out_dir = Path(tmp)
            pl = PrintLayoutEngine(logic=_HeadlessPrintLogic(eng))
            overrides = {
                46: {
                    "h_mm": 10.5,
                    "w0_mm": 6.8,
                    "w0_start_uv": (-3.4, -1.0),
                    "w0_end_uv": (3.4, -1.0),
                }
            }
            html_p, pdf_p, _ = pl.generate_8x10_layout(
                None,
                None,
                selected_fdis=[46],
                out_dir=out_dir,
                grayscale=True,
                measurement_overrides=overrides,
            )
            self.assertTrue(html_p.is_file())
            self.assertTrue(pdf_p.is_file())
            self.assertIsNotNone(pl.last_json_file)
            self.assertTrue(pl.last_json_file.is_file())
            rep_data = json.loads(pl.last_json_file.read_text(encoding="utf-8"))
            self.assertEqual(rep_data["software_version"], SOFTWARE_VERSION)
            self.assertTrue(len(rep_data["measurements"]) > 0)
            self.assertTrue(
                any(m["source_type"] == SOURCE_MANUALLY_CORRECTED for m in rep_data["measurements"])
            )
            for m in rep_data["measurements"]:
                self.assertIn(m["source_type"], VALID_SOURCE_TYPES)
                self.assertIn(m["unit"], ("mm", "HU"))

            plan_data = [
                {
                    "fdi": 46,
                    "jaw": "Mandible",
                    "source": "label_derived",
                    "height_mm": 10.5,
                    "w_crest_mm": 6.8,
                    "w_3mm": None,
                    "w_5mm": None,
                    "nerve_clearance_mm": None,
                    "hu": 680.0,
                    "misch": "D2",
                    "withheld_reason": None,
                    "pos_ras": [float(v) for v in fr.point],
                }
            ]
            j_path, csv_path = cbct_mod.CBCTPlannerLogic.export_treatment_plan_files(
                plan_data, out_dir, implants=[imp]
            )
            self.assertTrue(j_path.is_file())
            self.assertTrue(csv_path.is_file())
            tp = json.loads(j_path.read_text(encoding="utf-8"))
            self.assertEqual(tp["software_version"], SOFTWARE_VERSION)
            self.assertIn("report_envelope", tp)
            env_m = tp["report_envelope"]["measurements"]
            srcs = {m["name"]: m["source_type"] for m in env_m}
            self.assertEqual(srcs["bone_height_h"], SOURCE_MEASURED)
            self.assertEqual(srcs["width_3mm"], SOURCE_NOT_AVAILABLE)
            self.assertEqual(srcs["trabecular_gray"], SOURCE_ESTIMATED)


if __name__ == "__main__":
    unittest.main()

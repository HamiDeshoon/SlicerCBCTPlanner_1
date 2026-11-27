"""Unit tests for v1.1 implant sleeve-ring schema and backward compatibility."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.ImplantPlanner import ImplantPlannerLogic, PlacedImplant  # noqa: E402
from CBCTPlannerLib.Phase7Tools import PrinterPreset, SurgicalGuideGenerator  # noqa: E402


class TestGuideSleeveSchema(unittest.TestCase):
    def test_bundled_v11_library_loads_with_uninvented_null_rings(self) -> None:
        lib_path = MODULE_DIR / "Resources" / "generic_implant_library.json"
        raw = json.loads(lib_path.read_text(encoding="utf-8"))
        self.assertEqual(raw.get("schema_version"), "1.1")

        planner = ImplantPlannerLogic(lib_path)
        self.assertEqual(len(planner.library), len(raw["implants"]))
        for entry in planner.library.values():
            self.assertIsNone(entry.ring_outer_diameter_mm)
            self.assertIsNone(entry.ring_inner_diameter_mm)
            self.assertIsNone(entry.ring_height_mm)
            self.assertIsNone(entry.sleeve_offset_mm)
            self.assertTrue(entry.sleeved)

    def test_ring_entry_resolves_metal_ring_pocket(self) -> None:
        preset = PrinterPreset(id="sla", label="SLA", sleeve_clearance_mm=0.06, sleeve_wall_thickness_mm=1.2)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ring_lib.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": "1.1",
                        "implants": [
                            {
                                "id": "ring_sys_4.0x10",
                                "label": "Ring System Ø4.0×10",
                                "diameter_mm": 4.0,
                                "length_mm": 10.0,
                                "platform_diameter_mm": 4.2,
                                "ring_outer_diameter_mm": 5.0,
                                "ring_inner_diameter_mm": 4.1,
                                "ring_height_mm": 4.5,
                                "sleeve_offset_mm": 5.0,
                                "sleeved": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            planner = ImplantPlannerLogic(p)
            imp = planner.place_or_update_implant(
                fdi=46,
                crest_point_ras=[10.0, 10.0, 0.0],
                T=np.array([-1.0, 0.0, 0.0]),
                N=np.array([0.0, 1.0, 0.0]),
                B=np.array([0.0, 0.0, 1.0]),
                entry_id="ring_sys_4.0x10",
            )
            spec = SurgicalGuideGenerator.resolve_sleeve_dimensions(preset, imp)
            self.assertEqual(spec["sleeve_kind"], "metal_ring")
            self.assertAlmostEqual(spec["pocket_diameter_mm"], 5.0 + 2.0 * 0.06, places=3)
            self.assertAlmostEqual(spec["pocket_height_mm"], 4.5, places=3)
            self.assertAlmostEqual(spec["inner_bore_diameter_mm"], 4.1, places=3)
            self.assertAlmostEqual(spec["offset_above_crest_mm"], 5.0, places=3)
            self.assertIn("preset sleeve_height exceeds ring_height_mm", spec["warnings"])

    def test_ringless_entry_resolves_integral_tube_formula(self) -> None:
        preset = PrinterPreset(id="sla", label="SLA", sleeve_clearance_mm=0.06, sleeve_wall_thickness_mm=1.2)
        imp = PlacedImplant(
            fdi=36,
            entry_id="generic_4.0x10.0",
            label="Generic 4.0x10",
            diameter_mm=4.0,
            length_mm=10.0,
            platform_diameter_mm=4.0,
            apical_diameter_mm=3.4,
            crest_point_ras=[0.0, 0.0, 0.0],
            apex_point_ras=[0.0, 0.0, -10.0],
            axis_apical_ras=[0.0, 0.0, -1.0],
        )
        spec = SurgicalGuideGenerator.resolve_sleeve_dimensions(preset, imp)
        self.assertEqual(spec["sleeve_kind"], "integral_tube")
        self.assertIsNone(spec["pocket_diameter_mm"])
        self.assertIsNone(spec["pocket_height_mm"])
        self.assertAlmostEqual(spec["inner_bore_diameter_mm"], 4.12, places=3)
        self.assertAlmostEqual(spec["outer_diameter_mm"], 6.52, places=3)

    def test_v10_legacy_library_without_ring_keys_loads_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "v10.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "implants": [{"id": "old_35", "diameter_mm": 3.5, "length_mm": 10.0}],
                    }
                ),
                encoding="utf-8",
            )
            planner = ImplantPlannerLogic(p)
            e = planner.library["old_35"]
            self.assertIsNone(e.ring_outer_diameter_mm)
            self.assertIsNone(e.ring_inner_diameter_mm)
            self.assertIsNone(e.ring_height_mm)
            self.assertIsNone(e.sleeve_offset_mm)
            self.assertTrue(e.sleeved)


if __name__ == "__main__":
    unittest.main()

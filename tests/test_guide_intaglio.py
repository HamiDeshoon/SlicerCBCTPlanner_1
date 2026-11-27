"""Unit tests for anatomy-conforming intaglio surgical guide shell generation."""

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

from CBCTPlannerLib.ImplantPlanner import PlacedImplant  # noqa: E402
from CBCTPlannerLib.Phase7Tools import (  # noqa: E402
    GuideBuildReport,
    PrinterPreset,
    SurgicalGuideGenerator,
    check_edge_manifold,
)


class TestGuideIntaglioShell(unittest.TestCase):
    def _slab_phantom(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 0.25 mm spacing slab so tissue_offset=0.25 and thickness=2.0 quantize cleanly
        mask = np.zeros((44, 44, 60), dtype=np.uint8)
        mask[12:20, 14:30, 12:48] = 1
        aff = np.eye(4, dtype=np.float64)
        aff[0, 0] = 0.25
        aff[1, 1] = 0.25
        aff[2, 2] = 0.25
        arch_pts = np.array(
            [[3.0, 5.5, 5.0], [5.5, 5.5, 5.0], [8.5, 5.5, 5.0], [11.5, 5.5, 5.0]],
            dtype=np.float64,
        )
        return mask, aff, arch_pts

    def test_flat_slab_shell_thickness_offset_and_through_hole(self) -> None:
        mask, aff, arch_pts = self._slab_phantom()
        gen = SurgicalGuideGenerator()
        gen.presets["test_slab"] = PrinterPreset(
            id="test_slab",
            label="Test 0.25mm",
            guide_thickness_mm=2.0,
            tissue_offset_mm=0.25,
            sleeve_clearance_mm=0.05,
            sleeve_wall_thickness_mm=1.0,
            default_sleeve_height_mm=3.0,
            default_sleeve_offset_above_crest_mm=2.5,
        )
        imp = PlacedImplant(
            fdi=46,
            entry_id="ring_test",
            label="Ring Test",
            diameter_mm=3.5,
            length_mm=8.0,
            platform_diameter_mm=3.6,
            apical_diameter_mm=3.0,
            crest_point_ras=[7.5, 5.5, 5.0],  # z=20 in voxels (top of slab 12:20)
            apex_point_ras=[7.5, 5.5, -3.0],
            axis_apical_ras=[0.0, 0.0, -1.0],
            ring_outer_diameter_mm=4.4,
            ring_inner_diameter_mm=3.6,
            ring_height_mm=3.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            stl_p, json_p, sleeves = gen.export_surgical_guide_stl_and_json(
                arch_pts,
                [imp],
                Path(tmp),
                preset_id="test_slab",
                mask_zyx=mask,
                affine=aff,
                smooth=False,
                body_mode="auto",
            )
            self.assertTrue(stl_p.is_file())
            data = json.loads(json_p.read_text(encoding="utf-8"))
            self.assertIn("guide_build", data)
            self.assertEqual(data["guide_build"]["body_mode"], "intaglio")
            self.assertTrue(data["guide_build"]["sleeve_channels_open"])
            self.assertEqual(data["guide_build"]["island_count"], 1)
            # One shared guide-level seating axis, not a per-implant draw direction
            self.assertEqual(len(data["guide_build"]["seating_axis_ras"]), 3)
            self.assertAlmostEqual(data["guide_build"]["seating_axis_ras"][2], -1.0, places=3)
            self.assertIsNotNone(data["guide_build"]["undercut_blocked_fraction"])
            self.assertEqual(sleeves[0].sleeve_kind, "metal_ring")

            rep = gen.last_build_report
            assert isinstance(rep, GuideBuildReport)
            self.assertGreater(rep.volume_mm3, 100.0)
            self.assertIn("base derived from segmentation; verify fit", rep.warnings)

            # Verify shell sits strictly coronal to slab top (z=5.0 mm -> intaglio >= 5.0 + 0.25 - 0.35 mm)
            verts, _faces, _ = gen.generate_surgical_guide(
                arch_pts, [imp], preset_id="test_slab", mask_zyx=mask, affine=aff, smooth=False
            )
            self.assertGreater(float(verts[:, 2].max()), 6.5)

    def test_empty_mask_falls_back_cleanly_to_analytic_with_warning(self) -> None:
        empty = np.zeros((20, 20, 20), dtype=np.uint8)
        aff = np.diag([0.5, 0.5, 0.5, 1.0])
        arch_pts = np.array(
            [[2.0, 4.0, 5.0], [4.0, 4.0, 5.0], [6.0, 4.0, 5.0], [8.0, 4.0, 5.0]], dtype=np.float64
        )
        gen = SurgicalGuideGenerator()
        verts, faces, _ = gen.generate_surgical_guide(
            arch_pts, [], mask_zyx=empty, affine=aff, body_mode="intaglio"
        )
        self.assertGreater(len(verts), 10)
        self.assertGreater(len(faces), 10)
        assert gen.last_build_report is not None
        self.assertEqual(gen.last_build_report.body_mode, "analytic")
        self.assertTrue(any("analytic" in w.lower() for w in gen.last_build_report.warnings))

    def test_surface_extraction_works_without_skimage_or_vtk(self) -> None:
        """CI installs only ruff/numpy/scipy: the pure-NumPy surface fallback must still emit a mesh."""
        from unittest.mock import patch

        from CBCTPlannerLib.adt.geometry import extract_isosurface_from_label

        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[4:12, 4:12, 4:12] = 1
        aff = np.diag([0.5, 0.5, 0.5, 1.0])
        with patch.dict(sys.modules, {"skimage": None, "skimage.measure": None, "vtk": None}):
            verts, faces = extract_isosurface_from_label(mask, aff, [1])
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)

    def test_manifold_gate_distinguishes_closed_vs_broken_faces(self) -> None:
        good_tet = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
        self.assertTrue(check_edge_manifold(good_tet))
        broken = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        self.assertFalse(check_edge_manifold(broken))
        self.assertFalse(check_edge_manifold(np.zeros((0, 3), dtype=np.int64)))


if __name__ == "__main__":
    unittest.main()

"""Print layout rules and panoramic strip geometry."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.core.print_layout import slice_count_for_selection  # noqa: E402
from CBCTPlannerLib.core.labels import sort_teeth_anatomically  # noqa: E402
from CBCTPlannerLib.core_print_render import render_panoramic_strip  # noqa: E402


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

    def test_invalid_panoramic_is_explicitly_unavailable(self) -> None:
        render = render_panoramic_strip(np.empty((0, 0)), column_spacing_mm=0.5, tick_arcs_mm=[0.0])
        self.assertFalse(render.available)
        self.assertIsNotNone(render.reason)


if __name__ == "__main__":
    unittest.main()

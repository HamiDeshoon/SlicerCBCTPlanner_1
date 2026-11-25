import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "CBCTPlanner"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CBCTPlannerLib.core_print_render import render_cross_section  # noqa: E402


class TestInlineLabelsAndCustomOverlays(unittest.TestCase):
    def test_inline_line_labels_and_custom_overlays_render(self) -> None:
        u = np.linspace(-10.0, 10.0, 41)
        v = np.linspace(-15.0, 15.0, 61)
        cs = SimpleNamespace(
            u_coords=u,
            v_coords=v,
            img_cs=np.full((len(v), len(u)), 200.0, dtype=np.float32),
            mask_cs=np.zeros((len(v), len(u)), dtype=np.int16),
            crest_uv=(0.0, 5.0),
            basal_uv=(0.0, -8.0),
            width_endpoints_uv={0.0: ((-3.5, 4.0), (3.5, 4.0))},
            canal_centroid_uv=(0.0, -9.5),
            vital_contour_uv=None,
            sinus_floor_uv=None,
            crest_profile_uv=None,
            ridge_axis_angle_deg=0.0,
            is_mandible=True,
            measurements={},
        )
        meas = SimpleNamespace(
            bone_height_mm=13.0,
            crest_width_mm=7.0,
            bone_width_at_depths_mm={0.0: 7.0},
            misch_density_class="D2",
            provenance="label_derived",
            withheld_reason=None,
        )
        custom = [
            {"kind": "line", "p1_uv": (-4.0, 0.0), "p2_uv": (4.0, 0.0), "text": "Check ridge"},
            {"kind": "circle", "center_uv": (2.0, -3.0), "radius_mm": 2.0, "text": "Abnormal?"},
        ]
        rnd = render_cross_section(cs, meas, scale=3, show_footer=False, inline_line_labels=True, custom_overlays=custom)
        self.assertTrue(rnd.available)
        texts = [t for _x, _y, t in rnd.labels]
        self.assertTrue(any(t.startswith("H:") for t in texts))
        self.assertTrue(any(t.startswith("W0 ") for t in texts))
        self.assertIn("IAN", texts)
        self.assertIn("Check ridge", texts)
        self.assertIn("Abnormal?", texts)


if __name__ == "__main__":
    unittest.main()

"""Tests for Phase 1 contracts, case_state round-trips, and Phase 3 OPT-01/OPT-03 caches."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.PanoramicCurveLogic import PanoramicCurveLogic  # noqa: E402
from CBCTPlannerLib.core.case_state import (  # noqa: E402
    deserialize_overrides_payload,
    serialize_corrected_keypoints_jsonl,
    serialize_overrides_payload,
)
from CBCTPlannerLib.core.contracts import (  # noqa: E402
    SOURCE_ESTIMATED,
    SOURCE_INTERPOLATED,
    SOURCE_MANUALLY_CORRECTED,
    SOURCE_MEASURED,
    SOURCE_MODEL_PREDICTED,
    SOURCE_NOT_AVAILABLE,
    StructuredMeasurementEntry,
    StructuredReportEnvelope,
    UNIT_MM,
    normalize_provenance,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


class ContractsAndCaseStateTests(unittest.TestCase):
    def test_provenance_normalization_and_envelope_serialization(self) -> None:
        self.assertEqual(normalize_provenance("label_derived", value=10.5), SOURCE_MEASURED)
        self.assertEqual(normalize_provenance("ai_stunet", value=10.5), SOURCE_MODEL_PREDICTED)
        self.assertEqual(normalize_provenance("interpolated_boundary", value=10.5), SOURCE_INTERPOLATED)
        self.assertEqual(normalize_provenance("estimated", value=10.5), SOURCE_ESTIMATED)
        self.assertEqual(
            normalize_provenance("label_derived", manually_corrected=True, value=10.5),
            SOURCE_MANUALLY_CORRECTED,
        )
        self.assertEqual(normalize_provenance("label_derived", value=None), SOURCE_NOT_AVAILABLE)

        entry = StructuredMeasurementEntry(
            name="bone_height_h",
            value=12.3456,
            unit=UNIT_MM,
            source_type="label_derived",
            method="caliper_ian_inset",
            structure="mandible",
            fdi="46",
            uncertainty_mm=0.5,
        )
        env = StructuredReportEnvelope(measurements=[entry])
        d = env.to_dict()
        self.assertEqual(d["schema_version"], "cbctplanner-report/1.0")
        self.assertEqual(d["measurements"][0]["unit"], "mm")
        self.assertEqual(d["measurements"][0]["source_type"], SOURCE_MEASURED)
        self.assertEqual(d["measurements"][0]["value"], 12.346)

    def test_case_state_overrides_and_jsonl_round_trip(self) -> None:
        overrides = {
            46: {
                "h_mm": 11.2,
                "w0_mm": 7.4,
                "w0_start_uv": (-3.7, -1.0),
                "w0_end_uv": (3.7, -1.0),
                "custom_overlays": [{"kind": "line", "p0": [0.0, 0.0], "p1": [1.0, -5.0]}],
            }
        }
        payload = serialize_overrides_payload(overrides)
        restored = deserialize_overrides_payload(payload + " ")
        self.assertEqual(restored[46]["h_mm"], 11.2)
        self.assertEqual(restored[46]["w0_start_uv"], (-3.7, -1.0))
        self.assertEqual(len(restored[46]["custom_overlays"]), 1)

        jsonl = serialize_corrected_keypoints_jsonl(overrides)
        self.assertIn('"fdi": 46', jsonl)
        self.assertIn('"custom_overlays"', jsonl)


class PanoramicSlabCacheInvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        PanoramicCurveLogic.clear_panoramic_caches()
        img = np.zeros((24, 32, 32), dtype=np.float32)
        img[8:16, 12:18, 6:26] = 600.0
        img[10:14, 14:16, 12:20] = 1800.0
        self.img = img
        self.aff = np.eye(4, dtype=np.float64)
        self.vol = CBCTVolume(self.img, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3), self.aff)
        self.pts = np.array([[8.0, 15.0, 12.0], [13.0, 14.0, 12.0], [19.0, 14.0, 12.0], [24.0, 15.0, 12.0]])
        self.tang = np.tile(np.array([[-1.0, 0.0, 0.0]]), (4, 1))
        self.norm = np.tile(np.array([[0.0, -1.0, 0.0]]), (4, 1))
        self.bino = np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1))

    def test_cache_keys_isolate_modes_offsets_affine_and_mutations(self) -> None:
        raysum = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol, self.pts, self.tang, self.norm, self.bino, projection_mode="RaySum"
        )
        mip = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol, self.pts, self.tang, self.norm, self.bino, projection_mode="MIP"
        )
        minip = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol, self.pts, self.tang, self.norm, self.bino, projection_mode="MinIP"
        )
        mean = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol, self.pts, self.tang, self.norm, self.bino, projection_mode="Mean"
        )
        self.assertGreater(float(np.max(np.abs(mip - raysum))), 50.0)
        self.assertGreater(float(np.max(np.abs(mip - minip))), 100.0)
        self.assertGreater(float(np.max(np.abs(raysum - mean))), 10.0)

        # Returned copy must not mutate internal cache
        mip_before = mip.copy()
        mip[0, 0] = -99999.0
        mip_again = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol, self.pts, self.tang, self.norm, self.bino, projection_mode="MIP"
        )
        np.testing.assert_allclose(mip_again, mip_before)

        # Offset invalidation
        shifted = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol,
            self.pts,
            self.tang,
            self.norm,
            self.bino,
            projection_mode="MIP",
            buccolingual_offset_mm=4.0,
        )
        self.assertEqual(shifted.shape, mip_before.shape)

        # Single-voxel mutation invalidation
        self.vol.image[12, 15, 16] = 3500.0
        mip_mutated = PanoramicCurveLogic.sample_panoramic_slab_array(
            self.vol, self.pts, self.tang, self.norm, self.bino, projection_mode="MIP"
        )
        self.assertGreater(float(np.max(mip_mutated) - np.max(mip_before)), 200.0)

        # Affine mutation invalidation
        shifted_aff = self.aff.copy()
        shifted_aff[0, 3] = 30.0
        vol_shifted = CBCTVolume(self.img, (1.0, 1.0, 1.0), (30.0, 0.0, 0.0), np.eye(3), shifted_aff)
        mip_shifted_aff = PanoramicCurveLogic.sample_panoramic_slab_array(
            vol_shifted, self.pts, self.tang, self.norm, self.bino, projection_mode="MIP"
        )
        self.assertGreater(float(np.max(np.abs(mip_shifted_aff - mip_mutated))), 100.0)


if __name__ == "__main__":
    unittest.main()

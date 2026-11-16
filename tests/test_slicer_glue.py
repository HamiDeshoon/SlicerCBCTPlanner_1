"""Mock-harness tests for Slicer-only glue that cannot run in headless CI.

Covers: VR segment-visibility isolation/restoration and segment re-encoding when
the volume is canonicalized to RAS.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import CBCTPlanner as planner_module  # noqa: E402
from CBCTPlanner import CBCTPlannerLogic  # noqa: E402
from CBCTPlannerLib.core.orientation import apply_ops, canonicalize_to_ras  # noqa: E402


class _Segment:
    def __init__(self, name: str, visible: bool = True, fdi: str | None = None) -> None:
        self._name = name
        self._visible = visible
        self._fdi = fdi

    def GetName(self) -> str:
        return self._name

    def GetSegmentVisibility(self) -> bool:
        return self._visible

    def GetTag(self, tag: str) -> str | None:
        return self._fdi if tag in ("CBCTPlanner.FDI", "FDI") else None


class _Segmentation:
    def __init__(self, segments: list[_Segment]) -> None:
        self._segments = segments

    def GetNumberOfSegments(self) -> int:
        return len(self._segments)

    def GetNthSegment(self, index: int) -> _Segment:
        return self._segments[index]

    def GetNthSegmentID(self, index: int) -> str:
        return f"Segment_{index}"


class _SegmentationNode:
    def __init__(self, segments: list[_Segment]) -> None:
        self._segmentation = _Segmentation(segments)
        self.visibility: dict[str, bool] = {
            f"Segment_{i}": seg.GetSegmentVisibility() for i, seg in enumerate(segments)
        }
        self._by_id = {f"Segment_{i}": seg for i, seg in enumerate(segments)}
        self.reference_volume = None
        self.attrs: dict[str, str] = {}

    def GetSegmentation(self) -> _Segmentation:
        return self._segmentation

    def SetSegmentVisibility(self, segment_id: str, visible: bool) -> None:
        self.visibility[segment_id] = bool(visible)
        if segment_id in self._by_id:
            self._by_id[segment_id]._visible = bool(visible)

    def SetReferenceImageGeometryParameterFromVolumeNode(self, volume: object) -> None:
        self.reference_volume = volume

    def SetAttribute(self, key: str, value: str) -> None:
        self.attrs[key] = value


class VRSegmentVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.volume = object()
        self.node = _SegmentationNode(
            [
                _Segment("Tooth 11", True, "11"),
                _Segment("Tooth 46", True, "46"),
                _Segment("Mandible", True),
            ]
        )
        self.logic = object.__new__(CBCTPlannerLogic)

    @contextmanager
    def _active_slicer(self):
        with patch.object(planner_module, "slicer", SimpleNamespace()):
            yield

    def test_isolates_quadrant_and_restores_every_segment(self) -> None:
        with self._active_slicer():
            with self.logic.quadrant_vr_isolation(self.node, [11]):
                self.assertEqual(
                    [self.node.visibility[f"Segment_{i}"] for i in range(3)],
                    [True, False, True],
                )
        self.assertEqual([self.node.visibility[f"Segment_{i}"] for i in range(3)], [True, True, True])

    def test_restores_originally_hidden_segments(self) -> None:
        self.node.SetSegmentVisibility("Segment_1", False)
        with self._active_slicer():
            with self.logic.quadrant_vr_isolation(self.node, [11]):
                self.assertFalse(self.node.visibility["Segment_1"])
        self.assertEqual([self.node.visibility[f"Segment_{i}"] for i in range(3)], [True, False, True])

    def test_restores_visibility_when_body_raises(self) -> None:
        with self._active_slicer():
            with self.assertRaises(RuntimeError):
                with self.logic.quadrant_vr_isolation(self.node, [11]):
                    raise RuntimeError("capture failed")
        self.assertEqual([self.node.visibility[f"Segment_{i}"] for i in range(3)], [True, True, True])


class _VolumeNode:
    def __init__(self, name: str = "CBCT") -> None:
        self._name = name
        self.attrs: dict[str, str] = {}
        self.matrix = None

    def GetName(self) -> str:
        return self._name

    def GetTransformNodeID(self) -> str:
        return ""

    def GetDisplayNode(self) -> None:
        return None

    def SetAttribute(self, key: str, value: str) -> None:
        self.attrs[key] = value

    def SetIJKToRASMatrix(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix)


class OrientationSegmentReencodeTests(unittest.TestCase):
    def test_standardize_reencodes_each_segment_with_permuted_ops(self) -> None:
        arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        affine = np.diag([-1.0, 1.0, -1.0, 1.0])
        result = canonicalize_to_ras(arr, affine)
        self.assertTrue(result.ops)

        seg_arrays = {"Segment_0": (arr > 20).astype(np.uint8), "Segment_1": (arr < 10).astype(np.uint8)}
        written: dict[str, np.ndarray] = {}
        target = _VolumeNode("CBCT_RAS")
        fake_slicer = SimpleNamespace(
            mrmlScene=SimpleNamespace(AddNewNodeByClass=lambda _cls, _name: target),
            util=SimpleNamespace(
                updateVolumeFromArray=lambda _node, _image: None,
                vtkMatrixFromArray=lambda matrix: matrix,
                arrayFromSegmentBinaryLabelmap=lambda _node, segment_id, _vol: seg_arrays[segment_id],
                updateSegmentBinaryLabelmapFromArray=lambda image, _node, segment_id, _vol: (
                    written.__setitem__(segment_id, np.asarray(image))
                ),
            ),
        )
        segmentation = _SegmentationNode([_Segment("Tooth 11", True, "11"), _Segment("Tooth 21", True, "21")])

        logic = object.__new__(CBCTPlannerLogic)
        with (
            patch.object(planner_module, "slicer", fake_slicer),
            patch.object(CBCTPlannerLogic, "_volume_array_and_affine", lambda _self, _node: (arr, affine)),
            patch.object(
                CBCTPlannerLogic, "_extract_combined_label_mask", lambda _self, _v, _s: (None, True)
            ),
        ):
            updated_volume, report = logic.standardize_orientation(_VolumeNode(), segmentation)

        self.assertIs(updated_volume, target)
        self.assertEqual(set(written), set(seg_arrays))
        for segment_id, source in seg_arrays.items():
            np.testing.assert_array_equal(written[segment_id], apply_ops(source, result.ops))
        self.assertIs(segmentation.reference_volume, target)
        self.assertEqual(segmentation.attrs.get("CBCTPlanner.Orientation"), "RAS")
        self.assertEqual(target.attrs.get("CBCTPlanner.Orientation"), "RAS")
        self.assertTrue(any("2 segment(s) re-encoded" in note for note in report.notes))


if __name__ == "__main__":
    unittest.main()

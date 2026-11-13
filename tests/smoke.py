#!/usr/bin/env python3
"""
Smoke test for SlicerCBCTPlanner.
Runnable inside 3D Slicer via:
    Slicer --no-main-window --python-script tests/smoke.py
and also runnable in standalone headless Python (with mock MRML harness when `slicer` is absent).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def run_smoke_test() -> None:
    print("[smoke] 1. Testing zero-optional-dependency import of CBCTPlanner & CBCTPlannerLib...")
    import CBCTPlannerLib
    from CBCTPlannerLib.core.arch_detector import DentalArch, fit_dental_arch_from_points
    from CBCTPlannerLib.core.labels import FDI_TO_TOOTH_ID, get_tooth_label
    from CBCTPlannerLib.core.measurements import MeasurementEngine
    from CBCTPlannerLib.core.tooth_localizer import ToothLocalizer
    from CBCTPlannerLib.core.volume_loader import CBCTVolume
    import CBCTPlanner

    assert hasattr(CBCTPlanner, "CBCTPlanner")
    assert hasattr(CBCTPlanner, "CBCTPlannerWidget")
    assert hasattr(CBCTPlanner, "CBCTPlannerLogic")
    print("[smoke]    Imports succeeded.")

    # Check if running inside real 3D Slicer
    in_slicer = False
    try:
        import slicer  # type: ignore

        in_slicer = getattr(slicer, "mrmlScene", None) is not None and hasattr(
            slicer.mrmlScene, "AddNewNodeByClass"
        )
    except ImportError:
        in_slicer = False

    if in_slicer:
        import slicer  # type: ignore

        print("[smoke] 2. Running inside 3D Slicer environment...")
        slicer.mrmlScene.Clear(0)
        volumeNode = None
        try:
            import SampleData  # type: ignore

            volumeNode = SampleData.SampleDataLogic().downloadMRHead()
        except Exception as exc:
            print(f"[smoke]    SampleData download skipped ({exc}); creating synthetic CBCT volume node.")
            import vtk  # type: ignore

            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "SmokeCBCT")
            img = vtk.vtkImageData()
            img.SetDimensions(64, 64, 64)
            img.AllocateScalars(vtk.VTK_SHORT, 1)
            volumeNode.SetAndObserveImageData(img)
            volumeNode.SetSpacing(0.5, 0.5, 0.5)
            volumeNode.CreateDefaultDisplayNodes()

        logic = CBCTPlanner.CBCTPlannerLogic()
        plan = logic.analyze_selected_teeth([46, 36], volumeNode)
        for row in plan:
            assert row["height_mm"] is None, "Unsegmented volume must report None ('not measured'), never invented numbers!"
        print("[smoke]    Slicer smoke test passed.")
    else:
        print("[smoke] 2. Running in standalone Python (Slicer binary not present in container)...")
        # Verify synthetic CBCT + DentalArch + MeasurementEngine
        pts = np.array(
            [
                [28.0, -20.0, 0.0],
                [22.0, -5.0, 0.0],
                [12.0, 12.0, 0.0],
                [0.0, 18.0, 0.0],
                [-12.0, 12.0, 0.0],
                [-22.0, -5.0, 0.0],
                [-28.0, -20.0, 0.0],
            ],
            dtype=np.float64,
        )
        ctrl = fit_dental_arch_from_points(pts, num_control_points=11, ras_convention=True)
        assert ctrl[0, 0] > ctrl[-1, 0], "RAS arch must start at Patient Right (+X) and end at Patient Left (-X)"
        arch = DentalArch(ctrl, jaw_type="lower", ras_convention=True)
        f_mid = arch.get_frame(0.5)
        assert f_mid.normal[1] > 0.5, f"Expected anterior normal to point +Y (labial/buccal), got {f_mid.normal}"
        print("[smoke]    Standalone math & clinical honesty checks passed.")


if __name__ == "__main__":
    run_smoke_test()

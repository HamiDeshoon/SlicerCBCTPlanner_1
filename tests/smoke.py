#!/usr/bin/env python3
"""
Smoke and integration test runner for SlicerCBCTPlanner (O9).
Runnable inside 3D Slicer via:
    Slicer --no-main-window --python-script tests/smoke.py
and also runnable in standalone headless Python:
    python3 tests/smoke.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def run_smoke_test() -> None:
    print("[smoke] 1. Testing zero-optional-dependency import of CBCTPlanner & CBCTPlannerLib...")
    import CBCTPlanner
    import CBCTPlannerLib

    assert hasattr(CBCTPlanner, "CBCTPlanner")
    assert hasattr(CBCTPlanner, "CBCTPlannerWidget")
    assert hasattr(CBCTPlanner, "CBCTPlannerLogic")
    assert hasattr(CBCTPlanner, "CBCTPlannerTest")
    assert hasattr(CBCTPlannerLib, "CanalTracerLogic")
    assert hasattr(CBCTPlannerLib, "ImplantPlannerLogic")
    print("[smoke]    Imports succeeded.")

    in_slicer = False
    try:
        import slicer  # type: ignore

        in_slicer = getattr(slicer, "mrmlScene", None) is not None and hasattr(
            slicer.mrmlScene, "AddNewNodeByClass"
        )
    except ImportError:
        in_slicer = False

    if in_slicer:
        print("[smoke] 2. Running CBCTPlannerTest inside live 3D Slicer environment...")
        tester = CBCTPlanner.CBCTPlannerTest()
        tester.runTest()
        print("[smoke]    Live 3D Slicer smoke test passed.")
    else:
        print("[smoke] 2. Running full unittest suite in standalone Python (Slicer binary not present in container)...")
        suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
        print("[smoke]    All standalone unit & integration tests passed.")


if __name__ == "__main__":
    run_smoke_test()

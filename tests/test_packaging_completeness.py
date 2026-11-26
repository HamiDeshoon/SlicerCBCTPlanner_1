"""Packaging completeness, single-version-source, and clean-install tests."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
MODULE_CMAKE = MODULE_DIR / "CMakeLists.txt"
ROOT_CMAKE = REPO_ROOT / "CMakeLists.txt"
VERSION_FILE = MODULE_DIR / "Resources" / "VERSION"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"


def _parse_cmake_list(text: str, var_name: str, module_name: str = "CBCTPlanner") -> list[str]:
    pattern = rf"set\(\s*{re.escape(var_name)}\s+(.*?)\)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise AssertionError(f"CMake variable {var_name!r} not found in {MODULE_CMAKE}")
    items = []
    for raw in match.group(1).split():
        cleaned = raw.strip().replace("${MODULE_NAME}", module_name)
        if cleaned:
            items.append(cleaned.replace("\\", "/"))
    return items


class PackagingCompletenessTests(unittest.TestCase):
    def test_version_single_source_of_truth(self) -> None:
        self.assertTrue(VERSION_FILE.is_file(), f"Missing version file: {VERSION_FILE}")
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

        added_path = False
        module_str = str(MODULE_DIR)
        if module_str not in sys.path:
            sys.path.insert(0, module_str)
            added_path = True
        try:
            from CBCTPlannerLib.Utils import SOFTWARE_VERSION
        finally:
            if added_path and module_str in sys.path:
                sys.path.remove(module_str)

        self.assertEqual(SOFTWARE_VERSION, version)

        root_cmake_text = ROOT_CMAKE.read_text(encoding="utf-8")
        self.assertIn("CBCTPlanner/Resources/VERSION", root_cmake_text)
        self.assertIn("EXTENSION_VERSION", root_cmake_text)

        pyproject_text = PYPROJECT_FILE.read_text(encoding="utf-8")
        self.assertIn('dynamic = ["version"]', pyproject_text)
        self.assertIn('file = "CBCTPlanner/Resources/VERSION"', pyproject_text)

    def test_every_python_module_and_resource_listed_in_cmake(self) -> None:
        cmake_text = MODULE_CMAKE.read_text(encoding="utf-8")
        scripts = _parse_cmake_list(cmake_text, "MODULE_PYTHON_SCRIPTS")
        resources = _parse_cmake_list(cmake_text, "MODULE_PYTHON_RESOURCES")

        disk_py = sorted(
            p.relative_to(MODULE_DIR).as_posix()
            for p in MODULE_DIR.rglob("*.py")
            if "__pycache__" not in p.parts
        )
        missing_scripts = sorted(set(disk_py) - set(scripts))
        extra_scripts = sorted(set(scripts) - set(disk_py))
        self.assertEqual(missing_scripts, [], f"Unpackaged Python modules: {missing_scripts}")
        self.assertEqual(extra_scripts, [], f"Missing files listed in CMake: {extra_scripts}")

        disk_resources = sorted(
            p.relative_to(MODULE_DIR).as_posix()
            for p in (MODULE_DIR / "Resources").rglob("*")
            if p.is_file()
        )
        missing_res = sorted(set(disk_resources) - set(resources))
        extra_res = sorted(set(resources) - set(disk_resources))
        self.assertEqual(missing_res, [], f"Unpackaged Resources: {missing_res}")
        self.assertEqual(extra_res, [], f"Missing resources listed in CMake: {extra_res}")

    def test_clean_staged_install_imports_and_runs_isolated(self) -> None:
        cmake_text = MODULE_CMAKE.read_text(encoding="utf-8")
        staged_files = _parse_cmake_list(cmake_text, "MODULE_PYTHON_SCRIPTS") + _parse_cmake_list(
            cmake_text, "MODULE_PYTHON_RESOURCES"
        )
        expected_version = VERSION_FILE.read_text(encoding="utf-8").strip()

        with tempfile.TemporaryDirectory(prefix="cbct_clean_install_") as tmp:
            stage = Path(tmp) / "qt-scripted-modules"
            for rel in staged_files:
                src = MODULE_DIR / rel
                dst = stage / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            script = (
                "import importlib, json, os, sys; "
                f"expected_stage = os.path.normcase(os.path.realpath(r'{stage}')); "
                "import CBCTPlannerLib, CBCTPlanner, cli_batch; "
                "assert os.path.normcase(os.path.realpath(CBCTPlannerLib.__file__)).startswith(expected_stage), CBCTPlannerLib.__file__; "
                "assert CBCTPlannerLib.SOFTWARE_VERSION == "
                + repr(expected_version)
                + "; "
                "from CBCTPlannerLib.ImplantPlanner import ImplantPlannerLogic; "
                "from CBCTPlannerLib.Phase7Tools import SurgicalGuideGenerator, ProstheticPlanner; "
                "from CBCTPlannerLib.Utils import iconPath; "
                "assert os.path.isfile(iconPath('CBCTPlanner.png')); "
                "imp_logic = ImplantPlannerLogic(); "
                "assert len(imp_logic.library) > 0; "
                "guide_gen = SurgicalGuideGenerator(); "
                "assert len(guide_gen.presets) > 0; "
                "prosth = ProstheticPlanner(); "
                "assert len(prosth.crowns) > 0 and len(prosth.abutments) > 0; "
                "import numpy as np; "
                "from CBCTPlannerLib.core.volume_loader import CBCTVolume; "
                "from CBCTPlannerLib.core.arch_detector import DentalArch; "
                "from CBCTPlannerLib.core.measurements import MeasurementEngine; "
                "from CBCTPlannerLib.PanoramicCurveLogic import PanoramicCurveLogic; "
                "aff = np.eye(4, dtype=np.float64); "
                "img = np.full((24, 32, 32), -800.0, dtype=np.float32); "
                "img[6:18, 11:20, 8:24] = 900.0; "
                "mask = np.zeros_like(img, dtype=np.int32); "
                "mask[6:18, 11:20, 8:24] = 1; "
                "vol = CBCTVolume(img, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3), aff); "
                "arch = DentalArch(np.array([[10.0, 17.0, 12.0], [14.0, 15.0, 12.0], [18.0, 15.0, 12.0], [22.0, 17.0, 12.0]]), jaw_type='lower', ras_convention=True); "
                "eng = MeasurementEngine(vol, mask, arch=arch, arches={'lower': arch}, is_dental_segmentator_5class=False, extract_teeth_eagerly=False); "
                "meas, _ = eng.measure_at_arc_mm('lower', float(arch.total_arc_length) * 0.5); "
                "assert meas is not None and meas.bone_height_mm is not None; "
                "pts_p = np.asarray(arch.control_points, dtype=np.float64); "
                "tang = np.tile(np.array([[-1.0, 0.0, 0.0]]), (len(pts_p), 1)); "
                "norm = np.tile(np.array([[0.0, -1.0, 0.0]]), (len(pts_p), 1)); "
                "bino = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(pts_p), 1)); "
                "pano_arr = PanoramicCurveLogic.sample_panoramic_slab_array(vol, pts_p, tang, norm, bino, projection_mode='RaySum'); "
                "assert pano_arr.ndim == 2 and pano_arr.size > 0; "
                "modules = ["
                + ",".join(
                    repr(
                        s[:-3].replace("/", ".")
                        if not s.endswith("/__init__.py")
                        else s[: -len("/__init__.py")].replace("/", ".")
                    )
                    for s in _parse_cmake_list(cmake_text, "MODULE_PYTHON_SCRIPTS")
                    if s.endswith(".py")
                )
                + "]; "
                "[importlib.import_module(m) for m in modules]; "
                "print('INSTALLED_OK', len(modules))"
            )
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("PYTHONPATH", "PYTHONHOME")
            }
            if (Path(sys.base_prefix) / "Lib" / "encodings").exists():
                env["PYTHONHOME"] = str(Path(sys.base_prefix))
            env["PYTHONPATH"] = str(stage)
            env["PYTHONNOUSERSITE"] = "0"
            proc = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=str(tmp),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Clean staged import/execution failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )
            self.assertIn("INSTALLED_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()

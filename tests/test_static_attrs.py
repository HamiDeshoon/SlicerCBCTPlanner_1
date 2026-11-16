"""Static AST guards for dangling attributes and silent exception handlers.

These tests catch the class of bug that made Auto-Fit appear dead: a widget reads
`self.<attribute>` that is never created, and the resulting AttributeError is
swallowed by a bare `except Exception: pass`.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
WIDGET_FILE = ROOT / "CBCTPlanner" / "CBCTPlanner.py"
PRINT_FILE = ROOT / "CBCTPlanner" / "CBCTPlannerLib" / "PrintLayoutEngine.py"

# Attributes provided by ScriptedLoadableModuleWidget / VTKObservationMixin.
INHERITED_WIDGET_ATTRS = {
    "addObserver",
    "getParameterNode",
    "initializeParameterNode",
    "layout",
    "parent",
    "removeObserver",
    "removeObservers",
    "setParameterNode",
}

# Files whose `except Exception: pass` handlers must carry an explicit justification.
SILENT_EXCEPT_FILES = [
    WIDGET_FILE,
    ROOT / "CBCTPlanner" / "CBCTPlannerLib" / "PanoramicCurveLogic.py",
    ROOT / "CBCTPlanner" / "CBCTPlannerLib" / "PrintLayoutEngine.py",
    ROOT / "CBCTPlanner" / "CBCTPlannerLib" / "core" / "measurements.py",
]


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _class_members(cls: ast.ClassDef) -> set[str]:
    members: set[str] = set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            members.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    members.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            members.add(node.target.id)
    return members


def _self_attributes(cls: ast.ClassDef) -> tuple[set[str], set[str]]:
    assigned: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.attr)
            else:
                loaded.add(node.attr)
    return assigned, loaded


class TestStaticAttributes(unittest.TestCase):
    def test_widget_self_attributes_are_assigned_or_inherited(self) -> None:
        tree = ast.parse(WIDGET_FILE.read_text(encoding="utf-8"))
        cls = _class_node(tree, "CBCTPlannerWidget")
        assigned, loaded = _self_attributes(cls)
        known = assigned | _class_members(cls) | INHERITED_WIDGET_ATTRS
        missing = sorted(loaded - known)
        self.assertEqual(
            missing,
            [],
            f"CBCTPlannerWidget reads undefined attributes: {missing}",
        )

    def test_logic_methods_called_actually_exist(self) -> None:
        tree = ast.parse(WIDGET_FILE.read_text(encoding="utf-8"))
        logic_members = _class_members(_class_node(tree, "CBCTPlannerLogic"))
        missing: set[str] = set()

        for path in (WIDGET_FILE, PRINT_FILE):
            file_tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(file_tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if not isinstance(func.value, ast.Attribute):
                    continue
                if func.value.attr != "logic":
                    continue
                if func.attr not in logic_members:
                    missing.add(f"{path.name}: logic.{func.attr}")

        self.assertEqual(sorted(missing), [], f"Calls to missing CBCTPlannerLogic members: {sorted(missing)}")

    def test_no_bare_except_exception_pass(self) -> None:
        offenders: list[str] = []
        for path in SILENT_EXCEPT_FILES:
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler) or len(node.body) != 1:
                    continue
                if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                    continue
                body = node.body[0]
                if not isinstance(body, ast.Pass):
                    continue
                handler_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                body_line = lines[body.lineno - 1] if body.lineno - 1 < len(lines) else ""
                if "silent-ok" not in handler_line and "silent-ok" not in body_line:
                    offenders.append(f"{path.name}:{body.lineno}")
        self.assertEqual(
            offenders, [], f"Silent except-pass handlers need `# silent-ok: reason`: {offenders}"
        )

    def test_panoramic_flip_uses_shared_sampler_helpers(self) -> None:
        source = (ROOT / "CBCTPlanner" / "CBCTPlannerLib" / "PanoramicCurveLogic.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(source.count("_build_slab_geometry("), 3)  # definition + sync + async
        self.assertEqual(source.count("_finalize_projection("), 3)


if __name__ == "__main__":
    unittest.main()

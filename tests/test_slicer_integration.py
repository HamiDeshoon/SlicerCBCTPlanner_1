"""
Integration and feature tests for SlicerCBCTPlanner covering:
  - Phase 2 MRML/UI behavior (B10, B11, B14, B15, B16, B17, B18)
  - Phase 3 Performance & Slicer Phantom Integration (O1..O9)
  - Phase 4 UX & Reporting (B21, B22, F10)
  - Phase 5 P1 Features (F1..F5)
  - Phase 6 P2 Features (F6..F13, F18, F19)
Runnable via:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlanner import CBCTPlannerLogic  # noqa: E402
from CBCTPlannerLib import (  # noqa: E402
    CanalTracerLogic,
    CrossSectionNavigator,
    CursorSettings,
    ImplantPlannerLogic,
    PanoramicCurveLogic,
    UndoRedoManager,
    capture_slice_view_png_base64,
    compute_angle_between_three_points_deg,
    compute_occlusal_plane_from_segmentation,
    compute_pharyngeal_airway_metrics,
    compute_surface_to_surface_distance_mm,
    compute_tmj_condyle_frames,
    detect_automated_dental_tools_modules,
    export_reformatted_dicom_series,
)
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


class _MockDisplayNode:
    def __init__(self) -> None:
        self.color = (1.0, 1.0, 1.0)
        self.visible = True
        self.visible2d = True
        self.visible3d = True
        self.view_ids: list[str] = []
        self.line_diameter = 1.0
        self.start_modify_calls = 0

    def StartModify(self) -> int:
        self.start_modify_calls += 1
        return 0

    def EndModify(self, _prev: int) -> None:
        pass

    def SetColor(self, r: float, g: float, b: float) -> None:
        self.color = (float(r), float(g), float(b))

    def SetSelectedColor(self, r: float, g: float, b: float) -> None:
        pass

    def SetCurveLineSizeMode(self, _mode: int) -> None:
        pass

    def SetLineDiameter(self, d: float) -> None:
        self.line_diameter = float(d)

    def SetLineWidth(self, _w: float) -> None:
        pass

    def SetOpacity(self, _op: float) -> None:
        pass

    def SetVisibility(self, v: bool) -> None:
        self.visible = bool(v)

    def SetVisibility2D(self, v: bool) -> None:
        self.visible2d = bool(v)

    def SetVisibility3D(self, v: bool) -> None:
        self.visible3d = bool(v)

    def SetPropertiesLabelVisibility(self, _v: bool) -> None:
        pass

    def SetPointLabelsVisibility(self, _v: bool) -> None:
        pass

    def SetViewNodeIDs(self, ids: list[str]) -> None:
        self.view_ids = list(ids)


class _MockMRMLNode:
    def __init__(self, node_id: str, name: str, class_name: str) -> None:
        self._id = node_id
        self._name = name
        self._class = class_name
        self._mtime = 1
        self._attrs: dict[str, str] = {}
        self._params: dict[str, str] = {}
        self._refs: dict[str, str] = {}
        self._scene: Optional["_MockMRMLScene"] = None
        self._pts: list[list[float]] = []
        self._labels: dict[int, str] = {}
        self._disp = _MockDisplayNode()
        self._observers: dict[int, tuple[int, object]] = {}
        self._next_obs_tag = 1
        self._transform_id: Optional[str] = None
        self.start_modify_calls = 0

    def GetID(self) -> str:
        return self._id

    def GetName(self) -> str:
        return self._name

    def SetName(self, new_name: str) -> None:
        self._name = str(new_name)

    def GetMTime(self) -> int:
        return self._mtime

    def Modified(self) -> None:
        self._mtime += 1

    def StartModify(self) -> int:
        self.start_modify_calls += 1
        return 0

    def EndModify(self, _prev: int) -> None:
        pass

    def SetAttribute(self, k: str, v: str) -> None:
        self._attrs[str(k)] = str(v)

    def GetAttribute(self, k: str) -> Optional[str]:
        return self._attrs.get(str(k))

    def SetParameter(self, k: str, v: str) -> None:
        self._params[str(k)] = str(v)

    def GetParameter(self, k: str) -> str:
        return self._params.get(str(k), "")

    def SetNodeReferenceID(self, role: str, target_id: Optional[str]) -> None:
        self._refs[str(role)] = str(target_id or "")

    def GetNodeReferenceID(self, role: str) -> str:
        return self._refs.get(str(role), "")

    def GetNodeReference(self, role: str) -> Optional["_MockMRMLNode"]:
        tid = self._refs.get(str(role), "")
        if tid and self._scene is not None:
            return self._scene.GetNodeByID(tid)
        return None

    def CreateDefaultDisplayNodes(self) -> None:
        pass

    def GetDisplayNode(self) -> _MockDisplayNode:
        return self._disp

    def SetAndObservePolyData(self, _poly: object) -> None:
        pass

    def SetLocked(self, _locked: bool) -> None:
        pass

    def GetNumberOfControlPoints(self) -> int:
        return len(self._pts)

    def RemoveAllControlPoints(self) -> None:
        self._pts.clear()
        self._mtime += 1

    def AddControlPoint(self, x: float, y: float, z: float) -> int:
        self._pts.append([float(x), float(y), float(z)])
        self._mtime += 1
        return len(self._pts) - 1

    def GetNthControlPointPositionWorld(self, idx: int, buf: list[float]) -> None:
        pt = self._pts[idx]
        buf[0], buf[1], buf[2] = pt[0], pt[1], pt[2]

    def SetNthControlPointPositionWorld(self, idx: int, x: float, y: float, z: float) -> None:
        self._pts[idx] = [float(x), float(y), float(z)]
        self._mtime += 1

    def SetNthControlPointLabel(self, idx: int, lbl: str) -> None:
        self._labels[int(idx)] = str(lbl)

    def AddObserver(self, evt: int, callback: object) -> int:
        tag = self._next_obs_tag
        self._next_obs_tag += 1
        self._observers[tag] = (evt, callback)
        return tag

    def RemoveObserver(self, tag: int) -> None:
        self._observers.pop(int(tag), None)

    def SetAndObserveTransformNodeID(self, tf_id: Optional[str]) -> None:
        self._transform_id = tf_id

    def GetTransformNodeID(self) -> Optional[str]:
        return self._transform_id

    def SetMatrixTransformToParent(self, _mat: object) -> None:
        pass


class _MockMRMLScene:
    def __init__(self) -> None:
        self._nodes_by_id: dict[str, _MockMRMLNode] = {}
        self._counter = 1

    def AddNewNodeByClass(self, class_name: str, name: str = "") -> _MockMRMLNode:
        nid = f"{class_name}_{self._counter}"
        self._counter += 1
        node = _MockMRMLNode(nid, name or nid, class_name)
        node._scene = self
        self._nodes_by_id[nid] = node
        return node

    def GetNodeByID(self, nid: str) -> Optional[_MockMRMLNode]:
        return self._nodes_by_id.get(str(nid))

    def GetFirstNodeByName(self, name: str) -> Optional[_MockMRMLNode]:
        for nd in self._nodes_by_id.values():
            if nd.GetName() == name:
                return nd
        return None

    def RemoveNode(self, node: _MockMRMLNode) -> None:
        if node is not None:
            self._nodes_by_id.pop(node.GetID(), None)


class _MockSliceNode:
    def __init__(self, tag: str) -> None:
        self._id = f"vtkMRMLSliceNode{tag}"
        self.orientation = "Axial"
        self.slice_offset = 0.0
        self.ntp: Optional[tuple[float, ...]] = None
        self.slab_enabled = False
        self.slab_type = 2
        self.slab_thickness = 0.5
        self.start_modify_calls = 0

    def GetID(self) -> str:
        return self._id

    def StartModify(self) -> int:
        self.start_modify_calls += 1
        return 0

    def EndModify(self, _prev: int) -> None:
        pass

    def Modified(self) -> None:
        pass

    def SetOrientationToAxial(self) -> None:
        self.orientation = "Axial"

    def SetOrientationToSagittal(self) -> None:
        self.orientation = "Sagittal"

    def SetOrientationToCoronal(self) -> None:
        self.orientation = "Coronal"

    def SetSliceOffset(self, z: float) -> None:
        self.slice_offset = float(z)

    def SetSliceToRASByNTP(self, *args: float) -> None:
        self.ntp = tuple(float(a) for a in args)

    def SetSlabReconstructionEnabled(self, enabled: bool) -> None:
        self.slab_enabled = bool(enabled)

    def SetSlabReconstructionType(self, t: int) -> None:
        self.slab_type = int(t)

    def SetSlabReconstructionThickness(self, th: float) -> None:
        self.slab_thickness = float(th)


class _MockSliceWidget:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._snode = _MockSliceNode(tag)
        self._bg_id: Optional[str] = None
        self.fit_calls = 0

    def mrmlSliceNode(self) -> _MockSliceNode:
        return self._snode

    def mrmlSliceCompositeNode(self) -> "_MockSliceWidget":
        return self

    def SetBackgroundVolumeID(self, vid: str) -> None:
        self._bg_id = str(vid)

    def GetBackgroundVolumeID(self) -> Optional[str]:
        return self._bg_id

    def sliceLogic(self) -> "_MockSliceWidget":
        return self

    def GetSliceNode(self) -> _MockSliceNode:
        return self._snode

    def FitSliceToAll(self) -> None:
        self.fit_calls += 1


class _MockSlicerEnv:
    def __init__(self) -> None:
        self.mrmlScene = _MockMRMLScene()
        self.slice_widgets = {
            t: _MockSliceWidget(t) for t in ("Red", "Green", "Yellow", "YellowMinus", "YellowPlus")
        }

        class _LM:
            def __init__(self, widgets: dict[str, _MockSliceWidget]) -> None:
                self._w = widgets
                self.threeDViewCount = 0

            def sliceWidget(self, name: str) -> Optional[_MockSliceWidget]:
                return self._w.get(name)

        class _App:
            def __init__(self, lm: _LM) -> None:
                self._lm = lm

            def layoutManager(self) -> _LM:
                return self._lm

        self.app = _App(_LM(self.slice_widgets))

        class _MarkupsDisp:
            UseLineDiameter = 1

        self.vtkMRMLMarkupsDisplayNode = _MarkupsDisp

        class _Util:
            @staticmethod
            def vtkMatrixFromArray(arr: np.ndarray) -> np.ndarray:
                return arr

        self.util = _Util()


class TestSlicerIntegrationAndFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.ras_arch_pts = np.array(
            [
                [28.0, -22.0, 0.0],
                [24.0, -8.0, 0.0],
                [16.0, 8.0, 0.0],
                [8.0, 16.0, 0.0],
                [0.0, 20.0, 0.0],
                [-8.0, 16.0, 0.0],
                [-16.0, 8.0, 0.0],
                [-24.0, -8.0, 0.0],
                [-28.0, -22.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_b10_b11_b14_b15_b16_b17_b18_mrml_slice_and_parameter_node_behaviors(self) -> None:
        """Verify B10 (zoom preserved), B11 (Red scroll preserved), B15 (volume rename safe), and B16 (ParameterNode round-trip)."""
        mock_slicer = _MockSlicerEnv()
        with (
            patch("CBCTPlanner.slicer", mock_slicer),
            patch("CBCTPlannerLib.PanoramicCurveLogic.slicer", mock_slicer),
            patch("CBCTPlannerLib.CrossSectionNavigator.slicer", mock_slicer),
        ):
            logic = CBCTPlannerLogic()
            pNode = mock_slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", "Params")
            logic._parameterNode = pNode
            logic._setDefaultParameters(pNode)

            self.assertEqual(pNode.GetAttribute("CBCTPlanner.DefaultsInitialized"), "1")
            prev_mod_calls = pNode.start_modify_calls
            logic._setDefaultParameters(pNode)
            self.assertEqual(pNode.start_modify_calls, prev_mod_calls)

            volNode = mock_slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ScanA")
            pNode.SetNodeReferenceID("InputVolume", volNode.GetID())
            curveNode = logic.panoramic.create_or_update_arch_curve(
                volNode, jaw="Lower", points_ras=self.ras_arch_pts, parameterNode=pNode
            )
            self.assertIsNotNone(curveNode)

            volNode.SetName("ScanB_Renamed")
            resolved_curve = logic.get_curve_node_for_jaw(volNode, "Lower", parameterNode=pNode)
            self.assertIs(resolved_curve, curveNode)

            panoNode = mock_slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Pano")
            panoNode.SetAttribute("CBCTPlanner.IsPanoramic", "1")
            panoNode.SetAttribute("CBCTPlanner.TotalArcMm", "95.0")
            panoNode.SetAttribute("CBCTPlanner.HeightMm", "50.0")
            self.assertTrue(logic.panoramic.display_panoramic_in_slice_view(panoNode, "Green"))
            self.assertEqual(mock_slicer.slice_widgets["Green"].fit_calls, 1)
            self.assertFalse(logic.panoramic.display_panoramic_in_slice_view(panoNode, "Green"))
            self.assertEqual(mock_slicer.slice_widgets["Green"].fit_calls, 1)

            red_snode = mock_slicer.slice_widgets["Red"].GetSliceNode()
            red_snode.SetSliceOffset(14.5)
            settings_drag = CursorSettings(
                sync_axial_slice=False, slab_enabled=True, slab_mode="MIP", slab_thickness_mm=2.5
            )
            logic.navigator.update_cursor_at_arc_mm(
                volNode, curveNode, arc_mm=40.0, settings=settings_drag, parameterNode=pNode
            )
            self.assertAlmostEqual(red_snode.slice_offset, 14.5)
            yellow_snode = mock_slicer.slice_widgets["Yellow"].GetSliceNode()
            self.assertTrue(yellow_snode.slab_enabled)
            self.assertAlmostEqual(yellow_snode.slab_thickness, 2.5)

            logic.jump_to_tooth_site(46, volNode, settings=CursorSettings(sync_axial_slice=True))
            self.assertAlmostEqual(red_snode.slice_offset, 0.0, places=4)

            custom_state = {
                "ActiveJaw": "Upper",
                "ArchMode": "Shared Occlusal Arch",
                "CrossSectionWidthMm": "32.0",
                "SlabEnabled": "true",
                "SlabMode": "MIP",
                "SlabThicknessMm": "3.0",
                "TickSpacingMm": "4.0",
                "CrossSectionTiltDeg": "-15.0",
                "LightboxStepMm": "3.5",
                "SelectedTeethFDI": "16,26,46",
            }
            logic.save_ui_state_to_parameter_node(pNode, custom_state)
            loaded_state = logic.load_ui_state_from_parameter_node(pNode)
            for k, v in custom_state.items():
                self.assertEqual(loaded_state[k], v)

    def test_o1_in_process_map_coordinates_panoramic_accuracy_and_timing(self) -> None:
        """O1: Verify in-process map_coordinates panoramic slab sampler matches analytical intensity and runs in < 250 ms."""
        z_grid, _, _ = np.meshgrid(
            np.arange(60, dtype=np.float32),
            np.arange(60, dtype=np.float32),
            np.arange(60, dtype=np.float32),
            indexing="ij",
        )
        img = 100.0 + 10.0 * z_grid
        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4, dtype=np.float64),
        )

        arch_pts = np.array(
            [
                [48.0, 15.0, 30.0],
                [38.0, 35.0, 30.0],
                [30.0, 45.0, 30.0],
                [22.0, 35.0, 30.0],
                [12.0, 15.0, 30.0],
            ],
            dtype=np.float64,
        )
        pano_logic = PanoramicCurveLogic()

        class _C:
            def GetID(self) -> str:
                return "C1"

            def GetMTime(self) -> int:
                return 1

            def GetNumberOfControlPoints(self) -> int:
                return len(arch_pts)

            def GetNthControlPointPositionWorld(self, i: int, b: list[float]) -> None:
                b[0], b[1], b[2] = float(arch_pts[i, 0]), float(arch_pts[i, 1]), float(arch_pts[i, 2])

            def GetCurveWorld(self) -> None:
                return None

        pts, _, tangents, normals, binormals = pano_logic.get_curve_points_and_frames(_C(), num_samples=50)
        t0 = time.perf_counter()
        proj_mean = pano_logic.sample_panoramic_slab_array(
            vol,
            pts,
            tangents,
            normals,
            binormals,
            slab_thickness_mm=10.0,
            height_mm=20.0,
            spacing_mm=1.0,
            projection_mode="Mean",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 250.0)
        mid_row = proj_mean.shape[0] // 2
        expected_val = proj_mean[mid_row, 0]
        self.assertLess(float(np.max(np.abs(proj_mean[mid_row, :] - expected_val))), 0.05)
        self.assertAlmostEqual(float(np.mean(proj_mean)), 400.0, delta=1.0)

    def test_o3_o8_zero_copy_volume_and_pyproject_ruff_config(self) -> None:
        """O3 & O8: Verify CBCTVolume zero-copy view and pyproject.toml ruff configuration."""
        raw = np.zeros((20, 20, 20), dtype=np.int16)
        vol = CBCTVolume(
            image=raw,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4),
            copy_image=False,
        )
        self.assertTrue(np.shares_memory(raw, vol.image))

        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.ruff]", pyproject)
        self.assertIn('target-version = "py39"', pyproject)

    def test_f1_mandibular_canal_and_root_canal_extraction_and_proximity_warning(self) -> None:
        """F1: Verify mandibular canal extraction from class 5 mask, root canal extraction, and 2 mm safety alert."""
        img = np.zeros((40, 40, 60), dtype=np.float32)
        mask = np.zeros((40, 40, 60), dtype=np.uint16)
        mask[11:14, 19:22, 35:52] = 5
        mask[11:14, 19:22, 8:25] = 5
        mask[18:28, 20:22, 42:44] = 146

        affine = np.eye(4, dtype=np.float64)
        affine[0, 3] = -30.0
        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(-30.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=affine,
        )

        canals = CanalTracerLogic.extract_canals_from_mask(vol, mask, is_5class=True)
        self.assertIn("Right", canals)
        self.assertIn("Left", canals)
        self.assertGreaterEqual(len(canals["Right"].control_points_ras), 4)

        c_pt = canals["Right"].control_points_ras[len(canals["Right"].control_points_ras) // 2]
        near_res = CanalTracerLogic.evaluate_canal_proximity(
            c_pt + np.array([0.0, 0.0, 1.2]), canals["Right"], safety_margin_mm=2.0
        )
        self.assertTrue(near_res.is_violation)
        self.assertIn("WARNING", near_res.warning_message)

        safe_res = CanalTracerLogic.evaluate_canal_proximity(
            c_pt + np.array([0.0, 0.0, 5.0]), canals["Right"], safety_margin_mm=2.0
        )
        self.assertFalse(safe_res.is_violation)

        root_canal = CanalTracerLogic.extract_root_canal_for_tooth(46, vol, mask)
        self.assertIsNotNone(root_canal)

    def test_f2_f11_f12_f19_implant_library_placement_parallelism_stl_and_confirmation_gate(self) -> None:
        """F2, F11, F12, F19: Verify generic JSON implant library, parallelism, canal clearance, STL export, and confirmation gate."""
        planner = ImplantPlannerLogic()
        self.assertGreaterEqual(len(planner.get_library_entries()), 10)

        T = np.array([-1.0, 0.0, 0.0])
        N = np.array([0.0, 1.0, 0.0])
        B = np.array([0.0, 0.0, 1.0])

        imp46 = planner.place_or_update_implant(
            fdi=46,
            crest_point_ras=[20.0, 10.0, 25.0],
            T=T,
            N=N,
            B=B,
            entry_id="generic_4.0x10.0",
            buccolingual_tilt_deg=0.0,
            mesiodistal_tilt_deg=0.0,
            is_ai_suggestion=False,
        )
        imp47 = planner.place_or_update_implant(
            fdi=47,
            crest_point_ras=[30.0, 10.0, 25.0],
            T=T,
            N=N,
            B=B,
            entry_id="generic_4.5x10.0",
            buccolingual_tilt_deg=0.0,
            mesiodistal_tilt_deg=10.0,
            is_ai_suggestion=True,
        )

        self.assertAlmostEqual(imp46.parallelism_deg_by_fdi["47"], 10.0, places=1)
        self.assertFalse(imp47.clinician_confirmed)

        self.assertEqual([i.fdi for i in planner.get_reportable_implants()], [46])
        planner.confirm_ai_suggestion(47)
        self.assertEqual([i.fdi for i in planner.get_reportable_implants()], [46, 47])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, stl_paths = planner.export_implants_json_and_stl(Path(tmpdir))
            self.assertTrue(json_path.exists())
            self.assertEqual(len(stl_paths), 2)
            stl_text = stl_paths[0].read_text(encoding="utf-8")
            self.assertIn("solid Implant_46", stl_text)
            self.assertIn("facet normal", stl_text)

    def test_f5_dynamic_lightbox_and_density_profile_along_line(self) -> None:
        """F5: Verify N-slice dynamic lightbox offsets and line intensity profile."""
        offsets_5 = CrossSectionNavigator.compute_dynamic_lightbox_offsets(
            center_arc_mm=50.0, total_len_mm=100.0, step_mm=2.5, count=5
        )
        self.assertEqual(offsets_5, [45.0, 47.5, 50.0, 52.5, 55.0])

        img = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4),
        )
        dists, vals = CrossSectionNavigator.compute_density_profile_along_line(
            vol, [1.0, 2.0, 2.0], [7.0, 2.0, 2.0], num_samples=7
        )
        self.assertEqual(len(dists), 7)
        self.assertAlmostEqual(float(dists[-1]), 6.0, places=5)
        self.assertGreater(float(vals[-1]), float(vals[0]))

    def test_f6_f7_f8_f9_f11_f13_f18_advanced_clinical_tools(self) -> None:
        """F6, F7, F8, F9, F11, F13, F18, B21: Verify occlusal SVD, airway mCSA, TMJ frames, angle/surface distance, DICOM, screenshot, and Undo/Redo."""
        cusps = np.array(
            [
                [25.0, -15.0, 5.0],
                [15.0, 10.0, 5.0],
                [0.0, 20.0, 5.0],
                [-15.0, 10.0, 5.0],
                [-25.0, -15.0, 5.0],
            ],
            dtype=np.float64,
        )
        normal, center = compute_occlusal_plane_from_segmentation(cusps)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(center[2]), 5.0)

        mods = detect_automated_dental_tools_modules()
        self.assertIn("ASO", mods)
        self.assertIn("AMASSS", mods)

        air_img = np.full((30, 30, 30), 200.0, dtype=np.float32)
        air_img[5:25, 12:18, 12:18] = -850.0
        air_img[15, :, :] = 200.0
        air_img[15, 13:16, 13:17] = -850.0
        vol_air = CBCTVolume(
            image=air_img,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.eye(4),
        )
        air_res = compute_pharyngeal_airway_metrics(vol_air, seed_ras=[14.0, 14.0, 10.0], threshold_hu=-400.0)
        self.assertIsNotNone(air_res)
        assert air_res is not None
        self.assertAlmostEqual(air_res.min_cross_sectional_area_mm2, 12.0, places=2)
        self.assertAlmostEqual(air_res.min_csa_z_ras_mm, 15.0, places=2)
        self.assertAlmostEqual(air_res.total_volume_mm3, 19 * 36.0 + 12.0, places=2)

        tmj_frames = compute_tmj_condyle_frames(condylar_angle_deg=15.0)
        self.assertIn("Right", tmj_frames)
        self.assertIn("Left", tmj_frames)
        self.assertAlmostEqual(
            float(np.dot(tmj_frames["Right"].sagittal_normal_ras, tmj_frames["Right"].coronal_normal_ras)),
            0.0,
            places=6,
        )

        ang = compute_angle_between_three_points_deg([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(ang, 90.0, places=4)
        s_dist, _, _ = compute_surface_to_surface_distance_mm(
            np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 10.0]]), radius_a_mm=2.0, radius_b_mm=1.5
        )
        self.assertAlmostEqual(s_dist, 6.5, places=3)

        state = {"x": 10.0}
        urm = UndoRedoManager()
        urm.push("move x", undo_fn=lambda: state.update(x=10.0), redo_fn=lambda: state.update(x=25.0))
        state["x"] = 25.0
        self.assertTrue(urm.can_undo)
        urm.undo()
        self.assertEqual(state["x"], 10.0)
        self.assertTrue(urm.can_redo)
        urm.redo()
        self.assertEqual(state["x"], 25.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            written = export_reformatted_dicom_series(
                np.zeros((32, 64), dtype=np.float32), (0.5, 0.5), Path(tmpdir)
            )
            self.assertGreaterEqual(len(written), 1)
            self.assertTrue(written[0].exists())

        class _FakeResult:
            def GetNumberOfTuples(self) -> int:
                return 32

            def GetNumberOfValues(self) -> int:
                return 32

            def GetValue(self, i: int) -> int:
                return (i + 65) % 256

        class _FakeVTK:
            class vtkWindowToImageFilter:
                def SetInput(self, _rw: object) -> None:
                    pass

                def SetInputBufferTypeToRGB(self) -> None:
                    pass

                def ReadFrontBufferOff(self) -> None:
                    pass

                def Update(self) -> None:
                    pass

                def GetOutputPort(self) -> object:
                    return object()

            class vtkPNGWriter:
                def WriteToMemoryOn(self) -> None:
                    pass

                def SetInputConnection(self, _port: object) -> None:
                    pass

                def Write(self) -> None:
                    pass

                def GetResult(self) -> _FakeResult:
                    return _FakeResult()

        class _FakeSV:
            def renderWindow(self) -> object:
                return object()

            def forceRender(self) -> None:
                pass

        class _FakeSW:
            def sliceView(self) -> _FakeSV:
                return _FakeSV()

        with patch("CBCTPlannerLib.AdvancedTools.vtk", _FakeVTK):
            uri = capture_slice_view_png_base64("Yellow", slice_widget=_FakeSW())
            self.assertIsNotNone(uri)
            assert uri is not None
            self.assertTrue(uri.startswith("data:image/png;base64,"))


class PanoramicSamplerAndFrameTests(unittest.TestCase):
    def _phantom(self) -> tuple[CBCTVolume, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shape = (40, 48, 48)
        z = np.arange(shape[0], dtype=np.float32)[:, None, None]
        img = np.broadcast_to(100.0 + 3.0 * z, shape).copy()
        vol = CBCTVolume(
            image=img,
            spacing=(1.0, 1.0, 1.0),
            origin=(-20.0, -20.0, -10.0),
            direction=np.eye(3),
            affine=np.array(
                [[1.0, 0.0, 0.0, -20.0], [0.0, 1.0, 0.0, -20.0], [0.0, 0.0, 1.0, -10.0], [0.0, 0.0, 0.0, 1.0]]
            ),
        )
        theta = np.linspace(0.0, np.pi, 40)
        pts = np.column_stack([12.0 * np.cos(theta), 12.0 * np.sin(theta), np.zeros_like(theta)])
        tangents = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        binormals = np.tile(np.array([0.0, 0.0, 1.0]), (len(theta), 1))
        return vol, pts, tangents, normals, binormals

    def test_fast_sampler_matches_legacy_world_loop(self) -> None:
        vol, pts, tangents, normals, binormals = self._phantom()
        logic = PanoramicCurveLogic()
        base, disp, offsets, _n, _dy, _dz = logic._build_slab_geometry(
            pts_arch=pts,
            normals=normals,
            binormals=binormals,
            slab_thickness_mm=6.0,
            height_mm=20.0,
            spacing_mm=1.0,
            buccolingual_offset_mm=0.0,
            flip_vertical=False,
        )
        accum = None
        for u_val in offsets:
            layer_pts = base + float(u_val) * disp[None, :, :]
            sampled = vol.sample_world_points(layer_pts, order=1, cval=-1000.0)
            accum = sampled.copy() if accum is None else accum + sampled
        legacy = (accum / float(len(offsets))).astype(np.float32)
        fast = logic.sample_panoramic_slab_array(
            vol,
            pts,
            tangents,
            normals,
            binormals,
            slab_thickness_mm=6.0,
            height_mm=20.0,
            spacing_mm=1.0,
            projection_mode="Mean",
        )
        np.testing.assert_allclose(fast, legacy, atol=1e-3)

    def test_tangent_smoothing_reduces_jitter_and_keeps_endpoints(self) -> None:
        t = np.linspace(0.0, 1.0, 60)
        base = np.column_stack([np.cos(t * 2.0), np.sin(t * 2.0), t])
        rng = np.random.default_rng(7)
        noisy = base + rng.normal(scale=0.01, size=base.shape)
        smoothed = PanoramicCurveLogic._smooth_series(noisy)
        self.assertEqual(smoothed.shape, noisy.shape)
        self.assertLess(
            np.linalg.norm(np.diff(smoothed, n=2, axis=0)), np.linalg.norm(np.diff(noisy, n=2, axis=0))
        )
        self.assertLess(np.max(np.abs(smoothed - noisy)), 0.02)

    def test_curve_frames_remain_orthonormal_after_smoothing(self) -> None:
        logic = PanoramicCurveLogic()
        points = np.array(
            [
                [10.0, 0.0, 0.0],
                [12.0, 2.0, 0.3],
                [11.0, 4.0, -0.2],
                [8.0, 5.0, 0.1],
                [5.0, 4.0, 0.0],
                [4.0, 2.0, 0.2],
            ]
        )
        _, _, tangents, normals, binormals = logic.get_curve_points_and_frames(
            type(
                "_Curve",
                (),
                {
                    "GetID": lambda self: "smoothed",
                    "GetMTime": lambda self: 1,
                    "GetNumberOfControlPoints": lambda self: len(points),
                    "GetNthControlPointPositionWorld": lambda self, i, b: b.__setitem__(
                        slice(None), points[i]
                    ),
                    "GetCurveWorld": lambda self: None,
                },
            )(),
            num_samples=30,
        )
        for tangent, normal, binormal in zip(tangents, normals, binormals):
            frame = np.column_stack([tangent, normal, binormal])
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(np.cross(normal, tangent), binormal, atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(frame)), -1.0, places=6)


if __name__ == "__main__":
    unittest.main()


class TestPhase7AndCheckpointInference(unittest.TestCase):
    """Tests for Phase 7 (F14-F20) and Custom AI Checkpoint Inference Engine."""

    def test_checkpoint_state_dict_normalization_and_shape_safe_loading(self):
        from CBCTPlannerLib.CheckpointInference import (
            CheckpointInferenceEngine,
            CheckpointSpec,
            infer_architecture_dims_from_state_dict,
            load_state_dict_shape_safe,
            normalize_cbct_volume,
            normalize_state_dict_keys,
            sliding_window_inference_3d,
        )

        # 1. Verify wrapped & prefixed state_dict normalization (nnUNet / DDP / torch.compile)
        raw_ckpt = {
            "network_weights": {
                "module._orig_mod.enc1.0.weight": np.zeros((16, 1, 3, 3, 3), dtype=np.float32),
                "module._orig_mod.enc1.0.bias": np.zeros((16,), dtype=np.float32),
                "module._orig_mod.head.weight": np.zeros((6, 16, 1, 1, 1), dtype=np.float32),
                "module._orig_mod.head.bias": np.zeros((6,), dtype=np.float32),
            }
        }
        cleaned = normalize_state_dict_keys(raw_ckpt)
        self.assertIn("enc1.0.weight", cleaned)
        self.assertIn("head.weight", cleaned)
        in_ch, num_classes, base_f = infer_architecture_dims_from_state_dict(cleaned)
        self.assertEqual((in_ch, num_classes, base_f), (1, 6, 16))

        # 2. Verify shape-safe state_dict loader skips mismatched tensors instead of raising RuntimeError
        class _DummyParam:
            def __init__(self, shape):
                self.shape = shape

        class _DummyModule:
            def __init__(self):
                self._sd = {
                    "enc1.0.weight": _DummyParam((16, 1, 3, 3, 3)),
                    "head.weight": _DummyParam((6, 16, 1, 1, 1)),
                }
                self.loaded = None

            def state_dict(self):
                return self._sd

            def load_state_dict(self, sd, strict=False):
                self.loaded = sd

        dummy_mod = _DummyModule()
        candidate_sd = {
            "enc1.0.weight": _DummyParam((16, 1, 3, 3, 3)),  # matches
            "head.weight": _DummyParam((32, 16, 1, 1, 1)),  # mismatched class count (32 vs 6)
            "extra.layer": _DummyParam((8, 8)),  # unexpected key
        }
        loaded_count, skipped_mismatch = load_state_dict_shape_safe(dummy_mod, candidate_sd)
        self.assertEqual(loaded_count, 1)
        self.assertEqual(len(skipped_mismatch), 2)
        self.assertIn("head.weight", skipped_mismatch[0])

        # 3. Verify 3D sliding-window Gaussian blending & checkpoint .npy inference
        vol = np.linspace(-500, 1500, 24 * 24 * 24, dtype=np.float32).reshape((24, 24, 24))
        norm_ct = normalize_cbct_volume(vol, mode="ct_hu")
        self.assertAlmostEqual(float(np.mean(norm_ct)), 0.0, places=3)
        norm_vol = normalize_cbct_volume(vol, mode="minmax")
        self.assertTrue(np.all(norm_vol >= 0.0) and np.all(norm_vol <= 1.0))

        def _dummy_predictor(patch_zyx):
            pd, ph, pw = patch_zyx.shape[-3:]
            p3 = patch_zyx.reshape((pd, ph, pw))
            out = np.zeros((3, pd, ph, pw), dtype=np.float32)
            out[0] = 0.2
            out[1] = np.where(p3 > 0.5, 0.9, 0.1)
            out[2] = np.where(p3 > 0.8, 1.2, 0.0)
            return out

        seg = sliding_window_inference_3d(
            norm_vol,
            predict_patch_fn=_dummy_predictor,
            patch_size_zyx=(12, 12, 12),
            num_classes=3,
            overlap=0.5,
        )
        self.assertEqual(seg.shape, (24, 24, 24))
        self.assertIn(2, np.unique(seg))

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_npy = Path(tmpdir) / "my_checkpoint_labels.npy"
            expected_labels = np.zeros((16, 16, 16), dtype=np.uint8)
            expected_labels[4:8, 4:8, 4:8] = 2  # Mandible
            expected_labels[8:12, 4:8, 4:8] = 4  # Lower Teeth
            expected_labels[6:7, 6:7, 2:14] = 5  # Mandibular Canal
            np.save(str(ckpt_npy), expected_labels)

            engine = CheckpointInferenceEngine()
            spec = CheckpointSpec(
                checkpoint_path=Path(ckpt_npy),
                label_scheme="5class",
            )
            labels_out, summary = engine.run_checkpoint_on_array(
                np.zeros((16, 16, 16), dtype=np.float32), spec
            )
            self.assertEqual(summary.format_type, "numpy")
            self.assertEqual(labels_out.shape, (16, 16, 16))
            self.assertIn(2, np.unique(labels_out))
            self.assertIn(4, np.unique(labels_out))
            self.assertIn(5, np.unique(labels_out))

    def test_phase7_f14_to_f20_tools_and_headless_cli(self):
        from CBCTPlannerLib.CheckpointInference import CheckpointSpec
        from CBCTPlannerLib.ImplantPlanner import PlacedImplant
        from CBCTPlannerLib.Phase7Tools import (
            ProstheticPlanner,
            SurgicalGuideGenerator,
            compute_3d_cephalometry,
            export_dicom_seg_and_sr,
            run_automated_dental_tool,
            run_headless_batch_pipeline,
        )
        from CBCTPlannerLib.core.volume_loader import CBCTVolume

        # F14: Surgical guide with hollow sleeve bore & printer presets
        guide_gen = SurgicalGuideGenerator()
        self.assertTrue(len(guide_gen.presets) >= 3)
        arch_pts = np.array(
            [[-25.0 + 50.0 * t, 20.0 * np.sin(np.pi * t), 0.0] for t in np.linspace(0, 1, 30)],
            dtype=np.float64,
        )
        imp = PlacedImplant(
            fdi=36,
            entry_id="generic_4.1x10",
            label="Generic 4.1 x 10.0 mm",
            diameter_mm=4.1,
            length_mm=10.0,
            platform_diameter_mm=4.1,
            apical_diameter_mm=3.5,
            crest_point_ras=[0.0, 20.0, 0.0],
            apex_point_ras=[0.0, 20.0, -10.0],
            axis_apical_ras=[0.0, 0.0, -1.0],
            clinician_confirmed=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            stl_p, json_p, sleeves = guide_gen.export_surgical_guide_stl_and_json(
                arch_pts,
                [imp],
                tmp_path,
                support_type="tooth_supported",
                preset_id="sla_dlp_standard_50um",
            )
            self.assertTrue(stl_p.is_file())
            self.assertTrue(json_p.is_file())
            self.assertEqual(len(sleeves), 1)
            self.assertEqual(sleeves[0].fdi, 36)
            self.assertGreater(sleeves[0].inner_bore_diameter_mm, imp.diameter_mm)

            # F15: DICOM SEG & Structured Report export
            seg_arr = np.zeros((12, 16, 16), dtype=np.uint8)
            seg_arr[2:10, 4:12, 4:12] = 2
            seg_dcm, sr_dcm = export_dicom_seg_and_sr(
                seg_arr,
                (0.5, 0.5, 0.5),
                [{"fdi": 36, "height_mm": 13.5, "width_mm": 7.2, "nerve_dist_mm": 3.8, "hu_mean": 620.0}],
                [imp],
                tmp_path,
                patient_id="TEST_PT_01",
            )
            self.assertTrue(seg_dcm.is_file())
            self.assertTrue(sr_dcm.is_file())

            # F16: Generic prosthetic abutment & crown library proposal
            prosthetic = ProstheticPlanner()
            self.assertTrue(len(prosthetic.abutments) >= 3)
            proposal, v_p, f_p = prosthetic.generate_proposal_for_implant(
                imp, occlusal_normal=(0.0, 0.0, 1.0)
            )
            self.assertEqual(proposal.fdi, 36)
            self.assertGreater(proposal.crown_to_implant_ratio, 0.5)
            self.assertGreater(len(v_p), 20)
            self.assertGreater(len(f_p), 20)

            # F17: 3D Cephalometry angles & classifications
            landmarks = {
                "S": [0.0, 0.0, 60.0],
                "N": [0.0, 70.0, 60.0],
                "A": [0.0, 66.0, 20.0],
                "B": [0.0, 63.0, -15.0],
                "Or": [0.0, 58.0, 42.0],
                "Po": [0.0, -5.0, 45.0],
                "Me": [0.0, 60.0, -45.0],
                "Go": [0.0, 5.0, -25.0],
                "Gn": [0.0, 62.0, -42.0],
                "U1_tip": [0.0, 70.0, 5.0],
                "U1_apex": [0.0, 62.0, 22.0],
                "L1_tip": [0.0, 68.0, 4.0],
                "L1_apex": [0.0, 58.0, -14.0],
            }
            ceph_list = compute_3d_cephalometry(landmarks)
            ceph_by_name = {m.name: m for m in ceph_list}
            self.assertIn("SNA", ceph_by_name)
            self.assertIn("SNB", ceph_by_name)
            self.assertIn("ANB", ceph_by_name)
            self.assertIn("Class", ceph_by_name["ANB"].interpretation)

            # F18: AutomatedDentalTools feature-detected execution wrapper
            adt_res = run_automated_dental_tool("ASO", {"input_volume": "test"})
            self.assertEqual(adt_res["status"], "not_installed")

            # F20: Headless Batch CLI with Custom Checkpoint Inference
            vol_syn = np.full((24, 32, 32), -200.0, dtype=np.float32)
            vol_syn[4:20, 8:24, 6:26] = 750.0
            ckpt_syn = np.zeros((24, 32, 32), dtype=np.uint8)
            ckpt_syn[4:16, 8:24, 6:26] = 2  # Mandible
            ckpt_syn[14:19, 10:22, 8:24] = 4  # Lower teeth
            ckpt_syn[6:8, 12:20, 6:26] = 5  # Canal

            ckpt_path = tmp_path / "custom_ckpt.npy"
            out_batch = tmp_path / "batch_out"
            np.save(str(ckpt_path), ckpt_syn)

            cbct_vol = CBCTVolume(
                image=vol_syn,
                spacing=(0.5, 0.5, 0.5),
                origin=(-8.0, -8.0, -6.0),
                direction=np.eye(3, dtype=np.float64),
                affine=np.diag([-0.5, -0.5, 0.5, 1.0]),
            )
            summary = run_headless_batch_pipeline(
                volume=cbct_vol,
                output_dir=out_batch,
                checkpoint_spec=CheckpointSpec(checkpoint_path=ckpt_path, label_scheme="5class"),
                is_5class=True,
                teeth_fdi_to_measure=(36, 46),
                place_generic_implants=True,
            )
            self.assertEqual(summary["checkpoint_summary"]["format_type"], "numpy")
            self.assertTrue(Path(summary["plan_json"]).is_file())
            self.assertTrue(Path(summary["plan_csv"]).is_file())
            self.assertTrue(Path(summary["report_html"]).is_file())
            self.assertTrue(Path(summary["surgical_guide_stl"]).is_file())
            self.assertTrue(Path(summary["dicom_sr"]).is_file())


if __name__ == "__main__":
    unittest.main()

"""Dental Arch Curve Visualization and In-Process Panoramic Curved Planar Reformation (CPR).

Fixes & Upgrades:
  - B5: Right-handed (det = +1) displacement frame `(-N, B, T)` matching grid frame `[axis_x, axis_y, axis_z]`
    (`det = +1`) to eliminate buccolingual mirroring in straightened CPR volumes.
  - B6: Fixed `<= 1.5 mm` frame sampling step along the arch independent of output voxel spacing.
  - B9: Bounded LRU frame cache (`max_entries=8`) and automatic removal of finished/cancelled CLI nodes.
  - B10: `display_panoramic_in_slice_view` calls `FitSliceToAll()` only on first display or when geometry changes.
  - B13: Sub-sample piecewise line-segment projection in `project_point_to_curve_arc_mm`.
  - B15 & O5: Stores and resolves arch curve and panoramic volume nodes via ParameterNode reference IDs.
  - O1 & F3: In-process `scipy.ndimage.map_coordinates` (`order=1`) panoramic slab generator supporting
    `Mean`, `MIP`, `MinIP`, and `RaySum` (pseudo-pan), configurable `height_mm` and `buccolingual_offset_mm`,
    and non-blocking chunked `QTimer` execution with progress and cancellation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np

try:
    import qt  # type: ignore
    import slicer  # type: ignore
except ImportError:
    qt = None
    slicer = None

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
except ImportError:
    vtk = None
    vtk_to_numpy = None

from .Utils import BoundedLRUCache
from .core.arch_detector import DentalArch, project_point_to_polyline_arc_mm
from .core.volume_loader import CBCTVolume


class PanoramicCurveLogic:
    """Handles dental arch curve visualization, bounded frame caching, and in-process panoramic CPR."""

    MAX_CPR_FRAME_SPACING_MM: float = 1.5

    def __init__(self, max_cache_entries: int = 8) -> None:
        self.max_frame_spacing_mm: float = self.MAX_CPR_FRAME_SPACING_MM
        # B9: Bounded LRU cache keyed by (node_id, mtime, num_samples, occlusal_tuple)
        self._frame_cache: BoundedLRUCache[
            Tuple[str, int, int, Tuple[float, float, float]],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = BoundedLRUCache(max_entries=max_cache_entries)
        self._active_cli_node: Any = None
        self._active_cli_observer: Optional[int] = None
        self._cancel_requested: bool = False
        self._chunk_timer: Any = None
        # B10: Track fitted panoramic geometry per slice view -> (pano_node_id, fov_w_mm, fov_h_mm)
        self._fitted_pano_geometry: Dict[str, Tuple[str, float, float]] = {}
        self.last_generation_elapsed_sec: float = 0.0

    @staticmethod
    def pano_x(arc_mm: float) -> float:
        """Convert physical arc-length along arch (0 = Patient Right, L = Patient Left) to Panoramic RAS X."""
        return -float(arc_mm)

    @staticmethod
    def arc_mm(pano_x: float, max_arc_mm: Optional[float] = None) -> float:
        """Convert Panoramic RAS X coordinate back to physical arc-length in mm along the arch."""
        val = max(0.0, -float(pano_x))
        if max_arc_mm is not None:
            val = min(float(max_arc_mm), val)
        return val

    @staticmethod
    def compute_cpr_displacement_frame(
        T: np.ndarray, N: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """B5: Return a strictly right-handed (`det = +1`) displacement frame `(disp_x, disp_y, disp_z)`.

        Because `N = T x B` (Buccal), `det([N, B, T]) = -1.0` (left-handed).
        Mapping `disp_x = -N` (Buccal-to-Lingual), `disp_y = B` (Inferior-to-Superior), and
        `disp_z = T` (Right-to-Left along arch) yields `det([-N, B, T]) = +1.0`, matching the
        grid direction matrix `[axis_x, axis_y, axis_z]` (`det = +1.0`) and eliminating mirroring.
        """
        t_u = np.asarray(T, dtype=np.float64) / max(1e-8, float(np.linalg.norm(T)))
        n_u = np.asarray(N, dtype=np.float64) / max(1e-8, float(np.linalg.norm(N)))
        b_u = np.asarray(B, dtype=np.float64) / max(1e-8, float(np.linalg.norm(B)))
        disp_x = -n_u
        disp_y = b_u
        disp_z = t_u
        if float(np.linalg.det(np.column_stack([disp_x, disp_y, disp_z]))) < 0.0:
            disp_x = -disp_x
        return disp_x, disp_y, disp_z

    @classmethod
    def compute_cpr_num_frames(
        cls, total_len_mm: float, max_frame_spacing_mm: Optional[float] = None
    ) -> int:
        """B6: Compute number of CPR frames using a fixed `<= 1.5 mm` frame spacing independent of voxel spacing."""
        step = min(
            2.0,
            max(0.25, float(max_frame_spacing_mm or cls.MAX_CPR_FRAME_SPACING_MM)),
        )
        return max(8, int(np.ceil(float(total_len_mm) / step)) + 1)

    @staticmethod
    def compute_cpr_direction_matrix(
        curve_start_ras: np.ndarray,
        curve_end_ras: np.ndarray,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """Compute a strictly right-handed (det = +1) 4x4 grid direction matrix for CPR straightening."""
        up = np.asarray(occlusal_normal, dtype=np.float64).reshape(3)
        up_norm = float(np.linalg.norm(up))
        axis_y = up / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if axis_y[2] < 0:
            axis_y = -axis_y

        chord = np.asarray(curve_end_ras, dtype=np.float64) - np.asarray(
            curve_start_ras, dtype=np.float64
        )
        chord_in_plane = chord - np.dot(chord, axis_y) * axis_y
        c_norm = float(np.linalg.norm(chord_in_plane))
        axis_z = (
            chord_in_plane / c_norm
            if c_norm > 1e-8
            else np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        )

        axis_x = np.cross(axis_y, axis_z)
        x_norm = float(np.linalg.norm(axis_x))
        axis_x = axis_x / x_norm if x_norm > 1e-8 else np.array([0.0, -1.0, 0.0], dtype=np.float64)

        axis_z = np.cross(axis_x, axis_y)
        axis_z /= max(1e-8, float(np.linalg.norm(axis_z)))

        mat = np.eye(4, dtype=np.float64)
        mat[0:3, 0] = axis_x
        mat[0:3, 1] = axis_y
        mat[0:3, 2] = axis_z
        return mat

    @staticmethod
    def _reference_role_for_jaw(jaw: str) -> str:
        jaw_low = jaw.lower()
        if jaw_low.startswith("up"):
            return "UpperArchCurve"
        if jaw_low.startswith("sh") or jaw_low.startswith("oc"):
            return "SharedArchCurve"
        return "LowerArchCurve"

    @classmethod
    def resolve_curve_node(
        cls,
        volumeNode: Any,
        jaw: str = "Lower",
        parameterNode: Any = None,
    ) -> Any:
        """B15 & O5: Resolve arch curve node by ParameterNode reference ID first, falling back to name migration."""
        role = cls._reference_role_for_jaw(jaw)
        if parameterNode is not None and hasattr(parameterNode, "GetNodeReference"):
            ref_node = parameterNode.GetNodeReference(role)
            if ref_node is not None:
                return ref_node
        if slicer is None or getattr(slicer, "mrmlScene", None) is None:
            return None
        vol_name = volumeNode.GetName() if (volumeNode and hasattr(volumeNode, "GetName")) else "CBCT"
        jaw_cap = "Shared" if role == "SharedArchCurve" else jaw.capitalize()
        name = f"{vol_name}_{jaw_cap}_DentalArch"
        found = slicer.mrmlScene.GetFirstNodeByName(name)
        if found is not None and parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
            parameterNode.SetNodeReferenceID(role, found.GetID())
        return found

    @classmethod
    def create_or_update_arch_curve(
        cls,
        volumeNode: Any,
        jaw: str,
        points_ras: Optional[np.ndarray] = None,
        color: Optional[Tuple[float, float, float]] = None,
        visible: bool = True,
        show_labels: bool = False,
        line_diameter: float = 1.8,
        parameterNode: Any = None,
    ) -> Any:
        """Create or update a vtkMRMLMarkupsCurveNode and bind its ID in ParameterNode (B15, O7)."""
        if slicer is None:
            return None
        role = cls._reference_role_for_jaw(jaw)
        jaw_cap = "Shared" if role == "SharedArchCurve" else jaw.capitalize()
        curveNode = cls.resolve_curve_node(volumeNode, jaw=jaw, parameterNode=parameterNode)
        if not curveNode:
            vol_name = volumeNode.GetName() if volumeNode else "CBCT"
            name = f"{vol_name}_{jaw_cap}_DentalArch"
            curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
            curveNode.CreateDefaultDisplayNodes()

        if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
            parameterNode.SetNodeReferenceID(role, curveNode.GetID())

        # B3: Inherit parent alignment transform from volumeNode if present
        if (
            volumeNode is not None
            and hasattr(volumeNode, "GetTransformNodeID")
            and hasattr(curveNode, "SetAndObserveTransformNodeID")
        ):
            tf_id = volumeNode.GetTransformNodeID()
            if tf_id:
                curveNode.SetAndObserveTransformNodeID(tf_id)

        was_mod = curveNode.StartModify() if hasattr(curveNode, "StartModify") else None
        try:
            curveNode.SetAttribute("CBCTPlanner.Jaw", jaw_cap.lower())
            if points_ras is not None:
                pts = np.asarray(points_ras, dtype=np.float64)
                if len(pts) >= 2 and pts[0, 0] < pts[-1, 0]:
                    pts = pts[::-1]
                curveNode.RemoveAllControlPoints()
                for pt in pts:
                    curveNode.AddControlPoint(float(pt[0]), float(pt[1]), float(pt[2]))

            if hasattr(curveNode, "SetCurveTypeToCardinalSpline"):
                curveNode.SetCurveTypeToCardinalSpline()
        finally:
            if was_mod is not None and hasattr(curveNode, "EndModify"):
                curveNode.EndModify(was_mod)

        if color is None:
            if jaw_cap == "Lower":
                color = (0.02, 0.71, 0.83)
            elif jaw_cap == "Upper":
                color = (0.92, 0.70, 0.03)
            else:
                color = (0.45, 0.85, 0.35)

        dNode = curveNode.GetDisplayNode()
        if dNode:
            d_mod = dNode.StartModify() if hasattr(dNode, "StartModify") else None
            try:
                dNode.SetColor(*color)
                if hasattr(slicer.vtkMRMLMarkupsDisplayNode, "UseLineDiameter"):
                    dNode.SetCurveLineSizeMode(slicer.vtkMRMLMarkupsDisplayNode.UseLineDiameter)
                dNode.SetLineDiameter(line_diameter)
                dNode.SetVisibility(visible)
                dNode.SetVisibility2D(visible)
                dNode.SetVisibility3D(visible)
                if hasattr(dNode, "SetPointLabelsVisibility"):
                    dNode.SetPointLabelsVisibility(show_labels)
                if hasattr(dNode, "SetPropertiesLabelVisibility"):
                    dNode.SetPropertiesLabelVisibility(show_labels)
            finally:
                if d_mod is not None and hasattr(dNode, "EndModify"):
                    dNode.EndModify(d_mod)
        return curveNode

    @staticmethod
    def extract_world_points_from_curve_node(curveNode: Any) -> np.ndarray:
        """Extract curve world points using vtk_to_numpy (zero Python loops when polydata exists)."""
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return np.empty((0, 3), dtype=np.float64)

        curve_poly = curveNode.GetCurveWorld() if hasattr(curveNode, "GetCurveWorld") else None
        if (
            curve_poly is not None
            and curve_poly.GetPoints() is not None
            and curve_poly.GetNumberOfPoints() >= 2
        ):
            if vtk_to_numpy is not None:
                pts = np.asarray(vtk_to_numpy(curve_poly.GetPoints().GetData()), dtype=np.float64)
            else:
                n_poly = curve_poly.GetNumberOfPoints()
                pts = np.array([curve_poly.GetPoint(i) for i in range(n_poly)], dtype=np.float64)
        else:
            n_cp = curveNode.GetNumberOfControlPoints()
            pts = np.zeros((n_cp, 3), dtype=np.float64)
            buf = [0.0, 0.0, 0.0]
            for i in range(n_cp):
                curveNode.GetNthControlPointPositionWorld(i, buf)
                pts[i] = buf
        return pts

    def 50.0(
        self,
        curveNode: Any,
        num_samples: int = 200,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample points, arc-lengths, tangents, buccal normals, and occlusal binormals along curveNode.

        B9: Cached in a bounded LRU cache (`max_entries=8`) keyed by `(node_id, mtime, num_samples, occlusal_tuple)`.
        """
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            raise ValueError("Curve node must have at least 2 control points.")

        node_id = curveNode.GetID() if hasattr(curveNode, "GetID") else str(id(curveNode))
        mtime = int(curveNode.GetMTime()) if hasattr(curveNode, "GetMTime") else 0
        occ_tuple = (
            round(float(occlusal_normal[0]), 6),
            round(float(occlusal_normal[1]), 6),
            round(float(occlusal_normal[2]), 6),
        )
        cache_key = (node_id, mtime, int(num_samples), occ_tuple)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        raw_pts = self.extract_world_points_from_curve_node(curveNode)
        if len(raw_pts) < 2:
            raise ValueError("Curve node has fewer than 2 world points.")

        if raw_pts[0, 0] < raw_pts[-1, 0]:
            raw_pts = raw_pts[::-1].copy()

        diffs = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = float(cum_len[-1])
        if total_len < 1e-6:
            raise ValueError("Dental arch curve has zero length.")

        target_s = np.linspace(0.0, total_len, num_samples)
        # If raw_pts is only sparse control points (e.g. headless mock), evaluate smooth spline if >= 4 pts
        if len(raw_pts) <= 30 and len(raw_pts) >= 4:
            try:
                from scipy import interpolate  # type: ignore

                u_norm = cum_len / total_len
                k = min(3, len(raw_pts) - 1)
                tck, _ = interpolate.splprep(
                    [raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2]], u=u_norm, s=0.0, k=k
                )
                u_dense = np.linspace(0.0, 1.0, max(num_samples, 300))
                dense_pts = np.column_stack(interpolate.splev(u_dense, tck))
                dense_s = np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(np.diff(dense_pts, axis=0), axis=1))]
                )
                total_len = float(dense_s[-1])
                target_s = np.linspace(0.0, total_len, num_samples)
                target_u = np.interp(target_s, dense_s, u_dense)
                pts = np.column_stack(interpolate.splev(target_u, tck))
            except Exception:
                pts = np.column_stack([np.interp(target_s, cum_len, raw_pts[:, d]) for d in range(3)])
        else:
            pts = np.column_stack([np.interp(target_s, cum_len, raw_pts[:, d]) for d in range(3)])

        up = np.asarray(occ_tuple, dtype=np.float64)
        up_norm = float(np.linalg.norm(up))
        up = up / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up[2] < 0:
            up = -up

        raw_tangents = np.gradient(pts, axis=0)
        tangents = raw_tangents - np.outer(raw_tangents @ up, up)
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        t_norms[t_norms < 1e-8] = 1.0
        tangents = tangents / t_norms

        normals = np.cross(tangents, up)
        n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        n_norms[n_norms < 1e-8] = 1.0
        normals = normals / n_norms

        binormals = np.cross(normals, tangents)
        b_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
        b_norms[b_norms < 1e-8] = 1.0
        binormals = binormals / b_norms

        flip = (binormals @ up) < 0
        binormals[flip] *= -1.0

        result = (pts, target_s, tangents, normals, binormals)
        self._frame_cache[cache_key] = result
        return result

    def get_frame_at_arc_mm(
        self,
        curveNode: Any,
        arc_mm: float,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Interpolate exact (P, T, N, B, clamped_arc_mm, total_len_mm) at `arc_mm` along curveNode."""
        pts, arc_s, tangents, normals, binormals = self.get_curve_points_and_frames(
            curveNode, num_samples=250, occlusal_normal=occlusal_normal
        )
        total_len = float(arc_s[-1])
        s_clamped = float(np.clip(arc_mm, 0.0, total_len))

        P = np.array([np.interp(s_clamped, arc_s, pts[:, d]) for d in range(3)], dtype=np.float64)
        T = np.array([np.interp(s_clamped, arc_s, tangents[:, d]) for d in range(3)], dtype=np.float64)
        N = np.array([np.interp(s_clamped, arc_s, normals[:, d]) for d in range(3)], dtype=np.float64)
        B = np.array([np.interp(s_clamped, arc_s, binormals[:, d]) for d in range(3)], dtype=np.float64)

        T /= max(1e-8, float(np.linalg.norm(T)))
        N /= max(1e-8, float(np.linalg.norm(N)))
        B /= max(1e-8, float(np.linalg.norm(B)))
        return P, T, N, B, s_clamped, total_len

    def project_point_to_curve_arc_mm(
        self,
        curveNode: Any,
        world_pt: Sequence[float],
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> float:
        """B13: Project a 3D RAS point onto curveNode using sub-sample line-segment projection."""
        pt = np.asarray(world_pt, dtype=np.float64).reshape(3)
        pts, arc_s, _, _, _ = self.get_curve_points_and_frames(
            curveNode, num_samples=250, occlusal_normal=occlusal_normal
        )
        if hasattr(curveNode, "GetClosestPointPositionAlongCurveWorld"):
            closest_pt = [0.0, 0.0, 0.0]
            try:
                curveNode.GetClosestPointPositionAlongCurveWorld(
                    [float(pt[0]), float(pt[1]), float(pt[2])], closest_pt
                )
                pt = np.asarray(closest_pt, dtype=np.float64)
            except Exception:
                pass

        best_arc, _, _ = project_point_to_polyline_arc_mm(pts, arc_s, pt)
        return float(best_arc)

    def compute_straightening_transform(
        self,
        transformNode: Any,
        curveNode: Any,
        slice_size_mm: Sequence[float] = (25.0, 120.0),
        spacing_along_curve_mm: float = 0.5,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> float:
        """B5 & B6: Compute a right-handed (`det = +1`) vtkOrientedGridTransform with fixed `<= 1.5 mm` frame spacing."""
        if vtk is None or slicer is None:
            raise RuntimeError("VTK and Slicer are required for curved planar reformation.")

        _ = spacing_along_curve_mm
        _, arc_s_check, _, _, _ = self.get_curve_points_and_frames(
            curveNode, num_samples=60, occlusal_normal=occlusal_normal
        )
        total_len = float(arc_s_check[-1])
        if total_len < 5.0:
            raise ValueError("Dental arch curve is too short for panoramic reformation.")

        # B6: Fixed <= 1.5 mm frame spacing independent of output voxel spacing
        num_slices = self.compute_cpr_num_frames(total_len, self.max_frame_spacing_mm)

        pts, _, tangents, normals, binormals = self.get_curve_points_and_frames(
            curveNode, num_samples=num_slices, occlusal_normal=occlusal_normal
        )

        dir_arr = self.compute_cpr_direction_matrix(
            pts[0], pts[-1], occlusal_normal=occlusal_normal
        )
        grid_axis_x = dir_arr[0:3, 0]
        grid_axis_y = dir_arr[0:3, 1]
        grid_axis_z = dir_arr[0:3, 2]

        center_pt = np.mean(pts, axis=0)
        grid_origin = (
            center_pt
            - grid_axis_x * (slice_size_mm[0] / 2.0)
            - grid_axis_y * (slice_size_mm[1] / 2.0)
            - grid_axis_z * (total_len / 2.0)
        )

        grid_dims = [2, 2, num_slices]
        grid_spacing = [
            float(slice_size_mm[0]),
            float(slice_size_mm[1]),
            total_len / float(num_slices - 1),
        ]

        dir_mat = slicer.util.vtkMatrixFromArray(dir_arr)
        grid_image = vtk.vtkImageData()
        grid_image.SetOrigin(grid_origin)
        grid_image.SetDimensions(grid_dims)
        grid_image.SetSpacing(grid_spacing)
        grid_image.AllocateScalars(vtk.VTK_DOUBLE, 3)

        transform = slicer.vtkOrientedGridTransform()
        transform.SetDisplacementGridData(grid_image)
        transform.SetGridDirectionMatrix(dir_mat)
        transformNode.SetAndObserveTransformFromParent(transform)

        displacements = slicer.util.arrayFromGridTransform(transformNode)
        for k in range(num_slices):
            p_k = pts[k]
            # B5: Right-handed displacement frame (-N, B, T) matching det(dir_arr) == +1
            disp_x, disp_y, _ = self.compute_cpr_displacement_frame(
                tangents[k], normals[k], binormals[k]
            )
            for j in range(2):
                for i in range(2):
                    straight_ras = (
                        grid_origin
                        + i * grid_spacing[0] * grid_axis_x
                        + j * grid_spacing[1] * grid_axis_y
                        + k * grid_spacing[2] * grid_axis_z
                    )
                    input_ras = (
                        p_k
                        + (i - 0.5) * slice_size_mm[0] * disp_x
                        + (j - 0.5) * slice_size_mm[1] * disp_y
                    )
                    displacements[k, j, i] = input_ras - straight_ras

        slicer.util.arrayFromGridTransformModified(transformNode)
        return total_len

    def cancel_active_panoramic_cli(self) -> None:
        """B9 & O1: Cancel an in-flight panoramic job and remove any cancelled CLI node from mrmlScene."""
        self._cancel_requested = True
        if self._chunk_timer is not None:
            try:
                self._chunk_timer.stop()
            except Exception:
                pass
            self._chunk_timer = None

        if self._active_cli_node is not None:
            cli_node = self._active_cli_node
            self._active_cli_node = None
            try:
                if self._active_cli_observer is not None:
                    cli_node.RemoveObserver(self._active_cli_observer)
                self._active_cli_observer = None
                cli_node.Cancel()
            except Exception:
                pass
            if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
                try:
                    slicer.mrmlScene.RemoveNode(cli_node)
                except Exception:
                    pass

    @classmethod
    def sample_panoramic_slab_array(
        cls,
        cbct_volume: CBCTVolume,
        pts_arch: np.ndarray,
        tangents: np.ndarray,
        normals: np.ndarray,
        binormals: np.ndarray,
        slab_thickness_mm: float = 25.0,
        height_mm: float = 120.0,
        spacing_mm: float = 0.5,
        buccolingual_offset_mm: float = 0.0,
        projection_mode: str = "Mean",
      flip_vertical: bool = False,
    ) -> np.ndarray:
        """O1, B5, B6, F3: Pure in-process panoramic CPR slab sampler using `scipy.ndimage.map_coordinates`.

        Samples layers across `[-slab_thickness_mm/2 + offset, +slab_thickness_mm/2 + offset]` along
        the right-handed displacement axis `disp_x = -N` (`det = +1`) and projects across layers via
        `Mean`, `MIP`, `MinIP`, or `RaySum` (pseudo-pan).
        Returns a 2D array of shape `(dim_y, dim_z)` where:
          - row `j` (`0 .. dim_y - 1`) increases Superiorly along `+B`
          - col `k` (`0 .. dim_z - 1`) progresses from Patient Right (`arc_mm = 0`) to Patient Left (`arc_mm = L`).
        """
        diffs = np.linalg.norm(np.diff(pts_arch, axis=0), axis=1)
        cum_s = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = float(cum_s[-1])
        sp = max(0.15, float(spacing_mm))

        dim_z = max(4, int(round(total_len / sp)))
        dim_y = max(2, int(round(float(height_mm) / sp)))
        layer_step = max(sp, 0.75)
        n_layers = max(1, int(round(float(slab_thickness_mm) / layer_step)) + 1)

        s_grid = np.linspace(0.0, total_len, dim_z)
        p_cols = np.column_stack([np.interp(s_grid, cum_s, pts_arch[:, d]) for d in range(3)])
        n_cols = np.column_stack([np.interp(s_grid, cum_s, normals[:, d]) for d in range(3)])
        b_cols = np.column_stack([np.interp(s_grid, cum_s, binormals[:, d]) for d in range(3)])
        n_cols /= np.maximum(1e-8, np.linalg.norm(n_cols, axis=1, keepdims=True))
        b_cols /= np.maximum(1e-8, np.linalg.norm(b_cols, axis=1, keepdims=True))

        # B5: Right-handed displacement axis disp_x = -N
        disp_x_cols = -n_cols
        v_offsets = np.linspace(-0.5 * float(height_mm), 0.5 * float(height_mm), dim_y)
      if flip_vertical: v_offsets = v_offsets[::-1]
        u_offsets = (
            np.linspace(-0.5 * float(slab_thickness_mm), 0.5 * float(slab_thickness_mm), n_layers)
            + float(buccolingual_offset_mm)
        )

        # Base 2D curved surface at u=0: shape (dim_y, dim_z, 3)
        base_surface = p_cols[None, :, :] + v_offsets[:, None, None] * b_cols[None, :, :]

        mode = projection_mode.strip().upper()
        accum: Optional[np.ndarray] = None

        for u_val in u_offsets:
            layer_pts = base_surface + float(u_val) * disp_x_cols[None, :, :]
            sampled = cbct_volume.sample_world_points(layer_pts, order=1, cval=-1000.0)
            if accum is None:
                accum = sampled.copy()
            elif mode == "MIP":
                np.maximum(accum, sampled, out=accum)
            elif mode == "MINIP":
                np.minimum(accum, sampled, out=accum)
            else:
                accum += sampled

        assert accum is not None
        if mode == "RAYSUM":
            # Pseudo-cephalometric / pseudo-panoramic attenuation integral scaled to display range
            accum = accum / float(max(1, n_layers)) * 1.25
        elif mode not in ("MIP", "MINIP"):
            accum /= float(max(1, n_layers))

        return accum.astype(np.float32)

    def _populate_panoramic_node_from_array(
        self,
        volumeNode: Any,
        proj_yk: np.ndarray,
        spacing_mm: float,
        total_len_mm: float,
        height_mm: float,
        show_in_green_view: bool = True,
        parameterNode: Any = None,
        force_fit: bool = False,
    ) -> Any:
        """Write 2D panoramic array `(dim_y, dim_z)` into MRML panoramic volume node and display."""
        panoNode = None
        if parameterNode is not None and hasattr(parameterNode, "GetNodeReference"):
            panoNode = parameterNode.GetNodeReference("PanoramicVolume")
        if panoNode is None:
            vol_name = volumeNode.GetName() if volumeNode else "CBCT"
            pano_name = f"{vol_name}_Panoramic"
            panoNode = slicer.mrmlScene.GetFirstNodeByName(pano_name)
            if not panoNode:
                panoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", pano_name)
            if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
                parameterNode.SetNodeReferenceID("PanoramicVolume", panoNode.GetID())

        dim_y, dim_z = proj_yk.shape
        pano_img = vtk.vtkImageData()
        pano_img.SetDimensions(dim_z, dim_y, 1)
        scalar_type = (
            volumeNode.GetImageData().GetScalarType()
            if (volumeNode and volumeNode.GetImageData())
            else vtk.VTK_SHORT
        )
        pano_img.AllocateScalars(scalar_type, 1)
        panoNode.SetAndObserveImageData(pano_img)

        pano_arr = slicer.util.arrayFromVolume(panoNode)
        pano_arr[0, :, :] = proj_yk.astype(pano_arr.dtype)
        slicer.util.arrayFromVolumeModified(panoNode)

        pano_ijk = np.eye(4, dtype=np.float64)
        pano_ijk[0, 0] = -float(spacing_mm)
        pano_ijk[1, 1] = float(spacing_mm)
        pano_ijk[2, 2] = -1.0
        panoNode.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(pano_ijk))
        panoNode.SetAttribute("CBCTPlanner.IsPanoramic", "1")
        panoNode.SetAttribute("CBCTPlanner.SpacingMm", str(spacing_mm))
        panoNode.SetAttribute("CBCTPlanner.HeightMm", str(height_mm))
        panoNode.SetAttribute("CBCTPlanner.TotalArcMm", str(total_len_mm))
        panoNode.CreateDefaultDisplayNodes()
        if volumeNode and volumeNode.GetDisplayNode() and panoNode.GetDisplayNode():
            panoNode.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())

        if show_in_green_view:
            self.display_panoramic_in_slice_view(
                panoNode, slice_view_name="Green", force_fit=force_fit
            )
        return panoNode

    def generate_panoramic_volume(
        self,
        volumeNode: Any,
        curveNode: Any,
        slab_thickness_mm: float = 25.0,
        height_mm: float = 120.0,
        spacing_mm: float = 0.5,
        buccolingual_offset_mm: float = 0.0,
        projection_mode: str = "Mean",
        show_in_green_view: bool = True,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
        async_cli: bool = False,
        on_complete: Optional[Callable[[Any, Any], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
        parameterNode: Any = None,
        force_fit: bool = False,
    ) -> Tuple[Optional[Any], Any]:
        """O1, B5, B6, B10, F3: Generate Curved Planar Reformatted (CPR) Panoramic Radiograph in-process.

        Replaces the slow disk-serializing `ResampleScalarVectorDWIVolume` CLI with in-process
        `scipy.ndimage.map_coordinates` slab sampling. When `async_cli=True` and `qt` is available,
        processes slab layers across non-blocking `QTimer` chunks so the UI remains responsive.
        """
        if vtk is None or slicer is None:
            raise RuntimeError("VTK and Slicer are required for panoramic generation.")
        if not volumeNode or not curveNode:
            raise ValueError("Both a CBCT volume and a dental arch curve are required.")

        self.cancel_active_panoramic_cli()
        self._cancel_requested = False
        t0 = time.perf_counter()

        _, arc_s_check, _, _, _ = self.get_curve_points_and_frames(
            curveNode, num_samples=60, occlusal_normal=occlusal_normal
        )
        total_len = float(arc_s_check[-1])
        if total_len < 5.0:
            raise ValueError("Dental arch curve is too short for panoramic reformation.")

        num_frames = self.compute_cpr_num_frames(total_len, self.max_frame_spacing_mm)
        pts_arch, _, tangents, normals, binormals = self.get_curve_points_and_frames(
            curveNode, num_samples=num_frames, occlusal_normal=occlusal_normal
        )
        cbct_vol = CBCTVolume.from_slicer_volume_node(volumeNode, copy_image=False)

        if not async_cli or qt is None:
            proj_yk = self.sample_panoramic_slab_array(
                cbct_volume=cbct_vol,
                pts_arch=pts_arch,
                tangents=tangents,
                normals=normals,
                binormals=binormals,
                slab_thickness_mm=slab_thickness_mm,
                height_mm=height_mm,
                spacing_mm=spacing_mm,
                buccolingual_offset_mm=buccolingual_offset_mm,
                projection_mode=projection_mode,
            )
            self.last_generation_elapsed_sec = time.perf_counter() - t0
            panoNode = self._populate_panoramic_node_from_array(
                volumeNode=volumeNode,
                proj_yk=proj_yk,
                spacing_mm=spacing_mm,
                total_len_mm=total_len,
                height_mm=height_mm,
                show_in_green_view=show_in_green_view,
                parameterNode=parameterNode,
                force_fit=force_fit,
            )
            if on_progress is not None:
                on_progress(100)
            if on_complete is not None:
                on_complete(panoNode, None)
            return panoNode, None

        # O1: Non-blocking chunked QTimer slab sampling
        sp = max(0.15, float(spacing_mm))
        dim_z = max(4, int(round(total_len / sp)))
        dim_y = max(2, int(round(float(height_mm) / sp)))
        layer_step = max(sp, 0.75)
        n_layers = max(1, int(round(float(slab_thickness_mm) / layer_step)) + 1)

        diffs = np.linalg.norm(np.diff(pts_arch, axis=0), axis=1)
        cum_s = np.concatenate([[0.0], np.cumsum(diffs)])
        s_grid = np.linspace(0.0, total_len, dim_z)
        p_cols = np.column_stack([np.interp(s_grid, cum_s, pts_arch[:, d]) for d in range(3)])
        n_cols = np.column_stack([np.interp(s_grid, cum_s, normals[:, d]) for d in range(3)])
        b_cols = np.column_stack([np.interp(s_grid, cum_s, binormals[:, d]) for d in range(3)])
        n_cols /= np.maximum(1e-8, np.linalg.norm(n_cols, axis=1, keepdims=True))
        b_cols /= np.maximum(1e-8, np.linalg.norm(b_cols, axis=1, keepdims=True))
        disp_x_cols = -n_cols

        v_offsets = np.linspace(-0.5 * float(height_mm), 0.5 * float(height_mm), dim_y)
        u_offsets = (
            np.linspace(-0.5 * float(slab_thickness_mm), 0.5 * float(slab_thickness_mm), n_layers)
            + float(buccolingual_offset_mm)
        )
        base_surface = p_cols[None, :, :] + v_offsets[:, None, None] * b_cols[None, :, :]
        mode = projection_mode.strip().upper()
        state: Dict[str, Any] = {"idx": 0, "accum": None}

        timer = qt.QTimer()
        timer.setInterval(0)
        self._chunk_timer = timer

        def _process_next_chunk() -> None:
            if self._cancel_requested:
                timer.stop()
                self._chunk_timer = None
                return
            idx = int(state["idx"])
            chunk_end = min(n_layers, idx + 3)
            for l_i in range(idx, chunk_end):
                u_val = float(u_offsets[l_i])
                layer_pts = base_surface + u_val * disp_x_cols[None, :, :]
                sampled = cbct_vol.sample_world_points(layer_pts, order=1, cval=-1000.0)
                if state["accum"] is None:
                    state["accum"] = sampled.copy()
                elif mode == "MIP":
                    np.maximum(state["accum"], sampled, out=state["accum"])
                elif mode == "MINIP":
                    np.minimum(state["accum"], sampled, out=state["accum"])
                else:
                    state["accum"] += sampled

            state["idx"] = chunk_end
            if on_progress is not None:
                on_progress(int(round(100.0 * chunk_end / float(n_layers))))

            if chunk_end >= n_layers:
                timer.stop()
                self._chunk_timer = None
                accum = state["accum"]
                if mode == "RAYSUM":
                    accum = accum / float(max(1, n_layers)) * 1.25
                elif mode not in ("MIP", "MINIP"):
                    accum /= float(max(1, n_layers))
                self.last_generation_elapsed_sec = time.perf_counter() - t0
                panoNode = self._populate_panoramic_node_from_array(
                    volumeNode=volumeNode,
                    proj_yk=accum.astype(np.float32),
                    spacing_mm=spacing_mm,
                    total_len_mm=total_len,
                    height_mm=height_mm,
                    show_in_green_view=show_in_green_view,
                    parameterNode=parameterNode,
                    force_fit=force_fit,
                )
                if on_complete is not None:
                    on_complete(panoNode, None)

        timer.timeout.connect(_process_next_chunk)
        timer.start()
        return None, None

    def display_panoramic_in_slice_view(
        self,
        panoNode: Any,
        slice_view_name: str = "Green",
        force_fit: bool = False,
    ) -> bool:
        """B10: Display Panoramic Radiograph in `slice_view_name` and call `FitSliceToAll()` ONLY on first display or geometry change.

        Returns True if `FitSliceToAll()` was called, False if existing user zoom/pan was preserved.
        """
        if slicer is None:
            return False
        layoutManager = slicer.app.layoutManager() if getattr(slicer, "app", None) else None
        if not layoutManager or not panoNode:
            return False
        widget = layoutManager.sliceWidget(slice_view_name)
        if not widget:
            return False
        compositeNode = widget.mrmlSliceCompositeNode()
        pano_id = panoNode.GetID()
        prev_bg_id = compositeNode.GetBackgroundVolumeID() if compositeNode else None
        if compositeNode and prev_bg_id != pano_id:
            compositeNode.SetBackgroundVolumeID(pano_id)

        total_arc = float(panoNode.GetAttribute("CBCTPlanner.TotalArcMm") or 0.0)
        height_mm = float(panoNode.GetAttribute("CBCTPlanner.HeightMm") or 0.0)
        prev_geom = self._fitted_pano_geometry.get(slice_view_name)
        geometry_changed = (
            prev_geom is None
            or prev_geom[0] != pano_id
            or abs(prev_geom[1] - total_arc) > 2.0
            or abs(prev_geom[2] - height_mm) > 2.0
        )

        sliceLogic = widget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        did_fit = False
        if force_fit or prev_bg_id != pano_id or geometry_changed:
            was_mod = sliceNode.StartModify() if hasattr(sliceNode, "StartModify") else None
            try:
                sliceNode.SetOrientationToAxial()
                sliceLogic.FitSliceToAll()
            finally:
                if was_mod is not None and hasattr(sliceNode, "EndModify"):
                    sliceNode.EndModify(was_mod)
            self._fitted_pano_geometry[slice_view_name] = (pano_id, total_arc, height_mm)
            did_fit = True
        else:
            sliceNode.Modified()
        return did_fit

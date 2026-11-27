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

import logging
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import zlib
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
from .core.arch_detector import compute_parallel_transport_frames, project_point_to_polyline_arc_mm
from .core.volume_loader import CBCTVolume

logger = logging.getLogger(__name__)


class PanoramicCurveLogic:
    """Handles dental arch curve visualization, bounded frame caching, and in-process panoramic CPR."""

    MAX_CPR_FRAME_SPACING_MM: float = 1.5
    _SLAB_BASIS_CACHE: BoundedLRUCache[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = (
        BoundedLRUCache(max_entries=16)
    )
    _SLAB_PROJECTION_CACHE: BoundedLRUCache[Tuple[Any, ...], np.ndarray] = BoundedLRUCache(max_entries=8)

    @classmethod
    def clear_panoramic_caches(cls) -> None:
        """Clear shared slab geometry and projected panoramic LRU caches."""
        cls._SLAB_BASIS_CACHE.clear()
        cls._SLAB_PROJECTION_CACHE.clear()

    @staticmethod
    def _array_crc32_key(arr: np.ndarray) -> Tuple[Tuple[int, ...], str, int]:
        contig = np.ascontiguousarray(arr)
        return tuple(int(s) for s in contig.shape), str(contig.dtype), int(zlib.crc32(memoryview(contig)))

    @classmethod
    def _slab_geometry_key(
        cls,
        cbct_volume: CBCTVolume,
        pts_arch: np.ndarray,
        normals: np.ndarray,
        binormals: np.ndarray,
        slab_thickness_mm: float,
        height_mm: float,
        spacing_mm: float,
        buccolingual_offset_mm: float,
        flip_vertical: bool,
    ) -> Tuple[Any, ...]:
        aff = np.ascontiguousarray(cbct_volume.affine, dtype=np.float64)
        pts_c = np.ascontiguousarray(pts_arch, dtype=np.float64)
        norm_c = np.ascontiguousarray(normals, dtype=np.float64)
        bino_c = np.ascontiguousarray(binormals, dtype=np.float64)
        return (
            zlib.crc32(memoryview(aff)),
            cls._array_crc32_key(pts_c),
            cls._array_crc32_key(norm_c),
            cls._array_crc32_key(bino_c),
            round(float(slab_thickness_mm), 4),
            round(float(height_mm), 4),
            round(float(spacing_mm), 4),
            round(float(buccolingual_offset_mm), 4),
            bool(flip_vertical),
        )

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
    def compute_cpr_num_frames(cls, total_len_mm: float, max_frame_spacing_mm: Optional[float] = None) -> int:
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

        chord = np.asarray(curve_end_ras, dtype=np.float64) - np.asarray(curve_start_ras, dtype=np.float64)
        chord_in_plane = chord - np.dot(chord, axis_y) * axis_y
        c_norm = float(np.linalg.norm(chord_in_plane))
        axis_z = chord_in_plane / c_norm if c_norm > 1e-8 else np.array([-1.0, 0.0, 0.0], dtype=np.float64)

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

    @staticmethod
    def _smooth_series(points: np.ndarray, kernel: Sequence[float] = (1, 2, 3, 2, 1)) -> np.ndarray:
        """Light edge-preserving moving average to stop gradient tangents striping the CPR."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < len(kernel) + 4:
            return pts
        weights = np.asarray(kernel, dtype=np.float64)
        weights /= weights.sum()
        half = len(weights) // 2
        padded = np.pad(pts, ((half, half), (0, 0)), mode="edge")
        smoothed = np.zeros_like(pts)
        for offset, weight in enumerate(weights):
            smoothed += weight * padded[offset : offset + len(pts)]
        return smoothed

    def get_curve_points_and_frames(
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

        pts_for_frames = self._smooth_series(pts)
        tangents, normals, binormals = compute_parallel_transport_frames(
            pts_for_frames, occlusal_normal=up, ras_convention=True
        )

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
                logger.debug(
                    "Best-effort operation failed", exc_info=True
                )  # silent-ok: non-critical cleanup/probe

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

        dir_arr = self.compute_cpr_direction_matrix(pts[0], pts[-1], occlusal_normal=occlusal_normal)
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
            disp_x, disp_y, _ = self.compute_cpr_displacement_frame(tangents[k], normals[k], binormals[k])
            for j in range(2):
                for i in range(2):
                    straight_ras = (
                        grid_origin
                        + i * grid_spacing[0] * grid_axis_x
                        + j * grid_spacing[1] * grid_axis_y
                        + k * grid_spacing[2] * grid_axis_z
                    )
                    input_ras = (
                        p_k + (i - 0.5) * slice_size_mm[0] * disp_x + (j - 0.5) * slice_size_mm[1] * disp_y
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
                logger.debug(
                    "Best-effort operation failed", exc_info=True
                )  # silent-ok: non-critical cleanup/probe
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
                logger.debug(
                    "Best-effort operation failed", exc_info=True
                )  # silent-ok: non-critical cleanup/probe
            if slicer is not None and getattr(slicer, "mrmlScene", None) is not None:
                try:
                    slicer.mrmlScene.RemoveNode(cli_node)
                except Exception:
                    logger.debug(
                        "Best-effort operation failed", exc_info=True
                    )  # silent-ok: non-critical cleanup/probe

    @staticmethod
    def _build_slab_geometry(
        pts_arch: np.ndarray,
        normals: np.ndarray,
        binormals: np.ndarray,
        slab_thickness_mm: float,
        height_mm: float,
        spacing_mm: float,
        buccolingual_offset_mm: float,
        flip_vertical: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        """Shared CPR slab geometry for the synchronous and chunked samplers.

        `flip_vertical` is applied here exactly once so both paths produce identical arrays.
        Returns `(base_surface, disp_x_cols, u_offsets, n_layers, dim_y, dim_z)`.
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

        disp_x_cols = -n_cols
        v_offsets = np.linspace(-0.5 * float(height_mm), 0.5 * float(height_mm), dim_y)
        if flip_vertical:
            v_offsets = v_offsets[::-1]
        u_offsets = np.linspace(
            -0.5 * float(slab_thickness_mm), 0.5 * float(slab_thickness_mm), n_layers
        ) + float(buccolingual_offset_mm)
        base_surface = p_cols[None, :, :] + v_offsets[:, None, None] * b_cols[None, :, :]
        return base_surface, disp_x_cols, u_offsets, n_layers, dim_y, dim_z

    @staticmethod
    def _finalize_projection(accum: np.ndarray, mode: str, n_layers: int) -> np.ndarray:
        """Apply the projection normalisation shared by sync and async sampling."""
        if mode == "RAYSUM":
            # Pseudo-cephalometric / pseudo-panoramic attenuation integral scaled to display range
            accum = accum / float(max(1, n_layers)) * 1.25
        elif mode not in ("MIP", "MINIP"):
            accum = accum / float(max(1, n_layers))
        return accum.astype(np.float32)

    @staticmethod
    def _slab_voxel_basis(
        cbct_volume: CBCTVolume, base_surface: np.ndarray, disp_x_cols: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the world slab surface and displacement axis to voxel space once."""
        dim_y, dim_z = base_surface.shape[:2]
        base_vox = cbct_volume.world_to_voxel(base_surface.reshape(-1, 3)).reshape(dim_y, dim_z, 3)
        disp_vox = cbct_volume.world_direction_to_voxel(disp_x_cols)
        return base_vox, disp_vox

    @staticmethod
    def _sample_slab_layer(
        cbct_volume: CBCTVolume, base_vox: np.ndarray, disp_vox: np.ndarray, u_val: float
    ) -> np.ndarray:
        """Sample one slab layer from precomputed voxel coordinates (no per-layer affine)."""
        try:
            from scipy import ndimage  # type: ignore
        except ImportError:
            layer_pts = cbct_volume.voxel_to_world(base_vox + float(u_val) * disp_vox[None, :, :])
            return np.asarray(cbct_volume.sample_world_points(layer_pts, order=1, cval=-1000.0))
        u = float(u_val)
        coords = np.empty((3,) + base_vox.shape[:2], dtype=np.float64)
        np.multiply(disp_vox[None, :, 2], u, out=coords[0])
        coords[0] += base_vox[:, :, 2]
        np.multiply(disp_vox[None, :, 1], u, out=coords[1])
        coords[1] += base_vox[:, :, 1]
        np.multiply(disp_vox[None, :, 0], u, out=coords[2])
        coords[2] += base_vox[:, :, 0]
        nz, ny, nx = cbct_volume.image.shape
        in_fov = (
            (coords[0] >= -0.5)
            & (coords[0] <= nz - 0.5)
            & (coords[1] >= -0.5)
            & (coords[1] <= ny - 0.5)
            & (coords[2] >= -0.5)
            & (coords[2] <= nx - 0.5)
        )
        np.clip(coords[0], 0.0, max(0.0, nz - 1.0 - 1e-5), out=coords[0])
        np.clip(coords[1], 0.0, max(0.0, ny - 1.0 - 1e-5), out=coords[1])
        np.clip(coords[2], 0.0, max(0.0, nx - 1.0 - 1e-5), out=coords[2])
        sampled = ndimage.map_coordinates(
            cbct_volume.image,
            coords,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        sampled_arr = np.asarray(sampled, dtype=np.float32)
        if not np.all(in_fov):
            sampled_arr[~in_fov] = -1000.0
        return sampled_arr

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
        projection_mode: str = "RaySum",
        flip_vertical: bool = False,
    ) -> np.ndarray:
        """O1, B5, B6, F3: Pure in-process panoramic CPR slab sampler using `scipy.ndimage.map_coordinates`.

        Samples layers across `[-slab_thickness_mm/2 + offset, +slab_thickness_mm/2 + offset]` along
        the right-handed displacement axis `disp_x = -N` (`det = +1`) and projects across layers via
        `RaySum` (default pseudo-pan), `Mean`, `MIP`, or `MinIP`.
        Returns a 2D array of shape `(dim_y, dim_z)` where:
          - row `j` (`0 .. dim_y - 1`) increases Superiorly along `+B`
          - col `k` (`0 .. dim_z - 1`) progresses from Patient Right (`arc_mm = 0`) to Patient Left (`arc_mm = L`).
        """
        mode = projection_mode.strip().upper()
        geom_key = cls._slab_geometry_key(
            cbct_volume=cbct_volume,
            pts_arch=pts_arch,
            normals=normals,
            binormals=binormals,
            slab_thickness_mm=slab_thickness_mm,
            height_mm=height_mm,
            spacing_mm=spacing_mm,
            buccolingual_offset_mm=buccolingual_offset_mm,
            flip_vertical=flip_vertical,
        )
        vol_key = cls._array_crc32_key(cbct_volume.image)
        proj_key = (vol_key, geom_key, mode)
        cached_proj = cls._SLAB_PROJECTION_CACHE.get(proj_key)
        if cached_proj is not None:
            return cached_proj.copy()

        cached_basis = cls._SLAB_BASIS_CACHE.get(geom_key)
        if cached_basis is not None:
            base_vox, disp_vox, u_offsets, n_layers = cached_basis
        else:
            # B5: Right-handed displacement axis disp_x = -N; flip applied once in shared geometry
            base_surface, disp_x_cols, u_offsets, n_layers, _, _ = cls._build_slab_geometry(
                pts_arch=pts_arch,
                normals=normals,
                binormals=binormals,
                slab_thickness_mm=slab_thickness_mm,
                height_mm=height_mm,
                spacing_mm=spacing_mm,
                buccolingual_offset_mm=buccolingual_offset_mm,
                flip_vertical=flip_vertical,
            )
            base_vox, disp_vox = cls._slab_voxel_basis(cbct_volume, base_surface, disp_x_cols)
            cls._SLAB_BASIS_CACHE[geom_key] = (base_vox, disp_vox, u_offsets, n_layers)

        dim_y, dim_z = base_vox.shape[:2]
        accum: Optional[np.ndarray] = None
        try:
            from scipy import ndimage  # type: ignore

            # Pre-allocate reusable float32 coordinate & output buffers (eliminates per-layer allocs)
            nz, ny, nx = cbct_volume.image.shape
            max_z = np.float32(max(0.0, float(nz) - 1.0 - 1e-5))
            max_y = np.float32(max(0.0, float(ny) - 1.0 - 1e-5))
            max_x = np.float32(max(0.0, float(nx) - 1.0 - 1e-5))
            lim_z = np.float32(float(nz) - 0.5)
            lim_y = np.float32(float(ny) - 0.5)
            lim_x = np.float32(float(nx) - 0.5)
            half_neg = np.float32(-0.5)
            zero_f = np.float32(0.0)

            coords = np.empty((3, dim_y, dim_z), dtype=np.float32)
            layer_buf = np.empty((dim_y, dim_z), dtype=np.float32)
            in_fov = np.empty((dim_y, dim_z), dtype=bool)
            tmp_cmp = np.empty((dim_y, dim_z), dtype=bool)
            base_z = np.ascontiguousarray(base_vox[:, :, 2], dtype=np.float32)
            base_y = np.ascontiguousarray(base_vox[:, :, 1], dtype=np.float32)
            base_x = np.ascontiguousarray(base_vox[:, :, 0], dtype=np.float32)
            disp_z = np.ascontiguousarray(disp_vox[None, :, 2], dtype=np.float32)
            disp_y = np.ascontiguousarray(disp_vox[None, :, 1], dtype=np.float32)
            disp_x = np.ascontiguousarray(disp_vox[None, :, 0], dtype=np.float32)

            for u_val in u_offsets:
                u_f = np.float32(u_val)
                np.multiply(disp_z, u_f, out=coords[0])
                coords[0] += base_z
                np.multiply(disp_y, u_f, out=coords[1])
                coords[1] += base_y
                np.multiply(disp_x, u_f, out=coords[2])
                coords[2] += base_x

                np.greater_equal(coords[0], half_neg, out=in_fov)
                np.less_equal(coords[0], lim_z, out=tmp_cmp)
                in_fov &= tmp_cmp
                np.greater_equal(coords[1], half_neg, out=tmp_cmp)
                in_fov &= tmp_cmp
                np.less_equal(coords[1], lim_y, out=tmp_cmp)
                in_fov &= tmp_cmp
                np.greater_equal(coords[2], half_neg, out=tmp_cmp)
                in_fov &= tmp_cmp
                np.less_equal(coords[2], lim_x, out=tmp_cmp)
                in_fov &= tmp_cmp

                np.clip(coords[0], zero_f, max_z, out=coords[0])
                np.clip(coords[1], zero_f, max_y, out=coords[1])
                np.clip(coords[2], zero_f, max_x, out=coords[2])
                ndimage.map_coordinates(
                    cbct_volume.image,
                    coords,
                    output=layer_buf,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
                if not np.all(in_fov):
                    layer_buf[~in_fov] = -1000.0
                if accum is None:
                    accum = layer_buf.copy()
                elif mode == "MIP":
                    np.maximum(accum, layer_buf, out=accum)
                elif mode == "MINIP":
                    np.minimum(accum, layer_buf, out=accum)
                else:
                    accum += layer_buf
        except ImportError:
            for u_val in u_offsets:
                sampled = cls._sample_slab_layer(cbct_volume, base_vox, disp_vox, float(u_val))
                if accum is None:
                    accum = sampled
                elif mode == "MIP":
                    np.maximum(accum, sampled, out=accum)
                elif mode == "MINIP":
                    np.minimum(accum, sampled, out=accum)
                else:
                    accum += sampled

        assert accum is not None
        out = cls._finalize_projection(accum, mode, n_layers)
        cls._SLAB_PROJECTION_CACHE[proj_key] = out.copy()
        return out

    @staticmethod
    def _crop_panoramic_vertical_padding(
        proj_yk: np.ndarray,
        spacing_mm: float,
        height_mm: float,
        projection_mode: str = "RaySum",
        margin_mm: float = 2.0,
    ) -> Tuple[np.ndarray, float]:
        """Trim empty out-of-volume / external-air bands from top and bottom of panoramic array."""
        arr = np.asarray(proj_yk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 4:
            return arr, float(height_mm)

        mode = (projection_mode or "RaySum").strip().upper()
        scale = 1.25 if mode == "RAYSUM" else 1.0
        pad_cutoff = -1200.0 if mode == "RAYSUM" else -950.0
        valid_vals = arr[arr > pad_cutoff]
        if valid_vals.size == 0:
            return arr, float(height_mm)

        v_min = float(np.min(valid_vals))
        v_max = float(np.max(valid_vals))
        is_hu = v_min <= (-500.0 * scale) and (300.0 * scale) < v_max <= (10000.0 * scale)
        content_thresh = (-680.0 if mode == "RAYSUM" else -550.0) if is_hu else pad_cutoff

        row_p98 = np.percentile(arr, 98.0, axis=1)
        active = np.where(row_p98 > content_thresh)[0]
        if active.size == 0:
            active = np.where(row_p98 > pad_cutoff)[0]
        if active.size == 0:
            return arr, float(height_mm)

        sp = max(0.15, float(spacing_mm))
        pad_px = max(1, int(round(float(margin_mm) / sp)))
        r0 = max(0, int(active[0]) - pad_px)
        r1 = min(arr.shape[0], int(active[-1]) + 1 + pad_px)

        min_rows = min(arr.shape[0], max(4, int(round(20.0 / sp))))
        if (r1 - r0) < min_rows:
            mid = (r0 + r1) // 2
            r0 = max(0, mid - min_rows // 2)
            r1 = min(arr.shape[0], r0 + min_rows)
            r0 = max(0, r1 - min_rows)

        if r0 == 0 and r1 == arr.shape[0]:
            return arr, float(height_mm)
        cropped = arr[r0:r1, :]
        eff_height_mm = round(float(cropped.shape[0]) * sp, 2)
        return cropped, eff_height_mm

    @staticmethod
    def compute_panoramic_display_window_threshold(
        proj_yk: np.ndarray,
        projection_mode: str = "RaySum",
        source_is_hu: bool = True,
    ) -> Tuple[float, float, float, float]:
        """Compute `(window, level, lower_threshold, upper_threshold)` for transparent-air panoramic display."""
        arr = np.asarray(proj_yk, dtype=np.float32)
        if arr.size == 0:
            return 2000.0, 400.0, -937.5, 4000.0

        mode = (projection_mode or "RaySum").strip().upper()
        scale = 1.25 if mode == "RAYSUM" else 1.0
        pad_cutoff = -1200.0 if mode == "RAYSUM" else -950.0
        valid = arr[arr > pad_cutoff]
        if valid.size == 0:
            valid = arr.reshape(-1)

        if source_is_hu:
            lower_threshold = -750.0 * scale
        else:
            vmin = float(np.min(valid))
            p2, p99 = np.percentile(valid, [2.0, 99.0])
            if p2 <= vmin + 0.05 * max(1.0, float(p99 - vmin)):
                lower_threshold = float(p2)
            else:
                lower_threshold = max(pad_cutoff + 50.0, vmin - 1.0)

        fg = valid[valid > lower_threshold]
        if fg.size < 16:
            fg = valid
        lo, hi = np.percentile(fg, [2.0, 99.5])
        window = max(1.0, float(hi - lo))
        level = float(0.5 * (lo + hi))
        upper_threshold = max(float(np.max(arr)) + 1000.0, float(hi) + window)
        return window, level, float(lower_threshold), float(upper_threshold)

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
        projection_mode: str = "RaySum",
        source_is_hu: Optional[bool] = None,
        trim_vertical_padding: bool = True,
    ) -> Any:
        """Write 2D panoramic array `(dim_y, dim_z)` into MRML panoramic volume node and display."""
        if trim_vertical_padding:
            proj_yk, height_mm = self._crop_panoramic_vertical_padding(
                proj_yk,
                spacing_mm=spacing_mm,
                height_mm=height_mm,
                projection_mode=projection_mode,
            )

        if source_is_hu is None:
            source_is_hu = True
            if volumeNode is not None and slicer is not None:
                try:
                    from .VRVisualization import VRVisualizationLogic

                    src_arr = slicer.util.arrayFromVolume(volumeNode)
                    source_is_hu = VRVisualizationLogic.is_volume_hu_calibrated(src_arr)
                except Exception:
                    logger.debug("Could not inspect CBCT HU calibration for pano display", exc_info=True)

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
        unsigned_types = tuple(
            getattr(vtk, name)
            for name in ("VTK_UNSIGNED_CHAR", "VTK_UNSIGNED_SHORT", "VTK_UNSIGNED_INT")
            if hasattr(vtk, name)
        )
        if scalar_type in unsigned_types and float(np.min(proj_yk)) < 0.0:
            scalar_type = getattr(vtk, "VTK_FLOAT", scalar_type)
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
        panoNode.SetAttribute("CBCTPlanner.ProjectionMode", str(projection_mode))
        panoNode.CreateDefaultDisplayNodes()

        disp = panoNode.GetDisplayNode()
        if disp is not None:
            win, lvl, lo_th, hi_th = self.compute_panoramic_display_window_threshold(
                proj_yk,
                projection_mode=projection_mode,
                source_is_hu=bool(source_is_hu),
            )
            d_mod = disp.StartModify() if hasattr(disp, "StartModify") else None
            try:
                if hasattr(disp, "SetAutoWindowLevel"):
                    disp.SetAutoWindowLevel(False)
                if hasattr(disp, "SetWindowLevel"):
                    disp.SetWindowLevel(win, lvl)
                else:
                    if hasattr(disp, "SetWindow"):
                        disp.SetWindow(win)
                    if hasattr(disp, "SetLevel"):
                        disp.SetLevel(lvl)
                if hasattr(disp, "SetApplyThreshold"):
                    disp.SetApplyThreshold(True)
                if hasattr(disp, "SetAutoThreshold"):
                    disp.SetAutoThreshold(False)
                if hasattr(disp, "SetThreshold"):
                    disp.SetThreshold(lo_th, hi_th)
                else:
                    if hasattr(disp, "SetLowerThreshold"):
                        disp.SetLowerThreshold(lo_th)
                    if hasattr(disp, "SetUpperThreshold"):
                        disp.SetUpperThreshold(hi_th)
                if hasattr(disp, "SetInterpolate"):
                    disp.SetInterpolate(True)
                if hasattr(disp, "SetOpacity"):
                    disp.SetOpacity(1.0)
            finally:
                if d_mod is not None and hasattr(disp, "EndModify"):
                    disp.EndModify(d_mod)

        if show_in_green_view:
            self.display_panoramic_in_slice_view(panoNode, slice_view_name="Green", force_fit=force_fit)
        return panoNode

    def generate_panoramic_volume(
        self,
        volumeNode: Any,
        curveNode: Any,
        slab_thickness_mm: float = 25.0,
        height_mm: float = 120.0,
        spacing_mm: float = 0.5,
        buccolingual_offset_mm: float = 0.0,
        projection_mode: str = "RaySum",
        show_in_green_view: bool = True,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
        async_cli: bool = False,
        on_complete: Optional[Callable[[Any, Any], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
        parameterNode: Any = None,
        force_fit: bool = False,
        flip_vertical: bool = False,
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
                flip_vertical=flip_vertical,
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
                projection_mode=projection_mode,
            )
            if on_progress is not None:
                on_progress(100)
            if on_complete is not None:
                on_complete(panoNode, None)
            return panoNode, None

        # O1: Non-blocking chunked QTimer slab sampling (shared geometry with the sync path)
        base_surface, disp_x_cols, u_offsets, n_layers, _, _ = self._build_slab_geometry(
            pts_arch=pts_arch,
            normals=normals,
            binormals=binormals,
            slab_thickness_mm=slab_thickness_mm,
            height_mm=height_mm,
            spacing_mm=spacing_mm,
            buccolingual_offset_mm=buccolingual_offset_mm,
            flip_vertical=flip_vertical,
        )
        mode = projection_mode.strip().upper()
        base_vox, disp_vox = self._slab_voxel_basis(cbct_vol, base_surface, disp_x_cols)
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
                sampled = self._sample_slab_layer(cbct_vol, base_vox, disp_vox, u_val)
                if state["accum"] is None:
                    state["accum"] = sampled
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
                accum = self._finalize_projection(state["accum"], mode, n_layers)
                self.last_generation_elapsed_sec = time.perf_counter() - t0
                panoNode = self._populate_panoramic_node_from_array(
                    volumeNode=volumeNode,
                    proj_yk=accum,
                    spacing_mm=spacing_mm,
                    total_len_mm=total_len,
                    height_mm=height_mm,
                    show_in_green_view=show_in_green_view,
                    parameterNode=parameterNode,
                    force_fit=force_fit,
                    projection_mode=projection_mode,
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
        """B10: Display Panoramic Radiograph in `slice_view_name`, clear extra layers, and fit tightly.

        Calls `FitSliceToAll()` ONLY on first display or when geometry changes (or `force_fit=True`).
        Returns True if fitted, False if existing user zoom/pan was preserved.
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
        if compositeNode:
            if prev_bg_id != pano_id:
                compositeNode.SetBackgroundVolumeID(pano_id)
            if hasattr(compositeNode, "SetForegroundVolumeID"):
                compositeNode.SetForegroundVolumeID(None)
            if hasattr(compositeNode, "SetLabelVolumeID"):
                compositeNode.SetLabelVolumeID(None)
            if hasattr(compositeNode, "SetForegroundOpacity"):
                compositeNode.SetForegroundOpacity(0.0)
            if hasattr(compositeNode, "SetLabelOpacity"):
                compositeNode.SetLabelOpacity(0.0)

        total_arc = float(panoNode.GetAttribute("CBCTPlanner.TotalArcMm") or 0.0)
        height_mm = float(panoNode.GetAttribute("CBCTPlanner.HeightMm") or 0.0)
        if (total_arc <= 0.0 or height_mm <= 0.0) and hasattr(panoNode, "GetRASBounds"):
            bounds = [0.0] * 6
            panoNode.GetRASBounds(bounds)
            if total_arc <= 0.0:
                total_arc = abs(float(bounds[1]) - float(bounds[0]))
            if height_mm <= 0.0:
                height_mm = abs(float(bounds[3]) - float(bounds[2]))

        prev_geom = self._fitted_pano_geometry.get(slice_view_name)
        geometry_changed = (
            prev_geom is None
            or prev_geom[0] != pano_id
            or abs(prev_geom[1] - total_arc) > 2.0
            or abs(prev_geom[2] - height_mm) > 2.0
        )

        sliceLogic = widget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        if hasattr(sliceNode, "SetBackgroundColor"):
            sliceNode.SetBackgroundColor(0.0, 0.0, 0.0)
        if hasattr(sliceNode, "SetBackgroundColor2"):
            sliceNode.SetBackgroundColor2(0.0, 0.0, 0.0)

        did_fit = False
        if force_fit or prev_bg_id != pano_id or geometry_changed:
            was_mod = sliceNode.StartModify() if hasattr(sliceNode, "StartModify") else None
            try:
                sliceNode.SetOrientationToAxial()
                sliceLogic.FitSliceToAll()
                if (
                    total_arc > 1.0
                    and height_mm > 1.0
                    and hasattr(sliceNode, "GetDimensions")
                    and hasattr(sliceNode, "GetFieldOfView")
                    and hasattr(sliceNode, "SetFieldOfView")
                ):
                    dims = sliceNode.GetDimensions()
                    fov = sliceNode.GetFieldOfView()
                    vp_w = float(dims[0]) if dims and len(dims) >= 2 else 0.0
                    vp_h = float(dims[1]) if dims and len(dims) >= 2 else 0.0
                    if vp_w > 1.0 and vp_h > 1.0:
                        vp_aspect = vp_w / vp_h
                        pano_aspect = total_arc / height_mm
                        if vp_aspect >= pano_aspect:
                            fov_y = height_mm
                            fov_x = height_mm * vp_aspect
                        else:
                            fov_x = total_arc
                            fov_y = total_arc / vp_aspect
                        fov_z = float(fov[2]) if fov and len(fov) >= 3 else 1.0
                        sliceNode.SetFieldOfView(fov_x, fov_y, fov_z)
                        if hasattr(sliceNode, "UpdateMatrices"):
                            sliceNode.UpdateMatrices()
            finally:
                if was_mod is not None and hasattr(sliceNode, "EndModify"):
                    sliceNode.EndModify(was_mod)
            self._fitted_pano_geometry[slice_view_name] = (pano_id, total_arc, height_mm)
            did_fit = True
        else:
            sliceNode.Modified()
        if hasattr(widget, "sliceView") and hasattr(widget.sliceView(), "forceRender"):
            try:
                widget.sliceView().forceRender()
            except Exception:
                logger.debug("Panoramic sliceView.forceRender skipped", exc_info=True)
        return did_fit

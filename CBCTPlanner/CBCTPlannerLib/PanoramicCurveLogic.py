"""Dental Arch Curve Visualization and Panoramic Curved Planar Reformation (CPR).

Provides:
- Interactive Upper/Lower dental arch vtkMRMLMarkupsCurveNode creation & styling.
- RAS Right-to-Left (+X -> -X) frame computation with configurable occlusal-plane normal
  and MTime-based frame caching.
- Right-handed (det = +1) Curved Planar Reformation (CPR) straightening transform.
- Panoramic Radiograph generation with Patient Right on screen left, non-blocking CLI support,
  and exact `pano_x(arc_mm)` / `arc_mm(pano_x)` coordinate conversion helpers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
except ImportError:
    vtk = None
    vtk_to_numpy = None

from .core.arch_detector import DentalArch


class PanoramicCurveLogic:
    """Handles dental arch curve visualization, frame caching, and panoramic CPR generation."""

    def __init__(self) -> None:
        self.transform_spacing_factor: float = 5.0
        # Cache sampled curve frames keyed by (node_id, mtime, num_samples, occlusal_tuple)
        self._frame_cache: Dict[
            Tuple[str, int, int, Tuple[float, float, float]],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._active_cli_node: Any = None
        self._active_cli_observer: Optional[int] = None

    @staticmethod
    def pano_x(arc_mm: float) -> float:
        """Convert physical arc-length along arch (0 = Patient Right, L = Patient Left) to Panoramic RAS X.

        In Slicer's Axial slice view (Green Panoramic viewport), +X_RAS is on Screen Left and
        -X_RAS is on Screen Right. Mapping arc_mm -> -arc_mm places Patient Right (arc_mm = 0)
        on Screen Left and progresses toward Screen Right (-X_RAS) as arc_mm increases toward Patient Left.
        """
        return -float(arc_mm)

    @staticmethod
    def arc_mm(pano_x: float, max_arc_mm: Optional[float] = None) -> float:
        """Convert Panoramic RAS X coordinate back to physical arc-length in mm along the arch."""
        val = max(0.0, -float(pano_x))
        if max_arc_mm is not None:
            val = min(float(max_arc_mm), val)
        return val

    @staticmethod
    def compute_cpr_direction_matrix(
        curve_start_ras: np.ndarray,
        curve_end_ras: np.ndarray,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """Compute a strictly right-handed (det = +1) 4x4 grid direction matrix for CPR straightening.

        Columns of the 3x3 rotation block:
          - axis_x (col 0): Buccal-Lingual axis (perpendicular to arch chord & occlusal normal)
          - axis_y (col 1): Occlusal / Inferior-Superior axis (aligned with occlusal_normal)
          - axis_z (col 2): Along-arch progression (from Patient Right start to Patient Left end)
        Because axis_x = axis_y x axis_z, det([axis_x, axis_y, axis_z]) = +1.0 always.
        """
        up = np.asarray(occlusal_normal, dtype=np.float64).reshape(3)
        up_norm = float(np.linalg.norm(up))
        axis_y = up / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if axis_y[2] < 0:
            axis_y = -axis_y

        chord = np.asarray(curve_end_ras, dtype=np.float64) - np.asarray(curve_start_ras, dtype=np.float64)
        chord_in_plane = chord - np.dot(chord, axis_y) * axis_y
        c_norm = float(np.linalg.norm(chord_in_plane))
        axis_z = (
            chord_in_plane / c_norm
            if c_norm > 1e-8
            else np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        )

        # Right-handed cross product: axis_x = axis_y x axis_z -> det([axis_x, axis_y, axis_z]) = +1
        axis_x = np.cross(axis_y, axis_z)
        x_norm = float(np.linalg.norm(axis_x))
        axis_x = axis_x / x_norm if x_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Re-orthogonalize axis_z = axis_x x axis_y
        axis_z = np.cross(axis_x, axis_y)
        axis_z /= max(1e-8, float(np.linalg.norm(axis_z)))

        mat = np.eye(4, dtype=np.float64)
        mat[0:3, 0] = axis_x
        mat[0:3, 1] = axis_y
        mat[0:3, 2] = axis_z
        return mat

    @staticmethod
    def create_or_update_arch_curve(
        volumeNode: Any,
        jaw: str,
        points_ras: Optional[np.ndarray] = None,
        color: Optional[Tuple[float, float, float]] = None,
        visible: bool = True,
        show_labels: bool = False,
        line_diameter: float = 1.8,
    ) -> Any:
        """Create or update a vtkMRMLMarkupsCurveNode for the specified jaw ('Upper' or 'Lower').

        Ensures points_ras are ordered Patient Right (+X) -> Patient Left (-X) in RAS.
        If points_ras is None or empty, creates an empty curve node ready for interactive placement.
        """
        if slicer is None:
            return None
        jaw_cap = jaw.capitalize()
        vol_name = volumeNode.GetName() if volumeNode else "CBCT"
        name = f"{vol_name}_{jaw_cap}_DentalArch"
        curveNode = slicer.mrmlScene.GetFirstNodeByName(name)
        if not curveNode:
            curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
            curveNode.CreateDefaultDisplayNodes()

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

        if color is None:
            color = (0.02, 0.71, 0.83) if jaw_cap == "Lower" else (0.92, 0.70, 0.03)

        dNode = curveNode.GetDisplayNode()
        if dNode:
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
        return curveNode

    @staticmethod
    def extract_world_points_from_curve_node(curveNode: Any) -> np.ndarray:
        """Extract curve world points using vtk_to_numpy (zero Python point-by-point loops when polydata exists)."""
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            return np.empty((0, 3), dtype=np.float64)

        curve_poly = curveNode.GetCurveWorld() if hasattr(curveNode, "GetCurveWorld") else None
        if curve_poly is not None and curve_poly.GetPoints() is not None and curve_poly.GetNumberOfPoints() >= 2:
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

    def get_curve_points_and_frames(
        self,
        curveNode: Any,
        num_samples: int = 100,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample points, arc-lengths, tangents, buccal normals, and occlusal binormals along curveNode.

        Cached by curveNode MTime so repeated cursor moves do not recompute frames.
        """
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            raise ValueError("Curve node must have at least 2 control points.")

        node_id = curveNode.GetID() if hasattr(curveNode, "GetID") else str(id(curveNode))
        mtime = int(curveNode.GetMTime()) if hasattr(curveNode, "GetMTime") else 0
        occ_tuple = (float(occlusal_normal[0]), float(occlusal_normal[1]), float(occlusal_normal[2]))
        cache_key = (node_id, mtime, int(num_samples), occ_tuple)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        raw_pts = self.extract_world_points_from_curve_node(curveNode)
        if len(raw_pts) < 2:
            raise ValueError("Curve node has fewer than 2 world points.")

        # Enforce RAS Right (+X) -> Left (-X) progression
        if raw_pts[0, 0] < raw_pts[-1, 0]:
            raw_pts = raw_pts[::-1].copy()

        diffs = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = float(cum_len[-1])
        if total_len < 1e-6:
            raise ValueError("Dental arch curve has zero length.")

        target_s = np.linspace(0.0, total_len, num_samples)
        pts = np.column_stack([np.interp(target_s, cum_len, raw_pts[:, d]) for d in range(3)])

        up = np.asarray(occ_tuple, dtype=np.float64)
        up_norm = float(np.linalg.norm(up))
        up = up / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up[2] < 0:
            up = -up

        # Project tangent onto occlusal plane
        raw_tangents = np.gradient(pts, axis=0)
        tangents = raw_tangents - np.outer(raw_tangents @ up, up)
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        t_norms[t_norms < 1e-8] = 1.0
        tangents = tangents / t_norms

        # Buccal normal: N = T x Up (points outward/buccally along a RAS Right->Left arch)
        normals = np.cross(tangents, up)
        n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        n_norms[n_norms < 1e-8] = 1.0
        normals = normals / n_norms

        # Binormal: B = N x T = +Up
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
            curveNode, num_samples=200, occlusal_normal=occlusal_normal
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
        """Project a 3D RAS point onto curveNode using GetClosestPointPositionAlongCurveWorld and return arc_mm."""
        pt = np.asarray(world_pt, dtype=np.float64).reshape(3)
        pts, arc_s, _, _, _ = self.get_curve_points_and_frames(
            curveNode, num_samples=200, occlusal_normal=occlusal_normal
        )
        # First use Slicer's native GetClosestPointPositionAlongCurveWorld if available to snap to curve
        if hasattr(curveNode, "GetClosestPointPositionAlongCurveWorld"):
            closest_pt = [0.0, 0.0, 0.0]
            try:
                curveNode.GetClosestPointPositionAlongCurveWorld(
                    [float(pt[0]), float(pt[1]), float(pt[2])], closest_pt
                )
                pt = np.asarray(closest_pt, dtype=np.float64)
            except Exception:
                pass

        dists_sq = np.sum((pts - pt) ** 2, axis=1)
        idx = int(np.argmin(dists_sq))
        return float(arc_s[idx])

    def compute_straightening_transform(
        self,
        transformNode: Any,
        curveNode: Any,
        slice_size_mm: Sequence[float] = (20.0, 50.0),
        spacing_along_curve_mm: float = 0.5,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> float:
        """Compute a right-handed (det = +1) vtkOrientedGridTransform mapping straightened -> curve RAS space."""
        if vtk is None or slicer is None:
            raise RuntimeError("VTK and Slicer are required for curved planar reformation.")

        pts_check, arc_s_check, _, _, _ = self.get_curve_points_and_frames(
            curveNode, num_samples=50, occlusal_normal=occlusal_normal
        )
        total_len = float(arc_s_check[-1])
        if total_len < 5.0:
            raise ValueError("Dental arch curve is too short for panoramic reformation.")

        resampling_spacing = max(0.2, spacing_along_curve_mm * self.transform_spacing_factor)
        num_slices = max(4, int(round(total_len / resampling_spacing)) + 1)

        pts, _, _, normals, binormals = self.get_curve_points_and_frames(
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
            n_k = normals[k]
            b_k = binormals[k]
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
                        + (i - 0.5) * slice_size_mm[0] * n_k
                        + (j - 0.5) * slice_size_mm[1] * b_k
                    )
                    displacements[k, j, i] = input_ras - straight_ras

        slicer.util.arrayFromGridTransformModified(transformNode)
        return total_len

    def cancel_active_panoramic_cli(self) -> None:
        """Cancel an in-flight asynchronous panoramic resampling CLI job if running."""
        if self._active_cli_node is not None:
            try:
                if self._active_cli_observer is not None:
                    self._active_cli_node.RemoveObserver(self._active_cli_observer)
                self._active_cli_node.Cancel()
            except Exception:
                pass
            self._active_cli_node = None
            self._active_cli_observer = None

    def _project_straightened_to_panoramic(
        self,
        volumeNode: Any,
        straightNode: Any,
        spacing_mm: float,
        projection_mode: str = "Mean",
        show_in_green_view: bool = True,
    ) -> Any:
        vol_name = volumeNode.GetName()
        pano_name = f"{vol_name}_Panoramic"
        panoNode = slicer.mrmlScene.GetFirstNodeByName(pano_name)
        if not panoNode:
            panoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", pano_name)

        straight_arr = slicer.util.arrayFromVolume(straightNode)  # (Z=arch, Y=height, X=thickness)
        dim_z, dim_y, _ = straight_arr.shape
        if projection_mode.upper() == "MIP":
            proj_2d = np.max(straight_arr, axis=2)
        elif projection_mode.upper() == "MINIP":
            proj_2d = np.min(straight_arr, axis=2)
        else:
            proj_2d = np.mean(straight_arr, axis=2)

        pano_img = vtk.vtkImageData()
        pano_img.SetDimensions(dim_z, dim_y, 1)
        pano_img.AllocateScalars(straightNode.GetImageData().GetScalarType(), 1)
        panoNode.SetAndObserveImageData(pano_img)

        pano_arr = slicer.util.arrayFromVolume(panoNode)
        pano_arr[0, :, :] = proj_2d.T
        slicer.util.arrayFromVolumeModified(panoNode)

        # Set IJK->RAS so i=0 (Patient Right, arc_mm=0) is at X_RAS = 0 (Screen Left in Axial view)
        # and i=dim_z-1 (Patient Left, arc_mm=L) is at X_RAS = -L (Screen Right in Axial view),
        # with j increasing along +Y_RAS (Screen Up) and k along -Z_RAS so det(IJKToRAS) > 0.
        pano_ijk = np.eye(4, dtype=np.float64)
        pano_ijk[0, 0] = -float(spacing_mm)
        pano_ijk[1, 1] = float(spacing_mm)
        pano_ijk[2, 2] = -1.0
        panoNode.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(pano_ijk))
        panoNode.SetAttribute("CBCTPlanner.IsPanoramic", "1")
        panoNode.SetAttribute("CBCTPlanner.SpacingMm", str(spacing_mm))
        panoNode.SetAttribute("CBCTPlanner.HeightMm", str(dim_y * spacing_mm))
        panoNode.SetAttribute("CBCTPlanner.TotalArcMm", str(dim_z * spacing_mm))
        panoNode.CreateDefaultDisplayNodes()
        if volumeNode.GetDisplayNode():
            panoNode.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())

        if show_in_green_view:
            self.display_panoramic_in_slice_view(panoNode, slice_view_name="Green")
        return panoNode

    def generate_panoramic_volume(
        self,
        volumeNode: Any,
        curveNode: Any,
        slab_thickness_mm: float = 18.0,
        height_mm: float = 50.0,
        spacing_mm: float = 0.5,
        projection_mode: str = "Mean",
        show_in_green_view: bool = True,
        occlusal_normal: Sequence[float] = (0.0, 0.0, 1.0),
        async_cli: bool = False,
        on_complete: Optional[Callable[[Any, Any], None]] = None,
    ) -> Tuple[Optional[Any], Any]:
        """Generate a Curved Planar Reformatted (CPR) Panoramic Radiograph volume.

        Supports both synchronous (`async_cli=False`) and non-blocking asynchronous
        CLI execution (`async_cli=True`) with cancellation support.
        """
        if vtk is None or slicer is None:
            raise RuntimeError("VTK and Slicer are required for panoramic generation.")
        if not volumeNode or not curveNode:
            raise ValueError("Both a CBCT volume and a dental arch curve are required.")

        self.cancel_active_panoramic_cli()

        vol_name = volumeNode.GetName()
        transform_name = f"{vol_name}_CPR_Transform"
        transformNode = slicer.mrmlScene.GetFirstNodeByName(transform_name)
        if not transformNode:
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", transform_name)

        slice_size_mm = (float(slab_thickness_mm), float(height_mm))
        total_len = self.compute_straightening_transform(
            transformNode,
            curveNode,
            slice_size_mm=slice_size_mm,
            spacing_along_curve_mm=spacing_mm,
            occlusal_normal=occlusal_normal,
        )

        straight_name = f"{vol_name}_Straightened_CPR"
        straightNode = slicer.mrmlScene.GetFirstNodeByName(straight_name)
        if not straightNode:
            straightNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", straight_name)

        grid_tf = transformNode.GetTransformFromParentAs("vtkOrientedGridTransform")
        dir_mat = grid_tf.GetGridDirectionMatrix()
        grid_img = grid_tf.GetDisplacementGrid()
        grid_origin = grid_img.GetOrigin()

        ijk_to_ras = slicer.util.arrayFromVTKMatrix(dir_mat)
        ijk_to_ras = np.dot(ijk_to_ras, np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0]))
        ijk_to_ras[0:3, 3] = grid_origin

        dim_x = max(2, int(round(slice_size_mm[0] / spacing_mm)))
        dim_y = max(2, int(round(slice_size_mm[1] / spacing_mm)))
        dim_z = max(4, int(round(total_len / spacing_mm)))

        out_img = vtk.vtkImageData()
        out_img.SetExtent(0, dim_x - 1, 0, dim_y - 1, 0, dim_z - 1)
        out_img.AllocateScalars(
            volumeNode.GetImageData().GetScalarType(),
            volumeNode.GetImageData().GetNumberOfScalarComponents(),
        )
        straightNode.SetAndObserveImageData(out_img)
        straightNode.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(ijk_to_ras))

        params = {
            "inputVolume": volumeNode.GetID(),
            "outputVolume": straightNode.GetID(),
            "referenceVolume": straightNode.GetID(),
            "transformationFile": transformNode.GetID(),
            "interpolationType": "linear" if spacing_mm >= 1.0 else "bs",
        }

        if not async_cli:
            paramNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)
            if paramNode:
                slicer.mrmlScene.RemoveNode(paramNode)
            straightNode.CreateDefaultDisplayNodes()
            if volumeNode.GetDisplayNode():
                straightNode.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())
            panoNode = self._project_straightened_to_panoramic(
                volumeNode,
                straightNode,
                spacing_mm=spacing_mm,
                projection_mode=projection_mode,
                show_in_green_view=show_in_green_view,
            )
            if on_complete is not None:
                on_complete(panoNode, straightNode)
            return panoNode, straightNode

        # Non-blocking asynchronous CLI execution
        cliNode = slicer.cli.run(
            slicer.modules.resamplescalarvectordwivolume, None, params, wait_for_completion=False
        )
        self._active_cli_node = cliNode

        def _on_cli_modified(caller: Any, _event: Any) -> None:
            if not caller.IsBusy():
                if self._active_cli_observer is not None:
                    caller.RemoveObserver(self._active_cli_observer)
                self._active_cli_observer = None
                self._active_cli_node = None
                status = caller.GetStatusString()
                slicer.mrmlScene.RemoveNode(caller)
                if status == "Completed":
                    straightNode.CreateDefaultDisplayNodes()
                    if volumeNode.GetDisplayNode():
                        straightNode.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())
                    panoNode = self._project_straightened_to_panoramic(
                        volumeNode,
                        straightNode,
                        spacing_mm=spacing_mm,
                        projection_mode=projection_mode,
                        show_in_green_view=show_in_green_view,
                    )
                    if on_complete is not None:
                        on_complete(panoNode, straightNode)

        self._active_cli_observer = cliNode.AddObserver(
            vtk.vtkCommand.ModifiedEvent, _on_cli_modified
        )
        return None, straightNode

    @staticmethod
    def display_panoramic_in_slice_view(panoNode: Any, slice_view_name: str = "Green") -> None:
        """Display the generated Panoramic Radiograph in the target slice view and fit."""
        if slicer is None:
            return
        layoutManager = slicer.app.layoutManager() if getattr(slicer, "app", None) else None
        if not layoutManager or not panoNode:
            return
        widget = layoutManager.sliceWidget(slice_view_name)
        if not widget:
            return
        compositeNode = widget.mrmlSliceCompositeNode()
        compositeNode.SetBackgroundVolumeID(panoNode.GetID())
        sliceLogic = widget.sliceLogic()
        sliceNode = sliceLogic.GetSliceNode()
        sliceNode.SetOrientationToAxial()
        sliceLogic.FitSliceToAll()
        sliceNode.Modified()

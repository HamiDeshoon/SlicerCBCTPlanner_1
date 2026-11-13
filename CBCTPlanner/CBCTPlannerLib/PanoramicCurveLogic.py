"""Dental Arch Curve Visualization and Panoramic Curved Planar Reformation (CPR).

Provides:
- Interactive Upper/Lower dental arch vtkMRMLMarkupsCurveNode creation & styling.
- Parallel Transport Frame computation along the arch curve.
- Straightened volume resampling (CPR) and Panoramic Radiograph (Mean / MIP) generation
  inside 3D Slicer, inspired by PerkLab's CurvedPlanarReformat architecture.
"""

from typing import Optional, Sequence, Tuple
import numpy as np
import slicer

try:
    import vtk
except ImportError:
    vtk = None


class PanoramicCurveLogic:
    """Handles dental arch curve visualization and panoramic CPR generation."""

    def __init__(self) -> None:
        self.transform_spacing_factor: float = 5.0

    @staticmethod
    def create_or_update_arch_curve(
        volumeNode,
        jaw: str,
        points_ras: np.ndarray,
        color: Optional[Tuple[float, float, float]] = None,
        visible: bool = True,
        show_labels: bool = False,
        line_diameter: float = 1.8,
    ):
        """Create or update a vtkMRMLMarkupsCurveNode for the specified jaw ('Upper' or 'Lower')."""
        jaw_cap = jaw.capitalize()
        name = f"{volumeNode.GetName()}_{jaw_cap}_DentalArch"
        curveNode = slicer.mrmlScene.GetFirstNodeByName(name)
        if not curveNode:
            curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
            curveNode.CreateDefaultDisplayNodes()

        curveNode.RemoveAllControlPoints()
        for pt in points_ras:
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
    def get_curve_points_and_frames(
        curveNode,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample points, arc-lengths, tangents, normals, and upright binormals along curveNode.

        Returns (points, arc_lengths_mm, tangents, normals, binormals), each (N, 3) except arc_lengths (N,).
        """
        if not curveNode or curveNode.GetNumberOfControlPoints() < 2:
            raise ValueError("Curve node must have at least 2 control points.")

        curve_poly = curveNode.GetCurveWorld()
        if curve_poly is None or curve_poly.GetNumberOfPoints() < 2:
            # Fallback to control points directly
            n_cp = curveNode.GetNumberOfControlPoints()
            raw_pts = np.zeros((n_cp, 3), dtype=np.float64)
            buf = [0.0, 0.0, 0.0]
            for i in range(n_cp):
                curveNode.GetNthControlPointPositionWorld(i, buf)
                raw_pts[i] = buf
        else:
            n_poly = curve_poly.GetNumberOfPoints()
            raw_pts = np.array([curve_poly.GetPoint(i) for i in range(n_poly)], dtype=np.float64)

        # Uniformly resample by arc length
        diffs = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = float(cum_len[-1])
        if total_len < 1e-6:
            raise ValueError("Dental arch curve has zero length.")

        target_s = np.linspace(0.0, total_len, num_samples)
        pts = np.column_stack(
            [np.interp(target_s, cum_len, raw_pts[:, d]) for d in range(3)]
        )

        # Tangents via central differences
        tangents = np.gradient(pts, axis=0)
        t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        t_norms[t_norms < 1e-8] = 1.0
        tangents = tangents / t_norms

        # Upright Binormal = Superior (+Z), Normal = T x B (buccal-lingual)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normals = np.cross(tangents, up)
        n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        n_norms[n_norms < 1e-8] = 1.0
        normals = normals / n_norms

        binormals = np.cross(normals, tangents)
        b_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
        b_norms[b_norms < 1e-8] = 1.0
        binormals = binormals / b_norms

        # Ensure binormal points Superior (+Z)
        flip = binormals[:, 2] < 0
        binormals[flip] *= -1.0
        normals[flip] *= -1.0

        return pts, target_s, tangents, normals, binormals

    def compute_straightening_transform(
        self,
        transformNode,
        curveNode,
        slice_size_mm: Sequence[float] = (30.0, 50.0),
        spacing_along_curve_mm: float = 0.5,
    ) -> float:
        """Compute a vtkOrientedGridTransform mapping straightened space -> curve RAS space.

        Returns total curve length in mm.
        """
        if vtk is None:
            raise RuntimeError("VTK is required for curved planar reformation.")

        total_len = float(curveNode.GetCurveLengthWorld())
        if total_len < 5.0:
            raise ValueError("Dental arch curve is too short for panoramic reformation.")

        resampling_spacing = spacing_along_curve_mm * self.transform_spacing_factor
        num_slices = max(4, int(round(total_len / resampling_spacing)) + 1)

        pts, _, _, normals, binormals = self.get_curve_points_and_frames(
            curveNode, num_samples=num_slices
        )

        # Straightened grid axes: X = buccal-lingual, Y = inferior-superior, Z = along arch
        grid_axis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        grid_axis_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        grid_axis_z = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        center_pt = np.mean(pts, axis=0)
        grid_origin = (
            center_pt
            - grid_axis_x * (slice_size_mm[0] / 2.0)
            - grid_axis_y * (slice_size_mm[1] / 2.0)
            - grid_axis_z * (total_len / 2.0)
        )

        grid_dims = [2, 2, num_slices]
        grid_spacing = [float(slice_size_mm[0]), float(slice_size_mm[1]), total_len / float(num_slices - 1)]

        dir_arr = np.eye(4, dtype=np.float64)
        dir_arr[0:3, 0] = grid_axis_x
        dir_arr[0:3, 1] = grid_axis_y
        dir_arr[0:3, 2] = grid_axis_z
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

    def generate_panoramic_volume(
        self,
        volumeNode,
        curveNode,
        slab_thickness_mm: float = 18.0,
        height_mm: float = 50.0,
        spacing_mm: float = 0.5,
        projection_mode: str = "Mean",
        show_in_green_view: bool = True,
    ):
        """Generate a Curved Planar Reformatted (CPR) Panoramic Radiograph volume.

        Creates `<VolName>_Panoramic` (and intermediate `<VolName>_Straightened_CPR`),
        computes Mean or MIP projection across `slab_thickness_mm`, and optionally
        displays it in Slicer's Green slice view.
        """
        if vtk is None:
            raise RuntimeError("VTK is required for panoramic generation.")
        if not volumeNode or not curveNode:
            raise ValueError("Both a CBCT volume and a dental arch curve are required.")

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
        )

        # Build reference straightened volume
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
            "interpolationType": "bs",
        }
        paramNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, params)
        if paramNode:
            slicer.mrmlScene.RemoveNode(paramNode)

        straightNode.CreateDefaultDisplayNodes()
        if volumeNode.GetDisplayNode():
            straightNode.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())

        # Project straightened volume into a 2D Panoramic Radiograph
        pano_name = f"{vol_name}_Panoramic"
        panoNode = slicer.mrmlScene.GetFirstNodeByName(pano_name)
        if not panoNode:
            panoNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", pano_name)

        straight_arr = slicer.util.arrayFromVolume(straightNode)  # (Z=arch, Y=height, X=thickness)
        if projection_mode.upper() == "MIP":
            proj_2d = np.max(straight_arr, axis=2)
        elif projection_mode.upper() == "MINIP":
            proj_2d = np.min(straight_arr, axis=2)
        else:
            proj_2d = np.mean(straight_arr, axis=2)

        # Transpose so horizontal axis is Arch (Z) and vertical axis is Height (Y)
        pano_img = vtk.vtkImageData()
        pano_img.SetDimensions(dim_z, dim_y, 1)
        pano_img.AllocateScalars(out_img.GetScalarType(), 1)
        panoNode.SetAndObserveImageData(pano_img)

        pano_arr = slicer.util.arrayFromVolume(panoNode)  # shape (1, dim_y, dim_z)
        pano_arr[0, :, :] = proj_2d.T
        slicer.util.arrayFromVolumeModified(panoNode)

        pano_ijk = np.eye(4, dtype=np.float64)
        pano_ijk[0, 0] = spacing_mm
        pano_ijk[1, 1] = spacing_mm
        pano_ijk[2, 2] = 1.0
        panoNode.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(pano_ijk))
        panoNode.CreateDefaultDisplayNodes()
        if volumeNode.GetDisplayNode():
            panoNode.GetDisplayNode().CopyContent(volumeNode.GetDisplayNode())

        if show_in_green_view:
            self.display_panoramic_in_slice_view(panoNode, slice_view_name="Green")

        return panoNode, straightNode

    @staticmethod
    def display_panoramic_in_slice_view(panoNode, slice_view_name: str = "Green") -> None:
        """Display the generated Panoramic Radiograph in the target slice view and fit."""
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

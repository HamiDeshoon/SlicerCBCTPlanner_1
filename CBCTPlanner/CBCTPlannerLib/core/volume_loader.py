"""
CBCT Volume container and Slicer volume array adapter.
All external file readers (SimpleITK, pydicom, nibabel) are loaded lazily on demand;
Slicer integration uses arrayFromVolume + GetIJKToRASMatrix directly with zero file I/O
and zero full-volume copy (O3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from . import ensure_optional_package

logger = logging.getLogger("CBCTPlanner.volume_loader")


@dataclass
class VolumeMetadata:
    patient_id: str = "Anonymous"
    patient_name: str = "Anonymous"
    study_date: str = ""
    series_description: str = ""
    modality: str = "CT"
    kvp: Optional[float] = None
    tube_current_ma: Optional[float] = None
    slice_thickness_mm: Optional[float] = None
    pixel_spacing_mm: Optional[Tuple[float, float]] = None
    window_center: float = 400.0
    window_width: float = 1600.0


class CBCTVolume:
    """Encapsulates a 3D Dental CBCT volume with (Z, Y, X) array, spacing (sz, sy, sx), and 4x4 IJK->RAS affine."""

    def __init__(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: np.ndarray,
        affine: np.ndarray,
        metadata: Optional[VolumeMetadata] = None,
        sitk_image: Optional[Any] = None,
        copy_image: bool = False,
    ):
        # O3: Keep zero-copy view of the underlying volume array unless copy_image=True is requested
        arr = np.asarray(image)
        if copy_image:
            self.image = arr.astype(np.float32, copy=True)
        else:
            self.image = arr
        self.spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))  # (sz, sy, sx)
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
        self.direction = np.asarray(direction, dtype=np.float64).reshape((3, 3))
        self.affine = np.asarray(affine, dtype=np.float64).reshape((4, 4))
        self._inv_affine = np.linalg.inv(self.affine)
        self.metadata = metadata or VolumeMetadata()
        self._sitk_image = sitk_image

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape as (Z, Y, X)."""
        return self.image.shape  # type: ignore[return-value]

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        """Volume shape as (X, Y, Z)."""
        return (self.image.shape[2], self.image.shape[1], self.image.shape[0])

    def world_to_voxel(self, world_coord_mm: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        """Convert physical world coordinate(s) (x_mm, y_mm, z_mm) to continuous voxel index (i_x, j_y, k_z).

        Supports both a single 3-vector `(3,)` and a batch of points `(N, 3)`.
        """
        pts = np.asarray(world_coord_mm, dtype=np.float64)
        if pts.ndim == 1:
            pt_homog = np.array([pts[0], pts[1], pts[2], 1.0], dtype=np.float64) if len(pts) == 3 else pts
            vox_homog = self._inv_affine @ pt_homog
            return vox_homog[:3]
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homog = np.hstack([pts[:, :3], ones])
        vox = (self._inv_affine @ homog.T).T
        return vox[:, :3]

    def world_direction_to_voxel(self, world_dirs: np.ndarray) -> np.ndarray:
        """Map world-space direction vectors to voxel-space directions (rotation only)."""
        dirs = np.asarray(world_dirs, dtype=np.float64)
        return dirs @ self._inv_affine[:3, :3].T

    def voxel_to_world(self, voxel_index: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        """Convert voxel index `(i_x, j_y, k_z)` or `(N, 3)` batch to physical world coordinate `(x_mm, y_mm, z_mm)`."""
        idx = np.asarray(voxel_index, dtype=np.float64)
        if idx.ndim == 1:
            idx_homog = np.array([idx[0], idx[1], idx[2], 1.0], dtype=np.float64) if len(idx) == 3 else idx
            world_homog = self.affine @ idx_homog
            return world_homog[:3]
        ones = np.ones((idx.shape[0], 1), dtype=np.float64)
        homog = np.hstack([idx[:, :3], ones])
        world = (self.affine @ homog.T).T
        return world[:, :3]

    def sample_world_points(
        self,
        world_pts_ras: np.ndarray,
        order: int = 1,
        cval: float = -1000.0,
    ) -> np.ndarray:
        """Sample volume intensities at arbitrary 3D RAS coordinates `(..., 3)` using `scipy.ndimage.map_coordinates`."""
        pts = np.asarray(world_pts_ras, dtype=np.float64)
        orig_shape = pts.shape[:-1]
        flat_pts = pts.reshape(-1, 3)
        vox_ijk = self.world_to_voxel(flat_pts)  # (N, 3) as (i_x, j_y, k_z)
        coords_zyx = np.vstack([vox_ijk[:, 2], vox_ijk[:, 1], vox_ijk[:, 0]])

        try:
            from scipy import ndimage  # type: ignore

            sampled = ndimage.map_coordinates(
                self.image, coords_zyx, order=order, mode="constant", cval=float(cval)
            )
            return np.asarray(sampled, dtype=np.float64).reshape(orig_shape)
        except ImportError:
            # Pure numpy nearest-neighbor fallback if scipy is absent
            kz = np.rint(coords_zyx[0]).astype(np.int64)
            jy = np.rint(coords_zyx[1]).astype(np.int64)
            ix = np.rint(coords_zyx[2]).astype(np.int64)
            z_dim, y_dim, x_dim = self.shape
            valid = (kz >= 0) & (kz < z_dim) & (jy >= 0) & (jy < y_dim) & (ix >= 0) & (ix < x_dim)
            out = np.full(len(flat_pts), float(cval), dtype=np.float64)
            out[valid] = self.image[kz[valid], jy[valid], ix[valid]].astype(np.float64)
            return out.reshape(orig_shape)

    def sample_mask_world_points(
        self,
        mask: np.ndarray,
        world_pts_ras: np.ndarray,
    ) -> np.ndarray:
        """Nearest-neighbor label sampling of a `(Z, Y, X)` segmentation mask at arbitrary 3D RAS points `(..., 3)`."""
        pts = np.asarray(world_pts_ras, dtype=np.float64)
        orig_shape = pts.shape[:-1]
        flat_pts = pts.reshape(-1, 3)
        vox_ijk = self.world_to_voxel(flat_pts)
        ix = np.rint(vox_ijk[:, 0]).astype(np.int64)
        jy = np.rint(vox_ijk[:, 1]).astype(np.int64)
        kz = np.rint(vox_ijk[:, 2]).astype(np.int64)
        z_dim, y_dim, x_dim = mask.shape
        valid = (kz >= 0) & (kz < z_dim) & (jy >= 0) & (jy < y_dim) & (ix >= 0) & (ix < x_dim)
        out = np.zeros(len(flat_pts), dtype=mask.dtype)
        out[valid] = mask[kz[valid], jy[valid], ix[valid]]
        return out.reshape(orig_shape)

    def get_hu_at_point(
        self, world_coord_mm: Union[np.ndarray, List[float], Tuple[float, ...]]
    ) -> Optional[float]:
        """Sample intensity at a physical coordinate; returns None if outside volume bounds."""
        vox = self.world_to_voxel(world_coord_mm)
        vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        if 0 <= vz < self.shape[0] and 0 <= vy < self.shape[1] and 0 <= vx < self.shape[2]:
            return float(self.image[vz, vy, vx])
        return None

    @classmethod
    def from_slicer_volume_node(cls, volumeNode: Any, copy_image: bool = False) -> "CBCTVolume":
        """Construct a CBCTVolume directly from a Slicer vtkMRMLScalarVolumeNode without file I/O or full copy (O3)."""
        import slicer  # type: ignore
        import vtk  # type: ignore

        img = slicer.util.arrayFromVolume(volumeNode)  # (Z, Y, X) zero-copy view
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        affine = np.array(
            [[mat.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
        # If the volume node has a parent linear transform (B3 non-destructive alignment), compose it into affine
        if hasattr(volumeNode, "GetParentTransformNode") and volumeNode.GetParentTransformNode():
            parent_tf = volumeNode.GetParentTransformNode()
            tf_mat = vtk.vtkMatrix4x4()
            if hasattr(parent_tf, "GetMatrixTransformToWorld"):
                parent_tf.GetMatrixTransformToWorld(tf_mat)
            elif hasattr(parent_tf, "GetMatrixTransformToParent"):
                parent_tf.GetMatrixTransformToParent(tf_mat)
            parent_arr = np.array(
                [[tf_mat.GetElement(r, c) for c in range(4)] for r in range(4)],
                dtype=np.float64,
            )
            affine = parent_arr @ affine

        sx = float(np.linalg.norm(affine[:3, 0]))
        sy = float(np.linalg.norm(affine[:3, 1]))
        sz = float(np.linalg.norm(affine[:3, 2]))
        origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
        direction = affine[:3, :3].copy()
        for col, scale in enumerate((sx, sy, sz)):
            if scale > 1e-8:
                direction[:, col] /= scale
        metadata = VolumeMetadata(
            series_description=volumeNode.GetName() if hasattr(volumeNode, "GetName") else ""
        )
        return cls(
            image=img,
            spacing=(sz, sy, sx),
            origin=origin,
            direction=direction,
            affine=affine,
            metadata=metadata,
            copy_image=copy_image,
        )

    @classmethod
    def from_nifti(cls, file_path: Union[str, Path]) -> "CBCTVolume":
        """Lazily load a NIfTI file via nibabel if requested outside Slicer."""
        nib = ensure_optional_package("nibabel", "nibabel", "NIfTI file loading")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {path}")
        nii = nib.load(str(path))
        data_xyz = np.asarray(nii.get_fdata(), dtype=np.float32)
        image_zyx = np.transpose(data_xyz, (2, 1, 0))
        affine = np.asarray(nii.affine, dtype=np.float64)
        sx = float(np.linalg.norm(affine[:3, 0]))
        sy = float(np.linalg.norm(affine[:3, 1]))
        sz = float(np.linalg.norm(affine[:3, 2]))
        origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
        direction = affine[:3, :3] / np.maximum([sx, sy, sz], 1e-8)
        return cls(
            image=image_zyx,
            spacing=(sz, sy, sx),
            origin=origin,
            direction=direction,
            affine=affine,
        )

    @classmethod
    def from_dicom_directory(cls, folder_path: Union[str, Path]) -> "CBCTVolume":
        """Lazily load a DICOM series via SimpleITK if requested outside Slicer."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"DICOM folder not found: {folder}")
        sitk = ensure_optional_package("SimpleITK", "SimpleITK", "DICOM directory loading")
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(folder))
        if not series_ids:
            raise ValueError(f"No DICOM series found in {folder}")
        dcm_names = reader.GetGDCMSeriesFileNames(str(folder), series_ids[0])
        reader.SetFileNames(dcm_names)
        img_sitk = reader.Execute()
        arr_zyx = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        sx, sy, sz = img_sitk.GetSpacing()
        ox, oy, oz = img_sitk.GetOrigin()
        dir_mat = np.array(img_sitk.GetDirection(), dtype=np.float64).reshape((3, 3))
        lps_to_ras = np.diag([-1.0, -1.0, 1.0])
        dir_ras = lps_to_ras @ dir_mat
        orig_ras = lps_to_ras @ np.array([ox, oy, oz], dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = dir_ras @ np.diag([sx, sy, sz])
        affine[:3, 3] = orig_ras
        return cls(
            image=arr_zyx,
            spacing=(float(sz), float(sy), float(sx)),
            origin=(float(orig_ras[0]), float(orig_ras[1]), float(orig_ras[2])),
            direction=dir_ras,
            affine=affine,
            sitk_image=img_sitk,
        )

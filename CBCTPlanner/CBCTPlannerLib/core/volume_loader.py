"""
CBCT Volume container and Slicer volume array adapter.
All external file readers (SimpleITK, pydicom, nibabel) are loaded lazily on demand;
Slicer integration uses arrayFromVolume + GetIJKToRASMatrix directly with zero file I/O.
"""

from __future__ import annotations

import logging
import os
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
    ):
        self.image = np.asarray(image, dtype=np.float32)
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
        """Convert physical world coordinate (x_mm, y_mm, z_mm) to continuous voxel index (i_x, j_y, k_z)."""
        pt = np.asarray(world_coord_mm, dtype=np.float64).reshape(-1)
        pt_homog = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float64) if len(pt) == 3 else pt
        vox_homog = self._inv_affine @ pt_homog
        return vox_homog[:3]

    def voxel_to_world(self, voxel_index: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        """Convert voxel index (i_x, j_y, k_z) to physical world coordinate (x_mm, y_mm, z_mm)."""
        idx = np.asarray(voxel_index, dtype=np.float64).reshape(-1)
        idx_homog = np.array([idx[0], idx[1], idx[2], 1.0], dtype=np.float64) if len(idx) == 3 else idx
        world_homog = self.affine @ idx_homog
        return world_homog[:3]

    def get_hu_at_point(self, world_coord_mm: Union[np.ndarray, List[float], Tuple[float, ...]]) -> Optional[float]:
        """Sample intensity (HU) at a physical coordinate; returns None if outside volume bounds."""
        vox = self.world_to_voxel(world_coord_mm)
        vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        if 0 <= vz < self.shape[0] and 0 <= vy < self.shape[1] and 0 <= vx < self.shape[2]:
            return float(self.image[vz, vy, vx])
        return None

    @classmethod
    def from_slicer_volume_node(cls, volumeNode: Any) -> "CBCTVolume":
        """Construct a CBCTVolume directly from a Slicer vtkMRMLScalarVolumeNode without file I/O."""
        import slicer  # type: ignore
        import vtk  # type: ignore

        img = slicer.util.arrayFromVolume(volumeNode)  # (Z, Y, X)
        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        affine = np.array(
            [[mat.GetElement(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
        # Compute true physical voxel spacing along I, J, K columns of IJKToRAS
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
        """Lazily load a DICOM series via SimpleITK or pydicom if requested outside Slicer."""
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
        # Convert LPS SimpleITK geometry to RAS
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

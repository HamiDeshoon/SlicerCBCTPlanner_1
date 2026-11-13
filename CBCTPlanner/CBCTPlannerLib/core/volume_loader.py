"""
CBCT Volume Loader for DICOM series and NIfTI volumes.
Compliant with medical imaging directives: SimpleITK primary loader, pydicom for headers,
warning suppression for anonymized pseudo-UIDs, and millimetric coordinate transforms.
"""

import os
import glob
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom

try:
    import SimpleITK as sitk
    HAS_SITK = True
except (ImportError, Exception):
    sitk = None
    HAS_SITK = False

import nibabel as nib

# Suppress pydicom warnings for non-standard/anonymized UIDs per medical directives
logging.getLogger("pydicom").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


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
    """
    Encapsulates a 3D Dental CBCT volume with spatial coordinates,
    physical spacing (mm), affine orientation, and Hounsfield Unit (HU) intensities.
    """

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
        """
        Args:
            image: 3D array of shape (Z, Y, X) or (D, H, W) in HU
            spacing: (spacing_z, spacing_y, spacing_x) in millimeters
            origin: (origin_x, origin_y, origin_z) in world mm
            direction: 3x3 direction cosine matrix
            affine: 4x4 affine matrix from voxel index to world coordinate
            metadata: VolumeMetadata instance
            sitk_image: underlying SimpleITK Image object (optional)
        """
        self.image = image.astype(np.float32)
        self.spacing = tuple(float(s) for s in spacing)  # (sz, sy, sx)
        self.origin = tuple(float(o) for o in origin)
        self.direction = np.asarray(direction, dtype=np.float64).reshape((3, 3))
        self.affine = np.asarray(affine, dtype=np.float64)
        self.metadata = metadata or VolumeMetadata()
        self._sitk_image = sitk_image

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape as (Z, Y, X)."""
        return self.image.shape

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        """Volume shape as (X, Y, Z)."""
        return (self.image.shape[2], self.image.shape[1], self.image.shape[0])

    def world_to_voxel(self, world_coord_mm: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        """
        Convert physical world coordinate (x_mm, y_mm, z_mm) to voxel index (x, y, z).
        Note: To index into self.image[z, y, x], use int(v[2]), int(v[1]), int(v[0]).
        """
        pt = np.asarray(world_coord_mm, dtype=np.float64).reshape(-1)
        if len(pt) == 3:
            pt_homog = np.array([pt[0], pt[1], pt[2], 1.0])
        else:
            pt_homog = pt
        inv_affine = np.linalg.inv(self.affine)
        vox_homog = inv_affine @ pt_homog
        return vox_homog[:3]

    def voxel_to_world(self, voxel_index: Union[np.ndarray, List[float], Tuple[float, ...]]) -> np.ndarray:
        """
        Convert voxel index (x, y, z) to physical world coordinate (x_mm, y_mm, z_mm).
        """
        idx = np.asarray(voxel_index, dtype=np.float64).reshape(-1)
        if len(idx) == 3:
            idx_homog = np.array([idx[0], idx[1], idx[2], 1.0])
        else:
            idx_homog = idx
        world_homog = self.affine @ idx_homog
        return world_homog[:3]

    def get_hu_at_point(self, world_coord_mm: Union[np.ndarray, List[float], Tuple[float, ...]]) -> float:
        """Sample Hounsfield Unit at continuous physical coordinate (trilinear or nearest)."""
        vox = self.world_to_voxel(world_coord_mm)
        vx, vy, vz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        if 0 <= vz < self.shape[0] and 0 <= vy < self.shape[1] and 0 <= vx < self.shape[2]:
            return float(self.image[vz, vy, vx])
        return -1000.0  # Air default outside volume

    def to_sitk(self):
        """Return or create a SimpleITK Image object with correct spacing and origin."""
        if not HAS_SITK:
            return None
        if self._sitk_image is not None:
            return self._sitk_image
        img = sitk.GetImageFromArray(self.image)
        img.SetSpacing((self.spacing[2], self.spacing[1], self.spacing[0]))  # (sx, sy, sz)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction.flatten().tolist())
        return img

    @classmethod
    def from_dicom_directory(cls, folder_path: Union[str, Path]) -> "CBCTVolume":
        """
        Load a DICOM series from directory. Recursively searches for DICOM files,
        uses SimpleITK if available, otherwise falls back to pydicom slice stack.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"DICOM folder not found: {folder}")

        # Recursively search for .dcm, .DCM, or any non-directory files
        dcm_candidates = []
        for root, _, files in os.walk(str(folder)):
            for f in files:
                p = os.path.join(root, f)
                if f.lower().endswith((".dcm", ".ima")) or not "." in f:
                    dcm_candidates.append(p)
        
        # If no specific DICOM extension found, try all files
        if not dcm_candidates:
            for root, _, files in os.walk(str(folder)):
                for f in files:
                    dcm_candidates.append(os.path.join(root, f))

        dcm_files = [f for f in dcm_candidates if os.path.isfile(f) and not f.endswith((".txt", ".json", ".xml", ".zip"))]
        if not dcm_files:
            raise ValueError(f"No DICOM files found in {folder}")

        if HAS_SITK:
            try:
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(str(folder))
                if series_ids:
                    # Select the series with the most slices (avoids choosing scout/localizer views)
                    chosen_series = max(series_ids, key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(folder), sid)))
                    file_names = reader.GetGDCMSeriesFileNames(str(folder), chosen_series)
                else:
                    file_names = dcm_files
                reader.SetFileNames(file_names)
                sitk_img = reader.Execute()
                try:
                    sitk_img = sitk.DICOMOrient(sitk_img, "LPS")
                except Exception as orient_err:
                    logging.debug(f"DICOMOrient LPS skipped on series: {orient_err}")
                arr = sitk.GetArrayFromImage(sitk_img)
                sp_sitk = sitk_img.GetSpacing()
                spacing = (sp_sitk[2], sp_sitk[1], sp_sitk[0])
                origin = sitk_img.GetOrigin()
                direction = np.array(sitk_img.GetDirection()).reshape((3, 3))
                affine = np.eye(4)
                affine[:3, :3] = direction @ np.diag([sp_sitk[0], sp_sitk[1], sp_sitk[2]])
                affine[:3, 3] = origin
            except Exception as sitk_err:
                logging.warning(f"SimpleITK series read failed ({sitk_err}), falling back to pydicom")
                HAS_SITK_FAILED = True
            else:
                HAS_SITK_FAILED = False
        else:
            HAS_SITK_FAILED = True

        if HAS_SITK_FAILED or not HAS_SITK:
            # Resilient pydicom fallback
            slices = []
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(f)
                    if hasattr(ds, "pixel_array"):
                        slices.append(ds)
                except Exception:
                    continue
            if not slices:
                raise ValueError("No readable DICOM image slices found")
            
            # Sort by InstanceNumber or ImagePositionPatient
            if hasattr(slices[0], "ImagePositionPatient"):
                slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            elif hasattr(slices[0], "InstanceNumber"):
                slices.sort(key=lambda s: int(s.InstanceNumber))

            # Stack 2D slices to 3D array (Z, Y, X)
            arr = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
            # Apply RescaleSlope and RescaleIntercept for HU
            first = slices[0]
            slope = float(getattr(first, "RescaleSlope", 1.0))
            intercept = float(getattr(first, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            # Spacing
            sx = float(first.PixelSpacing[1]) if hasattr(first, "PixelSpacing") else 0.4
            sy = float(first.PixelSpacing[0]) if hasattr(first, "PixelSpacing") else 0.4
            if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[1], "ImagePositionPatient"):
                sz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
            else:
                sz = float(getattr(first, "SliceThickness", 0.4))
            spacing = (sz, sy, sx)

            pos = getattr(first, "ImagePositionPatient", [0.0, 0.0, 0.0])
            origin = (float(pos[0]), float(pos[1]), float(pos[2]))
            direction = np.eye(3)
            affine = np.diag([sx, sy, sz, 1.0])
            affine[:3, 3] = origin
            sitk_img = None

        meta = VolumeMetadata()
        try:
            first_dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True, force=True)
            meta.patient_id = str(getattr(first_dcm, "PatientID", "Anonymous"))
            meta.patient_name = str(getattr(first_dcm, "PatientName", "Anonymous"))
            meta.study_date = str(getattr(first_dcm, "StudyDate", ""))
            meta.series_description = str(getattr(first_dcm, "SeriesDescription", "CBCT Scan"))
            meta.modality = str(getattr(first_dcm, "Modality", "CT"))
            meta.kvp = getattr(first_dcm, "KVP", None)
            meta.tube_current_ma = getattr(first_dcm, "XRayTubeCurrent", None)
            meta.slice_thickness_mm = float(getattr(first_dcm, "SliceThickness", spacing[0]))
            if hasattr(first_dcm, "PixelSpacing"):
                meta.pixel_spacing_mm = (float(first_dcm.PixelSpacing[0]), float(first_dcm.PixelSpacing[1]))
            if hasattr(first_dcm, "WindowCenter"):
                wc = first_dcm.WindowCenter
                meta.window_center = float(wc[0] if isinstance(wc, (list, pydicom.multival.MultiValue)) else wc)
            if hasattr(first_dcm, "WindowWidth"):
                ww = first_dcm.WindowWidth
                meta.window_width = float(ww[0] if isinstance(ww, (list, pydicom.multival.MultiValue)) else ww)
        except Exception as e:
            logging.warning(f"Could not parse DICOM header details: {e}")

        return cls(
            image=arr,
            spacing=spacing,
            origin=origin,
            direction=direction,
            affine=affine,
            metadata=meta,
            sitk_image=sitk_img,
        )

    @classmethod
    def from_nifti(cls, file_path: Union[str, Path]) -> "CBCTVolume":
        """Load NIfTI volume (.nii or .nii.gz) via nibabel or SimpleITK."""
        path = str(file_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NIfTI file not found: {path}")

        if HAS_SITK:
            sitk_img = sitk.ReadImage(path)
            try:
                sitk_img = sitk.DICOMOrient(sitk_img, "LPS")
            except Exception as orient_err:
                logging.debug(f"DICOMOrient LPS skipped on NIfTI: {orient_err}")
            arr = sitk.GetArrayFromImage(sitk_img)
            sp_sitk = sitk_img.GetSpacing()  # (sx, sy, sz)
            spacing = (sp_sitk[2], sp_sitk[1], sp_sitk[0])
            origin = sitk_img.GetOrigin()
            direction = np.array(sitk_img.GetDirection()).reshape((3, 3))
            affine = np.eye(4)
            affine[:3, :3] = direction @ np.diag([sp_sitk[0], sp_sitk[1], sp_sitk[2]])
            affine[:3, 3] = origin
        else:
            # nibabel load
            nii = nib.load(path)
            affine = nii.affine
            zooms = nii.header.get_zooms()
            # nibabel shape is (X, Y, Z) -> transpose to (Z, Y, X) for consistent array indexing
            arr_xyz = np.asanyarray(nii.dataobj, dtype=np.float32)
            arr = np.transpose(arr_xyz, (2, 1, 0))
            spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
            origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
            direction = np.eye(3)
            sitk_img = None

        stem = Path(path).stem.replace(".nii", "")
        meta = VolumeMetadata(
            patient_id=stem,
            series_description="NIfTI CBCT Scan",
            slice_thickness_mm=spacing[0],
            pixel_spacing_mm=(spacing[1], spacing[2]),
        )

        return cls(
            image=arr,
            spacing=spacing,
            origin=origin,
            direction=direction,
            affine=affine,
            metadata=meta,
            sitk_image=sitk_img,
        )

    @classmethod
    def from_file_or_dir(cls, path: Union[str, Path]) -> "CBCTVolume":
        """Auto-detect DICOM directory, ZIP archive, or NIfTI file."""
        import zipfile
        import tempfile
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {p}")

        if p.is_dir():
            # Check if there are NIfTI files inside first
            nii_files = list(p.glob("*.nii")) + list(p.glob("*.nii.gz"))
            if nii_files:
                return cls.from_nifti(nii_files[0])
            return cls.from_dicom_directory(p)
        elif str(p).lower().endswith((".nii", ".nii.gz")):
            return cls.from_nifti(p)
        elif str(p).lower().endswith((".dcm", ".ima")):
            # If an individual DICOM file is passed, check if sibling slices exist in parent directory
            parent = p.parent
            siblings = [f for f in parent.iterdir() if f.is_file() and f.suffix.lower() in (".dcm", ".ima")]
            if len(siblings) > 1:
                return cls.from_dicom_directory(parent)
            # Standalone single file (could be 3D multi-frame or single slice)
            if HAS_SITK:
                try:
                    sitk_img = sitk.ReadImage(str(p))
                    arr = sitk.GetArrayFromImage(sitk_img)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    sp_sitk = sitk_img.GetSpacing()
                    spacing = (sp_sitk[2] if len(sp_sitk) > 2 else 0.4, sp_sitk[1], sp_sitk[0])
                    origin = sitk_img.GetOrigin()
                    if len(origin) < 3:
                        origin = (origin[0], origin[1], 0.0)
                    direction = np.array(sitk_img.GetDirection())
                    if direction.size == 9:
                        direction = direction.reshape((3, 3))
                    else:
                        direction = np.eye(3)
                    affine = np.eye(4)
                    affine[:3, :3] = direction @ np.diag([spacing[2], spacing[1], spacing[0]])
                    affine[:3, 3] = origin
                    return cls(
                        image=arr.astype(np.float32),
                        spacing=spacing,
                        origin=origin,
                        direction=direction,
                        affine=affine,
                        metadata=VolumeMetadata(patient_id=p.stem, series_description="Single DICOM"),
                        sitk_image=sitk_img,
                    )
                except Exception:
                    pass
            return cls.from_dicom_directory(parent)
        elif str(p).lower().endswith(".zip"):
            extract_dir = tempfile.mkdtemp(prefix="cbct_unzip_")
            try:
                with zipfile.ZipFile(str(p), "r") as zf:
                    for member in zf.infolist():
                        target = Path(extract_dir) / member.filename
                        if not target.resolve().is_relative_to(Path(extract_dir).resolve()):
                            raise ValueError(f"Unsafe path detected in ZIP: {member.filename}")
                    zf.extractall(extract_dir)
                return cls.from_file_or_dir(extract_dir)
            finally:
                # Let from_file_or_dir load from extract_dir before removal or retain if needed
                pass
        else:
            raise ValueError(f"Unsupported input path: {p}")

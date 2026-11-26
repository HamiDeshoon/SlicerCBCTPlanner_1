"""Unified SlicerAutomatedDentalTools-compatible native package for CBCTPlanner (`adt`).

Consolidates upgraded ALI, ASO, CLI-C, DOCShapeAXI, VFACE, AutoCrop3D, and AutoMatrix
algorithms with C-level vectorization (`vtk`, `skimage`, `scipy`, `torch`/`torchvision` optional)
and complete offline/in-Slicer fallbacks without requiring external WSL2/Conda installs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..core.arch_detector import DentalArch
from ..core.volume_loader import CBCTVolume
from .ali import (
    ALILandmarkResult,
    export_ali_landmarks_mrk_json,
    extract_jaw_arch_anchor_bounds,
    identify_ali_landmarks,
    load_ali_landmarks_mrk_json,
)
from .aso import ASOOrientationResult, compute_aso_standardized_orientation
from .clic import (
    CLIC_CLASS_CODES,
    CLIC_CLASS_NAMES,
    ImpactedToothCLICResult,
    analyze_impacted_teeth_clic,
    run_clic_maskrcnn_inference,
)
from .doc_shapeaxi import DOCShapeAXIResult, run_doc_shapeaxi
from .geometry import (
    compute_vertex_curvature_saliency,
    extract_isosurface_from_label,
    filter_outlier_landmarks,
    kabsch_rigid_alignment,
    mesh_volume_and_area,
    read_surface,
    vectorized_median_filter_surface,
    vtk_icp_refine,
    vtk_matrix_to_numpy,
    write_surface_with_scalars,
)
from .vface import (
    VFACEAsymmetryResult,
    build_sagittal_mirror_matrix_4x4,
    compute_autocrop_roi_bounds_ras,
    compute_vface_asymmetry_analysis,
)


class AutomatedDentalToolsEngine:
    """Facade wrapping upgraded ALI, ASO, CLI-C, DOCShapeAXI, VFACE, and AutoCrop3D engines."""

    def __init__(self) -> None:
        self.last_ali_landmarks: Dict[str, ALILandmarkResult] = {}
        self.last_aso_result: Optional[ASOOrientationResult] = None
        self.last_clic_results: List[ImpactedToothCLICResult] = []
        self.last_clic_labelmap: Optional[np.ndarray] = None
        self.last_doc_shapeaxi_result: Optional[DOCShapeAXIResult] = None
        self.last_vface_result: Optional[VFACEAsymmetryResult] = None

    def identify_ali_landmarks(
        self,
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        include_per_tooth_landmarks: bool = True,
    ) -> Dict[str, ALILandmarkResult]:
        landmarks = identify_ali_landmarks(
            volume,
            mask=mask,
            is_5class=is_5class,
            include_per_tooth_landmarks=include_per_tooth_landmarks,
        )
        self.last_ali_landmarks = landmarks
        return landmarks

    @staticmethod
    def export_ali_landmarks_mrk_json(
        landmarks: Dict[str, ALILandmarkResult],
        out_file: Path,
    ) -> Path:
        return export_ali_landmarks_mrk_json(landmarks, out_file)

    @staticmethod
    def load_ali_landmarks_mrk_json(
        mrk_file: Path,
        allowed_labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, ALILandmarkResult]:
        return load_ali_landmarks_mrk_json(mrk_file, allowed_labels=allowed_labels)

    def compute_aso_standardized_orientation(
        self,
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        landmarks: Optional[Dict[str, ALILandmarkResult]] = None,
        reference_landmarks_ras: Optional[Dict[str, Sequence[float]]] = None,
        use_icp_refinement: bool = True,
    ) -> ASOOrientationResult:
        res = compute_aso_standardized_orientation(
            volume,
            mask=mask,
            is_5class=is_5class,
            landmarks=landmarks or (self.last_ali_landmarks or None),
            reference_landmarks_ras=reference_landmarks_ras,
            use_icp_refinement=use_icp_refinement,
        )
        self.last_aso_result = res
        return res

    def analyze_impacted_teeth_clic(
        self,
        volume: CBCTVolume,
        mask: np.ndarray,
        is_5class: bool = False,
        target_fdis: Sequence[str] = ("13", "23", "33", "43", "18", "28", "38", "48"),
        arch_lower: Optional[DentalArch] = None,
        arch_upper: Optional[DentalArch] = None,
        build_labelmap: bool = False,
    ) -> List[ImpactedToothCLICResult]:
        results, labelmap = analyze_impacted_teeth_clic(
            volume,
            mask,
            is_5class=is_5class,
            target_fdis=target_fdis,
            arch_lower=arch_lower,
            arch_upper=arch_upper,
            build_labelmap=build_labelmap,
        )
        self.last_clic_results = results
        self.last_clic_labelmap = labelmap
        return results

    def run_doc_shapeaxi(
        self,
        *,
        data_type: str = "Airway",
        task: str = "severity",
        surface_path: Optional[Path] = None,
        volume: Optional[CBCTVolume] = None,
        mask: Optional[np.ndarray] = None,
        out_vtk_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> DOCShapeAXIResult:
        res = run_doc_shapeaxi(
            data_type=data_type,
            task=task,
            surface_path=surface_path,
            volume=volume,
            mask=mask,
            out_vtk_path=out_vtk_path,
            checkpoint_path=checkpoint_path,
        )
        self.last_doc_shapeaxi_result = res
        return res

    def compute_vface_asymmetry_analysis(
        self,
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        is_5class: bool = False,
        landmarks: Optional[Dict[str, ALILandmarkResult]] = None,
    ) -> VFACEAsymmetryResult:
        res = compute_vface_asymmetry_analysis(
            volume,
            mask=mask,
            is_5class=is_5class,
            landmarks=landmarks or (self.last_ali_landmarks or None),
        )
        self.last_vface_result = res
        return res

    @staticmethod
    def compute_autocrop_roi_bounds_ras(
        volume: CBCTVolume,
        mask: Optional[np.ndarray] = None,
        padding_mm: float = 10.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_autocrop_roi_bounds_ras(volume, mask=mask, padding_mm=padding_mm)

    @staticmethod
    def build_sagittal_mirror_matrix_4x4(midline_x_ras: float = 0.0) -> np.ndarray:
        return build_sagittal_mirror_matrix_4x4(midline_x_ras=midline_x_ras)


__all__ = [
    "ALILandmarkResult",
    "ASOOrientationResult",
    "AutomatedDentalToolsEngine",
    "CLIC_CLASS_CODES",
    "CLIC_CLASS_NAMES",
    "DOCShapeAXIResult",
    "ImpactedToothCLICResult",
    "VFACEAsymmetryResult",
    "analyze_impacted_teeth_clic",
    "build_sagittal_mirror_matrix_4x4",
    "compute_aso_standardized_orientation",
    "compute_autocrop_roi_bounds_ras",
    "compute_vertex_curvature_saliency",
    "compute_vface_asymmetry_analysis",
    "export_ali_landmarks_mrk_json",
    "extract_isosurface_from_label",
    "extract_jaw_arch_anchor_bounds",
    "filter_outlier_landmarks",
    "identify_ali_landmarks",
    "kabsch_rigid_alignment",
    "load_ali_landmarks_mrk_json",
    "mesh_volume_and_area",
    "read_surface",
    "run_clic_maskrcnn_inference",
    "run_doc_shapeaxi",
    "vectorized_median_filter_surface",
    "vtk_icp_refine",
    "vtk_matrix_to_numpy",
    "write_surface_with_scalars",
]

"""Lossless RAS orientation handling for CBCT volumes and label masks.

Pure NumPy: no Slicer/VTK imports, so the geometry is unit-testable headlessly.
Arrays are ``(Z, Y, X)`` and the affine is a 4x4 IJK->RAS matrix where the IJK
columns are ``(x-index, y-index, z-index)``.

RAS convention: +X = Patient Right, +Y = Anterior, +Z = Superior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_RAS_AXIS_CODES = ("R", "A", "S")
_NEGATIVE_CODES = {"R": "L", "A": "P", "S": "I"}


def axcodes_from_affine(affine: np.ndarray) -> Tuple[str, str, str]:
    """Return the anatomical codes for the IJK axes, e.g. ``('L', 'P', 'S')``."""
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got {matrix.shape}")
    codes: List[str] = []
    for axis in range(3):
        direction = matrix[:3, axis]
        dominant = int(np.argmax(np.abs(direction)))
        sign = 1.0 if direction[dominant] >= 0 else -1.0
        code = _RAS_AXIS_CODES[dominant]
        codes.append(code if sign > 0 else _NEGATIVE_CODES[code])
    return codes[0], codes[1], codes[2]


def _obliquity_deg(affine: np.ndarray) -> float:
    matrix = np.asarray(affine, dtype=np.float64)
    max_angle = 0.0
    for axis in range(3):
        direction = matrix[:3, axis]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        cosine = float(np.max(np.abs(direction)) / norm)
        max_angle = max(max_angle, float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return max_angle


@dataclass(frozen=True)
class ArrayOps:
    """Permutation + per-output-axis flips applied to an ``(Z, Y, X)`` array.

    ``perm[out_axis]`` is the source array axis that becomes ``out_axis`` in the
    result (output order is Z, Y, X). ``flip[out_axis]`` reverses that axis.
    """

    perm: Tuple[int, int, int]
    flip: Tuple[bool, bool, bool]


@dataclass
class CanonResult:
    image_zyx: np.ndarray
    affine: np.ndarray
    ops: ArrayOps
    max_obliquity_deg: float
    is_oblique: bool


@dataclass
class OrientationReport:
    si_ok: Optional[bool] = None
    ap_ok: Optional[bool] = None
    lr_ok: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(value is not False for value in (self.si_ok, self.ap_ok, self.lr_ok))


def apply_ops(arr_zyx: np.ndarray, ops: ArrayOps) -> np.ndarray:
    """Apply the same permutation/flips used for an image to a label array."""
    data = np.asarray(arr_zyx)
    out = np.transpose(data, ops.perm)
    for axis, should_flip in enumerate(ops.flip):
        if should_flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def _axis_permutation_and_flips(affine: np.ndarray) -> ArrayOps:
    matrix = np.asarray(affine, dtype=np.float64)
    # Output array axes (Z, Y, X) must map to anatomical (S, A, R).
    target_vectors = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    perm: List[int] = []
    flips: List[bool] = []
    for target in target_vectors:
        scores = [abs(float(np.dot(matrix[:3, axis], target))) for axis in range(3)]
        ijk_axis = int(np.argmax(scores))
        component = float(np.dot(matrix[:3, ijk_axis], target))
        # array axis for IJK axis: i(0)->2, j(1)->1, k(2)->0
        perm.append(2 - ijk_axis)
        flips.append(component < 0.0)
    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"affine is degenerate; cannot determine axis permutation: {affine}")
    return ArrayOps(perm=tuple(perm), flip=tuple(flips))


def _canonical_affine_with_shape(
    affine: np.ndarray, ops: ArrayOps, shape_zyx: Tuple[int, int, int]
) -> np.ndarray:
    matrix = np.asarray(affine, dtype=np.float64)
    source_columns = [matrix[:3, 2 - ops.perm[out_axis]] for out_axis in range(3)]
    signs = [-1.0 if ops.flip[out_axis] else 1.0 for out_axis in range(3)]
    output_shape = tuple(int(shape_zyx[ops.perm[out_axis]]) for out_axis in range(3))
    translation = matrix[:3, 3].copy()
    # Output axis 0 (k/i/j) is written to IJK column 2, 1, 0 respectively.
    for ijk_axis, out_axis in enumerate((2, 1, 0)):
        if ops.flip[out_axis]:
            extent = int(output_shape[out_axis]) - 1
            translation = translation + source_columns[out_axis] * float(extent)
    new_matrix = np.eye(4, dtype=np.float64)
    for ijk_axis, out_axis in enumerate((2, 1, 0)):
        new_matrix[:3, ijk_axis] = signs[out_axis] * source_columns[out_axis]
    new_matrix[:3, 3] = translation
    return new_matrix


def canonicalize_to_ras(
    image_zyx: np.ndarray,
    affine: np.ndarray,
    *,
    max_oblique_deg: float = 5.0,
) -> CanonResult:
    """Permute/flip (never resample) so the result has axcodes ``('R', 'A', 'S')``."""
    image = np.asarray(image_zyx)
    matrix = np.asarray(affine, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image_zyx must be 3D (Z, Y, X), got shape {image.shape}")
    if matrix.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got {matrix.shape}")
    obliquity = _obliquity_deg(matrix)
    ops = _axis_permutation_and_flips(matrix)
    out = apply_ops(image, ops)
    new_affine = _canonical_affine_with_shape(matrix, ops, image.shape)
    return CanonResult(
        image_zyx=out,
        affine=new_affine,
        ops=ops,
        max_obliquity_deg=obliquity,
        is_oblique=obliquity > float(max_oblique_deg),
    )


def flip_axis(image_zyx: np.ndarray, affine: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the array along an anatomical axis and update the affine.

    ``axis`` is one of ``"LR"``, ``"AP"``, ``"SI"``. The operation is lossless
    (no resampling), keeps the volume inside the same world bounding box, and is
    idempotent: applying it twice restores the original array and affine.
    """
    key = str(axis).upper()
    if key not in ("LR", "AP", "SI"):
        raise ValueError(f"axis must be 'LR', 'AP' or 'SI', got {axis!r}")
    image = np.asarray(image_zyx)
    matrix = np.asarray(affine, dtype=np.float64)
    if image.ndim != 3:
        raise ValueError(f"image_zyx must be 3D (Z, Y, X), got shape {image.shape}")
    target = {"LR": 0, "AP": 1, "SI": 2}[key]
    scores = [abs(float(matrix[target, axis_index])) for axis_index in range(3)]
    ijk_axis = int(np.argmax(scores))
    array_axis = 2 - ijk_axis
    out = np.ascontiguousarray(np.flip(image, axis=array_axis))
    new_affine = matrix.copy()
    column = matrix[:3, ijk_axis].copy()
    extent = float(image.shape[array_axis] - 1)
    new_affine[:3, ijk_axis] = -column
    new_affine[:3, 3] = matrix[:3, 3] + column * extent
    return out, new_affine


def _world_centroid(voxel_indices_zyx: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    if len(voxel_indices_zyx) == 0:
        return None
    z_idx, y_idx, x_idx = voxel_indices_zyx
    mean = np.array([x_idx.mean(), y_idx.mean(), z_idx.mean(), 1.0], dtype=np.float64)
    return (np.asarray(affine, dtype=np.float64) @ mean)[:3]


def _mask_centroid(mask: np.ndarray, affine: np.ndarray, labels: Tuple[int, ...]) -> Optional[np.ndarray]:
    indices = np.where(np.isin(mask, labels))
    return _world_centroid(indices, affine)


def _tooth_cloud(mask: np.ndarray, affine: np.ndarray, labels: Tuple[int, ...]) -> np.ndarray:
    z_idx, y_idx, x_idx = np.where(np.isin(mask, labels))
    if len(x_idx) == 0:
        return np.empty((0, 3), dtype=np.float64)
    voxels = np.column_stack([x_idx, y_idx, z_idx, np.ones(len(x_idx))]).astype(np.float64)
    return (np.asarray(affine, dtype=np.float64) @ voxels.T).T[:, :3]


def _ap_orientation(points_xy: np.ndarray) -> Optional[bool]:
    """True when the arch is convex toward +Y (anterior at +Y)."""
    if len(points_xy) < 20:
        return None
    centroid = points_xy.mean(axis=0)
    centered = points_xy - centroid
    extent_x = float(np.percentile(np.abs(centered[:, 0]), 95))
    extent_y = float(np.percentile(np.abs(centered[:, 1]), 95))
    if extent_x < 6.0 or extent_y < 2.5:
        return None
    central = np.abs(centered[:, 0]) <= max(4.0, 0.35 * extent_x)
    posterior = np.abs(centered[:, 0]) >= max(6.0, 0.75 * extent_x)
    if not central.any() or not posterior.any():
        return None
    return bool(np.median(centered[central, 1]) > np.median(centered[posterior, 1]))


def verify_orientation(
    mask_zyx: np.ndarray,
    affine: np.ndarray,
    *,
    is_5class: bool,
) -> OrientationReport:
    """Check S-I, A-P and (when FDI labels exist) L-R orientation of a label mask."""
    mask = np.asarray(mask_zyx)
    report = OrientationReport()
    if mask.ndim != 3:
        report.notes.append(f"mask must be 3D, got shape {mask.shape}")
        return report

    if is_5class:
        upper_teeth = (3,)
        lower_teeth = (4,)
        upper_jaw = (1,)
        lower_jaw = (2,)
    else:
        from .labels import MANDIBULAR_TEETH_IDS, MAXILLARY_TEETH_IDS, UPPER_JAW_ID, LOWER_JAW_ID

        upper_teeth = tuple(MAXILLARY_TEETH_IDS)
        lower_teeth = tuple(MANDIBULAR_TEETH_IDS)
        upper_jaw = (UPPER_JAW_ID,)
        lower_jaw = (LOWER_JAW_ID,)

    lower_z = _mask_centroid(mask, affine, lower_teeth + lower_jaw)
    upper_z = _mask_centroid(mask, affine, upper_teeth + upper_jaw)
    if lower_z is not None and upper_z is not None:
        report.si_ok = bool(lower_z[2] < upper_z[2])
        if not report.si_ok:
            report.notes.append("Mandible/lower teeth are superior to maxilla/upper teeth (S-I inverted).")
    else:
        report.notes.append("S-I: missing upper or lower jaw/teeth labels; cannot determine.")

    cloud = _tooth_cloud(mask, affine, upper_teeth + lower_teeth)
    if len(cloud) == 0:
        report.notes.append("A-P: no tooth labels present; cannot determine.")
    else:
        report.ap_ok = _ap_orientation(cloud[:, :2])
        if report.ap_ok is None:
            report.notes.append("A-P: tooth cloud too small or degenerate; cannot determine.")
        elif not report.ap_ok:
            report.notes.append("Dental arch is convex toward -Y (A-P inverted).")

    if is_5class:
        report.notes.append("L-R: 5-class labels carry no FDI side information; cannot determine.")
    else:
        right = _mask_centroid(mask, affine, tuple(range(11, 19)) + tuple(range(41, 49)))
        left = _mask_centroid(mask, affine, tuple(range(21, 29)) + tuple(range(31, 39)))
        if right is not None and left is not None:
            report.lr_ok = bool(right[0] > left[0])
            if not report.lr_ok:
                report.notes.append("FDI right-side teeth are on -X (L-R inverted).")
        else:
            report.notes.append("L-R: FDI-labelled teeth missing; cannot determine.")

    if not report.notes:
        report.notes.append("Orientation is consistent with RAS.")
    return report

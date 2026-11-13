"""
Dental Arch Curve Detection and Curved Planar Reformation (CPR) Frame Computation.
All optional dependencies (scipy, scikit-image) are loaded lazily on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

from . import ensure_optional_package
from .labels import (
    LOWER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    UPPER_JAW_ID,
)
from .mask_centerline import CenterlineCurve

# FDI sequence from patient right posterior to patient left posterior for each jaw.
JAW_TOOTH_ORDER = {
    "lower": [
        "48", "47", "46", "45", "44", "43", "42", "41",
        "31", "32", "33", "34", "35", "36", "37", "38",
    ],
    "upper": [
        "18", "17", "16", "15", "14", "13", "12", "11",
        "21", "22", "23", "24", "25", "26", "27", "28",
    ],
}


def _prune_pathological_points(points: np.ndarray, max_deviation_mm: float = 8.0) -> np.ndarray:
    """Drop isolated control points that jump off the local path."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        return pts
    for _ in range(4):
        if len(pts) < 5:
            break
        worst_idx = -1
        worst_deviation = max_deviation_mm
        for j in range(1, len(pts) - 1):
            midpoint = 0.5 * (pts[j - 1] + pts[j + 1])
            deviation = float(np.linalg.norm(pts[j] - midpoint))
            if deviation > worst_deviation:
                worst_deviation = deviation
                worst_idx = j
        if worst_idx < 0:
            break
        pts = np.delete(pts, worst_idx, axis=0)
    return pts


def _order_points_around_arch(pts_xy: np.ndarray) -> np.ndarray:
    """Order 2D points along an open horseshoe from one posterior end to the other."""
    centroid = pts_xy.mean(axis=0)
    angles = np.arctan2(pts_xy[:, 1] - centroid[1], pts_xy[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 2.0 * np.pi]]))
    cut = int(np.argmax(gaps))
    return np.roll(order, -(cut + 1))


@dataclass
class ArchFrame:
    """Orthonormal frame at a specific point along the dental arch curve."""

    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray
    t: float
    arc_length_mm: float


class DentalArch:
    """Parametric 3D Dental Arch Spline with RAS Right-to-Left (+X to -X) orientation."""

    def __init__(
        self,
        control_points: np.ndarray,
        jaw_type: str = "lower",
        smoothing: float = 0.0,
        source: str = "manual",
        occlusal_normal: Optional[Sequence[float]] = None,
        ras_convention: bool = True,
    ):
        pts = np.asarray(control_points, dtype=np.float64)
        if len(pts) < 4:
            raise ValueError(f"Need at least 4 control points to fit a spline, got {len(pts)}")

        # In 3D Slicer RAS: +X is Patient Right, -X is Patient Left.
        # Orient curve from Patient Right (+X) -> Patient Left (-X) when ras_convention=True.
        if ras_convention and pts[0, 0] < pts[-1, 0]:
            pts = pts[::-1].copy()
        elif (not ras_convention) and pts[0, 0] > pts[-1, 0]:
            pts = pts[::-1].copy()

        self.control_points = pts
        self.jaw_type = jaw_type
        self.source = source
        self.ras_convention = ras_convention

        up_vec = np.asarray(occlusal_normal if occlusal_normal is not None else (0.0, 0.0, 1.0), dtype=np.float64)
        up_norm = float(np.linalg.norm(up_vec))
        self.occlusal_normal = up_vec / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if self.occlusal_normal[2] < 0:
            self.occlusal_normal = -self.occlusal_normal

        diffs = np.diff(self.control_points, axis=0)
        chord_dists = np.sqrt((diffs**2).sum(axis=1))
        u = np.zeros(len(self.control_points), dtype=np.float64)
        u[1:] = np.cumsum(chord_dists)
        total_chord = float(u[-1])
        if total_chord <= 1e-6:
            raise ValueError("Control points are degenerate (zero distance)")
        u_norm = u / total_chord

        self._use_scipy = False
        self.tck = None
        try:
            from scipy import interpolate  # type: ignore

            k = min(3, len(self.control_points) - 1)
            self.tck, _ = interpolate.splprep(
                [self.control_points[:, 0], self.control_points[:, 1], self.control_points[:, 2]],
                u=u_norm,
                s=smoothing,
                k=k,
            )
            self._use_scipy = True
        except ImportError:
            self._u_ctrl = u_norm

        dense_samples = 500
        u_dense = np.linspace(0.0, 1.0, dense_samples)
        pts_dense = self._eval_raw(u_dense)
        dense_diffs = np.diff(pts_dense, axis=0)
        seg_lengths = np.sqrt((dense_diffs**2).sum(axis=1))
        self.cum_arc_lengths = np.zeros(dense_samples, dtype=np.float64)
        self.cum_arc_lengths[1:] = np.cumsum(seg_lengths)
        self.total_arc_length = float(self.cum_arc_lengths[-1])
        self.u_dense = u_dense
        self.pts_dense = pts_dense
        self.t_dense = self.cum_arc_lengths / max(self.total_arc_length, 1e-9)

    def _eval_raw(self, u_vals: np.ndarray) -> np.ndarray:
        u_arr = np.atleast_1d(np.asarray(u_vals, dtype=np.float64))
        if self._use_scipy and self.tck is not None:
            from scipy import interpolate  # type: ignore

            return np.column_stack(interpolate.splev(u_arr, self.tck))
        return np.column_stack(
            [np.interp(u_arr, self._u_ctrl, self.control_points[:, d]) for d in range(3)]
        )

    def _eval_deriv_raw(self, u_val: float) -> np.ndarray:
        if self._use_scipy and self.tck is not None:
            from scipy import interpolate  # type: ignore

            return np.array(interpolate.splev(u_val, self.tck, der=1), dtype=np.float64)
        eps = 1e-3
        u0 = max(0.0, u_val - eps)
        u1 = min(1.0, u_val + eps)
        pts = self._eval_raw(np.array([u0, u1]))
        return (pts[1] - pts[0]) / max(1e-6, u1 - u0)

    def _arc_to_u(self, arc_mm: float) -> float:
        arc_clamped = float(np.clip(arc_mm, 0.0, self.total_arc_length))
        return float(np.interp(arc_clamped, self.cum_arc_lengths, self.u_dense))

    def evaluate_at(self, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, 1.0))
        return self.evaluate_at_mm(t * self.total_arc_length)

    def evaluate_at_mm(self, arc_mm: float) -> np.ndarray:
        u = self._arc_to_u(arc_mm)
        return self._eval_raw(np.array([u]))[0]

    def get_frame(self, t: float, occlusal_normal: Optional[Sequence[float]] = None) -> ArchFrame:
        """Compute local orthonormal frame (Point, Tangent, Normal=Buccal, Binormal=Superior)."""
        t = float(np.clip(t, 0.0, 1.0))
        arc_mm = t * self.total_arc_length
        return self.get_frame_at_mm(arc_mm, occlusal_normal=occlusal_normal)

    def get_frame_at_mm(
        self, arc_mm: float, occlusal_normal: Optional[Sequence[float]] = None
    ) -> ArchFrame:
        arc_clamped = float(np.clip(arc_mm, 0.0, self.total_arc_length))
        t = arc_clamped / max(1e-9, self.total_arc_length)
        u = self._arc_to_u(arc_clamped)

        pt = self._eval_raw(np.array([u]))[0]
        d_pt = self._eval_deriv_raw(u)

        up = (
            np.asarray(occlusal_normal, dtype=np.float64)
            if occlusal_normal is not None
            else self.occlusal_normal
        )
        up_norm = float(np.linalg.norm(up))
        up = up / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up[2] < 0:
            up = -up

        # Project tangent onto the occlusal plane so cross-sections stay perpendicular to occlusal plane
        t_proj = d_pt - np.dot(d_pt, up) * up
        t_norm = float(np.linalg.norm(t_proj))
        tangent = t_proj / t_norm if t_norm > 1e-8 else np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        # In RAS, for an arch running Patient Right (+X, posterior -Y) -> Anterior (+Y) -> Patient Left (-X, posterior -Y):
        # Tangent at anterior midline points toward Patient Left (-X, 0, 0).
        # T x Up = (-1, 0, 0) x (0, 0, 1) = (0, +1, 0) = +Y (Anterior / Labial / Buccal)!
        # At right molar (18/48), tangent points Anterior (+Y): (0, 1, 0) x (0, 0, 1) = (+1, 0, 0) = +X (Right Buccal)!
        # At left molar (28/38), tangent points Posterior (-Y): (0, -1, 0) x (0, 0, 1) = (-1, 0, 0) = -X (Left Buccal)!
        # Therefore N = T x Up ALWAYS points outward (Buccal/Labial) along a RAS Right->Left arch!
        normal = np.cross(tangent, up)
        n_norm = float(np.linalg.norm(normal))
        normal = normal / n_norm if n_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Verify outward buccal orientation against arch chord midpoint (posterior baseline)
        chord_mid = 0.5 * (self.control_points[0] + self.control_points[-1])
        outward_vec = pt - chord_mid
        outward_in_plane = outward_vec - np.dot(outward_vec, up) * up
        if np.linalg.norm(outward_in_plane) > 1e-3 and np.dot(normal, outward_in_plane) < 0:
            normal = -normal

        # Binormal = Up (orthonormalized as N x T so that T, N, B is a right-handed or upright frame with B = +Up)
        binormal = np.cross(normal, tangent)
        b_norm = float(np.linalg.norm(binormal))
        binormal = binormal / b_norm if b_norm > 1e-8 else up
        if np.dot(binormal, up) < 0:
            binormal = -binormal

        return ArchFrame(
            point=pt,
            tangent=tangent,
            normal=normal,
            binormal=binormal,
            t=t,
            arc_length_mm=arc_clamped,
        )

    def sample_frames_uniform(self, step_mm: float = 1.0) -> List[ArchFrame]:
        n_steps = max(2, int(np.round(self.total_arc_length / max(0.1, step_mm))))
        return [self.get_frame(i / float(n_steps)) for i in range(n_steps + 1)]

    def find_closest_parameter(self, point_mm: np.ndarray) -> Tuple[float, float]:
        pt = np.asarray(point_mm, dtype=np.float64).reshape(3)
        dists_sq = np.sum((self.pts_dense - pt) ** 2, axis=1)
        best_idx = int(np.argmin(dists_sq))
        best_t = float(self.t_dense[best_idx])

        dt = 0.02
        fine_ts = np.linspace(max(0.0, best_t - dt), min(1.0, best_t + dt), 50)
        arc_mm = fine_ts * self.total_arc_length
        u = np.interp(arc_mm, self.cum_arc_lengths, self.u_dense)
        fine_pts = self._eval_raw(u)
        fine_dists_sq = np.sum((fine_pts - pt) ** 2, axis=1)
        fine_idx = int(np.argmin(fine_dists_sq))
        return float(fine_ts[fine_idx]), float(np.sqrt(fine_dists_sq[fine_idx]))

    @classmethod
    def from_teeth_centroids(
        cls,
        teeth_centroids: List[Tuple[str, np.ndarray]],
        jaw_type: str = "lower",
        ras_convention: bool = True,
    ) -> "DentalArch":
        fdi_dict = {str(fdi): np.asarray(pt, dtype=np.float64) for fdi, pt in teeth_centroids}
        desired_order = JAW_TOOTH_ORDER.get(jaw_type, JAW_TOOTH_ORDER["lower"])
        ordered_pts = [fdi_dict[fdi] for fdi in desired_order if fdi in fdi_dict]
        if len(ordered_pts) < 4:
            raise ValueError(f"Found only {len(ordered_pts)} teeth for {jaw_type} jaw; need at least 4.")
        pruned = _prune_pathological_points(np.array(ordered_pts, dtype=np.float64))
        if len(pruned) < 4:
            raise ValueError(f"Only {len(pruned)} plausible teeth remain for {jaw_type} jaw; need at least 4.")
        return cls(
            control_points=pruned,
            jaw_type=jaw_type,
            smoothing=0.4 * len(pruned),
            source="predictions",
            ras_convention=ras_convention,
        )

    @classmethod
    def from_segmentation_mask(
        cls,
        mask: np.ndarray,
        affine: np.ndarray,
        jaw_type: str = "lower",
        ras_convention: bool = True,
    ) -> "DentalArch":
        skimage_morph = ensure_optional_package(
            "skimage.morphology", "scikit-image", "jawbone mask skeletonization"
        )
        jaw_id = LOWER_JAW_ID if jaw_type == "lower" else UPPER_JAW_ID
        jaw_binary = mask == jaw_id
        if not jaw_binary.any():
            raise ValueError(f"No voxels found for jawbone label {jaw_id} ({jaw_type})")

        axial_proj = jaw_binary.sum(axis=0) > 0
        skeleton = skimage_morph.skeletonize(axial_proj)
        sy, sx = np.where(skeleton)
        if len(sx) < 4:
            raise ValueError("Skeleton contains too few points to form an arch")

        pts = np.column_stack([sx, sy]).astype(np.float64)
        centroid = pts.mean(axis=0)
        radii = np.linalg.norm(pts - centroid, axis=1)
        median_radius = float(np.median(radii))
        if median_radius > 1e-6:
            in_band = radii < 1.45 * median_radius
            if in_band.sum() >= 8:
                pts = pts[in_band]

        ordered_indices = list(_order_points_around_arch(pts))
        z_indices = np.where(jaw_binary)[0]
        mean_z = float(np.mean(z_indices))
        step = max(1, len(ordered_indices) // 16)
        sub_indices = ordered_indices[::step]

        sub_pts = pts[sub_indices]
        vox_pts = np.column_stack([sub_pts[:, 0], sub_pts[:, 1], np.full(len(sub_pts), mean_z)])
        ones = np.ones((len(vox_pts), 1), dtype=np.float64)
        world_pts = (affine @ np.hstack([vox_pts, ones]).T).T[:, :3]

        return cls(
            control_points=world_pts,
            jaw_type=jaw_type,
            source="jawbone",
            ras_convention=ras_convention,
        )

    @classmethod
    def from_point_cloud(
        cls,
        points_3d: np.ndarray,
        jaw_type: str = "lower",
        num_control_points: int = 11,
        ras_convention: bool = True,
    ) -> "DentalArch":
        ctrl_pts = fit_dental_arch_from_points(
            points_3d,
            jaw_type=jaw_type,
            num_control_points=num_control_points,
            ras_convention=ras_convention,
        )
        return cls(
            control_points=ctrl_pts,
            jaw_type=jaw_type,
            smoothing=0.2 * len(ctrl_pts),
            source="point_cloud",
            ras_convention=ras_convention,
        )

    @classmethod
    def from_centerline_curve(
        cls,
        centerline: CenterlineCurve,
        jaw_type: str = "lower",
        ras_convention: bool = True,
    ) -> "DentalArch":
        return cls(
            control_points=centerline.control_points,
            jaw_type=jaw_type,
            smoothing=0.0,
            source=f"mask_centerline:{centerline.source}",
            ras_convention=ras_convention,
        )


def fit_dental_arch_from_points(
    points_3d: np.ndarray,
    jaw_type: str = "lower",
    num_control_points: int = 11,
    ras_convention: bool = True,
) -> np.ndarray:
    """Fit a parabolic dental arch Y(X), Z(X) to a 3D point cloud.

    In 3D Slicer RAS (`ras_convention=True`), +X is Patient Right and -X is Patient Left,
    so control points are ordered descending in X (+X -> -X).
    """
    _ = jaw_type
    pts = np.asarray(points_3d, dtype=np.float64)
    if len(pts) < 6:
        raise ValueError(f"Need at least 6 points to fit a dental arch, got {len(pts)}")

    med = np.median(pts, axis=0)
    dev = np.abs(pts - med)
    mad = np.median(dev, axis=0)
    mad[mad == 0] = 1.0
    valid = np.all(dev < 3.5 * mad, axis=1)
    filtered = pts[valid] if np.sum(valid) >= 6 else pts

    xs, ys, zs = filtered[:, 0], filtered[:, 1], filtered[:, 2]
    poly_y = np.polyfit(xs, ys, 2)
    poly_z = np.polyfit(xs, zs, 2)

    x_min = float(np.percentile(xs, 5))
    x_max = float(np.percentile(xs, 95))
    margin = (x_max - x_min) * 0.05
    if ras_convention:
        # Patient Right (+X) to Patient Left (-X)
        x_samples = np.linspace(x_max + margin, x_min - margin, num_control_points)
    else:
        # LPS: Patient Right (-X) to Patient Left (+X)
        x_samples = np.linspace(x_min - margin, x_max + margin, num_control_points)

    y_samples = np.polyval(poly_y, x_samples)
    z_samples = np.polyval(poly_z, x_samples)
    return np.column_stack([x_samples, y_samples, z_samples])

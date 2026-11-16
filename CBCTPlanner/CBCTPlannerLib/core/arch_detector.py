"""
Dental Arch Curve Detection, Sub-Sample Segment Projection (B13), FDI-to-Arc Mapping (B2),
and Principal Horseshoe Curve Fitting (B20) for U-shaped, V-shaped, and Tapered Arches.
All optional dependencies (scipy, scikit-image) are loaded lazily on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from . import ensure_optional_package
from .labels import (
    ANATOMICAL_MESIODISTAL_WIDTHS_MM,
    LOWER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    UPPER_JAW_ID,
)
from .mask_centerline import CenterlineCurve

# FDI sequence from patient right posterior to patient left posterior for each jaw.
JAW_TOOTH_ORDER: Dict[str, List[str]] = {
    "lower": [
        "48", "47", "46", "45", "44", "43", "42", "41",
        "31", "32", "33", "34", "35", "36", "37", "38",
    ],
    "upper": [
        "18", "17", "16", "15", "14", "13", "12", "11",
        "21", "22", "23", "24", "25", "26", "27", "28",
    ],
}


def project_point_to_polyline_arc_mm(
    pts: np.ndarray,
    arc_s: np.ndarray,
    point_mm: Sequence[float],
) -> Tuple[float, float, np.ndarray]:
    """B13: Exact projection of a 3D point onto a 3D polyline `(pts, arc_s)` by projecting onto each segment.

    Returns `(best_arc_mm, min_distance_mm, closest_point_ras)`.
    Achieves sub-sample accuracy instead of snapping to discrete sample vertices.
    """
    poly_pts = np.asarray(pts, dtype=np.float64)
    s_arr = np.asarray(arc_s, dtype=np.float64)
    pt = np.asarray(point_mm, dtype=np.float64).reshape(3)

    if len(poly_pts) == 0:
        return 0.0, float("inf"), pt.copy()
    if len(poly_pts) == 1:
        d = float(np.linalg.norm(poly_pts[0] - pt))
        return float(s_arr[0]), d, poly_pts[0].copy()

    p0 = poly_pts[:-1]
    p1 = poly_pts[1:]
    seg_vecs = p1 - p0
    seg_len_sq = np.sum(seg_vecs * seg_vecs, axis=1)
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)

    rel = pt[None, :] - p0
    t_raw = np.sum(rel * seg_vecs, axis=1) / safe_len_sq
    t_clamped = np.clip(np.where(seg_len_sq > 1e-12, t_raw, 0.0), 0.0, 1.0)

    proj_pts = p0 + t_clamped[:, None] * seg_vecs
    dists_sq = np.sum((proj_pts - pt[None, :]) ** 2, axis=1)
    best_i = int(np.argmin(dists_sq))

    best_t = float(t_clamped[best_i])
    best_arc = float(s_arr[best_i] + best_t * (s_arr[best_i + 1] - s_arr[best_i]))
    best_dist = float(np.sqrt(dists_sq[best_i]))
    best_pt = proj_pts[best_i].copy()
    return best_arc, best_dist, best_pt


def _voxel_grid_downsample(points: np.ndarray, grid_size_mm: float = 1.5) -> np.ndarray:
    """B20: Downsample a 3D point cloud using spatial voxel cells so dense mesh regions do not bias arch fitting."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= 32 or grid_size_mm <= 0.0:
        return pts
    mins = np.min(pts, axis=0)
    keys = np.floor((pts - mins) / float(grid_size_mm)).astype(np.int64)
    _, unique_inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = int(unique_inv.max()) + 1
    if n_cells < 6:
        return pts
    out = np.zeros((n_cells, 3), dtype=np.float64)
    counts = np.bincount(unique_inv).astype(np.float64)
    for d in range(3):
        out[:, d] = np.bincount(unique_inv, weights=pts[:, d]) / counts
    return out


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


@dataclass
class ToothArcSite:
    """B2: Mapped tooth site along the dental arch curve with explicit provenance."""

    fdi: str
    arc_mm: float
    t_normalized: float
    point_ras: np.ndarray
    source: str  # "segmentation", "calibrated_marker", "interpolated_from_neighbors", "approximate_width_table"
    is_approximate: bool
    is_missing_tooth: bool = False


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

        if ras_convention and pts[0, 0] < pts[-1, 0]:
            pts = pts[::-1].copy()
        elif (not ras_convention) and pts[0, 0] > pts[-1, 0]:
            pts = pts[::-1].copy()

        self.control_points = pts
        self.jaw_type = jaw_type
        self.source = source
        self.ras_convention = ras_convention

        up_vec = np.asarray(
            occlusal_normal if occlusal_normal is not None else (0.0, 0.0, 1.0), dtype=np.float64
        )
        up_norm = float(np.linalg.norm(up_vec))
        self.occlusal_normal = (
            up_vec / up_norm if up_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
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
        self._u_ctrl = u_norm

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
            pass

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

        t_proj = d_pt - np.dot(d_pt, up) * up
        t_norm = float(np.linalg.norm(t_proj))
        tangent = t_proj / t_norm if t_norm > 1e-8 else np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        normal = np.cross(tangent, up)
        n_norm = float(np.linalg.norm(normal))
        normal = normal / n_norm if n_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

        chord_mid = 0.5 * (self.control_points[0] + self.control_points[-1])
        outward_vec = pt - chord_mid
        outward_in_plane = outward_vec - np.dot(outward_vec, up) * up
        if np.linalg.norm(outward_in_plane) > 1e-3 and np.dot(normal, outward_in_plane) < 0:
            normal = -normal

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
        """B13: Sub-sample segment projection followed by local spline refinement."""
        pt = np.asarray(point_mm, dtype=np.float64).reshape(3)
        best_arc, best_dist, _ = project_point_to_polyline_arc_mm(
            self.pts_dense, self.cum_arc_lengths, pt
        )
        best_t = float(np.clip(best_arc / max(self.total_arc_length, 1e-9), 0.0, 1.0))

        if self._use_scipy and self.tck is not None:
            dt = 0.005
            fine_ts = np.linspace(max(0.0, best_t - dt), min(1.0, best_t + dt), 41)
            fine_arcs = fine_ts * self.total_arc_length
            u = np.interp(fine_arcs, self.cum_arc_lengths, self.u_dense)
            fine_pts = self._eval_raw(u)
            ref_arc, ref_dist, _ = project_point_to_polyline_arc_mm(fine_pts, fine_arcs, pt)
            if ref_dist <= best_dist:
                best_arc = ref_arc
                best_dist = ref_dist
                best_t = float(np.clip(best_arc / max(self.total_arc_length, 1e-9), 0.0, 1.0))
        return best_t, best_dist

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
            in_band = radii < 2.6 * median_radius
            if in_band.sum() >= 8:
                pts = pts[in_band]

        ordered_indices = list(_order_points_around_arch(pts))
        z_indices = np.where(jaw_binary)[0]
        mean_z = float(np.mean(z_indices))
        step = max(1, len(ordered_indices) // 16)
        sub_indices = ordered_indices[::step]
        if ordered_indices[-1] not in sub_indices:
            sub_indices.append(ordered_indices[-1])
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
            smoothing=0.1 * len(ctrl_pts),
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
    """B20: Fit a principal-curve spline along a 3D dental arch point cloud.

    Replaces the legacy quadratic `y(x)` fit and stride sampling:
      1. Spatial voxel-grid downsampling (`1.5 mm` cells) eliminates mesh vertex-density bias.
      2. Orders points around the open horseshoe arch from Patient Right (`+X` in RAS) through
         Anterior (`+Y`) to Patient Left (`-X`).
      3. Computes binned curvilinear medians along the horseshoe angle so U-shaped (parallel
         posterior legs), V-shaped, and tapered arches are tracked faithfully without parabolic distortion.
      4. Evaluates a smooth cubic B-spline at `num_control_points` uniform arc-length intervals.
    """
    _ = jaw_type
    raw_pts = np.asarray(points_3d, dtype=np.float64)
    if len(raw_pts) < 4:
        raise ValueError(f"Need at least 4 points to fit a dental arch, got {len(raw_pts)}")

    # 1. Remove vertex-density bias via spatial voxel-grid averaging when given a dense surface cloud
    pts = _voxel_grid_downsample(raw_pts, grid_size_mm=1.5) if len(raw_pts) > 32 else raw_pts.copy()

    # 2. Robust outlier rejection in Z (occlusal band) and XY
    if len(pts) >= 12:
        med = np.median(pts, axis=0)
        dev = np.abs(pts - med)
        mad = np.median(dev, axis=0)
        mad = np.where(mad < 1.0, 1.0, mad)
        valid = np.all(dev < 4.0 * mad, axis=1)
        if int(np.sum(valid)) >= 6:
            pts = pts[valid]

    # 3. Order points around the open horseshoe arch in the XY plane
    # Ensure reference point is strictly posterior to all points by at least 25mm
    # In RAS coordinates, +Y is Anterior (front/incisors) and -Y is Posterior (back/molars).
    y_min = float(np.min(pts[:, 1]))
    x_mid = float(0.5 * (np.percentile(pts[:, 0], 5) + np.percentile(pts[:, 0], 95)))
    ref_xy = np.array([x_mid, y_min - 25.0], dtype=np.float64)

    # Since pts[:, 1] - ref_xy[1] >= 25.0 > 0, angles are strictly in (0, pi)
    # Right (+X) -> Midline (X=0) -> Left (-X) corresponds to angles increasing from ~0 to ~pi
    angles = np.arctan2(pts[:, 1] - ref_xy[1], pts[:, 0] - ref_xy[0])

    if len(pts) <= 60:
        order = np.argsort(angles)
        ordered = pts[order]
        ordered = _prune_pathological_points(ordered, max_deviation_mm=10.0)
    else:
        n_bins = min(max(num_control_points * 2, 21), max(12, len(pts) // 2))
        a_min = float(np.percentile(angles, 1))
        a_max = float(np.percentile(angles, 99))
        edges = np.linspace(a_min, a_max, n_bins + 1)
        bin_centers: List[np.ndarray] = []
        for b_idx in range(n_bins):
            if b_idx < n_bins - 1:
                in_bin = (angles >= edges[b_idx]) & (angles < edges[b_idx + 1])
            else:
                in_bin = (angles >= edges[b_idx]) & (angles <= edges[b_idx + 1])
            if np.any(in_bin):
                bin_centers.append(np.median(pts[in_bin], axis=0))
        ordered = np.array(bin_centers, dtype=np.float64)
        ordered = _prune_pathological_points(ordered, max_deviation_mm=10.0)

    if len(ordered) < 4:
        order = np.argsort(pts[:, 0])[::-1] if ras_convention else np.argsort(pts[:, 0])
        ordered = pts[order]

    if ras_convention and ordered[0, 0] < ordered[-1, 0]:
        ordered = ordered[::-1].copy()
    elif (not ras_convention) and ordered[0, 0] > ordered[-1, 0]:
        ordered = ordered[::-1].copy()

    # 4. Resample uniformly along arc length via cubic spline
    diffs = np.diff(ordered, axis=0)
    seg_d = np.linalg.norm(diffs, axis=1)
    keep = np.concatenate([[True], seg_d > 1e-4])
    ordered = ordered[keep]
    if len(ordered) < 4:
        return ordered

    diffs = np.diff(ordered, axis=0)
    cum_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    total_s = float(cum_s[-1])
    if total_s <= 1e-6:
        return ordered
    u_norm = cum_s / total_s

    try:
        from scipy import interpolate  # type: ignore

        k = min(3, len(ordered) - 1)
        smoothing = 0.01 * len(ordered) if len(raw_pts) > 60 else 0.0
        tck, _ = interpolate.splprep(
            [ordered[:, 0], ordered[:, 1], ordered[:, 2]],
            u=u_norm,
            s=smoothing,
            k=k,
        )
        u_dense = np.linspace(0.0, 1.0, 300)
        dense_pts = np.column_stack(interpolate.splev(u_dense, tck))
        dense_s = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(dense_pts, axis=0), axis=1))]
        )
        target_s = np.linspace(0.0, float(dense_s[-1]), num_control_points)
        target_u = np.interp(target_s, dense_s, u_dense)
        ctrl = np.column_stack(interpolate.splev(target_u, tck))
    except ImportError:
        target_s = np.linspace(0.0, total_s, num_control_points)
        ctrl = np.column_stack([np.interp(target_s, cum_s, ordered[:, d]) for d in range(3)])

    if ras_convention and ctrl[0, 0] < ctrl[-1, 0]:
        ctrl = ctrl[::-1].copy()
    elif (not ras_convention) and ctrl[0, 0] > ctrl[-1, 0]:
        ctrl = ctrl[::-1].copy()
    return ctrl


def compute_fdi_arc_positions_mm(
    total_arc_length_mm: float,
    jaw_type: str = "lower",
    arch: Optional[DentalArch] = None,
    curve_pts: Optional[np.ndarray] = None,
    curve_arc_s: Optional[np.ndarray] = None,
    segmented_centroids: Optional[Dict[str, np.ndarray]] = None,
    calibrated_arc_mm: Optional[Dict[str, float]] = None,
) -> Dict[str, ToothArcSite]:
    """B2: Map every FDI tooth in `jaw_type` to its arc-length position along the curve."""
    jaw_key = "lower" if jaw_type.lower().startswith("low") else "upper"
    order = JAW_TOOTH_ORDER[jaw_key]
    L = max(1.0, float(total_arc_length_mm))

    if arch is not None and (curve_pts is None or curve_arc_s is None):
        curve_pts = arch.pts_dense
        curve_arc_s = arch.cum_arc_lengths

    def _eval_point_at_arc(s_val: float) -> np.ndarray:
        s_c = float(np.clip(s_val, 0.0, L))
        if arch is not None:
            return arch.evaluate_at_mm(s_c)
        if curve_pts is not None and curve_arc_s is not None and len(curve_pts) >= 2:
            return np.array(
                [np.interp(s_c, curve_arc_s, curve_pts[:, d]) for d in range(3)],
                dtype=np.float64,
            )
        return np.zeros(3, dtype=np.float64)

    widths = np.array([ANATOMICAL_MESIODISTAL_WIDTHS_MM.get(fdi, 7.5) for fdi in order], dtype=np.float64)
    cum_centers = np.cumsum(widths) - 0.5 * widths
    span_start = 0.04 * L
    span_end = 0.96 * L
    norm_w = (cum_centers - cum_centers[0]) / max(1e-6, float(cum_centers[-1] - cum_centers[0]))
    table_arcs = span_start + norm_w * (span_end - span_start)

    anchored_arc: Dict[int, float] = {}
    anchored_source: Dict[int, str] = {}

    cal_map = {str(k): float(v) for k, v in (calibrated_arc_mm or {}).items()}
    seg_map = {str(k): np.asarray(v, dtype=np.float64) for k, v in (segmented_centroids or {}).items()}

    for idx, fdi in enumerate(order):
        if fdi in cal_map:
            anchored_arc[idx] = float(np.clip(cal_map[fdi], 0.0, L))
            anchored_source[idx] = "calibrated_marker"
        elif fdi in seg_map:
            c_pt = seg_map[fdi]
            if curve_pts is not None and curve_arc_s is not None and len(curve_pts) >= 2:
                proj_s, _, _ = project_point_to_polyline_arc_mm(curve_pts, curve_arc_s, c_pt)
                anchored_arc[idx] = float(np.clip(proj_s, 0.0, L))
            elif arch is not None:
                t_val, _ = arch.find_closest_parameter(c_pt)
                anchored_arc[idx] = float(np.clip(t_val * L, 0.0, L))
            else:
                anchored_arc[idx] = float(table_arcs[idx])
            anchored_source[idx] = "segmentation"

    results: Dict[str, ToothArcSite] = {}
    anchored_indices = sorted(anchored_arc.keys())

    for idx, fdi in enumerate(order):
        if idx in anchored_arc:
            s_val = anchored_arc[idx]
            src = anchored_source[idx]
            is_approx = False
            is_missing = False
        elif len(anchored_indices) >= 1:
            left_neighbors = [i for i in anchored_indices if i < idx]
            right_neighbors = [i for i in anchored_indices if i > idx]
            if left_neighbors and right_neighbors:
                i0 = left_neighbors[-1]
                i1 = right_neighbors[0]
                dw_total = float(cum_centers[i1] - cum_centers[i0])
                frac = float(cum_centers[idx] - cum_centers[i0]) / max(1e-6, dw_total)
                s_val = float(np.clip(anchored_arc[i0] + frac * (anchored_arc[i1] - anchored_arc[i0]), 0.0, L))
            elif left_neighbors:
                i0 = left_neighbors[-1]
                scale = (span_end - span_start) / max(1e-6, float(cum_centers[-1] - cum_centers[0]))
                s_val = float(np.clip(anchored_arc[i0] + (cum_centers[idx] - cum_centers[i0]) * scale, 0.0, L))
            else:
                i1 = right_neighbors[0]
                scale = (span_end - span_start) / max(1e-6, float(cum_centers[-1] - cum_centers[0]))
                s_val = float(np.clip(anchored_arc[i1] - (cum_centers[i1] - cum_centers[idx]) * scale, 0.0, L))
            src = "interpolated_from_neighbors"
            is_approx = True
            is_missing = bool(seg_map) and (fdi not in seg_map)
        else:
            s_val = float(table_arcs[idx])
            src = "approximate_width_table"
            is_approx = True
            is_missing = False

        results[fdi] = ToothArcSite(
            fdi=fdi,
            arc_mm=round(s_val, 3),
            t_normalized=round(s_val / L, 6),
            point_ras=_eval_point_at_arc(s_val),
            source=src,
            is_approximate=is_approx,
            is_missing_tooth=is_missing,
        )
    return results

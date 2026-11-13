"""
Dental Arch Curve Detection and Curved Planar Reformation (CPR) Frame Computation.
Extracts 3D dental arch splines from jawbone or teeth segmentation,
supports manual refinement, and computes the Frenet/orthogonal reslicing frames.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import interpolate
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .labels import (
    LOWER_JAW_ID,
    UPPER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    LABEL_REGISTRY,
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
    """Drop isolated control points that jump off the local path.

    Each interior point is compared against the midpoint of its path neighbours;
    a predicted centroid that a segmentation error has thrown far off the arch is
    removed before the spline is fitted, so one bad tooth cannot fold the curve.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 5:
        return pts

    # Remove one worst point per pass: dropping every offending point at once would
    # also discard the two neighbours of an outlier, because the outlier poisons
    # their local midpoints too.
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
    """Order 2D points along an open horseshoe from one posterior end to the other.

    Polar ordering around the centroid, started just after the largest angular gap,
    follows the dental arch instead of jumping between its two posterior branches.
    """
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
    point: np.ndarray      # (3,) world coordinates [x, y, z] in mm
    tangent: np.ndarray    # (3,) unit vector along the arch trajectory (mesiodistal)
    normal: np.ndarray     # (3,) unit vector perpendicular to arch (buccolingual)
    binormal: np.ndarray   # (3,) unit vector perpendicular to arch plane (apicocoronal / Z)
    t: float               # normalized parameter [0.0, 1.0]
    arc_length_mm: float   # distance in mm from arch start


class DentalArch:
    """
    Parametric 3D Dental Arch Spline.
    Provides uniform arc-length sampling, tangent/normal frame vectors,
    and cross-sectional plane definitions for simulated panoramic & transverse views.
    """

    def __init__(
        self,
        control_points: np.ndarray,
        jaw_type: str = "lower",
        smoothing: float = 0.0,
        source: str = "manual",
    ):
        """
        Args:
            control_points: (N, 3) array of ordered 3D points in mm along the arch
            jaw_type: "lower" (mandible) or "upper" (maxilla)
            smoothing: spline smoothing factor (0.0 = interpolating)
            source: how the curve was obtained ("predictions", "jawbone", "manual")
        """
        self.control_points = np.asarray(control_points, dtype=np.float64)
        self.jaw_type = jaw_type
        self.source = source
        if len(self.control_points) < 4:
            raise ValueError(f"Need at least 4 control points to fit a cubic spline, got {len(self.control_points)}")

        # Fit parametric spline (k=3 for cubic)
        k = min(3, len(self.control_points) - 1)
        # Parameterize by cumulative chord distance
        diffs = np.diff(self.control_points, axis=0)
        chord_dists = np.sqrt((diffs**2).sum(axis=1))
        u = np.zeros(len(self.control_points))
        u[1:] = np.cumsum(chord_dists)
        total_chord = u[-1]
        if total_chord <= 1e-6:
            raise ValueError("Control points are degenerate (zero distance)")
        u_norm = u / total_chord

        tck, _ = interpolate.splprep(
            [self.control_points[:, 0], self.control_points[:, 1], self.control_points[:, 2]],
            u=u_norm,
            s=smoothing,
            k=k,
        )
        self.tck = tck

        # Precompute dense arc-length table for accurate arc-length reparameterization
        dense_samples = 500
        u_dense = np.linspace(0.0, 1.0, dense_samples)
        pts_dense = np.column_stack(interpolate.splev(u_dense, self.tck))
        dense_diffs = np.diff(pts_dense, axis=0)
        seg_lengths = np.sqrt((dense_diffs**2).sum(axis=1))
        self.cum_arc_lengths = np.zeros(dense_samples)
        self.cum_arc_lengths[1:] = np.cumsum(seg_lengths)
        self.total_arc_length = float(self.cum_arc_lengths[-1])
        self.u_dense = u_dense
        # Bolt optimization: Cache dense 3D points and normalized arc parameters
        # to enable O(1) vectorized coarse lookup in find_closest_parameter without
        # repeated scalar spline evaluations.
        self.pts_dense = pts_dense
        self.t_dense = self.cum_arc_lengths / max(self.total_arc_length, 1e-9)

    def _arc_to_u(self, arc_mm: float) -> float:
        """Convert physical arc length in mm to spline parameter u in [0, 1]."""
        arc_clamped = np.clip(arc_mm, 0.0, self.total_arc_length)
        return float(np.interp(arc_clamped, self.cum_arc_lengths, self.u_dense))

    def evaluate_at(self, t: float) -> np.ndarray:
        """
        Evaluate 3D point on arch curve at normalized arc-length parameter t in [0.0, 1.0].
        0.0 = right posterior (tooth #48/18), 0.5 = anterior midline, 1.0 = left posterior (tooth #38/28).
        """
        t = float(np.clip(t, 0.0, 1.0))
        arc_mm = t * self.total_arc_length
        u = self._arc_to_u(arc_mm)
        pt = interpolate.splev(u, self.tck)
        return np.array([pt[0], pt[1], pt[2]], dtype=np.float64)

    def evaluate_at_mm(self, arc_mm: float) -> np.ndarray:
        """Evaluate 3D point on arch curve at physical arc length from start (in mm)."""
        u = self._arc_to_u(arc_mm)
        pt = interpolate.splev(u, self.tck)
        return np.array([pt[0], pt[1], pt[2]], dtype=np.float64)

    def get_frame(self, t: float) -> ArchFrame:
        """
        Compute the local anatomical orthonormal frame (Point, Tangent, Normal, Binormal) at t in [0, 1].
        - Tangent: along arch (mesiodistal)
        - Normal: horizontal bucco-lingual normal (pointing outward/buccally)
        - Binormal: apicocoronal (Z-axis direction)
        """
        t = float(np.clip(t, 0.0, 1.0))
        arc_mm = t * self.total_arc_length
        u = self._arc_to_u(arc_mm)

        # First derivative (tangent)
        pt = np.array(interpolate.splev(u, self.tck), dtype=np.float64)
        d_pt = np.array(interpolate.splev(u, self.tck, der=1), dtype=np.float64)

        t_norm = np.linalg.norm(d_pt)
        if t_norm > 1e-8:
            tangent = d_pt / t_norm
        else:
            tangent = np.array([0.0, 1.0, 0.0])

        # Anatomical dental coordinate convention:
        # Cross product of tangent with patient superior-inferior axis gives the buccal-lingual normal.
        z_up = np.array([0.0, 0.0, 1.0])
        normal = np.cross(tangent, z_up)
        n_norm = np.linalg.norm(normal)
        if n_norm > 1e-8:
            normal = normal / n_norm
        else:
            normal = np.array([1.0, 0.0, 0.0])

        # Ensure normal points outward (buccal direction away from jaw center)
        # Compute vector from curve centroid to current point
        center = np.mean(self.control_points, axis=0)
        outward_ref = pt - center
        if np.dot(normal, outward_ref) < 0:
            normal = -normal

        # Binormal = Tangent x Normal (points apicocoronally along Z)
        binormal = np.cross(tangent, normal)
        b_norm = np.linalg.norm(binormal)
        if b_norm > 1e-8:
            binormal = binormal / b_norm
        else:
            binormal = z_up

        # Clinically and anatomically, the vertical slice axis (binormal) must strictly point cranial/superior (+Z)
        if binormal[2] < 0:
            binormal = -binormal

        return ArchFrame(
            point=pt,
            tangent=tangent,
            normal=normal,
            binormal=binormal,
            t=t,
            arc_length_mm=arc_mm,
        )

    def sample_frames_uniform(self, step_mm: float = 1.0) -> List[ArchFrame]:
        """Generate uniformly spaced frames along the entire arch at step_mm intervals."""
        n_steps = max(2, int(np.round(self.total_arc_length / step_mm)))
        frames = []
        for i in range(n_steps + 1):
            t = i / float(n_steps)
            frames.append(self.get_frame(t))
        return frames

    def find_closest_parameter(self, point_mm: np.ndarray) -> Tuple[float, float]:
        """
        Find parameter t in [0, 1] on the arch curve closest to point_mm.
        Returns: (t, min_distance_mm)
        """
        pt = np.asarray(point_mm, dtype=np.float64).reshape(3)
        # Bolt optimization: Vectorized coarse search using cached dense points
        # instead of 200 sequential scalar spline evaluations.
        dists_sq = np.sum((self.pts_dense - pt) ** 2, axis=1)
        best_idx = int(np.argmin(dists_sq))
        best_t = self.t_dense[best_idx]

        # Refine with local fine search: evaluate all candidate points in a single
        # batched splev call rather than 50 separate evaluate_at calls.
        dt = 0.02
        fine_ts = np.linspace(max(0.0, best_t - dt), min(1.0, best_t + dt), 50)
        arc_mm = fine_ts * self.total_arc_length
        u = np.interp(arc_mm, self.cum_arc_lengths, self.u_dense)
        fine_pts = np.column_stack(interpolate.splev(u, self.tck))
        fine_dists_sq = np.sum((fine_pts - pt) ** 2, axis=1)
        fine_idx = int(np.argmin(fine_dists_sq))

        return float(fine_ts[fine_idx]), float(np.sqrt(fine_dists_sq[fine_idx]))

    @classmethod
    def from_teeth_centroids(
        cls,
        teeth_centroids: List[Tuple[str, np.ndarray]],
        jaw_type: str = "lower",
    ) -> "DentalArch":
        """
        Construct arch from ordered teeth centroids.
        Ordered FDI sequence:
        Lower jaw: 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38
        Upper jaw: 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28
        """
        fdi_dict = {fdi: pt for fdi, pt in teeth_centroids}
        desired_order = JAW_TOOTH_ORDER.get(jaw_type, JAW_TOOTH_ORDER["lower"])
        ordered_pts = [fdi_dict[fdi] for fdi in desired_order if fdi in fdi_dict]

        if len(ordered_pts) < 4:
            raise ValueError(f"Found only {len(ordered_pts)} teeth for {jaw_type} jaw; need at least 4.")

        ordered_pts = _prune_pathological_points(np.array(ordered_pts, dtype=np.float64))
        if len(ordered_pts) < 4:
            raise ValueError(f"Only {len(ordered_pts)} plausible teeth remain for {jaw_type} jaw; need at least 4.")

        # A light smoothing fit follows the predicted arch without chasing individual
        # centroid noise (root/crown asymmetry, partially erupted molars).
        return cls(
            control_points=ordered_pts,
            jaw_type=jaw_type,
            smoothing=0.4 * len(ordered_pts),
            source="predictions",
        )

    @classmethod
    def from_segmentation_mask(
        cls,
        mask: np.ndarray,
        affine: np.ndarray,
        jaw_type: str = "lower",
    ) -> "DentalArch":
        """
        Extract dental arch directly from jawbone segmentation mask.
        1. Extract jawbone voxels (label 1 for lower jaw, label 2 for upper jaw).
        2. Maximum intensity / thickness axial projection.
        3. Morphological skeletonization.
        4. Traverse skeleton to order points.
        5. Map voxel coordinates to world coordinates via affine.
        """
        jaw_id = LOWER_JAW_ID if jaw_type == "lower" else UPPER_JAW_ID
        jaw_binary = (mask == jaw_id)
        if not jaw_binary.any():
            raise ValueError(f"No voxels found for jawbone label {jaw_id} ({jaw_type})")

        # 2D axial projection (summing along Z)
        axial_proj = jaw_binary.sum(axis=0) > 0  # (Y, X)
        skeleton = skeletonize(axial_proj)

        sy, sx = np.where(skeleton)
        if len(sx) < 4:
            raise ValueError("Skeleton contains too few points to form an arch")

        pts = np.column_stack([sx, sy]).astype(np.float64)  # (X, Y)

        # The jawbone skeleton includes the rami, which sit outside the dental arch.
        # Keeping the points within a band around the centroid leaves the alveolar arc.
        centroid = pts.mean(axis=0)
        radii = np.linalg.norm(pts - centroid, axis=1)
        median_radius = float(np.median(radii))
        if median_radius > 1e-6:
            in_band = radii < 1.45 * median_radius
            if in_band.sum() >= 8:
                pts = pts[in_band]

        # Order the alveolar arc from one posterior end to the other.
        ordered_indices = list(_order_points_around_arch(pts))

        # Find median Z for each (X, Y) point from the 3D mask
        # To make it fast, use the mean Z of jawbone voxels
        z_indices, y_indices, x_indices = np.where(jaw_binary)
        mean_z = np.mean(z_indices)

        # Downsample ordered points to ~15-20 evenly spaced control points
        step = max(1, len(ordered_indices) // 16)
        sub_indices = ordered_indices[::step]

        # Map (X, Y, Z) voxel points to world mm
        sub_pts = pts[sub_indices]
        vox_pts = np.column_stack([sub_pts[:, 0], sub_pts[:, 1], np.full(len(sub_pts), mean_z)])
        # Apply affine: affine is 4x4, operates on [x, y, z, 1]^T
        ones = np.ones((len(vox_pts), 1))
        vox_homog = np.hstack([vox_pts, ones])
        world_pts = (affine @ vox_homog.T).T[:, :3]

        # Ensure order is right-to-left (standard dental panorama convention: patient right -> patient left)
        # In LPS world coordinates: +X is Left, -X is Right. So sort from -X to +X.
        if world_pts[0, 0] > world_pts[-1, 0]:
            world_pts = world_pts[::-1]

        return cls(control_points=world_pts, jaw_type=jaw_type, source="jawbone")

    @classmethod
    def from_point_cloud(
        cls,
        points_3d: np.ndarray,
        jaw_type: str = "lower",
        num_control_points: int = 11,
    ) -> "DentalArch":
        """
        Fit a smooth, high-precision dental arch curve directly from a 3D point cloud
        (e.g. segmented teeth surface vertices, crest points, or thresholded CBCT coordinates).
        """
        ctrl_pts = fit_dental_arch_from_points(
            points_3d,
            jaw_type=jaw_type,
            num_control_points=num_control_points,
        )
        return cls(
            control_points=ctrl_pts,
            jaw_type=jaw_type,
            smoothing=0.2 * len(ctrl_pts),
            source="point_cloud",
        )

    @classmethod
    def from_centerline_curve(
        cls,
        centerline: CenterlineCurve,
        jaw_type: str = "lower",
    ) -> "DentalArch":
        """Build a :class:`DentalArch` from a mask-derived :class:`CenterlineCurve`.

        The centerline control points already follow real anatomy (extracted from
        the nnUNet segmentation mask via 3D skeletonization + minimal-cost-path
        ordering), so we interpolate through them without re-smoothing away the
        anatomical curvature (Spee / Wilson curve) that a predefined parabola
        would destroy.
        """
        return cls(
            control_points=centerline.control_points,
            jaw_type=jaw_type,
            smoothing=0.0,
            source=f"mask_centerline:{centerline.source}",
        )


def fit_dental_arch_from_points(
    points_3d: np.ndarray,
    jaw_type: str = "lower",
    num_control_points: int = 11,
) -> np.ndarray:
    """
    Robust anatomical dental arch fitting to 3D point cloud (teeth or alveolar crest).
    Fits a smooth parabolic arch Y(X) = a*X^2 + b*X + c and Z(X) occlusal profile,
    generating ordered control points from Patient Right (-X) to Patient Left (+X).
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    if len(pts) < 6:
        raise ValueError(f"Need at least 6 points to fit a dental arch, got {len(pts)}")

    # Prune extreme statistical outliers (> 3.5 MAD from median)
    med = np.median(pts, axis=0)
    dev = np.abs(pts - med)
    mad = np.median(dev, axis=0)
    mad[mad == 0] = 1.0
    valid = np.all(dev < 3.5 * mad, axis=1)
    filtered = pts[valid] if np.sum(valid) >= 6 else pts

    xs = filtered[:, 0]
    ys = filtered[:, 1]
    zs = filtered[:, 2]

    # Fit quadratic parabola in horizontal plane: Y = a*X^2 + b*X + c
    poly_y = np.polyfit(xs, ys, 2)
    # Fit Z profile along arch (curve of Spee / Wilson)
    poly_z = np.polyfit(xs, zs, 2)

    x_min = float(np.percentile(xs, 5))
    x_max = float(np.percentile(xs, 95))
    margin = (x_max - x_min) * 0.05
    x_samples = np.linspace(x_min - margin, x_max + margin, num_control_points)
    y_samples = np.polyval(poly_y, x_samples)
    z_samples = np.polyval(poly_z, x_samples)

    ctrl_pts = np.column_stack([x_samples, y_samples, z_samples])
    # Ensure order is Patient Right to Patient Left (-X to +X in RAS/LPS)
    if ctrl_pts[0, 0] > ctrl_pts[-1, 0]:
        ctrl_pts = ctrl_pts[::-1]
    return ctrl_pts


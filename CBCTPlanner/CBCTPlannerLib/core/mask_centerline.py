"""
Mask-driven 3D centerline extraction for dental CBCT segmentation masks.

Replaces the *predefined* parabolic arch curve and the *hardcoded geometric
ellipse* with curves extracted directly from the nnUNet segmentation masks:

  * Mandibular canal (nerve) centerline from IAN labels 3/4, incisive canals
    43/44 and lingual canal 45 -> 3D skeletonization + minimal-cost-path
    (Dijkstra) ordering + B-spline fit.
  * Dental arch centerline from teeth + alveolar crest jawbone masks -> same
    3D skeletonization + endpoint-to-endpoint path ordering + B-spline.

Every curve is a B-spline with arc-length parameterization and orthonormal
Frenet frames (tangent / normal / binormal), so cross-sectional slices,
measurements and nerve-clearance checks track real patient anatomy instead of
a fixed ellipse or parabola.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import interpolate, sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize as skeletonize_3d

from .labels import (
    LEFT_IAN_ID,
    RIGHT_IAN_ID,
    LEFT_INCISIVE_CANAL_ID,
    RIGHT_INCISIVE_CANAL_ID,
    LINGUAL_CANAL_ID,
    LOWER_JAW_ID,
    UPPER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
)


# --------------------------------------------------------------------------- #
# Frame & curve container
# --------------------------------------------------------------------------- #
@dataclass
class CenterlineFrame:
    """Orthonormal frame at one point along a 3D centerline."""

    point: np.ndarray          # (3,) world mm [x, y, z]
    tangent: np.ndarray        # (3,) unit vector along the centerline
    normal: np.ndarray         # (3,) unit vector (bucco-lingual for arch)
    binormal: np.ndarray       # (3,) unit vector (apico-coronal, +Z superior)
    t: float                   # normalized parameter [0, 1]
    arc_length_mm: float       # distance from curve start in mm


class CenterlineCurve:
    """Smooth 3D centerline curve with arc-length parameterization and frames."""

    def __init__(self, control_points: np.ndarray, source: str = "mask"):
        self.control_points = np.asarray(control_points, dtype=np.float64)
        self.source = source
        if len(self.control_points) < 4:
            raise ValueError(
                f"CenterlineCurve needs >= 4 control points, got {len(self.control_points)}"
            )
        self._build_spline()

    # -- spline construction ----------------------------------------------- #
    def _build_spline(self) -> None:
        pts = self.control_points
        diffs = np.diff(pts, axis=0)
        chord = np.sqrt((diffs ** 2).sum(axis=1))
        u = np.zeros(len(pts))
        u[1:] = np.cumsum(chord)
        total = float(u[-1])
        if total <= 1e-6:
            raise ValueError("Degenerate centerline: zero total chord length")
        u_norm = u / total
        k = min(3, len(pts) - 1)
        self.tck, _ = interpolate.splprep(
            [pts[:, 0], pts[:, 1], pts[:, 2]],
            u=u_norm,
            s=0.0,            # interpolating (passes through mask points)
            k=k,
        )
        # Dense arc-length table for O(1) arc-length reparameterization.
        dense = 500
        ud = np.linspace(0.0, 1.0, dense)
        pts_dense = np.column_stack(interpolate.splev(ud, self.tck))
        d_dense = np.diff(pts_dense, axis=0)
        seg = np.sqrt((d_dense ** 2).sum(axis=1))
        self.cum_arc = np.zeros(dense)
        self.cum_arc[1:] = np.cumsum(seg)
        self.total_arc_length = float(self.cum_arc[-1])
        self.pts_dense = pts_dense
        self.u_dense = ud

    # -- evaluation --------------------------------------------------------- #
    def _u_for_t(self, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        arc = t * self.total_arc_length
        u_grid = np.linspace(0.0, 1.0, len(self.cum_arc))
        return float(np.interp(arc, self.cum_arc, u_grid))

    def evaluate_at(self, t: float) -> np.ndarray:
        """3D point on the curve at normalized arc-length t in [0, 1]."""
        u = self._u_for_t(t)
        pt = interpolate.splev(u, self.tck)
        return np.array([pt[0], pt[1], pt[2]], dtype=np.float64)

    def get_frame(self, t: float) -> CenterlineFrame:
        """Orthonormal Frenet frame at parameter t in [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        u = self._u_for_t(t)
        pt = np.array(interpolate.splev(u, self.tck), dtype=np.float64)
        d1 = np.array(interpolate.splev(u, self.tck, der=1), dtype=np.float64)
        d2 = np.array(interpolate.splev(u, self.tck, der=2), dtype=np.float64)

        t_norm = float(np.linalg.norm(d1))
        tangent = d1 / t_norm if t_norm > 1e-8 else np.array([1.0, 0.0, 0.0])

        # Curvature normal = component of second derivative perpendicular to tangent
        n_raw = d2 - np.dot(d2, tangent) * tangent
        n_norm = float(np.linalg.norm(n_raw))
        if n_norm > 1e-8:
            normal = n_raw / n_norm
        else:
            z_up = np.array([0.0, 0.0, 1.0])
            normal = np.cross(tangent, z_up)
            n_norm = float(np.linalg.norm(normal))
            if n_norm < 1e-8:
                normal = np.array([1.0, 0.0, 0.0])
            else:
                normal = normal / n_norm

        binormal = np.cross(tangent, normal)
        b_norm = float(np.linalg.norm(binormal))
        if b_norm > 1e-8:
            binormal = binormal / b_norm
        else:
            binormal = np.array([0.0, 0.0, 1.0])

        # Keep binormal strictly superior (+Z) so slice views stay upright
        if binormal[2] < 0:
            binormal = -binormal
            normal = -normal  # preserve right-handed frame

        return CenterlineFrame(
            point=pt,
            tangent=tangent,
            normal=normal,
            binormal=binormal,
            t=t,
            arc_length_mm=t * self.total_arc_length,
        )

    def sample_frames_uniform(self, step_mm: float = 1.0) -> List[CenterlineFrame]:
        n = max(2, int(round(self.total_arc_length / step_mm)))
        return [self.get_frame(i / float(n)) for i in range(n + 1)]

    def find_closest_parameter(self, point_mm: np.ndarray) -> Tuple[float, float]:
        pt = np.asarray(point_mm, dtype=np.float64).reshape(3)
        dists_sq = np.sum((self.pts_dense - pt) ** 2, axis=1)
        best_idx = int(np.argmin(dists_sq))
        # Return a normalized ARC-LENGTH parameter (matching evaluate_at/get_frame),
        # not the raw spline parameter u.
        best_t = float(self.cum_arc[best_idx] / max(self.total_arc_length, 1e-9))
        # Local refinement
        dt = 0.02
        fine_t = np.linspace(max(0.0, best_t - dt), min(1.0, best_t + dt), 50)
        arc = fine_t * self.total_arc_length
        u_grid = np.linspace(0.0, 1.0, len(self.cum_arc))
        u = np.interp(arc, self.cum_arc, u_grid)
        fine_pts = np.column_stack(interpolate.splev(u, self.tck))
        fd_sq = np.sum((fine_pts - pt) ** 2, axis=1)
        fi = int(np.argmin(fd_sq))
        return float(fine_t[fi]), float(np.sqrt(fd_sq[fi]))


# --------------------------------------------------------------------------- #
# Skeleton graph helpers
# --------------------------------------------------------------------------- #
_SKELETON_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def _skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Voxel coords (N,3) in (z,y,x) of skeleton endpoints (exactly one 26-neighbor)."""
    coords = np.argwhere(skel)
    endpoints: List[Tuple[int, int, int]] = []
    for (z, y, x) in coords:
        cnt = 0
        for dz, dy, dx in _SKELETON_OFFSETS:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (
                0 <= nz < skel.shape[0]
                and 0 <= ny < skel.shape[1]
                and 0 <= nx < skel.shape[2]
                and skel[nz, ny, nx]
            ):
                cnt += 1
        if cnt == 1:
            endpoints.append((z, y, x))
    if not endpoints:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(endpoints, dtype=np.int64)


def _skeleton_graph(skel: np.ndarray):
    """Adjacency graph of skeleton voxels. Returns (coords_zyx, index_map, adj)."""
    coords = np.argwhere(skel)
    index = {tuple(c): i for i, c in enumerate(coords)}
    adj: Dict[int, List[int]] = {i: [] for i in range(len(coords))}
    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in _SKELETON_OFFSETS:
            nb = (z + dz, y + dy, x + dx)
            j = index.get(nb)
            if j is not None:
                adj[i].append(j)
    return coords, index, adj


def _order_skeleton_path(skel: np.ndarray, start_zyx, end_zyx) -> Optional[np.ndarray]:
    """Dijkstra shortest path along the skeleton from start to end voxel.

    Returns ordered voxel coordinates (N,3) in (z,y,x), or None if no path.
    """
    coords, index, adj = _skeleton_graph(skel)
    n = len(coords)
    s = index.get(tuple(start_zyx))
    tgt = index.get(tuple(end_zyx))
    if s is None or tgt is None:
        return None
    rows, cols, data = [], [], []
    for i, nbs in adj.items():
        for j in nbs:
            rows.append(i)
            cols.append(j)
            data.append(1.0)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    dists, predecessors = dijkstra(
        graph, directed=False, indices=s, return_predecessors=True
    )
    if not np.isfinite(dists[tgt]):
        return None
    path: List[int] = []
    cur = tgt
    while cur != -9999:
        path.append(cur)
        cur = int(predecessors[cur])
        if cur == s:
            path.append(s)
            break
        if len(path) > n + 1:
            return None
    path.reverse()
    return coords[path]


def _farthest_endpoint_pair(endpoints: np.ndarray):
    """Pick the endpoint pair with the greatest Euclidean distance (proxy for full extent)."""
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_dist = -1.0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            d = float(np.linalg.norm(endpoints[i] - endpoints[j]))
            if d > best_dist:
                best_dist = d
                best = (endpoints[i], endpoints[j])
    return best


def _voxels_to_world(path_zyx: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map ordered (z,y,x) voxel indices to world mm via the 4x4 affine."""
    vox_xyz = path_zyx[:, [2, 1, 0]].astype(np.float64)
    homog = np.hstack([vox_xyz, np.ones((len(vox_xyz), 1))])
    return (affine @ homog.T).T[:, :3]


def _downsample(points: np.ndarray, max_ctrl: int) -> np.ndarray:
    if len(points) <= max_ctrl:
        return points
    step = max(1, len(points) // max_ctrl)
    return points[::step]


# --------------------------------------------------------------------------- #
# Centerline extractors
# --------------------------------------------------------------------------- #
def _centerline_from_binary(
    binary: np.ndarray,
    affine: np.ndarray,
    max_ctrl: int = 60,
) -> Optional[Tuple[CenterlineCurve, int]]:
    """Skeletonize one binary structure, order its farthest endpoints via
    Dijkstra, fit a B-spline, and return (curve, num_path_points).

    Returns None if the structure is absent, the skeleton is degenerate, or
    no >=4-point path can be ordered.  Used per-canal-label because the left
    and right IAN canals are disconnected structures - unioning them into one
    skeleton graph makes the farthest-endpoint Dijkstra fail
    (no path across disconnected components).
    """
    if not binary.any():
        return None
    skel = skeletonize_3d(binary)
    if not skel.any():
        return None
    endpoints = _skeleton_endpoints(skel)
    if len(endpoints) < 2:
        return None
    pair = _farthest_endpoint_pair(endpoints)
    if pair is None:
        return None
    start, end = pair
    path = _order_skeleton_path(skel, start, end)
    if path is None or len(path) < 4:
        return None
    world = _voxels_to_world(path, affine)
    ctrl = _downsample(world, max_ctrl=max_ctrl)
    if len(ctrl) < 4:
        ctrl = world
    return CenterlineCurve(ctrl, source="canal_mask"), len(path)


def extract_canal_centerline_3d(
    mask: np.ndarray,
    affine: np.ndarray,
    canal_label_ids: Optional[Sequence[int]] = None,
) -> Optional[CenterlineCurve]:
    """
    Extract the mandibular-canal (nerve) centerline from the prediction mask.

    Tries each canal label individually - IAN canals (3/4), mandibular
    incisive canals (43/44) and the lingual canal (45) - because left/right
    canals are disconnected structures; unioning them into one skeleton graph
    breaks the Dijkstra ordering (no path across disconnected components).
    Returns the longest valid centerline found, mapped to world mm through
    ``affine``.
    """
    if canal_label_ids is None:
        canal_label_ids = [
            LEFT_IAN_ID,
            RIGHT_IAN_ID,
            LEFT_INCISIVE_CANAL_ID,
            RIGHT_INCISIVE_CANAL_ID,
            LINGUAL_CANAL_ID,
        ]
    best: Optional[Tuple[CenterlineCurve, int]] = None
    for label_id in canal_label_ids:
        result = _centerline_from_binary(mask == label_id, affine)
        if result is None:
            continue
        if best is None or result[1] > best[1]:
            best = result
    if best is None:
        return None
    return best[0]


def extract_arch_centerline_3d(
    mask: np.ndarray,
    affine: np.ndarray,
    jaw_type: str = "lower",
    ras: bool = False,
) -> Optional[CenterlineCurve]:
    """Extract the dental-arch centerline from teeth + alveolar crest mask.

    Uses the jawbone label (1=lower, 2=upper) plus all tooth labels for the
    jaw, skeletonizes in 3D, orders the two farthest endpoints via Dijkstra,
    fits a B-spline, and orders control points patient-right -> patient-left.

    ``ras`` selects the world convention: patient right is -X in LPS (default,
    SimpleITK/nnU-Net pipeline) and +X in RAS (3D Slicer).
    """
    jaw_id = LOWER_JAW_ID if jaw_type == "lower" else UPPER_JAW_ID
    tooth_ids = MANDIBULAR_TEETH_IDS if jaw_type == "lower" else MAXILLARY_TEETH_IDS
    structure = np.isin(mask, [jaw_id] + list(tooth_ids))
    if not structure.any():
        return None
    skel = skeletonize_3d(structure)
    if not skel.any():
        return None
    endpoints = _skeleton_endpoints(skel)
    if len(endpoints) < 2:
        return None
    pair = _farthest_endpoint_pair(endpoints)
    if pair is None:
        return None
    start, end = pair
    path = _order_skeleton_path(skel, start, end)
    if path is None or len(path) < 4:
        return None
    world = _voxels_to_world(path, affine)
    # Order patient-right -> patient-left.  Right is -X in LPS, +X in RAS.
    right_first = (world[0, 0] > world[-1, 0]) if ras else (world[0, 0] < world[-1, 0])
    if not right_first:
        world = world[::-1]
    ctrl = _downsample(world, max_ctrl=40)
    if len(ctrl) < 4:
        ctrl = world
    return CenterlineCurve(ctrl, source=f"{jaw_type}_arch_mask")


def extract_centerlines_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    jaw_type: str = "lower",
) -> Dict[str, Optional[CenterlineCurve]]:
    """Convenience: extract canal + arch centerlines from one prediction mask."""
    return {
        "canal": extract_canal_centerline_3d(mask, affine),
        "arch": extract_arch_centerline_3d(mask, affine, jaw_type=jaw_type),
    }

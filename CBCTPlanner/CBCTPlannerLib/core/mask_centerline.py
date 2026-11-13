"""
Mask-driven 3D centerline extraction for dental CBCT segmentation masks.
All optional dependencies (scipy, scikit-image) are imported lazily on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from . import ensure_optional_package
from .labels import (
    LEFT_IAN_ID,
    LEFT_INCISIVE_CANAL_ID,
    LINGUAL_CANAL_ID,
    LOWER_JAW_ID,
    MANDIBULAR_TEETH_IDS,
    MAXILLARY_TEETH_IDS,
    RIGHT_IAN_ID,
    RIGHT_INCISIVE_CANAL_ID,
    UPPER_JAW_ID,
)


@dataclass
class CenterlineFrame:
    """Orthonormal frame at one point along a 3D centerline."""

    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    binormal: np.ndarray
    t: float
    arc_length_mm: float


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

    def _build_spline(self) -> None:
        pts = self.control_points
        diffs = np.diff(pts, axis=0)
        chord = np.sqrt((diffs**2).sum(axis=1))
        u = np.zeros(len(pts), dtype=np.float64)
        u[1:] = np.cumsum(chord)
        total = float(u[-1])
        if total <= 1e-6:
            raise ValueError("Degenerate centerline: zero total chord length")
        u_norm = u / total
        self._u_ctrl = u_norm
        self._use_scipy = False
        self.tck = None
        try:
            from scipy import interpolate  # type: ignore

            k = min(3, len(pts) - 1)
            self.tck, _ = interpolate.splprep(
                [pts[:, 0], pts[:, 1], pts[:, 2]],
                u=u_norm,
                s=0.0,
                k=k,
            )
            self._use_scipy = True
        except ImportError:
            pass

        dense = 500
        ud = np.linspace(0.0, 1.0, dense)
        pts_dense = self._eval_raw(ud)
        d_dense = np.diff(pts_dense, axis=0)
        seg = np.sqrt((d_dense**2).sum(axis=1))
        self.cum_arc = np.zeros(dense, dtype=np.float64)
        self.cum_arc[1:] = np.cumsum(seg)
        self.total_arc_length = float(self.cum_arc[-1])
        self.pts_dense = pts_dense
        self.u_dense = ud

    def _eval_raw(self, u_vals: np.ndarray) -> np.ndarray:
        u_arr = np.atleast_1d(np.asarray(u_vals, dtype=np.float64))
        if self._use_scipy and self.tck is not None:
            from scipy import interpolate  # type: ignore

            return np.column_stack(interpolate.splev(u_arr, self.tck))
        return np.column_stack(
            [np.interp(u_arr, self._u_ctrl, self.control_points[:, d]) for d in range(3)]
        )

    def _u_for_t(self, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        arc = t * self.total_arc_length
        return float(np.interp(arc, self.cum_arc, self.u_dense))

    def evaluate_at(self, t: float) -> np.ndarray:
        u = self._u_for_t(t)
        return self._eval_raw(np.array([u]))[0]

    def get_frame(self, t: float) -> CenterlineFrame:
        t = float(np.clip(t, 0.0, 1.0))
        u = self._u_for_t(t)
        pt = self._eval_raw(np.array([u]))[0]
        eps = 1e-3
        u0, u1 = max(0.0, u - eps), min(1.0, u + eps)
        pts_pair = self._eval_raw(np.array([u0, u1]))
        d1 = (pts_pair[1] - pts_pair[0]) / max(1e-6, u1 - u0)

        t_norm = float(np.linalg.norm(d1))
        tangent = d1 / t_norm if t_norm > 1e-8 else np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        z_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normal = np.cross(tangent, z_up)
        n_norm = float(np.linalg.norm(normal))
        normal = normal / n_norm if n_norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

        binormal = np.cross(normal, tangent)
        b_norm = float(np.linalg.norm(binormal))
        binormal = binormal / b_norm if b_norm > 1e-8 else z_up
        if binormal[2] < 0:
            binormal = -binormal

        return CenterlineFrame(
            point=pt,
            tangent=tangent,
            normal=normal,
            binormal=binormal,
            t=t,
            arc_length_mm=t * self.total_arc_length,
        )

    def sample_frames_uniform(self, step_mm: float = 1.0) -> List[CenterlineFrame]:
        n = max(2, int(round(self.total_arc_length / max(0.1, step_mm))))
        return [self.get_frame(i / float(n)) for i in range(n + 1)]

    def find_closest_parameter(self, point_mm: np.ndarray) -> Tuple[float, float]:
        pt = np.asarray(point_mm, dtype=np.float64).reshape(3)
        dists_sq = np.sum((self.pts_dense - pt) ** 2, axis=1)
        best_idx = int(np.argmin(dists_sq))
        best_t = float(self.cum_arc[best_idx] / max(self.total_arc_length, 1e-9))
        dt = 0.02
        fine_t = np.linspace(max(0.0, best_t - dt), min(1.0, best_t + dt), 50)
        arc = fine_t * self.total_arc_length
        u = np.interp(arc, self.cum_arc, self.u_dense)
        fine_pts = self._eval_raw(u)
        fd_sq = np.sum((fine_pts - pt) ** 2, axis=1)
        fi = int(np.argmin(fd_sq))
        return float(fine_t[fi]), float(np.sqrt(fd_sq[fi]))


_SKELETON_OFFSETS = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def _skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    coords = np.argwhere(skel)
    endpoints: List[Tuple[int, int, int]] = []
    for z, y, x in coords:
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
    scipy_sparse = ensure_optional_package("scipy.sparse", "scipy", "skeleton graph traversal")
    scipy_csgraph = ensure_optional_package("scipy.sparse.csgraph", "scipy", "skeleton Dijkstra path")
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
    graph = scipy_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    dists, predecessors = scipy_csgraph.dijkstra(
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
    vox_xyz = path_zyx[:, [2, 1, 0]].astype(np.float64)
    homog = np.hstack([vox_xyz, np.ones((len(vox_xyz), 1), dtype=np.float64)])
    return (affine @ homog.T).T[:, :3]


def _downsample(points: np.ndarray, max_ctrl: int) -> np.ndarray:
    if len(points) <= max_ctrl:
        return points
    step = max(1, len(points) // max_ctrl)
    return points[::step]


def _centerline_from_binary(
    binary: np.ndarray,
    affine: np.ndarray,
    max_ctrl: int = 60,
) -> Optional[Tuple[CenterlineCurve, int]]:
    if not binary.any():
        return None
    skimage_morph = ensure_optional_package(
        "skimage.morphology", "scikit-image", "3D mask skeletonization"
    )
    skel = skimage_morph.skeletonize(binary)
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
    return best[0] if best is not None else None


def extract_arch_centerline_3d(
    mask: np.ndarray,
    affine: np.ndarray,
    jaw_type: str = "lower",
    ras: bool = True,
) -> Optional[CenterlineCurve]:
    """Extract dental-arch centerline ordered Patient Right -> Patient Left (+X -> -X in RAS)."""
    jaw_id = LOWER_JAW_ID if jaw_type == "lower" else UPPER_JAW_ID
    tooth_ids = MANDIBULAR_TEETH_IDS if jaw_type == "lower" else MAXILLARY_TEETH_IDS
    structure = np.isin(mask, [jaw_id] + list(tooth_ids))
    if not structure.any():
        return None
    skimage_morph = ensure_optional_package(
        "skimage.morphology", "scikit-image", "3D arch skeletonization"
    )
    skel = skimage_morph.skeletonize(structure)
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
    right_first = (world[0, 0] > world[-1, 0]) if ras else (world[0, 0] < world[-1, 0])
    if not right_first:
        world = world[::-1]
    ctrl = _downsample(world, max_ctrl=40)
    if len(ctrl) < 4:
        ctrl = world
    return CenterlineCurve(ctrl, source=f"{jaw_type}_arch_mask")

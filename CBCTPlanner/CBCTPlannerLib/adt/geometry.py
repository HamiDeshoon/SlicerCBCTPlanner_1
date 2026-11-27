"""Vectorized 3D geometry, surface extraction, Kabsch/ICP rigid alignment, and landmark QC.

Consolidates and accelerates utilities from `SlicerAutomatedDentalTools/ADT/ADTLib/geometry.py`,
`ASO_CBCT_utils`, and `ShapeAXI/post_process.py` using NumPy SVD, SciPy `cKDTree`, `skimage`,
and optional VTK without Python per-vertex loops or external Conda/WSL2 requirements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("CBCTPlanner.adt.geometry")


def vtk_matrix_to_numpy(matrix: Any) -> np.ndarray:
    """Copy a 4x4 `vtkMatrix4x4` into a `(4, 4)` float64 NumPy array."""
    out = np.eye(4, dtype=np.float64)
    for r in range(4):
        for c in range(4):
            out[r, c] = float(matrix.GetElement(r, c))
    return out


def read_surface(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Read `.vtk`, `.vtp`, `.stl`, `.ply`, or `.obj` surface into `(verts, faces, polydata)`."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Surface file not found: {src}")
    ext = src.suffix.lower()
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

    if ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".stl":
        reader = vtk.vtkSTLReader()
    elif ext == ".ply":
        reader = vtk.vtkPLYReader()
    elif ext == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise ValueError(f"Unsupported surface extension: {ext}")
    reader.SetFileName(str(src))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64), poly

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    tri_poly = tri.GetOutput()
    verts = vtk_to_numpy(tri_poly.GetPoints().GetData()).astype(np.float64)
    polys = tri_poly.GetPolys()
    raw = vtk_to_numpy(polys.GetData()) if polys is not None else np.empty(0, dtype=np.int64)
    n_cells = int(tri_poly.GetNumberOfCells())
    faces = (
        raw.reshape(n_cells, 4)[:, 1:4].astype(np.int64)
        if n_cells > 0 and raw.size == n_cells * 4
        else np.empty((0, 3), dtype=np.int64)
    )
    return verts, faces, tri_poly


def write_surface_with_scalars(
    out_path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    scalar_arrays: Optional[Dict[str, np.ndarray]] = None,
    polydata: Optional[Any] = None,
    *,
    binary: bool = False,
) -> Path:
    """Write a 3D surface mesh (`.vtk` or `.vtp`) with optional per-vertex scalar arrays (e.g. Grad-CAM/Saliency)."""
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import numpy_to_vtk  # type: ignore

        if polydata is None:
            polydata = numpy_mesh_to_vtk_polydata(verts, faces)
        if scalar_arrays:
            pd = polydata.GetPointData()
            first_name: Optional[str] = None
            for name, vals in scalar_arrays.items():
                arr_np = np.ascontiguousarray(np.asarray(vals, dtype=np.float32))
                if arr_np.shape[0] != polydata.GetNumberOfPoints():
                    continue
                vtk_arr = numpy_to_vtk(arr_np, deep=True)
                vtk_arr.SetName(str(name))
                pd.AddArray(vtk_arr)
                if first_name is None:
                    first_name = str(name)
            if first_name is not None:
                pd.SetActiveScalars(first_name)

        writer = vtk.vtkXMLPolyDataWriter() if dst.suffix.lower() == ".vtp" else vtk.vtkPolyDataWriter()
        writer.SetFileName(str(dst))
        writer.SetInputData(polydata)
        if binary and hasattr(writer, "SetFileTypeToBinary"):
            writer.SetFileTypeToBinary()
        writer.Write()
        return dst
    except ImportError:
        logger.debug("vtk unavailable; writing ASCII legacy VTK POLYDATA directly", exc_info=True)
        v_arr = np.asarray(verts, dtype=np.float64)
        f_arr = np.asarray(faces, dtype=np.int64)
        lines = [
            "# vtk DataFile Version 4.2",
            "CBCTPlanner ADT surface output",
            "ASCII",
            "DATASET POLYDATA",
            f"POINTS {len(v_arr)} float",
        ]
        lines.extend(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}" for p in v_arr)
        lines.append(f"POLYGONS {len(f_arr)} {len(f_arr) * 4}")
        lines.extend(f"3 {int(f[0])} {int(f[1])} {int(f[2])}" for f in f_arr)
        if scalar_arrays:
            lines.append(f"POINT_DATA {len(v_arr)}")
            for name, vals in scalar_arrays.items():
                arr_np = np.asarray(vals, dtype=np.float32).ravel()
                if len(arr_np) != len(v_arr):
                    continue
                safe_name = str(name).replace(" ", "_") or "Pred"
                lines.append(f"SCALARS {safe_name} float 1")
                lines.append("LOOKUP_TABLE default")
                lines.extend(f"{float(val):.6f}" for val in arr_np)
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dst


def numpy_mesh_to_vtk_polydata(verts: np.ndarray, faces: np.ndarray) -> Any:
    """Convert `(V, 3)` float vertices and `(F, 3)` triangle indices into a `vtkPolyData`."""
    import vtk  # type: ignore
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray  # type: ignore

    poly = vtk.vtkPolyData()
    pts_vtk = vtk.vtkPoints()
    pts_vtk.SetData(numpy_to_vtk(np.ascontiguousarray(np.asarray(verts, dtype=np.float64)), deep=True))
    poly.SetPoints(pts_vtk)

    if len(faces) > 0:
        f_arr = np.asarray(faces, dtype=np.int64)
        cells = np.empty((len(f_arr), 4), dtype=np.int64)
        cells[:, 0] = 3
        cells[:, 1:4] = f_arr
        cell_arr = vtk.vtkCellArray()
        id_arr = numpy_to_vtkIdTypeArray(np.ascontiguousarray(cells.ravel()), deep=True)
        if hasattr(cell_arr, "SetData") and hasattr(id_arr, "GetNumberOfTuples"):
            conn_arr = numpy_to_vtkIdTypeArray(np.ascontiguousarray(f_arr.ravel()), deep=True)
            cell_arr.SetData(3, conn_arr)
        else:
            cell_arr.SetCells(len(f_arr), id_arr)
        poly.SetPolys(cell_arr)
    return poly


def extract_isosurface_from_label(
    mask_zyx: np.ndarray,
    affine: np.ndarray,
    label_ids: Sequence[int],
    *,
    step_size: int = 1,
    max_target_faces: int = 25000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract oriented `(verts_ras, faces)` surface from binary/multi-label 3D mask using C-level Marching Cubes."""
    binary = np.isin(mask_zyx, list(label_ids))
    if not np.any(binary):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    zyx = np.where(binary)
    z0 = max(0, int(zyx[0].min()) - 1)
    z1 = min(mask_zyx.shape[0], int(zyx[0].max()) + 2)
    y0 = max(0, int(zyx[1].min()) - 1)
    y1 = min(mask_zyx.shape[1], int(zyx[1].max()) + 2)
    x0 = max(0, int(zyx[2].min()) - 1)
    x1 = min(mask_zyx.shape[2], int(zyx[2].max()) + 2)
    sub = np.pad(binary[z0:z1, y0:y1, x0:x1].astype(np.float32), 1, mode="constant", constant_values=0.0)

    verts_zyx = np.empty((0, 3), dtype=np.float64)
    faces = np.empty((0, 3), dtype=np.int64)
    try:
        from skimage.measure import marching_cubes  # type: ignore

        verts_zyx, faces, _normals, _vals = marching_cubes(sub, level=0.5, step_size=max(1, int(step_size)))
        verts_zyx = verts_zyx - 1.0
        verts_ijk = np.column_stack(
            [
                verts_zyx[:, 2] + float(x0),
                verts_zyx[:, 1] + float(y0),
                verts_zyx[:, 0] + float(z0),
                np.ones(len(verts_zyx), dtype=np.float64),
            ]
        )
        verts_ras = (np.asarray(affine, dtype=np.float64) @ verts_ijk.T).T[:, :3]
        faces = np.asarray(faces, dtype=np.int64)
    except Exception:
        logger.debug(
            "skimage marching_cubes unavailable; using VTK/NumPy surface extraction fallback", exc_info=True
        )
        try:
            import vtk  # type: ignore
            from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # type: ignore

            img_vtk = vtk.vtkImageData()
            nz, ny, nx = sub.shape
            img_vtk.SetDimensions(int(nx), int(ny), int(nz))
            img_vtk.SetSpacing(1.0, 1.0, 1.0)
            arr_u8 = (sub > 0.5).astype(np.uint8, order="C")
            vtk_arr = numpy_to_vtk(arr_u8.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            img_vtk.GetPointData().SetScalars(vtk_arr)
            mc = vtk.vtkDiscreteMarchingCubes()
            mc.SetInputData(img_vtk)
            mc.SetValue(0, 1)
            mc.Update()
            poly = mc.GetOutput()
            if poly.GetNumberOfPoints() >= 3 and poly.GetNumberOfCells() >= 1:
                pts_xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64) - 1.0
                verts_ijk = np.column_stack(
                    [
                        pts_xyz[:, 0] + float(x0),
                        pts_xyz[:, 1] + float(y0),
                        pts_xyz[:, 2] + float(z0),
                        np.ones(len(pts_xyz), dtype=np.float64),
                    ]
                )
                verts_ras = (np.asarray(affine, dtype=np.float64) @ verts_ijk.T).T[:, :3]
                raw_cells = vtk_to_numpy(poly.GetPolys().GetData())
                nc = int(poly.GetNumberOfCells())
                if raw_cells.size == nc * 4:
                    faces = raw_cells.reshape(nc, 4)[:, 1:4].astype(np.int64)
        except Exception:
            logger.debug(
                "VTK marching cubes fallback unavailable; using pure-NumPy voxel surface builder",
                exc_info=True,
            )

        if faces.size == 0:
            ds = max(1, int(max(sub.shape) // 96))
            sub_b = sub[::ds, ::ds, ::ds] > 0.5
            pad = np.pad(sub_b, 1, mode="constant", constant_values=False)
            occ_z, occ_y, occ_x = np.where(sub_b)
            quad_list: List[np.ndarray] = []
            # Emit boundary quad corners in sub_b index units for each of the 6 face directions
            dirs = (
                ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
                ((+1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
                ((0, -1, 0), ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0))),
                ((0, +1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
                ((0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
                ((0, 0, +1), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
            )
            pz, py, px = occ_z + 1, occ_y + 1, occ_x + 1
            for (dz, dy, dx), corners in dirs:
                exposed = ~pad[pz + dz, py + dy, px + dx]
                if not np.any(exposed):
                    continue
                cz = occ_z[exposed][:, None]
                cy = occ_y[exposed][:, None]
                cx = occ_x[exposed][:, None]
                c_arr = np.asarray(corners, dtype=np.int64)
                quad_zyx = np.stack([cz + c_arr[:, 0], cy + c_arr[:, 1], cx + c_arr[:, 2]], axis=-1)
                quad_list.append(quad_zyx)
            if quad_list:
                all_quads = np.concatenate(quad_list, axis=0)  # (Q, 4, 3) in sub_b grid
                flat_pts = all_quads.reshape(-1, 3)
                uniq_pts, inv = np.unique(flat_pts, axis=0, return_inverse=True)
                q_idx = inv.reshape(-1, 4)
                faces = np.vstack([q_idx[:, [0, 1, 2]], q_idx[:, [0, 2, 3]]]).astype(np.int64)
                pts_sub = uniq_pts.astype(np.float64) * float(ds) - 1.0
                verts_ijk = np.column_stack(
                    [
                        pts_sub[:, 2] + float(x0),
                        pts_sub[:, 1] + float(y0),
                        pts_sub[:, 0] + float(z0),
                        np.ones(len(pts_sub), dtype=np.float64),
                    ]
                )
                verts_ras = (np.asarray(affine, dtype=np.float64) @ verts_ijk.T).T[:, :3]
            else:
                step = max(1, len(zyx[0]) // 3000)
                vox = np.column_stack(
                    [
                        zyx[2][::step],
                        zyx[1][::step],
                        zyx[0][::step],
                        np.ones(len(zyx[0][::step]), dtype=np.float64),
                    ]
                )
                verts_ras = (np.asarray(affine, dtype=np.float64) @ vox.T).T[:, :3]
                return verts_ras, faces

    # Optional uniform face decimation for massive high-res CBCT crowns/condyles
    if max_target_faces > 0 and len(faces) > max_target_faces * 2:
        try:
            import vtk  # type: ignore
            from vtk.util.numpy_support import vtk_to_numpy  # type: ignore

            poly = numpy_mesh_to_vtk_polydata(verts_ras, faces)
            dec = vtk.vtkDecimatePro()
            dec.SetInputData(poly)
            dec.SetTargetReduction(max(0.0, min(0.9, 1.0 - float(max_target_faces) / float(len(faces)))))
            dec.PreserveTopologyOn()
            dec.Update()
            out_p = dec.GetOutput()
            if out_p.GetNumberOfPoints() > 0 and out_p.GetNumberOfCells() > 0:
                verts_ras = vtk_to_numpy(out_p.GetPoints().GetData()).astype(np.float64)
                raw = vtk_to_numpy(out_p.GetPolys().GetData())
                nc = int(out_p.GetNumberOfCells())
                if raw.size == nc * 4:
                    faces = raw.reshape(nc, 4)[:, 1:4].astype(np.int64)
        except Exception:
            logger.debug("VTK mesh decimation skipped cleanly", exc_info=True)

    return verts_ras, faces


def vectorized_median_filter_surface(
    verts_ras: np.ndarray,
    scalars: np.ndarray,
    *,
    k_neighbors: int = 8,
    n_iters: int = 1,
) -> np.ndarray:
    """Fast vectorized k-NN surface scalar median filter replacing slow O(V^2) VTK Python loops."""
    pts = np.asarray(verts_ras, dtype=np.float64)
    vals = np.asarray(scalars, dtype=np.float64).copy()
    n = len(pts)
    if n < 4 or vals.shape[0] != n:
        return vals
    try:
        from scipy.spatial import cKDTree  # type: ignore

        k = min(max(2, int(k_neighbors)), n)
        tree = cKDTree(pts)
        _dists, nbrs = tree.query(pts, k=k)
        for _ in range(max(1, int(n_iters))):
            vals = np.median(vals[nbrs], axis=1)
        return vals
    except Exception:
        return vals


def mesh_volume_and_area(verts: np.ndarray, faces: np.ndarray) -> Tuple[float, float]:
    """Compute closed triangle mesh `(volume_mm3, surface_area_mm2)` via divergence theorem."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(v) < 4 or len(f) == 0:
        return 0.0, 0.0
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * float(np.sum(np.linalg.norm(cross, axis=1)))
    signed_vol = float(np.sum(v0 * np.cross(v1, v2))) / 6.0
    return abs(signed_vol), area


def compute_vertex_curvature_saliency(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Estimate normalized local geometric saliency (dihedral normal variation) per vertex in `[0, 1]`."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n_v = len(v)
    if n_v < 4 or len(f) == 0:
        return np.zeros(n_v, dtype=np.float32)

    tri_cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    tri_norm_len = np.maximum(1e-8, np.linalg.norm(tri_cross, axis=1, keepdims=True))
    face_normals = tri_cross / tri_norm_len

    # Accumulate area-weighted vertex normals
    v_norm = np.zeros((n_v, 3), dtype=np.float64)
    for col in range(3):
        np.add.at(v_norm, f[:, col], tri_cross)
    v_len = np.maximum(1e-8, np.linalg.norm(v_norm, axis=1, keepdims=True))
    v_norm /= v_len

    # Deviation of adjacent triangle normals from smooth vertex normal captures osteophytes / notches / clefts
    var_acc = np.zeros(n_v, dtype=np.float64)
    deg = np.zeros(n_v, dtype=np.float64)
    for col in range(3):
        idx = f[:, col]
        dot = np.clip(np.sum(v_norm[idx] * face_normals, axis=1), -1.0, 1.0)
        dev = 1.0 - np.abs(dot)
        np.add.at(var_acc, idx, dev)
        np.add.at(deg, idx, 1.0)
    raw = var_acc / np.maximum(1.0, deg)
    p99 = float(np.percentile(raw, 99)) if raw.size > 0 else 1.0
    if p99 > 1e-8:
        raw = np.clip(raw / p99, 0.0, 1.0)
    raw = vectorized_median_filter_surface(v, raw, k_neighbors=8, n_iters=1)
    return raw.astype(np.float32)


def kabsch_rigid_alignment(source_pts: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """Compute exact 4x4 proper rigid (SO(3) + translation) matrix mapping `source_pts -> target_pts`."""
    src = np.asarray(source_pts, dtype=np.float64)
    tgt = np.asarray(target_pts, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[0] < 3 or src.shape[1] != 3:
        raise ValueError("kabsch_rigid_alignment requires matching (N>=3, 3) point arrays")

    c_src = src.mean(axis=0)
    c_tgt = tgt.mean(axis=0)
    H = (src - c_src).T @ (tgt - c_tgt)
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if float(np.linalg.det(R)) < 0.0:
        Vt_corr = Vt.copy()
        Vt_corr[-1, :] *= -1.0
        R = Vt_corr.T @ U.T
    t = c_tgt - R @ c_src
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def filter_outlier_landmarks(
    source_landmarks: Dict[str, Sequence[float]],
    target_landmarks: Dict[str, Sequence[float]],
    *,
    max_rel_dist_dev: float = 0.35,
    max_dir_cos_dev: float = 0.55,
    min_keep: int = 3,
) -> Tuple[List[str], List[str]]:
    """Identify geometrically inconsistent landmarks using pairwise distance/direction consensus (`ASO_CBCT`).

    Returns `(kept_labels, rejected_labels)`.
    """
    common = [k for k in source_landmarks if k in target_landmarks]
    if len(common) <= min_keep:
        return common, []

    src = np.asarray([source_landmarks[k] for k in common], dtype=np.float64)
    tgt = np.asarray([target_landmarks[k] for k in common], dtype=np.float64)
    # Remove centroid before comparing relative pair vectors
    src_c = src - src.mean(axis=0)
    tgt_c = tgt - tgt.mean(axis=0)
    n = len(common)
    err_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bad = 0.0
        for j in range(n):
            if i == j:
                continue
            ds = float(np.linalg.norm(src_c[i] - src_c[j]))
            dt = float(np.linalg.norm(tgt_c[i] - tgt_c[j]))
            ref_d = max(5.0, dt)
            if abs(ds - dt) / ref_d > max_rel_dist_dev:
                bad += 1.0
            else:
                vs = (src_c[i] - src_c[j]) / max(1e-6, ds)
                vt = (tgt_c[i] - tgt_c[j]) / max(1e-6, dt)
                if float(np.dot(vs, vt)) < (1.0 - max_dir_cos_dev):
                    bad += 0.5
        err_scores[i] = bad / max(1, n - 1)

    kept: List[str] = []
    rejected: List[str] = []
    order = np.argsort(err_scores)
    for idx in order:
        if len(kept) < min_keep or err_scores[idx] < 0.45:
            kept.append(common[idx])
        else:
            rejected.append(common[idx])
    return kept, rejected


def vtk_icp_refine(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_iterations: int = 200,
) -> Optional[np.ndarray]:
    """Optional VTK rigid-body ICP refinement (`vtkIterativeClosestPointTransform`); returns 4x4 matrix."""
    try:
        import vtk  # type: ignore

        src = np.asarray(source_pts, dtype=np.float64)
        tgt = np.asarray(target_pts, dtype=np.float64)
        if len(src) < 3 or len(tgt) < 3:
            return None

        def _to_poly(pts: np.ndarray) -> Any:
            p = vtk.vtkPoints()
            for x, y, z in pts:
                p.InsertNextPoint(float(x), float(y), float(z))
            pd = vtk.vtkPolyData()
            pd.SetPoints(p)
            return pd

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(_to_poly(src))
        icp.SetTarget(_to_poly(tgt))
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(int(max_iterations))
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()
        return vtk_matrix_to_numpy(icp.GetMatrix())
    except Exception:
        logger.debug("VTK ICP unavailable cleanly", exc_info=True)
        return None

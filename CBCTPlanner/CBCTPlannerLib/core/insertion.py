"""Mask-space insertion axis & undercut evaluation for dental surgical guides.

Operates directly on 3D binary/label masks in `(Z, Y, X)` order using an affine
mapping voxel `(x, y, z, 1)` -> RAS mm. Uses pure NumPy integer ray sweeps along
the coronal draw direction (`-axis_apical_ras`) with zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

GEOMETRIC_DISCLAIMER = "geometric obstruction, not a clinical contraindication"
MAX_CROPPED_GRID_DIM = 128


@dataclass
class UndercutReport:
    """Result of evaluating path-of-insertion undercuts over a mask surface."""

    axis_apical_ras: Tuple[float, float, float]
    blocked_fraction: float
    blocked_voxel_count: int
    surface_voxel_count: int
    worst_depth_mm: float
    blocked_centroid_ras: Optional[Tuple[float, float, float]]
    is_insertable: bool
    voxel_mm: float
    warnings: List[str] = field(default_factory=list)


def _normalize_vec(
    vec: Sequence[float], default: Tuple[float, float, float] = (0.0, 0.0, -1.0)
) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-8:
        arr = np.asarray(default, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
    return arr / max(1e-8, norm)


def _crop_and_downsample_mask(
    mask_zyx: np.ndarray,
    affine: np.ndarray,
    *,
    label_ids: Optional[Sequence[int]],
    center_ras: Optional[Sequence[float]],
    roi_mm: float,
    max_grid_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build cropped binary `(Z, Y, X)` mask bounded by `max_grid_dim` and adjusted affine."""
    mask = np.asarray(mask_zyx)
    aff = np.asarray(affine, dtype=np.float64)
    if mask.ndim != 3 or mask.size == 0:
        return np.zeros((0, 0, 0), dtype=bool), aff, 1.0

    sp_xyz = np.maximum(np.linalg.norm(aff[:3, :3], axis=0), 1e-6)
    base_vox_mm = float(np.mean(sp_xyz))

    nz, ny, nx = mask.shape
    if center_ras is not None:
        c_ijk = (np.linalg.inv(aff) @ np.append(np.asarray(center_ras, dtype=np.float64).reshape(3), 1.0))[:3]
        half_xyz = np.maximum(2.0, np.ceil(float(roi_mm) / sp_xyz)).astype(int)
        x0, x1 = (
            max(0, int(math.floor(c_ijk[0])) - int(half_xyz[0])),
            min(nx, int(math.ceil(c_ijk[0])) + int(half_xyz[0]) + 1),
        )
        y0, y1 = (
            max(0, int(math.floor(c_ijk[1])) - int(half_xyz[1])),
            min(ny, int(math.ceil(c_ijk[1])) + int(half_xyz[1]) + 1),
        )
        z0, z1 = (
            max(0, int(math.floor(c_ijk[2])) - int(half_xyz[2])),
            min(nz, int(math.ceil(c_ijk[2])) + int(half_xyz[2]) + 1),
        )
        sub_raw = mask[z0:z1, y0:y1, x0:x1]
        occ = np.isin(sub_raw, list(label_ids)) if label_ids is not None else (sub_raw > 0)
    else:
        full_occ = np.isin(mask, list(label_ids)) if label_ids is not None else (mask > 0)
        if not np.any(full_occ):
            return np.zeros((0, 0, 0), dtype=bool), aff, base_vox_mm
        zz, yy, xx = np.where(full_occ)
        pad = 2
        z0, z1 = max(0, int(zz.min()) - pad), min(nz, int(zz.max()) + pad + 1)
        y0, y1 = max(0, int(yy.min()) - pad), min(ny, int(yy.max()) + pad + 1)
        x0, x1 = max(0, int(xx.min()) - pad), min(nx, int(xx.max()) + pad + 1)
        occ = full_occ[z0:z1, y0:y1, x0:x1]

    if not np.any(occ):
        return np.zeros((0, 0, 0), dtype=bool), aff, base_vox_mm

    # Update origin for the crop offset
    crop_aff = aff.copy()
    offset_ijk = np.array([float(x0), float(y0), float(z0), 1.0], dtype=np.float64)
    crop_aff[:3, 3] = (aff @ offset_ijk)[:3]

    max_dim = max(occ.shape)
    stride = max(1, int(math.ceil(max_dim / float(max(8, int(max_grid_dim))))))
    if stride > 1:
        occ = occ[::stride, ::stride, ::stride]
        crop_aff[:3, :3] = crop_aff[:3, :3] * float(stride)
        base_vox_mm *= float(stride)

    return occ, crop_aff, base_vox_mm


def compute_blocked_surface(
    mask_zyx: np.ndarray,
    affine: np.ndarray,
    axis_apical_ras: Sequence[float],
    *,
    label_ids: Optional[Sequence[int]] = None,
    center_ras: Optional[Sequence[float]] = None,
    roi_mm: float = 25.0,
    coronal_band_mm: Optional[float] = None,
) -> UndercutReport:
    """Evaluate surface undercuts blocking coronal-to-apical seating along `axis_apical_ras`.

    Checks every guide-contacting surface voxel's exposed air neighbors by marching toward
    the coronal side (`-axis_apical_ras`). Strictly apical underside faces (`dot(n, -u) < -0.55`)
    are excluded so flat/bottom bases do not trigger false undercuts.
    """
    u_apical = _normalize_vec(axis_apical_ras)
    axis_tuple = (round(float(u_apical[0]), 6), round(float(u_apical[1]), 6), round(float(u_apical[2]), 6))
    occ, sub_aff, voxel_mm = _crop_and_downsample_mask(
        mask_zyx,
        affine,
        label_ids=label_ids,
        center_ras=center_ras,
        roi_mm=roi_mm,
        max_grid_dim=MAX_CROPPED_GRID_DIM,
    )
    if occ.size < 2 or int(np.count_nonzero(occ)) < 2:
        return UndercutReport(
            axis_apical_ras=axis_tuple,
            blocked_fraction=0.0,
            blocked_voxel_count=0,
            surface_voxel_count=0,
            worst_depth_mm=0.0,
            blocked_centroid_ras=None,
            is_insertable=True,
            voxel_mm=round(voxel_mm, 4),
            warnings=[
                "empty or single-voxel mask; no surface evaluated",
                f"stair-step uncertainty ±{voxel_mm:.2f} mm (1 voxel)",
                GEOMETRIC_DISCLAIMER,
            ],
        )

    # Convert coronal draw vector (-u_apical) into voxel (X, Y, Z) -> (Z, Y, X)
    inv_linear = np.linalg.inv(sub_aff[:3, :3])
    draw_ijk = inv_linear @ (-u_apical)
    draw_zyx = np.array([draw_ijk[2], draw_ijk[1], draw_ijk[0]], dtype=np.float64)
    draw_norm = float(np.linalg.norm(draw_zyx))
    if draw_norm <= 1e-8:
        draw_zyx = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        draw_zyx = draw_zyx / draw_norm

    # Optional coronal depth window relative to the coronal-most occupied voxel
    if coronal_band_mm is not None and float(coronal_band_mm) > 0.0:
        sp_zyx = np.array(
            [np.linalg.norm(sub_aff[:3, 2]), np.linalg.norm(sub_aff[:3, 1]), np.linalg.norm(sub_aff[:3, 0])]
        )
        zz, yy, xx = np.where(occ)
        proj = zz * draw_zyx[0] * sp_zyx[0] + yy * draw_zyx[1] * sp_zyx[1] + xx * draw_zyx[2] * sp_zyx[2]
        top_proj = float(np.max(proj))
        keep_mask = np.zeros_like(occ, dtype=bool)
        keep_idx = (top_proj - proj) <= float(coronal_band_mm)
        keep_mask[zz[keep_idx], yy[keep_idx], xx[keep_idx]] = True
        occ = occ & keep_mask
        if int(np.count_nonzero(occ)) < 2:
            return UndercutReport(
                axis_apical_ras=axis_tuple,
                blocked_fraction=0.0,
                blocked_voxel_count=0,
                surface_voxel_count=0,
                worst_depth_mm=0.0,
                blocked_centroid_ras=None,
                is_insertable=True,
                voxel_mm=round(voxel_mm, 4),
                warnings=[
                    f"stair-step uncertainty ±{voxel_mm:.2f} mm (1 voxel)",
                    GEOMETRIC_DISCLAIMER,
                ],
            )

    pad = np.pad(occ, 1, mode="constant", constant_values=False)
    nz, ny, nx = occ.shape

    eroded = (
        pad[2:, 1:-1, 1:-1]
        & pad[:-2, 1:-1, 1:-1]
        & pad[1:-1, 2:, 1:-1]
        & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 1:-1, 2:]
        & pad[1:-1, 1:-1, :-2]
    )
    surface = occ & ~eroded
    surf_count = int(np.count_nonzero(surface))
    if surf_count == 0:
        return UndercutReport(
            axis_apical_ras=axis_tuple,
            blocked_fraction=0.0,
            blocked_voxel_count=0,
            surface_voxel_count=0,
            worst_depth_mm=0.0,
            blocked_centroid_ras=None,
            is_insertable=True,
            voxel_mm=round(voxel_mm, 4),
            warnings=[
                f"stair-step uncertainty ±{voxel_mm:.2f} mm (1 voxel)",
                GEOMETRIC_DISCLAIMER,
            ],
        )

    # Central-difference outward normals (-grad(occ)) at surface voxels avoid cardinal-face
    # rounding aliasing on curved oblique walls (e.g. rotated cylinders and tooth crowns).
    pad_f = pad.astype(np.float32)
    gz = pad_f[:-2, 1:-1, 1:-1] - pad_f[2:, 1:-1, 1:-1]
    gy = pad_f[1:-1, :-2, 1:-1] - pad_f[1:-1, 2:, 1:-1]
    gx = pad_f[1:-1, 1:-1, :-2] - pad_f[1:-1, 1:-1, 2:]

    sz_all, sy_all, sx_all = np.where(surface)
    n_zyx = np.column_stack(
        [gz[sz_all, sy_all, sx_all], gy[sz_all, sy_all, sx_all], gx[sz_all, sy_all, sx_all]]
    ).astype(np.float64)
    n_norms = np.linalg.norm(n_zyx, axis=1)
    valid_n = n_norms > 1e-6
    sz_all, sy_all, sx_all, n_zyx = (
        sz_all[valid_n],
        sy_all[valid_n],
        sx_all[valid_n],
        n_zyx[valid_n] / n_norms[valid_n][:, None],
    )

    linear = sub_aff[:3, :3]
    n_ijk = np.column_stack([n_zyx[:, 2], n_zyx[:, 1], n_zyx[:, 0]])
    n_ras = (linear @ n_ijk.T).T
    n_ras_norm = np.maximum(np.linalg.norm(n_ras, axis=1, keepdims=True), 1e-8)
    n_ras = n_ras / n_ras_norm
    cos_draw = n_ras @ (-u_apical)

    # Exclude strictly apical bottom faces (pointing opposite coronal draw direction)
    keep_surf = cos_draw >= -0.55
    if not np.any(keep_surf):
        return UndercutReport(
            axis_apical_ras=axis_tuple,
            blocked_fraction=0.0,
            blocked_voxel_count=0,
            surface_voxel_count=surf_count,
            worst_depth_mm=0.0,
            blocked_centroid_ras=None,
            is_insertable=True,
            voxel_mm=round(voxel_mm, 4),
            warnings=[
                f"stair-step uncertainty ±{voxel_mm:.2f} mm (1 voxel)",
                GEOMETRIC_DISCLAIMER,
            ],
        )

    sz_v = sz_all[keep_surf]
    sy_v = sy_all[keep_surf]
    sx_v = sx_all[keep_surf]
    n_v = n_zyx[keep_surf]

    # Scale step along dominant voxel axis (Bresenham-style step per iteration)
    dom = float(np.max(np.abs(draw_zyx)))
    step_zyx = draw_zyx / max(1e-6, dom)
    step_mm = float(
        np.linalg.norm(linear @ np.array([step_zyx[2], step_zyx[1], step_zyx[0]], dtype=np.float64))
    )
    max_steps = int(math.ceil(math.hypot(nz, math.hypot(ny, nx)))) + 1

    # Raymarch from 1.5 voxels outside the true outward surface normal along +draw vector.
    ez = sz_v.astype(np.float64) + 1.5 * n_v[:, 0]
    ey = sy_v.astype(np.float64) + 1.5 * n_v[:, 1]
    ex = sx_v.astype(np.float64) + 1.5 * n_v[:, 2]
    hit = np.zeros(sz_v.size, dtype=bool)
    max_depth_steps = 0
    for m in range(1, max_steps):
        cz = np.floor(ez + float(m) * step_zyx[0] + 0.5).astype(int)
        cy = np.floor(ey + float(m) * step_zyx[1] + 0.5).astype(int)
        cx = np.floor(ex + float(m) * step_zyx[2] + 0.5).astype(int)
        in_b = (cz >= 0) & (cz < nz) & (cy >= 0) & (cy < ny) & (cx >= 0) & (cx < nx)
        if not np.any(in_b):
            break
        idx_in = np.flatnonzero(in_b)
        occ_hit = occ[cz[idx_in], cy[idx_in], cx[idx_in]]
        if np.any(occ_hit):
            cand = idx_in[occ_hit]
            far_enough = (
                (np.abs(cz[cand] - sz_v[cand]) > 1)
                | (np.abs(cy[cand] - sy_v[cand]) > 1)
                | (np.abs(cx[cand] - sx_v[cand]) > 1)
            )
            newly = cand[far_enough]
            if newly.size > 0:
                hit[newly] = True
                if m > max_depth_steps:
                    max_depth_steps = m

    blocked_mask = np.zeros_like(occ, dtype=bool)
    if np.any(hit):
        blocked_mask[sz_v[hit], sy_v[hit], sx_v[hit]] = True

    blocked_count = int(np.count_nonzero(blocked_mask))
    blocked_fraction = float(blocked_count) / float(max(1, surf_count))
    worst_depth_mm = float(max_depth_steps) * step_mm if blocked_count > 0 else 0.0
    threshold = max(0.02, 3.0 / float(max(1, surf_count)))
    is_insertable = bool(blocked_fraction <= threshold)

    centroid_ras: Optional[Tuple[float, float, float]] = None
    if blocked_count > 0:
        bz, by, bx = np.where(blocked_mask)
        mean_ijk = np.array([float(bx.mean()), float(by.mean()), float(bz.mean()), 1.0], dtype=np.float64)
        c_world = (sub_aff @ mean_ijk)[:3]
        centroid_ras = (round(float(c_world[0]), 3), round(float(c_world[1]), 3), round(float(c_world[2]), 3))

    warnings = [
        f"stair-step uncertainty ±{voxel_mm:.2f} mm (1 voxel)",
        GEOMETRIC_DISCLAIMER,
    ]
    if not is_insertable:
        warnings.insert(
            0,
            f"Seating path blocked over {blocked_fraction * 100.0:.1f}% of surface "
            f"(worst depth {worst_depth_mm:.1f} mm).",
        )

    return UndercutReport(
        axis_apical_ras=axis_tuple,
        blocked_fraction=round(blocked_fraction, 4),
        blocked_voxel_count=blocked_count,
        surface_voxel_count=surf_count,
        worst_depth_mm=round(worst_depth_mm, 3),
        blocked_centroid_ras=centroid_ras,
        is_insertable=is_insertable,
        voxel_mm=round(voxel_mm, 4),
        warnings=warnings,
    )


def _orthonormal_basis(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(u[0])) < 0.8
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    e1 = np.cross(u, ref)
    e1 /= max(1e-8, float(np.linalg.norm(e1)))
    e2 = np.cross(u, e1)
    e2 /= max(1e-8, float(np.linalg.norm(e2)))
    return e1, e2


def suggest_insertion_axis(
    mask_zyx: np.ndarray,
    affine: np.ndarray,
    label_ids: Optional[Sequence[int]],
    *,
    seed_axis_ras: Sequence[float] = (0.0, 0.0, -1.0),
    max_tilt_deg: float = 20.0,
    n_candidates: int = 13,
    center_ras: Optional[Sequence[float]] = None,
    roi_mm: float = 25.0,
) -> Tuple[np.ndarray, UndercutReport, List[Tuple[float, np.ndarray]]]:
    """Search a tilt cone around `seed_axis_ras` for the lowest-undercut draw direction."""
    seed = _normalize_vec(seed_axis_ras)
    e1, e2 = _orthonormal_basis(seed)
    max_tilt = max(1.0, float(max_tilt_deg))
    n_total = max(1, int(n_candidates))

    candidates: List[np.ndarray] = [seed]
    if n_total > 1:
        # Concentric Fibonacci/golden-angle rings plus cardinal extrema up to max_tilt_deg
        ring_count = n_total - 1
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for k in range(ring_count):
            frac = math.sqrt(float(k + 1) / float(ring_count))
            ang_rad = math.radians(max_tilt * frac)
            phi = float(k) * golden_angle
            vec = math.cos(ang_rad) * seed + math.sin(ang_rad) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            candidates.append(_normalize_vec(vec))

    scored: List[Tuple[float, float, float, np.ndarray, UndercutReport]] = []

    def _eval(vec: np.ndarray) -> Tuple[float, float, float, np.ndarray, UndercutReport]:
        rep = compute_blocked_surface(
            mask_zyx,
            affine,
            vec,
            label_ids=label_ids,
            center_ras=center_ras,
            roi_mm=roi_mm,
        )
        tilt_from_seed = float(np.degrees(np.arccos(np.clip(float(np.dot(vec, seed)), -1.0, 1.0))))
        return (rep.blocked_fraction, rep.worst_depth_mm, tilt_from_seed, vec, rep)

    for cand in candidates:
        scored.append(_eval(cand))

    best = min(scored, key=lambda item: (item[0], item[1], item[2]))
    # Local 1-ring refinement within max_tilt around coarse winner when coarse axis is not clear
    if not best[4].is_insertable and n_total >= 5 and max_tilt >= 2.0:
        b_vec = best[3]
        be1, be2 = _orthonormal_basis(b_vec)
        fine_rad = math.radians(min(2.5, max_tilt * 0.15))
        for ang in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            cand_f = _normalize_vec(
                math.cos(fine_rad) * b_vec + math.sin(fine_rad) * (math.cos(ang) * be1 + math.sin(ang) * be2)
            )
            tilt_f = float(np.degrees(np.arccos(np.clip(float(np.dot(cand_f, seed)), -1.0, 1.0))))
            if tilt_f <= max_tilt + 0.5:
                scored.append(_eval(cand_f))
        best = min(scored, key=lambda item: (item[0], item[1], item[2]))

    best_axis, best_report = best[3], best[4]
    ranking = [(s[0], s[3]) for s in sorted(scored, key=lambda item: (item[0], item[1], item[2]))]
    return best_axis, best_report, ranking

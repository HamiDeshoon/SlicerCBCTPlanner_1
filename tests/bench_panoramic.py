"""Benchmark the panoramic slab sampler against the pre-Phase-4 layer loop.

Usage:
    python3 tests/bench_panoramic.py

Prints old/new timings, the measured speed-up, and the maximum array difference.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.PanoramicCurveLogic import PanoramicCurveLogic  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


def make_phantom() -> tuple[CBCTVolume, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape_zyx = (240, 320, 320)
    z = np.arange(shape_zyx[0], dtype=np.float32)[:, None, None]
    img = np.broadcast_to(100.0 + 2.0 * z, shape_zyx).copy()
    vol = CBCTVolume(
        image=img,
        spacing=(0.5, 0.5, 0.5),
        origin=(-80.0, -80.0, -60.0),
        direction=np.eye(3),
        affine=np.array(
            [[0.5, 0.0, 0.0, -80.0], [0.0, 0.5, 0.0, -80.0], [0.0, 0.0, 0.5, -60.0], [0.0, 0.0, 0.0, 1.0]]
        ),
    )
    theta = np.linspace(0.0, np.pi, 220)
    radius = 28.0
    pts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])
    tangents = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
    normals = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    binormals = np.tile(np.array([0.0, 0.0, 1.0]), (len(theta), 1))
    return vol, pts, tangents, normals, binormals


def legacy_sample(
    vol: CBCTVolume, base_surface: np.ndarray, disp_x_cols: np.ndarray, u_offsets: np.ndarray
) -> np.ndarray:
    """Pre-optimization loop: full world->voxel transform and sampler per layer."""
    accum = None
    for u_val in u_offsets:
        layer_pts = base_surface + float(u_val) * disp_x_cols[None, :, :]
        sampled = vol.sample_world_points(layer_pts, order=1, cval=-1000.0)
        accum = sampled.copy() if accum is None else accum + sampled
    assert accum is not None
    return (accum / float(len(u_offsets))).astype(np.float32)


def timed(fn, repeats: int = 2) -> tuple[float, np.ndarray]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    assert result is not None
    return best, result


def main() -> int:
    vol, pts, tangents, normals, binormals = make_phantom()
    logic = PanoramicCurveLogic()
    geometry = logic._build_slab_geometry(
        pts_arch=pts,
        normals=normals,
        binormals=binormals,
        slab_thickness_mm=25.0,
        height_mm=60.0,
        spacing_mm=0.5,
        buccolingual_offset_mm=0.0,
        flip_vertical=False,
    )
    base_surface, disp_x_cols, u_offsets, _n_layers, _dim_y, _dim_z = geometry

    old_s, old_arr = timed(lambda: legacy_sample(vol, base_surface, disp_x_cols, u_offsets))
    new_s, new_arr = timed(
        lambda: logic.sample_panoramic_slab_array(
            vol,
            pts,
            tangents,
            normals,
            binormals,
            slab_thickness_mm=25.0,
            height_mm=60.0,
            spacing_mm=0.5,
            projection_mode="Mean",
        )
    )
    max_diff = float(np.max(np.abs(old_arr.astype(np.float64) - new_arr.astype(np.float64))))
    print(f"shape={new_arr.shape} layers={len(u_offsets)}")
    print(f"old={old_s * 1000:.1f} ms")
    print(f"new={new_s * 1000:.1f} ms")
    print(f"speedup={old_s / max(new_s, 1e-9):.2f}x")
    print(f"max_diff={max_diff:.6f} HU")
    return 0 if max_diff <= 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())

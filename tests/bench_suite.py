"""Reproducible multi-pipeline performance & memory benchmark suite (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = REPO_ROOT / "CBCTPlanner"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.CanalTracer import CanalTracerLogic  # noqa: E402
from CBCTPlannerLib.EndodonticEngine import EndodonticEngine  # noqa: E402
from CBCTPlannerLib.ImplantPlanner import ImplantPlannerLogic  # noqa: E402
from CBCTPlannerLib.PanoramicCurveLogic import PanoramicCurveLogic  # noqa: E402
from CBCTPlannerLib.PrintLayoutEngine import PrintLayoutEngine  # noqa: E402
from CBCTPlannerLib.core.arch_detector import DentalArch, fit_arch_from_label_mask  # noqa: E402
from CBCTPlannerLib.core.measurements import MeasurementEngine  # noqa: E402
from CBCTPlannerLib.core.volume_loader import CBCTVolume  # noqa: E402


def _make_synthetic_cbct_case(
    shape: Tuple[int, int, int] = (96, 160, 160),
    spacing_mm: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tuple[CBCTVolume, np.ndarray]:
    nz, ny, nx = shape
    sz, sy, sx = spacing_mm
    aff = np.diag([sx, sy, sz, 1.0]).astype(np.float64)
    img = np.full(shape, -850.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.int32)

    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    cx, cy = nx * 0.5, ny * 0.65
    rx, ry = 32.0 / sx, 38.0 / sy
    u = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    in_arch = (u >= 0.72) & (u <= 1.28) & (yy <= cy + 6.0 / sy)

    mandible = in_arch & (zz >= int(0.18 * nz)) & (zz <= int(0.48 * nz))
    mask[mandible] = 1
    img[mandible] = 650.0

    # Synthetic lower IAN canal
    canal = (u >= 0.94) & (u <= 1.06) & (yy <= cy) & (zz >= int(0.24 * nz)) & (zz <= int(0.29 * nz))
    mask[canal] = 3
    img[canal] = -100.0

    # 8 synthetic lower teeth along arc
    angles = np.linspace(0.2, np.pi - 0.2, 8)
    fdis = [46, 45, 44, 41, 31, 34, 35, 36]
    pulp_fdis = [146, 145, 144, 141, 131, 134, 135, 136]
    for ang, fdi, pid in zip(angles, fdis, pulp_fdis):
        tx = cx + rx * np.cos(ang)
        ty = cy - ry * np.sin(ang)
        r_pix = 2.8 / sx
        d2 = (xx - tx) ** 2 + (yy - ty) ** 2
        tooth = (d2 <= r_pix**2) & (zz >= int(0.40 * nz)) & (zz <= int(0.64 * nz))
        mask[tooth] = fdi
        img[tooth] = 1600.0
        pulp = (d2 <= (0.8 / sx) ** 2) & (zz >= int(0.41 * nz)) & (zz <= int(0.58 * nz))
        mask[pulp] = pid
        img[pulp] = 80.0

    vol = CBCTVolume(img, spacing_mm, (0.0, 0.0, 0.0), np.eye(3), aff)
    return vol, mask


def _measure_call(label: str, fn: Callable[[], Any]) -> Dict[str, Any]:
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "step": label,
        "elapsed_ms": round(dt * 1000.0, 2),
        "peak_kib": round(peak / 1024.0, 1),
        "_res": res,
    }


def run_benchmark() -> Dict[str, Any]:
    vol, mask = _make_synthetic_cbct_case()
    records = []

    rec_arch = _measure_call(
        "arch_fit_lower",
        lambda: fit_arch_from_label_mask(mask, vol.affine, jaw="Mandible", is_5class=False),
    )
    fit = rec_arch.pop("_res")
    records.append(rec_arch)

    arch = DentalArch(fit.points_ras, jaw_type="lower", ras_convention=True)
    pts_dense = np.asarray(arch.pts_dense, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tang = np.gradient(pts_dense, axis=0)
    tang /= np.maximum(1e-8, np.linalg.norm(tang, axis=1, keepdims=True))
    norm = np.cross(tang, up)
    norm /= np.maximum(1e-8, np.linalg.norm(norm, axis=1, keepdims=True))
    bino = np.cross(norm, tang)
    bino /= np.maximum(1e-8, np.linalg.norm(bino, axis=1, keepdims=True))

    PanoramicCurveLogic.clear_panoramic_caches()
    for mode in ("RaySum", "MIP", "MinIP", "Mean"):
        rec_p = _measure_call(
            f"pano_{mode}_cold",
            lambda m=mode: PanoramicCurveLogic.sample_panoramic_slab_array(
                vol, pts_dense, tang, norm, bino, projection_mode=m
            ),
        )
        rec_p.pop("_res")
        records.append(rec_p)

    rec_warm = _measure_call(
        "pano_RaySum_warm",
        lambda: PanoramicCurveLogic.sample_panoramic_slab_array(
            vol, pts_dense, tang, norm, bino, projection_mode="RaySum"
        ),
    )
    rec_warm.pop("_res")
    records.append(rec_warm)

    eng = MeasurementEngine(vol, mask, arch=arch, arches={"lower": arch}, is_dental_segmentator_5class=False)
    rec_cs = _measure_call(
        "cross_section_21_sites",
        lambda: [
            eng.measure_at_arc_mm("lower", float(a))
            for a in np.linspace(5.0, float(arch.total_arc_length) - 5.0, 21)
        ],
    )
    rec_cs.pop("_res")
    records.append(rec_cs)

    imp_logic = ImplantPlannerLogic()
    for i, arc_f in enumerate(np.linspace(0.25, 0.75, 4)):
        f = arch.get_frame(float(arc_f))
        imp_logic.place_or_update_implant(
            fdi=46 - i,
            crest_point_ras=f.point,
            T=f.tangent,
            N=f.normal,
            B=f.binormal,
            diameter_mm=4.1,
            length_mm=10.0,
        )
    rec_imp = _measure_call(
        "implant_verify_and_stl",
        lambda: (
            imp_logic.recompute_all_verifications(volume=vol, mask=mask),
            [
                imp_logic.build_cylinder_mesh_triangles(
                    imp.crest_point_ras,
                    imp.apex_point_ras,
                    0.5 * imp.platform_diameter_mm,
                    0.5 * imp.apical_diameter_mm,
                )
                for imp in imp_logic.placed_implants.values()
            ],
        ),
    )
    rec_imp.pop("_res")
    records.append(rec_imp)

    rec_endo = _measure_call(
        "canal_extract_pulp_46",
        lambda: CanalTracerLogic().extract_root_canal_for_tooth(46, vol, mask),
    )
    rec_endo.pop("_res")
    records.append(rec_endo)

    rec_endo_graph = _measure_call(
        "endo_multiroot_graph_46",
        lambda: EndodonticEngine().analyze_tooth(46, mask, vol.affine, vol.image),
    )
    rec_endo_graph.pop("_res")
    records.append(rec_endo_graph)

    import tempfile

    class _HeadlessPrintLogic:
        def __init__(self, e: MeasurementEngine) -> None:
            self._engine = e

        def _get_or_build_measurement_engine(self, *_args: Any, **_kwargs: Any) -> MeasurementEngine:
            return self._engine

    with tempfile.TemporaryDirectory(prefix="cbct_bench_print_") as tmpdir:
        pl_engine = PrintLayoutEngine(logic=_HeadlessPrintLogic(eng))
        rec_print = _measure_call(
            "print_8x10_q4",
            lambda: pl_engine.generate_8x10_layout(
                None,
                None,
                selected_fdis=[46],
                out_dir=Path(tmpdir),
                grayscale=True,
            ),
        )
        rec_print.pop("_res")
        records.append(rec_print)

    return {
        "volume_shape_zyx": list(vol.image.shape),
        "spacing_mm": list(vol.spacing),
        "structures_count": int(len(np.unique(mask)) - 1),
        "benchmarks": records,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))

"""Real-CT verification under 3D Slicer's PythonSlicer (BL-10).

Run:
    "D:/Program Files/3D Slicer 5.12.3/bin/PythonSlicer.exe" \
        analysis/slicercbctplanner_1/verify_real_ct.py

Checks, on the ToothFairy3S_0021 CBCT (image + ToothFairy3 labelmap):
  1. `CBCTPlannerLib` imports cleanly inside Slicer's Python.
  2. Dentate `W0` bridges buccal->lingual bone across a socket (never a ~2 mm plate).
  3. Tilted ridges keep the crest/W0 on bone (crest voxel is jaw bone).
  4. `H` reaches the basal jaw end on non-canal slices and stops before the IAN.
  5. `RaySum` panoramic + 3D MIP render RGBA with transparent air and isotropic fill.
  6. The 8x10 film page composes and exports PNG/PDF at 300 DPI (8x10 in).
  7. Times the panoramic sampler and the MIP/RGBA capture (OPT profiling).
"""

from __future__ import annotations

import gzip
import io
import json
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "CBCTPlanner"))

SEG_PATH = REPO / "tests" / "data" / "ToothFairy3S_0021_seg.nii.gz"
IMG_PATH = Path("C:/Users/hamid/Downloads/inspect_ToothFairy3S_0021/ToothFairy3S_0021_0000.nii.gz")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def _is_num(token: bytes) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_nifti_gz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal NIfTI-1 reader -> ((Z, Y, X) array, RAS affine)."""
    with gzip.open(str(path), "rb") as fh:
        blob = fh.read()
    header = blob[:352]
    dim = struct.unpack("<8h", header[40:56])
    ndim = int(dim[0])
    shape_xyz = tuple(int(v) for v in dim[1 : 1 + ndim])
    datatype = int(struct.unpack("<h", header[70:72])[0])
    vox_offset = int(round(struct.unpack("<f", header[108:112])[0]))
    srow = [struct.unpack("<4f", header[o : o + 16]) for o in (280, 296, 312)]
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(srow, dtype=np.float64)
    dtypes = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16}
    data = np.frombuffer(blob[vox_offset:], dtype=dtypes[datatype], count=int(np.prod(shape_xyz)))
    return np.transpose(data.reshape(shape_xyz, order="F"), (2, 1, 0)).astype(np.float32), affine


def main() -> int:  # noqa: C901 - a linear verification script
    print("--- 1. import CBCTPlannerLib in Slicer's Python ---", flush=True)
    import CBCTPlannerLib as lib  # noqa: F401
    from CBCTPlannerLib import PanoramicCurveLogic, core_print_render  # noqa: F401
    from CBCTPlannerLib.PrintLayoutEngine import (
        CrossSectionSliceSpec,
        PrintLayoutEngine,
        _image_to_data_uri,
    )
    from CBCTPlannerLib.core.arch_detector import DentalArch, fit_arch_from_label_mask
    from CBCTPlannerLib.core.diagnostics import calibration_stamp
    from CBCTPlannerLib.core.labels import (
        LEFT_IAN_ID,
        LEFT_INCISIVE_CANAL_ID,
        LINGUAL_CANAL_ID,
        LOWER_JAW_ID,
        RIGHT_IAN_ID,
        RIGHT_INCISIVE_CANAL_ID,
        UPPER_JAW_ID,
    )
    from CBCTPlannerLib.core.measurements import MeasurementEngine
    from CBCTPlannerLib.core.orientation import (
        auto_fix_anatomical_orientation,
        canonicalize_to_ras,
        inverted_axes_from_report,
        verify_orientation,
    )
    from CBCTPlannerLib.core.volume_loader import CBCTVolume

    import slicer

    check("import CBCTPlannerLib (package)", True, str(lib.__file__))
    check("import slicer module", True, getattr(slicer, "__file__", "?"))
    print("--- 2. load real CT + labelmap ---", flush=True)
    t0 = time.perf_counter()
    img_zyx, affine = read_nifti_gz(IMG_PATH)
    seg_zyx, seg_affine = read_nifti_gz(SEG_PATH)
    print(f"image {img_zyx.shape} seg {seg_zyx.shape} in {time.perf_counter() - t0:.1f}s", flush=True)
    canon_img = canonicalize_to_ras(img_zyx, affine)
    canon_seg = canonicalize_to_ras(seg_zyx, seg_affine)
    # Same order as CBCTPlannerLogic.standardize_orientation(auto_fix_anatomy=True):
    # canonicalize to RAS, then mirror any anatomically inverted axis in voxel space.
    report = verify_orientation(canon_seg.image_zyx, canon_seg.affine, is_5class=False)
    print(f"orientation report: si_ok={report.si_ok} ap_ok={report.ap_ok} lr_ok={report.lr_ok} {report.notes}", flush=True)
    fixed_img, fixed_seg, flipped, report = auto_fix_anatomical_orientation(
        canon_img.image_zyx, canon_seg.image_zyx, canon_img.affine, is_5class=False
    )
    check(
        "anatomically inverted axis detected and auto-corrected",
        not inverted_axes_from_report(report),
        f"flipped axes: {list(flipped)}; post-fix si_ok={report.si_ok} ap_ok={report.ap_ok} lr_ok={report.lr_ok}",
    )
    mask = np.rint(fixed_seg).astype(np.int32)
    spacing = float(np.linalg.norm(canon_img.affine[:3, 0]))
    origin = tuple(float(v) for v in canon_img.affine[:3, 3])
    direction = canon_img.affine[:3, :3] / np.maximum(np.linalg.norm(canon_img.affine[:3, :3], axis=0), 1e-8)
    vol = CBCTVolume(
        np.asarray(fixed_img, dtype=np.float32), (spacing,) * 3, origin, direction, canon_img.affine
    )
    check("image/labelmap share a grid", vol.image.shape == mask.shape, str(mask.shape))
    check("HU range is a CBCT", float(np.min(vol.image)) < -500 and float(np.max(vol.image)) > 800,
          f"[{float(np.min(vol.image)):.0f}, {float(np.max(vol.image)):.0f}]")

    arches = {}
    for jaw_key, display in (("lower", "Mandible"), ("upper", "Maxilla")):
        fit = fit_arch_from_label_mask(mask, canon_img.affine, jaw=display, is_5class=False)
        if fit is None or not fit.accepted:
            print(f"arch {jaw_key}: NOT ACCEPTED ({getattr(fit, 'reason', 'none')})", flush=True)
            continue
        arches[jaw_key] = DentalArch(fit.points_ras, jaw_type=jaw_key, ras_convention=True)
        print(f"arch {jaw_key}: {len(fit.points_ras)} pts, L={arches[jaw_key].total_arc_length:.1f} mm", flush=True)
    check("both arches fit from the real labelmap", set(arches) == {"lower", "upper"}, str(sorted(arches)))
    if not arches:
        return 1

    engine = MeasurementEngine(
        vol,
        mask,
        arch=arches.get("lower"),
        arches=arches,
        is_dental_segmentator_5class=False,
        extract_teeth_eagerly=False,
    )

    # ------------------------------------------------------------------ sweep
    print("--- 3/4. cross-section sweep: W0 bridging, tilted crest, H vs IAN ---", flush=True)
    # Any canal label in the plane blocks a "height reaches the jaw end" claim;
    # the crossing geometry is still measured only for the IAN under the H ray.
    ian_ids = {LEFT_IAN_ID, RIGHT_IAN_ID}
    canal_ids = ian_ids | {LINGUAL_CANAL_ID, LEFT_INCISIVE_CANAL_ID, RIGHT_INCISIVE_CANAL_ID}
    rows: list[dict] = []
    for jaw_key in ("lower", "upper"):
        arch = arches.get(jaw_key)
        if arch is None:
            continue
        jaw_id = LOWER_JAW_ID if jaw_key == "lower" else UPPER_JAW_ID
        total = float(arch.total_arc_length)
        arcs = np.arange(1.0, total - 1.0, 0.5)
        t_jaw = time.perf_counter()
        for arc in arcs:
            meas, cs = engine.measure_at_arc_mm(jaw_key, float(arc))
            if cs is None:
                continue
            P, _T, N, B = (np.asarray(v, dtype=np.float64) for v in cs.frame)

            def uv_to_world(u: float, v: float) -> np.ndarray:
                return P + u * N + v * B

            def mask_at(world: np.ndarray) -> int:
                ijk = np.rint(vol.world_to_voxel(world)).astype(int)
                if np.any(ijk < 0) or np.any(ijk >= np.array(mask.shape[::-1])):
                    return -1
                return int(mask[ijk[2], ijk[1], ijk[0]])

            def mask_near(world: np.ndarray, ids: set[int]) -> bool:
                """True when any of the 27 voxels around `world` carries one of `ids`."""
                ijk = np.rint(vol.world_to_voxel(world)).astype(int)
                lo = np.maximum(ijk - 1, 0)
                hi = np.minimum(ijk + 2, np.array(mask.shape[::-1]))
                if np.any(hi <= lo):
                    return False
                patch = mask[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
                return bool(np.isin(patch, list(ids)).any())

            crest = getattr(cs, "crest_uv", None)
            basal = getattr(cs, "basal_uv", None)
            if crest is None or basal is None:
                continue
            # Is the crest (and both W0 rims) on jaw bone?
            h_len = float(np.hypot(basal[1] - crest[1], basal[0] - crest[0]))
            crest_label = mask_at(uv_to_world(*crest))
            crest_on_bone = mask_near(uv_to_world(*crest), {jaw_id})
            w0_ends = cs.width_endpoints_uv.get(0.0)
            end_labels = []
            ends_on_bone = False
            if w0_ends is not None:
                end_labels = [mask_at(uv_to_world(*pt)) for pt in w0_ends]
                ends_on_bone = all(mask_near(uv_to_world(*pt), {jaw_id}) for pt in w0_ends)
            # Dentition local to the ridge (not merely anywhere in the 30x50 mm window).
            tooth_ids = [int(i) for i in np.unique(cs.mask_cs) if 11 <= int(i) <= 87]
            tooth_in_slice = False
            if tooth_ids:
                vv, uu = np.where(np.isin(cs.mask_cs, tooth_ids))
                du = np.abs(cs.u_coords[uu] - crest[0])
                dv = np.abs(cs.v_coords[vv] - crest[1])
                tooth_in_slice = bool(np.any((du <= 6.0) & (dv <= 8.0)))
            # Two cortical walls present at the crest level (bone .. tooth .. bone)?
            two_walls = False
            if tooth_ids:
                band = cs.mask_cs[np.abs(cs.v_coords - crest[1]) <= 0.6, :]
                if band.size:
                    is_jaw = np.isin(band, [jaw_id]).any(axis=0).astype(np.int8)
                    is_tooth = np.isin(band, tooth_ids).any(axis=0)
                    # Contiguous cortical runs >= 3 columns (0.75 mm) on both sides of the tooth.
                    edges = np.diff(np.concatenate(([0], is_jaw, [0])))
                    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
                    walls = [(s, e) for s, e in zip(starts, stops) if e - s >= 3]
                    if len(walls) >= 2 and is_tooth.any():
                        ti = np.where(is_tooth)[0]
                        two_walls = bool(
                            walls[0][1] <= ti.min() and walls[-1][0] >= ti.max()
                        )

            # Independent world-Z probes: never reuse the engine's basal_v for these.
            crest_world = uv_to_world(*crest)
            end_world = uv_to_world(*basal)
            ijk = np.rint(vol.world_to_voxel(crest_world)).astype(int)
            y_lo, y_hi = max(0, ijk[1] - 2), ijk[1] + 3
            x_lo, x_hi = max(0, ijk[0] - 2), ijk[0] + 3
            column = mask[:, y_lo:y_hi, x_lo:x_hi]  # full-depth 5x5 column under the crest
            kk = np.where(column == jaw_id)[0]
            jaw_end_z = None
            if len(kk):
                # z increases superiorly after auto-fix: mandible ends at low z, maxilla at high z.
                k_end = int(kk.max()) if jaw_key == "upper" else int(kk.min())
                jaw_end_z = float(vol.voxel_to_world([ijk[0], ijk[1], k_end])[2])
            jaw_end_gap = None if jaw_end_z is None else float(abs(end_world[2] - jaw_end_z))

            # IAN: superior-most canal voxel projected onto this cross-section plane.
            canal_rows, canal_cols = np.where(np.isin(cs.mask_cs, list(canal_ids)))
            canal_present = bool(len(canal_rows))
            ian_rows, ian_cols = np.where(np.isin(cs.mask_cs, list(ian_ids)))
            canal_top_z = None
            canal_crossed = False
            ian_clearance_mm = None
            if len(ian_rows):
                # IAN voxels anywhere in the section: the H ray must stay above their top.
                zs = P[2] + cs.u_coords[ian_cols] * N[2] + cs.v_coords[ian_rows] * B[2]
                canal_top_z = float(np.max(zs))
                # Does the H caliper corridor itself pass over the canal body?
                pts = np.column_stack([cs.u_coords[ian_cols], cs.v_coords[ian_rows]])
                a_uv = np.array([crest[0], crest[1]], dtype=np.float64)
                d_uv = np.array([basal[0] - crest[0], basal[1] - crest[1]], dtype=np.float64)
                t_par = np.clip(((pts - a_uv) @ d_uv) / max(1e-9, float(d_uv @ d_uv)), 0.0, 1.0)
                perp = np.linalg.norm(pts - (a_uv + t_par[:, None] * d_uv), axis=1)
                ian_clearance_mm = float(perp.min())
                over_canal = (t_par <= 0.95) & (perp <= 2.0)
                canal_crossed = bool(over_canal.any())
            rows.append(
                {
                    "jaw": jaw_key,
                    "arc": round(float(arc), 2),
                    "fdi": meas.tooth_fdi if meas else "",
                    "w0": None if meas is None or meas.crest_width_mm is None else round(float(meas.crest_width_mm), 2),
                    "h": None if meas is None or meas.bone_height_mm is None else round(float(meas.bone_height_mm), 2),
                    "ridge_deg": None if cs.ridge_axis_angle_deg is None else round(float(cs.ridge_axis_angle_deg), 1),
                    "crest_on_bone": crest_on_bone,
                    "crest_label": crest_label,
                    "w0_ends_on_bone": ends_on_bone,
                    "w0_end_labels": end_labels,
                    "dentate": tooth_in_slice,
                    "two_walls": two_walls,
                    "h_ray_len": round(h_len, 2),
                    "jaw_end_gap_mm": None if jaw_end_gap is None else round(jaw_end_gap, 2),
                    "canal_present": canal_present,
                    "canal_top_z": None if canal_top_z is None else round(canal_top_z, 2),
                    "h_end_z": round(float(end_world[2]), 2),
                    "canal_crossed": canal_crossed,
                    "ian_clearance_mm": None if ian_clearance_mm is None else round(ian_clearance_mm, 2),
                    "vital_dist": (
                        None
                        if meas is None or meas.distance_to_vital_structure_mm is None
                        else round(float(meas.distance_to_vital_structure_mm), 2)
                    ),
                    "withheld": meas.withheld_reason if meas else cs.withheld_reason,
                }
            )
        print(f"  {jaw_key}: {len(arcs)} sections in {time.perf_counter() - t_jaw:.1f}s", flush=True)

    measured = [r for r in rows if r["w0"] is not None or r["h"] is not None]
    check("cross-sections measured on the real CT", len(measured) >= 20,
          f"{len(measured)}/{len(rows)} sections produced a measurement")

    dentate = [r for r in measured if r["dentate"] and r["w0"] is not None]
    two_wall = [r for r in dentate if r["two_walls"]]
    molars = [r for r in two_wall if str(r["fdi"]).startswith(("6", "7"))
              or str(r["fdi"])[-1] in ("6", "7")]
    bridged = [r for r in two_wall if r["w0"] >= 4.5]
    molar_bridged = [r for r in molars if r["w0"] >= 6.0]
    single_plate = [r for r in two_wall if r["w0"] < 3.0]
    one_wall = [r for r in dentate if not r["two_walls"]]
    check(
        "dentate W0 bridges buccal<->lingual bone across the socket (no 2 mm plate)",
        bool(two_wall) and len(bridged) == len(two_wall)
        and len(molar_bridged) == len(molars) and not single_plate,
        f"{len(bridged)}/{len(two_wall)} two-wall socket sites bridged "
        f"({len(molar_bridged)}/{len(molars)} molars >= 6 mm); W0 "
        f"min {min((r['w0'] for r in two_wall), default=0):.1f} / "
        f"median {np.median([r['w0'] for r in two_wall]) if two_wall else 0:.1f} / "
        f"max {max((r['w0'] for r in two_wall), default=0):.1f} mm; "
        f"sub-3mm plates: {[r['arc'] for r in single_plate]}; "
        f"{len(one_wall)} slices clip a single cortical wall (W0 there is a plate, not a bridge); "
        f"two-wall detail (arc, fdi, W0): {[(r['arc'], r['fdi'], r['w0']) for r in two_wall]}",
    )
    check(
        "dentate W0 rims both land on bone",
        bool(dentate) and sum(r["w0_ends_on_bone"] for r in dentate) / len(dentate) >= 0.8,
        f"{sum(r['w0_ends_on_bone'] for r in dentate)}/{len(dentate)}",
    )

    tilted = [r for r in measured if r["ridge_deg"] is not None and abs(r["ridge_deg"]) >= 15.0]
    on_bone = [r for r in tilted if r["w0_ends_on_bone"] and (r["w0"] or 0) >= 4.0]
    print(
        "  tilted detail (arc, deg, w0, crest_label, on_bone): "
        + str([(r["arc"], r["ridge_deg"], r["w0"], r["crest_label"], r["crest_on_bone"]) for r in tilted]),
        flush=True,
    )
    check(
        "tilted ridges (|ridge axis| >= 15 deg) keep W0 on bone",
        bool(tilted) and len(on_bone) == len(tilted),
        f"{len(on_bone)}/{len(tilted)} tilted sites with both W0 rims on jaw bone and W0 >= 4 mm "
        f"(max tilt {max((abs(r['ridge_deg']) for r in tilted), default=0):.1f} deg)",
    )

    non_canal = [r for r in measured if r["jaw"] == "lower" and not r["canal_present"] and r["h"]]
    reached = [r for r in non_canal if r["jaw_end_gap_mm"] is not None and r["jaw_end_gap_mm"] <= 1.5]
    check(
        "H reaches the basal jaw end on non-canal mandibular slices",
        bool(non_canal) and len(reached) / len(non_canal) >= 0.8,
        f"{len(reached)}/{len(non_canal)} within 1.5 mm of the independently measured jaw end; "
        f"median H {np.median([r['h'] for r in non_canal]) if non_canal else 0:.1f} mm",
    )

    canal_slices = [r for r in measured if r["jaw"] == "lower" and r["canal_present"]]
    stopped = [r for r in canal_slices if not r["canal_crossed"]]
    check(
        "H stops strictly before the IAN canal",
        bool(canal_slices) and len(stopped) == len(canal_slices),
        f"{len(stopped)}/{len(canal_slices)} canal slices keep the H caliper clear of the canal; "
        f"min lateral clearance "
        f"{min((r['ian_clearance_mm'] for r in canal_slices if r['ian_clearance_mm'] is not None), default=0):.2f} mm"
        f"; H end vs canal top: "
        f"{min((r['h_end_z'] - (r['canal_top_z'] or 0) for r in canal_slices), default=0):.2f} mm",
    )
    crossing = [
        (r["arc"], r["fdi"], r["h"], r["h_end_z"], r["canal_top_z"], r["ian_clearance_mm"])
        for r in canal_slices
        if r["canal_crossed"]
    ]
    if crossing:
        print(f"  IAN crossings (arc, fdi, H, h_end_z, canal_top_z, clearance): {crossing}", flush=True)
    gaps = sorted(r["jaw_end_gap_mm"] for r in non_canal if r["jaw_end_gap_mm"] is not None)
    print(f"  non-canal jaw-end gaps (mm): {[round(g, 2) for g in gaps]}", flush=True)
    outliers = [
        (r["arc"], r["fdi"], r["h"], r["jaw_end_gap_mm"], r["withheld"])
        for r in non_canal
        if r["jaw_end_gap_mm"] is not None and r["jaw_end_gap_mm"] > 2.0
    ]
    if outliers:
        print(f"  H outliers (arc, fdi, H, gap, withheld): {outliers}", flush=True)

    # ------------------------------------------------------------- panoramic
    print("--- 5. RaySum panoramic + RGBA captures ---", flush=True)
    arch = arches["lower"]
    total = float(arch.total_arc_length)
    arc_vals = np.arange(0.5, total - 0.5, 0.5)
    pts, tans, norms, bins = [], [], [], []
    for a in arc_vals:
        fr = arch.get_frame_at_mm(float(a))
        pts.append(fr.point)
        tans.append(fr.tangent)
        norms.append(fr.normal)
        bins.append(fr.binormal)
    t_pano = time.perf_counter()
    pano = PanoramicCurveLogic.sample_panoramic_slab_array(
        vol,
        np.asarray(pts),
        np.asarray(tans),
        np.asarray(norms),
        np.asarray(bins),
        slab_thickness_mm=25.0,
        height_mm=120.0,
        spacing_mm=0.5,
    )
    t_sample = time.perf_counter() - t_pano
    finite_frac = float(np.isfinite(pano).mean())
    check(
        "RaySum panoramic samples the real CT",
        pano.ndim == 2 and abs(pano.shape[1] - len(arc_vals)) <= 1 and finite_frac > 0.999
        and float(pano.std()) > 1.0,
        f"shape {pano.shape} (cols {len(arc_vals)}), {t_sample:.1f}s, std {float(pano.std()):.1f}, "
        f"finite {finite_frac:.4f}, range [{np.nanmin(pano):.0f}, {np.nanmax(pano):.0f}]",
    )

    ticks = [float(a) for a in np.linspace(0.0, total, 12)]
    t_r = time.perf_counter()
    strip = core_print_render.render_panoramic_strip(
        pano,
        column_spacing_mm=0.5,
        tick_arcs_mm=ticks,
        strip_height=340,
        scale=2,
        transparent_bg=True,
        crop_empty_vertical=True,
        preserve_aspect=True,
        target_size=(1800, 560),
    )
    t_strip = time.perf_counter() - t_r
    strip_img = strip.image
    alpha = np.asarray(strip_img.split()[3]) if strip_img is not None and strip_img.mode == "RGBA" else None
    fg_frac = None
    tight = False
    if alpha is not None and alpha.size:
        cols = np.where((alpha > 8).any(axis=0))[0]
        rows_ = np.where((alpha > 8).any(axis=1))[0]
        fg_frac = float(alpha[alpha > 8].size) / float(alpha.size)
        if len(cols) and len(rows_):
            fill_w = (cols[-1] - cols[0] + 1) / strip_img.width
            fill_h = (rows_[-1] - rows_[0] + 1) / strip_img.height
            tight = max(fill_w, fill_h) >= 0.9
    check(
        "panoramic strip renders RGBA with transparent air",
        bool(strip.available) and strip_img is not None and strip_img.mode == "RGBA" and alpha is not None
        and int(alpha.min()) == 0,
        f"{None if strip_img is None else strip_img.size} mode={None if strip_img is None else strip_img.mode} "
        f"alpha_min={None if alpha is None else int(alpha.min())} fg={0 if fg_frac is None else round(fg_frac, 3)}",
    )
    check("panoramic strip is isotropically tight-filled (limiting axis >= 90%)", tight,
          f"render {t_strip:.2f}s")

    t_v = time.perf_counter()
    mip = core_print_render.render_3d_mip_card(
        vol.image,
        mask=mask,
        spacing_zyx=(spacing, spacing, spacing),
        target_size=(780, 280),
        transparent_bg=True,
    )
    t_mip = time.perf_counter() - t_v
    mip_alpha = np.asarray(mip.image.split()[3]) if mip.image is not None and mip.image.mode == "RGBA" else None
    check(
        "3D MIP card renders RGBA with transparent air",
        bool(mip.available) and mip_alpha is not None and int(mip_alpha.min()) == 0,
        f"{None if mip.image is None else mip.image.size} mode={None if mip.image is None else mip.image.mode} "
        f"{t_mip:.2f}s",
    )

    # --------------------------------------------------------------- 8x10
    print("--- 6. 8x10 film export at 300 DPI ---", flush=True)
    fdi = next((r["fdi"] for r in measured if r["jaw"] == "lower" and r["fdi"]), "46")
    arc0 = None
    for r in measured:
        if r["jaw"] == "lower" and r["fdi"] == fdi:
            arc0 = r["arc"]
            break
    if arc0 is None:
        arc0 = total * 0.25
    specs: list[CrossSectionSliceSpec] = []
    offsets = np.linspace(-4.0, 4.0, 5)
    for idx, off in enumerate(offsets):
        m, c = engine.measure_at_arc_mm("lower", float(arc0 + off), fdi_hint=str(fdi))
        tile = None
        if c is not None:
            rnd = core_print_render.render_cross_section(c, m, scale=3, show_masks=False,
                                                         show_implant_guides=False, monochrome=True)
            tile = rnd.image
        uri = _image_to_data_uri(tile) if tile is not None else ""
        specs.append(
            CrossSectionSliceSpec(
                slice_index=idx + 1,
                total_slices=len(offsets),
                offset_mm=float(off),
                fdi=int(fdi),
                height_mm=m.bone_height_mm if m else None,
                height_estimated=bool(m and str(getattr(m, "provenance", "")).lower().startswith("estimated")),
                crest_width_mm=m.crest_width_mm if m else None,
                width_3mm=(m.bone_width_at_depths_mm.get(3.0) if m and m.bone_width_at_depths_mm else None),
                width_5mm=(m.bone_width_at_depths_mm.get(5.0) if m and m.bone_width_at_depths_mm else None),
                nerve_dist_mm=m.distance_to_vital_structure_mm if m else None,
                misch_class=None,
                mean_hu=m.mean_density_hu if m else None,
                is_mandible=True,
                image_data_uri=uri,
                _pil_image=tile,
                withheld_reason=(m.withheld_reason if m else getattr(c, "withheld_reason", None)),
            )
        )
    check("film specs carry rendered cross-sections", all(s._pil_image is not None for s in specs),
          f"{sum(s._pil_image is not None for s in specs)}/{len(specs)}")

    engine_pl = PrintLayoutEngine(logic=None)
    record = {
        "quadrant": 4,
        "jaw_label": "Mandible",
        "fdis": [int(fdi)],
        "calibration": calibration_stamp(engine),
        "slices": specs,
        "pano_b64": _image_to_data_uri(strip_img),
        "vr_b64": _image_to_data_uri(mip.image),
        "span_summary": {},
    }
    t_comp = time.perf_counter()
    page = engine_pl._compose_page_image(
        record, patient_name="ToothFairy3S_0021", page_number=1, total_pages=1, grayscale=True
    )
    t_compose = time.perf_counter() - t_comp
    out_dir = Path(tempfile.mkdtemp(prefix="cbct_verify_"))
    png_file = out_dir / "ToothFairy3S_0021_8x10.png"
    pdf_file = out_dir / "ToothFairy3S_0021_8x10.pdf"
    ok_export = False
    detail = ""
    if page is not None:
        page.save(png_file, dpi=(300, 300))
        page.save(pdf_file, "PDF", resolution=300.0)
        from PIL import Image as PILImage

        with PILImage.open(png_file) as pi:
            png_size = pi.size
            png_dpi = pi.info.get("dpi")
        pdf_raw = pdf_file.read_bytes()
        mediabox = None
        if pdf_raw.startswith(b"%PDF") and b"/MediaBox" in pdf_raw:
            seg = pdf_raw.split(b"/MediaBox", 1)[1][:80]
            nums = [float(x) for x in seg.replace(b"[", b" ").replace(b"]", b" ").split()[:4] if _is_num(x)]
            if len(nums) == 4:
                mediabox = (round((nums[2] - nums[0]) / 72.0, 2), round((nums[3] - nums[1]) / 72.0, 2))
        ok_export = (
            png_size == (2400, 3000)
            and png_file.stat().st_size > 50_000
            and mediabox == (8.0, 10.0)
            and pdf_file.stat().st_size > 10_000
        )
        detail = (
            f"png {png_size} {png_file.stat().st_size // 1024} KiB dpi={png_dpi} | "
            f"pdf {mediabox} in {pdf_file.stat().st_size // 1024} KiB | compose {t_compose:.1f}s"
        )
    check("8x10 page composes and exports PNG+PDF at 300 DPI", ok_export, detail)

    html = engine_pl._build_8x10_html(
        specs, record["pano_b64"], record["vr_b64"], 4, "Mandible", [int(fdi)],
        "ToothFairy3S_0021", "TF3S-0021", 1, 1, grayscale=True,
    )
    html_file = out_dir / "ToothFairy3S_0021_8x10.html"
    html_file.write_text(html, encoding="utf-8")
    transparent_css = ".pano-section, .pano-img, .vr-card, .vr-img { background: transparent !important; }" in html
    check(
        "8x10 HTML keeps transparent pano/VR backgrounds",
        transparent_css and "data:image/png;base64," in html and "@page" in html,
        f"{len(html)} chars -> {html_file}",
    )

    # ------------------------------------------------------------ profiling
    print("--- 7. profiling (OPT-01 / OPT-03) ---", flush=True)
    n_cols, n_rows = pano.shape
    print(
        f"RaySum pano {n_cols}x{n_rows} over {len(arc_vals)} arcs: {t_sample:.2f}s "
        f"({t_sample / max(1, len(arc_vals)) * 1000:.2f} ms/col, 25 mm slab)", flush=True)
    print(f"pano strip render: {t_strip:.2f}s | 3D MIP {vol.image.shape}: {t_mip:.2f}s | page compose: {t_compose:.1f}s", flush=True)

    # -------------------------------------------------------------- summary
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({"passed": len(RESULTS) - len(failed), "failed": failed, "rows": len(rows)}, indent=2))
    (out_dir / "sections.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"artifacts: {out_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

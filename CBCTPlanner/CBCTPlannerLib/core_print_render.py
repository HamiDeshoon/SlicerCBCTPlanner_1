"""Clinical renderers for cross-sections and the panoramic strip.

Pure numpy geometry with Pillow (and automatic Qt fallback in 3D Slicer).
Core design principle: the measurement engine determines all endpoints and values;
this renderer ONLY transforms engine-supplied coordinates into display pixels and
visualizes them. It never independently detects anatomy, modifies endpoints, or
recalculates measurements or implant dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core.labels import jaw_mask_from_labels, tooth_mask_from_labels
from .core.measurements import (
    IMPLANT_APICAL_SAFETY_MARGIN_MM,
    PROVENANCE_ESTIMATED,
    PROVENANCE_INTERPOLATED,
)

try:  # Pillow is optional in Slicer; callers render an "unavailable" tile when missing.
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without Pillow
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

MM_PER_ROW = 0.25
MM_PER_COL = 0.25
SAFETY_MARGIN_MM = IMPLANT_APICAL_SAFETY_MARGIN_MM

# Consistent clinical color system (Part 4.3)
COLOR_BONE_HEIGHT = (34, 222, 94)  # Vivid green: available bone height
COLOR_BONE_HEIGHT_TEXT = (74, 232, 128)
COLOR_ROOT_EXTENT = (192, 132, 252)  # Violet: tooth root extent (distinct from bone height)
COLOR_WIDTH_W0 = (0, 220, 255)  # Cyan: crestal width W0
COLOR_WIDTH_W3 = (255, 180, 0)  # Gold: width at 3 mm
COLOR_WIDTH_DEEP = (255, 120, 40)  # Orange: width at >= 5 mm
COLOR_CREST_MARKER = (250, 204, 21)  # Amber-yellow: alveolar crest marker
COLOR_VITAL_CANAL = (255, 64, 64)  # Red: IAN canal & safety margin
COLOR_VITAL_SINUS = (244, 114, 182)  # Pink: maxillary sinus floor
COLOR_IMPLANT = (56, 189, 248)  # Sky blue: suggested implant (planning aid)
COLOR_UNAVAILABLE = (148, 163, 184)  # Muted slate: unavailable/withheld

# Width caliper opacity: W0 is the primary crest measurement, W3/W5 are context and are
# drawn semi-transparent so they never compete with the bone surface underneath.
WIDTH_ALPHA_W0 = 255
WIDTH_ALPHA_W3 = 120
WIDTH_ALPHA_W5 = 95

_FONT_CANDIDATES_REGULAR = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "arial.ttf",
)

_FONT_CANDIDATES_BOLD = (
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "arialbd.ttf",
)


@lru_cache(maxsize=32)
def _load_font(size: int, bold: bool = False) -> Any:
    """Load a crisp TrueType font at ``size`` px with safe cross-platform fallbacks."""
    if not _PIL_AVAILABLE or ImageFont is None:
        return None
    px = max(10, int(size))
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path in candidates:
        try:
            if "/" in path and not Path(path).exists():
                continue
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1]))
    except Exception:
        return max(1, len(text) * 7), 13


def _draw_pill_label(
    draw: Any,
    x: float,
    y: float,
    text: str,
    *,
    fg: Tuple[int, int, int],
    bg: Tuple[int, int, int] = (8, 13, 24),
    border: Optional[Tuple[int, int, int]] = None,
    font: Any = None,
    pad_x: int = 4,
    pad_y: int = 2,
    bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, float]:
    """Draw a high-contrast dark-backed pill label clamped inside ``bounds``."""
    tw, th = _text_size(draw, text, font)
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y
    px = float(x)
    py = float(y)
    if bounds is not None:
        max_w, max_h = bounds
        px = min(max(1.0, px), max(1.0, float(max_w - box_w - 1)))
        py = min(max(1.0, py), max(1.0, float(max_h - box_h - 1)))
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        off_x, off_y = int(bb[0]), int(bb[1])
    except Exception:
        off_x, off_y = 0, 0
    rect = (px, py, px + box_w, py + box_h + 2)
    draw.rectangle(rect, fill=bg, outline=border, width=1 if border else 0)
    draw.text((px + pad_x - off_x, py + pad_y + 1 - off_y), text, fill=fg, font=font)
    return rect


def _draw_dashed_segment(
    draw: Any,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    fill: Tuple[int, int, int],
    dash: int = 6,
    gap: int = 4,
    width: int = 1,
) -> None:
    """Draw a dashed line between arbitrary 2D points ``p0`` and ``p1``."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-3:
        return
    ux = (x1 - x0) / dist
    uy = (y1 - y0) / dist
    cur = 0.0
    while cur < dist:
        seg_end = min(dist, cur + dash)
        draw.line(
            [(x0 + cur * ux, y0 + cur * uy), (x0 + seg_end * ux, y0 + seg_end * uy)],
            fill=fill,
            width=width,
        )
        cur += dash + gap


def _draw_endcap(
    draw: Any,
    pt: Tuple[float, float],
    perp_unit: Tuple[float, float],
    cap_r: float,
    *,
    fill: Tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a perpendicular caliper end-cap at ``pt``."""
    px, py = float(pt[0]), float(pt[1])
    nx, ny = float(perp_unit[0]), float(perp_unit[1])
    draw.line(
        [(px - cap_r * nx, py - cap_r * ny), (px + cap_r * nx, py + cap_r * ny)],
        fill=fill,
        width=width,
    )


def _draw_dashed_hline(
    draw: Any,
    x0: float,
    x1: float,
    y: float,
    *,
    fill: Tuple[int, int, int],
    dash: int = 6,
    gap: int = 4,
    width: int = 1,
) -> None:
    _draw_dashed_segment(draw, (min(x0, x1), y), (max(x0, x1), y), fill=fill, dash=dash, gap=gap, width=width)


def _draw_dashed_vline(
    draw: Any,
    x: float,
    y0: float,
    y1: float,
    *,
    fill: Tuple[int, int, int],
    dash: int = 6,
    gap: int = 4,
    width: int = 1,
) -> None:
    _draw_dashed_segment(draw, (x, min(y0, y1)), (x, max(y0, y1)), fill=fill, dash=dash, gap=gap, width=width)


@dataclass
class CrossSectionRender:
    image: Any
    overlay_px: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    labels: List[Tuple[float, float, str]] = field(default_factory=list)
    mm_per_px: float = MM_PER_COL
    available: bool = True
    reason: Optional[str] = None


def hu_window_to_rgb(img_cs: np.ndarray, window: float = 4000.0, level: float = 1000.0) -> np.ndarray:
    """Map CBCT intensity values to 8-bit grayscale using a window/level (with adaptive fallback)."""
    arr = np.asarray(img_cs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low = float(level) - float(window) / 2.0
    high = float(level) + float(window) / 2.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    if float(np.std(scaled)) < 0.01 and float(np.max(arr) - np.min(arr)) > 1.0:
        p2, p99 = np.percentile(arr, [2.0, 99.0])
        denom = max(1.0, float(p99 - p2))
        scaled = np.clip((arr - float(p2)) / denom, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _enhance_bone_clarity(gray_u8: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Fast pure-numpy 3x3 unsharp mask to sharpen cortical plates, roots, and canal walls."""
    arr = np.asarray(gray_u8, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3 or strength <= 0.0:
        return np.asarray(gray_u8, dtype=np.uint8)
    p = np.pad(arr, 1, mode="edge")
    blur = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    ) * (1.0 / 9.0)
    sharp = arr + float(strength) * (arr - blur)
    return np.clip(sharp, 0.0, 255.0).astype(np.uint8)


def _px(u: float, v: float, u_coords: np.ndarray, v_coords: np.ndarray, scale: int) -> Tuple[float, float]:
    """Map physical cross-section `(u, v)` mm coordinates to display pixel `(x, y)` coordinates."""
    u_lo, u_hi = float(u_coords[0]), float(u_coords[-1])
    v_lo, v_hi = float(min(v_coords[0], v_coords[-1])), float(max(v_coords[0], v_coords[-1]))
    col = (float(u) - u_lo) / max(1e-6, u_hi - u_lo)
    row = (v_hi - float(v)) / max(1e-6, v_hi - v_lo)
    width = (len(u_coords) - 1) * scale
    height = (len(v_coords) - 1) * scale
    return col * width, row * height


def _apply_anatomical_contour_tint(rgb: np.ndarray, mask_cs: Optional[np.ndarray]) -> np.ndarray:
    """Blend subtle anatomical boundary contours using the canonical label registry."""
    if mask_cs is None or mask_cs.shape != rgb.shape[:2]:
        return rgb
    out = rgb.astype(np.float32)
    m = np.asarray(mask_cs)
    edge = np.zeros(m.shape, dtype=bool)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]

    tooth_edge = edge & tooth_mask_from_labels(m)
    if np.any(tooth_edge):
        out[tooth_edge] = 0.35 * out[tooth_edge] + 0.65 * np.array([255.0, 215.0, 90.0], dtype=np.float32)

    jaw_edge = edge & jaw_mask_from_labels(m) & (~tooth_edge)
    if np.any(jaw_edge):
        out[jaw_edge] = 0.55 * out[jaw_edge] + 0.45 * np.array([80.0, 220.0, 180.0], dtype=np.float32)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def measurement_footer_text(meas: Optional[Any], bone_density: Optional[str] = None) -> str:
    """Return the tile footer `H ... / W0 ...` caption, printing `--` for a withheld metric."""
    if meas is None:
        base = "H --  W0 --"
    else:
        height_mm = getattr(meas, "bone_height_mm", None)
        crest_w = getattr(meas, "crest_width_mm", None)
        h_txt = f"{float(height_mm):.1f} mm" if height_mm is not None else "--"
        w_txt = f"{float(crest_w):.1f} mm" if crest_w is not None else "--"
        base = f"H {h_txt}  W0 {w_txt}"
    return f"{base}  {bone_density}" if bone_density else base


def _resolve_engine_overlay_data(cs: Any, meas: Optional[Any]) -> Dict[str, Any]:
    """Read ONLY pre-computed geometry and values from `cs` and `meas` without inferring anatomy."""
    is_mand = bool(getattr(cs, "is_mandible", True))
    records = dict(getattr(cs, "measurements", None) or {})
    if not records and meas is not None:
        records = dict(getattr(meas, "measurements", None) or {})

    # 1. Available bone height (exclusively from engine record or explicit cs crest/basal)
    h_rec = records.get("available_bone_height")
    crest_uv = getattr(cs, "crest_uv", None)
    basal_uv = getattr(cs, "basal_uv", None)
    height_mm = getattr(meas, "bone_height_mm", None) if meas is not None else None
    height_prov = (
        getattr(h_rec, "provenance", None)
        if h_rec is not None
        else getattr(cs, "provenance", "label_derived")
    )

    if h_rec is not None and h_rec.is_available:
        crest_uv = h_rec.start_uv_mm
        basal_uv = h_rec.end_uv_mm
        height_mm = h_rec.value_mm
    else:
        # Backward compatibility for minimal mock `CrossSectionData` objects in legacy tests
        if crest_uv is None and getattr(cs, "crest_v", None) is not None:
            crest_uv = (float(getattr(cs, "height_u", 0.0)), float(cs.crest_v))
        if basal_uv is None and crest_uv is not None:
            if getattr(cs, "basal_v", None) is not None:
                basal_uv = (float(crest_uv[0]), float(cs.basal_v))
            elif height_mm is not None and float(height_mm) > 0.0:
                bv = (
                    float(crest_uv[1]) - float(height_mm)
                    if is_mand
                    else float(crest_uv[1]) + float(height_mm)
                )
                basal_uv = (float(crest_uv[0]), bv)
        if height_mm is None and crest_uv is not None and basal_uv is not None:
            # Only derive displayed length if a measurement object was explicitly supplied or basal_v was set
            if getattr(cs, "basal_v", None) is not None or getattr(cs, "basal_uv", None) is not None:
                height_mm = round(math.hypot(basal_uv[0] - crest_uv[0], basal_uv[1] - crest_uv[1]), 2)

    # 2. Separate root extent (never substituted for available bone height!)
    r_rec = records.get("root_extent")
    root_start_uv = (
        r_rec.start_uv_mm
        if (r_rec is not None and r_rec.is_available)
        else getattr(cs, "root_start_uv", None)
    )
    root_apex_uv = (
        r_rec.end_uv_mm if (r_rec is not None and r_rec.is_available) else getattr(cs, "root_apex_uv", None)
    )
    root_extent_mm = (
        r_rec.value_mm
        if (r_rec is not None and r_rec.is_available)
        else (getattr(meas, "root_extent_mm", None) if meas is not None else None)
    )

    # 3. Width endpoints (prefer exact 2D ridge-aligned endpoints, fallback to legacy width_segments)
    width_endpoints: Dict[float, Tuple[Tuple[float, float], Tuple[float, float], Optional[float], str]] = {}
    raw_endpoints = dict(getattr(cs, "width_endpoints_uv", None) or {})
    raw_segments = dict(getattr(cs, "width_segments", None) or {})
    all_depths = sorted(set(raw_endpoints.keys()) | set(raw_segments.keys()))
    for depth in all_depths:
        w_rec = records.get(f"width_{depth:g}")
        prov = w_rec.provenance if w_rec is not None else "label_derived"
        if w_rec is not None and w_rec.is_available:
            width_endpoints[float(depth)] = (w_rec.start_uv_mm, w_rec.end_uv_mm, w_rec.value_mm, prov)
        elif depth in raw_endpoints:
            p0, p1 = raw_endpoints[depth]
            val = round(math.hypot(p1[0] - p0[0], p1[1] - p0[1]), 2)
            width_endpoints[float(depth)] = (p0, p1, val, prov)
        elif depth in raw_segments and crest_uv is not None:
            u_l, u_r = raw_segments[depth]
            v_lvl = float(crest_uv[1]) - float(depth) if is_mand else float(crest_uv[1]) + float(depth)
            val = round(abs(float(u_r) - float(u_l)), 2)
            width_endpoints[float(depth)] = ((float(u_l), v_lvl), (float(u_r), v_lvl), val, prov)

    # 4. Implant suggestion (exclusively from engine — never recalculated here!)
    imp_diam = (
        getattr(meas, "suggested_diameter_mm", None)
        if meas is not None
        else getattr(cs, "suggested_implant_diameter_mm", None)
    )
    imp_len = (
        getattr(meas, "suggested_length_mm", None)
        if meas is not None
        else getattr(cs, "suggested_implant_length_mm", None)
    )

    # 5. Approximate bone density code (exclusively from engine — never recalculated here!)
    misch_code = getattr(cs, "misch_code", None)
    if misch_code is None and meas is not None and getattr(meas, "misch_density_class", None):
        misch_code = str(meas.misch_density_class).split()[0].strip()

    return {
        "is_mand": is_mand,
        "crest_uv": crest_uv,
        "basal_uv": basal_uv,
        "height_mm": float(height_mm) if height_mm is not None else None,
        "height_prov": height_prov,
        "crest_profile_uv": getattr(cs, "crest_profile_uv", None),
        "root_start_uv": root_start_uv,
        "root_apex_uv": root_apex_uv,
        "root_extent_mm": float(root_extent_mm) if root_extent_mm is not None else None,
        "width_endpoints": width_endpoints,
        "canal_centroid_uv": getattr(cs, "canal_centroid_uv", None),
        "vital_contour_uv": getattr(cs, "vital_contour_uv", None),
        "sinus_floor_uv": getattr(cs, "sinus_floor_uv", None),
        "imp_diam": float(imp_diam) if imp_diam is not None else None,
        "imp_len": float(imp_len) if imp_len is not None else None,
        "misch_code": misch_code,
        "ridge_angle": getattr(cs, "ridge_axis_angle_deg", None),
        "withheld_reason": (
            getattr(meas, "withheld_reason", None)
            if meas is not None
            else getattr(cs, "withheld_reason", None)
        ),
    }


def render_cross_section(
    cs: Any,
    meas: Optional[Any],
    *,
    scale: int = 3,
    window: float = 4000.0,
    level: float = 1000.0,
    dark_film: bool = False,
    show_masks: bool = False,
    show_implant_guides: bool = True,
    enhance_bone_edges: bool = True,
    monochrome: bool = False,
) -> CrossSectionRender:
    """Render one arch cross-section strictly from engine-supplied measurement endpoints.

    Never detects anatomy, modifies endpoints, or fabricates missing measurements.

    ``monochrome=True`` renders a plain grayscale film tile (no implant envelope, root line,
    safety barrier, density or angle badges) for printing.
    """
    if monochrome:
        show_implant_guides = False
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    if cs is None:
        return CrossSectionRender(None, available=False, reason="no cross-section data")

    u_coords = np.asarray(cs.u_coords, dtype=np.float64)
    v_coords = np.asarray(cs.v_coords, dtype=np.float64)
    img_raw = np.asarray(cs.img_cs)
    geom = _resolve_engine_overlay_data(cs, meas)

    # `cs.img_cs[iv, iu]` is indexed by `v_coords[iv]`. `_px()` maps superior (`max(v)`) to row 0.
    # Flip vertically when `v_coords` is ascending so CBCT anatomy and overlays share one frame.
    img_arr = img_raw
    mask_arr = getattr(cs, "mask_cs", None)
    if len(v_coords) >= 2 and float(v_coords[-1]) > float(v_coords[0]):
        img_arr = img_arr[::-1, :]
        if mask_arr is not None:
            mask_arr = np.asarray(mask_arr)[::-1, :]

    gray = hu_window_to_rgb(img_arr, window=window, level=level)
    if enhance_bone_edges:
        gray = _enhance_bone_clarity(gray, strength=0.35)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if dark_film:
        rgb = 255 - rgb
    if show_masks:
        rgb = _apply_anatomical_contour_tint(rgb, mask_arr)

    image = Image.fromarray(rgb, mode="RGB")
    width = max(1, (len(u_coords) - 1) * scale)
    height = max(1, (len(v_coords) - 1) * scale)
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((width, height), getattr(resampling, "BICUBIC", Image.BILINEAR))
    # Draw every overlay on an RGBA layer so W3/W5 can be genuinely semi-transparent and
    # still read over bright cortical bone; the layer is composited once at the end.
    base_image = image.convert("RGBA")
    layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font_sm = _load_font(max(11, int(round(4.6 * scale))), bold=True)
    font_md = _load_font(max(13, int(round(5.4 * scale))), bold=True)
    font_lg = _load_font(max(14, int(round(6.2 * scale))), bold=True)
    footer_h = max(24, int(round(9.5 * scale)))
    safe_h = max(1, height - footer_h - 2)
    bounds = (width, safe_h)

    overlay: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    labels: List[Tuple[float, float, str]] = []
    occupied_rects: List[Tuple[float, float, float, float]] = []

    def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        return not (a[2] + 2 < b[0] or b[2] + 2 < a[0] or a[3] + 2 < b[1] or b[3] + 2 < a[1])

    def _place_non_overlapping_pill(
        pref_x: float,
        pref_y: float,
        text: str,
        *,
        fg: Tuple[int, int, int],
        border: Optional[Tuple[int, int, int]] = None,
        font: Any = None,
    ) -> Tuple[float, float, float, float]:
        tw, th = _text_size(draw, text, font)
        bw, bh = tw + 8, th + 4
        candidates = [
            (pref_x, pref_y),
            (pref_x, pref_y - bh - 4),
            (pref_x, pref_y + bh + 4),
            (max(20.0, pref_x - bw - 10), pref_y),
            (min(float(width - bw - 3), pref_x + bw + 10), pref_y),
            (max(20.0, pref_x - bw - 10), pref_y - bh - 4),
        ]
        chosen = candidates[0]
        for cx, cy in candidates:
            cx_c = min(max(18.0, cx), max(18.0, float(width - bw - 2)))
            cy_c = min(max(24.0, cy), max(24.0, float(safe_h - bh - 2)))
            cand_rect = (cx_c, cy_c, cx_c + bw, cy_c + bh)
            if not any(_rects_overlap(cand_rect, r) for r in occupied_rects):
                chosen = (cx_c, cy_c)
                break
        rect = _draw_pill_label(
            draw,
            chosen[0],
            chosen[1],
            text,
            fg=fg,
            bg=(8, 13, 24),
            border=border,
            font=font,
            bounds=bounds,
        )
        occupied_rects.append(rect)
        labels.append((chosen[0], chosen[1], text))
        return rect

    # Orientation badges: Lingual (L) left, Buccal (B) right, Superior (S) top-center.
    l_rect = _draw_pill_label(
        draw, 3, 3, "L", fg=COLOR_BONE_HEIGHT_TEXT, border=COLOR_BONE_HEIGHT, font=font_md, bounds=bounds
    )
    b_tw, _ = _text_size(draw, "B", font_md)
    b_rect = _draw_pill_label(
        draw,
        max(3, width - b_tw - 11),
        3,
        "B",
        fg=COLOR_BONE_HEIGHT_TEXT,
        border=COLOR_BONE_HEIGHT,
        font=font_md,
        bounds=bounds,
    )
    s_tw, _ = _text_size(draw, "S", font_sm)
    s_rect = _draw_pill_label(
        draw, max(3, (width - s_tw - 8) // 2), 3, "S", fg=COLOR_UNAVAILABLE, font=font_sm, bounds=bounds
    )
    occupied_rects.extend([l_rect, b_rect, s_rect])
    labels.extend([(3.0, 3.0, "L"), (float(max(3, width - b_tw - 11)), 3.0, "B")])

    # Millimetre ruler (left edge) and 10 mm scale bar on every tile
    ruler_colour = (245, 245, 245) if dark_film else (195, 210, 225)
    v_min, v_max = float(v_coords.min()), float(v_coords.max())
    for tick_mm in np.arange(np.ceil(v_min / 5.0) * 5.0, v_max, 5.0):
        _, tick_y = _px(u_coords[0], float(tick_mm), u_coords, v_coords, scale)
        if tick_y < 24 or tick_y > safe_h - 22:
            continue
        major = abs(float(tick_mm) % 10.0) < 1e-6
        tick_len = max(9, 4 * scale) if major else max(5, 2 * scale)
        draw.line([(0, tick_y), (tick_len, tick_y)], fill=ruler_colour, width=1)
        if major:
            draw.text((tick_len + 2, tick_y - 6), f"{float(tick_mm):+.0f}", fill=ruler_colour, font=font_sm)

    bar_px = 10.0 / max(1e-6, abs(u_coords[-1] - u_coords[0])) * width
    bar_y = max(28, safe_h - max(14, int(round(5.5 * scale))))
    bar_x0 = 8.0
    bar_x1 = bar_x0 + bar_px
    draw.line([(bar_x0, bar_y), (bar_x1, bar_y)], fill=ruler_colour, width=2)
    draw.line([(bar_x0, bar_y - 4), (bar_x0, bar_y + 4)], fill=ruler_colour, width=1)
    draw.line([(bar_x1, bar_y - 4), (bar_x1, bar_y + 4)], fill=ruler_colour, width=1)
    scale_rect = _draw_pill_label(
        draw, bar_x1 + 4, bar_y - 8, "10 mm", fg=ruler_colour, bg=(8, 13, 24), font=font_sm, bounds=bounds
    )
    occupied_rects.append(scale_rect)

    crest_uv = geom["crest_uv"]
    basal_uv = geom["basal_uv"]
    height_mm_val = geom["height_mm"]
    height_prov = geom["height_prov"]
    width_endpoints = geom["width_endpoints"]
    is_mand = bool(geom["is_mand"])

    # 1. Alveolar crest marker & detected ridge crest contour (when supplied by engine)
    if crest_uv is not None:
        crest_profile = geom["crest_profile_uv"]
        if crest_profile and len(crest_profile) >= 2:
            prof_px = [_px(pu, pv, u_coords, v_coords, scale) for pu, pv in crest_profile]
            for i_p in range(len(prof_px) - 1):
                draw.line([prof_px[i_p], prof_px[i_p + 1]], fill=(180, 145, 25), width=max(1, scale - 1))

        ep_x, ep_y = _px(crest_uv[0], crest_uv[1], u_coords, v_coords, scale)
        ep_r = max(3, int(round(1.8 * scale)))
        draw.ellipse(
            [ep_x - ep_r, ep_y - ep_r, ep_x + ep_r, ep_y + ep_r],
            outline=COLOR_CREST_MARKER,
            width=max(2, scale - 1),
        )

    # 2. Available Bone Height line: crest_uv -> basal_uv (NEVER replaced by root apex)
    if crest_uv is not None and basal_uv is not None and height_mm_val is not None and height_mm_val > 0.0:
        p0 = _px(crest_uv[0], crest_uv[1], u_coords, v_coords, scale)
        p1 = _px(basal_uv[0], basal_uv[1], u_coords, v_coords, scale)
        line_w = max(2, scale)
        cap_r = max(5, int(round(2.4 * scale)))
        is_estimated = height_prov in (PROVENANCE_ESTIMATED, PROVENANCE_INTERPOLATED)
        if is_estimated:
            _draw_dashed_segment(draw, p0, p1, fill=COLOR_BONE_HEIGHT, dash=6, gap=4, width=line_w)
        else:
            draw.line([p0, p1], fill=COLOR_BONE_HEIGHT, width=line_w)

        # Perpendicular end-caps aligned to the actual line angle
        ldx, ldy = p1[0] - p0[0], p1[1] - p0[1]
        ll = max(1e-6, math.hypot(ldx, ldy))
        perp_px = (-ldy / ll, ldx / ll)
        _draw_endcap(draw, p0, perp_px, cap_r, fill=COLOR_BONE_HEIGHT, width=max(2, line_w - 1))
        _draw_endcap(draw, p1, perp_px, cap_r, fill=COLOR_BONE_HEIGHT, width=max(2, line_w - 1))
        overlay.append((p0, p1))

        mid_hx = 0.5 * (p0[0] + p1[0])
        mid_hy = 0.5 * (p0[1] + p1[1])
        h_tag = f"H~:{height_mm_val:.1f} mm" if is_estimated else f"H:{height_mm_val:.1f} mm"
        _place_non_overlapping_pill(
            mid_hx + cap_r + 5,
            mid_hy - 8,
            h_tag,
            fg=COLOR_BONE_HEIGHT_TEXT,
            border=COLOR_BONE_HEIGHT,
            font=font_lg,
        )

        # Ridge inclination badge when non-vertical
        ridge_ang = geom["ridge_angle"]
        if show_implant_guides and ridge_ang is not None and abs(float(ridge_ang)) >= 2.0:
            _place_non_overlapping_pill(
                width - 96.0,
                26.0,
                f"Ang {float(ridge_ang):+.0f}deg",
                fg=(226, 232, 240),
                border=(100, 116, 139),
                font=font_sm,
            )

    # 4. Buccolingual Width calipers strictly from engine endpoints (ridge-aligned)
    has_w5 = 5.0 in width_endpoints
    for depth in sorted(width_endpoints.keys()):
        pt_l, pt_r, w_val, w_prov = width_endpoints[depth]
        start = _px(pt_l[0], pt_l[1], u_coords, v_coords, scale)
        end = _px(pt_r[0], pt_r[1], u_coords, v_coords, scale)
        is_primary = depth in (0.0, 3.0, 5.0) or (depth == 6.0 and not has_w5)
        # W3/W5 are context, not the primary crest measurement: draw them semi-transparent.
        if depth == 0.0:
            colour, alpha = COLOR_WIDTH_W0, WIDTH_ALPHA_W0
        elif depth <= 3.5:
            colour, alpha = COLOR_WIDTH_W3, WIDTH_ALPHA_W3
        else:
            colour, alpha = COLOR_WIDTH_DEEP, WIDTH_ALPHA_W5
        stroke = (int(colour[0]), int(colour[1]), int(colour[2]), int(alpha))

        w_line = max(2, scale) if is_primary else 1
        tick_r = max(4, int(round(1.8 * scale))) if is_primary else 2
        if w_prov in (PROVENANCE_ESTIMATED, PROVENANCE_INTERPOLATED):
            _draw_dashed_segment(draw, start, end, fill=stroke, dash=5, gap=3, width=w_line)
        else:
            draw.line([start, end], fill=stroke, width=w_line)

        wdx, wdy = end[0] - start[0], end[1] - start[1]
        wl = max(1e-6, math.hypot(wdx, wdy))
        w_perp = (-wdy / wl, wdx / wl)
        _draw_endcap(draw, start, w_perp, tick_r, fill=stroke, width=max(2, w_line - 1))
        _draw_endcap(draw, end, w_perp, tick_r, fill=stroke, width=max(2, w_line - 1))
        overlay.append((start, end))

        if is_primary and w_val is not None:
            lbl_str = f"W0 {w_val:.1f} mm" if depth == 0.0 else f"W{depth:g} {w_val:.1f} mm"
            if depth == 3.0:
                pref_x = min(start[0], end[0]) - _text_size(draw, lbl_str, font_md)[0] - 12
            else:
                pref_x = max(start[0], end[0]) + 6
            _place_non_overlapping_pill(
                pref_x,
                0.5 * (start[1] + end[1]) - 8,
                lbl_str,
                fg=colour,
                border=colour,
                font=font_md,
            )

    # 5. Vital anatomical structures (actual segmentation contour + marker)
    canal = geom["canal_centroid_uv"]
    vital_contour = geom["vital_contour_uv"]
    if is_mand and (vital_contour or canal is not None):
        if vital_contour and len(vital_contour) >= 3:
            c_px = [_px(float(cu), float(cv), u_coords, v_coords, scale) for cu, cv in vital_contour]
            for idx_c in range(len(c_px)):
                draw.line(
                    [c_px[idx_c], c_px[(idx_c + 1) % len(c_px)]], fill=COLOR_VITAL_CANAL, width=max(2, scale)
                )
        if canal is not None:
            cu, cv = canal
            cx, cy = _px(cu, cv, u_coords, v_coords, scale)
            radius_mm = 1.5
            radius_px = (
                radius_mm / max(1e-6, (u_coords[-1] - u_coords[0]) / max(1, len(u_coords) - 1)) * scale
            )
            if not (vital_contour and len(vital_contour) >= 3):
                draw.ellipse(
                    [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
                    outline=COLOR_VITAL_CANAL,
                    width=max(2, scale),
                )
            cross_r = max(2.0, radius_px * 0.45)
            draw.line([(cx - cross_r, cy), (cx + cross_r, cy)], fill=(255, 100, 100), width=1)
            draw.line([(cx, cy - cross_r), (cx, cy + cross_r)], fill=(255, 100, 100), width=1)
            _place_non_overlapping_pill(
                cx + radius_px + 5,
                cy - 8,
                "IAN",
                fg=(255, 96, 96),
                border=(239, 68, 68),
                font=font_md,
            )

    sinus_pts = geom["sinus_floor_uv"]
    if sinus_pts and len(sinus_pts) >= 2:
        px_pts = [_px(float(su), float(sv), u_coords, v_coords, scale) for su, sv in sinus_pts]
        for i_s in range(len(px_pts) - 1):
            draw.line([px_pts[i_s], px_pts[i_s + 1]], fill=COLOR_VITAL_SINUS, width=max(2, scale))
        mid_sx, mid_sy = px_pts[len(px_pts) // 2]
        _place_non_overlapping_pill(
            mid_sx + 6,
            mid_sy - 14,
            "SINUS",
            fg=(249, 168, 212),
            border=(236, 72, 153),
            font=font_sm,
        )

    bone_density = geom["misch_code"] if show_implant_guides else None

    # Crisp dark clinical footer bar at the bottom of the tile
    draw.rectangle([(0, height - footer_h), (width, height)], fill=(2, 4, 9))
    draw.line([(0, height - footer_h), (width, height - footer_h)], fill=(30, 41, 59), width=1)
    if meas is None and (height_mm_val is None or height_mm_val <= 0.0) and not width_endpoints:
        draw.text((6, height - footer_h + 4), "not measurable", fill=COLOR_UNAVAILABLE, font=font_md)
        labels.append((6.0, float(height - footer_h + 4), "not measurable"))
    else:
        if meas is not None:
            text = measurement_footer_text(meas, bone_density=bone_density)
        else:
            h_txt = f"{height_mm_val:.1f} mm" if height_mm_val is not None else "--"
            w0_val = width_endpoints[0.0][2] if 0.0 in width_endpoints else None
            w_txt = f"{w0_val:.1f} mm" if w0_val is not None else "--"
            text = f"H {h_txt}  W0 {w_txt}" + (f"  {bone_density}" if bone_density else "")
        draw.text((6, height - footer_h + 4), text, fill=(248, 250, 252), font=font_md)
        labels.append((6.0, float(height - footer_h + 4), text))

    image = Image.alpha_composite(base_image, layer).convert("RGB")
    if monochrome:
        gray = image.convert("L")
        image = Image.merge("RGB", (gray, gray, gray))

    return CrossSectionRender(
        image=image,
        overlay_px=overlay,
        labels=labels,
        mm_per_px=MM_PER_COL / float(scale),
        available=True,
    )


def render_panoramic_strip(
    pano_array: np.ndarray,
    *,
    column_spacing_mm: float,
    tick_arcs_mm: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    window: Optional[float] = None,
    level: Optional[float] = None,
    strip_height: int = 340,
    scale: int = 2,
    flip_vertical: bool = True,
    show_cross_section_lines: bool = True,
    highlight_indices: Optional[Sequence[int]] = None,
    monochrome: bool = False,
) -> CrossSectionRender:
    """Render a large, upright panoramic strip with full-height cross-section reference lines."""
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    arr = np.asarray(pano_array)
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            return CrossSectionRender(None, available=False, reason="empty panoramic volume")
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        return CrossSectionRender(None, available=False, reason="invalid panoramic array")

    if flip_vertical:
        arr = arr[::-1, :]

    if window is None or level is None:
        lo, hi = np.percentile(arr.astype(np.float32), [1.0, 99.0])
        level = float((lo + hi) / 2.0)
        window = float(max(1.0, hi - lo))
    gray = _enhance_bone_clarity(hu_window_to_rgb(arr, window=window, level=level), strength=0.30)
    image = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    img_w = max(1, arr.shape[1] * max(1, int(scale)))
    img_h = max(1, int(strip_height))
    image = image.resize((img_w, img_h), getattr(resampling, "BICUBIC", Image.BILINEAR))
    draw = ImageDraw.Draw(image)
    font = _load_font(max(13, int(round(img_h * 0.058))), bold=True)
    ticks: List[Tuple[float, float, str]] = []
    overlay: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    tick_bar_h = max(26, int(round(img_h * 0.11)))
    draw.rectangle([(0, img_h - tick_bar_h), (img_w, img_h)], fill=(2, 4, 9))
    _draw_pill_label(draw, 8, 8, "R", fg=COLOR_IMPLANT, bg=(2, 4, 9), border=COLOR_IMPLANT, font=font)
    _draw_pill_label(
        draw, max(8, img_w - 30), 8, "L", fg=COLOR_IMPLANT, bg=(2, 4, 9), border=COLOR_IMPLANT, font=font
    )
    total_arc_mm = float(arr.shape[1] - 1) * max(1e-6, float(column_spacing_mm))
    for mm_mark in range(0, int(total_arc_mm) + 1, 10):
        rx = (float(mm_mark) / max(1e-6, float(column_spacing_mm))) * max(1, int(scale))
        if 0 <= rx <= img_w - 1:
            draw.line([(rx, img_h - tick_bar_h), (rx, img_h - tick_bar_h + 5)], fill=(100, 116, 139), width=1)
    draw.line([(0, img_h - tick_bar_h), (img_w, img_h - tick_bar_h)], fill=(30, 41, 59), width=1)

    hi_set = set(highlight_indices or ())
    last_label_right = -1e9
    for index, arc_mm in enumerate(tick_arcs_mm):
        col = float(arc_mm) / max(1e-6, float(column_spacing_mm))
        if col < 0 or col > arr.shape[1] - 1:
            continue
        x = col * max(1, int(scale))
        is_hi = index in hi_set
        line_col = COLOR_BONE_HEIGHT_TEXT if is_hi else COLOR_WIDTH_W0
        line_w = max(2, int(scale)) if is_hi else max(1, int(scale) - 1)

        if show_cross_section_lines:
            if is_hi:
                draw.line([(x, 0), (x, img_h - tick_bar_h)], fill=line_col, width=line_w)
            else:
                _draw_dashed_vline(draw, x, 0, img_h - tick_bar_h, fill=line_col, dash=8, gap=5, width=line_w)
            overlay.append(((x, 0.0), (x, float(img_h - tick_bar_h))))
        else:
            draw.line(
                [(x, img_h - tick_bar_h - 10), (x, img_h - tick_bar_h)], fill=COLOR_BONE_HEIGHT, width=line_w
            )
            draw.line([(x, 0), (x, max(12, tick_bar_h // 2))], fill=COLOR_WIDTH_W0, width=line_w)

        draw.line([(x, 0), (x, 10)], fill=line_col, width=max(2, line_w))
        draw.line(
            [(x, img_h - tick_bar_h - 10), (x, img_h - tick_bar_h)], fill=line_col, width=max(2, line_w)
        )

        label = labels[index] if labels is not None and index < len(labels) else f"#{index + 1}"
        lbl_str = str(label)
        tw, _ = _text_size(draw, lbl_str, font)
        if lbl_str and x >= last_label_right + 6:
            draw.text(
                (min(img_w - tw - 2, x + 3), img_h - tick_bar_h + 4),
                lbl_str,
                fill=COLOR_BONE_HEIGHT_TEXT if is_hi else (226, 232, 240),
                font=font,
            )
            last_label_right = x + 3 + tw
        ticks.append((x, float(img_h - tick_bar_h + 4), lbl_str))

    if monochrome:
        gray = image.convert("L")
        image = Image.merge("RGB", (gray, gray, gray))
    return CrossSectionRender(image=image, overlay_px=overlay, labels=ticks, available=True)


def unavailable_tile(text: str, size: Tuple[int, int] = (640, 400), dark_film: bool = False) -> Any:
    """Render an explicit 'unavailable' tile; never fake clinical content."""
    if not _PIL_AVAILABLE:
        return None
    background = (15, 23, 42) if not dark_film else (245, 245, 245)
    foreground = (226, 232, 240) if not dark_film else (15, 23, 42)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    font = _load_font(max(14, size[1] // 20), bold=True)
    msg = f"{text} unavailable"
    tw, th = _text_size(draw, msg, font)
    draw.text(((size[0] - tw) // 2, (size[1] - th) // 2), msg, fill=foreground, font=font)
    return image


__all__ = [
    "CrossSectionRender",
    "SAFETY_MARGIN_MM",
    "hu_window_to_rgb",
    "measurement_footer_text",
    "render_cross_section",
    "render_panoramic_strip",
    "unavailable_tile",
]

"""Clinical renderers for cross-sections and the panoramic strip.

Pure numpy geometry with Pillow (and automatic Qt fallback in 3D Slicer). Every overlay
is derived from measured millimetre coordinates:
  - Green vertical Bone Height line (`H`) from alveolar crest (`crest_v`) to basal/canal limit (`basal_v`)
  - Cyan `W0` Crestal Buccolingual Width line
  - Gold-orange `W3` (3 mm depth) and orange `W5` (5+ mm depth) Buccolingual Width lines
    (legend colours in PrintLayoutEngine must match the RGB tuples below)
  - Red Inferior Alveolar Nerve Canal circle + clearance indicator
  - Subtle anatomical contours for segmented teeth/roots and jawbone cortical plates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Pillow is optional in Slicer; callers render an "unavailable" tile when missing.
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without Pillow
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

MM_PER_ROW = 0.25
MM_PER_COL = 0.25

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
    px = max(8, int(size))
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
        return max(1, len(text) * 6), 11


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
    pad_x: int = 3,
    pad_y: int = 1,
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
    rect = (px, py, px + box_w, py + box_h)
    draw.rectangle(rect, fill=bg, outline=border, width=1 if border else 0)
    draw.text((px + pad_x, py + pad_y - 1), text, fill=fg, font=font)
    return rect


def _draw_dashed_hline(
    draw: Any,
    x0: float,
    x1: float,
    y: float,
    *,
    fill: Tuple[int, int, int],
    dash: int = 5,
    gap: int = 4,
    width: int = 1,
) -> None:
    lo, hi = (min(x0, x1), max(x0, x1))
    cur = lo
    while cur < hi:
        seg_end = min(hi, cur + dash)
        draw.line([(cur, y), (seg_end, y)], fill=fill, width=width)
        cur += dash + gap


@dataclass
class CrossSectionRender:
    image: Any
    overlay_px: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    labels: List[Tuple[float, float, str]] = field(default_factory=list)
    mm_per_px: float = MM_PER_COL
    available: bool = True
    reason: Optional[str] = None


def hu_window_to_rgb(img_cs: np.ndarray, window: float = 4000.0, level: float = 1000.0) -> np.ndarray:
    """Map HU values to 8-bit grayscale using a window/level (with adaptive fallback for non-HU arrays)."""
    arr = np.asarray(img_cs, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low = float(level) - float(window) / 2.0
    high = float(level) + float(window) / 2.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    # If standard HU window leaves image flat (e.g. non-HU gray values), adapt to [p2, p99]
    if float(np.std(scaled)) < 0.01 and float(np.max(arr) - np.min(arr)) > 1.0:
        p2, p99 = np.percentile(arr, [2.0, 99.0])
        denom = max(1.0, float(p99 - p2))
        scaled = np.clip((arr - float(p2)) / denom, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _px(u: float, v: float, u_coords: np.ndarray, v_coords: np.ndarray, scale: int) -> Tuple[float, float]:
    col = (float(u) - float(u_coords[0])) / (float(u_coords[-1]) - float(u_coords[0]))
    row = (float(v_coords[-1]) - float(v)) / (float(v_coords[-1]) - float(v_coords[0]))
    width = (len(u_coords) - 1) * scale
    height = (len(v_coords) - 1) * scale
    return col * width, row * height


def _apply_anatomical_contour_tint(rgb: np.ndarray, mask_cs: Optional[np.ndarray]) -> np.ndarray:
    """Blend subtle vivid SlicerDentalSegmentator boundary contours for teeth/roots, bone, and canal."""
    if mask_cs is None or mask_cs.shape != rgb.shape[:2]:
        return rgb
    out = rgb.astype(np.float32)
    m = np.asarray(mask_cs)
    # Edge detection via 4-neighbor difference
    edge = np.zeros(m.shape, dtype=bool)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]

    # Tooth & root contour: vivid golden-ivory (labels 3, 4, or 11..48)
    tooth_edge = edge & ((m == 3) | (m == 4) | ((m >= 11) & (m <= 48)))
    if np.any(tooth_edge):
        out[tooth_edge] = 0.35 * out[tooth_edge] + 0.65 * np.array([255.0, 215.0, 90.0], dtype=np.float32)

    # Jawbone cortical contour: subtle teal-emerald (labels 1, 2)
    jaw_edge = edge & ((m == 1) | (m == 2)) & (~tooth_edge)
    if np.any(jaw_edge):
        out[jaw_edge] = 0.55 * out[jaw_edge] + 0.45 * np.array([80.0, 220.0, 180.0], dtype=np.float32)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def measurement_footer_text(meas: Optional[Any]) -> str:
    """Return the tile footer `H … / W0 …` caption, printing `--` for a withheld metric.

    A withheld crest width must never render as a fabricated `W0 0.0 mm`.
    """
    if meas is None:
        return "H --  W0 --"
    height_mm = getattr(meas, "bone_height_mm", None)
    crest_w = getattr(meas, "crest_width_mm", None)
    h_txt = f"{float(height_mm):.1f} mm" if height_mm is not None else "--"
    w_txt = f"{float(crest_w):.1f} mm" if crest_w is not None else "--"
    return f"H {h_txt}  W0 {w_txt}"


def render_cross_section(
    cs: Any,
    meas: Optional[Any],
    *,
    scale: int = 3,
    window: float = 4000.0,
    level: float = 1000.0,
    dark_film: bool = False,
) -> CrossSectionRender:
    """Render one arch cross-section with real millimetre overlays.

    `cs` is a `CrossSectionData` (duck-typed to keep this module Slicer-free).
    When Pillow is unavailable the result is explicitly unavailable.
    """
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    if cs is None:
        return CrossSectionRender(None, available=False, reason="no cross-section data")

    u_coords = np.asarray(cs.u_coords, dtype=np.float64)
    v_coords = np.asarray(cs.v_coords, dtype=np.float64)

    # `cs.img_cs[iv, iu]` is indexed by `v_coords[iv]` (ascending inferior -> superior),
    # whereas `_px()` maps `v_coords[-1]` (superior) to image row 0 (top). Flip vertically
    # so the CBCT anatomy, contour mask, and every measurement line share one exact frame.
    img_arr = np.asarray(cs.img_cs)
    mask_arr = getattr(cs, "mask_cs", None)
    if len(v_coords) >= 2 and float(v_coords[-1]) > float(v_coords[0]):
        img_arr = img_arr[::-1, :]
        if mask_arr is not None:
            mask_arr = np.asarray(mask_arr)[::-1, :]

    gray = hu_window_to_rgb(img_arr, window=window, level=level)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if dark_film:
        rgb = 255 - rgb
    rgb = _apply_anatomical_contour_tint(rgb, mask_arr)

    image = Image.fromarray(rgb, mode="RGB")
    width = max(1, (len(u_coords) - 1) * scale)
    height = max(1, (len(v_coords) - 1) * scale)
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((width, height), getattr(resampling, "BICUBIC", Image.BILINEAR))
    draw = ImageDraw.Draw(image)

    font_sm = _load_font(max(9, int(round(4.0 * scale))), bold=False)
    font_md = _load_font(max(10, int(round(4.5 * scale))), bold=True)
    footer_h = max(18, int(round(8.0 * scale)))
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
        bw, bh = tw + 6, th + 2
        candidates = [
            (pref_x, pref_y),
            (pref_x, pref_y - bh - 3),
            (pref_x, pref_y + bh + 3),
            (max(18.0, pref_x - bw - 8), pref_y),
            (min(float(width - bw - 3), pref_x + bw + 8), pref_y),
        ]
        chosen = candidates[0]
        for cx, cy in candidates:
            cx_c = min(max(16.0, cx), max(16.0, float(width - bw - 2)))
            cy_c = min(max(16.0, cy), max(16.0, float(safe_h - bh - 2)))
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
        return rect

    # Anatomical orientation badges: Lingual (L) left, Buccal (B) right, Superior (S) top-center.
    l_rect = _draw_pill_label(draw, 3, 3, "L", fg=(74, 222, 128), border=(34, 197, 94), font=font_md, bounds=bounds)
    b_tw, _ = _text_size(draw, "B", font_md)
    b_rect = _draw_pill_label(
        draw, max(3, width - b_tw - 9), 3, "B", fg=(74, 222, 128), border=(34, 197, 94), font=font_md, bounds=bounds
    )
    s_tw, _ = _text_size(draw, "S", font_sm)
    s_rect = _draw_pill_label(
        draw, max(3, (width - s_tw - 6) // 2), 3, "S", fg=(148, 163, 184), font=font_sm, bounds=bounds
    )
    occupied_rects.extend([l_rect, b_rect, s_rect])
    labels.extend([(3.0, 3.0, "L"), (float(max(3, width - b_tw - 9)), 3.0, "B")])

    # Millimetre ruler (left edge) and 10 mm scale bar, drawn on every tile so the image
    # carries a measurement reference even when no metric could be computed.
    ruler_colour = (245, 245, 245) if dark_film else (195, 210, 225)
    v_min, v_max = float(v_coords.min()), float(v_coords.max())
    for tick_mm in np.arange(np.ceil(v_min / 5.0) * 5.0, v_max, 5.0):
        _, tick_y = _px(u_coords[0], float(tick_mm), u_coords, v_coords, scale)
        if tick_y < 18 or tick_y > safe_h - 18:
            continue
        major = abs(float(tick_mm) % 10.0) < 1e-6
        tick_len = max(9, 4 * scale) if major else max(5, 2 * scale)
        draw.line([(0, tick_y), (tick_len, tick_y)], fill=ruler_colour, width=1)
        if major:
            draw.text((tick_len + 2, tick_y - 5), f"{float(tick_mm):+.0f}", fill=ruler_colour, font=font_sm)

    bar_px = 10.0 / max(1e-6, abs(u_coords[-1] - u_coords[0])) * width
    bar_y = max(24, safe_h - max(12, int(round(5.0 * scale))))
    bar_x0 = 8.0
    bar_x1 = bar_x0 + bar_px
    draw.line([(bar_x0, bar_y), (bar_x1, bar_y)], fill=ruler_colour, width=2)
    draw.line([(bar_x0, bar_y - 4), (bar_x0, bar_y + 4)], fill=ruler_colour, width=1)
    draw.line([(bar_x1, bar_y - 4), (bar_x1, bar_y + 4)], fill=ruler_colour, width=1)
    scale_rect = _draw_pill_label(
        draw, bar_x1 + 4, bar_y - 7, "10 mm", fg=ruler_colour, bg=(8, 13, 24), font=font_sm, bounds=bounds
    )
    occupied_rects.append(scale_rect)

    crest_v = getattr(cs, "crest_v", None)
    segments: Dict[float, Tuple[float, float]] = getattr(cs, "width_segments", {}) or {}
    is_mand = bool(getattr(cs, "is_mandible", True))

    # Subtle dashed crest reference guide (avoids colliding with the solid cyan W0 bar).
    if crest_v is not None:
        x_left, y_crest = _px(u_coords[0], crest_v, u_coords, v_coords, scale)
        x_right, _ = _px(u_coords[-1], crest_v, u_coords, v_coords, scale)
        if 0.0 in segments:
            w0_l, w0_r = segments[0.0]
            wx0, _ = _px(w0_l, crest_v, u_coords, v_coords, scale)
            wx1, _ = _px(w0_r, crest_v, u_coords, v_coords, scale)
            _draw_dashed_hline(draw, x_left + 12, max(x_left + 12, wx0 - 4), y_crest, fill=(250, 204, 21))
            _draw_dashed_hline(draw, min(x_right - 4, wx1 + 4), x_right - 4, y_crest, fill=(250, 204, 21))
        else:
            _draw_dashed_hline(draw, x_left + 12, x_right - 4, y_crest, fill=(250, 204, 21))
        overlay.append(((x_left, y_crest), (x_right, y_crest)))

    # Draw vertical Green Bone Height (H) caliper from crest_v to basal_v.
    height_mm_val = getattr(meas, "bone_height_mm", None) if meas is not None else None
    if crest_v is not None and height_mm_val is not None and float(height_mm_val) > 0.0:
        basal_v = getattr(cs, "basal_v", None)
        if basal_v is None:
            basal_v = float(crest_v) - float(height_mm_val) if is_mand else float(crest_v) + float(height_mm_val)
        h_u = float(getattr(cs, "height_u", 0.0))
        hx0, hy0 = _px(h_u, crest_v, u_coords, v_coords, scale)
        hx1, hy1 = _px(h_u, basal_v, u_coords, v_coords, scale)
        line_w = max(2, scale)
        cap_r = max(4, int(round(2.0 * scale)))
        draw.line([(hx0, hy0), (hx1, hy1)], fill=(34, 197, 94), width=line_w)
        draw.line([(hx0 - cap_r, hy0), (hx0 + cap_r, hy0)], fill=(34, 197, 94), width=max(2, line_w - 1))
        draw.line([(hx1 - cap_r, hy1), (hx1 + cap_r, hy1)], fill=(34, 197, 94), width=max(2, line_w - 1))
        overlay.append(((hx0, hy0), (hx1, hy1)))
        mid_hy = 0.5 * (hy0 + hy1)
        _place_non_overlapping_pill(
            hx0 + cap_r + 4,
            mid_hy - 6,
            f"H {float(height_mm_val):.1f}",
            fg=(74, 222, 128),
            border=(34, 197, 94),
            font=font_md,
        )

    # Draw horizontal Buccolingual Width calipers (W0 cyan, W3 gold-orange, W5 orange).
    # Suppress cluttered W6/W9 text labels when W5 is already present so labels never collide.
    has_w5 = 5.0 in segments
    for depth in sorted(segments.keys()):
        if crest_v is None:
            break
        u_left, u_right = segments[depth]
        v_level = crest_v - float(depth) if is_mand else crest_v + float(depth)
        start = _px(u_left, v_level, u_coords, v_coords, scale)
        end = _px(u_right, v_level, u_coords, v_coords, scale)
        is_primary = depth in (0.0, 3.0, 5.0) or (depth == 6.0 and not has_w5)
        if depth == 0.0:
            colour = (0, 220, 255)
        elif depth <= 3.5:
            colour = (255, 180, 0)
        else:
            colour = (255, 120, 40)

        w_line = max(2, scale) if is_primary else 1
        tick_r = max(3, int(round(1.5 * scale))) if is_primary else 2
        draw.line([start, end], fill=colour, width=w_line)
        draw.line([(start[0], start[1] - tick_r), (start[0], start[1] + tick_r)], fill=colour, width=max(1, w_line - 1))
        draw.line([(end[0], end[1] - tick_r), (end[0], end[1] + tick_r)], fill=colour, width=max(1, w_line - 1))
        overlay.append((start, end))

        if is_primary:
            w_val = abs(float(u_right) - float(u_left))
            lbl_str = f"W{depth:g} {w_val:.1f}" if width >= 160 else f"W{depth:g}"
            # Alternate preferred side (right for W0/W5, left for W3) to keep calipers uncluttered.
            if depth == 3.0:
                pref_x = start[0] - _text_size(draw, lbl_str, font_sm)[0] - 10
            else:
                pref_x = end[0] + 5
            _place_non_overlapping_pill(
                pref_x,
                start[1] - 6,
                lbl_str,
                fg=colour,
                border=colour,
                font=font_sm,
            )

    canal = getattr(cs, "canal_centroid_uv", None)
    if canal is not None:
        cu, cv = canal
        cx, cy = _px(cu, cv, u_coords, v_coords, scale)
        radius_mm = 1.5
        radius_px = radius_mm / max(1e-6, (u_coords[-1] - u_coords[0]) / max(1, len(u_coords) - 1)) * scale
        draw.ellipse(
            [cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px],
            outline=(255, 60, 60),
            width=max(2, scale),
        )
        # Subtle center crosshair inside the IAN ring
        cross_r = max(2.0, radius_px * 0.45)
        draw.line([(cx - cross_r, cy), (cx + cross_r, cy)], fill=(255, 100, 100), width=1)
        draw.line([(cx, cy - cross_r), (cx, cy + cross_r)], fill=(255, 100, 100), width=1)
        _place_non_overlapping_pill(
            cx + radius_px + 4,
            cy - 6,
            "IAN",
            fg=(255, 96, 96),
            border=(239, 68, 68),
            font=font_md,
        )

    # Crisp dark clinical footer bar at the bottom of the tile
    draw.rectangle([(0, height - footer_h), (width, height)], fill=(10, 15, 28))
    draw.line([(0, height - footer_h), (width, height - footer_h)], fill=(30, 41, 59), width=1)
    if meas is None:
        draw.text((6, height - footer_h + 3), "not measurable", fill=(248, 113, 113), font=font_md)
        labels.append((6.0, float(height - footer_h + 3), "not measurable"))
    else:
        text = measurement_footer_text(meas)
        draw.text((6, height - footer_h + 3), text, fill=(248, 250, 252), font=font_md)
        labels.append((6.0, float(height - footer_h + 3), text))

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
    strip_height: int = 160,
    scale: int = 1,
) -> CrossSectionRender:
    """Render a panoramic strip from the actual panoramic volume array.

    Column `k` maps to arc length `k * column_spacing_mm`. No screenshot or
    placeholder is used. Returns an explicitly unavailable render on bad input.
    """
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    arr = np.asarray(pano_array)
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            return CrossSectionRender(None, available=False, reason="empty panoramic volume")
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        return CrossSectionRender(None, available=False, reason="invalid panoramic array")

    if window is None or level is None:
        lo, hi = np.percentile(arr.astype(np.float32), [1.0, 99.0])
        level = float((lo + hi) / 2.0)
        window = float(max(1.0, hi - lo))
    gray = hu_window_to_rgb(arr, window=window, level=level)
    image = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    img_w = max(1, arr.shape[1] * scale)
    img_h = max(1, strip_height)
    image = image.resize((img_w, img_h), getattr(resampling, "BICUBIC", Image.BILINEAR))
    draw = ImageDraw.Draw(image)
    font = _load_font(max(10, int(round(img_h * 0.065))), bold=True)
    ticks: List[Tuple[float, float, str]] = []

    tick_bar_h = max(16, int(round(img_h * 0.11)))
    draw.rectangle([(0, img_h - tick_bar_h), (img_w, img_h)], fill=(8, 13, 24))

    last_label_right = -1e9
    for index, arc_mm in enumerate(tick_arcs_mm):
        col = float(arc_mm) / float(column_spacing_mm)
        if col < 0 or col > arr.shape[1] - 1:
            continue
        x = col * scale
        draw.line([(x, img_h - tick_bar_h - 8), (x, img_h - tick_bar_h)], fill=(34, 197, 94), width=max(1, scale))
        draw.line([(x, 0), (x, max(10, tick_bar_h // 2))], fill=(0, 220, 255), width=max(1, scale))
        label = labels[index] if labels is not None and index < len(labels) else ""
        lbl_str = str(label)
        tw, _ = _text_size(draw, lbl_str, font)
        if lbl_str and x >= last_label_right + 4:
            draw.text((x + 2, img_h - tick_bar_h + 2), lbl_str, fill=(74, 222, 128), font=font)
            last_label_right = x + 2 + tw
        ticks.append((x, float(img_h - tick_bar_h + 2), lbl_str))

    return CrossSectionRender(image=image, labels=ticks, available=True)


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
    "hu_window_to_rgb",
    "measurement_footer_text",
    "render_cross_section",
    "render_panoramic_strip",
    "unavailable_tile",
]

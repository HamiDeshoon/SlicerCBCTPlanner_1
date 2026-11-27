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

from .core.labels import (
    CANAL_IDS,
    DENTALSEGMENTATOR_CANAL_ID,
    infer_mask_is_5class,
    jaw_mask_from_labels,
    tooth_mask_from_labels,
)
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


def is_estimated_provenance(provenance: Optional[str]) -> bool:
    """Return True when a record's provenance is not label-exact (drawn/labelled with `~`)."""
    return str(provenance or "") in (PROVENANCE_ESTIMATED, PROVENANCE_INTERPOLATED)


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

    # 2. Width endpoints (prefer exact 2D ridge-aligned endpoints, fallback to legacy width_segments)
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

    # 3. Approximate bone density code (exclusively from engine — never recalculated here!)
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
        "width_endpoints": width_endpoints,
        "canal_centroid_uv": getattr(cs, "canal_centroid_uv", None),
        "vital_contour_uv": getattr(cs, "vital_contour_uv", None),
        "sinus_floor_uv": getattr(cs, "sinus_floor_uv", None),
        "misch_code": misch_code,
        "ridge_angle": getattr(cs, "ridge_axis_angle_deg", None),
        "withheld_reason": (
            getattr(meas, "withheld_reason", None)
            if meas is not None
            else getattr(cs, "withheld_reason", None)
        ),
    }


def select_periodic_depth_key(
    slice_index: int,
    total_slices: int,
    *,
    is_center: bool = False,
    stride: int = 4,
) -> Tuple[float, ...]:
    """Select at most one depth (`W3` or `W5`) every `stride` slices to avoid repeating crowded tables."""
    if total_slices <= 1:
        return (3.0,)
    step = max(2, int(stride))
    if slice_index % step == 0:
        slot = slice_index // step
        return (3.0,) if (slot % 2 == 0) else (5.0,)
    if is_center:
        return (3.0,)
    return ()


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
    show_footer: bool = True,
    inline_line_labels: bool = False,
    custom_overlays: Optional[Sequence[Dict[str, Any]]] = None,
    show_depth_key: bool = True,
    depth_key_depths: Optional[Sequence[float]] = None,
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
    img_w = max(1, (len(u_coords) - 1) * scale)
    img_h = max(1, (len(v_coords) - 1) * scale)
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((img_w, img_h), getattr(resampling, "BICUBIC", Image.BILINEAR))

    font_sm = _load_font(max(11, int(round(4.6 * scale))), bold=True)
    font_md = _load_font(max(13, int(round(5.4 * scale))), bold=True)
    font_lg = _load_font(max(14, int(round(6.2 * scale))), bold=True)
    footer_h = max(24, int(round(9.5 * scale))) if show_footer else 0
    # The footer gets its OWN band below the anatomy: the tile is taller than the image, so a
    # caption or a deep basal end-cap can never be hidden behind the footer text.
    width = img_w
    height = img_h + footer_h
    safe_h = img_h
    bounds = (width, safe_h)
    # Reserved right margin for measurement captions: they sit outside the tooth/jaw silhouette
    # instead of on top of it.
    caption_gutter = max(64, int(round(14.0 * scale)))
    caption_cursor = [26.0]

    # Draw every overlay on an RGBA layer so W3/W5 can be genuinely semi-transparent and
    # still read over bright cortical bone; the layer is composited once at the end.
    base_image = Image.new("RGBA", (width, height), (2, 4, 9, 255))
    base_image.paste(image.convert("RGBA"), (0, 0))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    overlay: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    labels: List[Tuple[float, float, str]] = []
    occupied_rects: List[Tuple[float, float, float, float]] = []

    def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        return not (a[2] + 2 < b[0] or b[2] + 2 < a[0] or a[3] + 2 < b[1] or b[3] + 2 < a[1])

    def _clamp_pill(cx: float, cy: float, bw: int, bh: int) -> Tuple[float, float]:
        max_x = max(18.0, float(width - bw - 4))
        max_y = max(24.0, float(safe_h - bh - 4))
        return min(max(18.0, float(cx)), max_x), min(max(24.0, float(cy)), max_y)

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
            (min(float(width - bw - 4), pref_x + bw + 10), pref_y),
            (max(20.0, pref_x - bw - 10), pref_y - bh - 4),
        ]
        chosen = _clamp_pill(pref_x, pref_y, bw, bh)
        for cx, cy in candidates:
            cx_c, cy_c = _clamp_pill(cx, cy, bw, bh)
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
            bounds=(max(1, width - 3), max(1, safe_h - 2)),
        )
        occupied_rects.append(rect)
        labels.append((chosen[0], chosen[1], text))
        return rect

    def _draw_leader_to_anchor(
        rect: Tuple[float, float, float, float],
        anchor: Tuple[float, float],
        *,
        fg: Tuple[int, int, int],
    ) -> None:
        """Draw a crisp hairline leader from the nearest pill edge to the anatomical anchor point."""
        ax, ay = float(anchor[0]), float(anchor[1])
        rx0, ry0, rx1, ry1 = rect
        edge_x = min(max(ax, rx0), rx1)
        edge_y = min(max(ay, ry0), ry1)
        dist = math.hypot(ax - edge_x, ay - edge_y)
        l_col = (int(fg[0]), int(fg[1]), int(fg[2]), 235)
        if dist >= 1.5:
            draw.line([(edge_x, edge_y), (ax, ay)], fill=l_col, width=max(1, min(2, scale - 1)))
        dot_r = max(1.8, 0.7 * float(scale))
        draw.ellipse([ax - dot_r, ay - dot_r, ax + dot_r, ay + dot_r], fill=l_col)

    def _place_caption(
        anchor: Tuple[float, float],
        text: str,
        *,
        fg: Tuple[int, int, int],
        border: Optional[Tuple[int, int, int]] = None,
        font: Any = None,
        connect_leader: bool = True,
    ) -> Tuple[float, float, float, float]:
        """Place a measurement caption attached directly to its geometry (or gutter with leader)."""
        if inline_line_labels:
            rect = _place_non_overlapping_pill(
                float(anchor[0]) + 6.0,
                float(anchor[1]) - 10.0,
                text,
                fg=fg,
                border=border,
                font=font,
            )
            if connect_leader:
                _draw_leader_to_anchor(rect, anchor, fg=fg)
            return rect
        tw, th = _text_size(draw, text, font)
        bw, bh = tw + 8, th + 4
        px = max(2.0, float(width - caption_gutter - bw + 2))
        py = float(caption_cursor[0])
        while any(_rects_overlap((px, py, px + bw, py + bh), r) for r in occupied_rects):
            py += bh + 3
        py = float(min(max(24.0, py), max(24.0, float(safe_h - bh - 4))))
        rect = _draw_pill_label(
            draw,
            px,
            py,
            text,
            fg=fg,
            bg=(8, 13, 24),
            border=border,
            font=font,
            bounds=(max(1, width - 3), max(1, safe_h - 2)),
        )
        occupied_rects.append(rect)
        labels.append((px, py, text))
        caption_cursor[0] = py + bh + 3
        if connect_leader:
            _draw_leader_to_anchor(rect, anchor, fg=fg)
        else:
            stub = max(6.0, 3.0 * scale)
            draw.line(
                [(px - stub - 2, py + bh * 0.5), (px - 2, py + bh * 0.5)],
                fill=(int(fg[0]), int(fg[1]), int(fg[2]), 220),
                width=1,
            )
        return rect

    # Draw the bottom footer bar BEFORE calipers so a deep basal height end-cap is never occluded.
    if show_footer:
        draw.rectangle([(0, img_h), (width, height)], fill=(2, 4, 9, 220))
        draw.line([(0, img_h), (width, img_h)], fill=(30, 41, 59, 255), width=1)

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
    occupied_rects.append((bar_x0 - 2.0, bar_y - 5.0, bar_x1 + 2.0, bar_y + 5.0))
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

    # 2. Available Bone Height line: crest_uv (centered on W0 midline) -> basal_uv (NEVER replaced by root apex)
    if crest_uv is not None and basal_uv is not None and height_mm_val is not None and height_mm_val > 0.0:
        p0 = _px(crest_uv[0], crest_uv[1], u_coords, v_coords, scale)
        p1 = _px(basal_uv[0], basal_uv[1], u_coords, v_coords, scale)
        line_w = max(2, scale)
        cap_r = max(5, int(round(2.4 * scale)))
        is_estimated = is_estimated_provenance(height_prov)
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
        _place_caption(
            (mid_hx, mid_hy),
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
        if float(depth) > 0.0:
            # W3/W5 are reference widths, not crest geometry: they are drawn only inside the
            # corner depth key below, never as calipers across the tooth or jaw.
            continue
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
        if is_estimated_provenance(w_prov):
            _draw_dashed_segment(draw, start, end, fill=stroke, dash=5, gap=3, width=w_line)
        else:
            draw.line([start, end], fill=stroke, width=w_line)

        wdx, wdy = end[0] - start[0], end[1] - start[1]
        wl = max(1e-6, math.hypot(wdx, wdy))
        w_perp = (-wdy / wl, wdx / wl)
        _draw_endcap(draw, start, w_perp, tick_r, fill=stroke, width=max(2, w_line - 1))
        _draw_endcap(draw, end, w_perp, tick_r, fill=stroke, width=max(2, w_line - 1))
        overlay.append((start, end))

        if is_primary and w_val is not None and float(depth) == 0.0:
            _place_caption(
                (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])),
                f"W0 {w_val:.1f} mm",
                fg=colour,
                border=colour,
                font=font_md,
            )

    # Corner depth key: isolated periodic W3 or W5 caliper readout (never crowding every slice).
    if not show_depth_key:
        deep_depths: List[float] = []
    elif depth_key_depths is not None:
        deep_depths = [float(d) for d in depth_key_depths if float(d) > 0.0 and float(d) in width_endpoints]
    else:
        deep_depths = [d for d in sorted(width_endpoints.keys()) if float(d) > 0.0]
    if deep_depths:
        row_h = max(14, int(round(6.0 * scale)))
        box_w = max(78, int(round(28.0 * scale)))
        box_h = row_h * len(deep_depths) + 8
        min_x = 18.0 if width >= box_w + 24 else 4.0
        max_x = max(min_x, float(width - box_w - 4))
        min_y = 24.0 if safe_h >= box_h + 28 else 4.0
        max_y = max(min_y, float(safe_h - box_h - 4))
        left_x = min(max_x, min_x)
        bot_above_scale_y = max(min_y, min(max_y, float(bar_y - box_h - 10.0)))
        top_below_l_y = min(max_y, max(min_y, float(l_rect[3]) + 6.0))
        top_below_b_y = min(max_y, max(min_y, float(b_rect[3]) + 6.0))

        # Keep bottom-right default when clean; otherwise move to a non-overlapping corner.
        corner_candidates = [
            (max_x, max_y),
            (left_x, bot_above_scale_y),
            (left_x, top_below_l_y),
            (max_x, top_below_b_y),
        ]

        # Collect point samples from drawn caliper lines & vital structures to avoid occluding them.
        avoid_pts: List[Tuple[float, float]] = []
        for seg_p0, seg_p1 in overlay:
            for t_i in range(9):
                tf = t_i / 8.0
                avoid_pts.append(
                    (seg_p0[0] + tf * (seg_p1[0] - seg_p0[0]), seg_p0[1] + tf * (seg_p1[1] - seg_p0[1]))
                )
        if is_mand:
            if geom["canal_centroid_uv"] is not None:
                avoid_pts.append(
                    _px(geom["canal_centroid_uv"][0], geom["canal_centroid_uv"][1], u_coords, v_coords, scale)
                )
            for vu, vv in geom["vital_contour_uv"] or ():
                avoid_pts.append(_px(float(vu), float(vv), u_coords, v_coords, scale))
        for su, sv in geom["sinus_floor_uv"] or ():
            avoid_pts.append(_px(float(su), float(sv), u_coords, v_coords, scale))

        has_seg_mask = (
            mask_arr is not None and getattr(mask_arr, "ndim", 0) == 2 and bool(np.any(mask_arr > 0))
        )
        src_arr = mask_arr if has_seg_mask else gray

        def _corner_score(cx: float, cy: float) -> Tuple[int, int, int, float]:
            rect_c = (cx, cy, cx + box_w, cy + box_h)
            ov_area = 0.0
            for r in occupied_rects:
                ix = max(0.0, min(rect_c[2], r[2]) - max(rect_c[0], r[0]))
                iy = max(0.0, min(rect_c[3], r[3]) - max(rect_c[1], r[1]))
                ov_area += ix * iy
            pt_hits = sum(
                1
                for px_v, py_v in avoid_pts
                if (cx - 4.0) <= px_v <= (cx + box_w + 4.0) and (cy - 4.0) <= py_v <= (cy + box_h + 4.0)
            )
            anat_frac = 0.0
            if (
                src_arr is not None
                and getattr(src_arr, "ndim", 0) == 2
                and src_arr.shape[0] > 0
                and src_arr.shape[1] > 0
            ):
                c0 = max(
                    0,
                    min(
                        src_arr.shape[1] - 1, int(math.floor(cx / max(1e-6, float(width)) * src_arr.shape[1]))
                    ),
                )
                c1 = max(
                    c0 + 1,
                    min(
                        src_arr.shape[1],
                        int(math.ceil((cx + box_w) / max(1e-6, float(width)) * src_arr.shape[1])),
                    ),
                )
                r0 = max(
                    0,
                    min(
                        src_arr.shape[0] - 1,
                        int(math.floor(cy / max(1e-6, float(safe_h)) * src_arr.shape[0])),
                    ),
                )
                r1 = max(
                    r0 + 1,
                    min(
                        src_arr.shape[0],
                        int(math.ceil((cy + box_h) / max(1e-6, float(safe_h)) * src_arr.shape[0])),
                    ),
                )
                patch = src_arr[r0:r1, c0:c1]
                if has_seg_mask:
                    anat_frac = float(np.mean(patch > 0))
                elif dark_film:
                    anat_frac = float(np.mean(patch < 160))
                else:
                    anat_frac = float(np.mean(patch > 95))
            return (
                1 if ov_area > 0.0 else 0,
                pt_hits,
                1 if anat_frac > 0.03 else 0,
                ov_area + anat_frac * 100.0,
            )

        bx, by = corner_candidates[0]
        best_score = _corner_score(bx, by)
        if best_score[:3] != (0, 0, 0):
            for cand_x, cand_y in corner_candidates[1:]:
                sc = _corner_score(cand_x, cand_y)
                if sc < best_score:
                    best_score = sc
                    bx, by = cand_x, cand_y
                if sc[:3] == (0, 0, 0):
                    break

        draw.rectangle(
            [(bx, by), (bx + box_w, by + box_h)],
            fill=(2, 6, 23, 230),
            outline=(70, 84, 104, 255),
            width=1,
        )
        for i_d, d in enumerate(deep_depths):
            _pl, _pr, w_val, _prov = width_endpoints[d]
            colour = COLOR_WIDTH_W3 if float(d) <= 3.5 else COLOR_WIDTH_DEEP
            y = by + 4 + i_d * row_h
            draw.line(
                [(bx + 5, y + row_h * 0.5), (bx + 5 + max(8, 3 * scale), y + row_h * 0.5)],
                fill=colour,
                width=max(2, scale - 1),
            )
            txt = f"W{float(d):g} {w_val:.1f}" if w_val is not None else f"W{float(d):g} --"
            draw.text((bx + 10 + max(8, 3 * scale), y), txt, fill=colour, font=font_sm)
        occupied_rects.append((bx, by, bx + box_w, by + box_h))
        labels.append((bx, by, "W3/W5 key"))

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
            _place_caption(
                (cx, cy),
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
        _place_caption(
            (mid_sx, mid_sy),
            "SINUS",
            fg=(249, 168, 212),
            border=(236, 72, 153),
            font=font_sm,
        )

    # 6. Clinician/agent custom overlays (user-added lines and circles with text from Pre-Print Editor)
    if custom_overlays:
        u_span = max(1e-6, abs(float(u_coords[-1]) - float(u_coords[0])))
        px_per_mm = float(width) / u_span
        for ov in custom_overlays:
            if not isinstance(ov, dict):
                continue
            kind = str(ov.get("kind") or "").lower()
            ov_text = str(ov.get("text") or "").strip()
            if kind == "line" and ov.get("p1_uv") is not None and ov.get("p2_uv") is not None:
                p1_uv, p2_uv = ov["p1_uv"], ov["p2_uv"]
                lp1 = _px(float(p1_uv[0]), float(p1_uv[1]), u_coords, v_coords, scale)
                lp2 = _px(float(p2_uv[0]), float(p2_uv[1]), u_coords, v_coords, scale)
                draw.line([lp1, lp2], fill=(250, 204, 21, 255), width=max(2, scale))
                overlay.append((lp1, lp2))
                if ov_text:
                    _place_non_overlapping_pill(
                        0.5 * (lp1[0] + lp2[0]) + 4.0,
                        0.5 * (lp1[1] + lp2[1]) - 8.0,
                        ov_text,
                        fg=(250, 204, 21),
                        border=(234, 179, 8),
                        font=font_sm,
                    )
            elif kind == "circle" and ov.get("center_uv") is not None:
                cu, cv = ov["center_uv"]
                cx, cy = _px(float(cu), float(cv), u_coords, v_coords, scale)
                r_px = max(4.0, float(ov.get("radius_mm") or 2.0) * px_per_mm)
                draw.ellipse(
                    [cx - r_px, cy - r_px, cx + r_px, cy + r_px],
                    outline=(244, 114, 182, 255),
                    width=max(2, scale),
                )
                if ov_text:
                    _place_non_overlapping_pill(
                        cx + r_px + 3.0,
                        cy - 6.0,
                        ov_text,
                        fg=(244, 114, 182),
                        border=(236, 72, 153),
                        font=font_sm,
                    )

    bone_density = geom["misch_code"] if show_implant_guides else None

    # Draw footer text without re-filling over any basal end-cap that reached the bottom band.
    if show_footer:
        if meas is None and (height_mm_val is None or height_mm_val <= 0.0) and not width_endpoints:
            f_text = "not measurable"
            f_fill = COLOR_UNAVAILABLE
        else:
            if meas is not None:
                f_text = measurement_footer_text(meas, bone_density=bone_density)
            else:
                h_txt = f"{height_mm_val:.1f} mm" if height_mm_val is not None else "--"
                w0_val = width_endpoints[0.0][2] if 0.0 in width_endpoints else None
                w_txt = f"{w0_val:.1f} mm" if w0_val is not None else "--"
                f_text = f"H {h_txt}  W0 {w_txt}" + (f"  {bone_density}" if bone_density else "")
            f_fill = (248, 250, 252)

        f_tw, _ = _text_size(draw, f_text, font_md)
        f_x = 6.0
        if overlay:
            # Check whether the height line's endpoints sit inside the bottom-left footer text area.
            h_p0, h_p1 = overlay[0]
            for end_pt in (h_p0, h_p1):
                if end_pt[1] >= height - footer_h - 6 and end_pt[0] <= f_x + f_tw + 10:
                    alt_x = max(6.0, float(width - f_tw - 6))
                    if abs(end_pt[0] - alt_x) > abs(end_pt[0] - f_x):
                        f_x = alt_x
                    break
        f_y = float(img_h + 4)
        draw.text((f_x, f_y), f_text, fill=f_fill, font=font_md)
        labels.append((f_x, f_y, f_text))

    image = Image.alpha_composite(base_image, layer).convert("RGB")
    if monochrome:
        image = _to_monochrome(image)

    return CrossSectionRender(
        image=image,
        overlay_px=overlay,
        labels=labels,
        mm_per_px=MM_PER_COL / float(scale),
        available=True,
    )


def _to_monochrome(image: Any) -> Any:
    """Convert an RGB or RGBA Pillow image to grayscale while preserving any alpha channel."""
    if image.mode == "RGBA":
        alpha_ch = image.split()[3]
        gray_ch = image.convert("L")
        return Image.merge("RGBA", (gray_ch, gray_ch, gray_ch, alpha_ch))
    gray_ch = image.convert("L")
    return Image.merge("RGB", (gray_ch, gray_ch, gray_ch))


def _canal_mask_from_labels(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask selecting mandibular/incisive canals across 5-class & ToothFairy3."""
    arr = np.asarray(mask)
    if infer_mask_is_5class(arr):
        return arr == DENTALSEGMENTATOR_CANAL_ID
    return np.isin(arr, CANAL_IDS)


def _compute_foreground_mask(arr: np.ndarray, extra_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Identify anatomical foreground vs empty air/padding, filling interior radiolucent holes."""
    vals = np.asarray(arr, dtype=np.float32)
    lo_all, hi_all = np.percentile(vals, [2.0, 99.0])
    if hi_all <= lo_all + 1e-3:
        fg = np.ones(vals.shape, dtype=bool) if hi_all > -500.0 else np.zeros(vals.shape, dtype=bool)
    else:
        if lo_all < -700.0:
            air_cut = max(-650.0, float(lo_all) + 0.06 * float(hi_all - lo_all))
        else:
            air_cut = float(lo_all) + 0.08 * float(hi_all - lo_all)
        fg = vals > air_cut
    if extra_mask is not None and extra_mask.shape == fg.shape:
        fg |= np.asarray(extra_mask, dtype=bool)
    if fg.any():
        try:
            from scipy import ndimage  # type: ignore

            fg = ndimage.binary_fill_holes(fg)
        except ImportError:
            pass
    return fg


def _crop_vertical_air(
    arr: np.ndarray, fg: np.ndarray, *, pad_rows: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim empty top/bottom padding rows while keeping a clean anatomical margin."""
    n_rows, n_cols = arr.shape[:2]
    min_row_hits = max(2, int(round(0.02 * n_cols)))
    active_rows = np.flatnonzero(np.count_nonzero(fg, axis=1) >= min_row_hits)
    if active_rows.size == 0:
        active_rows = np.flatnonzero(fg.any(axis=1))
    if active_rows.size == 0:
        return arr, fg
    r0 = max(0, int(active_rows[0]) - int(pad_rows))
    r1 = min(n_rows, int(active_rows[-1]) + int(pad_rows) + 1)
    if r1 - r0 < 2:
        return arr, fg
    return arr[r0:r1, :], fg[r0:r1, :]


def _robust_window_level(arr: np.ndarray, fg: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute display `(window, level)` focused on non-air signal so padding cannot wash out bone."""
    vals = np.asarray(arr, dtype=np.float32)
    sample = vals[fg] if (fg is not None and int(np.count_nonzero(fg)) >= 16) else vals
    lo, hi = np.percentile(sample, [2.0, 99.5])
    if hi <= lo + 1e-3:
        lo, hi = np.percentile(vals, [1.0, 99.0])
    level = float(0.5 * (lo + hi))
    window = float(max(1.0, hi - lo))
    return window, level


def _foreground_alpha_u8(gray_u8: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Build an 8-bit alpha channel (`0` outside anatomical foreground, `255` inside)."""
    active = np.asarray(fg, dtype=bool) | (np.asarray(gray_u8) > 18)
    return np.where(active, 255, 0).astype(np.uint8)


def _resize_rgba_clean(image: Any, size: Tuple[int, int], rgb_resample: Any) -> Any:
    """Resize RGB smoothly while resizing any alpha channel with bilinear weights (no ringing)."""
    if image.mode == "RGBA":
        resampling = getattr(Image, "Resampling", Image)
        bilinear = getattr(resampling, "BILINEAR", Image.BILINEAR)
        r, g, b, a = image.split()
        rr, gg, bb = Image.merge("RGB", (r, g, b)).resize(size, rgb_resample).split()
        aa = a.resize(size, bilinear)
        return Image.merge("RGBA", (rr, gg, bb, aa))
    return image.resize(size, rgb_resample)


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
    transparent_bg: bool = False,
    crop_empty_vertical: bool = False,
    row_spacing_mm: Optional[float] = None,
    preserve_aspect: bool = False,
    target_size: Optional[Tuple[int, int]] = None,
) -> CrossSectionRender:
    """Render an upright panoramic strip with full-height cross-section reference lines."""
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

    col_sp = max(1e-3, float(column_spacing_mm))
    row_sp = max(1e-3, float(row_spacing_mm if row_spacing_mm is not None else col_sp))
    fg = _compute_foreground_mask(arr) if (transparent_bg or crop_empty_vertical) else None
    if crop_empty_vertical and fg is not None:
        pad_rows = max(2, int(round(3.0 / row_sp)))
        arr, fg = _crop_vertical_air(arr, fg, pad_rows=pad_rows)

    if window is None or level is None:
        if transparent_bg or crop_empty_vertical:
            window, level = _robust_window_level(arr, fg=fg)
        else:
            lo, hi = np.percentile(arr.astype(np.float32), [1.0, 99.0])
            level = float((lo + hi) / 2.0)
            window = float(max(1.0, hi - lo))

    gray = _enhance_bone_clarity(hu_window_to_rgb(arr, window=window, level=level), strength=0.30)
    if transparent_bg:
        fg_eff = fg if fg is not None else _compute_foreground_mask(arr)
        alpha = _foreground_alpha_u8(gray, fg_eff)
        rgba = np.dstack([gray, gray, gray, alpha]).astype(np.uint8)
        base_strip = Image.fromarray(rgba, mode="RGBA")
    else:
        base_strip = Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB")

    resampling = getattr(Image, "Resampling", Image)
    bicubic = getattr(resampling, "BICUBIC", Image.BILINEAR)
    isotropic_mode = (target_size is not None) or bool(preserve_aspect)
    if target_size is not None:
        max_w = max(1, int(target_size[0]))
        max_h = max(1, int(target_size[1]))
        tick_bar_h = max(28, int(round(max_h * 0.09)))
        avail_h = max(1, max_h - tick_bar_h)
        phys_w = max(col_sp, float(arr.shape[1]) * col_sp)
        phys_h = max(row_sp, float(arr.shape[0]) * row_sp)
        px_per_mm = min(float(max_w) / phys_w, float(avail_h) / phys_h)
        img_w = max(1, int(round(phys_w * px_per_mm)))
        anatomy_h = max(1, int(round(phys_h * px_per_mm)))
        img_h = anatomy_h + tick_bar_h
        x_scale = float(img_w) / float(max(1, arr.shape[1]))
    elif preserve_aspect:
        px_per_mm = float(max(1, int(scale))) / col_sp
        phys_w = max(col_sp, float(arr.shape[1]) * col_sp)
        phys_h = max(row_sp, float(arr.shape[0]) * row_sp)
        img_w = max(1, int(round(phys_w * px_per_mm)))
        anatomy_h = max(1, int(round(phys_h * px_per_mm)))
        tick_bar_h = max(26, int(round(anatomy_h * 0.12)))
        img_h = anatomy_h + tick_bar_h
        x_scale = float(img_w) / float(max(1, arr.shape[1]))
    else:
        x_scale = float(max(1, int(scale)))
        img_w = max(1, arr.shape[1] * max(1, int(scale)))
        img_h = max(1, int(strip_height))
        tick_bar_h = max(26, int(round(img_h * 0.11)))
        anatomy_h = img_h

    if isotropic_mode:
        resized_strip = _resize_rgba_clean(base_strip, (img_w, anatomy_h), bicubic)
        bg_fill = (0, 0, 0, 0) if transparent_bg else (2, 4, 9)
        image = Image.new("RGBA" if transparent_bg else "RGB", (img_w, img_h), bg_fill)
        image.paste(resized_strip, (0, 0))
    else:
        image = _resize_rgba_clean(base_strip, (img_w, img_h), bicubic)

    if transparent_bg:
        overlay_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)
    else:
        overlay_layer = None
        draw = ImageDraw.Draw(image)

    font = _load_font(max(13, int(round(img_h * 0.058))), bold=True)
    ticks: List[Tuple[float, float, str]] = []
    overlay: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    if not transparent_bg:
        draw.rectangle([(0, img_h - tick_bar_h), (img_w, img_h)], fill=(2, 4, 9))
    _draw_pill_label(draw, 8, 8, "R", fg=COLOR_IMPLANT, bg=(2, 4, 9), border=COLOR_IMPLANT, font=font)
    l_tw, _ = _text_size(draw, "L", font)
    _draw_pill_label(
        draw,
        max(8, img_w - l_tw - 14),
        8,
        "L",
        fg=COLOR_IMPLANT,
        bg=(2, 4, 9),
        border=COLOR_IMPLANT,
        font=font,
    )
    total_arc_mm = float(arr.shape[1] - 1) * col_sp
    for mm_mark in range(0, int(total_arc_mm) + 1, 10):
        rx = (float(mm_mark) / col_sp) * x_scale
        if 0 <= rx <= img_w - 1:
            draw.line([(rx, img_h - tick_bar_h), (rx, img_h - tick_bar_h + 5)], fill=(100, 116, 139), width=1)
    draw.line([(0, img_h - tick_bar_h), (img_w, img_h - tick_bar_h)], fill=(30, 41, 59), width=1)

    eff_scale = min(4, max(1, int(round(x_scale)))) if isotropic_mode else max(1, int(scale))
    hi_set = set(highlight_indices or ())
    last_label_right = -1e9
    for index, arc_mm in enumerate(tick_arcs_mm):
        col = float(arc_mm) / col_sp
        if col < 0 or col > arr.shape[1] - 1:
            continue
        x = col * x_scale
        is_hi = index in hi_set
        line_col = COLOR_BONE_HEIGHT_TEXT if is_hi else COLOR_WIDTH_W0
        line_w = max(2, eff_scale) if is_hi else max(1, eff_scale - 1)

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
            lx = min(float(img_w - tw - 8), x + 3.0)
            ly = float(img_h - tick_bar_h + 3)
            if transparent_bg:
                rect = _draw_pill_label(
                    draw,
                    lx,
                    ly,
                    lbl_str,
                    fg=COLOR_BONE_HEIGHT_TEXT if is_hi else (248, 250, 252),
                    bg=(15, 23, 42),
                    font=font,
                    pad_x=3,
                    pad_y=1,
                    bounds=(img_w, img_h),
                )
                last_label_right = rect[2]
            else:
                draw.text(
                    (min(img_w - tw - 2, x + 3), img_h - tick_bar_h + 4),
                    lbl_str,
                    fill=COLOR_BONE_HEIGHT_TEXT if is_hi else (226, 232, 240),
                    font=font,
                )
                last_label_right = x + 3 + tw
        ticks.append((x, float(img_h - tick_bar_h + 4), lbl_str))

    if transparent_bg and overlay_layer is not None:
        image = Image.alpha_composite(image, overlay_layer)
    if monochrome:
        image = _to_monochrome(image)
    return CrossSectionRender(image=image, overlay_px=overlay, labels=ticks, available=True)


def render_3d_mip_card(
    volume_image: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    spacing_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    quadrant: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    transparent_bg: bool = True,
    monochrome: bool = False,
    show_title: bool = True,
) -> CrossSectionRender:
    """Render a 3D coronal MIP projection card preserving physical aspect and transparent air background."""
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    img_3d = np.asarray(volume_image, dtype=np.float32)
    if img_3d.ndim != 3 or img_3d.size == 0:
        return CrossSectionRender(None, available=False, reason="invalid 3D volume")

    mip = np.max(img_3d, axis=1)[::-1, :]
    teeth_proj: Optional[np.ndarray] = None
    canal_proj: Optional[np.ndarray] = None
    extra_fg: Optional[np.ndarray] = None
    if mask is not None and np.asarray(mask).shape == img_3d.shape:
        m_arr = np.asarray(mask)
        teeth_proj = np.any(tooth_mask_from_labels(m_arr), axis=1)[::-1, :]
        canal_proj = np.any(_canal_mask_from_labels(m_arr), axis=1)[::-1, :]
        jaw_proj = np.any(jaw_mask_from_labels(m_arr), axis=1)[::-1, :]
        extra_fg = teeth_proj | canal_proj | jaw_proj

    fg = _compute_foreground_mask(mip, extra_mask=extra_fg)
    if fg.any():
        rows, cols = np.where(fg)
        pad = max(2, int(round(0.03 * max(mip.shape))))
        r0 = max(0, int(rows.min()) - pad)
        r1 = min(mip.shape[0], int(rows.max()) + pad + 1)
        c0 = max(0, int(cols.min()) - pad)
        c1 = min(mip.shape[1], int(cols.max()) + pad + 1)
        mip = mip[r0:r1, c0:c1]
        fg = fg[r0:r1, c0:c1]
        if teeth_proj is not None:
            teeth_proj = teeth_proj[r0:r1, c0:c1]
        if canal_proj is not None:
            canal_proj = canal_proj[r0:r1, c0:c1]

    win, lvl = _robust_window_level(mip, fg=fg)
    gray = hu_window_to_rgb(mip, window=win, level=lvl)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    if teeth_proj is not None and np.any(teeth_proj):
        rgb[teeth_proj] = 0.35 * rgb[teeth_proj] + 0.65 * np.array([255.0, 220.0, 120.0], dtype=np.float32)
    if canal_proj is not None and np.any(canal_proj):
        rgb[canal_proj] = np.array([255.0, 60.0, 50.0], dtype=np.float32)
    rgb_u8 = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    if transparent_bg:
        alpha = _foreground_alpha_u8(gray, fg)
        if teeth_proj is not None:
            alpha[teeth_proj] = 255
        if canal_proj is not None:
            alpha[canal_proj] = 255
        pil_img = Image.fromarray(np.dstack([rgb_u8, alpha]), mode="RGBA")
    else:
        pil_img = Image.fromarray(rgb_u8, mode="RGB")

    sz = max(1e-3, float(spacing_zyx[0]))
    sx = max(1e-3, float(spacing_zyx[2]))
    phys_w = max(sx, float(mip.shape[1]) * sx)
    phys_h = max(sz, float(mip.shape[0]) * sz)
    if target_size is not None:
        tw, th = max(1, int(target_size[0])), max(1, int(target_size[1]))
        scale_f = min(float(tw) / phys_w, float(th) / phys_h)
    else:
        scale_f = 440.0 / max(phys_w, phys_h)
    out_w = max(1, int(round(phys_w * scale_f)))
    out_h = max(1, int(round(phys_h * scale_f)))

    resampling = getattr(Image, "Resampling", Image)
    pil_img = _resize_rgba_clean(pil_img, (out_w, out_h), getattr(resampling, "LANCZOS", Image.BILINEAR))

    if show_title and quadrant is not None:
        title = f"3D Vivid MIP (Q{quadrant})"
        font = _load_font(max(12, int(round(out_h * 0.055))), bold=True)
        if transparent_bg:
            title_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            t_draw = ImageDraw.Draw(title_layer)
            _draw_pill_label(
                t_draw, 6, 6, title, fg=COLOR_IMPLANT, bg=(8, 13, 24), font=font, bounds=(out_w, out_h)
            )
            pil_img = Image.alpha_composite(pil_img, title_layer)
        else:
            t_draw = ImageDraw.Draw(pil_img)
            _draw_pill_label(
                t_draw, 6, 6, title, fg=COLOR_IMPLANT, bg=(8, 13, 24), font=font, bounds=(out_w, out_h)
            )

    if monochrome:
        pil_img = _to_monochrome(pil_img)
    return CrossSectionRender(image=pil_img, available=True)


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
    "render_3d_mip_card",
    "render_cross_section",
    "render_panoramic_strip",
    "select_periodic_depth_key",
    "unavailable_tile",
]

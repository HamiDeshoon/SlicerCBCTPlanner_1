"""Clinical renderers for cross-sections and the panoramic strip.

Pure numpy geometry with Pillow (and automatic Qt fallback in 3D Slicer). Every overlay
is derived from measured millimetre coordinates:
  - Clean, mask-free CBCT anatomy by default (`show_masks=False`) so cortical plates,
    trabecular bone, roots, and nerve canals remain unobstructed.
  - Green vertical Height line (`H` in mm) from crestal entry point to root apex or
    basal/canal/sinus anatomical limit, drawn on EVERY cross-section slice.
  - Cyan horizontal Entry-Point / Crestal Width line (`W0` in mm) + yellow/cyan entry
    crosshair target drawn on EVERY cross-section slice.
  - Gold-orange `W3` (3 mm depth) and orange `W5` (5 mm depth) Buccolingual Width lines
    in large, bold, high-contrast millimetre typography.
  - Implant deployment visual aids: 2.0 mm safety-margin line above IAN canal / basal limit,
    suggested safe implant cylinder envelope & axis, and Misch D1-D4 bone-density cue.
  - Large, upright panoramic strip (`flip_vertical=True`, default height 340 px) with
    full-height cross-section cut lines and clear slice/tooth labels.
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
SAFETY_MARGIN_MM = 2.0

_STANDARD_IMPLANT_DIAMETERS = (3.3, 3.5, 4.0, 4.5, 5.0)
_STANDARD_IMPLANT_LENGTHS = (8.0, 10.0, 11.5, 13.0)

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
    lo, hi = (min(x0, x1), max(x0, x1))
    cur = lo
    while cur < hi:
        seg_end = min(hi, cur + dash)
        draw.line([(cur, y), (seg_end, y)], fill=fill, width=width)
        cur += dash + gap


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
    lo, hi = (min(y0, y1), max(y0, y1))
    cur = lo
    while cur < hi:
        seg_end = min(hi, cur + dash)
        draw.line([(x, cur), (x, seg_end)], fill=fill, width=width)
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


def _enhance_bone_clarity(gray_u8: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Fast pure-numpy 3x3 unsharp mask to sharpen cortical plates, roots, and canal walls on X-ray film."""
    arr = np.asarray(gray_u8, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3 or strength <= 0.0:
        return np.asarray(gray_u8, dtype=np.uint8)
    p = np.pad(arr, 1, mode="edge")
    blur = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) * (1.0 / 9.0)
    sharp = arr + float(strength) * (arr - blur)
    return np.clip(sharp, 0.0, 255.0).astype(np.uint8)


def _px(u: float, v: float, u_coords: np.ndarray, v_coords: np.ndarray, scale: int) -> Tuple[float, float]:
    col = (float(u) - float(u_coords[0])) / max(1e-6, float(u_coords[-1]) - float(u_coords[0]))
    row = (float(v_coords[-1]) - float(v)) / max(1e-6, float(v_coords[-1]) - float(v_coords[0]))
    width = (len(u_coords) - 1) * scale
    height = (len(v_coords) - 1) * scale
    return col * width, row * height


def _apply_anatomical_contour_tint(rgb: np.ndarray, mask_cs: Optional[np.ndarray]) -> np.ndarray:
    """Blend subtle vivid SlicerDentalSegmentator boundary contours when explicitly enabled."""
    if mask_cs is None or mask_cs.shape != rgb.shape[:2]:
        return rgb
    out = rgb.astype(np.float32)
    m = np.asarray(mask_cs)
    edge = np.zeros(m.shape, dtype=bool)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]

    tooth_edge = edge & ((m == 3) | (m == 4) | ((m >= 11) & (m <= 48)))
    if np.any(tooth_edge):
        out[tooth_edge] = 0.35 * out[tooth_edge] + 0.65 * np.array([255.0, 215.0, 90.0], dtype=np.float32)

    jaw_edge = edge & ((m == 1) | (m == 2)) & (~tooth_edge)
    if np.any(jaw_edge):
        out[jaw_edge] = 0.55 * out[jaw_edge] + 0.45 * np.array([80.0, 220.0, 180.0], dtype=np.float32)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _classify_misch_bone_density(
    img_raw: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    h_u: float,
    crest_v: float,
    basal_v: float,
) -> Optional[str]:
    """Estimate Misch D1-D4 bone density along the planned osteotomy corridor (HU-calibrated volumes)."""
    arr = np.asarray(img_raw, dtype=np.float32)
    if arr.size == 0 or float(np.min(arr)) > -150.0:
        return None
    v_lo, v_hi = sorted((float(crest_v), float(basal_v)))
    u_mask = np.abs(u_coords - float(h_u)) <= 1.5
    v_mask = (v_coords >= v_lo) & (v_coords <= v_hi)
    if not (np.any(u_mask) and np.any(v_mask)):
        return None
    patch = arr[np.ix_(v_mask, u_mask)]
    bone_vals = patch[patch > 100.0]
    if bone_vals.size < 8:
        return None
    mean_hu = float(np.mean(bone_vals))
    if mean_hu >= 1250.0:
        return "D1"
    if mean_hu >= 850.0:
        return "D2"
    if mean_hu >= 350.0:
        return "D3"
    return "D4"


def _suggest_implant_size(
    height_mm: Optional[float],
    w0_mm: Optional[float],
    w3_mm: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Suggest a conservative implant (diameter_mm, length_mm) respecting 2 mm safety clearances."""
    if height_mm is None or float(height_mm) < 8.5:
        return None
    avail_len = float(height_mm) - SAFETY_MARGIN_MM
    valid_lens = [L for L in _STANDARD_IMPLANT_LENGTHS if L <= avail_len]
    if not valid_lens:
        return None
    width_candidates = [w for w in (w0_mm, w3_mm) if w is not None and float(w) > 0.0]
    if not width_candidates:
        return None
    min_w = min(float(w) for w in width_candidates)
    avail_diam = min_w - 2.0  # >= 1 mm buccal + >= 1 mm lingual bone plate
    valid_diams = [D for D in _STANDARD_IMPLANT_DIAMETERS if D <= avail_diam]
    if not valid_diams:
        return None
    return max(valid_diams), max(valid_lens)


def _infer_cross_section_geometry(
    cs: Any,
    meas: Optional[Any],
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> Dict[str, Any]:
    """Resolve crest/entry point, vertical limit (root apex or basal/canal), and width segments.

    Guarantees that every cross-section tile has a valid vertical line and horizontal
    entry-point line even when upstream `meas` or `crest_v` is partially populated.
    """
    is_mand = bool(getattr(cs, "is_mandible", True))
    img_raw = np.asarray(cs.img_cs, dtype=np.float32)
    mask_raw = getattr(cs, "mask_cs", None)
    segments: Dict[float, Tuple[float, float]] = dict(getattr(cs, "width_segments", {}) or {})
    crest_v = getattr(cs, "crest_v", None)
    basal_v = getattr(cs, "basal_v", None)
    root_apex_v = getattr(cs, "root_apex_v", None)
    h_u = getattr(cs, "height_u", None)
    canal = getattr(cs, "canal_centroid_uv", None)

    # Identify bone/tooth foreground mask directly from HU or intensity if needed
    if img_raw.size > 0:
        if float(np.min(img_raw)) < -150.0:
            fg = img_raw >= 250.0
        else:
            p70 = float(np.percentile(img_raw, 70.0))
            fg = img_raw >= max(p70, float(np.mean(img_raw) + 0.5 * np.std(img_raw)))
    else:
        fg = np.zeros((len(v_coords), len(u_coords)), dtype=bool)

    # 1. Resolve horizontal axis center (`h_u`)
    if h_u is None:
        if 0.0 in segments:
            h_u = 0.5 * (float(segments[0.0][0]) + float(segments[0.0][1]))
        elif np.any(fg):
            u_idx = int(round(float(np.mean(np.where(fg)[1]))))
            u_idx = min(max(0, u_idx), len(u_coords) - 1)
            h_u = float(u_coords[u_idx])
        else:
            h_u = float(0.5 * (u_coords[0] + u_coords[-1]))
    else:
        h_u = float(h_u)

    # 2. Resolve crest / entry point (`crest_v`)
    if crest_v is None and np.any(fg):
        center_cols = np.abs(u_coords - h_u) <= 3.0
        col_fg = fg[:, center_cols] if np.any(center_cols) else fg
        v_hits = np.where(np.any(col_fg, axis=1))[0]
        if v_hits.size > 0:
            crest_idx = int(v_hits[-1]) if is_mand else int(v_hits[0])
            crest_v = float(v_coords[crest_idx])

    # 3. Resolve root apex from tooth mask if present and not already supplied
    has_tooth_root = False
    if root_apex_v is not None:
        root_apex_v = float(root_apex_v)
        has_tooth_root = True
    elif mask_raw is not None and np.asarray(mask_raw).shape == img_raw.shape:
        m = np.asarray(mask_raw)
        tooth_mask = (m == 3) | (m == 4) | ((m >= 11) & (m <= 48))
        center_cols = np.abs(u_coords - h_u) <= 3.5
        t_col = tooth_mask[:, center_cols] if np.any(center_cols) else tooth_mask
        t_rows = np.where(np.any(t_col, axis=1))[0]
        if t_rows.size >= 3:
            # Mandibular tooth root apex is inferior-most (lowest v); maxillary is superior-most
            apex_idx = int(t_rows[0]) if is_mand else int(t_rows[-1])
            root_apex_v = float(v_coords[apex_idx])
            has_tooth_root = True

    # 4. Resolve basal/anatomical limit (`basal_v`) and vertical height (`height_mm`)
    height_mm = getattr(meas, "bone_height_mm", None) if meas is not None else None
    if height_mm is not None:
        height_mm = float(height_mm)

    if crest_v is not None:
        crest_v = float(crest_v)
        if basal_v is None:
            if has_tooth_root and root_apex_v is not None and abs(crest_v - root_apex_v) >= 2.0:
                basal_v = float(root_apex_v)
            elif height_mm is not None and height_mm > 0.0:
                basal_v = crest_v - height_mm if is_mand else crest_v + height_mm
            elif canal is not None:
                basal_v = float(canal[1]) + (1.5 if is_mand else -1.5)
            elif np.any(fg):
                center_cols = np.abs(u_coords - h_u) <= 3.0
                col_fg = fg[:, center_cols] if np.any(center_cols) else fg
                v_hits = np.where(np.any(col_fg, axis=1))[0]
                if v_hits.size >= 2:
                    basal_idx = int(v_hits[0]) if is_mand else int(v_hits[-1])
                    basal_v = float(v_coords[basal_idx])
        else:
            basal_v = float(basal_v)

        if (height_mm is None or height_mm <= 0.0) and basal_v is not None:
            height_mm = abs(crest_v - float(basal_v))

        # 5. Guarantee horizontal entry-point width segment (`0.0` mm depth) on every slice
        if 0.0 not in segments and np.any(fg):
            v_idx = int(np.argmin(np.abs(v_coords - crest_v)))
            # Sample a 3-row band at the crestal entry point for robust width detection
            r0 = max(0, v_idx - 1)
            r1 = min(fg.shape[0], v_idx + 2)
            row_fg = np.any(fg[r0:r1, :], axis=0)
            u_hits = np.where(row_fg)[0]
            if u_hits.size >= 2:
                segments[0.0] = (float(u_coords[int(u_hits[0])]), float(u_coords[int(u_hits[-1])]))

    return {
        "is_mand": is_mand,
        "crest_v": crest_v,
        "basal_v": basal_v,
        "root_apex_v": root_apex_v,
        "has_tooth_root": has_tooth_root,
        "h_u": h_u,
        "height_mm": height_mm,
        "segments": segments,
        "canal": canal,
    }


def measurement_footer_text(meas: Optional[Any], bone_density: Optional[str] = None) -> str:
    """Return the tile footer `H ... / W0 ...` caption, printing `--` for a withheld metric.

    A withheld crest width must never render as a fabricated `W0 0.0 mm`.
    """
    if meas is None:
        base = "H --  W0 --"
    else:
        height_mm = getattr(meas, "bone_height_mm", None)
        crest_w = getattr(meas, "crest_width_mm", None)
        h_txt = f"{float(height_mm):.1f} mm" if height_mm is not None else "--"
        w_txt = f"{float(crest_w):.1f} mm" if crest_w is not None else "--"
        base = f"H {h_txt}  W0 {w_txt}"
    return f"{base}  {bone_density}" if bone_density else base


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
) -> CrossSectionRender:
    """Render one arch cross-section with clean anatomy and large millimetre overlays.

    `cs` is a `CrossSectionData` (duck-typed to keep this module Slicer-free).
    Segmentation masks are OFF by default (`show_masks=False`) so CBCT bone stays clear.
    Every tile renders both the vertical measurement line (to root apex or anatomical limit)
    and the horizontal entry-point line in large high-contrast millimetre fonts.
    """
    if not _PIL_AVAILABLE:
        return CrossSectionRender(None, available=False, reason="Pillow is not installed")
    if cs is None:
        return CrossSectionRender(None, available=False, reason="no cross-section data")

    u_coords = np.asarray(cs.u_coords, dtype=np.float64)
    v_coords = np.asarray(cs.v_coords, dtype=np.float64)

    img_raw = np.asarray(cs.img_cs)
    geom = _infer_cross_section_geometry(cs, meas, u_coords, v_coords)

    # `cs.img_cs[iv, iu]` is indexed by `v_coords[iv]` (ascending inferior -> superior),
    # whereas `_px()` maps `v_coords[-1]` (superior) to image row 0 (top). Flip vertically
    # so the CBCT anatomy and every measurement line share one exact frame.
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
    draw = ImageDraw.Draw(image)

    # Large, bold clinical typography for effortless reading in window & print views
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

    # Anatomical orientation badges: Lingual (L) left, Buccal (B) right, Superior (S) top-center.
    l_rect = _draw_pill_label(draw, 3, 3, "L", fg=(74, 222, 128), border=(34, 197, 94), font=font_md, bounds=bounds)
    b_tw, _ = _text_size(draw, "B", font_md)
    b_rect = _draw_pill_label(
        draw, max(3, width - b_tw - 11), 3, "B", fg=(74, 222, 128), border=(34, 197, 94), font=font_md, bounds=bounds
    )
    s_tw, _ = _text_size(draw, "S", font_sm)
    s_rect = _draw_pill_label(
        draw, max(3, (width - s_tw - 8) // 2), 3, "S", fg=(148, 163, 184), font=font_sm, bounds=bounds
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

    crest_v = geom["crest_v"]
    basal_v = geom["basal_v"]
    h_u = float(geom["h_u"])
    height_mm_val = geom["height_mm"]
    segments: Dict[float, Tuple[float, float]] = geom["segments"]
    is_mand = bool(geom["is_mand"])
    canal = geom["canal"]

    # Suggested implant envelope & axis guide (drawn underneath primary calipers for clarity)
    w0_mm = abs(segments[0.0][1] - segments[0.0][0]) if 0.0 in segments else None
    w3_mm = abs(segments[3.0][1] - segments[3.0][0]) if 3.0 in segments else None
    suggested_impl = _suggest_implant_size(height_mm_val, w0_mm, w3_mm) if show_implant_guides else None
    if show_implant_guides and suggested_impl is not None and crest_v is not None:
        imp_d, imp_l = suggested_impl
        half_d = 0.5 * imp_d
        imp_end_v = float(crest_v) - imp_l if is_mand else float(crest_v) + imp_l
        ix0, iy0 = _px(h_u - half_d, crest_v, u_coords, v_coords, scale)
        ix1, iy1 = _px(h_u + half_d, imp_end_v, u_coords, v_coords, scale)
        x_lo, x_hi = min(ix0, ix1), max(ix0, ix1)
        y_lo, y_hi = min(iy0, iy1), max(iy0, iy1)
        _draw_dashed_vline(draw, x_lo, y_lo, y_hi, fill=(56, 189, 248), dash=5, gap=4, width=max(1, scale - 1))
        _draw_dashed_vline(draw, x_hi, y_lo, y_hi, fill=(56, 189, 248), dash=5, gap=4, width=max(1, scale - 1))
        _draw_dashed_hline(draw, x_lo, x_hi, y_hi if is_mand else y_lo, fill=(56, 189, 248), dash=4, gap=3, width=max(1, scale - 1))
        _place_non_overlapping_pill(
            3.0,
            26.0,
            f"Impl {imp_d:.1f}x{imp_l:g} mm",
            fg=(125, 211, 252),
            border=(56, 189, 248),
            font=font_sm,
        )

    # 1. Horizontal Entry-Point / Crest Line (drawn on EVERY single image)
    if crest_v is not None:
        x_left, y_crest = _px(u_coords[0], crest_v, u_coords, v_coords, scale)
        x_right, _ = _px(u_coords[-1], crest_v, u_coords, v_coords, scale)
        if 0.0 in segments:
            w0_l, w0_r = segments[0.0]
            wx0, _ = _px(w0_l, crest_v, u_coords, v_coords, scale)
            wx1, _ = _px(w0_r, crest_v, u_coords, v_coords, scale)
            _draw_dashed_hline(draw, x_left + 12, max(x_left + 12, wx0 - 4), y_crest, fill=(250, 204, 21), width=max(1, scale - 1))
            _draw_dashed_hline(draw, min(x_right - 4, wx1 + 4), x_right - 4, y_crest, fill=(250, 204, 21), width=max(1, scale - 1))
        else:
            _draw_dashed_hline(draw, x_left + 12, x_right - 4, y_crest, fill=(250, 204, 21), width=max(2, scale - 1))
        overlay.append(((x_left, y_crest), (x_right, y_crest)))

        # Entry-point target crosshair at (h_u, crest_v)
        ep_x, ep_y = _px(h_u, crest_v, u_coords, v_coords, scale)
        ep_r = max(4, int(round(2.0 * scale)))
        draw.ellipse(
            [ep_x - ep_r, ep_y - ep_r, ep_x + ep_r, ep_y + ep_r],
            outline=(250, 204, 21),
            width=max(2, scale - 1),
        )
        draw.line([(ep_x - ep_r - 3, ep_y), (ep_x + ep_r + 3, ep_y)], fill=(250, 204, 21), width=max(1, scale - 1))
        draw.line([(ep_x, ep_y - ep_r - 3), (ep_x, ep_y + ep_r + 3)], fill=(250, 204, 21), width=max(1, scale - 1))

    # 2. Vertical Measurement Line from Entry Point to Root Apex / Anatomical Limit (EVERY single image)
    if crest_v is not None and basal_v is not None and height_mm_val is not None and float(height_mm_val) > 0.0:
        hx0, hy0 = _px(h_u, crest_v, u_coords, v_coords, scale)
        hx1, hy1 = _px(h_u, basal_v, u_coords, v_coords, scale)
        line_w = max(2, scale)
        cap_r = max(5, int(round(2.4 * scale)))
        draw.line([(hx0, hy0), (hx1, hy1)], fill=(34, 197, 94), width=line_w)
        draw.line([(hx0 - cap_r, hy0), (hx0 + cap_r, hy0)], fill=(34, 197, 94), width=max(2, line_w - 1))
        draw.line([(hx1 - cap_r, hy1), (hx1 + cap_r, hy1)], fill=(34, 197, 94), width=max(2, line_w - 1))
        overlay.append(((hx0, hy0), (hx1, hy1)))
        mid_hy = 0.5 * (hy0 + hy1)
        h_prefix = "Root" if geom["has_tooth_root"] else "H"
        _place_non_overlapping_pill(
            hx0 + cap_r + 5,
            mid_hy - 8,
            f"{h_prefix} {float(height_mm_val):.1f} mm",
            fg=(74, 222, 128),
            border=(34, 197, 94),
            font=font_lg,
        )

        # Prosthetic / Implant Long-Axis trajectory extending 4 mm coronally above crest
        if show_implant_guides:
            ext_v = float(crest_v) + 4.0 if is_mand else float(crest_v) - 4.0
            ax_x0, ax_y0 = _px(h_u, ext_v, u_coords, v_coords, scale)
            ax_x1, ax_y1 = _px(h_u, crest_v, u_coords, v_coords, scale)
            _draw_dashed_vline(draw, ax_x0, ax_y0, ax_y1, fill=(226, 232, 240), dash=3, gap=3, width=1)
            ridge_ang = getattr(cs, "ridge_axis_angle_deg", None)
            if ridge_ang is not None and abs(float(ridge_ang)) >= 2.0:
                _place_non_overlapping_pill(
                    width - 96.0,
                    26.0,
                    f"Ang {float(ridge_ang):+.0f}deg",
                    fg=(226, 232, 240),
                    border=(100, 116, 139),
                    font=font_sm,
                )

        # 2.0 mm Apical Safety Margin barrier coronal to basal limit / IAN canal / sinus floor
        if show_implant_guides and float(height_mm_val) > SAFETY_MARGIN_MM + 1.0:
            safe_v = float(basal_v) + SAFETY_MARGIN_MM if is_mand else float(basal_v) - SAFETY_MARGIN_MM
            sx0, sy = _px(h_u - 5.0, safe_v, u_coords, v_coords, scale)
            sx1, _ = _px(h_u + 5.0, safe_v, u_coords, v_coords, scale)
            _draw_dashed_hline(draw, sx0, sx1, sy, fill=(248, 113, 113), dash=5, gap=3, width=max(2, scale - 1))
            _place_non_overlapping_pill(
                sx0 - 52.0,
                sy - 8,
                "2mm SAFE",
                fg=(252, 165, 165),
                border=(239, 68, 68),
                font=font_sm,
            )

    # 3. Horizontal Buccolingual Width calipers (W0 Entry cyan, W3 gold-orange, W5 orange)
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
        tick_r = max(4, int(round(1.8 * scale))) if is_primary else 2
        draw.line([start, end], fill=colour, width=w_line)
        draw.line([(start[0], start[1] - tick_r), (start[0], start[1] + tick_r)], fill=colour, width=max(2, w_line - 1))
        draw.line([(end[0], end[1] - tick_r), (end[0], end[1] + tick_r)], fill=colour, width=max(2, w_line - 1))
        overlay.append((start, end))

        if is_primary:
            w_val = abs(float(u_right) - float(u_left))
            if depth == 0.0:
                lbl_str = f"W0 {w_val:.1f} mm"
            else:
                lbl_str = f"W{depth:g} {w_val:.1f} mm"
            if depth == 3.0:
                pref_x = start[0] - _text_size(draw, lbl_str, font_md)[0] - 12
            else:
                pref_x = end[0] + 6
            _place_non_overlapping_pill(
                pref_x,
                start[1] - 8,
                lbl_str,
                fg=colour,
                border=colour,
                font=font_md,
            )

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

    sinus_pts = getattr(cs, "sinus_floor_uv", None)
    if sinus_pts and len(sinus_pts) >= 2:
        px_pts = [_px(float(su), float(sv), u_coords, v_coords, scale) for su, sv in sinus_pts]
        for i_s in range(len(px_pts) - 1):
            draw.line([px_pts[i_s], px_pts[i_s + 1]], fill=(244, 114, 182), width=max(2, scale))
        mid_sx, mid_sy = px_pts[len(px_pts) // 2]
        _place_non_overlapping_pill(
            mid_sx + 6,
            mid_sy - 14,
            "SINUS",
            fg=(249, 168, 212),
            border=(236, 72, 153),
            font=font_sm,
        )

    bone_density = (
        _classify_misch_bone_density(img_raw, u_coords, v_coords, h_u, crest_v, basal_v)
        if (show_implant_guides and crest_v is not None and basal_v is not None)
        else None
    )

    # Crisp dark clinical footer bar at the bottom of the tile
    draw.rectangle([(0, height - footer_h), (width, height)], fill=(2, 4, 9))
    draw.line([(0, height - footer_h), (width, height - footer_h)], fill=(30, 41, 59), width=1)
    if meas is None and (height_mm_val is None or float(height_mm_val) <= 0.0):
        draw.text((6, height - footer_h + 4), "not measurable", fill=(248, 113, 113), font=font_md)
        labels.append((6.0, float(height - footer_h + 4), "not measurable"))
    else:
        if meas is not None:
            text = measurement_footer_text(meas, bone_density=bone_density)
        else:
            h_txt = f"{float(height_mm_val):.1f} mm" if height_mm_val is not None else "--"
            w_txt = f"{float(w0_mm):.1f} mm" if w0_mm is not None else "--"
            text = f"H {h_txt}  W0 {w_txt}" + (f"  {bone_density}" if bone_density else "")
        draw.text((6, height - footer_h + 4), text, fill=(248, 250, 252), font=font_md)
        labels.append((6.0, float(height - footer_h + 4), text))

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
) -> CrossSectionRender:
    """Render a large, upright panoramic strip with full-height cross-section reference lines.

    Column `k` maps to arc length `k * column_spacing_mm`. `flip_vertical=True` (default)
    places superior at the top so the print panoramic matches the upright clinical view.
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

    # Panoramic array row 0 is inferior (z-ascending); flip vertically so superior is top.
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
    # Radiographic patient orientation markers (R = Patient Right, L = Patient Left)
    _draw_pill_label(draw, 8, 8, "R", fg=(56, 189, 248), bg=(2, 4, 9), border=(56, 189, 248), font=font)
    _draw_pill_label(draw, max(8, img_w - 30), 8, "L", fg=(56, 189, 248), bg=(2, 4, 9), border=(56, 189, 248), font=font)
    # 10 mm calibrated arch ruler ticks along the panoramic bottom border
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
        line_col = (74, 222, 128) if is_hi else (0, 220, 255)
        line_w = max(2, int(scale)) if is_hi else max(1, int(scale) - 1)

        # Draw full-height cross-section cut line across the panoramic image
        if show_cross_section_lines:
            if is_hi:
                draw.line([(x, 0), (x, img_h - tick_bar_h)], fill=line_col, width=line_w)
            else:
                _draw_dashed_vline(draw, x, 0, img_h - tick_bar_h, fill=line_col, dash=8, gap=5, width=line_w)
            overlay.append(((x, 0.0), (x, float(img_h - tick_bar_h))))
        else:
            draw.line([(x, img_h - tick_bar_h - 10), (x, img_h - tick_bar_h)], fill=(34, 197, 94), width=line_w)
            draw.line([(x, 0), (x, max(12, tick_bar_h // 2))], fill=(0, 220, 255), width=line_w)

        # Top & bottom anchor ticks
        draw.line([(x, 0), (x, 10)], fill=line_col, width=max(2, line_w))
        draw.line([(x, img_h - tick_bar_h - 10), (x, img_h - tick_bar_h)], fill=line_col, width=max(2, line_w))

        label = labels[index] if labels is not None and index < len(labels) else f"#{index + 1}"
        lbl_str = str(label)
        tw, _ = _text_size(draw, lbl_str, font)
        if lbl_str and x >= last_label_right + 6:
            draw.text(
                (min(img_w - tw - 2, x + 3), img_h - tick_bar_h + 4),
                lbl_str,
                fill=(74, 222, 128) if is_hi else (226, 232, 240),
                font=font,
            )
            last_label_right = x + 3 + tw
        ticks.append((x, float(img_h - tick_bar_h + 4), lbl_str))

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

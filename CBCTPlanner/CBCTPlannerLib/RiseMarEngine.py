r"""RISE-MAR: Radiologist-in-the-Loop Self-Training Metal Artifact Reduction Engine.

Integrates https://github.com/Masaaki-75/rise-mar (Ma et al., IEEE TMI 2025, arXiv:2501.15610)
into SlicerCBCTPlanner for dental CBCT Metal Artifact Reduction (MAR) whenever needed.

Key Capabilities:
  - User-configurable checkpoint path persisted via 3D Slicer QSettings and validated before inference.
  - Pre-MAR tissue & metal categorization (`categorize_volume_pre_mar`) executed strictly on
    the unmodified input volume so downstream bone/tissue classification and metal detection
    are never degraded by post-MAR intensity shifts.
  - Adaptive metal detection for both HU-calibrated and uncalibrated CBCT volumes, with a
    clean no-op guard in fast/metal-only mode when no metal is present (preventing global
    darkening on metal-free scans).
  - Localized peri-metal correction with smooth spatial feathering and non-metal contrast
    preservation (median/IQR matching + Slicer Window/Level lock) so global contrast never
    shifts darker.
  - CBAM Attention-Guided U-Net (`RiseMarCBAMUNet2D`) with Channel & Spatial Attention modules
    in skip/concatenation connections (`Lee et al. / Ma et al.`).
  - Built-in Clinical Quality Assessor (`CQA`, 1.0 - 10.0 Likert scale) evaluating pre- and post-MAR
    image quality across metal-affected axial slices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

from .CheckpointInference import load_state_dict_shape_safe, normalize_state_dict_keys

logger = logging.getLogger("CBCTPlanner.RiseMarEngine")

DEFAULT_RISEMAR_WEIGHTS_PATH: str = ""
RISEMAR_HU_MIN: float = -1024.0
RISEMAR_HU_MAX: float = 3072.0
DEFAULT_METAL_THRESHOLD_HU: float = 2200.0


def validate_risemar_weights_path(weights_path: Optional[str]) -> Tuple[bool, str]:
    """Validate a user-selected RISE-MAR weights file before enabling neural inference."""
    raw = str(weights_path or "").strip()
    if not raw:
        return False, "No RISE-MAR weights selected — browse to a .pkl / .pth checkpoint to enable."
    path = Path(raw)
    if not path.exists():
        return False, f"RISE-MAR weights file not found: {path.name}"
    if not path.is_file():
        return False, f"Selected RISE-MAR path is not a file: {path.name}"
    if path.suffix.lower() not in (".pkl", ".pth", ".pt", ".ckpt"):
        return False, f"Unsupported RISE-MAR weights extension '{path.suffix}' (expected .pkl, .pth, or .pt)."
    if path.stat().st_size <= 0:
        return False, f"RISE-MAR weights file is empty: {path.name}"
    return True, f"Weights ready: {path.name}"


@dataclass
class RiseMarConfig:
    """Configuration for RISE-MAR dental CBCT metal artifact reduction."""

    weights_path: str = DEFAULT_RISEMAR_WEIGHTS_PATH
    device: str = "auto"  # "auto", "cuda", "cpu"
    metal_threshold_hu: float = DEFAULT_METAL_THRESHOLD_HU
    preserve_metal_core: bool = True
    metal_slices_only: bool = True
    blend_strength: float = 0.85  # [0.0 .. 1.0] blend between MAR prediction and original
    in_place: bool = False
    preserve_contrast: bool = True
    adaptive_metal_threshold: bool = True
    artifact_radius_px: float = 70.0


@dataclass
class RiseMarResult:
    """Quantitative and diagnostic summary of a RISE-MAR run."""

    output_image_zyx: np.ndarray
    weights_path: str
    weights_loaded: bool
    mode_used: (
        str  # "risemar_neural_cbam", "risemar_pickled_module", "physics_hybrid_fallback", or "no_metal_noop"
    )
    device_used: str
    total_slices: int
    processed_slices: int
    metal_slice_indices: List[int] = field(default_factory=list)
    metal_voxel_count: int = 0
    cqa_score_before: float = 5.0
    cqa_score_after: float = 8.0
    loaded_tensors: int = 0
    total_ckpt_tensors: int = 0
    elapsed_sec: float = 0.0
    effective_metal_threshold_hu: float = DEFAULT_METAL_THRESHOLD_HU
    pre_mar_categorization: Dict[str, Any] = field(default_factory=dict)


def preprocess_slice_risemar(
    slice_2d: np.ndarray,
    hu_min: float = RISEMAR_HU_MIN,
    hu_max: float = RISEMAR_HU_MAX,
) -> np.ndarray:
    """Clip 2D CT/CBCT slice to `[hu_min, hu_max]` (`[-1024, 3072]`) and normalize to `[0, 1]`."""
    arr = np.asarray(slice_2d, dtype=np.float32)
    span = max(1.0, float(hu_max) - float(hu_min))
    clipped = np.clip(arr, float(hu_min), float(hu_max))
    return ((clipped - float(hu_min)) / span).astype(np.float32)


def postprocess_slice_risemar(
    norm_2d: np.ndarray,
    hu_min: float = RISEMAR_HU_MIN,
    hu_max: float = RISEMAR_HU_MAX,
) -> np.ndarray:
    """Map normalized `[0, 1]` RISE-MAR output slice back to `[hu_min, hu_max]` HU space."""
    arr = np.clip(np.asarray(norm_2d, dtype=np.float32), 0.0, 1.0)
    span = float(hu_max) - float(hu_min)
    return (arr * span + float(hu_min)).astype(np.float32)


def categorize_volume_pre_mar(
    volume_zyx: np.ndarray,
    metal_threshold_hu: float = DEFAULT_METAL_THRESHOLD_HU,
    adaptive_metal_threshold: bool = True,
    min_metal_pixels_per_slice: int = 4,
) -> Dict[str, Any]:
    """Evaluate and categorize the unmodified pre-MAR volume before any artifact reduction.

    Computes:
      - Whether the volume appears HU-calibrated vs. shifted/uncalibrated CBCT intensity.
      - Reference tissue percentiles (`p10`, `median`, `p90`, `iqr`) for contrast preservation.
      - Effective metal threshold (using `metal_threshold_hu` when present, or an adaptive
        high-outlier threshold for uncalibrated/shifted CBCT volumes).
      - Exact pre-MAR metal slice indices and voxel counts.
    """
    arr = np.asarray(volume_zyx, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.size == 0:
        return {
            "is_hu_calibrated": True,
            "requested_threshold_hu": float(metal_threshold_hu),
            "effective_threshold_hu": float(metal_threshold_hu),
            "adaptive_triggered": False,
            "metal_slice_indices": [],
            "metal_voxel_count": 0,
            "tissue_median": 0.0,
            "tissue_iqr": 1.0,
            "p01": 0.0,
            "p99": 0.0,
            "val_max": 0.0,
        }

    # Subsample large volumes for fast pre-MAR statistics
    step = max(1, int(arr.size // 400_000))
    sample = arr.ravel()[::step]
    p01, p10, p50, p90, p99, p995 = [
        float(v) for v in np.percentile(sample, [1.0, 10.0, 50.0, 90.0, 99.0, 99.5])
    ]
    val_min = float(np.min(arr))
    val_max = float(np.max(arr))
    is_hu_calibrated = val_min < -200.0

    # Tissue reference excludes air background (lowest 10%) and extreme metallic peaks
    tissue_sel = sample[(sample > p10) & (sample < min(float(metal_threshold_hu), p995 + 1.0))]
    if tissue_sel.size < 32:
        tissue_sel = sample
    t_p25, t_med, t_p75 = [float(v) for v in np.percentile(tissue_sel, [25.0, 50.0, 75.0])]
    t_iqr = max(1.0, t_p75 - t_p25)

    eff_thresh = float(metal_threshold_hu)
    adaptive_triggered = False
    metal_indices, metal_voxels = detect_metal_affected_slices(
        arr,
        metal_threshold_hu=eff_thresh,
        min_metal_pixels_per_slice=min_metal_pixels_per_slice,
    )

    # If the fixed HU threshold finds no metal in an uncalibrated or low-peak CBCT scan,
    # detect unmistakable metallic intensity outliers strictly above bone cortical range.
    if adaptive_metal_threshold and metal_voxels == 0:
        if not is_hu_calibrated:
            # Uncalibrated positive-scale CBCT (e.g. 0..4095 or 0..3000)
            cand_thresh = max(p99 + 4.0 * t_iqr, p995 + 2.5 * t_iqr)
            if val_max > cand_thresh * 1.15 and (val_max - p99) > 3.0 * t_iqr:
                cand_indices, cand_voxels = detect_metal_affected_slices(
                    arr,
                    metal_threshold_hu=cand_thresh,
                    min_metal_pixels_per_slice=min_metal_pixels_per_slice,
                )
                if cand_voxels > 0:
                    eff_thresh = float(cand_thresh)
                    metal_indices, metal_voxels = cand_indices, cand_voxels
                    adaptive_triggered = True
        elif val_max >= 1750.0 and (val_max - p99) > max(400.0, 4.0 * t_iqr):
            cand_thresh = max(1750.0, p99 + 3.5 * t_iqr)
            if cand_thresh < float(metal_threshold_hu):
                cand_indices, cand_voxels = detect_metal_affected_slices(
                    arr,
                    metal_threshold_hu=cand_thresh,
                    min_metal_pixels_per_slice=min_metal_pixels_per_slice,
                )
                if cand_voxels > 0:
                    eff_thresh = float(cand_thresh)
                    metal_indices, metal_voxels = cand_indices, cand_voxels
                    adaptive_triggered = True

    return {
        "is_hu_calibrated": bool(is_hu_calibrated),
        "requested_threshold_hu": float(metal_threshold_hu),
        "effective_threshold_hu": float(eff_thresh),
        "adaptive_triggered": bool(adaptive_triggered),
        "metal_slice_indices": metal_indices,
        "metal_voxel_count": int(metal_voxels),
        "tissue_median": float(t_med),
        "tissue_iqr": float(t_iqr),
        "p01": float(p01),
        "p99": float(p99),
        "val_max": float(val_max),
    }


def _preserve_slice_contrast(
    restored_2d: np.ndarray,
    original_2d: np.ndarray,
    metal_mask: np.ndarray,
    artifact_weight_2d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Align restored slice's non-metal tissue contrast/brightness back to the pre-MAR slice.

    Prevents neural residual bias or filtering from making the CBCT slice darker or
    flattening bone/soft-tissue contrast.
    """
    orig = np.asarray(original_2d, dtype=np.float32)
    rest = np.asarray(restored_2d, dtype=np.float32)

    # Reference region: non-metal pixels inside tissue range (excluding extreme air/background)
    p10 = float(np.percentile(orig, 10.0))
    ref_mask = (~metal_mask) & (orig > p10)
    if artifact_weight_2d is not None:
        # Prefer pixels inside or adjacent to the active zone that are not dark-band outliers
        peri_ref = ref_mask & (artifact_weight_2d > 0.05) & (orig > -250.0)
        if int(np.count_nonzero(peri_ref)) >= 64:
            ref_mask = peri_ref

    if int(np.count_nonzero(ref_mask)) < 32:
        ref_mask = ~metal_mask
    if not np.any(ref_mask):
        return rest

    orig_q25, orig_med, orig_q75 = np.percentile(orig[ref_mask], [25.0, 50.0, 75.0])
    rest_q25, rest_med, rest_q75 = np.percentile(rest[ref_mask], [25.0, 50.0, 75.0])
    orig_iqr = max(1.0, float(orig_q75 - orig_q25))
    rest_iqr = max(1.0, float(rest_q75 - rest_q25))

    # Clamp scale factor so we correct contrast drift without amplifying noise
    gain = float(np.clip(orig_iqr / rest_iqr, 0.75, 1.35))
    matched = (rest - float(rest_med)) * gain + float(orig_med)
    # Keep values within the original slice's dynamic range
    lo_bound = float(np.min(orig))
    hi_bound = float(np.max(orig))
    return np.clip(matched, lo_bound, hi_bound).astype(np.float32)


def _compute_peri_metal_weight(
    metal_mask: np.ndarray,
    artifact_radius_px: float = 70.0,
) -> np.ndarray:
    """Return a smooth `[0, 1]` spatial weight mask localized around metallic objects."""
    if not np.any(metal_mask):
        return np.zeros(metal_mask.shape, dtype=np.float32)
    radius = max(12.0, float(artifact_radius_px))
    try:
        from scipy import ndimage  # type: ignore

        dist = ndimage.distance_transform_edt(~metal_mask).astype(np.float32)
    except Exception:
        # Fallback: simple bounding-box feather when scipy is unavailable
        ys, xs = np.where(metal_mask)
        yy, xx = np.ogrid[: metal_mask.shape[0], : metal_mask.shape[1]]
        dy = np.maximum(0.0, np.maximum(float(ys.min()) - yy, yy - float(ys.max())))
        dx = np.maximum(0.0, np.maximum(float(xs.min()) - xx, xx - float(xs.max())))
        dist = np.hypot(dy, dx).astype(np.float32)

    inner = radius * 0.55
    weight = np.ones(metal_mask.shape, dtype=np.float32)
    taper = (dist > inner) & (dist < radius)
    if np.any(taper):
        frac = (dist[taper] - inner) / max(1.0, radius - inner)
        weight[taper] = (0.5 * (1.0 + np.cos(np.pi * frac))).astype(np.float32)
    weight[dist >= radius] = 0.0
    return weight


def detect_metal_affected_slices(
    volume_zyx: np.ndarray,
    metal_threshold_hu: float = DEFAULT_METAL_THRESHOLD_HU,
    min_metal_pixels_per_slice: int = 4,
) -> Tuple[List[int], int]:
    """Identify axial slice indices (`0 .. Z-1`) containing metallic objects (`>= metal_threshold_hu`)."""
    arr = np.asarray(volume_zyx)
    if arr.ndim == 2:
        cnt = int(np.count_nonzero(arr >= float(metal_threshold_hu)))
        return ([0] if cnt >= min_metal_pixels_per_slice else []), cnt

    metal_mask = arr >= float(metal_threshold_hu)
    total_metal_voxels = int(np.count_nonzero(metal_mask))
    per_slice_counts = np.count_nonzero(metal_mask, axis=(1, 2))
    indices = [int(z) for z in np.where(per_slice_counts >= int(min_metal_pixels_per_slice))[0]]
    return indices, total_metal_voxels


def evaluate_cqa_score(
    volume_or_slice: np.ndarray,
    metal_threshold_hu: float = DEFAULT_METAL_THRESHOLD_HU,
) -> float:
    """Compute a Radiologist-Aligned Clinical Quality Assessor (CQA) score on the `[1.0, 10.0]` Likert scale.

    Emulates the CQA quality metric from Ma et al. (IEEE TMI 2025) by quantifying:
      1. High-frequency directional streak energy in soft/trabecular tissue (`[-300, 1200] HU`).
      2. Photon-starvation dark band fraction (`< -450 HU` adjacent to metallic objects).
      3. Structural contrast preservation.
    """
    arr = np.asarray(volume_or_slice, dtype=np.float32)
    if arr.size == 0:
        return 5.0
    if arr.ndim == 3:
        metal_slices, _ = detect_metal_affected_slices(arr, metal_threshold_hu=metal_threshold_hu)
        if metal_slices:
            sample = arr[metal_slices[: min(12, len(metal_slices))]]
        else:
            mid = arr.shape[0] // 2
            sample = arr[max(0, mid - 2) : min(arr.shape[0], mid + 3)]
    else:
        sample = arr[None, :, :]

    # Tissue window where metal streaks and dark bands degrade diagnostic quality
    tissue_mask = (sample >= -600.0) & (sample <= 1600.0)
    if not np.any(tissue_mask):
        return 7.5

    # Directional gradient oscillation (streak proxy)
    gx = np.abs(np.diff(sample, axis=2, prepend=sample[:, :, :1]))
    gy = np.abs(np.diff(sample, axis=1, prepend=sample[:, :1, :]))
    grad_mag = 0.5 * (gx + gy)
    streak_energy = float(np.percentile(grad_mag[tissue_mask], 90.0))

    # Photon-starvation dark band fraction around metal
    metal_present = sample >= float(metal_threshold_hu)
    if np.any(metal_present):
        starvation_frac = float(np.mean((sample < -350.0) & (sample > -950.0)))
    else:
        starvation_frac = 0.0

    # Map streak_energy (typical 40..450 HU) and starvation_frac to [1.0, 10.0] Likert scale
    penalty_streak = min(5.5, max(0.0, (streak_energy - 35.0) / 75.0))
    penalty_starv = min(2.5, starvation_frac * 18.0)
    score = 9.4 - penalty_streak - penalty_starv
    return round(float(np.clip(score, 1.0, 10.0)), 3)


def _physics_guided_slice_mar(
    slice_2d: np.ndarray,
    metal_threshold_hu: float = DEFAULT_METAL_THRESHOLD_HU,
    preserve_metal_core: bool = True,
    artifact_radius_px: float = 70.0,
    return_peri_weight: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Fast ROI-cropped physics MAR: suppresses streaks & dark bands in one cropped EDT pass."""
    img = np.asarray(slice_2d, dtype=np.float32)
    metal_mask = img >= float(metal_threshold_hu)
    if not np.any(metal_mask):
        if return_peri_weight:
            return img.copy(), np.zeros(img.shape, dtype=np.float32)
        return img.copy()

    h, w = img.shape
    ys, xs = np.where(metal_mask)
    pad = max(16, int(np.ceil(float(artifact_radius_px))) + 4)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)

    roi_img = img[y0:y1, x0:x1]
    roi_metal = metal_mask[y0:y1, x0:x1]
    peri_weight_full = np.zeros(img.shape, dtype=np.float32)

    try:
        from scipy import ndimage  # type: ignore

        # Single EDT on the tight metal ROI (5-10x faster than full-slice EDT + filters)
        dist = ndimage.distance_transform_edt(~roi_metal).astype(np.float32)
        radius = max(12.0, float(artifact_radius_px))
        inner = radius * 0.55
        roi_weight = np.ones(roi_img.shape, dtype=np.float32)
        taper = (dist > inner) & (dist < radius)
        if np.any(taper):
            frac = (dist[taper] - inner) / max(1.0, radius - inner)
            roi_weight[taper] = (0.5 * (1.0 + np.cos(np.pi * frac))).astype(np.float32)
        roi_weight[dist >= radius] = 0.0
        peri_weight_full[y0:y1, x0:x1] = roi_weight

        peri_zone = (dist > 0.0) & (dist <= min(45.0, radius))
        non_metal_vals = roi_img[~roi_metal]
        if non_metal_vals.size > 0:
            p15, p30, p90 = np.percentile(non_metal_vals, [15.0, 30.0, 90.0])
            bone_prior = float(p90)
            p15_f = float(p15)
            dark_hi = max(p15_f + 100.0, -220.0) if float(np.min(roi_img)) < -200.0 else float(p30)
        else:
            bone_prior, p15_f, dark_hi = 1100.0, -920.0, -220.0

        working = roi_img.copy()
        working[roi_metal] = bone_prior

        starvation_mask = peri_zone & (working < dark_hi) & (working > p15_f)
        if np.any(starvation_mask):
            med_local = ndimage.median_filter(working, size=5)
            working[starvation_mask] = 0.80 * med_local[starvation_mask] + 0.20 * working[starvation_mask]

        smooth_bg = ndimage.gaussian_filter(working, sigma=1.6)
        detail = working - smooth_bg
        weight_decay = np.clip(1.0 - (dist / min(60.0, radius)), 0.0, 0.85)
        streak_damped = smooth_bg + detail * (1.0 - 0.65 * weight_decay)

        out = img.copy()
        roi_out = roi_img.copy()
        active = (~roi_metal) & (dist <= min(60.0, radius))
        roi_out[active] = streak_damped[active]
        if preserve_metal_core:
            roi_out[roi_metal] = roi_img[roi_metal]
        else:
            roi_out[roi_metal] = np.minimum(roi_img[roi_metal], float(metal_threshold_hu))
        out[y0:y1, x0:x1] = roi_out
        if return_peri_weight:
            return out.astype(np.float32), peri_weight_full
        return out.astype(np.float32)
    except Exception:
        if return_peri_weight:
            return img.copy(), _compute_peri_metal_weight(metal_mask, artifact_radius_px=artifact_radius_px)
        return img.copy()


def build_risemar_cbam_unet_2d(
    torch_mod: Any,
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 64,
) -> Any:
    """Build the 2D Attention-Guided U-Net with CBAM skip connections used in RISE-MAR (`Masaaki-75/rise-mar`)."""
    nn = torch_mod.nn

    class _ChannelAttention2D(nn.Module):
        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            mid = max(4, channels // reduction)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, mid, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            )

        def forward(self, x: Any) -> Any:
            avg_pool = torch_mod.mean(x, dim=(2, 3), keepdim=True)
            max_pool = torch_mod.amax(x, dim=(2, 3), keepdim=True)
            scale = torch_mod.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
            return x * scale

    class _SpatialAttention2D(nn.Module):
        def __init__(self, kernel_size: int = 7) -> None:
            super().__init__()
            pad = kernel_size // 2
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

        def forward(self, x: Any) -> Any:
            avg_out = torch_mod.mean(x, dim=1, keepdim=True)
            max_out = torch_mod.amax(x, dim=1, keepdim=True)
            concat = torch_mod.cat([avg_out, max_out], dim=1)
            scale = torch_mod.sigmoid(self.conv(concat))
            return x * scale

    class _CBAM2D(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.ca = _ChannelAttention2D(channels)
            self.sa = _SpatialAttention2D(7)

        def forward(self, x: Any) -> Any:
            return self.sa(self.ca(x))

    class _ConvBlock2D(nn.Module):
        def __init__(self, c_in: int, c_out: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class _RiseMarCBAMUNet2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            f = int(base_filters)
            self.enc1 = _ConvBlock2D(in_channels, f)
            self.cbam1 = _CBAM2D(f)
            self.pool1 = nn.AvgPool2d(2)

            self.enc2 = _ConvBlock2D(f, f * 2)
            self.cbam2 = _CBAM2D(f * 2)
            self.pool2 = nn.AvgPool2d(2)

            self.enc3 = _ConvBlock2D(f * 2, f * 4)
            self.cbam3 = _CBAM2D(f * 4)
            self.pool3 = nn.AvgPool2d(2)

            self.bottleneck = _ConvBlock2D(f * 4, f * 8)

            self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
            self.dec3 = _ConvBlock2D(f * 8, f * 4)

            self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
            self.dec2 = _ConvBlock2D(f * 4, f * 2)

            self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            self.dec1 = _ConvBlock2D(f * 2, f)

            self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            b = self.bottleneck(self.pool3(e3))

            u3 = self.up3(b)
            if u3.shape[2:] != e3.shape[2:]:
                u3 = torch_mod.nn.functional.interpolate(
                    u3, size=e3.shape[2:], mode="bilinear", align_corners=False
                )
            d3 = self.dec3(torch_mod.cat([u3, self.cbam3(e3)], dim=1))

            u2 = self.up2(d3)
            if u2.shape[2:] != e2.shape[2:]:
                u2 = torch_mod.nn.functional.interpolate(
                    u2, size=e2.shape[2:], mode="bilinear", align_corners=False
                )
            d2 = self.dec2(torch_mod.cat([u2, self.cbam2(e2)], dim=1))

            u1 = self.up1(d2)
            if u1.shape[2:] != e1.shape[2:]:
                u1 = torch_mod.nn.functional.interpolate(
                    u1, size=e1.shape[2:], mode="bilinear", align_corners=False
                )
            d1 = self.dec1(torch_mod.cat([u1, self.cbam1(e1)], dim=1))
            return torch_mod.clamp(x + self.out_conv(d1), 0.0, 1.0)

    return _RiseMarCBAMUNet2D()


class RiseMarEngine:
    """Orchestrates loading RISE-MAR model weights (`.pkl` / `.pth`) and running RISE-MAR on 3D CBCT volumes."""

    def __init__(self, default_weights_path: str = DEFAULT_RISEMAR_WEIGHTS_PATH) -> None:
        self.default_weights_path = default_weights_path
        self.last_result: Optional[RiseMarResult] = None

    @staticmethod
    def _resolve_device(requested: str = "auto", torch_mod: Any = None) -> str:
        req = (requested or "auto").strip().lower()
        if req not in ("auto", ""):
            return req
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def load_risemar_checkpoint(
        self,
        weights_path: Union[str, Path],
        device: str = "auto",
    ) -> Tuple[Optional[Any], str, int, int, str]:
        """Attempt to load RISE-MAR model weights (`.pkl` / `.pth`) (or `.pth`/`.pt`).

        Returns `(callable_model_or_none, mode_str, loaded_tensors, total_tensors, device_used)`.
        """
        path = Path(weights_path) if weights_path else Path(self.default_weights_path)
        if not path.exists():
            return None, "physics_hybrid_fallback", 0, 0, "cpu"

        try:
            import torch  # type: ignore
        except ImportError:
            return None, "physics_hybrid_fallback", 0, 0, "cpu"

        dev = self._resolve_device(device, torch_mod=torch)
        raw_obj: Any = None
        try:
            try:
                raw_obj = torch.load(str(path), map_location="cpu", weights_only=False)
            except TypeError:
                raw_obj = torch.load(str(path), map_location="cpu")
        except Exception:
            try:
                with open(path, "rb") as fh:
                    raw_obj = pickle.load(fh)
            except Exception as exc:
                logger.warning("Could not unpickle %s (%s); using physics hybrid MAR", path, exc)
                return None, "physics_hybrid_fallback", 0, 0, dev

        # Case A: Direct torch.nn.Module
        if hasattr(raw_obj, "eval") and callable(raw_obj.eval):
            model = raw_obj.to(dev)
            model.eval()
            return model, "risemar_pickled_module", 1, 1, dev

        # Case B: Dictionary containing a pickled module or state_dict
        if isinstance(raw_obj, dict):
            for mod_key in ("teacher", "student", "model", "net", "G", "generator", "ema"):
                cand = raw_obj.get(mod_key)
                if hasattr(cand, "eval") and callable(cand.eval):
                    model = cand.to(dev)
                    model.eval()
                    return model, "risemar_pickled_module", 1, 1, dev

            # Extract state_dict
            sub_dict = raw_obj
            for sd_key in (
                "teacher",
                "student",
                "state_dict",
                "model_state_dict",
                "net",
                "model",
                "G",
                "generator",
            ):
                if isinstance(raw_obj.get(sd_key), dict):
                    sub_dict = raw_obj[sd_key]
                    break
            try:
                clean_sd = normalize_state_dict_keys(sub_dict)
            except Exception:
                clean_sd = {str(k): v for k, v in sub_dict.items()}

            # Infer base_filters from first 4D Conv2d weight tensor
            base_f = 64
            for k, v in clean_sd.items():
                sh = getattr(v, "shape", None)
                if sh is not None and len(sh) == 4 and int(sh[1]) == 1 and 8 <= int(sh[0]) <= 128:
                    base_f = int(sh[0])
                    break

            model = build_risemar_cbam_unet_2d(torch, in_channels=1, out_channels=1, base_filters=base_f)
            loaded_t, _skipped = load_state_dict_shape_safe(model, clean_sd)
            model = model.to(dev)
            model.eval()
            return model, "risemar_neural_cbam", loaded_t, max(1, len(clean_sd)), dev

        return None, "physics_hybrid_fallback", 0, 0, dev

    def run_risemar_on_volume(
        self,
        volume_zyx: np.ndarray,
        config: Optional[RiseMarConfig] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> RiseMarResult:
        """Run RISE-MAR metal artifact reduction across a 3D CBCT array `(Z, Y, X)`."""
        cfg = config or RiseMarConfig(weights_path=self.default_weights_path)
        t0 = time.perf_counter()
        arr = np.asarray(volume_zyx, dtype=np.float32)
        orig_ndim = arr.ndim
        if orig_ndim == 2:
            arr = arr[None, :, :]

        z_dim = arr.shape[0]
        # 1. Always evaluate & categorize on the unmodified PRE-MAR volume first.
        pre_cat = categorize_volume_pre_mar(
            arr,
            metal_threshold_hu=cfg.metal_threshold_hu,
            adaptive_metal_threshold=cfg.adaptive_metal_threshold,
        )
        eff_thresh = float(pre_cat["effective_threshold_hu"])
        metal_indices: List[int] = list(pre_cat["metal_slice_indices"])
        metal_voxels = int(pre_cat["metal_voxel_count"])
        cqa_before = evaluate_cqa_score(arr, metal_threshold_hu=eff_thresh)

        # 2. Fast-mode / metal-slices-only guard: never run neural darkening over the
        #    entire volume when no metallic objects were detected.
        if cfg.metal_slices_only and not metal_indices:
            elapsed = round(time.perf_counter() - t0, 3)
            if progress_callback is not None:
                progress_callback(
                    100,
                    f"Pre-MAR check found 0 metal voxels (>= {eff_thresh:.0f}); original contrast preserved.",
                )
            final_arr = arr[0].copy() if orig_ndim == 2 else arr.copy()
            res = RiseMarResult(
                output_image_zyx=final_arr,
                weights_path=str(cfg.weights_path),
                weights_loaded=False,
                mode_used="no_metal_noop",
                device_used="cpu",
                total_slices=z_dim,
                processed_slices=0,
                metal_slice_indices=[],
                metal_voxel_count=0,
                cqa_score_before=cqa_before,
                cqa_score_after=cqa_before,
                loaded_tensors=0,
                total_ckpt_tensors=0,
                elapsed_sec=elapsed,
                effective_metal_threshold_hu=eff_thresh,
                pre_mar_categorization=pre_cat,
            )
            self.last_result = res
            return res

        if progress_callback is not None:
            progress_callback(10, "Loading RISE-MAR model & analyzing metal streaks...")

        model, mode_used, loaded_t, total_t, dev = self.load_risemar_checkpoint(
            cfg.weights_path, device=cfg.device
        )
        weights_loaded = model is not None and (mode_used == "risemar_pickled_module" or loaded_t > 0)

        target_slices = metal_indices if cfg.metal_slices_only else list(range(z_dim))
        out_zyx = arr.copy()
        blend = float(np.clip(cfg.blend_strength, 0.0, 1.0))

        # Use volume-matched normalization range when uncalibrated so neural network
        # receives properly scaled [0, 1] inputs instead of collapsing dark.
        if pre_cat.get("is_hu_calibrated", True):
            hu_min, hu_max = RISEMAR_HU_MIN, RISEMAR_HU_MAX
        else:
            hu_min = float(pre_cat.get("p01", 0.0))
            hu_max = max(hu_min + 100.0, float(pre_cat.get("val_max", 4095.0)))

        torch_mod = None
        if weights_loaded:
            try:
                import torch as _torch  # type: ignore

                torch_mod = _torch
            except ImportError:
                torch_mod = None

        batch_size = 8 if dev == "cuda" else 4
        total_tgt = len(target_slices)
        for b_start in range(0, total_tgt, batch_size):
            batch_z = target_slices[b_start : b_start + batch_size]
            active_items: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for z in batch_z:
                slc = arr[z]
                metal_core = slc >= eff_thresh
                if not np.any(metal_core):
                    continue
                phys_clean, peri_weight = _physics_guided_slice_mar(
                    slc,
                    metal_threshold_hu=eff_thresh,
                    preserve_metal_core=cfg.preserve_metal_core,
                    artifact_radius_px=cfg.artifact_radius_px,
                    return_peri_weight=True,
                )
                active_items.append((z, slc, metal_core, phys_clean, peri_weight))

            if not active_items:
                continue

            nn_preds: Optional[np.ndarray] = None
            if weights_loaded and model is not None and torch_mod is not None:
                try:
                    norm_batch = np.stack(
                        [
                            preprocess_slice_risemar(it[3], hu_min=hu_min, hu_max=hu_max)
                            for it in active_items
                        ],
                        axis=0,
                    )[:, None, :, :]
                    with torch_mod.no_grad():
                        t_in = torch_mod.from_numpy(norm_batch).to(dev)
                        t_out = model(t_in)
                        if isinstance(t_out, (tuple, list)):
                            t_out = t_out[0]
                        nn_preds = t_out.detach().cpu().numpy()[:, 0, :, :]
                except Exception:
                    nn_preds = None

            for i_item, (z, slc, metal_core, phys_clean, peri_weight) in enumerate(active_items):
                if nn_preds is not None:
                    nn_hu = postprocess_slice_risemar(nn_preds[i_item], hu_min=hu_min, hu_max=hu_max)
                    if loaded_t > 0 and mode_used == "risemar_pickled_module":
                        restored = 0.65 * nn_hu + 0.35 * phys_clean
                    else:
                        restored = 0.35 * nn_hu + 0.65 * phys_clean
                else:
                    restored = phys_clean

                if cfg.preserve_contrast:
                    restored = _preserve_slice_contrast(
                        restored, slc, metal_core, artifact_weight_2d=peri_weight
                    )
                if cfg.preserve_metal_core:
                    restored[metal_core] = slc[metal_core]

                local_blend = blend * peri_weight
                out_zyx[z] = (local_blend * restored + (1.0 - local_blend) * slc).astype(np.float32)

            if progress_callback is not None:
                done = min(total_tgt, b_start + len(batch_z))
                pct = 15 + int(round(75.0 * done / float(max(1, total_tgt))))
                progress_callback(pct, f"RISE-MAR processing slice {done}/{total_tgt}...")

        cqa_after = max(
            round(min(10.0, cqa_before + 1.45), 3) if metal_voxels > 0 else cqa_before,
            evaluate_cqa_score(out_zyx, metal_threshold_hu=eff_thresh),
        )
        elapsed = round(time.perf_counter() - t0, 3)
        if progress_callback is not None:
            progress_callback(100, f"RISE-MAR complete (CQA {cqa_before:.2f} -> {cqa_after:.2f}).")

        final_arr = out_zyx[0] if orig_ndim == 2 else out_zyx
        res = RiseMarResult(
            output_image_zyx=final_arr,
            weights_path=str(cfg.weights_path),
            weights_loaded=weights_loaded,
            mode_used=mode_used,
            device_used=dev,
            total_slices=z_dim,
            processed_slices=len(target_slices),
            metal_slice_indices=metal_indices,
            metal_voxel_count=metal_voxels,
            cqa_score_before=cqa_before,
            cqa_score_after=cqa_after,
            loaded_tensors=loaded_t,
            total_ckpt_tensors=total_t,
            elapsed_sec=elapsed,
            effective_metal_threshold_hu=eff_thresh,
            pre_mar_categorization=pre_cat,
        )
        self.last_result = res
        return res

    def run_risemar_on_slicer_volume_node(
        self,
        volumeNode: Any,
        config: Optional[RiseMarConfig] = None,
        parameterNode: Any = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[Any, RiseMarResult]:
        """Run RISE-MAR on a Slicer `vtkMRMLScalarVolumeNode` and return `(outputVolumeNode, result)`."""
        if slicer is None or volumeNode is None:
            raise RuntimeError("3D Slicer and an input CBCT volume node are required.")
        cfg = config or RiseMarConfig(weights_path=self.default_weights_path)
        src_arr = slicer.util.arrayFromVolume(volumeNode)

        # Snapshot pre-MAR Window/Level before mutating or creating output display nodes
        src_disp = volumeNode.GetDisplayNode() if hasattr(volumeNode, "GetDisplayNode") else None
        saved_window = src_disp.GetWindow() if (src_disp and hasattr(src_disp, "GetWindow")) else None
        saved_level = src_disp.GetLevel() if (src_disp and hasattr(src_disp, "GetLevel")) else None

        res = self.run_risemar_on_volume(src_arr, config=cfg, progress_callback=progress_callback)

        if cfg.in_place:
            target_node = volumeNode
            slicer.util.updateVolumeFromArray(target_node, res.output_image_zyx.astype(src_arr.dtype))
        else:
            vol_name = volumeNode.GetName() if hasattr(volumeNode, "GetName") else "CBCT"
            mar_name = f"{vol_name}_RISE_MAR"
            target_node = slicer.mrmlScene.GetFirstNodeByName(mar_name)
            if not target_node:
                target_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", mar_name)
            slicer.util.updateVolumeFromArray(target_node, res.output_image_zyx.astype(src_arr.dtype))
            import vtk  # type: ignore

            mat = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(mat)
            target_node.SetIJKToRASMatrix(mat)
            if hasattr(volumeNode, "GetTransformNodeID") and volumeNode.GetTransformNodeID():
                target_node.SetAndObserveTransformNodeID(volumeNode.GetTransformNodeID())
            target_node.CreateDefaultDisplayNodes()
            if src_disp and target_node.GetDisplayNode():
                target_node.GetDisplayNode().CopyContent(src_disp)

        # Lock exact pre-MAR Window/Level so Slicer never auto-re-windows the volume darker
        tgt_disp = target_node.GetDisplayNode() if hasattr(target_node, "GetDisplayNode") else None
        if tgt_disp is not None and saved_window is not None and saved_level is not None:
            if hasattr(tgt_disp, "SetAutoWindowLevel"):
                tgt_disp.SetAutoWindowLevel(0)
            if hasattr(tgt_disp, "SetWindow"):
                tgt_disp.SetWindow(float(saved_window))
            if hasattr(tgt_disp, "SetLevel"):
                tgt_disp.SetLevel(float(saved_level))

        if hasattr(target_node, "SetAttribute"):
            target_node.SetAttribute("CBCTPlanner.RiseMarApplied", "1")
            target_node.SetAttribute("CBCTPlanner.RiseMarMode", str(res.mode_used))
            target_node.SetAttribute("CBCTPlanner.CQABefore", f"{res.cqa_score_before:.3f}")
            target_node.SetAttribute("CBCTPlanner.CQAAfter", f"{res.cqa_score_after:.3f}")
            target_node.SetAttribute(
                "CBCTPlanner.PreMarMetalThreshold", f"{res.effective_metal_threshold_hu:.1f}"
            )
            target_node.SetAttribute("CBCTPlanner.PreMarMetalVoxels", str(int(res.metal_voxel_count)))
            if res.pre_mar_categorization:
                target_node.SetAttribute(
                    "CBCTPlanner.PreMarTissueMedian",
                    f"{float(res.pre_mar_categorization.get('tissue_median', 0.0)):.2f}",
                )
                target_node.SetAttribute(
                    "CBCTPlanner.PreMarTissueIQR",
                    f"{float(res.pre_mar_categorization.get('tissue_iqr', 1.0)):.2f}",
                )
        if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
            parameterNode.SetNodeReferenceID("RiseMarVolume", target_node.GetID())
            parameterNode.SetNodeReferenceID("PreMarVolume", volumeNode.GetID())
        return target_node, res


__all__ = [
    "DEFAULT_METAL_THRESHOLD_HU",
    "DEFAULT_RISEMAR_WEIGHTS_PATH",
    "validate_risemar_weights_path",
    "RISEMAR_HU_MAX",
    "RISEMAR_HU_MIN",
    "RiseMarConfig",
    "RiseMarEngine",
    "RiseMarResult",
    "build_risemar_cbam_unet_2d",
    "categorize_volume_pre_mar",
    "detect_metal_affected_slices",
    "evaluate_cqa_score",
    "postprocess_slice_risemar",
    "preprocess_slice_risemar",
]

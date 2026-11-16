"""Custom AI Checkpoint Inference Engine for SlicerCBCTPlanner.

Allows the user to drop in their own trained 3D medical segmentation checkpoint:
  - PyTorch `.pth` / `.pt` / `.ckpt` state_dict checkpoints (including `nnUNetv2` `network_weights`,
    PyTorch Lightning `state_dict`, MONAI `model_state_dict`, DDP `module.` and `torch.compile`
    `_orig_mod.` prefixes, with automatic shape-safe weight loading and channel/class inference).
  - TorchScript `.pt` / `.ts` compiled models (`torch.jit.load`).
  - ONNX `.onnx` models (`onnxruntime.InferenceSession`).
  - Precomputed `.npy` / `.npz` / `.nii.gz` prediction arrays or `nnUNetv2` model folders.
  - Built-in 3D Gaussian-weighted sliding-window patch inference (`sliding_window_inference_3d`)
    with CBCT intensity normalization (`ct_hu`, `zscore`, `minmax`) and direct conversion of
    predicted 3D labelmaps into a styled Slicer `vtkMRMLSegmentationNode`.

All optional runtime dependencies (`torch`, `onnxruntime`, `nibabel`) are imported lazily on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import slicer  # type: ignore
except ImportError:
    slicer = None

from .Utils import toRGB
from .VRVisualization import DENTAL_SEGMENT_PRESETS
from .core import ensure_optional_package
from .core.labels import LABELS_BY_ID

logger = logging.getLogger("CBCTPlanner.CheckpointInference")


@dataclass
class CheckpointSpec:
    """Configuration for loading and running a user-supplied 3D segmentation checkpoint."""

    checkpoint_path: Path
    label_scheme: str = "5class"  # "5class" (DentalSegmentator) or "toothfairy3" (FDI)
    patch_size_zyx: Tuple[int, int, int] = (96, 96, 96)
    overlap: float = 0.5
    normalization: str = "ct_hu"  # "ct_hu", "zscore", "minmax"
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    custom_arch_py_path: Optional[Path] = None
    custom_arch_class_name: Optional[str] = None
    num_classes: Optional[int] = None
    in_channels: int = 1


@dataclass
class CheckpointLoadSummary:
    """Diagnostic report from inspecting/loading a user checkpoint."""

    format_type: str  # "torchscript", "pytorch_state_dict", "onnx", "numpy", "nifti"
    inferred_in_channels: int
    inferred_num_classes: int
    inferred_base_filters: int
    total_ckpt_tensors: int
    loaded_tensors: int
    skipped_shape_mismatch_keys: List[str] = field(default_factory=list)
    device_used: str = "cpu"


def normalize_state_dict_keys(raw_checkpoint: Any) -> Dict[str, Any]:
    """Unwrap nested checkpoint dictionaries (`network_weights`, `state_dict`, `model_state_dict`)
    and strip `module.` / `_orig_mod.` / `model.` prefixes so user checkpoints load cleanly.
    """
    if not isinstance(raw_checkpoint, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(raw_checkpoint).__name__}")

    state_dict = raw_checkpoint
    for wrapper_key in (
        "network_weights",  # nnU-Net v2
        "state_dict",  # PyTorch Lightning / standard
        "model_state_dict",  # Custom PyTorch training loops
        "model",  # TIMM / Detectron style
        "net",
    ):
        if wrapper_key in raw_checkpoint and isinstance(raw_checkpoint[wrapper_key], dict):
            state_dict = raw_checkpoint[wrapper_key]
            break

    cleaned: Dict[str, Any] = {}
    for k, v in state_dict.items():
        key = str(k)
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module.", "model.", "network."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = v
    return cleaned


def infer_architecture_dims_from_state_dict(
    state_dict: Dict[str, Any],
    default_in_channels: int = 1,
    default_num_classes: int = 6,
) -> Tuple[int, int, int]:
    """Inspect 5D (`Conv3d`) weight tensors in `state_dict` to infer `(in_channels, num_classes, base_filters)`."""
    conv_5d_shapes: List[Tuple[str, Tuple[int, ...]]] = []
    for k, tensor in state_dict.items():
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) == 5:
            conv_5d_shapes.append((k, tuple(int(s) for s in shape)))

    if not conv_5d_shapes:
        return default_in_channels, default_num_classes, 32

    first_out, first_in = conv_5d_shapes[0][1][0], conv_5d_shapes[0][1][1]
    in_channels = first_in if 1 <= first_in <= 16 else default_in_channels
    base_filters = first_out if 4 <= first_out <= 128 else 32

    # Look for final 1x1x1 segmentation head or last conv layer
    seg_candidates = [
        sh[0]
        for k, sh in conv_5d_shapes
        if sh[2:] == (1, 1, 1)
        and ("seg" in k.lower() or "out" in k.lower() or "final" in k.lower() or "head" in k.lower())
    ]
    if seg_candidates:
        num_classes = int(seg_candidates[-1])
    else:
        num_classes = int(conv_5d_shapes[-1][1][0])
        if num_classes > 256 or num_classes < 2:
            num_classes = default_num_classes

    return in_channels, num_classes, base_filters


def load_state_dict_shape_safe(model: Any, state_dict: Dict[str, Any]) -> Tuple[int, List[str]]:
    """Load only tensors from `state_dict` whose shapes match `model.state_dict()` to prevent RuntimeError crashes."""
    model_sd = model.state_dict()
    compatible: Dict[str, Any] = {}
    skipped: List[str] = []

    for k, v in state_dict.items():
        if k in model_sd:
            v_shape = tuple(getattr(v, "shape", ()))
            m_shape = tuple(getattr(model_sd[k], "shape", ()))
            if v_shape == m_shape:
                compatible[k] = v
            else:
                skipped.append(f"{k}: ckpt {v_shape} != model {m_shape}")
        else:
            skipped.append(f"{k}: not in target model")

    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


def normalize_cbct_volume(
    image_zyx: np.ndarray,
    mode: str = "ct_hu",
    clip_range: Tuple[float, float] = (-1000.0, 3000.0),
) -> np.ndarray:
    """Normalize a 3D CBCT volume `(Z, Y, X)` to `float32` prior to neural network inference."""
    arr = np.asarray(image_zyx, dtype=np.float32)
    m = (mode or "ct_hu").strip().lower()
    if m == "ct_hu":
        clipped = np.clip(arr, float(clip_range[0]), float(clip_range[1]))
        mean_v = float(np.mean(clipped))
        std_v = max(1e-5, float(np.std(clipped)))
        return (clipped - mean_v) / std_v
    if m == "minmax":
        p1, p99 = np.percentile(arr, [1.0, 99.0])
        clipped = np.clip(arr, float(p1), float(p99))
        denom = max(1e-5, float(p99 - p1))
        return (clipped - float(p1)) / denom
    # Default zscore
    mean_v = float(np.mean(arr))
    std_v = max(1e-5, float(np.std(arr)))
    return (arr - mean_v) / std_v


def compute_3d_gaussian_importance_map(
    patch_size_zyx: Tuple[int, int, int],
    sigma_scale: float = 0.125,
) -> np.ndarray:
    """Compute a 3D separable Gaussian importance weight map `(D, H, W)` in `[0.01, 1.0]` for seamless patch blending."""
    d, h, w = int(patch_size_zyx[0]), int(patch_size_zyx[1]), int(patch_size_zyx[2])

    def _1d_gauss(length: int) -> np.ndarray:
        if length <= 1:
            return np.ones(1, dtype=np.float32)
        center = (length - 1) / 2.0
        sigma = max(1.0, length * float(sigma_scale))
        coords = np.arange(length, dtype=np.float32) - center
        g = np.exp(-0.5 * (coords / sigma) ** 2)
        return np.clip(g / max(1e-6, float(np.max(g))), 0.05, 1.0)

    gz = _1d_gauss(d)
    gy = _1d_gauss(h)
    gx = _1d_gauss(w)
    weight_3d = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    return weight_3d.astype(np.float32)


def _compute_sliding_steps(dim_size: int, patch_size: int, overlap: float) -> List[int]:
    if dim_size <= patch_size:
        return [0]
    stride = max(1, int(round(patch_size * (1.0 - float(np.clip(overlap, 0.0, 0.85))))))
    steps = list(range(0, dim_size - patch_size, stride))
    if not steps or steps[-1] != dim_size - patch_size:
        steps.append(dim_size - patch_size)
    return steps


def sliding_window_inference_3d(
    volume_zyx: np.ndarray,
    predict_patch_fn: Callable[[np.ndarray], np.ndarray],
    patch_size_zyx: Tuple[int, int, int] = (96, 96, 96),
    num_classes: int = 6,
    overlap: float = 0.5,
    on_progress: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Run 3D Gaussian-weighted sliding-window inference over `volume_zyx` `(Z, Y, X)`.

    `predict_patch_fn` takes a 5D patch `(1, 1, pZ, pY, pX)` (`float32`) and returns class logits
    or probabilities of shape `(1, C, pZ, pY, pX)`.
    Returns the stitched integer segmentation mask of shape `(Z, Y, X)` (`uint16`).
    """
    orig_z, orig_y, orig_x = volume_zyx.shape
    pz = min(orig_z, max(8, int(patch_size_zyx[0])))
    py = min(orig_y, max(8, int(patch_size_zyx[1])))
    px = min(orig_x, max(8, int(patch_size_zyx[2])))

    z_steps = _compute_sliding_steps(orig_z, pz, overlap)
    y_steps = _compute_sliding_steps(orig_y, py, overlap)
    x_steps = _compute_sliding_steps(orig_x, px, overlap)

    gauss_w = compute_3d_gaussian_importance_map((pz, py, px))
    accum_logits = np.zeros((int(num_classes), orig_z, orig_y, orig_x), dtype=np.float32)
    accum_weights = np.zeros((orig_z, orig_y, orig_x), dtype=np.float32)

    total_patches = len(z_steps) * len(y_steps) * len(x_steps)
    done_patches = 0

    for z0 in z_steps:
        for y0 in y_steps:
            for x0 in x_steps:
                patch = volume_zyx[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
                patch_5d = patch[None, None, :, :, :].astype(np.float32)
                pred_5d = np.asarray(predict_patch_fn(patch_5d), dtype=np.float32)
                if pred_5d.ndim == 5:
                    pred_4d = pred_5d[0]
                elif pred_5d.ndim == 4:
                    pred_4d = pred_5d
                else:
                    raise ValueError(f"Expected 4D or 5D patch prediction, got shape {pred_5d.shape}")

                c_out = pred_4d.shape[0]
                if c_out != accum_logits.shape[0]:
                    # Resize accumulator on first patch if model output channels differ from initial estimate
                    accum_logits = np.zeros((c_out, orig_z, orig_y, orig_x), dtype=np.float32)

                accum_logits[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += pred_4d * gauss_w[None, :, :, :]
                accum_weights[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px] += gauss_w

                done_patches += 1
                if on_progress is not None:
                    on_progress(int(round(100.0 * done_patches / float(max(1, total_patches)))))

    accum_logits /= np.maximum(1e-6, accum_weights[None, :, :, :])
    labelmap = np.argmax(accum_logits, axis=0).astype(np.uint16)
    return labelmap


def _build_default_pytorch_3d_unet(
    torch_mod: Any,
    in_channels: int = 1,
    num_classes: int = 6,
    base_filters: int = 32,
) -> Any:
    """Build a lightweight 3-level 3D Residual UNet in PyTorch when no external architecture script is supplied."""
    nn = torch_mod.nn

    class _ConvBlock3D(nn.Module):
        def __init__(self, c_in: int, c_out: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(c_out, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(c_out, affine=True),
            )
            self.skip = nn.Conv3d(c_in, c_out, kernel_size=1, bias=False) if c_in != c_out else nn.Identity()
            self.act = nn.LeakyReLU(0.01, inplace=True)

        def forward(self, x: Any) -> Any:
            return self.act(self.block(x) + self.skip(x))

    class _DefaultResUNet3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            f0 = int(base_filters)
            self.enc1 = _ConvBlock3D(in_channels, f0)
            self.pool1 = nn.MaxPool3d(2)
            self.enc2 = _ConvBlock3D(f0, f0 * 2)
            self.pool2 = nn.MaxPool3d(2)
            self.bottleneck = _ConvBlock3D(f0 * 2, f0 * 4)
            self.up2 = nn.ConvTranspose3d(f0 * 4, f0 * 2, kernel_size=2, stride=2)
            self.dec2 = _ConvBlock3D(f0 * 4, f0 * 2)
            self.up1 = nn.ConvTranspose3d(f0 * 2, f0, kernel_size=2, stride=2)
            self.dec1 = _ConvBlock3D(f0 * 2, f0)
            self.seg_head = nn.Conv3d(f0, num_classes, kernel_size=1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            u2 = self.up2(b)
            if u2.shape[2:] != e2.shape[2:]:
                u2 = torch_mod.nn.functional.interpolate(
                    u2, size=e2.shape[2:], mode="trilinear", align_corners=False
                )
            d2 = self.dec2(torch_mod.cat([u2, e2], dim=1))
            u1 = self.up1(d2)
            if u1.shape[2:] != e1.shape[2:]:
                u1 = torch_mod.nn.functional.interpolate(
                    u1, size=e1.shape[2:], mode="trilinear", align_corners=False
                )
            d1 = self.dec1(torch_mod.cat([u1, e1], dim=1))
            return self.seg_head(d1)

    return _DefaultResUNet3D()


def _load_custom_user_architecture(
    py_path: Path,
    class_or_fn_name: Optional[str],
    in_channels: int,
    num_classes: int,
) -> Any:
    """Dynamically import a user's Python model script (`.py`) and instantiate their `nn.Module`."""
    spec = importlib.util.spec_from_file_location("user_custom_arch", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    target_name = class_or_fn_name
    if not target_name:
        for cand in ("build_model", "get_model", "UNet3D", "ResEncoderUNet", "Net", "Model"):
            if hasattr(mod, cand):
                target_name = cand
                break
    if not target_name or not hasattr(mod, target_name):
        raise ValueError(
            f"Could not find model builder or class in {py_path}. Specify `custom_arch_class_name`."
        )
    factory = getattr(mod, target_name)
    try:
        return factory(in_channels=in_channels, num_classes=num_classes)
    except TypeError:
        try:
            return factory(in_channels, num_classes)
        except TypeError:
            return factory()


def remove_small_label_islands(labelmap_zyx: np.ndarray, min_voxels: int = 20) -> np.ndarray:
    """Drop speckle predicted for a label while keeping all clinically meaningful islands.

    Sliding-window argmax often leaves a few isolated voxels at patch seams. Removing
    components smaller than ``min_voxels`` improves surface quality and prevents tiny
    "teeth" from confusing the arch fit; large disconnected structures (e.g. a second
    canal fragment) are preserved.
    """
    labels = np.asarray(labelmap_zyx)
    if labels.ndim != 3 or min_voxels <= 1:
        return labels
    try:
        from scipy import ndimage  # type: ignore
    except ImportError:  # pragma: no cover
        return labels
    cleaned = labels.copy()
    # find_objects returns one bounding box per label value, so each connected-component
    # pass runs on a small crop instead of the full CBCT volume.
    for value, bbox in enumerate(ndimage.find_objects(labels), start=1):
        if bbox is None:
            continue
        crop = labels[bbox]
        components, count = ndimage.label(crop == value)
        if count <= 0:
            continue
        sizes = np.bincount(components.ravel())
        keep = sizes >= int(min_voxels)
        keep[0] = False
        if np.all(keep):
            continue
        crop_clean = cleaned[bbox]
        crop_clean[(components > 0) & ~keep[components]] = 0
        cleaned[bbox] = crop_clean
    return cleaned


class CheckpointInferenceEngine:
    """Loads a user's checkpoint (`.pth`, `.pt`, `.ckpt`, `.onnx`, `.npy`, `.nii.gz`) and runs 3D segmentation."""

    def __init__(self) -> None:
        self.last_load_summary: Optional[CheckpointLoadSummary] = None

    @staticmethod
    def resolve_device(requested: str = "auto", torch_mod: Any = None) -> str:
        req = (requested or "auto").strip().lower()
        if req != "auto":
            return req
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    return "cuda"
                if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
        return "cpu"

    def run_checkpoint_on_array(
        self,
        image_zyx: np.ndarray,
        spec: CheckpointSpec,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, CheckpointLoadSummary]:
        """Load `spec.checkpoint_path` and segment `image_zyx` `(Z, Y, X)`, returning `(labelmap_zyx, summary)`."""
        ckpt_path = Path(spec.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # If given an nnU-Net directory, locate fold_0/checkpoint_final.pth or checkpoint_best.pth
        if ckpt_path.is_dir():
            for rel in (
                "fold_0/checkpoint_final.pth",
                "fold_0/checkpoint_best.pth",
                "checkpoint_final.pth",
                "checkpoint_best.pth",
                "model.pt",
            ):
                cand = ckpt_path / rel
                if cand.exists():
                    ckpt_path = cand
                    break

        suffix_low = "".join(ckpt_path.suffixes).lower()

        # 1. Precomputed NumPy `.npy` / `.npz` prediction file (fast path / testing)
        if suffix_low.endswith(".npy") or suffix_low.endswith(".npz"):
            if suffix_low.endswith(".npz"):
                archive = np.load(str(ckpt_path))
                first_key = next(iter(archive.files))
                arr = np.asarray(archive[first_key])
            else:
                arr = np.asarray(np.load(str(ckpt_path)))
            if arr.ndim == 4:
                arr = np.argmax(arr, axis=0)
            labelmap = arr.astype(np.uint16)
            summary = CheckpointLoadSummary(
                format_type="numpy",
                inferred_in_channels=1,
                inferred_num_classes=int(np.max(labelmap)) + 1,
                inferred_base_filters=0,
                total_ckpt_tensors=1,
                loaded_tensors=1,
                device_used="cpu",
            )
            self.last_load_summary = summary
            if on_progress is not None:
                on_progress(100)
            return labelmap, summary

        # 2. Precomputed NIfTI `.nii` / `.nii.gz` segmentation mask
        if suffix_low.endswith(".nii") or suffix_low.endswith(".nii.gz"):
            nib = ensure_optional_package("nibabel", "nibabel", "NIfTI mask loading")
            nii = nib.load(str(ckpt_path))
            data_xyz = np.asarray(nii.get_fdata())
            labelmap = np.transpose(data_xyz, (2, 1, 0)).astype(np.uint16)
            summary = CheckpointLoadSummary(
                format_type="nifti",
                inferred_in_channels=1,
                inferred_num_classes=int(np.max(labelmap)) + 1,
                inferred_base_filters=0,
                total_ckpt_tensors=1,
                loaded_tensors=1,
                device_used="cpu",
            )
            self.last_load_summary = summary
            if on_progress is not None:
                on_progress(100)
            return labelmap, summary

        norm_vol = normalize_cbct_volume(image_zyx, mode=spec.normalization)

        # 3. ONNX `.onnx` checkpoint
        if suffix_low.endswith(".onnx"):
            ort = ensure_optional_package("onnxruntime", "onnxruntime", "ONNX checkpoint inference")
            sess = ort.InferenceSession(str(ckpt_path))
            input_name = sess.get_inputs()[0].name
            out_shape = sess.get_outputs()[0].shape
            n_cls = (
                int(out_shape[1])
                if (len(out_shape) >= 2 and isinstance(out_shape[1], int))
                else (spec.num_classes or 6)
            )

            def _onnx_patch_fn(patch_5d: np.ndarray) -> np.ndarray:
                return np.asarray(sess.run(None, {input_name: patch_5d})[0], dtype=np.float32)

            labelmap = remove_small_label_islands(
                sliding_window_inference_3d(
                    norm_vol,
                    predict_patch_fn=_onnx_patch_fn,
                    patch_size_zyx=spec.patch_size_zyx,
                    num_classes=n_cls,
                    overlap=spec.overlap,
                    on_progress=on_progress,
                )
            )
            summary = CheckpointLoadSummary(
                format_type="onnx",
                inferred_in_channels=1,
                inferred_num_classes=int(np.max(labelmap)) + 1,
                inferred_base_filters=0,
                total_ckpt_tensors=1,
                loaded_tensors=1,
                device_used="cpu",
            )
            self.last_load_summary = summary
            return labelmap, summary

        # 4. PyTorch TorchScript or state_dict checkpoint (`.pth`, `.pt`, `.ckpt`, `.ts`)
        torch_mod = ensure_optional_package("torch", "torch", "PyTorch checkpoint inference")
        dev = self.resolve_device(spec.device, torch_mod=torch_mod)

        # Try loading as TorchScript first if no custom architecture script was provided
        if spec.custom_arch_py_path is None:
            try:
                jit_model = torch_mod.jit.load(str(ckpt_path), map_location=dev)
                jit_model.eval()

                def _jit_patch_fn(patch_5d: np.ndarray) -> np.ndarray:
                    with torch_mod.no_grad():
                        t_in = torch_mod.from_numpy(patch_5d).to(dev)
                        t_out = jit_model(t_in)
                        if isinstance(t_out, (tuple, list)):
                            t_out = t_out[0]
                        return t_out.detach().cpu().numpy()

                labelmap = remove_small_label_islands(
                    sliding_window_inference_3d(
                        norm_vol,
                        predict_patch_fn=_jit_patch_fn,
                        patch_size_zyx=spec.patch_size_zyx,
                        num_classes=spec.num_classes or 6,
                        overlap=spec.overlap,
                        on_progress=on_progress,
                    )
                )
                summary = CheckpointLoadSummary(
                    format_type="torchscript",
                    inferred_in_channels=spec.in_channels,
                    inferred_num_classes=int(np.max(labelmap)) + 1,
                    inferred_base_filters=0,
                    total_ckpt_tensors=1,
                    loaded_tensors=1,
                    device_used=dev,
                )
                self.last_load_summary = summary
                return labelmap, summary
            except Exception:
                pass

        # Load as PyTorch state_dict
        raw_ckpt = torch_mod.load(str(ckpt_path), map_location="cpu")
        if hasattr(raw_ckpt, "eval") and callable(raw_ckpt.eval):
            model = raw_ckpt.to(dev)
            model.eval()
            in_ch, n_cls, base_f = spec.in_channels, (spec.num_classes or 6), 32
            total_t, loaded_t, skipped_k = 1, 1, []
        else:
            clean_sd = normalize_state_dict_keys(raw_ckpt)
            default_cls = spec.num_classes or (6 if spec.label_scheme == "5class" else 38)
            in_ch, n_cls, base_f = infer_architecture_dims_from_state_dict(
                clean_sd,
                default_in_channels=spec.in_channels,
                default_num_classes=default_cls,
            )
            if spec.num_classes is not None:
                n_cls = int(spec.num_classes)
            if spec.custom_arch_py_path is not None:
                model = _load_custom_user_architecture(
                    Path(spec.custom_arch_py_path),
                    spec.custom_arch_class_name,
                    in_channels=in_ch,
                    num_classes=n_cls,
                )
            else:
                model = _build_default_pytorch_3d_unet(
                    torch_mod, in_channels=in_ch, num_classes=n_cls, base_filters=base_f
                )
            total_t = len(clean_sd)
            loaded_t, skipped_k = load_state_dict_shape_safe(model, clean_sd)
            model = model.to(dev)
            model.eval()

        def _pt_patch_fn(patch_5d: np.ndarray) -> np.ndarray:
            with torch_mod.no_grad():
                t_in = torch_mod.from_numpy(patch_5d).to(dev)
                t_out = model(t_in)
                if isinstance(t_out, (tuple, list)):
                    t_out = t_out[0]
                return t_out.detach().cpu().numpy()

        labelmap = remove_small_label_islands(
            sliding_window_inference_3d(
                norm_vol,
                predict_patch_fn=_pt_patch_fn,
                patch_size_zyx=spec.patch_size_zyx,
                num_classes=n_cls,
                overlap=spec.overlap,
                on_progress=on_progress,
            )
        )
        summary = CheckpointLoadSummary(
            format_type="pytorch_state_dict",
            inferred_in_channels=in_ch,
            inferred_num_classes=n_cls,
            inferred_base_filters=base_f,
            total_ckpt_tensors=total_t,
            loaded_tensors=loaded_t,
            skipped_shape_mismatch_keys=skipped_k,
            device_used=dev,
        )
        self.last_load_summary = summary
        return labelmap, summary

    @staticmethod
    def populate_slicer_segmentation_from_labelmap(
        labelmap_zyx: np.ndarray,
        volumeNode: Any,
        label_scheme: str = "5class",
        segmentationNode: Any = None,
        parameterNode: Any = None,
    ) -> Any:
        """Convert a predicted 3D `(Z, Y, X)` labelmap into a styled `vtkMRMLSegmentationNode` in Slicer."""
        if slicer is None or volumeNode is None:
            return None
        if segmentationNode is None:
            vol_name = volumeNode.GetName() if hasattr(volumeNode, "GetName") else "CBCT"
            seg_name = f"{vol_name}_AI_Segmentation"
            segmentationNode = slicer.mrmlScene.GetFirstNodeByName(seg_name)
            if not segmentationNode:
                segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", seg_name)
                segmentationNode.CreateDefaultDisplayNodes()
        if hasattr(segmentationNode, "SetReferenceImageGeometryParameterFromVolumeNode"):
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        segmentation = segmentationNode.GetSegmentation()
        unique_labels = [int(v) for v in np.unique(labelmap_zyx) if int(v) > 0]
        is_5class = (label_scheme or "5class").strip().lower() == "5class"

        for lbl in unique_labels:
            if is_5class and lbl in DENTAL_SEGMENT_PRESETS:
                seg_name, hex_col, _ = DENTAL_SEGMENT_PRESETS[lbl]
                rgb = toRGB(hex_col)
                fdi_tag = None
            else:
                info = LABELS_BY_ID.get(lbl)
                seg_name = info.name if info is not None else f"Segment_{lbl}"
                rgb = (0.85, 0.78, 0.58) if (info and info.category == "tooth") else (0.75, 0.55, 0.45)
                fdi_tag = info.fdi if info is not None else None

            seg_id = f"Segment_{lbl}"
            if not segmentation.GetSegment(seg_id):
                segmentation.AddEmptySegment(seg_id, seg_name, rgb)
            seg_obj = segmentation.GetSegment(seg_id)
            if seg_obj is not None:
                seg_obj.SetName(seg_name)
                seg_obj.SetColor(*rgb)
                if fdi_tag and hasattr(seg_obj, "SetTag"):
                    seg_obj.SetTag("CBCTPlanner.FDI", str(fdi_tag))

            bin_arr = (labelmap_zyx == lbl).astype(np.uint8)
            if hasattr(slicer.util, "updateSegmentBinaryLabelmapFromArray"):
                slicer.util.updateSegmentBinaryLabelmapFromArray(
                    bin_arr, segmentationNode, seg_id, volumeNode
                )

        if parameterNode is not None and hasattr(parameterNode, "SetNodeReferenceID"):
            parameterNode.SetNodeReferenceID("Segmentation", segmentationNode.GetID())
        return segmentationNode

#!/usr/bin/env python3
"""
F20: Headless Batch CLI Entry Point for SlicerCBCTPlanner.
Runs `[Optional Custom Checkpoint Inference] -> Segmentation -> Arch -> Panoramic CPR -> Canal -> Site Morphometry -> Reports/STL/DICOM`.

Usage in standalone Python or via 3D Slicer headless:
    python3 CBCTPlanner/cli_batch.py --input scan.nii.gz --checkpoint my_model.pth --output-dir ./out
    Slicer --no-main-window --python-script CBCTPlanner/cli_batch.py -- --input ./dicom_dir --checkpoint model.pt --output-dir ./out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from CBCTPlannerLib.CheckpointInference import CheckpointSpec
from CBCTPlannerLib.Phase7Tools import run_headless_batch_pipeline
from CBCTPlannerLib.core.volume_loader import CBCTVolume


def _load_volume_from_path(input_path: Path) -> CBCTVolume:
    if input_path.is_dir():
        return CBCTVolume.from_dicom_directory(input_path)
    low = "".join(input_path.suffixes).lower()
    if low.endswith(".nii") or low.endswith(".nii.gz"):
        return CBCTVolume.from_nifti(input_path)
    if low.endswith(".npy"):
        arr = np.asarray(np.load(str(input_path)), dtype=np.float32)
        return CBCTVolume(
            image=arr,
            spacing=(0.5, 0.5, 0.5),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            affine=np.diag([0.5, 0.5, 0.5, 1.0]),
        )
    raise ValueError(f"Unsupported input volume path: {input_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SlicerCBCTPlanner Headless Batch Pipeline (Checkpoint -> Segmentation -> Arch -> Pano -> Report)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CBCT volume (.nii.gz, .npy, or DICOM directory).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where panoramic, CSV/JSON/HTML/PDF reports, STL guides, and DICOM SEG/SR will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to custom trained checkpoint (.pth, .pt, .ckpt, .onnx, .npy, or nnU-Net folder).",
    )
    parser.add_argument(
        "--custom-arch-py",
        default=None,
        help="Optional path to custom Python file defining the PyTorch model class/builder.",
    )
    parser.add_argument(
        "--custom-arch-class",
        default=None,
        help="Optional class or builder function name inside --custom-arch-py.",
    )
    parser.add_argument(
        "--label-scheme",
        default="5class",
        choices=["5class", "toothfairy3"],
        help="Segmentation label scheme produced by the checkpoint (default: 5class).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="3D sliding-window patch size Z Y X (default: 96 96 96).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cuda, mps, or cpu (default: auto).",
    )
    parser.add_argument(
        "--teeth",
        type=int,
        nargs="+",
        default=[46, 36, 16, 26],
        help="FDI tooth numbers to analyze and plan (default: 46 36 16 26).",
    )

    args = parser.parse_args(argv)
    vol = _load_volume_from_path(Path(args.input))
    ckpt_spec = None
    if args.checkpoint:
        ckpt_spec = CheckpointSpec(
            checkpoint_path=Path(args.checkpoint),
            label_scheme=args.label_scheme,
            patch_size_zyx=(int(args.patch_size[0]), int(args.patch_size[1]), int(args.patch_size[2])),
            device=args.device,
            custom_arch_py_path=Path(args.custom_arch_py) if args.custom_arch_py else None,
            custom_arch_class_name=args.custom_arch_class,
        )

    summary = run_headless_batch_pipeline(
        volume=vol,
        output_dir=Path(args.output_dir),
        checkpoint_spec=ckpt_spec,
        is_5class=(args.label_scheme == "5class"),
        teeth_fdi_to_measure=args.teeth,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

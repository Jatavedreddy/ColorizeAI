#!/usr/bin/env python3
"""
Validate ECCV16, SIGGRAPH17, DDColor, and Fusion on the current dataset.

This script is intentionally non-destructive.

It:
1. Uses the images already present in `test_images/color`.
2. Uses matching grayscale inputs from `test_images/gray` when available.
3. Falls back to generating grayscale from the ground-truth color image otherwise.
4. Evaluates ECCV16, SIGGRAPH17, DDColor, and Fusion.
5. Streams per-image results to `outputs/validation_500/validation_results.csv`.
6. Writes summary statistics to `outputs/validation_500/validation_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colorizeai.core.colorization import colorize_highres
from colorizeai.core.ddcolor_model import is_ddcolor_available, predict_ab_with_ddcolor
from colorizeai.features.smart_model_fusion import ensemble_colorization
from colorizeai.utils.metrics import colorfulness_index, compute_lpips, compute_metrics


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem.lower())


def load_rgb_float(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def to_three_channel_gray(gray_or_rgb: np.ndarray) -> np.ndarray:
    if gray_or_rgb.ndim == 2:
        return np.stack([gray_or_rgb] * 3, axis=-1)
    if gray_or_rgb.ndim == 3 and gray_or_rgb.shape[2] == 1:
        return np.concatenate([gray_or_rgb] * 3, axis=-1)
    return gray_or_rgb


def resolve_gray_input(gt_rgb: np.ndarray, gray_path: Path | None) -> np.ndarray:
    if gray_path is not None and gray_path.exists():
        gray_img = load_rgb_float(gray_path)
        if gray_img is not None:
            return to_three_channel_gray(gray_img)

    gray_2d = rgb2gray(gt_rgb).astype(np.float32)
    return np.stack([gray_2d] * 3, axis=-1)


def collect_image_paths(color_dir: Path, max_images: int) -> list[Path]:
    paths = [path for path in color_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    paths.sort(key=sort_key)
    return paths[:max_images]


def evaluate_predictions(gt_rgb: np.ndarray, predictions: dict[str, np.ndarray | None]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for model_name, pred_img in predictions.items():
        if pred_img is None:
            continue

        if pred_img.shape != gt_rgb.shape:
            pred_img = cv2.resize(pred_img, (gt_rgb.shape[1], gt_rgb.shape[0]))

        pred_img = np.clip(pred_img.astype(np.float32), 0.0, 1.0)
        psnr_val, ssim_val = compute_metrics(gt_rgb, pred_img)
        lpips_val = compute_lpips(gt_rgb, pred_img)
        colorfulness_val = colorfulness_index(pred_img)

        rows.append(
            {
                "model": model_name,
                "psnr": float(psnr_val),
                "ssim": float(ssim_val),
                "lpips": float(lpips_val) if lpips_val is not None else np.nan,
                "colorfulness": float(colorfulness_val) if colorfulness_val is not None else np.nan,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate colorization models on the current image set")
    parser.add_argument("--max-images", type=int, default=500, help="Maximum number of images to evaluate")
    parser.add_argument(
        "--color-dir",
        type=Path,
        default=Path(__file__).parent.parent / "test_images" / "color",
        help="Directory containing color ground-truth images",
    )
    parser.add_argument(
        "--gray-dir",
        type=Path,
        default=Path(__file__).parent.parent / "test_images" / "gray",
        help="Directory containing grayscale input images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "validation_500",
        help="Directory to write validation results",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if not args.color_dir.exists():
        print(f"Error: color directory not found: {args.color_dir}")
        return

    image_paths = collect_image_paths(args.color_dir, args.max_images)
    if not image_paths:
        print("No images found to process.")
        return

    has_ddcolor = is_ddcolor_available()
    print(f"Found {len(image_paths)} images to evaluate.")
    print(f"Using grayscale directory: {args.gray_dir if args.gray_dir.exists() else 'generated from ground truth'}")
    print(f"DDColor available: {has_ddcolor}")

    csv_path = args.output_dir / "validation_results.csv"
    summary_path = args.output_dir / "validation_summary.csv"

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image", "model", "psnr", "ssim", "lpips", "colorfulness"],
        )
        writer.writeheader()

        for img_path in tqdm(image_paths, desc="Validating models"):
            gt_rgb = load_rgb_float(img_path)
            if gt_rgb is None:
                continue

            gray_path = args.gray_dir / img_path.name if args.gray_dir.exists() else None
            gray_rgb = resolve_gray_input(gt_rgb, gray_path)
            gray_uint8 = (rgb2gray(gray_rgb) * 255.0).astype(np.uint8)

            predictions: dict[str, np.ndarray | None] = {
                "eccv16": None,
                "siggraph17": None,
                "ddcolor": None,
                "fusion": None,
            }

            try:
                eccv_pred, siggraph_pred = colorize_highres(gray_rgb, strength=1.0, use_ddcolor=False)
                predictions["eccv16"] = eccv_pred.astype(np.float32)
                predictions["siggraph17"] = siggraph_pred.astype(np.float32)
            except Exception as exc:
                print(f"Warning: classic model inference failed for {img_path.name}: {exc}")

            ddcolor_pred: np.ndarray | None = None
            if has_ddcolor:
                try:
                    gray_bgr_uint8 = cv2.cvtColor((gray_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    dd_out_bgr = predict_ab_with_ddcolor(gray_bgr_uint8, input_size=512, model_size="large")
                    if dd_out_bgr is not None:
                        ddcolor_pred = cv2.cvtColor(dd_out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        predictions["ddcolor"] = ddcolor_pred
                except Exception as exc:
                    print(f"Warning: DDColor inference failed for {img_path.name}: {exc}")

            if predictions["eccv16"] is not None and predictions["siggraph17"] is not None:
                try:
                    fused_pred, _, _ = ensemble_colorization(
                        gray_uint8,
                        predictions["eccv16"],
                        predictions["siggraph17"],
                        ddcolor_result=ddcolor_pred,
                    )
                    predictions["fusion"] = fused_pred.astype(np.float32)
                except Exception as exc:
                    print(f"Warning: fusion failed for {img_path.name}: {exc}")

            rows = evaluate_predictions(gt_rgb, predictions)
            for row in rows:
                row["image"] = img_path.name
                writer.writerow(row)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No valid results were generated.")
        return

    summary = df.groupby("model")[["psnr", "ssim", "lpips", "colorfulness"]].mean(numeric_only=True)
    summary.to_csv(summary_path)

    print("\n--- Validation Summary (Mean Metrics) ---")
    print(summary)
    print(f"\nDetailed results saved to {csv_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

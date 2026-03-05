#!/usr/bin/env python3
"""
Validation Script: Compare Fusion, DDColor, ECCV16, Siggraph17 on First 500 Images

This script:
1. Selects the first 500 images (numerically sorted) from test_images/color.
2. Deletes the remaining images from test_images/color and test_images/gray.
3. Validates the 4 models on these 500 images.
4. Computes mean metrics: PSNR, SSIM, LPIPS, Colorfulness.
5. Saves results incrementally to outputs/validation_500/validation_results.csv.
"""

import sys
import os
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb, rgb2gray

from colorizeai.core.models import get_models
from colorizeai.core.ddcolor_model import predict_ab_with_ddcolor, is_ddcolor_available
from colorizeai.features.smart_model_fusion import ensemble_colorization
from colorizeai.utils.metrics import colorfulness_index, compute_metrics, compute_lpips

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_image(path):
    """Load image as RGB float [0, 1]"""
    path_str = str(path)
    if not os.path.exists(path_str):
        return None
    img = cv2.imread(path_str)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def _predict_ab_classic(model, img_rgb, device):
    """Deep copy of classic model prediction logic."""
    h, w = img_rgb.shape[:2]
    # Resize to 256x256 for model
    img_small = cv2.resize(img_rgb, (256, 256))
    img_lab = rgb2lab(img_small)
    img_l = img_lab[:, :, 0]
    
    tens = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        out_ab = model(tens)
        
    ab = out_ab.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab_up = cv2.resize(ab, (w, h))
    
    # Combine with original L (from original image to keep sharpness)
    img_lab_orig = rgb2lab(img_rgb)
    img_lab_out = np.zeros_like(img_lab_orig)
    img_lab_out[:, :, 0] = img_lab_orig[:, :, 0]
    img_lab_out[:, :, 1:] = ab_up
    
    return np.clip(lab2rgb(img_lab_out), 0, 1)

def cleanup_dataset(color_dir, gray_dir, limit=500):
    """
    Keeps only the first `limit` images (numerically sorted).
    Deletes the rest.
    """
    print(f"Cleaning up dataset to keep first {limit} images...")
    
    # Get all regular files
    color_files = [f for f in color_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    # Sort numerically
    def sort_key(f):
        try:
            return int(f.stem)
        except ValueError:
            return f.stem

    color_files.sort(key=sort_key)

    files_to_keep = color_files[:limit]
    files_to_delete = color_files[limit:]

    print(f"Found {len(color_files)} images. Keeping {len(files_to_keep)}, deleting {len(files_to_delete)}.")

    # Delete excess color files
    for f in files_to_delete:
        try:
            f.unlink()
        except Exception as e:
            print(f"Error deleting {f}: {e}")

    # Delete excess gray files (match by name)
    if gray_dir.exists():
        gray_files = [f for f in gray_dir.iterdir() if f.is_file()]
        names_to_keep = set(f.name for f in files_to_keep)
        
        deleted_gray_count = 0
        for f in gray_files:
            if f.name not in names_to_keep and f.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                try:
                    f.unlink()
                    deleted_gray_count += 1
                except Exception as e:
                    print(f"Error deleting gray file {f}: {e}")
        print(f"Deleted {deleted_gray_count} corresponding gray images.")
    
    return files_to_keep

def main():
    # --- Configuration ---
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    COLOR_DIR = PROJECT_ROOT / "test_images" / "color"
    GRAY_DIR = PROJECT_ROOT / "test_images" / "gray"
    OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "validation_500"
    ensure_dir(OUTPUT_ROOT)

    if not COLOR_DIR.exists():
        print(f"Error: {COLOR_DIR} not found.")
        return

    # --- Dataset Cleanup ---
    image_paths = cleanup_dataset(COLOR_DIR, GRAY_DIR, limit=500)
    
    if not image_paths:
        print("No images found to process.")
        return

    # --- Load Models ---
    print("Loading models...")
    try:
        (eccv16_model, siggraph17_model), device = get_models()
    except Exception as e:
        print(f"Error loading classic models: {e}")
        return

    has_ddcolor = is_ddcolor_available()
    print(f"DDColor available: {has_ddcolor}")

    # --- Prepare Output File ---
    csv_path = OUTPUT_ROOT / "validation_results.csv"
    
    # Define columns
    columns = ["image", "model", "psnr", "ssim", "lpips", "colorfulness"]
    
    # Write header
    with open(csv_path, 'w') as f:
        f.write(",".join(columns) + "\n")

    print(f"Starting validation on {len(image_paths)} images...")
    print(f"Results will be streamed to {csv_path}")

    pbar = tqdm(image_paths)
    for img_path in pbar:
        img_name = img_path.name
        
        # Load Ground Truth
        gt_rgb = load_image(img_path)
        if gt_rgb is None:
            continue
            
        # Create Input Grayscale (uint8 for Fusion input)
        gray_img_2d_float = rgb2gray(gt_rgb) # 0-1
        gray_img_uint8 = (gray_img_2d_float * 255).astype(np.uint8)
        
        # ----------------- Run Models -----------------
        predictions = {}
        
        # Store raw outputs for Fusion
        out_eccv16 = None
        out_siggraph17 = None
        out_ddcolor_rgb = None

        # 1. ECCV16
        try:
            out_eccv16 = _predict_ab_classic(eccv16_model, gt_rgb, device)
            predictions['eccv16'] = out_eccv16
        except Exception:
            pass

        # 2. SIGGRAPH17
        try:
            out_siggraph17 = _predict_ab_classic(siggraph17_model, gt_rgb, device)
            predictions['siggraph17'] = out_siggraph17
        except Exception:
            pass

        # 3. DDColor
        predictions['ddcolor'] = None
        if has_ddcolor:
            try:
                # DDColor expects BGR uint8
                gt_bgr_uint8 = (cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
                dd_out_bgr = predict_ab_with_ddcolor(gt_bgr_uint8)
                
                if dd_out_bgr is not None:
                    # Convert to RGB uint8 for Fusion
                    out_ddcolor_rgb = cv2.cvtColor(dd_out_bgr, cv2.COLOR_BGR2RGB)
                    # Convert to RGB float for Metrics
                    predictions['ddcolor'] = out_ddcolor_rgb.astype(np.float32) / 255.0
            except Exception:
                pass

        # 4. Fusion
        predictions['fusion'] = None
        # Only run fusion if we have the components
        if out_eccv16 is not None and out_siggraph17 is not None:
            try:
                fusion_res_rgb, _, _ = ensemble_colorization(
                    gray_img_uint8, 
                    out_eccv16, 
                    out_siggraph17, 
                    ddcolor_result=out_ddcolor_rgb # Passing RGB uint8
                )
                predictions['fusion'] = fusion_res_rgb
            except Exception as e:
                # print(f"Fusion fail: {e}")
                pass

        # ----------------- Compute Metrics & Write Row -----------------
        for model_name, pred_img in predictions.items():
            if pred_img is None:
                continue
                
            # Resize pred if needed
            if pred_img.shape != gt_rgb.shape:
                pred_img = cv2.resize(pred_img, (gt_rgb.shape[1], gt_rgb.shape[0]))

            psnr_val, ssim_val = compute_metrics(gt_rgb, pred_img)
            
            # LPIPS
            try:
                lpips_val = compute_lpips(gt_rgb, pred_img)
                if hasattr(lpips_val, 'item'):
                    lpips_val = lpips_val.item()
            except:
                lpips_val = ""
            
            # Colorfulness
            try:
                cf_val = colorfulness_index(pred_img)
            except:
                cf_val = ""

            # Write row immediately
            with open(csv_path, 'a') as f:
                vals = [
                    img_name,
                    model_name,
                    f"{psnr_val:.4f}",
                    f"{ssim_val:.4f}",
                    f"{lpips_val:.4f}" if isinstance(lpips_val, float) else str(lpips_val),
                    f"{cf_val:.4f}" if isinstance(cf_val, float) else str(cf_val)
                ]
                f.write(",".join(vals) + "\n")

    # --- Final Summary ---
    try:
        print("\nComputing summary statistics...")
        df = pd.read_csv(csv_path)
        if not df.empty:
            summary = df.groupby("model").mean(numeric_only=True)
            print("\n--- Validation Summary (Mean Metrics) ---")
            print(summary)
            
            summary_path = OUTPUT_ROOT / "validation_summary.csv"
            summary.to_csv(summary_path)
            print(f"Summary saved to {summary_path}")
    except Exception as e:
        print(f"Error computing summary: {e}")

if __name__ == "__main__":
    main()

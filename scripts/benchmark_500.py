#!/usr/bin/env python3
"""
Validation Script: Filter 500 images & Benchmark 4 models.

Usage:
    python scripts/benchmark_500.py
"""

import sys
import shutil
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import warnings

# Import project modules
from colorizeai.core.models import get_models
from colorizeai.core.ddcolor_model import predict_ab_with_ddcolor, is_ddcolor_available
from colorizeai.features.smart_model_fusion import ensemble_colorization
from colorizeai.utils.metrics import colorfulness_index, compute_metrics, compute_lpips

# Filter warnings
warnings.filterwarnings("ignore")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_image_rgb(path):
    """Load image as RGB float [0, 1]"""
    p = str(path)
    if not os.path.exists(p):
        return None
    img = cv2.imread(p)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    return img

def cleanup_and_select_images(limit=500):
    """
    Keep first `limit` images in test_images/color.
    Delete the rest.
    Sync test_images/gray.
    """
    root = Path(__file__).parent.parent
    color_dir = root / "test_images" / "color"
    gray_dir = root / "test_images" / "gray"
    
    if not color_dir.exists():
        print(f"Directory not found: {color_dir}")
        return []

    # Gather all images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = [f for f in color_dir.iterdir() if f.suffix.lower() in exts]
    
    # Sort numerically by filename
    def sort_key(f):
        try:
            return int(f.stem)
        except ValueError:
            return f.stem
    
    all_files.sort(key=sort_key)
    
    keep_files = all_files[:limit]
    delete_files = all_files[limit:]
    
    print(f"Total images found: {len(all_files)}")
    print(f"Keeping: {len(keep_files)}")
    print(f"Deleting: {len(delete_files)}")
    
    # Delete excess color images
    for f in delete_files:
        try:
            f.unlink()
        except OSError as e:
            print(f"Error deleting {f}: {e}")
            
    # Sync gray directory if it exists
    if gray_dir.exists():
        gray_files = [f for f in gray_dir.iterdir() if f.suffix.lower() in exts]
        keep_names = {f.name for f in keep_files}
        deleted_gray = 0
        for f in gray_files:
            if f.name not in keep_names:
                try:
                    f.unlink()
                    deleted_gray += 1
                except OSError as e:
                    print(f"Error deleting {f}: {e}")
        print(f"Deleted {deleted_gray} gray images to match.")
        
    return keep_files

def run_classic_model(model, img_rgb, device):
    """Run ECCV16 or SIGGRAPH17"""
    h, w = img_rgb.shape[:2]
    
    # Resize to 256x256 for model input usually
    img_small = cv2.resize(img_rgb, (256, 256))
    img_lab = rgb2lab(img_small)
    img_l = img_lab[:, :, 0]
    
    tens = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        out_ab = model(tens)
        
    ab = out_ab.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab_up = cv2.resize(ab, (w, h))
    
    # Combine with original L
    img_lab_orig = rgb2lab(img_rgb)
    img_lab_out = np.zeros_like(img_lab_orig)
    img_lab_out[:, :, 0] = img_lab_orig[:, :, 0]
    img_lab_out[:, :, 1:] = ab_up
    
    return np.clip(lab2rgb(img_lab_out), 0, 1)

def main():
    # Setup
    output_dir = Path("outputs/validation_500")
    ensure_dir(output_dir)
    
    print("--- 1. Cleaning Dataset ---")
    image_list = cleanup_and_select_images(limit=500)
    if not image_list:
        print("No images to process.")
        return

    print("\n--- 2. Loading Models ---")
    try:
        models_classic, device = get_models()
        if models_classic:
            eccv16_model, siggraph17_model = models_classic
        else:
            print("Failed to load classic models.")
            return
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    has_ddcolor = is_ddcolor_available()
    print(f"DDColor Available: {has_ddcolor}")
    
    metrics_data = []
    
    print("\n--- 3. Running Validation ---")
    for img_path in tqdm(image_list):
        img_name = img_path.name
        
        # Ground Truth
        gt_rgb = load_image_rgb(img_path)
        if gt_rgb is None:
            continue
            
        # Prepare inputs
        # Grayscale for fusion engine (0-255 uint8)
        gray_img_u8 = (rgb2gray(gt_rgb) * 255).astype(np.uint8)
        
        # --- Run Models ---
        results = {}
        
        # ECCV16
        try:
            results['eccv16'] = run_classic_model(eccv16_model, gt_rgb, device)
        except Exception:
            results['eccv16'] = None
            
        # SIGGRAPH17
        try:
            results['siggraph17'] = run_classic_model(siggraph17_model, gt_rgb, device)
        except Exception:
            results['siggraph17'] = None
            
        # DDColor
        if has_ddcolor:
            try:
                # DDColor expects BGR input (float or uint8)
                # Convert GT RGB to BGR
                gt_bgr_u8 = (gt_rgb[:, :, ::-1] * 255).astype(np.uint8)
                dd_out_bgr = predict_ab_with_ddcolor(gt_bgr_u8, input_size=512)
                if dd_out_bgr is not None:
                    # Convert BGR uint8 -> RGB float
                    results['ddcolor'] = cv2.cvtColor(dd_out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                else:
                    results['ddcolor'] = None
            except Exception as e:
                # print(f"DDColor Error: {e}")
                results['ddcolor'] = None
        else:
             results['ddcolor'] = None
        
        # Smart Fusion
        # Needs eccv16, siggraph17
        if results.get('eccv16') is not None and results.get('siggraph17') is not None:
            try:
                fusion_out, _, _ = ensemble_colorization(
                    gray_img_u8,
                    results['eccv16'],
                    results['siggraph17'],
                    ddcolor_result=results.get('ddcolor')
                )
                results['fusion'] = fusion_out
            except Exception as e:
                # print(f"Fusion Error: {e}")
                results['fusion'] = None
        else:
            results['fusion'] = None
            
        # --- Compute Metrics ---
        # Base names: eccv16, siggraph17, ddcolor, fusion
        for model_name, pred_rgb in results.items():
            if pred_rgb is None:
                continue
                
            # PSNR, SSIM
            psnr, ssim = compute_metrics(gt_rgb, pred_rgb)
            
            # LPIPS
            lpips_val = compute_lpips(gt_rgb, pred_rgb)
            
            # Colorfulness (of prediction)
            cf = colorfulness_index(pred_rgb)
            
            metrics_data.append({
                "Image": img_name,
                "Model": model_name,
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": lpips_val if lpips_val is not None else np.nan,
                "Colorfulness": cf if cf is not None else np.nan
            })

    # --- Save Results ---
    if not metrics_data:
        print("No results generated.")
        return

    df = pd.DataFrame(metrics_data)
    csv_path = output_dir / "validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Summary
    print("\n--- Summary ---")
    summary = df.groupby("Model")[["PSNR", "SSIM", "LPIPS", "Colorfulness"]].mean()
    print(summary)
    summary.to_csv(output_dir / "summary_metrics.csv")

if __name__ == "__main__":
    main()

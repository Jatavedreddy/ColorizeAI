#!/usr/bin/env python3
"""
Benchmark Script: Smart Model Fusion vs. Base Models

This script processes images to compare:
1. DDColor (Base SOTA)
2. SIGGRAPH17 (Classic)
3. Smart Fusion (Our Hybrid Approach)

It generates:
- Visual comparisons (side-by-side grids)
- Quantitative metrics in a CSV
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from colorizeai.core.models import get_models
from colorizeai.core.ddcolor_model import predict_ab_with_ddcolor, is_ddcolor_available
from colorizeai.features.smart_model_fusion import ensemble_colorization
from colorizeai.utils.metrics import colorfulness_index, compute_metrics

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_image(path):
    """Load image as RGB float [0, 1]"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def _predict_ab_classic(model, img_rgb, device):
    """Refactored helper to get result from a classic model."""
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
    
    # Combine with original L
    img_lab_orig = rgb2lab(img_rgb)
    img_lab_out = np.zeros_like(img_lab_orig)
    img_lab_out[:, :, 0] = img_lab_orig[:, :, 0]
    img_lab_out[:, :, 1:] = ab_up
    
    return np.clip(lab2rgb(img_lab_out), 0, 1)

def main():
    # --- Configuration ---
    INPUT_DIR = Path("assets/sample_images/performance evaluation") # GT Images
    OUTPUT_ROOT = Path("outputs/benchmark_fusion")
    ensure_dir(OUTPUT_ROOT)
    
    # Subdirs for results
    dirs = {
        "ddcolor": OUTPUT_ROOT / "ddcolor",
        "siggraph17": OUTPUT_ROOT / "siggraph17",
        "fusion": OUTPUT_ROOT / "smart_fusion",
        "comparison": OUTPUT_ROOT / "grids"
    }
    for d in dirs.values():
        ensure_dir(d)
        
    # --- Load Models ---
    print("Loading models...")
    (eccv16_model, siggraph17_model), device = get_models()
    has_ddcolor = is_ddcolor_available()
    print(f"DDColor available: {has_ddcolor}")

    # --- Processing ---
    image_files = sorted(list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png")))
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    results_data = []

    print(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        img_name = img_path.name
        img_gt = load_image(img_path)
        if img_gt is None: 
            continue
            
        # Simulate Grayscale Input
        img_gray = cv2.cvtColor((img_gt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img_gray_3ch = np.stack([img_gray]*3, axis=-1) / 255.0 # For shape compatibility logic if needed
        
        # 1. Run SIGGRAPH17 (Classic Baseline)
        pred_sig = _predict_ab_classic(siggraph17_model, img_gt, device)
        
        # 2. Run ECCV16 (Needed for fusion primarily)
        pred_eccv = _predict_ab_classic(eccv16_model, img_gt, device)
        
        # 3. Run DDColor (SOTA)
        pred_dd = None
        if has_ddcolor:
            img_bgr = cv2.cvtColor((img_gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            dd_out_bgr = predict_ab_with_ddcolor(img_bgr, input_size=512)
            if dd_out_bgr is not None:
                pred_dd = cv2.cvtColor(dd_out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        if pred_dd is None:
            # Fallback if DDColor fails or not installed, use SIGGRAPH as "best" proxy for logic
            pred_dd = pred_sig.copy() 

        # 4. Run Smart Fusion
        # Note: ensemble_colorization expects uint8 grayscale image for analysis
        pred_fusion, weights, chars = ensemble_colorization(
            img_gray, 
            pred_eccv, 
            pred_sig, 
            ddcolor_result=pred_dd
        )
        
        # --- Metrics ---
        # Calculate PSNR against GT
        psnr_dd = compute_psnr(img_gt, pred_dd, data_range=1.0)
        psnr_sig = compute_psnr(img_gt, pred_sig, data_range=1.0)
        psnr_fusion = compute_psnr(img_gt, pred_fusion, data_range=1.0)
        
        # Colorfulness (Higher isn't always better, but good to track)
        cf_gt = colorfulness_index(img_gt)
        cf_fusion = colorfulness_index(pred_fusion)
        
        results_data.append({
            "Image": img_name,
            "PSNR_DDColor": psnr_dd,
            "PSNR_Siggraph": psnr_sig,
            "PSNR_Fusion": psnr_fusion,
            "Weight_DD": weights.get('ddcolor', 0),
            "Weight_Sig": weights.get('siggraph17', 0),
            "Texture": chars.get('texture_complexity', 0),
            "Contrast": chars.get('contrast', 0)
        })

        # --- Save Images ---
        def save_rgb(path, img):
            cv2.imwrite(str(path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
        save_rgb(dirs["ddcolor"] / img_name, pred_dd)
        save_rgb(dirs["siggraph17"] / img_name, pred_sig)
        save_rgb(dirs["fusion"] / img_name, pred_fusion)
        
        # Generate Grid: [GT | DDColor | SIGGRAPH | Fusion]
        # Labeling images
        h, w, c = img_gt.shape
        def add_label(img, text):
            # Create a copy to draw on
            out = (img * 255).astype(np.uint8).copy()
            # Add white background strip for text
            cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return out

        grid_row = np.hstack([
            add_label(img_gt, "Ground Truth"),
            add_label(pred_dd, f"DDColor ({psnr_dd:.1f} dB)"),
            add_label(pred_sig, f"SIGAPP17 ({psnr_sig:.1f} dB)"),
            add_label(pred_fusion, f"Smart Fusion ({psnr_fusion:.1f} dB)")
        ])
        
        cv2.imwrite(str(dirs["comparison"] / f"grid_{img_name}"), cv2.cvtColor(grid_row, cv2.COLOR_RGB2BGR))

    # --- Save CSV ---
    df = pd.DataFrame(results_data)
    csv_path = OUTPUT_ROOT / "fusion_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark Complete. Results saved to {csv_path}")
    print(f"Visual grids saved to {dirs['comparison']}")
    
    # Calculate Wins
    wins_fusion = sum(df['PSNR_Fusion'] > df['PSNR_DDColor'])
    print(f"\nSmart Fusion beat pure DDColor in {wins_fusion}/{len(df)} cases (PSNR metric)")

if __name__ == "__main__":
    main()

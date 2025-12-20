import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colorizeai.utils.metrics import (
    compute_metrics,
    compute_ciede2000,
    compute_lpips,
    colorfulness_index,
    compute_ab_mse
)
from colorizeai.utils.fid import (
    INCEPTION_V3_FID,
    get_activations,
    calculate_activation_statistics,
    calculate_frechet_distance
)

def load_image(path):
    """Load image as RGB float [0, 1]"""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def compute_fid_score(gt_paths, pred_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Compute FID score between two sets of images"""
    print("Computing FID score...")
    
    # Load images into tensors
    def load_batch(paths):
        tensors = []
        for p in paths:
            img = load_image(p)
            # Convert to tensor (C, H, W)
            t = torch.from_numpy(img.transpose(2, 0, 1)).float()
            tensors.append(t)
        return tensors

    # Initialize Inception model
    block_idx = INCEPTION_V3_FID.DEFAULT_BLOCK_INDEX
    model = INCEPTION_V3_FID(output_blocks=[block_idx])
    
    # Get activations
    gt_tensors = load_batch(gt_paths)
    pred_tensors = load_batch(pred_paths)
    
    mu1, sigma1 = calculate_activation_statistics(
        get_activations(gt_tensors, model, device=device)
    )
    mu2, sigma2 = calculate_activation_statistics(
        get_activations(pred_tensors, model, device=device)
    )
    
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value

def main():
    parser = argparse.ArgumentParser(description="Evaluate colorization metrics")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth images")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted images")
    parser.add_argument("--output", type=str, default="metrics_report.csv", help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    
    if not gt_dir.exists():
        print(f"Error: GT directory {gt_dir} does not exist")
        return
    if not pred_dir.exists():
        print(f"Error: Pred directory {pred_dir} does not exist")
        return
        
    pred_files = sorted([f for f in pred_dir.glob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    results = []
    gt_paths = []
    pred_paths = []
    
    print(f"Found {len(pred_files)} images to evaluate")
    
    for pred_file in tqdm(pred_files):
        # Assume same filename in GT dir
        gt_file = gt_dir / pred_file.name
        
        if not gt_file.exists():
            # Try matching without extension or different extension if needed
            # For now strict matching
            print(f"Warning: GT file for {pred_file.name} not found, skipping")
            continue
            
        try:
            gt_img = load_image(gt_file)
            pred_img = load_image(pred_file)
            
            # Resize GT to match Pred if needed (or vice versa)
            if gt_img.shape != pred_img.shape:
                gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]))
            
            psnr, ssim = compute_metrics(gt_img, pred_img)
            lpips_val = compute_lpips(gt_img, pred_img)
            ciede = compute_ciede2000(gt_img, pred_img)
            cf_pred = colorfulness_index(pred_img)
            cf_gt = colorfulness_index(gt_img)
            
            results.append({
                "filename": pred_file.name,
                "PSNR": psnr,
                "SSIM": ssim,
                "LPIPS": lpips_val,
                "CIEDE2000": ciede,
                "Colorfulness_Pred": cf_pred,
                "Colorfulness_GT": cf_gt
            })
            
            gt_paths.append(gt_file)
            pred_paths.append(pred_file)
            
        except Exception as e:
            print(f"Error processing {pred_file.name}: {e}")
            
    if not results:
        print("No results generated")
        return
        
    df = pd.DataFrame(results)
    
    # Compute FID
    try:
        fid_score = compute_fid_score(gt_paths, pred_paths, device=args.device)
        print(f"FID Score: {fid_score}")
    except Exception as e:
        print(f"Error computing FID: {e}")
        fid_score = None
        
    # Summary
    summary = df.mean(numeric_only=True)
    print("\n=== Performance Summary ===")
    print(f"PSNR: {summary['PSNR']:.4f}")
    print(f"SSIM: {summary['SSIM']:.4f}")
    print(f"LPIPS: {summary['LPIPS']:.4f}")
    print(f"CIEDE2000: {summary['CIEDE2000']:.4f}")
    print(f"Colorfulness (Pred): {summary['Colorfulness_Pred']:.4f}")
    print(f"Colorfulness (GT): {summary['Colorfulness_GT']:.4f}")
    if fid_score is not None:
        print(f"FID: {fid_score:.4f}")
        
    # Save detailed results
    df.to_csv(args.output, index=False)
    print(f"\nDetailed results saved to {args.output}")
    
    # Save summary
    summary_file = Path(args.output).with_name("metrics_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=== Performance Summary ===\n")
        f.write(f"PSNR: {summary['PSNR']:.4f}\n")
        f.write(f"SSIM: {summary['SSIM']:.4f}\n")
        f.write(f"LPIPS: {summary['LPIPS']:.4f}\n")
        f.write(f"CIEDE2000: {summary['CIEDE2000']:.4f}\n")
        f.write(f"Colorfulness (Pred): {summary['Colorfulness_Pred']:.4f}\n")
        f.write(f"Colorfulness (GT): {summary['Colorfulness_GT']:.4f}\n")
        if fid_score is not None:
            f.write(f"FID: {fid_score:.4f}\n")

if __name__ == "__main__":
    main()

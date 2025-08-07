"""
Metric calculation utilities
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Returns (PSNR, SSIM) for images in RGB float [0,1]."""
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
    return psnr_val, ssim_val

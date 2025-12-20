"""Metric calculation utilities.

All functions here are defensive: if an optional dependency is missing, they
return ``None`` instead of raising, allowing the UI to degrade gracefully.

Image format expectations:
- All inputs are RGB float arrays in range [0,1] unless otherwise noted.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, Sequence
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color as skcolor

# Optional deps guarded
try:  # LPIPS
    import lpips  # type: ignore
    _lpips_model = lpips.LPIPS(net="alex")  # lazy init once
except Exception:  # pragma: no cover
    _lpips_model = None

try:  # torch-fidelity for FID/KID (heavy; single-image FID is meaningless, we skip if <2 images)
    from torch_fidelity import calculate_metrics as torch_fid_calculate  # type: ignore
except Exception:  # pragma: no cover
    torch_fid_calculate = None

try:  # psutil for memory capture
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _to_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    return arr


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Returns (PSNR, SSIM) for RGB float images in [0,1]."""
    gt = _to_3d(gt)
    pred = _to_3d(pred)
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
    return psnr_val, ssim_val


def compute_ciede2000(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    """Mean CIEDE2000 ΔE between two RGB images in [0,1]. Lower is better.
    Returns None if computation fails.
    """
    try:
        gt_lab = skcolor.rgb2lab(_to_3d(gt))
        pred_lab = skcolor.rgb2lab(_to_3d(pred))
        
        # Try using skimage's implementation if available (v0.19+)
        if hasattr(skcolor, 'deltaE_ciede2000'):
            de = skcolor.deltaE_ciede2000(gt_lab, pred_lab)
            return float(de.mean())
            
        # Fallback to Euclidean in Lab (ΔE76)
        L1, a1, b1 = gt_lab[..., 0], gt_lab[..., 1], gt_lab[..., 2]
        L2, a2, b2 = pred_lab[..., 0], pred_lab[..., 1], pred_lab[..., 2]
        de = np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
        return float(de.mean())
    except Exception:
        return None


def compute_ab_mse(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    """Mean squared error only on chroma (a,b) channels in Lab space."""
    try:
        gt_lab = skcolor.rgb2lab(_to_3d(gt))
        pred_lab = skcolor.rgb2lab(_to_3d(pred))
        mse = ((gt_lab[..., 1] - pred_lab[..., 1]) ** 2 + (gt_lab[..., 2] - pred_lab[..., 2]) ** 2).mean()
        return float(mse)
    except Exception:
        return None


def compute_lpips(gt: np.ndarray, pred: np.ndarray) -> Optional[float]:
    """LPIPS distance (lower better). Returns None if lpips not installed."""
    if _lpips_model is None:
        return None
    try:
        import torch
        # Convert to tensor in [-1,1]
        def prep(x: np.ndarray) -> torch.Tensor:
            x = _to_3d(x)
            t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)
            t = t * 2.0 - 1.0
            return t
        with torch.no_grad():
            d = _lpips_model(prep(gt), prep(pred)).item()
        return float(d)
    except Exception:
        return None


def compute_fid(samples_a: Sequence[np.ndarray], samples_b: Sequence[np.ndarray]) -> Optional[float]:
    """Approximate FID between two sets of images. Returns None if unavailable or too few samples."""
    if torch_fid_calculate is None or len(samples_a) < 2 or len(samples_b) < 2:
        return None
    try:
        # torch-fidelity expects paths or tensors; quickest path: create temp dirs is heavy.
        # For now return None to avoid overhead unless enough samples (stub placeholder).
        return None  # Placeholder – full FID requires directory serialization.
    except Exception:
        return None


def compute_memory_usage_mb() -> Optional[float]:
    """Returns current RSS memory in MB (psutil required)."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def compute_flicker_variance(frames: Sequence[np.ndarray]) -> Optional[float]:
    """Simple temporal flicker metric: variance of chroma over time per pixel averaged.
    Returns None if <2 frames.
    """
    if len(frames) < 2:
        return None
    try:
        labs = [skcolor.rgb2lab(_to_3d(f)) for f in frames]
        ab_stack = np.stack([lab[..., 1:3] for lab in labs], axis=0)  # (T,H,W,2)
        var = ab_stack.var(axis=0).mean()
        return float(var)
    except Exception:
        return None


def summarize_image_pair(gt: np.ndarray, pred: np.ndarray, extra: bool = True) -> dict:
    """Convenience wrapper returning all implemented metrics for a pair.
    If *extra* is False only PSNR/SSIM are returned.
    """
    psnr, ssim = compute_metrics(gt, pred)
    summary = {"psnr": psnr, "ssim": ssim}
    if extra:
        summary.update({
            "deltaE": compute_ciede2000(gt, pred),
            "ab_mse": compute_ab_mse(gt, pred),
            "lpips": compute_lpips(gt, pred)
        })
    return summary


def format_metric(value: Optional[float], fmt: str = ":.4f", na: str = "—") -> str:
    """Format a metric value with fallback for None."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return na
    try:
        return f"{value:{fmt}}"
    except Exception:
        return f"{value}"


# -------------------- Diversity & Palette Metrics --------------------

def colorfulness_index(rgb: np.ndarray) -> Optional[float]:
    """Hasler–Süsstrunk colorfulness index. Higher is richer color.
    Returns None on error.
    """
    try:
        rgb = _to_3d(rgb)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        std_rg, std_yb = rg.std(), yb.std()
        mean_rg, mean_yb = rg.mean(), yb.mean()
        # Common formulation
        return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))
    except Exception:
        return None


def chroma_entropy(rgb: np.ndarray, bins: int = 32) -> Optional[float]:
    """Entropy of chroma distribution in Lab ab-space. Higher suggests broader palette.
    Returns None on error.
    """
    try:
        lab = skcolor.rgb2lab(_to_3d(rgb))
        ab = lab[..., 1:3].reshape(-1, 2)
        # Range of a,b roughly [-128, 127]
        hist, _, _ = np.histogram2d(ab[:, 0], ab[:, 1], bins=bins, range=[[-128, 127], [-128, 127]])
        p = hist.ravel().astype(np.float64)
        p = p / (p.sum() + 1e-12)
        ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
        return float(ent)
    except Exception:
        return None


def palette_distance_kmeans(gt: np.ndarray, pred: np.ndarray, k: int = 8, random_state: int = 0) -> Optional[float]:
    """Distance between palettes by k-means on ab channels with Hungarian matching.
    Lower is better. Returns None if sklearn/scipy not available.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return None
    try:
        gt_lab = skcolor.rgb2lab(_to_3d(gt))
        pr_lab = skcolor.rgb2lab(_to_3d(pred))
        gt_ab = gt_lab[..., 1:3].reshape(-1, 2)
        pr_ab = pr_lab[..., 1:3].reshape(-1, 2)

        gt_km = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(gt_ab)
        pr_km = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(pr_ab)

        C1, w1 = gt_km.cluster_centers_, np.bincount(gt_km.labels_, minlength=k).astype(np.float64)
        C2, w2 = pr_km.cluster_centers_, np.bincount(pr_km.labels_, minlength=k).astype(np.float64)
        w1 /= (w1.sum() + 1e-12)
        w2 /= (w2.sum() + 1e-12)

        # Cost: Euclidean distances between centers (Lab ab space)
        cost = np.linalg.norm(C1[:, None, :] - C2[None, :, :], axis=-1)
        r_ind, c_ind = linear_sum_assignment(cost)

        # Weighted average distance using min of weights to be conservative
        dist = 0.0
        for i, j in zip(r_ind, c_ind):
            dist += cost[i, j] * 0.5 * (w1[i] + w2[j])
        return float(dist)
    except Exception:
        return None


def composite_score(metrics: dict, weights: Optional[dict] = None) -> Optional[float]:
    """Composite score combining distortion, perceptual, and diversity terms.
    Higher is better. All keys optional; missing values are ignored.

    Default weights: {psnr:0.2, ssim:0.2, inv_lpips:0.2, inv_deltaE:0.2, colorfulness:0.1, entropy:0.1}
    Normalization: psnr/40 (clamped 0-1), ssim in [0,1], inv_lpips=1-lpips, inv_deltaE=1-min(ΔE/20,1).
    """
    try:
        w = {
            "psnr": 0.2,
            "ssim": 0.2,
            "inv_lpips": 0.2,
            "inv_deltaE": 0.2,
            "colorfulness": 0.1,
            "entropy": 0.1,
        }
        if weights:
            w.update(weights)

        # Normalize pieces
        parts = []
        def clamp01(x):
            return max(0.0, min(1.0, x))

        if (v := metrics.get("psnr")) is not None:
            parts.append(w["psnr"] * clamp01(float(v) / 40.0))
        if (v := metrics.get("ssim")) is not None:
            parts.append(w["ssim"] * clamp01(float(v)))
        if (v := metrics.get("lpips")) is not None:
            parts.append(w["inv_lpips"] * clamp01(1.0 - float(v)))
        if (v := metrics.get("deltaE")) is not None:
            parts.append(w["inv_deltaE"] * clamp01(1.0 - min(float(v) / 20.0, 1.0)))
        if (v := metrics.get("colorfulness")) is not None:
            # Rough scale: colorfulness ~ [0,>1]; clamp modestly
            parts.append(w["colorfulness"] * clamp01(float(v)))
        if (v := metrics.get("entropy")) is not None:
            # Entropy max ~ log2(bins^2) for chroma_entropy with bins=32 => ~10
            parts.append(w["entropy"] * clamp01(float(v) / 10.0))

        if not parts:
            return None
        return float(sum(parts))
    except Exception:
        return None


__all__ = [
    "compute_metrics",
    "compute_ciede2000",
    "compute_ab_mse",
    "compute_lpips",
    "compute_fid",
    "compute_memory_usage_mb",
    "compute_flicker_variance",
    "summarize_image_pair",
    "format_metric",
    # diversity & palette
    "colorfulness_index",
    "chroma_entropy",
    "palette_distance_kmeans",
    "composite_score",
]

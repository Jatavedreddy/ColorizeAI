"""
Interactive Color Hints Feature
Allows users to provide color hints by scribbling on the image
"""

import numpy as np
import cv2
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import felzenszwalb

def apply_color_hints(
    grayscale_img: np.ndarray,
    base_colorized: np.ndarray,
    color_hints: list,  # List of (x, y, r, g, b) tuples
    hint_radius: int = 20,
    propagation_strength: float = 0.7
) -> np.ndarray:
    """
    Apply user-provided color hints to guide colorization
    
    Args:
        grayscale_img: Original grayscale image
        base_colorized: Base colorization from models
        color_hints: List of color hints as (x, y, r, g, b) tuples
        hint_radius: Radius around each hint point to apply color
        propagation_strength: How strongly to propagate hint colors
    
    Returns:
        Colorized image with applied hints
    """
    if not color_hints:
        return base_colorized
    
    if grayscale_img.ndim == 2:
        guide_source = np.stack((grayscale_img,) * 3, axis=-1)
    elif grayscale_img.ndim == 3 and grayscale_img.shape[2] == 1:
        guide_source = np.concatenate((grayscale_img,) * 3, axis=-1)
    else:
        guide_source = grayscale_img

    h, w = grayscale_img.shape[:2]
    result = base_colorized.copy()

    if guide_source.dtype in (np.float32, np.float64):
        guide_rgb = np.clip(guide_source, 0, 1)
    else:
        guide_rgb = guide_source.astype(np.float64) / 255.0

    if result.dtype != np.float64:
        result = np.clip(result.astype(np.float64), 0, 1)
    
    # Convert to LAB for better color blending
    gray_lab = rgb2lab(guide_rgb)
    base_lab = rgb2lab(result)
    
    # Segment by luminance to constrain propagation to object boundaries
    luminance_01 = gray_lab[:, :, 0] / 100.0
    segments = felzenszwalb(luminance_01, scale=120, sigma=0.8, min_size=80)

    y_coords, x_coords = np.ogrid[:h, :w]
    
    for x, y, r, g, b in color_hints:
        # Ensure coordinates are within bounds
        x, y = max(0, min(w-1, int(x))), max(0, min(h-1, int(y)))
        
        # Convert hint color to LAB
        hint_rgb = np.array([[[r/255.0, g/255.0, b/255.0]]])
        hint_lab = rgb2lab(hint_rgb)[0, 0]
        
        # Get the segment that contains this hint
        hint_segment = segments[y, x]
        segment_mask = (segments == hint_segment)
        if not np.any(segment_mask):
            continue
        
        # Build a smooth propagation field inside the selected segment.
        # Using only a hard radius creates visible dots in final output.
        hint_luminance = gray_lab[y, x, 0]
        luminance_diff = np.abs(gray_lab[:, :, 0] - hint_luminance)
        segment_diffs = luminance_diff[segment_mask]
        adaptive_sigma = max(8.0, float(np.percentile(segment_diffs, 75)))
        luminance_weights = np.exp(-((luminance_diff ** 2) / (2.0 * adaptive_sigma ** 2)))

        distance_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
        spatial_sigma = max(12.0, float(hint_radius))
        spatial_weights = np.exp(-(distance_sq / (2.0 * spatial_sigma ** 2)))

        combined_weights = (0.8 * luminance_weights + 0.2 * spatial_weights) * segment_mask.astype(np.float64)
        max_weight = float(np.max(combined_weights))
        if max_weight > 0:
            combined_weights /= max_weight

        # Keep a small minimum blend in the segment to avoid single-pixel artifacts.
        combined_weights = np.where(segment_mask, 0.15 + 0.85 * combined_weights, 0.0)
        combined_weights = cv2.GaussianBlur(combined_weights.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
        combined_weights = np.where(segment_mask, combined_weights, 0.0)
        blend_weights = np.clip(propagation_strength * combined_weights, 0.0, 1.0)
        
        # Apply the hint color with varying strength
        base_lab[:, :, 1] = (1 - blend_weights) * base_lab[:, :, 1] + blend_weights * hint_lab[1]
        base_lab[:, :, 2] = (1 - blend_weights) * base_lab[:, :, 2] + blend_weights * hint_lab[2]
    
    # Convert back to RGB
    result_rgb = lab2rgb(base_lab)
    return np.clip(result_rgb, 0, 1)

def smart_color_propagation(
    grayscale_img: np.ndarray,
    color_hints: list,
    base_colorized: np.ndarray
) -> np.ndarray:
    """
    Intelligent color propagation using image structure
    """
    if not color_hints:
        return base_colorized
    
    # Use edge-preserving filtering for better color propagation
    gray_float = grayscale_img.astype(np.float32) / 255.0
    
    # Create a guide image for propagation
    guide = cv2.bilateralFilter(gray_float, 9, 75, 75)
    
    # Apply hints with structure-aware propagation
    result = apply_color_hints(grayscale_img, base_colorized, color_hints)
    
    return result

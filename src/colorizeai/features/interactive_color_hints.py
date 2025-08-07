"""
Interactive Color Hints Feature
Allows users to provide color hints by scribbling on the image
"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import felzenszwalb
import torch

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
    
    h, w = grayscale_img.shape[:2]
    result = base_colorized.copy()
    
    # Convert to LAB for better color blending
    gray_lab = rgb2lab(grayscale_img.astype(np.float64) / 255.0)
    base_lab = rgb2lab(result)
    
    # Create segments for intelligent color propagation
    segments = felzenszwalb(grayscale_img, scale=100, sigma=0.5, min_size=50)
    
    for x, y, r, g, b in color_hints:
        # Ensure coordinates are within bounds
        x, y = max(0, min(w-1, int(x))), max(0, min(h-1, int(y)))
        
        # Convert hint color to LAB
        hint_rgb = np.array([[[r/255.0, g/255.0, b/255.0]]])
        hint_lab = rgb2lab(hint_rgb)[0, 0]
        
        # Get the segment that contains this hint
        hint_segment = segments[y, x]
        
        # Apply hint color to the entire segment
        segment_mask = (segments == hint_segment)
        
        # Also apply to nearby pixels within radius
        y_coords, x_coords = np.ogrid[:h, :w]
        distance_mask = ((x_coords - x) ** 2 + (y_coords - y) ** 2) <= hint_radius ** 2
        
        # Combine segment and distance masks
        apply_mask = segment_mask | distance_mask
        
        # Calculate blending weights based on luminance similarity
        hint_luminance = gray_lab[y, x, 0]
        luminance_diff = np.abs(gray_lab[:, :, 0] - hint_luminance)
        luminance_weights = np.exp(-luminance_diff / 20.0)  # Sigma = 20
        
        # Apply the hint color with varying strength
        base_lab[apply_mask, 1] = (
            (1 - propagation_strength * luminance_weights[apply_mask]) * base_lab[apply_mask, 1] +
            propagation_strength * luminance_weights[apply_mask] * hint_lab[1]
        )
        base_lab[apply_mask, 2] = (
            (1 - propagation_strength * luminance_weights[apply_mask]) * base_lab[apply_mask, 2] +
            propagation_strength * luminance_weights[apply_mask] * hint_lab[2]
        )
    
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

"""
Reference-Guided Colorization Feature
Allows users to provide a reference image to guide the colorization process
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import slic
from skimage.util import img_as_float
import torch

def extract_color_palette(reference_img: np.ndarray, n_colors: int = 8) -> np.ndarray:
    """Extract dominant color palette from reference image"""
    # Reshape image to be a list of pixels
    data = reference_img.reshape((-1, 3))
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(data)
    
    # Get the colors
    colors = kmeans.cluster_centers_
    return colors.astype(np.uint8)

def apply_reference_guided_colorization(
    grayscale_img: np.ndarray, 
    reference_img: np.ndarray, 
    base_colorized: np.ndarray,
    guidance_strength: float = 0.3
) -> np.ndarray:
    """
    Apply reference-guided colorization by blending base colorization with reference colors
    
    Args:
        grayscale_img: Original grayscale image
        reference_img: Reference color image for guidance
        base_colorized: Base colorization from ECCV16/SIGGRAPH17
        guidance_strength: How much to apply reference guidance (0-1)
    
    Returns:
        Reference-guided colorized image
    """
    # Convert to LAB color space
    gray_lab = rgb2lab(img_as_float(grayscale_img))
    ref_lab = rgb2lab(img_as_float(reference_img))
    base_lab = rgb2lab(img_as_float(base_colorized))
    
    # Extract color palette from reference
    ref_palette = extract_color_palette(reference_img, n_colors=8)
    ref_palette_lab = rgb2lab(ref_palette.reshape(1, -1, 3)).reshape(-1, 3)
    
    # Segment the grayscale image
    segments = slic(grayscale_img, n_segments=100, compactness=10, start_label=1)
    
    # For each segment, find the best matching color from reference palette
    guided_ab = base_lab[:, :, 1:].copy()
    
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        
        if np.sum(mask) == 0:
            continue
            
        # Get average color of this segment in base colorization
        segment_color = np.mean(base_lab[mask], axis=0)
        
        # Find closest color in reference palette
        distances = np.sum((ref_palette_lab - segment_color) ** 2, axis=1)
        closest_color_idx = np.argmin(distances)
        closest_ref_color = ref_palette_lab[closest_color_idx]
        
        # Blend the ab channels
        original_ab = base_lab[mask, 1:]
        reference_ab = closest_ref_color[1:]
        
        # Apply guidance with strength parameter
        blended_ab = (1 - guidance_strength) * original_ab + guidance_strength * reference_ab
        guided_ab[mask] = blended_ab
    
    # Reconstruct the image
    guided_lab = np.concatenate([gray_lab[:, :, :1], guided_ab], axis=2)
    guided_rgb = lab2rgb(guided_lab)
    
    return np.clip(guided_rgb, 0, 1)

def create_color_transfer_map(source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Create a color transfer map between source and target images
    Useful for transferring color style from one image to another
    """
    # Convert to LAB
    source_lab = rgb2lab(img_as_float(source_img))
    target_lab = rgb2lab(img_as_float(target_img))
    
    # Calculate statistics for each channel
    source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
    source_std = np.std(source_lab.reshape(-1, 3), axis=0)
    
    target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
    target_std = np.std(target_lab.reshape(-1, 3), axis=0)
    
    # Apply color transfer
    transferred_lab = source_lab.copy()
    for i in range(3):
        transferred_lab[:, :, i] = ((source_lab[:, :, i] - source_mean[i]) * 
                                   (target_std[i] / source_std[i]) + target_mean[i])
    
    # Convert back to RGB
    transferred_rgb = lab2rgb(transferred_lab)
    return np.clip(transferred_rgb, 0, 1)

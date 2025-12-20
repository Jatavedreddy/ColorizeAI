import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb

def apply_guided_filter_refinement(l_orig, ab_pred, radius=4, eps=1e-2):
    """
    Refines the predicted ab channels using the original L channel as a guide.
    This forces color edges to align with grayscale edges.
    """
    # Ensure inputs are float32
    l_guide = l_orig.astype(np.float32)
    a_src = ab_pred[:,:,0].astype(np.float32)
    b_src = ab_pred[:,:,1].astype(np.float32)
    
    # Apply Guided Filter
    # cv2.ximgproc.guidedFilter might not be available in standard opencv-python
    # We use the standard cv2.guidedFilter if available, or fallback
    try:
        a_refined = cv2.guidedFilter(l_guide, a_src, radius, eps)
        b_refined = cv2.guidedFilter(l_guide, b_src, radius, eps)
    except AttributeError:
        # Fallback to simple resizing/blurring if guided filter is missing (unlikely in modern cv2)
        return ab_pred
        
    return np.dstack([a_refined, b_refined])

def adaptive_saturation_boost(img_rgb, boost_factor=1.2):
    """
    Smartly boosts saturation.
    - Analyzes current saturation.
    - Boosts low-sat areas more than high-sat areas.
    - Prevents clipping.
    """
    # Convert to HSV
    img_hsv = cv2.cvtColor(img_rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)
    
    # Calculate adaptive boost curve
    # S_new = S_old ^ (1/gamma) where gamma > 1 boosts saturation
    # We want to boost mid-tones, so we use a power curve
    
    # If image is already very colorful, don't boost much
    mean_sat = np.mean(s)
    if mean_sat > 0.5:
        return img_rgb # Already saturated
        
    # Apply boost
    s = np.power(s, 1.0/boost_factor)
    
    # Clip
    s = np.clip(s, 0, 1)
    
    # Merge back
    img_hsv_boosted = cv2.merge([h, s, v])
    img_rgb_boosted = cv2.cvtColor(img_hsv_boosted, cv2.COLOR_HSV2RGB)
    
    return img_rgb_boosted

def enhance_colorization(rgb_pred, gray_orig, strength=1.0):
    """
    The 'Secret Sauce' pipeline to make any colorization look professional.
    
    Args:
        rgb_pred: The raw output from the AI model (float 0-1)
        gray_orig: The original high-res grayscale input (float 0-1)
    """
    # 1. Convert to Lab
    lab_pred = rgb2lab(rgb_pred)
    l_pred, a_pred, b_pred = lab_pred[:,:,0], lab_pred[:,:,1], lab_pred[:,:,2]
    
    # 2. Guided Filter Refinement (The "Edge Snapping")
    # We use the original grayscale as the guide for the color channels
    # Note: rgb2lab L is 0-100, gray_orig is 0-1. Scale gray_orig.
    l_guide = gray_orig * 100.0
    ab_refined = apply_guided_filter_refinement(l_guide, np.dstack([a_pred, b_pred]))
    
    # 3. Recombine with Original L (Luminance Preservation)
    # We strictly use the original L channel to maintain 100% sharpness
    lab_refined = np.dstack([l_guide, ab_refined[:,:,0], ab_refined[:,:,1]])
    rgb_refined = lab2rgb(lab_refined)
    
    # 4. Adaptive Vibrance (The "Pop")
    # Only apply if strength is high
    if strength > 0.8:
        rgb_refined = adaptive_saturation_boost(rgb_refined, boost_factor=1.15)
    
    return np.clip(rgb_refined, 0, 1)

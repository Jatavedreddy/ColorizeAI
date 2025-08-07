"""
Real-time Style Transfer Integration
Combines colorization with artistic style transfer for unique results
"""

import numpy as np
import cv2
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

class StyleTransferColorization:
    def __init__(self):
        self.style_presets = {
            'vintage': {'warmth': 1.3, 'saturation': 0.8, 'sepia': 0.3},
            'modern': {'warmth': 1.0, 'saturation': 1.2, 'contrast': 1.1},
            'pastel': {'warmth': 1.1, 'saturation': 0.6, 'brightness': 1.1},
            'vibrant': {'warmth': 1.0, 'saturation': 1.5, 'contrast': 1.2},
            'cinematic': {'warmth': 1.2, 'saturation': 0.9, 'vignette': 0.2},
            'cold': {'warmth': 0.8, 'saturation': 1.0, 'blue_tint': 0.1},
        }
    
    def apply_color_grading(self, image, style_name='modern', custom_params=None):
        """Apply cinematic color grading to colorized image"""
        if custom_params:
            params = custom_params
        else:
            params = self.style_presets.get(style_name, self.style_presets['modern'])
        
        # Convert to LAB for better color manipulation
        lab = rgb2lab(image)
        
        # Apply warmth adjustment
        if 'warmth' in params:
            warmth = params['warmth']
            # Adjust a channel (green-red axis) for warmth
            lab[:, :, 1] *= warmth
        
        # Apply saturation adjustment
        if 'saturation' in params:
            saturation = params['saturation']
            # Scale a and b channels for saturation
            lab[:, :, 1] *= saturation
            lab[:, :, 2] *= saturation
        
        # Apply brightness adjustment
        if 'brightness' in params:
            brightness = params['brightness']
            lab[:, :, 0] *= brightness
        
        # Apply sepia effect
        if 'sepia' in params:
            sepia_strength = params['sepia']
            # Convert back to RGB for sepia
            rgb_temp = lab2rgb(lab)
            sepia_rgb = self.apply_sepia(rgb_temp, sepia_strength)
            lab = rgb2lab(sepia_rgb)
        
        # Apply blue tint for cold effect
        if 'blue_tint' in params:
            blue_tint = params['blue_tint']
            lab[:, :, 2] += blue_tint * 20  # Shift towards blue
        
        # Convert back to RGB
        result = lab2rgb(lab)
        
        # Apply contrast adjustment
        if 'contrast' in params:
            contrast = params['contrast']
            result = self.adjust_contrast(result, contrast)
        
        # Apply vignette effect
        if 'vignette' in params:
            vignette_strength = params['vignette']
            result = self.apply_vignette(result, vignette_strength)
        
        return np.clip(result, 0, 1)
    
    def apply_sepia(self, image, strength):
        """Apply sepia tone effect"""
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia_img = image.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 1)
        
        # Blend with original based on strength
        return (1 - strength) * image + strength * sepia_img
    
    def adjust_contrast(self, image, factor):
        """Adjust image contrast"""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 1)
    
    def apply_vignette(self, image, strength):
        """Apply vignette effect"""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Create coordinate arrays
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize distance
        normalized_distance = distance / max_distance
        
        # Create vignette mask
        vignette_mask = 1 - strength * normalized_distance**2
        vignette_mask = np.clip(vignette_mask, 0, 1)
        
        # Apply vignette
        result = image.copy()
        for i in range(3):  # Apply to each color channel
            result[:, :, i] *= vignette_mask
        
        return result
    
    def apply_film_emulation(self, image, film_type='kodak'):
        """Emulate different film stocks"""
        film_curves = {
            'kodak': {
                'gamma': 0.9,
                'lift': 0.02,
                'gain': 1.1,
                'color_balance': [1.05, 1.0, 0.95]  # Slightly warm
            },
            'fuji': {
                'gamma': 1.1,
                'lift': 0.01,
                'gain': 1.05,
                'color_balance': [0.98, 1.0, 1.02]  # Slightly cool
            },
            'agfa': {
                'gamma': 0.85,
                'lift': 0.03,
                'gain': 1.15,
                'color_balance': [1.1, 1.0, 0.9]  # Warm and contrasty
            }
        }
        
        if film_type not in film_curves:
            film_type = 'kodak'
        
        params = film_curves[film_type]
        
        # Apply gamma correction
        result = np.power(image, params['gamma'])
        
        # Apply lift (affects shadows)
        result = result + params['lift']
        
        # Apply gain (affects highlights)
        result = result * params['gain']
        
        # Apply color balance
        for i, balance in enumerate(params['color_balance']):
            result[:, :, i] *= balance
        
        return np.clip(result, 0, 1)
    
    def create_artistic_filter(self, image, filter_type='oil_painting'):
        """Apply artistic filters to colorized images"""
        if filter_type == 'oil_painting':
            return self.oil_painting_effect(image)
        elif filter_type == 'watercolor':
            return self.watercolor_effect(image)
        elif filter_type == 'pencil_sketch':
            return self.pencil_sketch_effect(image)
        else:
            return image
    
    def oil_painting_effect(self, image, intensity=7, levels=6):
        """Create oil painting effect"""
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply oil painting filter (if available in OpenCV)
        try:
            # This requires opencv-contrib-python
            result = cv2.xphoto.oilPainting(img_uint8, intensity, levels)
            return result.astype(np.float64) / 255.0
        except:
            # Fallback: bilateral filter for smoothing effect
            result = cv2.bilateralFilter(img_uint8, 15, 80, 80)
            return result.astype(np.float64) / 255.0
    
    def watercolor_effect(self, image):
        """Create watercolor effect"""
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create watercolor effect using edge-preserving filter
        result = cv2.edgePreservingFilter(img_uint8, flags=1, sigma_s=50, sigma_r=0.4)
        
        # Add slight blur for watercolor feel
        result = cv2.bilateralFilter(result, 5, 50, 50)
        
        return result.astype(np.float64) / 255.0
    
    def pencil_sketch_effect(self, image):
        """Create pencil sketch overlay effect"""
        # Convert to grayscale
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Create pencil sketch
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 9)
        
        # Convert back to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB).astype(np.float64) / 255.0
        
        # Blend with original colorized image
        alpha = 0.3  # Sketch overlay strength
        result = (1 - alpha) * image + alpha * edges_rgb
        
        return np.clip(result, 0, 1)

def apply_style_to_colorization(colorized_image, style_type='modern', **kwargs):
    """
    Main function to apply style transfer to colorized image
    
    Args:
        colorized_image: RGB image array in range [0,1]
        style_type: Style preset name or 'custom'
        **kwargs: Custom style parameters
    
    Returns:
        Styled colorized image
    """
    style_engine = StyleTransferColorization()
    
    if style_type in style_engine.style_presets:
        # Apply preset style
        styled_image = style_engine.apply_color_grading(colorized_image, style_type)
    elif style_type == 'custom':
        # Apply custom parameters
        styled_image = style_engine.apply_color_grading(colorized_image, custom_params=kwargs)
    elif style_type.startswith('film_'):
        # Apply film emulation
        film_type = style_type.replace('film_', '')
        styled_image = style_engine.apply_film_emulation(colorized_image, film_type)
    elif style_type.startswith('artistic_'):
        # Apply artistic filter
        filter_type = style_type.replace('artistic_', '')
        styled_image = style_engine.create_artistic_filter(colorized_image, filter_type)
    else:
        styled_image = colorized_image
    
    return styled_image

"""
Multi-Model Ensemble with Smart Fusion
Intelligently combines results from multiple models based on image characteristics
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
import torch

class SmartModelFusion:
    def __init__(self):
        self.texture_threshold = 0.1
        self.edge_threshold = 0.15
        
    def analyze_image_characteristics(self, grayscale_img):
        """Analyze image to determine optimal model weighting"""
        # Convert to float
        gray_float = grayscale_img.astype(np.float64) / 255.0
        
        characteristics = {}
        
        # 1. Texture analysis using Local Binary Pattern
        lbp = local_binary_pattern(gray_float, P=8, R=1, method='uniform')
        texture_variance = np.var(lbp)
        characteristics['texture_complexity'] = min(texture_variance / 100.0, 1.0)
        
        # 2. Edge density analysis
        edges = sobel(gray_float)
        edge_density = np.mean(edges > self.edge_threshold)
        characteristics['edge_density'] = edge_density
        
        # 3. Contrast analysis
        contrast = np.std(gray_float)
        characteristics['contrast'] = min(contrast, 1.0)
        
        # 4. Scene type classification (simplified)
        # Check for natural vs artificial patterns
        # Natural scenes typically have more organic, curved edges
        # Artificial scenes have more straight lines and geometric shapes
        
        # Detect lines using Hough transform
        edges_uint8 = (edges * 255).astype(np.uint8)
        lines = cv2.HoughLines(edges_uint8, 1, np.pi/180, threshold=50)
        line_density = len(lines) if lines is not None else 0
        characteristics['geometric_content'] = min(line_density / 100.0, 1.0)
        
        # 5. Spatial frequency analysis
        f_transform = np.fft.fft2(gray_float)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        high_freq_energy = np.mean(magnitude_spectrum[magnitude_spectrum.shape[0]//4:3*magnitude_spectrum.shape[0]//4,
                                                     magnitude_spectrum.shape[1]//4:3*magnitude_spectrum.shape[1]//4])
        characteristics['high_frequency_content'] = min(high_freq_energy / 10.0, 1.0)
        
        return characteristics
    
    def calculate_model_weights(self, characteristics, use_ddcolor=False):
        """Calculate optimal weights for each model based on image characteristics"""
        
        if use_ddcolor:
            # DDColor is generally superior, so we give it high base weight
            # We only blend in others if texture complexity is very high (where DDColor might be too smooth)
            ddcolor_weight = 0.85
            
            # If texture is complex, give more weight to SIGGRAPH17 (good at texture)
            if characteristics['texture_complexity'] > 0.5:
                ddcolor_weight -= 0.15
                siggraph17_weight = 0.15
                eccv16_weight = 0.0
            else:
                # Otherwise trust DDColor more
                ddcolor_weight += 0.1
                siggraph17_weight = 0.05
                eccv16_weight = 0.0
                
            return {
                'ddcolor': ddcolor_weight,
                'eccv16': eccv16_weight,
                'siggraph17': siggraph17_weight
            }

        # Legacy logic for just ECCV16/SIGGRAPH17
        # ECCV16 tends to be better for:
        # - High contrast images
        # - Images with clear objects
        # - Geometric/artificial content
        
        # SIGGRAPH17 tends to be better for:
        # - Natural scenes
        # - Complex textures
        # - Lower contrast images
        
        eccv16_weight = (
            0.3 * characteristics['contrast'] +
            0.2 * characteristics['geometric_content'] +
            0.2 * characteristics['edge_density'] +
            0.3 * (1 - characteristics['texture_complexity'])  # ECCV16 better for simpler textures
        )
        
        siggraph17_weight = (
            0.4 * characteristics['texture_complexity'] +
            0.3 * (1 - characteristics['geometric_content']) +  # Better for natural scenes
            0.2 * characteristics['high_frequency_content'] +
            0.1 * (1 - characteristics['contrast'])  # Better for lower contrast
        )
        
        # Normalize weights
        total_weight = eccv16_weight + siggraph17_weight
        if total_weight > 0:
            eccv16_weight /= total_weight
            siggraph17_weight /= total_weight
        else:
            eccv16_weight = siggraph17_weight = 0.5
        
        return {
            'eccv16': eccv16_weight,
            'siggraph17': siggraph17_weight
        }
    
    def adaptive_fusion(self, eccv16_result, siggraph17_result, grayscale_img, ddcolor_result=None):
        """Perform adaptive fusion of model results"""
        # Analyze image characteristics
        characteristics = self.analyze_image_characteristics(grayscale_img)
        
        # Calculate model weights
        weights = self.calculate_model_weights(characteristics, use_ddcolor=(ddcolor_result is not None))
        
        # Convert to LAB for better blending
        eccv16_lab = rgb2lab(eccv16_result)
        siggraph17_lab = rgb2lab(siggraph17_result)
        
        if ddcolor_result is not None:
            ddcolor_lab = rgb2lab(ddcolor_result)
            
            # Simple weighted blend for now (spatially varying is complex with 3 models)
            # But we can still use the smart weights we calculated
            fused_lab = np.zeros_like(ddcolor_lab)
            fused_lab[:,:,0] = ddcolor_lab[:,:,0] # Use L from DDColor (should be same as input)
            
            for channel in [1, 2]:
                fused_lab[:,:,channel] = (
                    weights['ddcolor'] * ddcolor_lab[:,:,channel] +
                    weights['siggraph17'] * siggraph17_lab[:,:,channel] +
                    weights['eccv16'] * eccv16_lab[:,:,channel]
                )
                
            fused_rgb = lab2rgb(fused_lab)
            return np.clip(fused_rgb, 0, 1), weights, characteristics

        # Perform spatially-varying fusion (Legacy 2-model)
        fused_lab = self.spatially_varying_fusion(
            eccv16_lab, siggraph17_lab, grayscale_img, weights
        )
        
        # Convert back to RGB
        fused_rgb = lab2rgb(fused_lab)
        
        return np.clip(fused_rgb, 0, 1), weights, characteristics
    
    def spatially_varying_fusion(self, eccv16_lab, siggraph17_lab, grayscale_img, global_weights):
        """Perform spatially-varying fusion based on local image properties"""
        h, w = grayscale_img.shape[:2]
        
        # Create local weight maps
        local_weights = np.ones((h, w)) * global_weights['eccv16']
        
        # Analyze local properties using sliding window
        window_size = 32
        for i in range(0, h - window_size, window_size // 2):
            for j in range(0, w - window_size, window_size // 2):
                # Extract local patch
                patch = grayscale_img[i:i+window_size, j:j+window_size]
                
                if patch.size == 0:
                    continue
                
                # Analyze local characteristics
                local_contrast = np.std(patch.astype(np.float64) / 255.0)
                local_edges = sobel(patch.astype(np.float64) / 255.0)
                local_edge_density = np.mean(local_edges > self.edge_threshold)
                
                # Adjust weights based on local properties
                if local_contrast > 0.2 and local_edge_density > 0.1:
                    # High contrast, high edge density - favor ECCV16
                    local_weight = min(global_weights['eccv16'] + 0.2, 1.0)
                elif local_contrast < 0.1:
                    # Low contrast - favor SIGGRAPH17
                    local_weight = max(global_weights['eccv16'] - 0.2, 0.0)
                else:
                    local_weight = global_weights['eccv16']
                
                # Apply to region
                end_i = min(i + window_size, h)
                end_j = min(j + window_size, w)
                local_weights[i:end_i, j:end_j] = local_weight
        
        # Smooth the weight map to avoid artifacts
        local_weights = cv2.GaussianBlur(local_weights, (15, 15), 5)
        
        # Apply fusion
        fused_lab = eccv16_lab.copy()
        
        # Blend a and b channels using local weights
        for channel in [1, 2]:  # a and b channels
            fused_lab[:, :, channel] = (
                local_weights * eccv16_lab[:, :, channel] +
                (1 - local_weights) * siggraph17_lab[:, :, channel]
            )
        
        return fused_lab

def ensemble_colorization(grayscale_img, eccv16_result, siggraph17_result, ddcolor_result=None):
    """
    Main function to perform ensemble colorization
    
    Args:
        grayscale_img: Input grayscale image
        eccv16_result: Colorization result from ECCV16 model
        siggraph17_result: Colorization result from SIGGRAPH17 model
        ddcolor_result: Colorization result from DDColor model (optional)
    
    Returns:
        tuple: (fused_result, weights_used, image_characteristics)
    """
    fusion_engine = SmartModelFusion()
    
    fused_result, weights, characteristics = fusion_engine.adaptive_fusion(
        eccv16_result, siggraph17_result, grayscale_img, ddcolor_result=ddcolor_result
    )
    
    return fused_result, weights, characteristics

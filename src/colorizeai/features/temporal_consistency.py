"""
Temporal Consistency for Video Colorization
Reduces flickering and maintains color consistency across video frames
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from skimage.color import rgb2lab, lab2rgb
from collections import deque

class TemporalConsistencyEngine:
    def __init__(self, consistency_strength=0.3, history_size=5):
        self.consistency_strength = consistency_strength
        self.history_size = history_size
        self.frame_history = deque(maxlen=history_size)
        self.flow_history = deque(maxlen=history_size-1)
        
    def estimate_optical_flow(self, prev_frame, curr_frame):
        """Estimate optical flow between consecutive frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        # Use Farneback optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, None, None,
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        return flow
    
    def warp_frame(self, frame, flow):
        """Warp a frame using optical flow"""
        h, w = frame.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow vectors
        map_x = map_x + flow[:, :, 0]
        map_y = map_y + flow[:, :, 1]
        
        # Warp the frame
        warped = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), 
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped
    
    def apply_temporal_consistency(self, current_colorized, current_gray):
        """Apply temporal consistency to current frame"""
        if len(self.frame_history) == 0:
            # First frame - no consistency to apply
            self.frame_history.append({
                'colorized': current_colorized.copy(),
                'grayscale': current_gray.copy()
            })
            return current_colorized
        
        # Get previous frame
        prev_frame_data = self.frame_history[-1]
        prev_colorized = prev_frame_data['colorized']
        prev_gray = prev_frame_data['grayscale']
        
        # Estimate optical flow
        flow = self.estimate_optical_flow(prev_gray, current_gray)
        
        # Warp previous colorized frame to current frame
        warped_prev = self.warp_frame(prev_colorized, flow)
        
        # Blend current colorization with warped previous frame
        # Use temporal consistency strength
        consistent_frame = self.temporal_blend(
            current_colorized, warped_prev, current_gray, prev_gray
        )
        
        # Store current frame in history
        self.frame_history.append({
            'colorized': consistent_frame.copy(),
            'grayscale': current_gray.copy()
        })
        
        return consistent_frame
    
    def temporal_blend(self, current_colorized, warped_prev, current_gray, prev_gray):
        """Intelligent blending based on scene changes"""
        # Calculate scene change score
        gray_diff = np.mean(np.abs(current_gray.astype(float) - prev_gray.astype(float)))
        scene_change_score = min(gray_diff / 50.0, 1.0)  # Normalize to [0,1]
        
        # Adaptive consistency strength based on scene change
        adaptive_strength = self.consistency_strength * (1 - scene_change_score)
        
        # Convert to LAB for better blending
        current_lab = rgb2lab(current_colorized)
        warped_lab = rgb2lab(warped_prev)
        
        # Blend only the a and b channels (preserve luminance from current frame)
        blended_lab = current_lab.copy()
        blended_lab[:, :, 1:] = (
            (1 - adaptive_strength) * current_lab[:, :, 1:] +
            adaptive_strength * warped_lab[:, :, 1:]
        )
        
        # Convert back to RGB
        blended_rgb = lab2rgb(blended_lab)
        return np.clip(blended_rgb, 0, 1)
    
    def reset(self):
        """Reset the temporal consistency engine"""
        self.frame_history.clear()
        self.flow_history.clear()

def apply_temporal_smoothing(frames_colorized, frames_grayscale, smoothing_strength=0.3):
    """
    Apply temporal smoothing to a sequence of colorized frames
    
    Args:
        frames_colorized: List of colorized frames
        frames_grayscale: List of corresponding grayscale frames
        smoothing_strength: Strength of temporal smoothing (0-1)
    
    Returns:
        List of temporally consistent frames
    """
    if len(frames_colorized) < 2:
        return frames_colorized
    
    engine = TemporalConsistencyEngine(consistency_strength=smoothing_strength)
    consistent_frames = []
    
    for i, (colorized, grayscale) in enumerate(zip(frames_colorized, frames_grayscale)):
        consistent_frame = engine.apply_temporal_consistency(colorized, grayscale)
        consistent_frames.append(consistent_frame)
    
    return consistent_frames

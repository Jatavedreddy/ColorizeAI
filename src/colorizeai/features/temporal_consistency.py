"""
Temporal Consistency for Video Colorization
Reduces flickering and maintains color consistency across video frames
"""

import numpy as np
"""Temporal consistency utilities with adaptive optical flow and scene change detection."""

from collections import deque
from typing import List
import numpy as np
import cv2
from skimage.color import rgb2lab, lab2rgb


class TemporalConsistencyEngine:
    """Enhances temporal coherence by adaptive blending of successive frames."""

    def __init__(self, consistency_strength: float = 0.3, history_size: int = 5):
        self.consistency_strength = consistency_strength
        self.history_size = history_size
        self.frame_history = deque(maxlen=history_size)
        self.scene_change_threshold = 0.15
        self.last_motion_level = 0.0

    def reset(self):
        self.frame_history.clear()
        self.scene_change_threshold = 0.15
        self.last_motion_level = 0.0

    def _estimate_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        frame_diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
        motion_level = min(frame_diff / 30.0, 1.0)
        motion_level = 0.7 * self.last_motion_level + 0.3 * motion_level
        self.last_motion_level = motion_level
        levels = max(3, min(5, int(3 + motion_level * 2)))
        winsize = max(15, min(21, int(15 + motion_level * 6)))
        iterations = max(3, min(5, int(3 + motion_level * 2)))
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=levels, winsize=winsize,
                iterations=iterations, poly_n=5, poly_sigma=1.2, flags=0
            )
        except Exception:
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
            motion_level = 0.0
        return flow, motion_level

    def _scene_change_score(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        if prev_gray is None:
            return 1.0
        hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
        hist_curr = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
        hist_diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
        hist_score = max(0, 1 - hist_diff)
        gray_diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
        gray_score = min(gray_diff / 50.0, 1.0)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(prev_gray, curr_gray, data_range=255)
            struct_score = 1 - ssim_score
        except Exception:
            struct_score = gray_score
        combined = 0.4 * hist_score + 0.3 * gray_score + 0.3 * struct_score
        if combined > self.scene_change_threshold:
            self.scene_change_threshold = min(0.25, combined * 1.1)
        else:
            self.scene_change_threshold = max(0.15, self.scene_change_threshold * 0.95)
        return combined

    def _warp(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y + flow[:, :, 1]).astype(np.float32)
        try:
            return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        except Exception:
            return frame

    def _blend(self, curr_rgb: np.ndarray, warped_prev_rgb: np.ndarray, prev_gray: np.ndarray, curr_gray: np.ndarray, flow: np.ndarray) -> np.ndarray:
        scene_score = self._scene_change_score(prev_gray, curr_gray)
        if scene_score > 0.3:
            return curr_rgb
        flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        max_flow = np.percentile(flow_mag, 95) or 1.0
        flow_quality = np.exp(-flow_mag / (max_flow * 2))
        alpha = self.consistency_strength * (1 - scene_score) * flow_quality
        alpha = np.clip(alpha, 0, 0.8)
        curr_lab = rgb2lab(curr_rgb)
        prev_lab = rgb2lab(warped_prev_rgb)
        blended_lab = curr_lab.copy()
        blended_lab[:, :, 1:] = (1 - alpha[:, :, None]) * curr_lab[:, :, 1:] + alpha[:, :, None] * prev_lab[:, :, 1:]
        blended = lab2rgb(blended_lab)
        return np.clip(blended, 0, 1)

    def apply_temporal_consistency(self, curr_rgb: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        if len(self.frame_history) == 0:
            self.frame_history.append((curr_rgb.copy(), curr_gray.copy()))
            return curr_rgb
        prev_rgb, prev_gray = self.frame_history[-1]
        flow, _ = self._estimate_flow(prev_gray, curr_gray)
        warped_prev = self._warp(prev_rgb, flow)
        blended = self._blend(curr_rgb, warped_prev, prev_gray, curr_gray, flow)
        self.frame_history.append((blended.copy(), curr_gray.copy()))
        return blended


def apply_temporal_smoothing(frames_colorized: List[np.ndarray], frames_grayscale: List[np.ndarray], smoothing_strength: float = 0.3) -> List[np.ndarray]:
    if len(frames_colorized) < 2:
        return frames_colorized
    engine = TemporalConsistencyEngine(consistency_strength=smoothing_strength)
    result = []
    for c, g in zip(frames_colorized, frames_grayscale):
        result.append(engine.apply_temporal_consistency(c, g))
    return result

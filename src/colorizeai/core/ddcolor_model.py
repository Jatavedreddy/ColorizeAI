"""
DDColor Integration - Base Colorization Model

This module provides a comprehensive wrapper for the DDColor model, which uses
a ConvNeXt encoder and multi-scale decoder for high-quality image colorization.

Architecture:
- Encoder: ConvNeXt-T (tiny) or ConvNeXt-L (large)
- Decoder: MultiScaleColorDecoder with 100 queries, 3 scales, 9 layers
- Input: RGB grayscale image [N, 3, H, W] (converted to Lab internally)
- Output: ab channels [N, 2, H, W] in Lab space

Weights discovery:
- Environment variable `DDCOLOR_WEIGHTS` pointing to model checkpoint
- Or auto-search in: ../../../ddcolor/DDColor-master copy/modelscope/damo/cv_ddcolor_image-colorization/
- Or fallback paths: weights/ddcolor.pt, pretrain/ddcolor.pt

The model is loaded from the DDColor project and integrated as the primary colorizer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add DDColor project to path
_current_file = Path(__file__).resolve()
_ddcolor_project_path = _current_file.parents[4] / "ddcolor" / "DDColor-master copy"
if _ddcolor_project_path.exists():
    sys.path.insert(0, str(_ddcolor_project_path))

try:
    from basicsr.archs.ddcolor_arch import DDColor as DDColorArch
    _DDCOLOR_ARCH_AVAILABLE = True
except ImportError:
    _DDCOLOR_ARCH_AVAILABLE = False
    DDColorArch = None

try:
    from .models import get_device as _get_device
except Exception:
    def _get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device("cpu")


# Global state
_ddcolor_model: Optional[torch.nn.Module] = None
_ddcolor_device: Optional[torch.device] = None
_ddcolor_load_error: Optional[str] = None
_ddcolor_input_size: int = 512
_ddcolor_model_size: str = 'large'


def _find_weights_path() -> Optional[Path]:
    """Search for DDColor weights in multiple locations"""
    env = os.getenv("DDCOLOR_WEIGHTS")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    
    # Primary location: DDColor project folder
    if _ddcolor_project_path.exists():
        primary = _ddcolor_project_path / "modelscope" / "damo" / "cv_ddcolor_image-colorization" / "pytorch_model.pt"
        if primary.exists():
            return primary
    
    # Fallback locations
    candidates = [
        Path("weights/ddcolor.pt"),
        Path("pretrain/ddcolor.pt"),
        Path("../ddcolor/DDColor-master copy/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"),
    ]
    for c in candidates:
        c_abs = c.expanduser().resolve()
        if c_abs.exists():
            return c_abs
    
    return None


def load_ddcolor(
    device: Optional[torch.device] = None,
    input_size: int = 512,
    model_size: str = 'large'
) -> Tuple[Optional[torch.nn.Module], torch.device]:
    """Load the DDColor model.

    Args:
        device: Target device (cuda/mps/cpu). Auto-detected if None.
        input_size: Input resolution (256, 512, etc.)
        model_size: 'tiny' or 'large' for ConvNeXt-T or ConvNeXt-L

    Returns:
        (model_or_None, device)
    """
    global _ddcolor_model, _ddcolor_device, _ddcolor_load_error, _ddcolor_input_size, _ddcolor_model_size
    
    # Return cached if already loaded with same config
    if _ddcolor_model is not None and _ddcolor_input_size == input_size and _ddcolor_model_size == model_size:
        return _ddcolor_model, _ddcolor_device or _get_device()

    if not _DDCOLOR_ARCH_AVAILABLE:
        _ddcolor_load_error = "DDColor architecture not available. Check that basicsr is importable."
        return None, device or _get_device()

    if device is None:
        device = _get_device()
    _ddcolor_device = device
    _ddcolor_input_size = input_size
    _ddcolor_model_size = model_size

    weights_path = _find_weights_path()
    if weights_path is None:
        _ddcolor_model = None
        _ddcolor_load_error = (
            "DDColor weights not found. Please ensure the DDColor project is present at "
            "../ddcolor/DDColor-master copy/ with weights in modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
        )
        return None, device

    try:
        encoder_name = 'convnext-t' if model_size == 'tiny' else 'convnext-l'
        model = DDColorArch(
            encoder_name=encoder_name,
            decoder_name='MultiScaleColorDecoder',
            input_size=[input_size, input_size],
            num_output_channels=2,
            last_norm='Spectral',
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['params'], strict=False)
        model.eval()
        
        _ddcolor_model = model
        _ddcolor_load_error = None
        print(f"✓ DDColor ({model_size}) loaded successfully from {weights_path}")
        return _ddcolor_model, device
    except Exception as e:
        _ddcolor_model = None
        _ddcolor_load_error = f"Failed to load DDColor: {e}"
        print(f"✗ DDColor loading failed: {e}")
        return None, device


def get_ddcolor() -> Tuple[Optional[torch.nn.Module], torch.device, Optional[str]]:
    """Get the global DDColor model if available.

    Returns (model_or_None, device, error_message_or_None).
    """
    model, device = load_ddcolor()
    return model, device, _ddcolor_load_error


def is_ddcolor_available() -> bool:
    """Check if DDColor model is loaded and ready."""
    model, _, _ = get_ddcolor()
    return model is not None


@torch.no_grad()
def predict_ab_with_ddcolor(
    img_bgr: np.ndarray,
    input_size: Optional[int] = None,
    model_size: str = 'large'
) -> np.ndarray:
    """
    Colorize a BGR image using DDColor.

    Args:
        img_bgr: Input BGR image (uint8 or float). Can be color or grayscale.
        input_size: Processing resolution. Uses model's default if None.
        model_size: 'tiny' or 'large'

    Returns:
        Colorized BGR image (uint8), same size as input.
        Returns None if DDColor is not available.
    """
    model, device = load_ddcolor(input_size=input_size or _ddcolor_input_size, model_size=model_size)
    if model is None:
        return None

    # Normalize to float [0, 1]
    if img_bgr.dtype == np.uint8:
        img = (img_bgr / 255.0).astype(np.float32)
    else:
        img = img_bgr.astype(np.float32)

    height, width = img.shape[:2]
    
    # Convert BGR -> Lab, extract L channel
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

    # Resize to model input size
    use_size = input_size or _ddcolor_input_size
    img_resized = cv2.resize(img, (use_size, use_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    
    # Create grayscale Lab -> RGB for model input
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    # To tensor [1, 3, H, W]
    tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    
    # Predict ab channels
    output_ab = model(tensor_gray_rgb).cpu()  # (1, 2, H, W)

    # Resize ab to original resolution
    output_ab_resized = F.interpolate(output_ab, size=(height, width))[0].float().numpy().transpose(1, 2, 0)
    
    # Concatenate with original L
    output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

    # Back to uint8
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    return output_img


@torch.no_grad()
def predict_ab_channels(
    img_l_normalized: np.ndarray,
    device: Optional[torch.device] = None
) -> Optional[np.ndarray]:
    """
    Predict ab channels from L channel (for integration with existing pipeline).
    
    Args:
        img_l_normalized: L channel normalized 0..100, shape (H, W) or (H, W, 1)
        device: Device to use
    
    Returns:
        ab channels (H, W, 2) in range ~-128..128, or None if unavailable
    """
    model, dev = load_ddcolor(device=device)
    if model is None:
        return None
    
    if img_l_normalized.ndim == 2:
        img_l_normalized = img_l_normalized[:, :, np.newaxis]
    
    h, w = img_l_normalized.shape[:2]
    
    # Build a gray Lab image
    lab_gray = np.concatenate([img_l_normalized, np.zeros((h, w, 2))], axis=2)
    # Convert to RGB
    rgb_gray = cv2.cvtColor(lab_gray.astype(np.float32), cv2.COLOR_LAB2RGB)
    
    # Resize to model input
    use_size = _ddcolor_input_size
    rgb_resized = cv2.resize(rgb_gray, (use_size, use_size))
    
    # To tensor
    tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(dev)
    ab_out = model(tensor).cpu()
    
    # Resize back
    ab_np = F.interpolate(ab_out, size=(h, w))[0].float().numpy().transpose(1, 2, 0)
    return ab_np


class DDColorPipeline:
    """
    High-level pipeline wrapper for DDColor colorization.
    Provides both image and batch processing.
    """
    def __init__(self, model_size: str = 'large', input_size: int = 512, device: Optional[torch.device] = None):
        self.model_size = model_size
        self.input_size = input_size
        self.device = device or _get_device()
        self.model, self.device = load_ddcolor(device=self.device, input_size=input_size, model_size=model_size)
        
        if self.model is None:
            raise RuntimeError(f"Failed to load DDColor: {_ddcolor_load_error}")
    
    @torch.no_grad()
    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """Colorize a single BGR image."""
        return predict_ab_with_ddcolor(img_bgr, input_size=self.input_size, model_size=self.model_size)
    
    @torch.no_grad()
    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Colorize a batch of images."""
        results = []
        for img in images:
            result = self.process(img)
            if result is not None:
                results.append(result)
        return results


# Utility function for quick colorization
def colorize_image(
    img: np.ndarray,
    model_size: str = 'large',
    input_size: int = 512
) -> Optional[np.ndarray]:
    """
    Quick colorization of a single image.
    
    Args:
        img: Input image (BGR or RGB, uint8 or float)
        model_size: 'tiny' or 'large'
        input_size: Processing resolution
    
    Returns:
        Colorized image (same format as input), or None if failed
    """
    return predict_ab_with_ddcolor(img, input_size=input_size, model_size=model_size)


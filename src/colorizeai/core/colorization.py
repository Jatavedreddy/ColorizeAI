"""
Core colorization algorithms
"""

import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

from .models import get_models
from ..features.smart_model_fusion import ensemble_colorization
from ..features.reference_guided_colorization import apply_reference_guided_colorization
from ..features.interactive_color_hints import apply_color_hints
from ..features.style_transfer_colorization import apply_style_to_colorization

def _predict_ab(model, l_channel: np.ndarray) -> np.ndarray:
    """Runs the model on a 256×256 L-channel and returns ab [H,W,2] in range-128…128."""
    tens = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0).float()
    
    # Get device from models
    _, device = get_models()
    tens = tens.to(device)
    
    with torch.no_grad():
        out_ab = model(tens)
    ab = (
        out_ab.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )  # (256,256,2), still ‑128…128
    return ab

def colorize_highres(image_np: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Basic colorization with both models and blend with the original.

    Returns two arrays in RGB float [0,1] matching the original resolution.
    """
    # Get models
    (colorizer_eccv16, colorizer_siggraph17), device = get_models()
    
    # Ensure float in [0,1]
    if image_np.dtype != np.float64 and image_np.dtype != np.float32:
        img = image_np.astype(np.float64) / 255.0
    else:
        img = np.clip(image_np, 0, 1)

    h, w = img.shape[:2]

    # Extract original L channel
    img_lab_orig = rgb2lab(img)
    img_l_orig = img_lab_orig[:, :, 0]

    # Down-scale to 256×256 for model input
    img_small = resize(img, (256, 256), preserve_range=True)
    img_lab_small = rgb2lab(img_small)
    img_l_small = img_lab_small[:, :, 0]

    # Predict ab for both models
    ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)
    ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)

    # Upsample ab maps to original resolution
    ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)
    ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)

    def _lab_to_rgb(l_orig: np.ndarray, ab_up: np.ndarray) -> np.ndarray:
        lab = np.concatenate((l_orig[:, :, np.newaxis], ab_up), axis=2)
        rgb = lab2rgb(lab)
        return rgb

    out_eccv_rgb = _lab_to_rgb(img_l_orig, ab_eccv_up)
    out_sig_rgb = _lab_to_rgb(img_l_orig, ab_sig_up)

    # Blend with original for fine detail
    strength = float(strength)
    out_eccv_rgb = (1 - strength) * img + strength * out_eccv_rgb
    out_sig_rgb = (1 - strength) * img + strength * out_sig_rgb

    return out_eccv_rgb, out_sig_rgb

def colorize_highres_enhanced(
    image_np: np.ndarray, 
    strength: float = 1.0,
    use_ensemble: bool = True,
    reference_img: np.ndarray = None,
    color_hints: list = None,
    style_type: str = 'modern'
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Enhanced colorization with multiple techniques
    
    Returns:
        tuple: (eccv_result, siggraph_result, metadata)
    """
    # Get models
    (colorizer_eccv16, colorizer_siggraph17), device = get_models()
    
    # Ensure float in [0,1]
    if image_np.dtype != np.float64 and image_np.dtype != np.float32:
        img = image_np.astype(np.float64) / 255.0
    else:
        img = np.clip(image_np, 0, 1)

    h, w = img.shape[:2]

    # Extract original L channel
    img_lab_orig = rgb2lab(img)
    img_l_orig = img_lab_orig[:, :, 0]

    # Down-scale to 256×256 for model input
    img_small = resize(img, (256, 256), preserve_range=True)
    img_lab_small = rgb2lab(img_small)
    img_l_small = img_lab_small[:, :, 0]

    # Predict ab for both models
    ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)
    ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)

    # Upsample ab maps to original resolution
    ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)
    ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)

    def _lab_to_rgb(l_orig: np.ndarray, ab_up: np.ndarray) -> np.ndarray:
        lab = np.concatenate((l_orig[:, :, np.newaxis], ab_up), axis=2)
        rgb = lab2rgb(lab)
        return rgb

    out_eccv_rgb = _lab_to_rgb(img_l_orig, ab_eccv_up)
    out_sig_rgb = _lab_to_rgb(img_l_orig, ab_sig_up)

    # Apply ensemble fusion if requested
    metadata = {}
    if use_ensemble:
        try:
            fused_result, weights, characteristics = ensemble_colorization(
                (img * 255).astype(np.uint8), out_eccv_rgb, out_sig_rgb
            )
            metadata['ensemble_weights'] = weights
            metadata['image_characteristics'] = characteristics
            
            # Use fused result as the primary output
            ensemble_result = fused_result
        except Exception as e:
            print(f"Ensemble fusion failed: {e}")
            ensemble_result = out_sig_rgb  # Fallback
            metadata['ensemble_error'] = str(e)
    else:
        ensemble_result = out_sig_rgb

    # Apply reference guidance if provided
    if reference_img is not None:
        try:
            ensemble_result = apply_reference_guided_colorization(
                (img * 255).astype(np.uint8),
                reference_img,
                ensemble_result,
                guidance_strength=0.3
            )
            metadata['reference_guided'] = True
        except Exception as e:
            print(f"Reference guidance failed: {e}")
            metadata['reference_error'] = str(e)

    # Apply color hints if provided
    if color_hints and len(color_hints) > 0:
        try:
            ensemble_result = apply_color_hints(
                (img * 255).astype(np.uint8),
                ensemble_result,
                color_hints,
                hint_radius=20,
                propagation_strength=0.7
            )
            metadata['color_hints_applied'] = len(color_hints)
        except Exception as e:
            print(f"Color hints failed: {e}")
            metadata['hints_error'] = str(e)

    # Apply style transfer
    if style_type != 'none':
        try:
            ensemble_result = apply_style_to_colorization(ensemble_result, style_type)
            metadata['style_applied'] = style_type
        except Exception as e:
            print(f"Style transfer failed: {e}")
            metadata['style_error'] = str(e)

    # Blend with original for fine detail
    strength = float(strength)
    out_eccv_rgb = (1 - strength) * img + strength * out_eccv_rgb
    out_sig_rgb = (1 - strength) * img + strength * ensemble_result

    return out_eccv_rgb, out_sig_rgb, metadata

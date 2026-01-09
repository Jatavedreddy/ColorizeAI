"""
Core colorization algorithms

This module provides the main colorization pipeline using DDColor as the base model,
with optional fusion with ECCV16/SIGGRAPH17 for enhanced results.
"""

import numpy as np
import torch
import cv2
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

from .models import get_models
from .ddcolor_model import (
    predict_ab_with_ddcolor, 
    predict_ab_channels,
    is_ddcolor_available,
    colorize_image as ddcolor_colorize
)
from ..features.smart_model_fusion import ensemble_colorization
from ..features.reference_guided_colorization import apply_reference_guided_colorization
from ..features.interactive_color_hints import apply_color_hints
from ..features.style_transfer_colorization import apply_style_to_colorization
from ..features.post_processing import enhance_colorization


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


def colorize_highres(image_np: np.ndarray, strength: float = 1.0, use_ddcolor: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    High-resolution colorization with DDColor as primary, with fallback options.

    Args:
        image_np: Input RGB image (uint8 or float [0,1])
        strength: Blend strength (0=grayscale, 1=full color)
        use_ddcolor: Use DDColor if available (True by default)

    Returns:
        tuple: (eccv_result, primary_result) - two arrays in RGB float [0,1] at original resolution.
               primary_result uses DDColor if available, else SIGGRAPH17
    """
    # Ensure float in [0,1]
    if image_np.dtype != np.float64 and image_np.dtype != np.float32:
        img = image_np.astype(np.float64) / 255.0
    else:
        img = np.clip(image_np, 0, 1)

    h, w = img.shape[:2]

    # Try DDColor first if enabled and available
    primary_rgb = None
    if use_ddcolor and is_ddcolor_available():
        try:
            # DDColor works on BGR
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            colored_bgr = predict_ab_with_ddcolor(img_bgr, input_size=512, model_size='large')
            if colored_bgr is not None:
                primary_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                print("✓ Using DDColor for primary colorization")
        except Exception as e:
            print(f"⚠ DDColor prediction failed: {e}, falling back to SIGGRAPH17")
            primary_rgb = None
    
    # Fallback to classic models if DDColor not used or failed
    if primary_rgb is None:
        (colorizer_eccv16, colorizer_siggraph17), device = get_models()
        
        # Extract original L channel
        img_lab_orig = rgb2lab(img)
        img_l_orig = img_lab_orig[:, :, 0]

        # Down-scale to 256×256 for model input
        img_small = resize(img, (256, 256), preserve_range=True)
        img_lab_small = rgb2lab(img_small)
        img_l_small = img_lab_small[:, :, 0]

        # Predict ab
        ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)
        ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)

        # Upsample to original resolution
        ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)
        ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)

        def _lab_to_rgb(l_orig: np.ndarray, ab_up: np.ndarray) -> np.ndarray:
            lab = np.concatenate((l_orig[:, :, np.newaxis], ab_up), axis=2)
            rgb = lab2rgb(lab)
            return rgb

        out_eccv_rgb = _lab_to_rgb(img_l_orig, ab_eccv_up)
        primary_rgb = _lab_to_rgb(img_l_orig, ab_sig_up)
        
        print("✓ Using SIGGRAPH17/ECCV16 for colorization")
    else:
        # Also compute ECCV for comparison output
        (colorizer_eccv16, _), device = get_models()
        img_lab_orig = rgb2lab(img)
        img_l_orig = img_lab_orig[:, :, 0]
        img_small = resize(img, (256, 256), preserve_range=True)
        img_lab_small = rgb2lab(img_small)
        img_l_small = img_lab_small[:, :, 0]
        ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)
        ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)
        out_eccv_rgb = lab2rgb(np.concatenate((img_l_orig[:, :, np.newaxis], ab_eccv_up), axis=2))

    # Blend with original for detail preservation
    strength = float(strength)
    out_eccv_rgb = (1 - strength) * img + strength * out_eccv_rgb
    primary_rgb = (1 - strength) * img + strength * primary_rgb

    return out_eccv_rgb, primary_rgb


def colorize_highres_enhanced(
    image_np: np.ndarray,
    strength: float = 1.0,
    use_ensemble: bool = True,
    reference_img: np.ndarray = None,
    color_hints: list = None,
    style_type: str = 'none',
    use_ddcolor: bool = True,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Enhanced colorization pipeline with multiple features.
    
    Uses DDColor as the base model when available, with optional:
    - Model ensemble (DDColor + ECCV16/SIGGRAPH17 fusion)
    - Reference-guided colorization
    - Interactive color hints
    - Style transfer/presets
    
    Args:
        image_np: Input RGB image (uint8 or float [0,1])
        strength: Blend strength (0=grayscale, 1=full color)
        use_ensemble: Enable model fusion for enhanced results
        reference_img: Optional reference image for color guidance
        color_hints: Optional list of color hints (points/strokes)
        style_type: Style preset ('modern', 'vintage', 'cinematic', etc.)
        use_ddcolor: Use DDColor if available (True by default)
    
    Returns:
        tuple: (eccv_result, primary_result, metadata)
               - eccv_result: ECCV16 baseline output
               - primary_result: DDColor-based enhanced result with all features
               - metadata: dict with processing info, weights, characteristics
    """
    # Handle argument aliases for compatibility with new UI calls
    if 'reference_image' in kwargs and reference_img is None:
        reference_img = kwargs['reference_image']
    if 'style' in kwargs:
        style = kwargs['style']
        if style is not None and style != 'none':
             style_type = style

    # Ensure float in [0,1]
    # Check for grayscale input (2D or 1-channel) and convert to RGB
    if image_np.ndim == 2:
        image_np = np.stack((image_np,)*3, axis=-1)
    elif image_np.ndim == 3 and image_np.shape[2] == 1:
        image_np = np.concatenate((image_np,)*3, axis=-1)

    if image_np.dtype != np.float64 and image_np.dtype != np.float32:
        img = image_np.astype(np.float64) / 255.0
    else:
        img = np.clip(image_np, 0, 1)

    h, w = img.shape[:2]
    metadata = {"ddcolor_used": False, "features_applied": []}

    # Validating kwargs
    reference_strength = kwargs.get('reference_strength', 0.9) # FORCE HIGH DEFAULT

    print(f"DEBUG: Colorize Start. Input: {img.shape}, RefImg: {reference_img is not None}, RefStrength: {reference_strength}")

    # Try DDColor first if enabled
    primary_rgb = None
    out_sig_rgb = None # Initialize to avoid UnboundLocalError

    if use_ddcolor and is_ddcolor_available():
        try:
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            colored_bgr = predict_ab_with_ddcolor(img_bgr, input_size=512, model_size='large')
            if colored_bgr is not None:
                primary_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                metadata["ddcolor_used"] = True
                print("✓ Enhanced pipeline using DDColor base")
        except Exception as e:
            print(f"⚠ DDColor failed: {e}, using classic models")
            primary_rgb = None
    
    # Fallback to classic models
    if primary_rgb is None:
        (colorizer_eccv16, colorizer_siggraph17), device = get_models()
        
        img_lab_orig = rgb2lab(img)
        img_l_orig = img_lab_orig[:, :, 0]
        img_small = resize(img, (256, 256), preserve_range=True)
        img_lab_small = rgb2lab(img_small)
        img_l_small = img_lab_small[:, :, 0]

        ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)
        ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)

        ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)
        ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)

        def _lab_to_rgb(l_orig: np.ndarray, ab_up: np.ndarray) -> np.ndarray:
            lab = np.concatenate((l_orig[:, :, np.newaxis], ab_up), axis=2)
            return lab2rgb(lab)

        out_eccv_rgb = _lab_to_rgb(img_l_orig, ab_eccv_up)
        primary_rgb = _lab_to_rgb(img_l_orig, ab_sig_up)
        out_sig_rgb = primary_rgb # Capture SIGGRAPH17 result for fusion
    else:
        # Compute ECCV for comparison
        (colorizer_eccv16, colorizer_siggraph17), _ = get_models()
        img_lab_orig = rgb2lab(img)
        img_l_orig = img_lab_orig[:, :, 0]
        img_small = resize(img, (256, 256), preserve_range=True)
        img_lab_small = rgb2lab(img_small)
        img_l_small = img_lab_small[:, :, 0]
        
        ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)
        ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)
        out_eccv_rgb = lab2rgb(np.concatenate((img_l_orig[:, :, np.newaxis], ab_eccv_up), axis=2))
        
        # If fusion is requested, we also need SIGGRAPH17 result
        if use_ensemble:
            try:
                ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)
                ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)
                out_sig_rgb = lab2rgb(np.concatenate((img_l_orig[:, :, np.newaxis], ab_sig_up), axis=2))
            except Exception as e:
                print(f"Warning: Could not compute SIGGRAPH17 for fusion: {e}")
                out_sig_rgb = out_eccv_rgb # Fallback

    # Apply ensemble fusion if requested
    ensemble_result = primary_rgb
    if use_ensemble:
        try:
            print("Running Smart Fusion...")
            image_uint8 = (img * 255).astype(np.uint8)
            
            # Convert to grayscale for analysis (fix for 2D array error)
            if image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
                gray_input = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray_input = image_uint8
            
            # Ensure siggraph input is available
            siggraph_input = out_sig_rgb
            if siggraph_input is None:
                siggraph_input = primary_rgb if not metadata["ddcolor_used"] else out_eccv_rgb

            ddcolor_input = primary_rgb if metadata["ddcolor_used"] else None
            
            fused_result, weights, characteristics = ensemble_colorization(
                gray_input, 
                out_eccv_rgb, 
                siggraph_input, 
                ddcolor_result=ddcolor_input
            )
            metadata['ensemble_weights'] = weights
            metadata['image_characteristics'] = characteristics
            ensemble_result = fused_result
            metadata['features_applied'].append('ensemble')
            print(f"Fusion complete. Weights: {weights}")
        except Exception as e:
            print(f"⚠ Ensemble fusion failed: {e}")
            metadata['ensemble_error'] = str(e)

            metadata['image_characteristics'] = characteristics
            ensemble_result = fused_result
            metadata['features_applied'].append('ensemble')
        except Exception as e:
            print(f"⚠ Ensemble fusion failed: {e}")
            metadata['ensemble_error'] = str(e)
    elif use_ensemble and metadata["ddcolor_used"]:
        # For DDColor, use Smart Fusion to blend with classic models based on texture
        try:
            (colorizer_eccv16, colorizer_siggraph17), _ = get_models()
            img_lab_orig = rgb2lab(img)
            img_l_orig = img_lab_orig[:, :, 0]
            img_small = resize(img, (256, 256), preserve_range=True)
            img_lab_small = rgb2lab(img_small)
            img_l_small = img_lab_small[:, :, 0]
            
            # Get predictions from classic models
            ab_sig = _predict_ab(colorizer_siggraph17, img_l_small)
            ab_sig_up = resize(ab_sig, (h, w), preserve_range=True)
            sig_rgb = lab2rgb(np.concatenate((img_l_orig[:, :, np.newaxis], ab_sig_up), axis=2))
            
            ab_eccv = _predict_ab(colorizer_eccv16, img_l_small)
            ab_eccv_up = resize(ab_eccv, (h, w), preserve_range=True)
            eccv_rgb = lab2rgb(np.concatenate((img_l_orig[:, :, np.newaxis], ab_eccv_up), axis=2))
            
            # Use Smart Fusion with all 3 models
            fused_result, weights, characteristics = ensemble_colorization(
                (img * 255).astype(np.uint8), 
                eccv_rgb, 
                sig_rgb, 
                ddcolor_result=primary_rgb
            )
            
            metadata['ensemble_weights'] = weights
            metadata['image_characteristics'] = characteristics
            ensemble_result = fused_result
            metadata['features_applied'].append('smart_fusion_ddcolor')
            print(f"✓ Smart Fusion applied: {weights}")
            
        except Exception as e:
            print(f"⚠ Smart Fusion failed: {e}")
            metadata['ensemble_error'] = str(e)
            ensemble_result = primary_rgb
            metadata['features_applied'].append('ddcolor_sig_fusion')
        except Exception as e:
            print(f"⚠ DDColor-SIG fusion failed: {e}")

    # Apply reference guidance if provided
    if reference_img is not None:
        try:
            print(f"DEBUG: Applying Reference Guidance. RefShape: {reference_img.shape}")
            ensemble_result = apply_reference_guided_colorization(
                (img * 255).astype(np.uint8),
                reference_img,
                ensemble_result,
                guidance_strength=reference_strength
            )
            metadata['reference_guided'] = True
            metadata['features_applied'].append('reference_guided')
            print("DEBUG: Reference Guidance Applied.")
        except Exception as e:
            print(f"⚠ Reference guidance failed: {e}")
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
            metadata['features_applied'].append('color_hints')
        except Exception as e:
            print(f"⚠ Color hints failed: {e}")
            metadata['hints_error'] = str(e)

    # Apply style transfer/presets
    if style_type and style_type != 'none':
        try:
            ensemble_result = apply_style_to_colorization(ensemble_result, style_type)
            metadata['style_applied'] = style_type
            metadata['features_applied'].append('style_transfer')
        except Exception as e:
            print(f"⚠ Style transfer failed: {e}")
            metadata['style_error'] = str(e)

    # Blend with original for fine detail preservation
    strength = float(strength)
    out_eccv_rgb = (1 - strength) * img + strength * out_eccv_rgb
    ensemble_result = (1 - strength) * img + strength * ensemble_result
    
    # FINAL STEP: Advanced Post-Processing (The "Secret Sauce")
    # This applies Guided Filter refinement and Adaptive Vibrance
    # We apply this to the final result to ensure maximum quality
    try:
        ensemble_result = enhance_colorization(ensemble_result, img, strength=strength)
        metadata['features_applied'].append('post_processing_enhancement')
    except Exception as e:
        print(f"⚠ Post-processing failed: {e}")

    return out_eccv_rgb, ensemble_result, metadata


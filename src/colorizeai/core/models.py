"""
Model loading and management
"""

import torch
from .colorizers import eccv16, siggraph17

def get_device():
    """Get the best available device (CUDA/CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(device=None):
    """Load and initialize the colorization models"""
    if device is None:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Load models
    colorizer_eccv16 = eccv16(pretrained=True).to(device).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).to(device).eval()
    
    return colorizer_eccv16, colorizer_siggraph17

# Global model instances (will be initialized when needed)
_models = None
_device = None

def get_models():
    """Get the global model instances, loading them if necessary"""
    global _models, _device
    if _models is None:
        _device = get_device()
        _models = load_models(_device)
    return _models, _device

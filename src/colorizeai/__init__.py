"""
ColorizeAI - Advanced Image and Video Colorization Suite

A comprehensive AI-powered colorization system featuring:
- Basic ECCV16 and SIGGRAPH17 model colorization
- Smart Model Fusion with intelligent weighting
- Reference-Guided Colorization
- Interactive Color Hints
- Temporal Consistency for videos
- Cinematic Style Transfer
- Batch processing with progress tracking
- Video caching for instant reprocessing

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.colorization import colorize_highres, colorize_highres_enhanced
from .core.models import load_models, get_device
from .utils.cache import CacheManager, VideoCacheManager

__all__ = [
    "colorize_highres",
    "colorize_highres_enhanced", 
    "load_models",
    "get_device",
    "CacheManager",
    "VideoCacheManager"
]

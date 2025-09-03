# 🎨 ColorizeAI - Complete Image & Video Colorization Suite

A comprehensive AI-powered colorization system that transforms black and white images and videos into vibrant colored versions using state-of-the-art deep learning models.

## 🚀 Features

### **Core Colorization**
- **ECCV16 Model**: First successful deep colorization model (vibrant, saturated colors)
- **SIGGRAPH17 Model**: Improved model with more realistic and spatially coherent results
- **High-resolution processing** preserving original image quality
- **Adjustable colorization strength** for artistic control

### **🔥 5 Unique Enhanced Features**

1. **🧠 Smart Model Fusion**
   - Analyzes image characteristics (texture, contrast, edges)
   - Automatically weights ECCV16 vs SIGGRAPH17 models
   - Spatially-varying fusion for optimal results

2. **🎯 Reference-Guided Colorization**
   - Upload reference images to guide color choices
   - Extracts color palette and applies intelligently
   - Perfect for matching specific color schemes

3. **🖌️ Interactive Color Hints**
   - Add color hints by specifying x,y coordinates and RGB values
   - Smart propagation based on image structure
   - Fine-tune specific regions with precision

4. **⏳ Temporal Consistency (Videos)**
   - Optical flow tracking between frames
   - Reduces flickering in videos
   - Maintains color coherence across time

5. **🎭 Cinematic Style Transfer**
   - Film emulation (Kodak, Fuji, Agfa)
   - Color grading presets (vintage, modern, cinematic)
   - Artistic filters integration

### **⚡ Performance Features**
- **Batch Processing**: Handle multiple images with progress tracking
- **Video Optimization**: Frame skipping, resolution control, fast mode
- **Memory Management**: Efficient processing of large files

## 🏗️ Project Structure

```
ColorizeAI/
├── main.py                    # Main application entry point
├── setup.py                   # Package setup and dependencies
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/colorizeai/           # Main package
│   ├── __init__.py
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── models.py         # Model loading and management
│   │   ├── colorization.py   # Core colorization algorithms
│   │   └── colorizers/       # Pre-trained model implementations
│   ├── features/             # Enhanced features
│   │   ├── __init__.py
│   │   ├── smart_model_fusion.py
│   │   ├── reference_guided_colorization.py
│   │   ├── interactive_color_hints.py
│   │   ├── temporal_consistency.py
│   │   └── style_transfer_colorization.py
│   ├── handlers/             # Request handlers
│   │   └── __init__.py
│   ├── ui/                   # User interface components
│   │   └── __init__.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── metrics.py        # Quality metrics
├── tests/                    # Test files
│   └── test_video_feature.py
├── docs/                     # Documentation
│   ├── ANALYSIS_AND_FIXES.md
│   ├── UNIQUE_FEATURES.md
│   └── guide.txt
├── assets/                   # Sample assets
│   ├── sample_images/
│   └── sample_videos/
├── outputs/                  # Generated outputs
│   └── videos/               # Processed video files
└── flagged/                  # Gradio flagged content
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jatavedreddy/ColorizeAI.git
   cd ColorizeAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Open your browser** and navigate to the displayed URL (typically `http://127.0.0.1:7860`)

### Development Installation

For development with additional tools:
```bash
pip install -e ".[dev]"
```

## 💡 Usage

### **Basic Image Colorization**
1. Navigate to "Basic Single Image" tab
2. Upload a black and white image
3. Adjust colorization strength (0-1)
4. Click "Colorize Image"
5. Compare results with interactive sliders

### **Enhanced Image Colorization**
1. Navigate to "Enhanced Single Image" tab
2. Upload your black and white image
3. Optionally upload a reference image for color guidance
4. Choose a style preset (modern, vintage, cinematic, etc.)
5. Add color hints in JSON format: `[{"x":100,"y":50,"r":255,"g":0,"b":0}]`
6. Enable Smart Model Fusion for best results
7. Click "Enhanced Colorization"

### **Batch Processing**
1. Navigate to "Batch Processing" tab
2. Upload multiple images (JPG, PNG, BMP, TIFF)
3. Set colorization strength
4. Click "Process Batch"
5. Download ZIP file with results

### **Video Colorization**
1. Navigate to "Video Colorization" tab
2. Upload a video file (MP4, AVI, MOV)
3. Configure settings:
   - **Fast Mode**: Recommended for speed (3-5x faster)
   - **Frame Skip**: Process every Nth frame (higher = faster)
   - **Resolution**: Output video resolution
   - **Temporal Consistency**: Enhanced feature for flicker-free results
   - **Style**: Apply cinematic color grading
4. Click "Process Video"
5. **Note**: Each video is processed fresh (no caching layer).

## 🔧 Advanced Features

### **Color Hints Format**
Add precise color guidance using JSON format:
```json
[
  {"x": 100, "y": 50, "r": 255, "g": 0, "b": 0},
  {"x": 200, "y": 150, "r": 0, "g": 255, "b": 0}
]
```

### **Style Presets**
- **Film Emulation**: kodak, fuji, agfa
- **Artistic**: oil_painting, watercolor
- **Color Grading**: vintage, modern, cinematic, pastel, vibrant, cold

### **Performance Tips**
- Use **Fast Mode** for videos longer than 30 seconds
- **Frame Skip 5+** recommended for long videos
- **720p resolution** offers best speed/quality balance
- Videos are processed freshly each run (no caching layer)
- Large images (>1024px) are auto-resized in batch mode for speed

## 📊 Quality Metrics

When ground-truth images are provided, the system computes:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

## 🧠 Technical Details

### **Models Used**
- **ECCV16**: Colorful Image Colorization (Zhang et al., 2016)
- **SIGGRAPH17**: Real-time User-guided Image Colorization (Zhang et al., 2017)

### **Enhanced Algorithms**
- **Smart Fusion**: Texture and contrast analysis for optimal model weighting
- **Reference Guidance**: Color palette extraction and application
- **Color Propagation**: Structure-aware hint spreading
- **Temporal Consistency**: Optical flow-based frame alignment
- **Style Transfer**: Neural style adaptation for colorization

### **System Requirements**
- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU recommended (CPU supported)
- **Memory**: 4GB+ RAM, 8GB+ for large videos
- **Storage**: ~2GB for models


## 🙏 Acknowledgments

- Original ECCV16 and SIGGRAPH17 colorization papers and implementations
- PyTorch and the deep learning community
- Gradio for the excellent UI framework
- All contributors to the open-source libraries used


**⭐ Star this repository if you find it useful!**

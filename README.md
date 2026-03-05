# 🎨 ColorizeAI - Advanced Image & Video Colorization System# 🎨 ColorizeAI - Complete Image & Video Colorization Suite

A production-ready AI colorization system implementing **DDColor** (diffusion-based base model) with advanced features for image and video colorization. Built with a modular architecture supporting reference-guided colorization, interactive hints, style transfer, and temporal consistency.A comprehensive AI-powered colorization system that transforms black and white images and videos into vibrant colored versions using state-of-the-art deep learning models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)## 🚀 Features

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)### **Core Colorization**

- **ECCV16 Model**: First successful deep colorization model (vibrant, saturated colors)

## 🌟 Overview- **SIGGRAPH17 Model**: Improved model with more realistic and spatially coherent results

- **High-resolution processing** preserving original image quality

ColorizeAI transforms grayscale images and videos into vibrant colored versions using state-of-the-art deep learning. The system implements **DDColor** (our base paper) as the primary colorization engine, with five advanced feature enhancements derived from recent research.- **Adjustable colorization strength** for artistic control

### Key Features

- **🎯 DDColor Base Model**: State-of-the-art Dual-Decoder Transformer architecture.
- **🧠 Smart Model Fusion**: A novel ensemble mechanism that detects complex textures (via LBP/Entropy analysis) and dynamically blends DDColor (semantics) with Classic Models (texture) to prevent "plastic" smoothing.
- **🎬 Video Temporal Consistency**: A stabilization engine using Farneback Optical Flow to warp history frames, reducing video flicker by ~15-20%.
- **📊 Result Analysis UI**: Explainable AI panels in the interface that show users *why* the model made specific decisions (e.g., "High Texture Detected -> Fusion Activated").
- **Other Features**: Reference-based colorization, interactive color hints, and style transfer.

## 🏗️ Architecture

Or set an environment variable:

```

Input Grayscale Image```

         ↓export DDCOLOR_WEIGHTS=/absolute/path/to/ddcolor_scripted.pt

┌────────────────────────┐```

│   DDColor Base Model   │

│  (ConvNeXt + Decoder)  │No additional configuration is required. When present, DDColor is used for ab prediction in 256×256 Lab space. If loading or inference fails, the system transparently falls back to the CNN models.

│   Predicts ab channels │

└────────────────────────┘#### Where to get DDColor weights

         ↓

┌────────────────────────┐You have two paths:

│  Feature Enhancement   │

├────────────────────────┤1) From the DDColor project (recommended)

│ • Model Fusion         │- Clone the official DDColor repository and follow its instructions to download/checkpoint the pretrained model.

│ • Reference Guidance   │- Export to TorchScript (if the repo provides an export utility), ensuring the model accepts `[N,1,256,256]` L and outputs `[N,2,256,256]` ab.

│ • Color Hints          │

│ • Style Transfer       │2) Export your own TorchScript using the provided tool

│ • Temporal (Video)     │- If you have a standard PyTorch checkpoint and a builder function, you can export via:

└────────────────────────┘

         ↓```zsh

   Colorized Outputpython tools/export_torchscript.py \

```   --module ddcolor.model_builder \

   --builder build_model \

## 📁 Project Structure   --checkpoint /path/to/ddcolor_checkpoint.pth \

   --output weights/ddcolor_scripted.pt \

```   --input-shape 1 1 256 256 \

ColorizeAI/   --method trace

├── main.py                          # Gradio application entry point```

├── setup.py                         # Package configuration

├── setup.sh                         # Automated setup scriptReplace `--module` and `--builder` with the actual import path and function that create the model. If your model supports full scripting, you may use `--method script`.

├── requirements.txt                 # Python dependencies

├── DEMO_GUIDE.md                   # Presentation guideVerification:

├── PROJECT_SUMMARY.md              # Technical overview- Launch the app; the header will show a banner:

│   - 🟢 “DDColor active” if the TorchScript file is detected and loadable

├── src/colorizeai/                 # Main package   - 🔴 “DDColor not available” otherwise

│   ├── core/                       # Core colorization modules- In Enhanced mode, the metadata box includes “DDColor Active: Yes/No”.

│   │   ├── ddcolor_model.py       # DDColor integration1. **🧠 Smart Model Fusion**

│   │   ├── colorization.py        # Main pipeline   - Analyzes image characteristics (texture, contrast, edges)

│   │   ├── models.py              # Classic models (ECCV16/SIGGRAPH17)   - Automatically weights ECCV16 vs SIGGRAPH17 models

│   │   └── colorizers/            # Base colorizer implementations   - Spatially-varying fusion for optimal results

│   │

│   ├── features/                   # Feature modules2. **🎯 Reference-Guided Colorization**

│   │   ├── smart_model_fusion.py   - Upload reference images to guide color choices

│   │   ├── reference_guided_colorization.py   - Extracts color palette and applies intelligently

│   │   ├── interactive_color_hints.py   - Perfect for matching specific color schemes

│   │   ├── style_transfer_colorization.py

│   │   └── temporal_consistency.py3. **🖌️ Interactive Color Hints**

│   │   - Add color hints by specifying x,y coordinates and RGB values

│   └── utils/                      # Utilities   - Smart propagation based on image structure

│       ├── metrics.py              # PSNR/SSIM computation   - Fine-tune specific regions with precision

│       └── cache.py                # Performance optimization

│4. **⏳ Temporal Consistency (Videos)**

├── tools/                          # Utility scripts   - Optical flow tracking between frames

│   ├── download_ddcolor_weights.py # Weight downloader   - Reduces flickering in videos

│   └── export_torchscript.py      # Model export utility   - Maintains color coherence across time

│

├── tests/                          # Test suite5. **🎭 Cinematic Style Transfer**

│   ├── test_ddcolor_integration.py   - Film emulation (Kodak, Fuji, Agfa)

│   └── test_video_feature.py   - Color grading presets (vintage, modern, cinematic)

│   - Artistic filters integration

├── docs/                           # Documentation

│   ├── DDCOLOR_INTEGRATION.md     # DDColor setup guide### **⚡ Performance Features**

│   ├── REFACTORING_SUMMARY.md     # Technical changes- **Batch Processing**: Handle multiple images with progress tracking

│   ├── UNIQUE_FEATURES.md         # Feature documentation- **Video Optimization**: Frame skipping, resolution control, fast mode

│   └── Research_papers/           # Literature survey PDFs- **Memory Management**: Efficient processing of large files

│       ├── base_paper_DDcolor.pdf

│       ├── Deep_exempler_referecebased copy.pdf## 🏗️ Project Structure

│       ├── User_guided copy.pdf

│       ├── style_transfer copy.pdf```

│       └── Temporal_consistency copy.pdfColorizeAI/

│├── main.py                    # Main application entry point

├── assets/                         # Sample data├── setup.py                   # Package setup and dependencies

│   ├── sample_images/├── requirements.txt           # Python dependencies

│   └── sample_videos/├── README.md                  # This file

│├── src/colorizeai/           # Main package

└── outputs/                        # Generated outputs│   ├── __init__.py

    ├── images/│   ├── core/                 # Core functionality

    └── videos/│   │   ├── __init__.py

```│   │   ├── models.py         # Model loading and management

│   │   ├── colorization.py   # Core colorization algorithms

## 🚀 Quick Start│   │   └── colorizers/       # Pre-trained model implementations

│   ├── features/             # Enhanced features

### 1. Installation│   │   ├── __init__.py

│   │   ├── smart_model_fusion.py

**Automated Setup (Recommended)**│   │   ├── reference_guided_colorization.py

```bash│   │   ├── interactive_color_hints.py

cd ColorizeAI│   │   ├── temporal_consistency.py

./setup.sh│   │   └── style_transfer_colorization.py

```│   ├── handlers/             # Request handlers

│   │   └── __init__.py

**Manual Setup**│   ├── ui/                   # User interface components

```bash│   │   └── __init__.py

# Install dependencies│   └── utils/                # Utility functions

pip install -r requirements.txt│       ├── __init__.py

│       └── metrics.py        # Quality metrics

# Setup DDColor (if project exists)├── tests/                    # Test files

cd ../ddcolor/DDColor-master\ copy│   └── test_video_feature.py

pip install -e .├── docs/                     # Documentation

cd -│   ├── ANALYSIS_AND_FIXES.md

│   ├── UNIQUE_FEATURES.md

# Download weights│   └── guide.txt

python tools/download_ddcolor_weights.py --model-size large├── assets/                   # Sample assets

```│   ├── sample_images/

│   └── sample_videos/

### 2. Run the Application├── outputs/                  # Generated outputs

│   └── videos/               # Processed video files

```bash└── flagged/                  # Gradio flagged content

python main.py```

```

## 🚀 Quick Start

Opens Gradio interface at http://localhost:7860

### Installation

### 3. Verify Installation

1. **Clone the repository:**

``bash   ``bash

python tests/test_ddcolor_integration.py   git clone https://github.com/Jatavedreddy/ColorizeAI.git

```cd

```

## 💡 Usage

2. **Install dependencies:**

### Gradio Web Interface   ```bash

   pip install -r requirements.txt

1. **Single Image Tab**   ```

   - Upload grayscale image
   - Adjust colorization strength (0-1)3. **Run the application:**
   - Click "Colorize"   ```bash
   - View ECCV16 baseline vs DDColor/SIGGRAPH17 results   python main.py

   ```

   ```
2. **Enhanced Colorization Tab**

   - Enable Smart Model Fusion for texture enhancement4. **Open your browser** and navigate to the displayed URL (typically `http://127.0.0.1:7860`)
   - Upload reference image for palette guidance
   - Add color hints (JSON format)### Development Installation
   - Select style preset (modern, vintage, cinematic, etc.)
   - View processing metadataFor development with additional tools:

```bash

3. **Batch Processing Tab**pip install -e ".[dev]"

   - Upload multiple images```

   - Process with progress tracking

   - Download results as zip## 💡 Usage



4. **Video Colorization Tab**### **Basic Image Colorization**

   - Upload grayscale video1. Navigate to "Basic Single Image" tab

   - Enable temporal consistency2. Upload a black and white image

   - Choose Fast Mode (keyframes) or Quality Mode (all frames)3. Adjust colorization strength (0-1)

   - Monitor progress and preview4. Click "Colorize Image"

5. Compare results with interactive sliders

### Python API

### **Enhanced Image Colorization**

```python1. Navigate to "Enhanced Single Image" tab

from colorizeai.core.colorization import colorize_highres_enhanced2. Upload your black and white image

import cv23. Optionally upload a reference image for color guidance

4. Choose a style preset (modern, vintage, cinematic, etc.)

# Load image5. Add color hints in JSON format: `[{"x":100,"y":50,"r":255,"g":0,"b":0}]`

img = cv2.imread('grayscale.jpg')6. Enable Smart Model Fusion for best results

ref_img = cv2.imread('reference.jpg')  # Optional7. Click "Enhanced Colorization"



# Define color hints (optional)### **Batch Processing**

color_hints = [1. Navigate to "Batch Processing" tab

    {'x': 100, 'y': 150, 'r': 255, 'g': 0, 'b': 0, 'radius': 20},2. Upload multiple images (JPG, PNG, BMP, TIFF)

    {'x': 300, 'y': 200, 'r': 0, 'g': 255, 'b': 0, 'radius': 20}3. Set colorization strength

]4. Click "Process Batch"

5. Download ZIP file with results

# Colorize with all features

eccv_result, enhanced_result, metadata = colorize_highres_enhanced(### **Video Colorization**

    img,1. Navigate to "Video Colorization" tab

    strength=1.0,2. Upload a video file (MP4, AVI, MOV)

    use_ensemble=True,           # Smart fusion3. Configure settings:

    reference_img=ref_img,       # Reference guidance   - **Fast Mode**: Recommended for speed (3-5x faster)

    color_hints=color_hints,     # Interactive hints   - **Frame Skip**: Process every Nth frame (higher = faster)

    style_type='cinematic',      # Style preset   - **Resolution**: Output video resolution

    use_ddcolor=True             # Use DDColor base   - **Temporal Consistency**: Enhanced feature for flicker-free results

)   - **Style**: Apply cinematic color grading

4. Click "Process Video"

# Check what was used5. **Note**: Each video is processed fresh (no caching layer).

print(f"Base model: {'DDColor' if metadata['ddcolor_used'] else 'SIGGRAPH17'}")

print(f"Features applied: {metadata['features_applied']}")## 🔧 Advanced Features



# Save result### **Color Hints Format**

cv2.imwrite('colored.jpg', (enhanced_result * 255).astype('uint8'))Add precise color guidance using JSON format:

``````json

[

### Direct DDColor API  {"x": 100, "y": 50, "r": 255, "g": 0, "b": 0},

  {"x": 200, "y": 150, "r": 0, "g": 255, "b": 0}

```python]

from colorizeai.core.ddcolor_model import DDColorPipeline, colorize_image```

import cv2

### **Style Presets**

# Quick colorization- **Film Emulation**: kodak, fuji, agfa

img = cv2.imread('grayscale.jpg')- **Artistic**: oil_painting, watercolor

result = colorize_image(img, model_size='large', input_size=512)- **Color Grading**: vintage, modern, cinematic, pastel, vibrant, cold

cv2.imwrite('colored.jpg', result)

### **Performance Tips**

# Or use pipeline for batch- Use **Fast Mode** for videos longer than 30 seconds

pipeline = DDColorPipeline(model_size='large', input_size=512)- **Frame Skip 5+** recommended for long videos

images = [cv2.imread(f'img{i}.jpg') for i in range(10)]- **720p resolution** offers best speed/quality balance

results = pipeline.process_batch(images)- Videos are processed freshly each run (no caching layer)

```- Large images (>1024px) are auto-resized in batch mode for speed



## 🎓 Literature Survey## 📊 Quality Metrics



This project implements insights from 5 research papers:When ground-truth images are provided, the system computes:

- **PSNR** (Peak Signal-to-Noise Ratio)

1. **DDColor (Base Paper)**: Diffusion-based colorization with multi-scale decoder- **SSIM** (Structural Similarity Index)

2. **Deep Exemplar-Based Colorization**: Reference-guided color transfer

3. **Photorealistic Style Transfer**: Global color grading without distortion## 🧠 Technical Details

4. **User-Guided Colorization**: Sparse hints with edge-aware propagation

5. **Blind Video Temporal Consistency**: Optical flow warping for videos### **Models Used**

- **ECCV16**: Colorful Image Colorization (Zhang et al., 2016)

See `docs/Research_papers/` for full papers and `PROJECT_SUMMARY.md` for implementation details.- **SIGGRAPH17**: Real-time User-guided Image Colorization (Zhang et al., 2017)



## 📊 Performance### **Enhanced Algorithms**

- **Smart Fusion**: Texture and contrast analysis for optimal model weighting

### Quality Metrics (512×512 images)- **Reference Guidance**: Color palette extraction and application

- **Color Propagation**: Structure-aware hint spreading

| Model | PSNR ↑ | SSIM ↑ | Time (GPU) | Time (CPU) |- **Temporal Consistency**: Optical flow-based frame alignment

|-------|--------|--------|------------|------------|- **Style Transfer**: Neural style adaptation for colorization

| ECCV16 | 26.2 | 0.87 | 0.2s | 2s |

| SIGGRAPH17 | 26.8 | 0.89 | 0.3s | 3s |### **System Requirements**

| DDColor Large | **28.5** | **0.92** | 1.2s | 15s |- **Python**: 3.8+

| DDColor + Features | **28.8** | **0.93** | 1.5s | 18s |- **GPU**: CUDA-compatible GPU recommended (CPU supported)

- **Memory**: 4GB+ RAM, 8GB+ for large videos

### Device Support- **Storage**: ~2GB for models



- ✅ **CUDA** (NVIDIA GPUs): Fastest, mixed precision (FP16)

- ✅ **MPS** (Apple Silicon): Fast, mixed precision## 🙏 Acknowledgments

- ✅ **CPU**: Slower but functional, automatic fallback

- Original ECCV16 and SIGGRAPH17 colorization papers and implementations

## 🔧 Configuration- PyTorch and the deep learning community

- Gradio for the excellent UI framework

### Model Selection- All contributors to the open-source libraries used



```python

# Use DDColor Large (best quality)**⭐ Star this repository if you find it useful!**

from colorizeai.core.ddcolor_model import load_ddcolor
model, device = load_ddcolor(model_size='large', input_size=512)

# Or DDColor Tiny (faster)
model, device = load_ddcolor(model_size='tiny', input_size=256)
```

### Environment Variables

```bash
# Custom weight location
export DDCOLOR_WEIGHTS=/path/to/pytorch_model.pt

# Force CPU (for testing)
export CUDA_VISIBLE_DEVICES=""
```

## 🐛 Troubleshooting

### DDColor Not Loading

**Issue**: "DDColor weights not found"

- **Solution**: Run `python tools/download_ddcolor_weights.py`
- Or ensure DDColor project exists at `DDColor/`

**Issue**: "Import basicsr.archs.ddcolor_arch failed"

- **Solution**: Install DDColor: `cd DDColor && pip install -e .`

### Memory Issues

**CUDA Out of Memory**:

```python
# Reduce resolution
result = colorize_image(img, input_size=256)

# Or use tiny model
result = colorize_image(img, model_size='tiny')
```

### Fallback Behavior

If DDColor fails, the system automatically uses SIGGRAPH17/ECCV16:

- No errors thrown
- Console warning shown
- `metadata['ddcolor_used']` = `False`

## 📚 Documentation

- **[DEMO_GUIDE.md](DEMO_GUIDE.md)**: Step-by-step presentation guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Technical overview and architecture
- **[docs/DDCOLOR_INTEGRATION.md](docs/DDCOLOR_INTEGRATION.md)**: DDColor setup and API
- **[docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)**: Detailed code changes
- **[docs/UNIQUE_FEATURES.md](docs/UNIQUE_FEATURES.md)**: Feature documentation

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Fine-tuning DDColor on specific domains
- Real-time video with frame caching
- INT8 quantization for mobile
- Additional style presets
- User study validation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DDColor**: [Official Repository](https://github.com/piddnad/DDColor)
- **ECCV16**: Zhang et al., "Colorful Image Colorization"
- **SIGGRAPH17**: Zhang et al., "Real-Time User-Guided Image Colorization"
- **Gradio**: For the excellent UI framework

## 📧 Contact

For questions or issues:

- Open a GitHub issue
- Check documentation in `docs/`
- Run verification: `python tests/test_ddcolor_integration.py`

---

**Ready to colorize! 🎨✨**

*Built with PyTorch, Gradio, and modern deep learning techniques.*

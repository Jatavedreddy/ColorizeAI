#!/bin/bash
# ColorizeAI Startup Script
# Runs the application with the correct conda environment

set -e

echo "=========================================="
echo "🎨 ColorizeAI Launcher"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda not found. Please install Anaconda/Miniconda."
    exit 1
fi

# Check if colorize environment exists
if ! conda env list | grep -q "^colorize "; then
    echo "❌ Error: 'colorize' environment not found."
    echo ""
    echo "Please create it with:"
    echo "  conda create -n colorize python=3.10"
    echo "  conda activate colorize"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✓ Using conda environment: colorize"
echo "✓ Python: $(conda run -n colorize python --version)"
echo ""

# Check if DDColor weights exist
WEIGHTS_PATH="DDColor/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
if [ -f "$WEIGHTS_PATH" ]; then
    SIZE=$(du -h "$WEIGHTS_PATH" | cut -f1)
    echo "✓ DDColor weights found (${SIZE})"
else
    echo "⚠ Warning: DDColor weights not found at:"
    echo "  $WEIGHTS_PATH"
    echo ""
    echo "The app will work but DDColor will be inactive."
    echo "To download weights, run:"
    echo "  python tools/download_ddcolor_weights.py --model-size large"
    echo ""
fi

echo ""
echo "🚀 Starting ColorizeAI..."
echo "   URL: http://127.0.0.1:7860"
echo "   Using: main_fixed.py (Gradio 5.x compatible)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Activate environment and run the fixed version
conda run -n colorize python main_fixed.py

#!/bin/bash
# Setup script for ColorizeAI with DDColor integration

set -e

echo "======================================"
echo "ColorizeAI + DDColor Setup"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the ColorizeAI directory"
    exit 1
fi

echo "Step 1: Installing ColorizeAI dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Setting up DDColor project..."

# Check if DDColor project exists
DDCOLOR_PATH="../ddcolor/DDColor-master copy"
if [ -d "$DDCOLOR_PATH" ]; then
    echo "✓ DDColor project found at $DDCOLOR_PATH"
    
    # Install DDColor dependencies
    echo "Installing DDColor dependencies..."
    cd "$DDCOLOR_PATH"
    pip install -e .
    cd - > /dev/null
    
    echo "✓ DDColor dependencies installed"
else
    echo "⚠ DDColor project not found at $DDCOLOR_PATH"
    echo "Please ensure the DDColor project is placed at the correct location"
    echo "Expected location: $(realpath ../ddcolor/DDColor-master\ copy)"
fi

echo ""
echo "Step 3: Checking for DDColor weights..."

WEIGHTS_PATH="$DDCOLOR_PATH/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
if [ -f "$WEIGHTS_PATH" ]; then
    echo "✓ DDColor weights found"
    SIZE=$(du -h "$WEIGHTS_PATH" | cut -f1)
    echo "  File size: $SIZE"
else
    echo "⚠ DDColor weights not found"
    echo ""
    echo "Would you like to download them now? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        python tools/download_ddcolor_weights.py --model-size large
    else
        echo "You can download weights later with:"
        echo "  python tools/download_ddcolor_weights.py --model-size large"
    fi
fi

echo ""
echo "Step 4: Verifying installation..."

python -c "
from colorizeai.core.ddcolor_model import is_ddcolor_available, get_ddcolor
import sys

model, device, error = get_ddcolor()
if model:
    print(f'✓ DDColor loaded successfully on {device}')
    sys.exit(0)
else:
    print(f'⚠ DDColor not available: {error}')
    print('The system will fall back to ECCV16/SIGGRAPH17 models')
    sys.exit(0)  # Not a fatal error
"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To start ColorizeAI:"
echo "  python main.py"
echo ""
echo "The Gradio interface will open in your browser."
echo ""
echo "For more information, see:"
echo "  - docs/DDCOLOR_INTEGRATION.md"
echo "  - docs/UNIQUE_FEATURES.md"
echo "  - README.md"
echo ""

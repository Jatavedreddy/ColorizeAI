#!/bin/bash
# Setup script for ColorizeAI with DDColor integration

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root (parent of scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "======================================"
echo "ColorizeAI + DDColor Setup"
echo "======================================"
echo "Working directory: $PWD"
echo ""

# Check if we're in the right directory (check for main.py)
if [ ! -f "main.py" ]; then
    echo "Error: Could not find main.py in project root."
    exit 1
fi

echo "Step 1: Installing ColorizeAI dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 2: Setting up DDColor..."
if [ -d "DDColor" ]; then
    echo "DDColor directory found."
    # Install DDColor dependencies if needed, or just ensure it's there
    # Assuming requirements.txt covers it or it's self-contained
else
    echo "Warning: DDColor directory not found."
fi

echo ""
echo "Setup complete!"
echo "Run with: ./scripts/run.sh"

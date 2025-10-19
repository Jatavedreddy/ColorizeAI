#!/usr/bin/env python3
"""
DDColor Model Weight Downloader

This script downloads the DDColor pretrained weights from Hugging Face
and places them in the correct location for the ColorizeAI project.
"""

import os
import sys
from pathlib import Path
import urllib.request
import shutil

def download_file(url: str, dest_path: Path, desc: str = ""):
    """Download a file with progress indication"""
    print(f"Downloading {desc}...")
    print(f"URL: {url}")
    print(f"Destination: {dest_path}")
    
    # Create parent directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r{desc}: {percent}% complete")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print(f"\n✓ Downloaded successfully to {dest_path}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_ddcolor_weights(model_size: str = 'large'):
    """
    Download DDColor model weights.
    
    Args:
        model_size: 'tiny' or 'large'
    """
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Target location in DDColor project folder
    ddcolor_project = project_root.parent / "ddcolor" / "DDColor-master copy"
    weights_dir = ddcolor_project / "modelscope" / "damo" / "cv_ddcolor_image-colorization"
    weights_file = weights_dir / "pytorch_model.pt"
    
    # Alternative: also save to ColorizeAI weights folder
    alt_weights_dir = project_root / "weights"
    alt_weights_file = alt_weights_dir / "ddcolor.pt"
    
    print("=" * 60)
    print("DDColor Model Weight Downloader")
    print("=" * 60)
    print(f"Model size: {model_size}")
    print(f"Target directory: {weights_dir}")
    print()
    
    # Check if weights already exist
    if weights_file.exists():
        print(f"✓ Weights already exist at {weights_file}")
        size_mb = weights_file.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
        
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    # Hugging Face model URLs
    # Note: Update these URLs with actual DDColor Hugging Face model repository
    if model_size == 'large':
        url = "https://huggingface.co/piddnad/ddcolor-models/resolve/main/pytorch_model.pt"
        desc = "DDColor Large Model"
    else:
        url = "https://huggingface.co/piddnad/ddcolor-models/resolve/main/pytorch_model_tiny.pt"
        desc = "DDColor Tiny Model"
    
    print(f"\nAttempting to download {desc}...")
    print("Note: This may take several minutes depending on your connection.")
    print("Model size is approximately 500-1500 MB.")
    print()
    
    # Try downloading
    success = download_file(url, weights_file, desc)
    
    if success:
        # Also copy to alternative location
        print(f"\nCopying to alternative location: {alt_weights_file}")
        alt_weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_file, alt_weights_file)
        print(f"✓ Copy complete")
        
        print("\n" + "=" * 60)
        print("✓ DDColor weights downloaded successfully!")
        print("=" * 60)
        print("\nYou can now run ColorizeAI with DDColor support:")
        print("  python main.py")
        print()
        return True
    else:
        print("\n" + "=" * 60)
        print("✗ Download failed")
        print("=" * 60)
        print("\nManual download instructions:")
        print(f"1. Visit: {url}")
        print(f"2. Download the file manually")
        print(f"3. Place it at: {weights_file}")
        print()
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DDColor model weights")
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['tiny', 'large'],
        default='large',
        help="Model size to download (default: large)"
    )
    
    args = parser.parse_args()
    
    success = download_ddcolor_weights(args.model_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

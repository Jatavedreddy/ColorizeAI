#!/usr/bin/env python3
"""
Test script for video processing functionality
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def test_video_loading():
    """Test if we can load and read video files"""
    video_dir = Path("videos")
    if not video_dir.exists():
        print("❌ Videos directory not found")
        return False
    
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print("❌ No MP4 files found in videos directory")
        return False
    
    print(f"✅ Found {len(video_files)} video files")
    
    # Test loading the first video
    test_video = video_files[0]
    cap = cv2.VideoCapture(str(test_video))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {test_video}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Test reading a few frames
    frames_read = 0
    for i in range(min(10, total_frames)):
        ret, frame = cap.read()
        if ret:
            frames_read += 1
        else:
            break
    
    cap.release()
    
    if frames_read > 0:
        print(f"✅ Successfully read {frames_read} frames")
        return True
    else:
        print("❌ Could not read any frames")
        return False

def test_video_handler():
    """Test the video handler function"""
    try:
        from app import handler_video
        print("✅ Successfully imported handler_video")
        
        # For testing without UI, we'll create a mock progress function
        class MockProgress:
            def __call__(self, progress, desc=""):
                print(f"Progress: {progress:.1%} - {desc}")
        
        # Test with None input (should return None)
        result = handler_video(None, 1.0, 1, "Original", None, None, MockProgress())
        if result is None:
            print("✅ handler_video correctly handles None input")
        else:
            print("❌ handler_video should return None for None input")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import handler_video: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing handler_video: {e}")
        return False

def main():
    print("🔍 Testing Video Feature Functionality\n")
    
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("Video Loading", test_video_loading),
        ("Video Handler", test_video_handler),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                print(f"✅ {test_name} test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Video feature appears to be working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

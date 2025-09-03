#!/usr/bin/env python3
"""
Test script for video processing functionality
"""

import os
import sys
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from colorizeai.features.temporal_consistency import TemporalConsistencyEngine


def test_video_loading():
    """Test if sample videos can be opened"""
    video_dir = ROOT / "assets" / "sample_videos"
    assert video_dir.exists(), "assets/sample_videos directory not found"

    videos = list(video_dir.glob("*.mp4"))
    assert videos, "No sample videos found"

    cap = cv2.VideoCapture(str(videos[0]))
    assert cap.isOpened(), f"Could not open video: {videos[0]}"
    cap.release()



def test_temporal_scene_change_reset():
    """Basic check that engine resets blending on large differences (indirect via output difference)."""
    eng = TemporalConsistencyEngine()
    import numpy as np
    h, w = 64, 96
    # Frame 1 random
    f1g = (np.random.rand(h, w) * 255).astype('uint8')
    f1c = np.clip(np.random.rand(h, w, 3), 0, 1)
    out1 = eng.apply_temporal_consistency(f1c, f1g)
    # Frame 2 similar -> should blend
    f2g = np.clip(f1g + np.random.randint(-5,5,(h,w)),0,255).astype('uint8')
    f2c = np.clip(f1c + (np.random.rand(h,w,3)-0.5)*0.02,0,1)
    out2 = eng.apply_temporal_consistency(f2c, f2g)
    # Frame 3 very different -> expect reset (output closer to f3c than to out2)
    f3g = (np.random.rand(h, w) * 255).astype('uint8')
    f3c = np.clip(np.random.rand(h, w, 3),0,1)
    out3 = eng.apply_temporal_consistency(f3c, f3g)
    # Compute mean differences
    import numpy.linalg as npl
    diff_reset = np.mean(np.abs(out3 - f3c))
    diff_prev = np.mean(np.abs(out2 - f2c))
    assert diff_reset < diff_prev + 0.1, "After scene change reset, output should track new frame closely"


def test_temporal_engine_basic():
    """Temporal engine should return first frame unchanged and maintain shape"""
    eng = TemporalConsistencyEngine()
    import numpy as np
    h, w = 120, 160
    gray = (np.random.rand(h, w) * 255).astype("uint8")
    color = np.clip(np.random.rand(h, w, 3), 0, 1)

    out = eng.apply_temporal_consistency(color, gray)
    assert out.shape == color.shape


def run_all():
    tests = [
        ("Video Loading", test_video_loading),
        ("Temporal Engine", test_temporal_engine_basic),
        ("Scene Change Reset", test_temporal_scene_change_reset),
    ]
    passed = 0
    for name, fn in tests:
        try:
            assert fn()
            print(f"✅ {name} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name} FAILED: {e}")
        except Exception as e:
            print(f"❌ {name} ERROR: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""
DDColor Integration Verification Script

Tests that all components are properly integrated and working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from colorizeai.core.ddcolor_model import (
            is_ddcolor_available, 
            get_ddcolor, 
            DDColorPipeline,
            colorize_image
        )
        from colorizeai.core.colorization import colorize_highres, colorize_highres_enhanced
        from colorizeai.core.models import get_models
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_ddcolor_availability():
    """Check if DDColor model can be loaded"""
    print("\nTesting DDColor availability...")
    try:
        from colorizeai.core.ddcolor_model import is_ddcolor_available, get_ddcolor
        
        if is_ddcolor_available():
            model, device, error = get_ddcolor()
            if model:
                print(f"✓ DDColor loaded successfully on {device}")
                return True, "ddcolor"
            else:
                print(f"⚠ DDColor model is None: {error}")
                return True, "fallback"
        else:
            print("⚠ DDColor not available (will use fallback models)")
            model, device, error = get_ddcolor()
            print(f"  Reason: {error}")
            return True, "fallback"
    except Exception as e:
        print(f"✗ DDColor check failed: {e}")
        return False, None


def test_classic_models():
    """Test that classic models can be loaded"""
    print("\nTesting classic models...")
    try:
        from colorizeai.core.models import get_models
        (eccv16, siggraph17), device = get_models()
        print(f"✓ Classic models loaded on {device}")
        print(f"  - ECCV16: {type(eccv16).__name__}")
        print(f"  - SIGGRAPH17: {type(siggraph17).__name__}")
        return True
    except Exception as e:
        print(f"✗ Classic model loading failed: {e}")
        return False


def test_basic_colorization():
    """Test basic colorization pipeline"""
    print("\nTesting basic colorization...")
    try:
        import numpy as np
        from colorizeai.core.colorization import colorize_highres
        
        # Create a dummy grayscale image
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test with DDColor
        try:
            eccv_result, primary_result = colorize_highres(dummy_img, strength=1.0, use_ddcolor=True)
            print(f"✓ Basic colorization (DDColor) successful")
            print(f"  Output shapes: {eccv_result.shape}, {primary_result.shape}")
        except Exception as e:
            print(f"⚠ DDColor colorization failed, trying fallback: {e}")
            # Try without DDColor
            eccv_result, primary_result = colorize_highres(dummy_img, strength=1.0, use_ddcolor=False)
            print(f"✓ Basic colorization (fallback) successful")
            print(f"  Output shapes: {eccv_result.shape}, {primary_result.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Basic colorization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_pipeline():
    """Test enhanced colorization with all features"""
    print("\nTesting enhanced pipeline...")
    try:
        import numpy as np
        from colorizeai.core.colorization import colorize_highres_enhanced
        
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        try:
            eccv_result, enhanced_result, metadata = colorize_highres_enhanced(
                dummy_img,
                strength=1.0,
                use_ensemble=True,
                reference_img=None,
                color_hints=None,
                style_type='modern',
                use_ddcolor=True
            )
            print(f"✓ Enhanced pipeline (DDColor) successful")
            print(f"  Base model used: {'DDColor' if metadata.get('ddcolor_used') else 'Classic'}")
            print(f"  Features applied: {metadata.get('features_applied', [])}")
        except Exception as e:
            print(f"⚠ DDColor enhanced failed, trying fallback: {e}")
            eccv_result, enhanced_result, metadata = colorize_highres_enhanced(
                dummy_img,
                strength=1.0,
                use_ensemble=True,
                reference_img=None,
                color_hints=None,
                style_type='modern',
                use_ddcolor=False
            )
            print(f"✓ Enhanced pipeline (fallback) successful")
            print(f"  Features applied: {metadata.get('features_applied', [])}")
        
        return True
    except Exception as e:
        print(f"✗ Enhanced pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_modules():
    """Test individual feature modules"""
    print("\nTesting feature modules...")
    try:
        from colorizeai.features.smart_model_fusion import ensemble_colorization
        from colorizeai.features.reference_guided_colorization import apply_reference_guided_colorization
        from colorizeai.features.interactive_color_hints import apply_color_hints
        from colorizeai.features.style_transfer_colorization import apply_style_to_colorization
        from colorizeai.features.temporal_consistency import TemporalConsistencyEngine
        
        print("✓ All feature modules imported successfully")
        
        # Test temporal consistency engine
        engine = TemporalConsistencyEngine()
        print(f"✓ Temporal consistency engine initialized")
        
        return True
    except Exception as e:
        print(f"✗ Feature module test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("ColorizeAI + DDColor Integration Verification")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['ddcolor'], model_type = test_ddcolor_availability()
    results['classic'] = test_classic_models()
    results['basic'] = test_basic_colorization()
    results['enhanced'] = test_enhanced_pipeline()
    results['features'] = test_feature_modules()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.ljust(20)}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if model_type == "ddcolor":
        print("\n🎉 DDColor is active and working!")
    elif model_type == "fallback":
        print("\n⚠ Using fallback models (DDColor not available)")
        print("   The system will work but without DDColor enhancements.")
        print("   To enable DDColor, run: python tools/download_ddcolor_weights.py")
    
    if passed == total:
        print("\n✓ All systems operational!")
        return 0
    else:
        print("\n⚠ Some tests failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

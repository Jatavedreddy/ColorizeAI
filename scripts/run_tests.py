#!/usr/bin/env python3
"""
ColorizeAI Test Runner

Runs all tests and generates comprehensive reports.
Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --accuracy         # Run only accuracy tests
    python run_tests.py --performance      # Run only performance tests
    python run_tests.py --report           # Generate detailed report
"""

import sys
import argparse
from pathlib import Path

# Add src and project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

def run_all_tests(verbose=True):
    """Run all test suites"""
    print("="*70)
    print("🧪 ColorizeAI Test Suite")
    print("="*70)
    print()
    
    all_passed = True
    
    # Run accuracy tests
    print("\n📊 Running Accuracy Tests...")
    print("-"*70)
    try:
        from tests.test_accuracy import run_tests as run_accuracy_tests
        passed = run_accuracy_tests()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"❌ Error running accuracy tests: {e}")
        all_passed = False
    
    # Run performance tests
    print("\n\n⚡ Running Performance Tests...")
    print("-"*70)
    try:
        from tests.test_performance import run_performance_tests
        passed = run_performance_tests()
        all_passed = all_passed and passed
    except ImportError as e:
        print(f"⚠️  Skipping performance tests (missing dependency: {e})")
        print("   Install with: pip install psutil")
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS")
    print("="*70)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED")
    print("="*70)
    
    return all_passed


def run_accuracy_only():
    """Run only accuracy tests"""
    from tests.test_accuracy import run_tests
    return run_tests()


def run_performance_only():
    """Run only performance tests"""
    try:
        from tests.test_performance import run_performance_tests
        return run_performance_tests()
    except ImportError:
        print("❌ Performance tests require psutil: pip install psutil")
        return False


def generate_report():
    """Generate detailed test report"""
    print("="*70)
    print("📋 ColorizeAI Test Report")
    print("="*70)
    print()
    
    # System info
    import platform
    import torch
    from colorizeai.core.ddcolor_model import is_ddcolor_available
    
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print(f"   DDColor: {'✅ Available' if is_ddcolor_available() else '❌ Not Available'}")
    print()
    
    # Run tests
    success = run_all_tests(verbose=True)
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Run ColorizeAI tests')
    parser.add_argument('--accuracy', action='store_true', help='Run only accuracy tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance tests')
    parser.add_argument('--report', action='store_true', help='Generate detailed report')
    
    args = parser.parse_args()
    
    if args.accuracy:
        success = run_accuracy_only()
    elif args.performance:
        success = run_performance_only()
    elif args.report:
        success = generate_report()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

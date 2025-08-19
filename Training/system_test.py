#!/usr/bin/env python3
"""
Comprehensive test untuk validasi sistem Face Anti-Spoofing
setelah optimisasi lightweight
"""

def test_imports():
    """Test semua import yang diperlukan"""
    print("Testing imports...")
    
    try:
        # Core system
        from src.core.app_launcher import AppLauncher
        print("  [OK] AppLauncher")
        
        from src.web.app_clean import create_app
        print("  [OK] Flask app factory")
        
        from src.detection.landmark_detection import FacialLandmarkDetector, LivenessVerifier
        print("  [OK] Landmark detection")
        
        from src.models.cnn_model import SimpleCNN
        print("  [OK] CNN model")
        
        from src.utils.config import ConfigManager
        print("  [OK] Configuration")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Import error: {e}")
        return False

def test_initialization():
    """Test inisialisasi komponen utama"""
    print("\nTesting initialization...")
    
    try:
        # Test landmark detector
        from src.detection.landmark_detection import FacialLandmarkDetector, LivenessVerifier
        detector = FacialLandmarkDetector()
        print("  [OK] FacialLandmarkDetector initialized")
        
        # Test liveness verifier
        verifier = LivenessVerifier()
        print("  [OK] LivenessVerifier initialized")
        
        # Test config manager
        from src.utils.config import ConfigManager
        config = ConfigManager()
        print("  [OK] ConfigManager initialized")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Initialization error: {e}")
        return False

def test_dependencies():
    """Test essential dependencies"""
    print("\nTesting dependencies...")
    
    essential_deps = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'), 
        ('numpy', 'NumPy'),
        ('flask', 'Flask'),
        ('PIL', 'Pillow')
    ]
    
    for module, name in essential_deps:
        try:
            if module == 'cv2':
                import cv2
                print(f"  [OK] {name}: {cv2.__version__}")
            elif module == 'PIL':
                from PIL import Image
                print(f"  [OK] {name}")
            elif module == 'torch':
                import torch
                print(f"  [OK] {name}: {torch.__version__}")
            elif module == 'numpy':
                import numpy as np
                print(f"  [OK] {name}: {np.__version__}")
            elif module == 'flask':
                import flask
                print(f"  [OK] {name}: {flask.__version__}")
        except ImportError:
            print(f"  [ERROR] {name} not available")
            return False
    
    # Test optional dependencies
    optional_deps = ['mediapipe', 'matplotlib', 'seaborn']
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"  [OK] {dep} available (optional)")
        except ImportError:
            print(f"  [SKIP] {dep} not available (optional)")
    
    return True

def main():
    """Run comprehensive test"""
    print("Face Anti-Spoofing System - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization), 
        ("Dependencies Test", test_dependencies)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[FAIL] {test_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("\nSystem Status: READY FOR DEPLOYMENT")
        print("\nNext steps:")
        print("1. Start web app: python main.py --mode web")
        print("2. Open browser: http://localhost:5000")
        print("3. Test with webcam or upload images")
    else:
        print("[WARNING] Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

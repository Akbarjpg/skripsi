#!/usr/bin/env python3
"""
Comprehensive test untuk validasi sistem Face Anti-Spoofing
setelah optimisasi lightweight
"""

def test_imports():
    """Test semua import yang diperlukan"""
    print("🔍 Testing imports...")
    
    try:
        # Core system
        from src.core.app_launcher import AppLauncher
        print("  ✅ AppLauncher")
        
        from src.web.app_clean import create_app
        print("  ✅ Flask app factory")
        
        from src.detection.landmark_detection import FacialLandmarkDetector, LivenessVerifier
        print("  ✅ Landmark detection")
        
        from src.models.cnn_model import LivenessDetectionCNN
        print("  ✅ CNN model")
        
        from src.utils.config import ConfigManager
        print("  ✅ Configuration")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_initialization():
    """Test inisialisasi komponen utama"""
    print("\n🛠️  Testing initialization...")
    
    try:
        # Test landmark detector
        from src.detection.landmark_detection import FacialLandmarkDetector, LivenessVerifier
        detector = FacialLandmarkDetector()
        print("  ✅ FacialLandmarkDetector initialized")
        
        # Test liveness verifier
        verifier = LivenessVerifier()
        print("  ✅ LivenessVerifier initialized")
        
        # Test config manager
        from src.utils.config import ConfigManager
        config = ConfigManager()
        print("  ✅ ConfigManager initialized")
        
        return True
    except Exception as e:
        print(f"  ❌ Initialization error: {e}")
        return False

def test_web_app():
    """Test Flask app creation"""
    print("\n🌐 Testing web application...")
    
    try:
        from src.web.app_clean import create_app
        app, socketio = create_app()
        print("  ✅ Flask app created successfully")
        
        # Test basic routes
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("  ✅ Home route working")
            else:
                print(f"  ⚠️  Home route returned status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"  ❌ Web app error: {e}")
        return False

def test_dependencies():
    """Test essential dependencies"""
    print("\n📦 Testing dependencies...")
    
    essential_deps = [
        'torch', 'cv2', 'numpy', 'flask', 'PIL'
    ]
    
    for dep in essential_deps:
        try:
            if dep == 'cv2':
                import cv2
                print(f"  ✅ OpenCV: {cv2.__version__}")
            elif dep == 'PIL':
                from PIL import Image
                print("  ✅ Pillow")
            elif dep == 'torch':
                import torch
                print(f"  ✅ PyTorch: {torch.__version__}")
            elif dep == 'numpy':
                import numpy as np
                print(f"  ✅ NumPy: {np.__version__}")
            elif dep == 'flask':
                import flask
                print(f"  ✅ Flask: {flask.__version__}")
        except ImportError:
            print(f"  ❌ {dep} not available")
            return False
    
    # Test optional dependencies
    optional_deps = ['mediapipe', 'matplotlib', 'seaborn']
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep} available (optional)")
        except ImportError:
            print(f"  ⚠️  {dep} not available (optional)")
    
    return True

def main():
    """Run comprehensive test"""
    print("🎯 Face Anti-Spoofing System - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization), 
        ("Web App Test", test_web_app),
        ("Dependencies Test", test_dependencies)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n🚀 System Status: READY FOR DEPLOYMENT")
        print("\nNext steps:")
        print("1. Start web app: python main.py --mode web")
        print("2. Open browser: http://localhost:5000")
        print("3. Test with webcam or upload images")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

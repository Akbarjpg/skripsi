#!/usr/bin/env python3
"""
Quick Installation Test for Step 3 System
Run this to verify all dependencies are installed correctly
"""

def test_step3_dependencies():
    """Test all Step 3 dependencies"""
    print("🧪 TESTING STEP 3 DEPENDENCIES")
    print("=" * 50)
    
    # Test core dependencies
    tests = [
        ("OpenCV", "import cv2; print(f'OpenCV {cv2.__version__}')"),
        ("MediaPipe", "import mediapipe as mp; print(f'MediaPipe {mp.__version__}')"),
        ("PyTorch", "import torch; print(f'PyTorch {torch.__version__}')"),
        ("NumPy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("Pygame", "import pygame; pygame.mixer.init(); print('Pygame OK')"),
        ("pyttsx3", "import pyttsx3; engine = pyttsx3.init(); engine.stop(); print('pyttsx3 OK')"),
        ("Flask", "import flask; print(f'Flask {flask.__version__}')"),
        ("Flask-SocketIO", "import flask_socketio; print('Flask-SocketIO OK')"),
        ("Pillow", "from PIL import Image; print('Pillow OK')"),
        ("scikit-learn", "import sklearn; print(f'scikit-learn {sklearn.__version__}')")
    ]
    
    passed = 0
    failed = 0
    
    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    print(f"\n📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL DEPENDENCIES READY!")
        test_step3_imports()
        return True
    else:
        print("❌ Some dependencies missing. Check installation.")
        return False

def test_step3_imports():
    """Test Step 3 specific imports"""
    print(f"\n{'='*50}")
    print("🎯 TESTING STEP 3 SPECIFIC COMPONENTS")
    print("=" * 50)
    
    try:
        # Test if we can import Step 3 modules (if they exist)
        print("Testing Step 3 modules...")
        
        # These might not exist yet if user is setting up fresh
        step3_tests = [
            "from src.challenge.distance_challenge import DistanceChallenge",
            "from src.challenge.audio_feedback import AudioFeedbackSystem", 
            "from src.challenge.challenge_response import ChallengeResponseSystem"
        ]
        
        for test in step3_tests:
            try:
                exec(test)
                print(f"✅ {test.split()[-1]}")
            except ImportError:
                print(f"⚠️  {test.split()[-1]} - not found (need to copy source files)")
            except Exception as e:
                print(f"❌ {test.split()[-1]} - error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Step 3 import test failed: {e}")
        return False

def test_camera_quick():
    """Quick camera test"""
    print(f"\n{'='*50}")
    print("📷 QUICK CAMERA TEST")
    print("=" * 50)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"✅ Camera working - Frame: {frame.shape}")
                return True
            else:
                print("❌ Cannot read from camera")
        else:
            print("❌ Cannot open camera")
        return False
        
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return False

def test_audio_quick():
    """Quick audio test"""
    print(f"\n{'='*50}")
    print("🔊 QUICK AUDIO TEST") 
    print("=" * 50)
    
    try:
        import pygame
        pygame.mixer.init()
        print("✅ Pygame audio initialized")
        
        import pyttsx3
        engine = pyttsx3.init()
        engine.stop()
        print("✅ Text-to-speech ready")
        return True
        
    except Exception as e:
        print(f"❌ Audio test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STEP 3 QUICK INSTALLATION TEST")
    print("=" * 50)
    
    deps_ok = test_step3_dependencies()
    camera_ok = test_camera_quick()
    audio_ok = test_audio_quick()
    
    print(f"\n{'='*50}")
    print("📋 FINAL RESULTS")
    print("=" * 50)
    
    if deps_ok:
        print("✅ Dependencies: OK")
    else:
        print("❌ Dependencies: FAILED")
    
    if camera_ok:
        print("✅ Camera: OK")
    else:
        print("⚠️  Camera: Not working")
    
    if audio_ok:
        print("✅ Audio: OK") 
    else:
        print("⚠️  Audio: Not working")
    
    if deps_ok:
        print(f"\n🎉 STEP 3 SYSTEM READY!")
        print("📋 Next steps:")
        print("1. Copy source files (src/ folder)")
        print("2. Run: python test_step3_enhanced_challenges.py")
        print("3. Or run: python -m src.web.app")
    else:
        print(f"\n❌ Setup incomplete. Install missing dependencies.")

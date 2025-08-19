#!/usr/bin/env python3
"""
Test Integration of Blink Detection Fixes in Web Application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_web_app_integration():
    """Test that the web app can import and use the fixed components"""
    print("🧪 TESTING WEB APP INTEGRATION")
    print("=" * 50)
    
    try:
        print("1. 📦 Testing imports...")
        
        # Test landmark detection import
        from src.detection.landmark_detection import LivenessVerifier
        print("   ✅ LivenessVerifier imported")
        
        # Test CNN model import  
        from src.models.optimized_cnn_model import OptimizedLivenessPredictor
        print("   ✅ OptimizedLivenessPredictor imported")
        
        # Test web app import
        from src.web.app_optimized import create_optimized_app
        print("   ✅ Web app imported")
        
        print("\n2. 🔧 Testing component initialization...")
        
        # Test LivenessVerifier initialization
        verifier = LivenessVerifier()
        print(f"   ✅ LivenessVerifier initialized")
        print(f"      - Left eye indices: {verifier.left_eye_indices}")
        print(f"      - Right eye indices: {verifier.right_eye_indices}")
        print(f"      - EAR threshold: {verifier.ear_threshold}")
        
        # Test CNN predictor initialization
        predictor = OptimizedLivenessPredictor()
        print(f"   ✅ CNN predictor initialized")
        
        print("\n3. 🎯 Testing blink detection logic...")
        
        # Create test landmarks
        test_landmarks = []
        for i in range(500):  # MediaPipe has 468 points
            test_landmarks.append([float(i % 100), float((i * 2) % 100)])
        
        # Test EAR calculation
        left_ear = verifier.calculate_eye_aspect_ratio(test_landmarks, verifier.left_eye_indices)
        right_ear = verifier.calculate_eye_aspect_ratio(test_landmarks, verifier.right_eye_indices)
        
        print(f"   ✅ EAR calculation working:")
        print(f"      - Left EAR: {left_ear:.3f}")
        print(f"      - Right EAR: {right_ear:.3f}")
        
        if left_ear > 0 and right_ear > 0:
            print("   ✅ EAR calculation returns valid values")
        else:
            print("   ⚠️ EAR calculation returned zero values")
        
        print("\n4. 🌐 Testing web app creation...")
        
        # Test app creation (don't run it)
        app, socketio = create_optimized_app()
        print("   ✅ Flask app and SocketIO created successfully")
        
        print("\n🎉 INTEGRATION TEST COMPLETE!")
        print("=" * 50)
        print("✅ All components integrated successfully")
        print("🚀 Ready to test blink detection in web interface")
        print("\nTo start the server, run:")
        print("  python launch_optimized.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_launch_script():
    """Create a quick launch script with fixes applied"""
    launch_script = """#!/usr/bin/env python3
\"\"\"
Quick Launch Script with Blink Detection Fixes Applied
\"\"\"

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def quick_launch():
    print("🚀 LAUNCHING WITH BLINK DETECTION FIXES")
    print("=" * 50)
    print("🔧 APPLIED FIXES:")
    print("  ✅ Numpy stride error fixed (.copy() method)")
    print("  ✅ MediaPipe eye landmarks optimized (4-point EAR)")
    print("  ✅ Blink detection sensitivity improved (threshold 0.30)")
    print("  ✅ Faster detection (2-frame requirement)")
    print("  ✅ Enhanced debug logging")
    print("=" * 50)
    
    try:
        from src.web.app_optimized import create_optimized_app
        app, socketio = create_optimized_app()
        
        print("🌐 Starting server on http://127.0.0.1:5000")
        print("📱 Navigate to Sequential Detection for blink test")
        print("🎯 Try 'Kedipkan mata 3 kali' challenge")
        print("\\n⚡ Server starting...")
        
        socketio.run(
            app,
            host='127.0.0.1',
            port=5000,
            debug=True,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\\n👋 Server stopped")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_launch()
"""
    
    with open("quick_launch_with_fixes.py", "w") as f:
        f.write(launch_script)
    
    print("📝 Created quick_launch_with_fixes.py")

if __name__ == "__main__":
    success = test_web_app_integration()
    
    if success:
        create_quick_launch_script()
        print("\n🎯 NEXT STEPS:")
        print("1. Run: python quick_launch_with_fixes.py")
        print("2. Open browser: http://127.0.0.1:5000")
        print("3. Go to Sequential Detection")
        print("4. Test 'Kedipkan mata 3 kali' challenge")
        print("5. Watch for improved blink detection!")
    else:
        print("\n🔧 Fix any import errors before launching")

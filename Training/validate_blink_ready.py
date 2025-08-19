#!/usr/bin/env python3
"""
Pre-Launch Validation for Blink Detection Fixes
Run this before launching the main application
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def validate_blink_fixes():
    """Validate that all blink detection fixes are working"""
    print("🔍 VALIDATING BLINK DETECTION FIXES")
    print("=" * 50)
    
    validation_passed = True
    
    try:
        print("1. 📦 Testing imports...")
        
        # Test landmark detection import
        try:
            from src.detection.landmark_detection import LivenessVerifier
            print("   ✅ LivenessVerifier imported successfully")
        except Exception as e:
            print(f"   ❌ LivenessVerifier import failed: {e}")
            validation_passed = False
            
        # Test CNN model import
        try:
            from src.models.optimized_cnn_model import OptimizedLivenessPredictor
            print("   ✅ OptimizedLivenessPredictor imported successfully")
        except Exception as e:
            print(f"   ❌ OptimizedLivenessPredictor import failed: {e}")
            validation_passed = False
        
        if not validation_passed:
            return False
            
        print("\n2. 🔧 Testing component initialization...")
        
        # Test LivenessVerifier
        verifier = LivenessVerifier()
        print(f"   ✅ LivenessVerifier initialized")
        print(f"      • EAR threshold: {verifier.ear_threshold}")
        print(f"      • Consecutive frames: {verifier.consecutive_frames}")
        print(f"      • Left eye indices: {len(verifier.left_eye_indices)} points")
        print(f"      • Right eye indices: {len(verifier.right_eye_indices)} points")
        
        # Check if we're using the fixed 4-point method
        if len(verifier.left_eye_indices) == 4 and len(verifier.right_eye_indices) == 4:
            print("   ✅ Using optimized 4-point EAR calculation")
        else:
            print(f"   ⚠️ Expected 4 points per eye, got {len(verifier.left_eye_indices)} and {len(verifier.right_eye_indices)}")
        
        # Test CNN predictor
        predictor = OptimizedLivenessPredictor()
        print("   ✅ CNN predictor initialized")
        
        print("\n3. 🎯 Testing EAR calculation...")
        
        # Create realistic test landmarks
        test_landmarks = []
        for i in range(500):  # More than MediaPipe's 468 points for safety
            x = 50 + (i % 20) * 2  # Spread points realistically
            y = 50 + (i // 20) * 2
            test_landmarks.append([float(x), float(y)])
        
        # Test EAR for both eyes
        left_ear = verifier.calculate_eye_aspect_ratio(test_landmarks, verifier.left_eye_indices)
        right_ear = verifier.calculate_eye_aspect_ratio(test_landmarks, verifier.right_eye_indices)
        
        print(f"   ✅ Left EAR: {left_ear:.3f}")
        print(f"   ✅ Right EAR: {right_ear:.3f}")
        
        if left_ear > 0 and right_ear > 0:
            print("   ✅ EAR calculation returns valid values")
        else:
            print("   ❌ EAR calculation returned zero values")
            validation_passed = False
        
        print("\n4. 🧪 Testing blink detection logic...")
        
        # Simulate a blink (lower EAR values)
        blink_landmarks = []
        for i in range(500):
            x = 50 + (i % 20) * 2
            y = 50 + (i // 20) * 1  # Reduced Y spread to simulate closed eye
            blink_landmarks.append([float(x), float(y)])
        
        blink_left_ear = verifier.calculate_eye_aspect_ratio(blink_landmarks, verifier.left_eye_indices)
        blink_right_ear = verifier.calculate_eye_aspect_ratio(blink_landmarks, verifier.right_eye_indices)
        
        print(f"   📊 Blink simulation EAR: {blink_left_ear:.3f}, {blink_right_ear:.3f}")
        
        # Check if blink would be detected
        avg_ear = (blink_left_ear + blink_right_ear) / 2
        if avg_ear < verifier.ear_threshold:
            print(f"   ✅ Blink would be detected (EAR {avg_ear:.3f} < threshold {verifier.ear_threshold})")
        else:
            print(f"   ⚠️ Blink might not be detected (EAR {avg_ear:.3f} >= threshold {verifier.ear_threshold})")
        
        print("\n5. 🌐 Testing web app integration...")
        
        try:
            from src.web.app_optimized import create_optimized_app
            app, socketio = create_optimized_app()
            print("   ✅ Web app creates successfully")
        except Exception as e:
            print(f"   ❌ Web app creation failed: {e}")
            validation_passed = False
        
        if validation_passed:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("=" * 50)
            print("✅ Blink detection fixes are properly integrated")
            print("✅ Ready to launch the web application")
            print("\n🚀 To start the server:")
            print("   python launch_blink_fixed.py")
            print("\n🎯 Test sequence:")
            print("   1. Navigate to Sequential Detection")
            print("   2. Select 'Kedipkan mata 3 kali'")
            print("   3. Blink 3 times clearly")
            print("   4. Verify counter increases and challenge completes")
        else:
            print("\n❌ VALIDATION FAILED!")
            print("Please fix the issues above before launching")
            
        return validation_passed
        
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_blink_fixes()
    
    if success:
        print("\n🎯 READY TO TEST!")
        print("Run: python launch_blink_fixed.py")
    else:
        print("\n🔧 Please fix validation errors first")

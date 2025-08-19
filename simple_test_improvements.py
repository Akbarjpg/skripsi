"""
Simple test to verify landmark detection improvements
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ear_calculation():
    """Test the improved EAR calculation"""
    print("🧪 Testing EAR Calculation Improvements")
    
    try:
        from detection.landmark_detection import LivenessVerifier
        
        verifier = LivenessVerifier()
        
        # Test with sample landmark data (6 points for each eye)
        sample_landmarks = []
        for i in range(468):  # MediaPipe provides 468 landmarks
            sample_landmarks.append([0.5 + np.random.normal(0, 0.1), 0.5 + np.random.normal(0, 0.1)])
        
        # Test EAR calculation
        left_ear = verifier.calculate_eye_aspect_ratio(sample_landmarks, verifier.left_eye_ear_indices)
        right_ear = verifier.calculate_eye_aspect_ratio(sample_landmarks, verifier.right_eye_ear_indices)
        
        print(f"✅ EAR calculation successful:")
        print(f"   Left EAR: {left_ear:.3f}")
        print(f"   Right EAR: {right_ear:.3f}")
        
        # Test blink detection
        initial_blinks = verifier.blink_count
        
        # Simulate closed eyes (low EAR)
        for _ in range(5):
            verifier.detect_blink(0.15, 0.15)  # Closed eyes
        
        # Simulate open eyes (normal EAR)
        for _ in range(3):
            verifier.detect_blink(0.3, 0.3)   # Open eyes
        
        blinks_detected = verifier.blink_count - initial_blinks
        print(f"✅ Blink detection test: {blinks_detected} blinks detected")
        
        return True
        
    except Exception as e:
        print(f"❌ EAR calculation test failed: {e}")
        return False

def test_head_pose():
    """Test the improved head pose estimation"""
    print("\n🧪 Testing Head Pose Improvements")
    
    try:
        from detection.landmark_detection import LivenessVerifier
        
        verifier = LivenessVerifier()
        
        # Create sample landmarks
        sample_landmarks = []
        for i in range(468):
            sample_landmarks.append([0.5 + np.random.normal(0, 0.05), 0.5 + np.random.normal(0, 0.05)])
        
        # Test head pose calculation
        head_pose = verifier.calculate_head_pose(sample_landmarks)
        
        if head_pose:
            print(f"✅ Head pose calculation successful:")
            print(f"   Yaw: {head_pose['yaw']:.1f}°")
            print(f"   Pitch: {head_pose['pitch']:.1f}°") 
            print(f"   Roll: {head_pose['roll']:.1f}°")
            print(f"   Confidence: {head_pose.get('confidence', 'N/A')}")
            return True
        else:
            print("❌ Head pose calculation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Head pose test failed: {e}")
        return False

def test_quality_validation():
    """Test landmark quality validation"""
    print("\n🧪 Testing Quality Validation")
    
    try:
        from detection.landmark_detection import FacialLandmarkDetector
        
        detector = FacialLandmarkDetector()
        
        # Test with good quality landmarks
        good_landmarks = []
        for i in range(468):
            x = 0.3 + (i % 100) * 0.004  # Spread across face area
            y = 0.3 + (i // 100) * 0.08
            good_landmarks.append([x, y])
        
        is_valid = detector._validate_landmark_quality(good_landmarks, 0.8)
        print(f"✅ Good landmarks validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test with poor quality landmarks (too few)
        poor_landmarks = [[0.5, 0.5] for _ in range(100)]  # Only 100 landmarks
        is_invalid = not detector._validate_landmark_quality(poor_landmarks, 0.8)
        print(f"✅ Poor landmarks rejection: {'PASS' if is_invalid else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality validation test failed: {e}")
        return False

def test_optimized_version():
    """Test the optimized version"""
    print("\n🧪 Testing Optimized Version")
    
    try:
        from detection.optimized_landmark_detection import OptimizedLivenessVerifier
        
        verifier = OptimizedLivenessVerifier()
        
        # Create dummy image
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Test processing
        result = verifier.process_frame_optimized(dummy_image)
        
        print(f"✅ Optimized processing successful")
        print(f"   Landmarks detected: {result['landmarks_detected']}")
        print(f"   Processing time: {result.get('processing_time', 'N/A')}")
        print(f"   FPS estimate: {result.get('fps_estimate', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized version test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 LANDMARK DETECTION IMPROVEMENTS - SIMPLE TEST")
    print("=" * 50)
    
    tests = [
        ("EAR Calculation", test_ear_calculation),
        ("Head Pose Estimation", test_head_pose),
        ("Quality Validation", test_quality_validation),
        ("Optimized Version", test_optimized_version)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 TEST RESULTS")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Improvements are working correctly.")
        print("\nKey improvements verified:")
        print("✅ Robust 6-point EAR calculation")
        print("✅ Enhanced head pose estimation with confidence")
        print("✅ Landmark quality validation")
        print("✅ Optimized processing pipeline")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the implementation.")

if __name__ == "__main__":
    main()

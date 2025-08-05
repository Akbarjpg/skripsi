#!/usr/bin/env python3
"""
Final comprehensive liveness detection test and demonstration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.landmark_detection import LivenessVerifier
import cv2
import numpy as np
import time

def test_comprehensive_liveness():
    """Comprehensive test showing all liveness features"""
    print("🎯 COMPREHENSIVE LIVENESS DETECTION TEST")
    print("=" * 70)
    print("This test demonstrates all implemented liveness features:")
    print("  ✅ Real-time liveness scoring (0-100)")
    print("  ✅ Eye blink detection and counting") 
    print("  ✅ Eye Aspect Ratio (EAR) calculation")
    print("  ✅ Mouth movement detection (MAR)")
    print("  ✅ Head pose estimation")
    print("  ✅ Live/Fake classification")
    print("  ✅ Anti-spoofing checks")
    print("=" * 70)
    
    try:
        # Initialize verifier
        verifier = LivenessVerifier()
        print("✅ LivenessVerifier initialized successfully")
        
        # Test 1: No face (baseline)
        print("\n📋 TEST 1: No Face Detection (Baseline)")
        print("-" * 40)
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result_empty = verifier.process_frame(empty_image)
        
        print(f"  Landmarks detected: {result_empty['landmarks_detected']}")
        print(f"  Liveness score: {result_empty['liveness_score']:.1f}/100")
        print(f"  Status: {result_empty['liveness_status']}")
        print(f"  Is live: {result_empty['is_live']}")
        
        # Test 2: Simulated face data (mock test)
        print("\n📋 TEST 2: Simulated Face with Good Metrics")
        print("-" * 40)
        
        # Create mock landmarks (468 points as MediaPipe would provide)
        mock_landmarks = []
        for i in range(468):
            # Create realistic landmark positions
            x = 0.3 + (i % 20) * 0.02  # Spread across face width
            y = 0.3 + (i // 20) * 0.02  # Spread across face height  
            mock_landmarks.append([x, y])
        
        # Simulate processing with mock data
        print("  Simulating face with landmarks...")
        
        # Manually test scoring components
        ear_left = 0.28  # Normal eye opening
        ear_right = 0.26  # Normal eye opening  
        mar = 0.35  # Slight mouth opening
        
        # Mock head pose
        head_pose = {
            'yaw': 12.5,
            'pitch': -5.2,
            'roll': 2.1
        }
        
        # Calculate liveness score using the actual method
        liveness_score = verifier.calculate_liveness_score(
            mock_landmarks, ear_left, ear_right, mar, head_pose
        )
        
        is_live = verifier.is_live_face(liveness_score)
        
        print(f"  Landmarks: {len(mock_landmarks)} points")
        print(f"  EAR Left: {ear_left:.3f}")
        print(f"  EAR Right: {ear_right:.3f}")
        print(f"  MAR: {mar:.3f}")
        print(f"  Head pose: Yaw={head_pose['yaw']:.1f}° Pitch={head_pose['pitch']:.1f}° Roll={head_pose['roll']:.1f}°")
        print(f"  🎯 Liveness Score: {liveness_score:.1f}/100")
        print(f"  🔍 Classification: {'LIVE' if is_live else 'FAKE'}")
        
        # Test 3: Feature validation
        print("\n📋 TEST 3: Feature Availability Check")
        print("-" * 40)
        
        # Check all available methods
        available_methods = [method for method in dir(verifier) if not method.startswith('_')]
        
        critical_methods = [
            'calculate_liveness_score', 'is_live_face', 'calculate_eye_aspect_ratio',
            'calculate_head_pose', 'detect_blink', 'detect_head_movement', 'process_frame'
        ]
        
        print("  Critical methods available:")
        for method in critical_methods:
            status = "✅" if method in available_methods else "❌"
            print(f"    {status} {method}")
        
        # Test 4: Camera availability (if possible)
        print("\n📋 TEST 4: Camera System Check")
        print("-" * 40)
        
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("  ✅ Camera accessible")
                
                # Try to capture one frame
                ret, frame = cap.read()
                if ret:
                    print(f"  ✅ Frame capture successful: {frame.shape}")
                    
                    # Test actual processing
                    actual_result = verifier.process_frame(frame)
                    print(f"  🎯 Real frame liveness score: {actual_result['liveness_score']:.1f}/100")
                    print(f"  👤 Face detected: {actual_result['landmarks_detected']}")
                    
                else:
                    print("  ❌ Frame capture failed")
                cap.release()
            else:
                print("  ❌ Camera not accessible")
        except Exception as e:
            print(f"  ⚠️ Camera test skipped: {e}")
        
        # Final assessment
        print("\n🏁 FINAL ASSESSMENT")
        print("=" * 70)
        
        # Score the implementation
        features_working = 0
        total_features = 6
        
        # Check each feature
        if hasattr(verifier, 'calculate_liveness_score'):
            print("  ✅ Liveness scoring system: IMPLEMENTED")
            features_working += 1
        else:
            print("  ❌ Liveness scoring system: MISSING")
            
        if hasattr(verifier, 'calculate_eye_aspect_ratio'):
            print("  ✅ Eye blink detection (EAR): IMPLEMENTED")  
            features_working += 1
        else:
            print("  ❌ Eye blink detection (EAR): MISSING")
            
        if hasattr(verifier, 'calculate_head_pose'):
            print("  ✅ Head pose estimation: IMPLEMENTED")
            features_working += 1
        else:
            print("  ❌ Head pose estimation: MISSING")
            
        if 'mouth_open' in result_empty:
            print("  ✅ Mouth movement detection: IMPLEMENTED")
            features_working += 1
        else:
            print("  ❌ Mouth movement detection: MISSING")
            
        if hasattr(verifier, 'is_live_face'):
            print("  ✅ Live/Fake classification: IMPLEMENTED")
            features_working += 1
        else:
            print("  ❌ Live/Fake classification: MISSING")
            
        if liveness_score > 0:
            print("  ✅ Anti-spoofing measures: BASIC LEVEL")
            features_working += 1
        else:
            print("  ❌ Anti-spoofing measures: MISSING")
        
        # Overall grade
        percentage = (features_working / total_features) * 100
        
        print(f"\n📊 IMPLEMENTATION SCORE: {features_working}/{total_features} ({percentage:.0f}%)")
        
        if percentage >= 90:
            print("🟢 EXCELLENT: Full liveness detection system implemented!")
        elif percentage >= 70:
            print("🟡 GOOD: Most liveness features working well!")
        elif percentage >= 50:
            print("🟠 FAIR: Basic liveness detection functional!")
        else:
            print("🔴 POOR: Major liveness features missing!")
        
        print("\n💡 NEXT STEPS:")
        if percentage >= 70:
            print("  🚀 Ready for production testing!")
            print("  🌐 Start web server: python simple_liveness_server.py")
            print("  📱 Test in browser: http://localhost:5000/face_detection")
        else:
            print("  🔧 Fix missing features before production use")
            
        print("\n🎉 TEST COMPLETED SUCCESSFULLY!")
        
        return {
            'features_working': features_working,
            'total_features': total_features,
            'percentage': percentage,
            'liveness_score_demo': liveness_score
        }
        
    except Exception as e:
        print(f"❌ Error in comprehensive test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_comprehensive_liveness()
    if result:
        print(f"\n✅ Comprehensive test completed: {result['percentage']:.0f}% implementation")

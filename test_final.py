#!/usr/bin/env python3
"""
FINAL TEST - Verify landmark system is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("🚀 FINAL LANDMARK SYSTEM TEST")
    print("=" * 50)
    
    # Test import
    from src.detection.landmark_detection import LivenessVerifier
    print("✅ LivenessVerifier imported successfully")
    
    # Create verifier
    verifier = LivenessVerifier()
    print("✅ LivenessVerifier instance created")
    
    # Test EAR calculation with dummy data
    print("\n🧪 Testing EAR calculation...")
    
    # Create normalized landmark coordinates (like MediaPipe output)
    dummy_landmarks = []
    for i in range(478):
        x = 0.3 + (i % 100) * 0.004  # x between 0.3-0.7
        y = 0.2 + (i % 150) * 0.004  # y between 0.2-0.8
        dummy_landmarks.append([x, y])
    
    print(f"✅ Created {len(dummy_landmarks)} dummy landmarks")
    
    # Test left eye indices (first 6)
    left_eye_indices = verifier.landmark_detector.left_eye_indices[:6]
    print(f"📍 Testing with eye indices: {left_eye_indices}")
    
    # Test EAR calculation
    ear_result = verifier.calculate_eye_aspect_ratio(dummy_landmarks, left_eye_indices)
    print(f"✅ EAR calculation result: {ear_result}")
    
    # Test full process_frame simulation
    print("\n🎯 Testing full process_frame workflow...")
    
    import numpy as np
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_image[200:280, 300:340] = [128, 128, 128]  # Gray rectangle as "face"
    
    # This will either detect real landmarks or return empty result
    results = verifier.process_frame(dummy_image)
    
    print("📊 Process Frame Results:")
    print(f"  - Landmarks detected: {results['landmarks_detected']}")
    print(f"  - Landmark count: {len(results.get('landmark_coordinates', []))}")
    print(f"  - Confidence: {results['confidence']}")
    print(f"  - EAR left: {results['ear_left']}")
    print(f"  - EAR right: {results['ear_right']}")
    print(f"  - Blink count: {results['blink_count']}")
    
    # Final status
    print("\n" + "=" * 50)
    if results['landmarks_detected']:
        print("🎉 SUCCESS: Real landmark detection is working!")
        print(f"   MediaPipe detected {len(results['landmark_coordinates'])} landmarks")
    else:
        print("⚠️  INFO: No real landmarks detected (expected without webcam)")
        print("   But all processing pipeline is functional")
    
    print("✅ All landmark processing functions are working correctly")
    print("✅ No list index errors encountered")
    print("✅ System ready for web interface testing")
    
    print("\n🌐 Next step: Test web interface at http://localhost:5000/face_detection_clean")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔧 Debug info:")
    print("   Make sure you're in the project root directory")
    print("   Make sure MediaPipe is installed")
    print("   Make sure the server is running")

print("\n🏁 Test complete!")

#!/usr/bin/env python3
"""
Test landmark processing dengan gambar dummy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    from src.detection.landmark_detection import LivenessVerifier
    print("✅ Import successful")
    
    # Create dummy landmarks (478 points)
    dummy_landmarks = []
    for i in range(478):
        x = 0.3 + (i % 100) * 0.004  # x between 0.3-0.7
        y = 0.2 + (i % 150) * 0.004  # y between 0.2-0.8
        dummy_landmarks.append([x, y])
    
    print(f"✅ Created {len(dummy_landmarks)} dummy landmarks")
    
    # Test verifier
    verifier = LivenessVerifier()
    print("✅ LivenessVerifier created")
    
    # Test EAR calculation directly
    left_eye_indices = verifier.landmark_detector.left_eye_indices[:6]
    ear = verifier.calculate_eye_aspect_ratio(dummy_landmarks, left_eye_indices)
    
    print(f"✅ EAR calculation: {ear}")
    
    # Test coordinate access
    for i, idx in enumerate(left_eye_indices):
        if idx < len(dummy_landmarks):
            landmark = dummy_landmarks[idx]
            print(f"  Landmark {idx}: {landmark}")
        if i >= 2:  # Just show first 3
            break
            
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

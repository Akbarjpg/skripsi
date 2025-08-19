#!/usr/bin/env python3
"""
Simple test for blink detection functionality
"""

import os
import sys
import cv2
import numpy as np

print("=== SIMPLE BLINK TEST ===")

# Test 1: Import check
try:
    from src.detection.landmark_detection import LivenessVerifier
    print("✓ Import LivenessVerifier - SUCCESS")
except Exception as e:
    print(f"✗ Import LivenessVerifier - FAILED: {e}")
    sys.exit(1)

# Test 2: Initialize verifier
try:
    verifier = LivenessVerifier()
    print("✓ Initialize LivenessVerifier - SUCCESS")
except Exception as e:
    print(f"✗ Initialize LivenessVerifier - FAILED: {e}")
    sys.exit(1)

# Test 3: Check eye indices
try:
    print(f"✓ Left eye indices: {verifier.left_eye_indices}")
    print(f"✓ Right eye indices: {verifier.right_eye_indices}")
    print(f"✓ Total indices: {len(verifier.left_eye_indices) + len(verifier.right_eye_indices)}")
except Exception as e:
    print(f"✗ Eye indices check - FAILED: {e}")

# Test 4: EAR calculation with dummy data
try:
    # Create dummy landmarks (4 points for each eye)
    dummy_landmarks = []
    for i in range(500):  # MediaPipe has 468 points, we'll create 500 to be safe
        dummy_landmarks.append([float(i % 100), float((i * 2) % 100)])
    
    # Test EAR calculation
    left_ear = verifier.calculate_eye_aspect_ratio(dummy_landmarks, verifier.left_eye_indices)
    right_ear = verifier.calculate_eye_aspect_ratio(dummy_landmarks, verifier.right_eye_indices)
    
    print(f"✓ Left EAR calculation: {left_ear:.3f}")
    print(f"✓ Right EAR calculation: {right_ear:.3f}")
    
    if left_ear > 0 and right_ear > 0:
        print("✓ EAR calculation - SUCCESS")
    else:
        print("✗ EAR calculation returned zero values")

except Exception as e:
    print(f"✗ EAR calculation - FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== SIMPLE BLINK TEST COMPLETE ===")

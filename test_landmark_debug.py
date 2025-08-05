#!/usr/bin/env python3
"""
Quick test untuk landmark detection dengan debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.landmark_detection import LivenessVerifier
import cv2
import numpy as np

def test_quick():
    """Test quick landmark detection"""
    print("=== QUICK LANDMARK DETECTION TEST ===")
    
    verifier = LivenessVerifier()
    print("LivenessVerifier initialized")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("Camera opened, processing frame...")
    
    # Read one frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        cap.release()
        return
    
    print(f"Frame read: {frame.shape}")
    
    # Process frame
    results = verifier.process_frame(frame)
    
    # Print results
    print("=== RESULTS ===")
    print(f"Landmarks detected: {results['landmarks_detected']}")
    print(f"Confidence: {results['confidence']}")
    print(f"Number of landmarks: {len(results['landmark_coordinates'])}")
    print(f"EAR left: {results['ear_left']}")
    print(f"EAR right: {results['ear_right']}")
    print(f"MAR: {results['mar']}")
    print(f"Blink count: {results['blink_count']}")
    print(f"Mouth open: {results['mouth_open']}")
    
    if results['landmarks_detected'] and len(results['landmark_coordinates']) > 0:
        print("SUCCESS: Landmarks detected and processed!")
        print(f"First 5 landmarks: {results['landmark_coordinates'][:5]}")
    else:
        print("FAILED: No landmarks detected")
    
    cap.release()

if __name__ == "__main__":
    test_quick()

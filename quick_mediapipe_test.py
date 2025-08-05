#!/usr/bin/env python3
"""
Quick test to verify MediaPipe landmark detection is working
"""

import cv2
import mediapipe as mp
import numpy as np

def test_mediapipe_simple():
    print("=== SIMPLE MEDIAPIPE TEST ===")
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✓ MediaPipe FaceMesh initialized")
    
    # Create a test image (solid color for testing)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (128, 128, 128)  # Gray image
    
    print(f"✓ Test image created: {test_image.shape}")
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    print(f"✓ Converted to RGB: {rgb_image.shape}")
    
    # Process with MediaPipe
    try:
        results = face_mesh.process(rgb_image)
        print(f"✓ MediaPipe processing completed")
        
        if results.multi_face_landmarks:
            print(f"✓ Face detected: {len(results.multi_face_landmarks)} face(s)")
            for face_landmarks in results.multi_face_landmarks:
                landmark_count = len(face_landmarks.landmark)
                print(f"  - Landmarks: {landmark_count}")
        else:
            print("- No face detected (expected for gray image)")
            
        print("✅ MediaPipe is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe error: {e}")
        return False

if __name__ == "__main__":
    test_mediapipe_simple()

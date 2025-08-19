#!/usr/bin/env python3
"""
Simple MediaPipe test script to verify landmark detection is working
"""

import cv2
import mediapipe as mp
import numpy as np

def test_mediapipe():
    print("=== MEDIAPIPE STANDALONE TEST ===")
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✓ MediaPipe FaceMesh initialized")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
        
    print("✓ Webcam opened")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
            
        frame_count += 1
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_face_landmarks:
            print(f"Frame {frame_count}: {len(results.multi_face_landmarks)} face(s) detected")
            
            for face_landmarks in results.multi_face_landmarks:
                # Count landmarks
                landmark_count = len(face_landmarks.landmark)
                print(f"  Landmarks: {landmark_count}")
                
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        else:
            print(f"Frame {frame_count}: No face detected")
        
        # Show frame
        cv2.imshow('MediaPipe Test', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Test only 100 frames
        if frame_count >= 100:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_mediapipe()

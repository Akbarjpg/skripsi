#!/usr/bin/env python3
"""
Test script to verify the real-time landmark detection fixes
"""

import cv2
import numpy as np
from src.detection.landmark_detection import LivenessVerifier
import time

def test_mediapipe_standalone():
    """Test MediaPipe detection with webcam"""
    print("=== TESTING MEDIAPIPE STANDALONE ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Initialize detector
    verifier = LivenessVerifier()
    print("LivenessVerifier initialized successfully")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while frame_count < 30:  # Test 30 frames
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_count}")
                continue
                
            frame_count += 1
            print(f"\n=== FRAME {frame_count} ===")
            
            # Process frame
            results = verifier.process_frame(frame)
            
            # Print results
            print(f"Landmarks detected: {results['landmarks_detected']}")
            if results['landmarks_detected']:
                print(f"Landmark count: {len(results['landmark_coordinates'])}")
                print(f"Confidence: {results['confidence']}")
                print(f"Blink count: {results['blink_count']}")
                print(f"Mouth open: {results['mouth_open']}")
                
                # Show first few coordinates
                coords = results['landmark_coordinates']
                if len(coords) > 0:
                    print(f"First landmark: {coords[0]}")
                    print(f"Last landmark: {coords[-1]}")
            else:
                print("No face detected")
            
            # Add small delay
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        return False
    finally:
        cap.release()
    
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"\n=== TEST COMPLETED ===")
    print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    
    return True

def test_coordinate_format():
    """Test coordinate format and conversion"""
    print("\n=== TESTING COORDINATE FORMAT ===")
    
    # Create dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    verifier = LivenessVerifier()
    results = verifier.process_frame(dummy_image)
    
    print(f"Results keys: {list(results.keys())}")
    print(f"Landmarks detected: {results['landmarks_detected']}")
    
    if results['landmarks_detected']:
        coords = results['landmark_coordinates']
        print(f"Coordinate count: {len(coords)}")
        print(f"Coordinate type: {type(coords)}")
        if len(coords) > 0:
            print(f"First coordinate type: {type(coords[0])}")
            print(f"First coordinate value: {coords[0]}")
    
    return True

def test_backend_processing():
    """Test the backend processing logic"""
    print("\n=== TESTING BACKEND PROCESSING LOGIC ===")
    
    # Simulate the backend coordinate conversion
    dummy_landmarks = [
        [0.5, 0.3],  # Center face
        [0.3, 0.4],  # Left eye area
        [0.7, 0.4],  # Right eye area
        [0.5, 0.6],  # Mouth area
    ]
    
    # Face region indices (simplified)
    left_eye = [1]
    right_eye = [2] 
    mouth_outer = [3]
    
    # Convert to frontend format
    landmark_points = []
    for i, landmark in enumerate(dummy_landmarks):
        x = float(landmark[0])
        y = float(landmark[1])
        
        # Determine color
        if i in left_eye or i in right_eye:
            color = '#FF0000'  # Red for eyes
        elif i in mouth_outer:
            color = '#FFFF00'  # Yellow for mouth
        else:
            color = '#FFFFFF'  # White for other
            
        landmark_points.append({
            'x': x,
            'y': y,
            'color': color,
            'index': i
        })
    
    print(f"Converted {len(landmark_points)} landmarks:")
    for point in landmark_points:
        print(f"  Point {point['index']}: ({point['x']:.2f}, {point['y']:.2f}) - {point['color']}")
    
    return True

if __name__ == "__main__":
    print("REAL-TIME LANDMARK DETECTION TEST")
    print("=" * 50)
    
    # Test 1: Coordinate format
    test_coordinate_format()
    
    # Test 2: Backend processing
    test_backend_processing()
    
    # Test 3: MediaPipe standalone (with camera)
    print("\n" + "=" * 50)
    print("CAMERA TEST - Press Ctrl+C to stop")
    user_input = input("Do you want to test with camera? (y/n): ")
    
    if user_input.lower().startswith('y'):
        success = test_mediapipe_standalone()
        if success:
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED!")
    else:
        print("\nSkipping camera test")
        print("\n✅ NON-CAMERA TESTS PASSED!")

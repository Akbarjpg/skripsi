#!/usr/bin/env python3
"""
Advanced liveness detection test with camera
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.landmark_detection import LivenessVerifier
import cv2
import numpy as np

def test_camera_liveness():
    """Test liveness detection with camera"""
    print("📷 TESTING LIVENESS DETECTION WITH CAMERA")
    print("=" * 60)
    print("Instructions:")
    print("  👁️  Blink your eyes naturally")
    print("  👄  Open and close your mouth") 
    print("  🔄  Turn your head left and right")
    print("  ⬆️⬇️  Nod your head up and down")
    print("  ❌  Try holding a photo to test anti-spoofing")
    print("  📱  Try showing phone screen with face")
    print("  🔑  Press 'q' to quit")
    print()
    
    try:
        # Initialize verifier
        verifier = LivenessVerifier()
        print("✅ LivenessVerifier initialized")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
            
        print("✅ Camera opened successfully")
        print("🔴 Starting liveness detection...")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
                
            frame_count += 1
            
            # Process every 5th frame for better performance
            if frame_count % 5 == 0:
                # Process frame for liveness
                results = verifier.process_frame(frame)
                
                # Draw results on frame
                if results['landmarks_detected']:
                    # Draw liveness score
                    score = results['liveness_score']
                    status = results['liveness_status']
                    is_live = results['is_live']
                    
                    # Color based on liveness
                    if is_live:
                        color = (0, 255, 0)  # Green for live
                        text_color = (0, 0, 0)
                    elif score > 40:
                        color = (0, 255, 255)  # Yellow for uncertain
                        text_color = (0, 0, 0)
                    else:
                        color = (0, 0, 255)  # Red for fake
                        text_color = (255, 255, 255)
                    
                    # Draw liveness info
                    cv2.rectangle(frame, (10, 10), (450, 200), color, -1)
                    cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), 2)
                    
                    y_offset = 35
                    cv2.putText(frame, f"LIVENESS SCORE: {score:.1f}/100", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                    y_offset += 30
                    
                    cv2.putText(frame, f"STATUS: {status}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    y_offset += 25
                    
                    cv2.putText(frame, f"IS LIVE: {'YES' if is_live else 'NO'}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                    y_offset += 25
                    
                    # Draw metrics
                    cv2.putText(frame, f"Blinks: {results['blink_count']}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 20
                    
                    cv2.putText(frame, f"EAR: L={results['ear_left']:.3f} R={results['ear_right']:.3f}", 
                               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 20
                    
                    cv2.putText(frame, f"MAR: {results['mar']:.3f}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    y_offset += 20
                    
                    if results['mouth_open']:
                        cv2.putText(frame, "MOUTH OPEN", (20, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                    if results['head_movement']:
                        cv2.putText(frame, "HEAD MOVEMENT", (250, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                else:
                    # No face detected
                    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 255), -1)
                    cv2.putText(frame, "NO FACE DETECTED", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Please look at camera", (20, 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Print results to console every 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: Score={results['liveness_score']:.1f}, "
                          f"Status={results['liveness_status']}, Live={results['is_live']}")
            
            cv2.imshow('🔍 Advanced Liveness Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset counters
                verifier.reset_counters()
                print("🔄 Counters reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test completed")
        
    except Exception as e:
        print(f"❌ Error in camera test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_liveness()

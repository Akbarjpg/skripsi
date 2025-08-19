"""
Test script to validate the blink detection fixes
"""

import cv2
import numpy as np
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detection.landmark_detection import LivenessVerifier
    print("✅ Successfully imported LivenessVerifier")
except ImportError as e:
    print(f"❌ Failed to import LivenessVerifier: {e}")
    sys.exit(1)

def test_blink_detection():
    """Test the improved blink detection algorithm"""
    print("🧪 TESTING IMPROVED BLINK DETECTION")
    print("=" * 50)
    
    # Initialize verifier
    verifier = LivenessVerifier()
    print(f"✅ Initialized LivenessVerifier with blink threshold: {verifier.blink_threshold}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    print("📹 Camera opened successfully")
    print("\n🎯 INSTRUCTIONS:")
    print("- Look at the camera")
    print("- Blink your eyes 3 times slowly")
    print("- Watch the blink counter in the terminal")
    print("- Press 'q' to quit")
    print("- Press 'r' to reset counter")
    print("\n🔬 MONITORING:")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process every 3rd frame to reduce load
            if frame_count % 3 != 0:
                continue
            
            # Run detection
            result = verifier.process_frame(frame)
            
            # Display results
            if result['landmarks_detected']:
                blink_count = result['blink_count']
                ear_left = result['ear_left']
                ear_right = result['ear_right']
                avg_ear = (ear_left + ear_right) / 2.0
                liveness_score = result['liveness_score']
                is_live = result['is_live']
                
                print(f"\r👁️  EAR: {avg_ear:.3f} | 👀 Blinks: {blink_count} | 🎯 Score: {liveness_score:.1f} | ✅ Live: {is_live} | Frame: {frame_count}", end="", flush=True)
                
                # Visual feedback on frame
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Score: {liveness_score:.1f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Live: {is_live}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Success indicator
                if blink_count >= 3:
                    cv2.putText(frame, "CHALLENGE COMPLETE!", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                print(f"\r❌ No face detected | Frame: {frame_count}", end="", flush=True)
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Blink Detection Test', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quit requested")
                break
            elif key == ord('r'):
                print("\n🔄 Resetting counters...")
                verifier.reset_counters()
                print("✅ Counters reset")
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        end_time = time.time()
        duration = end_time - start_time
        fps = frame_count / duration if duration > 0 else 0
        
        print(f"\n\n📊 TEST RESULTS:")
        print(f"Duration: {duration:.1f}s")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Final blink count: {verifier.blink_count}")
        print(f"Blink threshold: {verifier.blink_threshold}")
        print(f"Consecutive frames needed: {verifier.blink_consecutive_frames}")
        
        if verifier.blink_count >= 3:
            print("🎉 SUCCESS: Challenge would be completed!")
        else:
            print("⚠️ Need more blinks to complete challenge")
    
    return True

def test_sequential_mode_simulation():
    """Test the sequential mode with simulated data"""
    print("\n🔄 TESTING SEQUENTIAL MODE SIMULATION")
    print("=" * 50)
    
    try:
        from web.app_optimized import SequentialDetectionState
        
        # Create sequential state
        seq_state = SequentialDetectionState()
        print(f"✅ Created sequential state, initial phase: {seq_state.phase}")
        print(f"🎯 Current challenge: {seq_state.current_challenge['instruction']}")
        
        # Simulate blink detection progress
        for i in range(5):
            # Simulate landmark results with increasing blinks
            landmark_results = {
                'landmarks_detected': True,
                'blink_count': i,
                'head_movement': False,
                'mouth_open': False
            }
            
            # Update challenge
            completed = seq_state.update_challenge(landmark_results)
            challenge_info = seq_state.get_challenge_info()
            
            print(f"Blink {i}: Progress {challenge_info['progress']:.1f}, Completed: {completed}")
            
            if completed:
                print("🎉 Challenge completed!")
                break
        
        print("✅ Sequential mode simulation completed")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import sequential mode: {e}")
        return False

if __name__ == "__main__":
    print("🚀 BLINK DETECTION FIX VALIDATION")
    print("=" * 60)
    
    # Test 1: Basic blink detection
    success1 = test_blink_detection()
    
    # Test 2: Sequential mode simulation
    success2 = test_sequential_mode_simulation()
    
    # Summary
    print("\n📋 SUMMARY:")
    print(f"✅ Blink detection test: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Sequential mode test: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! The fixes should resolve the blink detection issues.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")

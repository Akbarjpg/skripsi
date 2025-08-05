#!/usr/bin/env python3
"""
Headless liveness detection test (no GUI display)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.landmark_detection import LivenessVerifier
import cv2
import numpy as np
import time

def test_headless_liveness():
    """Test liveness detection without GUI (headless mode)"""
    print("📷 TESTING LIVENESS DETECTION (HEADLESS MODE)")
    print("=" * 60)
    print("This test captures frames from camera but doesn't display them")
    print("Perfect for systems without GUI support!")
    print()
    print("🎯 Test will run for 30 seconds and show results...")
    print("⏰ Starting in 3 seconds...")
    time.sleep(3)
    
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
        print("🔴 Starting headless liveness detection...")
        print("📊 Live results will be shown below:")
        print("-" * 60)
        
        frame_count = 0
        start_time = time.time()
        test_duration = 30  # 30 seconds
        
        # Track best scores
        best_score = 0
        best_status = "NO_FACE"
        total_blinks = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
                
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Stop after test duration
            if elapsed > test_duration:
                break
            
            # Process every 10th frame for better performance
            if frame_count % 10 == 0:
                # Process frame for liveness
                results = verifier.process_frame(frame)
                
                if results['landmarks_detected']:
                    score = results['liveness_score']
                    status = results['liveness_status']
                    is_live = results['is_live']
                    blinks = results['blink_count']
                    
                    # Update best score
                    if score > best_score:
                        best_score = score
                        best_status = status
                    
                    total_blinks = blinks
                    
                    # Print live results every 2 seconds
                    if frame_count % 60 == 0:  # Every 60 frames (roughly 2 seconds)
                        remaining = test_duration - elapsed
                        print(f"⏱️  {remaining:.0f}s left | "
                              f"Score: {score:.1f}/100 | "
                              f"Status: {status} | "
                              f"Live: {'✅' if is_live else '❌'} | "
                              f"Blinks: {blinks}")
                        
                        # Show detailed metrics
                        if frame_count % 120 == 0:  # Every 4 seconds
                            ear_avg = (results['ear_left'] + results['ear_right']) / 2
                            print(f"   📊 EAR: {ear_avg:.3f} | "
                                  f"MAR: {results['mar']:.3f} | "
                                  f"Mouth: {'Open' if results['mouth_open'] else 'Closed'} | "
                                  f"Head: {'Moving' if results['head_movement'] else 'Still'}")
                
                else:
                    # No face detected
                    if frame_count % 60 == 0:
                        remaining = test_duration - elapsed
                        print(f"⏱️  {remaining:.0f}s left | 👤 NO FACE DETECTED - Please look at camera")
        
        cap.release()
        
        # Final results
        print("-" * 60)
        print("🏁 TEST COMPLETED!")
        print("=" * 60)
        print("📊 FINAL RESULTS:")
        print(f"  🎯 Best Liveness Score: {best_score:.1f}/100")
        print(f"  📈 Best Status: {best_status}")
        print(f"  👁️  Total Blinks Detected: {total_blinks}")
        print(f"  ⏱️  Test Duration: {test_duration} seconds")
        print(f"  🖼️  Frames Processed: {frame_count // 10}")
        
        # Assessment
        print("\n🔍 ASSESSMENT:")
        if best_score >= 80:
            print("🟢 EXCELLENT: Strong liveness detection! System working perfectly.")
        elif best_score >= 60:
            print("🟡 GOOD: Decent liveness detection. Try more natural movements.")
        elif best_score >= 40:
            print("🟠 FAIR: Basic liveness detected. Blink more and move head naturally.")
        elif best_score >= 20:
            print("🔴 POOR: Weak liveness signal. Are you looking at the camera?")
        else:
            print("❌ FAILED: No significant liveness detected. Check camera/lighting.")
        
        print("\n💡 TIPS FOR BETTER SCORES:")
        print("  • Blink naturally (not forced)")
        print("  • Turn head slowly left and right")
        print("  • Ensure good lighting on your face")
        print("  • Look directly at camera")
        print("  • Avoid holding the camera/device")
        
        return {
            'best_score': best_score,
            'best_status': best_status,
            'total_blinks': total_blinks,
            'frames_processed': frame_count // 10
        }
        
    except Exception as e:
        print(f"❌ Error in headless camera test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_headless_liveness()
    if result:
        print(f"\n✅ Test completed successfully with score: {result['best_score']:.1f}/100")

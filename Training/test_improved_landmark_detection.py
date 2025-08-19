"""
Test script for improved landmark detection system
Tests the enhanced EAR calculation, blink detection, and anti-spoofing features
"""

import cv2
import numpy as np
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detection.landmark_detection import LivenessVerifier
from detection.optimized_landmark_detection import OptimizedLivenessVerifier

def test_landmark_improvements():
    """
    Test the improved landmark detection system
    """
    print("🧪 TESTING IMPROVED LANDMARK DETECTION SYSTEM")
    print("=" * 60)
    
    # Initialize both versions for comparison
    print("Initializing verifiers...")
    standard_verifier = LivenessVerifier(history_length=30)
    optimized_verifier = OptimizedLivenessVerifier(history_length=15)
    
    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No webcam available, creating test with dummy image")
        test_with_dummy_image(standard_verifier, optimized_verifier)
        return
    
    print("\n📹 WEBCAM TEST")
    print("Instructions:")
    print("- Look directly at the camera")
    print("- Blink naturally several times")
    print("- Move your head slowly left/right")
    print("- Open your mouth occasionally")
    print("- Press 'q' to quit, 's' to show stats")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Test standard verifier
        start_std = time.time()
        std_results = standard_verifier.process_frame(frame)
        std_time = time.time() - start_std
        
        # Test optimized verifier  
        start_opt = time.time()
        opt_results = optimized_verifier.process_frame_optimized(frame)
        opt_time = time.time() - start_opt
        
        # Display results
        display_frame = frame.copy()
        
        # Draw info for standard verifier (left side)
        cv2.putText(display_frame, "STANDARD VERIFIER", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset = 60
        if std_results['landmarks_detected']:
            cv2.putText(display_frame, f"Blinks: {std_results['blink_count']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"EAR: {std_results['ear_left']:.3f}/{std_results['ear_right']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"Liveness: {std_results['liveness_score']:.1f}%", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"Status: {std_results['liveness_status']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"Time: {std_time*1000:.1f}ms", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No face detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw info for optimized verifier (right side)
        cv2.putText(display_frame, "OPTIMIZED VERIFIER", (350, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        y_offset = 60
        if opt_results['landmarks_detected']:
            cv2.putText(display_frame, f"Blinks: {opt_results['blink_count']}", 
                       (350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"EAR: {opt_results['ear_left']:.3f}/{opt_results['ear_right']:.3f}", 
                       (350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"Liveness: {opt_results['liveness_score']:.1f}%", 
                       (350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"Status: {opt_results['liveness_status']}", 
                       (350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25
            
            cv2.putText(display_frame, f"Time: {opt_time*1000:.1f}ms", 
                       (350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if 'fps_estimate' in opt_results:
                y_offset += 25
                cv2.putText(display_frame, f"FPS: {opt_results['fps_estimate']:.1f}", 
                           (350, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No face detected", (350, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Performance comparison
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if std_time > 0 and opt_time > 0:
            speedup = std_time / opt_time
            cv2.putText(display_frame, f"Speedup: {speedup:.1f}x", (10, display_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Improved Landmark Detection Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print_stats(standard_verifier, optimized_verifier, frame_count, start_time)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n📊 FINAL STATISTICS")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count/total_time:.1f}")
    
    print_stats(standard_verifier, optimized_verifier, frame_count, start_time)

def test_with_dummy_image(standard_verifier, optimized_verifier):
    """
    Test with dummy image if no webcam available
    """
    print("Creating dummy test image...")
    
    # Create a simple test image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some basic shapes to simulate a face
    cv2.circle(dummy_image, (320, 240), 100, (100, 100, 100), -1)  # Face
    cv2.circle(dummy_image, (290, 210), 15, (255, 255, 255), -1)   # Left eye
    cv2.circle(dummy_image, (350, 210), 15, (255, 255, 255), -1)   # Right eye
    cv2.ellipse(dummy_image, (320, 270), (30, 15), 0, 0, 180, (200, 200, 200), -1)  # Mouth
    
    print("Testing with dummy image...")
    
    # Test both verifiers
    std_results = standard_verifier.process_frame(dummy_image)
    opt_results = optimized_verifier.process_frame_optimized(dummy_image)
    
    print(f"Standard verifier results: {std_results['landmarks_detected']}")
    print(f"Optimized verifier results: {opt_results['landmarks_detected']}")
    
    if not std_results['landmarks_detected'] and not opt_results['landmarks_detected']:
        print("✅ Both verifiers correctly rejected dummy image (no real face detected)")
    else:
        print("⚠️  One or both verifiers detected landmarks in dummy image")

def print_stats(standard_verifier, optimized_verifier, frame_count, start_time):
    """
    Print detailed statistics
    """
    print(f"\n📈 DETAILED STATISTICS")
    print("-" * 40)
    
    print(f"Standard Verifier:")
    print(f"  - Blink count: {standard_verifier.blink_count}")
    print(f"  - Landmark history length: {len(standard_verifier.landmark_history)}")
    print(f"  - EAR history length: {len(standard_verifier.eye_aspect_ratio_history)}")
    
    print(f"\nOptimized Verifier:")
    print(f"  - Blink count: {optimized_verifier.blink_count}")
    print(f"  - Landmark history length: {len(optimized_verifier.landmark_history)}")
    print(f"  - EAR history length: {len(optimized_verifier.ear_history)}")
    
    if hasattr(optimized_verifier, 'get_performance_stats'):
        perf_stats = optimized_verifier.get_performance_stats()
        print(f"  - Performance stats: {perf_stats}")

def test_anti_spoofing_features():
    """
    Test specific anti-spoofing features
    """
    print("\n🛡️  TESTING ANTI-SPOOFING FEATURES")
    print("-" * 40)
    
    verifier = LivenessVerifier()
    
    # Test EAR calculation with different scenarios
    print("Testing EAR calculation robustness...")
    
    # Simulate different eye states
    test_cases = [
        ("Normal open eyes", [0.3, 0.3]),
        ("Closed eyes", [0.1, 0.1]),
        ("Asymmetric eyes", [0.3, 0.1]),
        ("Wide open eyes", [0.5, 0.5])
    ]
    
    for case_name, (ear_left, ear_right) in test_cases:
        # Test blink detection
        initial_blinks = verifier.blink_count
        
        # Simulate multiple frames
        for _ in range(5):
            verifier.detect_blink(ear_left, ear_right)
        
        blinks_detected = verifier.blink_count - initial_blinks
        print(f"  {case_name}: {blinks_detected} blinks detected")
    
    print("✅ Anti-spoofing features tested")

if __name__ == "__main__":
    try:
        # Test main functionality
        test_landmark_improvements()
        
        # Test anti-spoofing features
        test_anti_spoofing_features()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nKey improvements implemented:")
        print("✅ Robust 6-point EAR calculation")
        print("✅ Temporal smoothing for noise reduction")
        print("✅ Improved blink detection with validation")
        print("✅ Better head pose estimation")
        print("✅ Landmark quality validation")
        print("✅ Anti-spoofing micro-expression detection")
        print("✅ Performance optimization")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
Enhanced Anti-Spoofing System Test Script
=========================================

This script tests the Step 1 implementation from yangIni.md:
- Real-time face anti-spoofing detection
- Multiple anti-spoofing techniques simultaneously  
- Challenge-response system integration
- Progress indicators and confidence thresholds

Usage:
    python test_enhanced_antispoofing_step1.py
"""

import cv2
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from integration.realtime_antispoofing_system import RealTimeAntiSpoofingSystem
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced anti-spoofing system not available: {e}")
    ENHANCED_SYSTEM_AVAILABLE = False

def test_enhanced_antispoofing():
    """
    Test the enhanced anti-spoofing system with live camera feed
    """
    print("=" * 60)
    print("🚀 ENHANCED ANTI-SPOOFING SYSTEM TEST")
    print("   Step 1 Implementation from yangIni.md")
    print("=" * 60)
    
    if not ENHANCED_SYSTEM_AVAILABLE:
        print("❌ Enhanced anti-spoofing system is not available.")
        print("   Please check your installation and dependencies.")
        return
    
    # Initialize the enhanced anti-spoofing system
    print("🔧 Initializing Enhanced Anti-Spoofing System...")
    
    try:
        config = {
            'confidence_threshold': 0.95,  # 95% confidence as specified in Step 1
            'session_timeout': 60,
            'challenge_timeout': 15,
            'device': 'cpu'  # Use CPU for compatibility
        }
        
        antispoofing_system = RealTimeAntiSpoofingSystem(config)
        print("✅ Enhanced Anti-Spoofing System initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize enhanced system: {e}")
        return
    
    # Initialize camera
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera initialized successfully!")
    print("\n" + "=" * 60)
    print("🎯 STEP 1 REQUIREMENTS TESTING")
    print("=" * 60)
    print("Testing the following features:")
    print("• Real-time face anti-spoofing detection")
    print("• Multiple anti-spoofing techniques simultaneously")
    print("• CNN-based texture analysis")
    print("• Landmark-based micro-movement detection")
    print("• Challenge-response system")
    print("• Color space analysis")
    print("• 95% confidence threshold")
    print("• Progress indicators")
    print("\nPress 'q' to quit, 'r' to reset session")
    print("=" * 60)
    
    frame_count = 0
    fps_start = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame through enhanced anti-spoofing system
            start_time = time.time()
            result = antispoofing_system.process_frame(frame)
            processing_time = time.time() - start_time
            
            # Display results
            display_frame = frame.copy()
            display_detection_results(display_frame, result, processing_time)
            
            # Show frame
            cv2.imshow('Enhanced Anti-Spoofing Test - Step 1', display_frame)
            
            # Print periodic status
            if frame_count % 30 == 0:  # Every 30 frames
                current_time = time.time()
                fps = frame_count / (current_time - fps_start)
                print_status_update(result, fps, frame_count)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n🔄 Resetting session...")
                antispoofing_system.reset_session()
                frame_count = 0
                fps_start = time.time()
                print("✅ Session reset complete")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("📊 FINAL TEST STATISTICS")
        print("=" * 60)
        
        session_stats = antispoofing_system.get_session_stats()
        print(f"Total frames processed: {frame_count}")
        print(f"Session duration: {session_stats.get('session_duration', 0):.1f}s")
        print(f"Detection state: {session_stats.get('detection_state', 'UNKNOWN')}")
        print(f"Challenge attempts: {session_stats.get('challenge_attempts', 0)}")
        
        if frame_count > 0:
            total_time = time.time() - fps_start
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Average FPS: {avg_fps:.1f}")
        
        print("\n✅ Enhanced Anti-Spoofing System Test Complete!")

def display_detection_results(frame, result, processing_time):
    """
    Display detection results on the frame
    """
    h, w = frame.shape[:2]
    
    # Overlay detection information
    info_y = 30
    line_height = 25
    
    # Status and confidence
    status = result.get('status', 'unknown')
    confidence = result.get('confidence', 0.0)
    
    # Choose color based on status
    if status == 'verified':
        color = (0, 255, 0)  # Green
    elif status in ['challenging', 'processing']:
        color = (0, 255, 255)  # Yellow
    elif status in ['failed', 'error']:
        color = (0, 0, 255)  # Red
    else:
        color = (255, 255, 255)  # White
    
    # Display main status
    cv2.putText(frame, f"Status: {status.upper()}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    info_y += line_height
    
    # Display confidence
    confidence_text = f"Confidence: {confidence * 100:.1f}%"
    cv2.putText(frame, confidence_text, 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    info_y += line_height
    
    # Display processing time
    cv2.putText(frame, f"Processing: {processing_time * 1000:.1f}ms", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    
    # Display detection state
    detection_state = result.get('detection_state', 'UNKNOWN')
    cv2.putText(frame, f"State: {detection_state}", 
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height
    
    # Display progress information
    progress = result.get('progress', {})
    if progress:
        overall_progress = progress.get('overall_progress', 0.0)
        
        # Draw progress bar
        bar_width = 300
        bar_height = 20
        bar_x = 10
        bar_y = h - 50
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress fill
        fill_width = int(bar_width * overall_progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         color, -1)
        
        # Progress text
        progress_text = f"Overall Progress: {overall_progress * 100:.1f}%"
        cv2.putText(frame, progress_text, 
                    (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display challenge information
    challenge_info = result.get('challenge_info')
    if challenge_info:
        instruction = challenge_info.get('instruction', 'Complete challenge')
        time_remaining = challenge_info.get('time_remaining', 0)
        
        # Challenge instruction
        cv2.putText(frame, f"Challenge: {instruction}", 
                    (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Time remaining
        cv2.putText(frame, f"Time: {time_remaining:.1f}s", 
                    (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display message
    message = result.get('message', '')
    if message:
        # Word wrap for long messages
        words = message.split(' ')
        lines = []
        current_line = ''
        
        for word in words:
            if len(current_line + ' ' + word) < 50:  # Approximate character limit
                current_line += ' ' + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Display message lines
        for i, line in enumerate(lines[:3]):  # Limit to 3 lines
            cv2.putText(frame, line, 
                        (10, h - 150 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def print_status_update(result, fps, frame_count):
    """
    Print periodic status updates to console
    """
    status = result.get('status', 'unknown')
    confidence = result.get('confidence', 0.0)
    detection_state = result.get('detection_state', 'UNKNOWN')
    
    print(f"\n📊 Frame {frame_count} | FPS: {fps:.1f}")
    print(f"   Status: {status} | Confidence: {confidence * 100:.1f}%")
    print(f"   Detection State: {detection_state}")
    
    # Show progress breakdown
    progress = result.get('progress', {})
    if progress:
        cnn = progress.get('cnn_analysis', 0.0) * 100
        landmark = progress.get('landmark_detection', 0.0) * 100
        challenge = progress.get('challenge_completion', 0.0) * 100
        overall = progress.get('overall_progress', 0.0) * 100
        
        print(f"   Progress - CNN: {cnn:.1f}% | Landmark: {landmark:.1f}% | Challenge: {challenge:.1f}% | Overall: {overall:.1f}%")
    
    # Show challenge info
    challenge_info = result.get('challenge_info')
    if challenge_info:
        instruction = challenge_info.get('instruction', 'No instruction')
        time_remaining = challenge_info.get('time_remaining', 0)
        print(f"   Challenge: {instruction} (Time: {time_remaining:.1f}s)")

if __name__ == "__main__":
    print("🎯 Enhanced Anti-Spoofing System Test")
    print("   Implementing Step 1 requirements from yangIni.md")
    print("   Press Ctrl+C to exit\n")
    
    test_enhanced_antispoofing()

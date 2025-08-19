#!/usr/bin/env python3
"""
Test script untuk enhanced challenge-response system
Testing Phase 2 improvements dengan proper anti-spoofing
"""

import cv2
import time
import numpy as np
from typing import Dict, Optional

# Import enhanced challenge system
from src.challenge.challenge_response import (
    ChallengeResponseSystem, ChallengeDifficulty, ChallengeType,
    BlinkChallenge, SmileChallenge, HeadMovementChallenge
)

# Import enhanced landmark detection
from src.detection.landmark_detection import EnhancedLivenessVerifier

def test_enhanced_challenge_system():
    """Test enhanced challenge system dengan real camera input"""
    print("=" * 60)
    print("🚀 ENHANCED CHALLENGE-RESPONSE SYSTEM TEST")
    print("   Phase 2: Improved Challenge System")
    print("=" * 60)
    
    # Initialize systems
    print("🔧 Initializing enhanced systems...")
    verifier = EnhancedLivenessVerifier(
        ear_threshold=0.25,
        ear_consec_frames=3,
        mouth_threshold=0.6,
        confidence_threshold=0.7
    )
    
    challenge_system = ChallengeResponseSystem()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Systems initialized successfully!")
    print("\n📋 Test Controls:")
    print("   'n' - Start new random challenge")
    print("   's' - Start sequence challenge") 
    print("   'e' - Start easy challenge")
    print("   'm' - Start medium challenge")
    print("   'h' - Start hard challenge")
    print("   'q' - Quit test")
    print("\n🎯 Challenge Types Available:")
    print("   • Enhanced Blink Detection (temporal smoothing)")
    print("   • Smart Smile Detection (mouth aspect ratio)")
    print("   • Directional Head Movement (left/right/up/down)")
    print("   • Sequential Multi-step Challenges")
    print("=" * 60)
    
    frame_count = 0
    test_start_time = time.time()
    challenge_stats = {
        'total_challenges': 0,
        'successful_challenges': 0,
        'failed_challenges': 0,
        'by_difficulty': {'EASY': 0, 'MEDIUM': 0, 'HARD': 0},
        'by_type': {}
    }
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process with enhanced landmark detection
            detection_results = verifier.process_frame(frame)
            
            # Process challenge if active
            challenge_result = None
            if challenge_system.current_challenge:
                challenge_result = challenge_system.process_frame(detection_results)
                
                # Update stats if challenge completed
                if challenge_result:
                    challenge_stats['total_challenges'] += 1
                    if challenge_result.success:
                        challenge_stats['successful_challenges'] += 1
                    else:
                        challenge_stats['failed_challenges'] += 1
                    
                    # Track by difficulty and type
                    difficulty = challenge_result.difficulty.value
                    challenge_type = challenge_result.challenge_type.value
                    
                    challenge_stats['by_difficulty'][difficulty] = challenge_stats['by_difficulty'].get(difficulty, 0) + 1
                    challenge_stats['by_type'][challenge_type] = challenge_stats['by_type'].get(challenge_type, 0) + 1
                    
                    # Print result
                    print(f"\\n{'✅' if challenge_result.success else '❌'} Challenge Result:")
                    print(f"   Type: {challenge_type}")
                    print(f"   Difficulty: {difficulty}")
                    print(f"   Success: {challenge_result.success}")
                    print(f"   Confidence: {challenge_result.confidence:.2f}")
                    print(f"   Quality Score: {challenge_result.quality_score:.2f}")
                    print(f"   Intentional Score: {challenge_result.intentional_score:.2f}")
                    print(f"   Duration: {challenge_result.duration:.1f}s")
            
            # Enhanced visual feedback
            display_frame = frame.copy()
            _draw_enhanced_interface(display_frame, detection_results, challenge_system, challenge_stats, current_time - test_start_time)
            
            cv2.imshow('Enhanced Challenge-Response System', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                if not challenge_system.current_challenge:
                    challenge_system.start_challenge('random')
            elif key == ord('s'):
                if not challenge_system.current_challenge:
                    challenge_system.start_challenge('sequence')
            elif key == ord('e'):
                if not challenge_system.current_challenge:
                    challenge_system.start_challenge('random', ChallengeDifficulty.EASY)
            elif key == ord('m'):
                if not challenge_system.current_challenge:
                    challenge_system.start_challenge('random', ChallengeDifficulty.MEDIUM)
            elif key == ord('h'):
                if not challenge_system.current_challenge:
                    challenge_system.start_challenge('random', ChallengeDifficulty.HARD)
            
            # Show FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - test_start_time
                fps = frame_count / elapsed
                print(f"📊 Performance: {fps:.1f} FPS | Frames: {frame_count}")
    
    except KeyboardInterrupt:
        print("\\n⏹️  Test interrupted by user")
    
    finally:
        # Cleanup and final statistics
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - test_start_time
        print("\\n" + "=" * 60)
        print("📈 ENHANCED CHALLENGE SYSTEM TEST RESULTS")
        print("=" * 60)
        print(f"⏱️  Total Test Duration: {total_time:.1f} seconds")
        print(f"🎬 Frames Processed: {frame_count}")
        print(f"📊 Average FPS: {frame_count / total_time:.1f}")
        
        print(f"\\n🎯 Challenge Statistics:")
        print(f"   Total Challenges: {challenge_stats['total_challenges']}")
        print(f"   Successful: {challenge_stats['successful_challenges']}")
        print(f"   Failed: {challenge_stats['failed_challenges']}")
        
        if challenge_stats['total_challenges'] > 0:
            success_rate = (challenge_stats['successful_challenges'] / challenge_stats['total_challenges']) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
        
        print(f"\\n📊 By Difficulty:")
        for difficulty, count in challenge_stats['by_difficulty'].items():
            if count > 0:
                print(f"   {difficulty}: {count}")
        
        print(f"\\n📊 By Type:")
        for challenge_type, count in challenge_stats['by_type'].items():
            if count > 0:
                print(f"   {challenge_type}: {count}")
        
        print("\\n✅ Enhanced challenge system test completed!")
        print("=" * 60)

def _draw_enhanced_interface(frame, detection_results: Dict, challenge_system, stats: Dict, elapsed_time: float):
    """Draw enhanced interface dengan improved visual feedback"""
    height, width = frame.shape[:2]
    
    # Enhanced challenge status display
    challenge = challenge_system.current_challenge
    if challenge:
        # Challenge header
        header_text = f"🎯 {challenge.challenge_type.value.title()} Challenge"
        cv2.putText(frame, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Difficulty indicator
        difficulty_color = {
            'EASY': (0, 255, 0),
            'MEDIUM': (0, 255, 255), 
            'HARD': (0, 0, 255)
        }
        color = difficulty_color.get(challenge.difficulty.value, (255, 255, 255))
        cv2.putText(frame, f"Difficulty: {challenge.difficulty.value}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Description
        cv2.putText(frame, challenge.description, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress indicators
        if hasattr(challenge, 'detected_blinks') and challenge.challenge_type == ChallengeType.BLINK:
            progress_text = f"Blinks: {challenge.detected_blinks}/{challenge.required_blinks}"
            cv2.putText(frame, progress_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif hasattr(challenge, 'detected_smiles') and challenge.challenge_type == ChallengeType.SMILE:
            progress_text = f"Smiles: {challenge.detected_smiles}/{challenge.required_smiles}"
            cv2.putText(frame, progress_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif hasattr(challenge, 'completed_directions') and challenge.challenge_type == ChallengeType.HEAD_MOVEMENT:
            progress_text = f"Directions: {len(challenge.completed_directions)}/{len(challenge.required_directions)}"
            cv2.putText(frame, progress_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show required directions
            directions_text = f"Required: {', '.join(challenge.required_directions)}"
            cv2.putText(frame, directions_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Time remaining
        time_remaining = max(0, challenge.duration - challenge.get_elapsed_time())
        time_text = f"Time: {time_remaining:.1f}s"
        time_color = (0, 0, 255) if time_remaining < 5 else (0, 255, 0)
        cv2.putText(frame, time_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
        
        # Progress bar
        progress = min(1.0, challenge.get_elapsed_time() / challenge.duration)
        bar_width = 300
        bar_height = 20
        cv2.rectangle(frame, (10, 160), (10 + bar_width, 160 + bar_height), (100, 100, 100), 2)
        cv2.rectangle(frame, (10, 160), (10 + int(bar_width * progress), 160 + bar_height), time_color, -1)
    
    else:
        cv2.putText(frame, "Press 'n'=Random, 's'=Sequence, 'e'=Easy, 'm'=Medium, 'h'=Hard", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Enhanced detection info
    if detection_results.get('landmarks_detected', False):
        y_offset = 220
        
        # Detection quality indicators
        confidence = detection_results.get('detection_confidence', 0)
        quality_text = f"Detection Quality: {confidence:.2f}"
        quality_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.putText(frame, quality_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
        
        # EAR values
        ear_left = detection_results.get('ear_left', 0)
        ear_right = detection_results.get('ear_right', 0)
        cv2.putText(frame, f"EAR: L:{ear_left:.3f} R:{ear_right:.3f}", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Head pose
        pitch = detection_results.get('head_pitch', 0)
        yaw = detection_results.get('head_yaw', 0)
        cv2.putText(frame, f"Head: Yaw:{yaw:.0f}° Pitch:{pitch:.0f}°", 
                   (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Blink count
        blink_count = detection_results.get('blink_count', 0)
        cv2.putText(frame, f"Total Blinks: {blink_count}", 
                   (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Test statistics
    cv2.putText(frame, f"Test Time: {elapsed_time:.0f}s", (width - 200, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Challenges: {stats['total_challenges']}", (width - 200, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if stats['total_challenges'] > 0:
        success_rate = (stats['successful_challenges'] / stats['total_challenges']) * 100
        cv2.putText(frame, f"Success: {success_rate:.0f}%", (width - 200, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

if __name__ == "__main__":
    test_enhanced_challenge_system()

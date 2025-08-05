"""
Challenge-Response System untuk Face Anti-Spoofing
Memberikan tantangan acak dan memverifikasi respons real-time
"""

import random
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import logging
from dataclasses import dataclass, asdict
from collections import deque

class ChallengeType(Enum):
    """Enum untuk tipe-tipe challenge"""
    BLINK = "blink"
    MOUTH_OPEN = "mouth_open"
    HEAD_LEFT = "head_left"
    HEAD_RIGHT = "head_right"
    HEAD_UP = "head_up"
    HEAD_DOWN = "head_down"
    SMILE = "smile"
    SEQUENCE = "sequence"

@dataclass
class ChallengeResult:
    """Data class untuk hasil challenge"""
    challenge_id: str
    challenge_type: ChallengeType
    success: bool
    response_time: float
    confidence_score: float
    details: Dict
    timestamp: float
    
    def to_dict(self):
        """Convert to JSON-serializable dictionary"""
        from ..utils import json_serializable
        return json_serializable({
            'challenge_id': self.challenge_id,
            'challenge_type': self.challenge_type.value,  # Convert enum to string
            'success': self.success,
            'response_time': float(self.response_time),
            'confidence_score': float(self.confidence_score),
            'details': self.details,
            'timestamp': float(self.timestamp)
        })

class Challenge:
    """Base class untuk semua challenge"""
    
    def __init__(self, challenge_id: str, challenge_type: ChallengeType, 
                 duration: float = 3.0, description: str = ""):
        self.challenge_id = challenge_id
        self.challenge_type = challenge_type
        self.duration = duration
        self.description = description
        self.start_time = None
        self.completed = False
        self.success = False
        self.response_data = []
        
    def start(self):
        """Mulai challenge"""
        self.start_time = time.time()
        self.completed = False
        self.success = False
        self.response_data = []
        
    def is_expired(self):
        """Check apakah challenge sudah expired"""
        if self.start_time is None:
            return False
        return time.time() - self.start_time > self.duration
        
    def process_response(self, detection_results: Dict) -> bool:
        """
        Process response dari facial detection
        Returns True jika challenge completed
        """
        raise NotImplementedError
        
    def get_result(self) -> ChallengeResult:
        """Get hasil challenge"""
        response_time = time.time() - self.start_time if self.start_time else 0
        
        return ChallengeResult(
            challenge_id=self.challenge_id,
            challenge_type=self.challenge_type,
            success=self.success,
            response_time=response_time,
            confidence_score=self._calculate_confidence(),
            details=self._get_details(),
            timestamp=time.time()
        )
        
    def _calculate_confidence(self) -> float:
        """Calculate confidence score berdasarkan response quality"""
        return 1.0 if self.success else 0.0
        
    def _get_details(self) -> Dict:
        """Get detail information tentang response"""
        return {'response_data_length': len(self.response_data)}

class BlinkChallenge(Challenge):
    """Challenge untuk kedipan mata"""
    
    def __init__(self, challenge_id: str, required_blinks: int = 2, duration: float = 5.0):
        super().__init__(
            challenge_id, 
            ChallengeType.BLINK, 
            duration,
            f"Kedip {required_blinks} kali dalam {duration} detik"
        )
        self.required_blinks = required_blinks
        self.initial_blink_count = None
        
    def process_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_blinks = detection_results.get('blink_count', 0)
        
        # Set initial blink count
        if self.initial_blink_count is None:
            self.initial_blink_count = current_blinks
            
        self.response_data.append({
            'timestamp': time.time(),
            'blink_count': current_blinks,
            'ear_left': detection_results.get('ear_left', 0),
            'ear_right': detection_results.get('ear_right', 0)
        })
        
        # Check jika sudah mencapai target
        blinks_performed = current_blinks - self.initial_blink_count
        if blinks_performed >= self.required_blinks:
            self.success = True
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
        
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        # Calculate confidence berdasarkan EAR consistency
        ear_values = [(d['ear_left'] + d['ear_right']) / 2 for d in self.response_data]
        if len(ear_values) < 5:
            return 0.5
            
        # Check for clear blink patterns (low EAR values)
        min_ear = min(ear_values)
        max_ear = max(ear_values)
        ear_range = max_ear - min_ear
        
        # Good blink should have significant EAR variation
        confidence = min(1.0, ear_range / 0.3)  # 0.3 is typical EAR range for blink
        return confidence

class MouthOpenChallenge(Challenge):
    """Challenge untuk membuka mulut"""
    
    def __init__(self, challenge_id: str, duration: float = 3.0, 
                 open_threshold: float = 0.6, min_open_time: float = 1.0):
        super().__init__(
            challenge_id,
            ChallengeType.MOUTH_OPEN,
            duration,
            f"Buka mulut selama {min_open_time} detik"
        )
        self.open_threshold = open_threshold
        self.min_open_time = min_open_time
        self.mouth_open_start = None
        self.mouth_open_duration = 0.0
        
    def process_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_time = time.time()
        mar = detection_results.get('mar', 0)
        mouth_open = detection_results.get('mouth_open', False)
        
        self.response_data.append({
            'timestamp': current_time,
            'mar': mar,
            'mouth_open': mouth_open
        })
        
        # Track mouth open duration
        if mouth_open and mar > self.open_threshold:
            if self.mouth_open_start is None:
                self.mouth_open_start = current_time
            else:
                self.mouth_open_duration = current_time - self.mouth_open_start
        else:
            self.mouth_open_start = None
            self.mouth_open_duration = 0.0
            
        # Check success
        if self.mouth_open_duration >= self.min_open_time:
            self.success = True
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
        
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        # Calculate confidence berdasarkan MAR consistency
        mar_values = [d['mar'] for d in self.response_data]
        max_mar = max(mar_values)
        
        # Higher MAR indicates better mouth opening
        confidence = min(1.0, max_mar / 1.0)  # Normalize to max expected MAR
        return confidence

class HeadMovementChallenge(Challenge):
    """Challenge untuk gerakan kepala"""
    
    def __init__(self, challenge_id: str, direction: str, duration: float = 4.0,
                 angle_threshold: float = 20.0, min_hold_time: float = 1.0):
        super().__init__(
            challenge_id,
            getattr(ChallengeType, f"HEAD_{direction.upper()}"),
            duration,
            f"Putar kepala ke {direction} dan tahan selama {min_hold_time} detik"
        )
        self.direction = direction.lower()
        self.angle_threshold = angle_threshold
        self.min_hold_time = min_hold_time
        self.target_reached = False
        self.hold_start_time = None
        self.hold_duration = 0.0
        
    def process_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        head_pose = detection_results.get('head_pose')
        if head_pose is None:
            return False
            
        current_time = time.time()
        yaw = head_pose['yaw']
        pitch = head_pose['pitch']
        
        self.response_data.append({
            'timestamp': current_time,
            'yaw': yaw,
            'pitch': pitch,
            'roll': head_pose['roll']
        })
        
        # Check direction
        target_reached = False
        if self.direction == 'left' and yaw > self.angle_threshold:
            target_reached = True
        elif self.direction == 'right' and yaw < -self.angle_threshold:
            target_reached = True
        elif self.direction == 'up' and pitch < -self.angle_threshold:
            target_reached = True
        elif self.direction == 'down' and pitch > self.angle_threshold:
            target_reached = True
            
        # Track hold duration
        if target_reached:
            if self.hold_start_time is None:
                self.hold_start_time = current_time
            else:
                self.hold_duration = current_time - self.hold_start_time
        else:
            self.hold_start_time = None
            self.hold_duration = 0.0
            
        # Check success
        if self.hold_duration >= self.min_hold_time:
            self.success = True
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
        
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        # Calculate confidence berdasarkan angle consistency
        if self.direction in ['left', 'right']:
            angles = [abs(d['yaw']) for d in self.response_data]
        else:  # up, down
            angles = [abs(d['pitch']) for d in self.response_data]
            
        max_angle = max(angles)
        confidence = min(1.0, max_angle / (self.angle_threshold * 2))
        return confidence

class SequenceChallenge(Challenge):
    """Challenge berupa sequence of actions"""
    
    def __init__(self, challenge_id: str, sequence: List[ChallengeType], 
                 duration: float = 10.0):
        super().__init__(
            challenge_id,
            ChallengeType.SEQUENCE,
            duration,
            f"Lakukan: {' -> '.join([c.value for c in sequence])}"
        )
        self.sequence = sequence
        self.current_step = 0
        self.step_results = []
        
    def process_response(self, detection_results: Dict) -> bool:
        if self.current_step >= len(self.sequence):
            self.success = True
            self.completed = True
            return True
            
        current_challenge_type = self.sequence[self.current_step]
        
        # Check current step completion
        step_completed = False
        
        if current_challenge_type == ChallengeType.BLINK:
            # Simple blink detection (could be enhanced)
            if detection_results.get('blink_count', 0) > len(self.step_results):
                step_completed = True
                
        elif current_challenge_type == ChallengeType.MOUTH_OPEN:
            if detection_results.get('mouth_open', False):
                step_completed = True
                
        elif current_challenge_type == ChallengeType.HEAD_LEFT:
            head_pose = detection_results.get('head_pose', {})
            if head_pose.get('yaw', 0) > 15:
                step_completed = True
                
        elif current_challenge_type == ChallengeType.HEAD_RIGHT:
            head_pose = detection_results.get('head_pose', {})
            if head_pose.get('yaw', 0) < -15:
                step_completed = True
        
        if step_completed:
            self.step_results.append({
                'step': self.current_step,
                'challenge_type': current_challenge_type,
                'timestamp': time.time(),
                'detection_results': detection_results.copy()
            })
            self.current_step += 1
            
        # Check overall completion
        if self.current_step >= len(self.sequence):
            self.success = True
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False

class ChallengeResponseSystem:
    """
    Main system untuk mengelola challenge-response
    """
    
    def __init__(self, anti_replay_window: float = 300.0):  # 5 minutes
        self.current_challenge = None
        self.challenge_history = deque(maxlen=100)
        self.anti_replay_window = anti_replay_window
        self.used_challenges = set()
        
    def generate_random_challenge(self) -> Challenge:
        """Generate random challenge"""
        challenge_types = [
            ChallengeType.BLINK,
            ChallengeType.MOUTH_OPEN,
            ChallengeType.HEAD_LEFT,
            ChallengeType.HEAD_RIGHT,
            ChallengeType.HEAD_UP,
            ChallengeType.HEAD_DOWN
        ]
        
        challenge_type = random.choice(challenge_types)
        challenge_id = f"{challenge_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Anti-replay: pastikan challenge tidak diulang dalam window tertentu
        while challenge_id in self.used_challenges:
            challenge_id = f"{challenge_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
            
        self.used_challenges.add(challenge_id)
        
        # Create specific challenge
        if challenge_type == ChallengeType.BLINK:
            return BlinkChallenge(challenge_id, required_blinks=random.randint(2, 4))
        elif challenge_type == ChallengeType.MOUTH_OPEN:
            return MouthOpenChallenge(challenge_id, min_open_time=random.uniform(1.0, 2.0))
        elif challenge_type == ChallengeType.HEAD_LEFT:
            return HeadMovementChallenge(challenge_id, 'left')
        elif challenge_type == ChallengeType.HEAD_RIGHT:
            return HeadMovementChallenge(challenge_id, 'right')
        elif challenge_type == ChallengeType.HEAD_UP:
            return HeadMovementChallenge(challenge_id, 'up')
        elif challenge_type == ChallengeType.HEAD_DOWN:
            return HeadMovementChallenge(challenge_id, 'down')
            
    def generate_sequence_challenge(self, num_steps: int = 3) -> SequenceChallenge:
        """Generate sequence challenge"""
        available_types = [
            ChallengeType.BLINK,
            ChallengeType.MOUTH_OPEN,
            ChallengeType.HEAD_LEFT,
            ChallengeType.HEAD_RIGHT
        ]
        
        sequence = random.sample(available_types, min(num_steps, len(available_types)))
        challenge_id = f"sequence_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return SequenceChallenge(challenge_id, sequence)
        
    def start_challenge(self, challenge_type: str = 'random') -> Challenge:
        """Start new challenge"""
        if challenge_type == 'random':
            challenge = self.generate_random_challenge()
        elif challenge_type == 'sequence':
            challenge = self.generate_sequence_challenge()
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
            
        challenge.start()
        self.current_challenge = challenge
        
        logging.info(f"Started challenge: {challenge.challenge_id} - {challenge.description}")
        return challenge
        
    def process_frame(self, detection_results: Dict) -> Optional[ChallengeResult]:
        """
        Process frame dengan detection results
        Returns ChallengeResult jika challenge completed
        """
        if self.current_challenge is None:
            return None
            
        # Process response
        completed = self.current_challenge.process_response(detection_results)
        
        if completed:
            result = self.current_challenge.get_result()
            self.challenge_history.append(result)
            self.current_challenge = None
            
            logging.info(f"Challenge completed: {result.challenge_id}, Success: {result.success}")
            return result
            
        return None
        
    def get_current_challenge_status(self) -> Optional[Dict]:
        """Get status challenge saat ini"""
        if self.current_challenge is None:
            return None
            
        elapsed_time = time.time() - self.current_challenge.start_time
        remaining_time = max(0, self.current_challenge.duration - elapsed_time)
        
        return {
            'challenge_id': self.current_challenge.challenge_id,
            'challenge_type': self.current_challenge.challenge_type.value,
            'description': self.current_challenge.description,
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'progress': min(1.0, elapsed_time / self.current_challenge.duration)
        }
        
    def cleanup_old_challenges(self):
        """Cleanup old challenges dari anti-replay protection"""
        current_time = time.time()
        cutoff_time = current_time - self.anti_replay_window
        
        # Remove old challenge IDs
        self.used_challenges = {
            cid for cid in self.used_challenges 
            if current_time - float(cid.split('_')[1]) < self.anti_replay_window
        }
        
    def get_challenge_statistics(self) -> Dict:
        """Get statistics dari completed challenges"""
        if not self.challenge_history:
            return {}
            
        total_challenges = len(self.challenge_history)
        successful_challenges = sum(1 for r in self.challenge_history if r.success)
        
        success_rate = successful_challenges / total_challenges if total_challenges > 0 else 0
        avg_response_time = np.mean([r.response_time for r in self.challenge_history])
        avg_confidence = np.mean([r.confidence_score for r in self.challenge_history])
        
        # Per-type statistics
        type_stats = {}
        for result in self.challenge_history:
            challenge_type = result.challenge_type.value
            if challenge_type not in type_stats:
                type_stats[challenge_type] = {'total': 0, 'success': 0, 'response_times': []}
                
            type_stats[challenge_type]['total'] += 1
            if result.success:
                type_stats[challenge_type]['success'] += 1
            type_stats[challenge_type]['response_times'].append(result.response_time)
        
        return {
            'total_challenges': total_challenges,
            'successful_challenges': successful_challenges,
            'success_rate': success_rate,
            'average_response_time': avg_response_time,
            'average_confidence': avg_confidence,
            'type_statistics': type_stats
        }

def test_challenge_response_system():
    """
    Test function untuk challenge-response system
    """
    from src.detection.landmark_detection import LivenessVerifier
    
    # Initialize systems
    verifier = LivenessVerifier()
    challenge_system = ChallengeResponseSystem()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Testing Challenge-Response System...")
    print("Instructions will appear on screen")
    print("Press 'n' for new challenge, 'q' to quit")
    
    current_challenge = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process facial landmarks
        detection_results = verifier.process_frame(frame)
        
        # Process challenge if active
        challenge_result = challenge_system.process_frame(detection_results)
        
        # Handle completed challenges
        if challenge_result:
            print(f"Challenge Result: {challenge_result.success}, "
                  f"Time: {challenge_result.response_time:.2f}s, "
                  f"Confidence: {challenge_result.confidence_score:.3f}")
            current_challenge = None
        
        # Draw challenge status
        challenge_status = challenge_system.get_current_challenge_status()
        
        if challenge_status:
            # Draw challenge description
            cv2.putText(frame, challenge_status['description'], (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Draw timer
            remaining = challenge_status['remaining_time']
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw progress bar
            progress = challenge_status['progress']
            bar_width = 300
            bar_height = 20
            cv2.rectangle(frame, (10, 90), (10 + bar_width, 90 + bar_height), (255, 255, 255), 2)
            cv2.rectangle(frame, (10, 90), (10 + int(bar_width * progress), 90 + bar_height), (0, 255, 0), -1)
            
        else:
            cv2.putText(frame, "Press 'n' for new challenge", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detection info
        if detection_results['landmarks_detected']:
            y_offset = 130
            cv2.putText(frame, f"Blinks: {detection_results['blink_count']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if detection_results['mouth_open']:
                cv2.putText(frame, "MOUTH OPEN", (10, y_offset + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if detection_results['head_pose']:
                pose = detection_results['head_pose']
                cv2.putText(frame, f"Head: Y:{pose['yaw']:.0f}° P:{pose['pitch']:.0f}°", 
                           (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Challenge-Response Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n') and challenge_status is None:
            # Start new random challenge
            challenge_system.start_challenge('random')
        elif key == ord('s') and challenge_status is None:
            # Start sequence challenge
            challenge_system.start_challenge('sequence')
    
    # Print statistics
    stats = challenge_system.get_challenge_statistics()
    print("\\n=== Challenge Statistics ===")
    for key, value in stats.items():
        if key != 'type_statistics':
            print(f"{key}: {value}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_challenge_response_system()

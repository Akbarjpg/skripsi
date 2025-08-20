"""
Enhanced Challenge-Response System untuk Face Anti-Spoofing
Sistem tantangan yang lebih robust untuk membedakan manusia asli vs foto/video/deepfake
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
import math

# Import audio feedback system
try:
    from .audio_feedback import AudioFeedbackSystem, ChallengeInstructions, AudioType
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio feedback not available")

# Import distance challenge
try:
    from .distance_challenge import DistanceChallenge
    DISTANCE_CHALLENGE_AVAILABLE = True
except ImportError:
    DISTANCE_CHALLENGE_AVAILABLE = False
    print("Warning: Distance challenge not available")

class ChallengeType(Enum):
    """Enhanced enum untuk tipe-tipe challenge anti-spoofing"""
    BLINK = "blink"
    MOUTH_OPEN = "mouth_open"
    HEAD_MOVEMENT = "head_movement"  # NEW: Head movement detection (up/down, left/right)
    SEQUENCE = "sequence"
    # NEW: Anti-spoofing specific challenges
    COVER_EYE = "cover_eye"
    MOVE_CLOSER = "move_closer"
    MOVE_FARTHER = "move_farther"
    LOOK_AROUND = "look_around"
    NOD = "nod"

class ChallengeDifficulty(Enum):
    """Challenge difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ANTI_SPOOFING = "anti_spoofing"

@dataclass
class ChallengeResult:
    """Enhanced data class untuk hasil challenge dengan detail analytics"""
    challenge_id: str
    challenge_type: ChallengeType
    difficulty: ChallengeDifficulty
    success: bool
    response_time: float
    confidence_score: float
    quality_score: float  # NEW: Quality of response
    intentional_score: float  # NEW: Intentional vs accidental movement
    details: Dict
    timestamp: float
    progress_data: List[Dict]  # NEW: Progress tracking data
    
    def to_dict(self):
        """Convert to JSON-serializable dictionary"""
        from ..utils import json_serializable
        return json_serializable({
            'challenge_id': self.challenge_id,
            'challenge_type': self.challenge_type.value,
            'difficulty': self.difficulty.value,
            'success': self.success,
            'response_time': float(self.response_time),
            'confidence_score': float(self.confidence_score),
            'quality_score': float(self.quality_score),
            'intentional_score': float(self.intentional_score),
            'details': self.details,
            'timestamp': float(self.timestamp),
            'progress_data': self.progress_data
        })

class Challenge:
    """Enhanced base class untuk semua challenge dengan progress tracking"""
    
    def __init__(self, challenge_id: str, challenge_type: ChallengeType, 
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 15.0, description: str = ""):  # Extended timeout to 15s
        self.challenge_id = challenge_id
        self.challenge_type = challenge_type
        self.difficulty = difficulty
        self.duration = duration
        self.description = description
        self.start_time = None
        self.completed = False
        self.success = False
        self.response_data = []
        self.progress_data = []  # NEW: Track progress for visual feedback
        self.quality_metrics = []  # NEW: Track response quality
        
        # Movement intentionality tracking
        self.movement_history = deque(maxlen=30)  # 30 frames of movement data
        self.intentional_threshold = 0.7  # Threshold for intentional movement
        
    def start(self):
        """Mulai challenge dengan enhanced tracking"""
        self.start_time = time.time()
        self.completed = False
        self.success = False
        self.response_data = []
        self.progress_data = []
        self.quality_metrics = []
        self.movement_history.clear()
        
    def is_expired(self):
        """Check apakah challenge sudah expired"""
        if self.start_time is None:
            return False
        return time.time() - self.start_time > self.duration
        
    def get_progress_percentage(self) -> float:
        """Get current progress percentage (0-100)"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return min(100.0, (elapsed / self.duration) * 100.0)
    
    def get_time_remaining(self) -> float:
        """Get remaining time in seconds"""
        if self.start_time is None:
            return self.duration
        elapsed = time.time() - self.start_time
        return max(0.0, self.duration - elapsed)
    
    def calculate_intentional_score(self) -> float:
        """
        Calculate if movement appears intentional vs accidental
        Returns score 0-1 where 1 = clearly intentional
        """
        if len(self.movement_history) < 10:
            return 0.5  # Not enough data
        
        movements = list(self.movement_history)
        
        # Analyze movement patterns
        # Intentional movements typically have:
        # 1. Sustained direction
        # 2. Gradual acceleration/deceleration
        # 3. Clear start and end points
        
        # Calculate movement consistency
        directions = []
        magnitudes = []
        
        for i in range(1, len(movements)):
            dx = movements[i]['x'] - movements[i-1]['x']
            dy = movements[i]['y'] - movements[i-1]['y']
            magnitude = math.sqrt(dx*dx + dy*dy)
            direction = math.atan2(dy, dx) if magnitude > 0.001 else 0
            
            directions.append(direction)
            magnitudes.append(magnitude)
        
        if not directions:
            return 0.5
        
        # Calculate directional consistency
        direction_variance = np.var(directions) if len(directions) > 1 else 0
        consistency_score = max(0, 1.0 - direction_variance / (math.pi * 0.5))
        
        # Calculate magnitude smoothness (intentional movements are smoother)
        magnitude_variance = np.var(magnitudes) if len(magnitudes) > 1 else 0
        smoothness_score = max(0, 1.0 - magnitude_variance * 10)
        
        # Combined intentional score
        intentional_score = (consistency_score * 0.6 + smoothness_score * 0.4)
        return min(1.0, max(0.0, intentional_score))
        
    def update_movement_tracking(self, landmarks_data: Dict):
        """Update movement tracking for intentional detection"""
        if not landmarks_data.get('landmarks_detected', False):
            return
        
        # Use nose tip as primary reference point
        head_pose = landmarks_data.get('head_pose', {})
        if head_pose:
            self.movement_history.append({
                'timestamp': time.time(),
                'x': head_pose.get('yaw', 0),
                'y': head_pose.get('pitch', 0),
                'confidence': head_pose.get('confidence', 0)
            })
        
    def process_response(self, detection_results: Dict) -> bool:
        """
        Enhanced response processing dengan quality tracking
        Returns True jika challenge completed
        """
        # Update movement tracking
        self.update_movement_tracking(detection_results)
        
        # Add progress tracking
        progress_entry = {
            'timestamp': time.time(),
            'progress_percent': self.get_progress_percentage(),
            'time_remaining': self.get_time_remaining(),
            'detection_quality': self._assess_detection_quality(detection_results)
        }
        self.progress_data.append(progress_entry)
        
        # Call specific challenge processing
        return self._process_specific_response(detection_results)
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        """Override in subclasses for specific challenge logic"""
        raise NotImplementedError
    
    def _assess_detection_quality(self, detection_results: Dict) -> float:
        """Assess quality of current detection (0-1)"""
        if not detection_results.get('landmarks_detected', False):
            return 0.0
        
        quality_factors = []
        
        # Landmark confidence
        confidence = detection_results.get('confidence', 0)
        quality_factors.append(confidence)
        
        # Face size (not too small, not too large)
        landmark_count = len(detection_results.get('landmark_coordinates', []))
        if landmark_count > 400:  # MediaPipe provides 468 for good detection
            quality_factors.append(0.9)
        elif landmark_count > 200:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Head pose confidence
        head_pose = detection_results.get('head_pose', {})
        if head_pose and 'confidence' in head_pose:
            quality_factors.append(head_pose['confidence'])
        
        return np.mean(quality_factors) if quality_factors else 0.0
        
    def get_result(self) -> ChallengeResult:
        """Enhanced result dengan quality dan intentional scoring"""
        response_time = time.time() - self.start_time if self.start_time else 0
        
        return ChallengeResult(
            challenge_id=self.challenge_id,
            challenge_type=self.challenge_type,
            difficulty=self.difficulty,
            success=self.success,
            response_time=response_time,
            confidence_score=self._calculate_confidence(),
            quality_score=self._calculate_quality_score(),
            intentional_score=self.calculate_intentional_score(),
            details=self._get_details(),
            timestamp=time.time(),
            progress_data=self.progress_data.copy()
        )
        
    def _calculate_confidence(self) -> float:
        """Calculate confidence score berdasarkan response quality"""
        return 1.0 if self.success else 0.0
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score of the response"""
        if not self.quality_metrics:
            return 0.5
        return np.mean(self.quality_metrics)
        
    def _get_details(self) -> Dict:
        """Get detail information tentang response"""
        return {
            'response_data_length': len(self.response_data),
            'progress_data_length': len(self.progress_data),
            'movement_consistency': self.calculate_intentional_score(),
            'average_quality': self._calculate_quality_score()
        }

class BlinkChallenge(Challenge):
    """Enhanced blink challenge dengan improved detection dan anti-spoofing"""
    
    def __init__(self, challenge_id: str, required_blinks: int = 3, 
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 15.0):  # Extended timeout
        super().__init__(
            challenge_id, 
            ChallengeType.BLINK, 
            difficulty,
            duration,
            f"Kedip mata {required_blinks} kali dengan natural (dalam {duration:.0f} detik)"
        )
        self.required_blinks = required_blinks
        self.initial_blink_count = None
        self.detected_blinks = 0
        self.last_blink_time = None
        self.blink_intervals = []  # Track time between blinks
        self.ear_history = deque(maxlen=20)  # Track EAR for quality assessment
        
        # Anti-spoofing thresholds
        self.min_blink_interval = 0.3  # Minimum time between natural blinks
        self.max_blink_interval = 3.0  # Maximum time for responsive blinking
        self.ear_variation_threshold = 0.15  # Minimum EAR variation for valid blink
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_time = time.time()
        current_blinks = detection_results.get('blink_count', 0)
        ear_left = detection_results.get('ear_left', 0)
        ear_right = detection_results.get('ear_right', 0)
        avg_ear = (ear_left + ear_right) / 2.0
        
        # Track EAR history for quality assessment
        self.ear_history.append(avg_ear)
        
        # Set initial blink count on first frame
        if self.initial_blink_count is None:
            self.initial_blink_count = current_blinks
            
        # Detect new blinks
        blinks_performed = current_blinks - self.initial_blink_count
        if blinks_performed > self.detected_blinks:
            # New blink detected
            new_blinks = blinks_performed - self.detected_blinks
            
            for _ in range(new_blinks):
                # Validate blink quality
                if self._validate_blink_quality(current_time):
                    self.detected_blinks += 1
                    self.last_blink_time = current_time
                    print(f"Valid blink detected! Count: {self.detected_blinks}/{self.required_blinks}")
        
        # Store response data
        self.response_data.append({
            'timestamp': current_time,
            'blink_count': current_blinks,
            'detected_blinks': self.detected_blinks,
            'ear_left': ear_left,
            'ear_right': ear_right,
            'avg_ear': avg_ear
        })
        
        # Update quality metrics
        quality = self._assess_detection_quality(detection_results)
        self.quality_metrics.append(quality)
        
        # Check success
        if self.detected_blinks >= self.required_blinks:
            self.success = self._validate_overall_blink_pattern()
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
    
    def _validate_blink_quality(self, current_time: float) -> bool:
        """Validate if the detected blink is natural and intentional"""
        
        # Check minimum interval between blinks (prevent rapid fake blinks)
        if (self.last_blink_time is not None and 
            current_time - self.last_blink_time < self.min_blink_interval):
            return False
        
        # Check EAR variation (natural blinks should have significant EAR change)
        if len(self.ear_history) >= 10:
            recent_ears = list(self.ear_history)[-10:]
            ear_range = max(recent_ears) - min(recent_ears)
            if ear_range < self.ear_variation_threshold:
                return False  # Not enough EAR variation for natural blink
        
        # Record blink interval for pattern analysis
        if self.last_blink_time is not None:
            interval = current_time - self.last_blink_time
            self.blink_intervals.append(interval)
        
        return True
    
    def _validate_overall_blink_pattern(self) -> bool:
        """Validate overall blinking pattern for anti-spoofing"""
        if len(self.blink_intervals) < 2:
            return True  # Not enough data to validate pattern
        
        # Check for too regular intervals (suspicious of automation)
        interval_variance = np.var(self.blink_intervals)
        if interval_variance < 0.1:  # Too regular
            return False
        
        # Check for reasonable timing
        avg_interval = np.mean(self.blink_intervals)
        if avg_interval < self.min_blink_interval or avg_interval > self.max_blink_interval:
            return False
        
        return True
        
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        confidence_factors = []
        
        # EAR variation quality
        if len(self.ear_history) > 5:
            ear_values = list(self.ear_history)
            ear_range = max(ear_values) - min(ear_values)
            ear_factor = min(1.0, ear_range / 0.3)  # Normalize to expected range
            confidence_factors.append(ear_factor)
        
        # Blink timing naturalness
        if len(self.blink_intervals) > 1:
            interval_variance = np.var(self.blink_intervals)
            timing_factor = min(1.0, interval_variance / 0.5)  # Reward natural variation
            confidence_factors.append(timing_factor)
        
        # Response completeness
        completion_factor = min(1.0, self.detected_blinks / self.required_blinks)
        confidence_factors.append(completion_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0


class HeadMovementChallenge(Challenge):
    """Enhanced head movement challenge with proper pose detection and anti-spoofing"""
    
    def __init__(self, challenge_id: str, required_movements: int = 3,
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 15.0):
        super().__init__(
            challenge_id,
            ChallengeType.HEAD_MOVEMENT,
            difficulty,
            duration,
            f"Gerakan kepala alami {required_movements} kali (kiri/kanan, atas/bawah) dalam {duration:.0f} detik"
        )
        self.required_movements = required_movements
        self.detected_movements = 0
        self.last_movement_time = None
        self.movement_history = []
        self.pose_history = deque(maxlen=30)  # Track head pose history
        self.baseline_pose = None
        
        # Anti-spoofing thresholds
        self.min_movement_interval = 0.8  # Minimum time between movements
        self.max_movement_interval = 8.0  # Maximum time for responsive movement
        self.movement_threshold = 15.0    # Minimum angle change for movement (degrees)
        self.movement_duration_min = 0.3  # Minimum duration for movement
        self.movement_duration_max = 2.5  # Maximum duration for movement
        
        # State tracking
        self.current_movement_start = None
        self.is_currently_moving = False
        self.last_pose = None
        self.movement_directions = ['left', 'right', 'up', 'down']
        self.detected_directions = set()
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_time = time.time()
        landmarks = detection_results.get('landmarks', [])
        
        # Calculate head pose angles
        pose_angles = self._calculate_head_pose(landmarks)
        if pose_angles is None:
            return False
        
        self.pose_history.append(pose_angles)
        
        # Establish baseline if not set
        if self.baseline_pose is None and len(self.pose_history) >= 10:
            # Use median of first samples as baseline (neutral position)
            poses = list(self.pose_history)[:10]
            self.baseline_pose = {
                'yaw': np.median([p['yaw'] for p in poses]),
                'pitch': np.median([p['pitch'] for p in poses]),
                'roll': np.median([p['roll'] for p in poses])
            }
        
        # Detect movement
        if self.baseline_pose is not None:
            movement_detected = self._detect_movement_state(pose_angles, current_time)
            
            if movement_detected and self.detected_movements < self.required_movements:
                self.detected_movements += 1
                print(f"Natural head movement detected! Count: {self.detected_movements}/{self.required_movements}")
        
        # Store response data
        self.response_data.append({
            'timestamp': current_time,
            'pose_angles': pose_angles,
            'baseline_pose': self.baseline_pose,
            'is_moving': self.is_currently_moving,
            'detected_movements': self.detected_movements,
            'detected_directions': list(self.detected_directions)
        })
        
        # Update quality metrics
        quality = self._assess_detection_quality(detection_results)
        self.quality_metrics.append(quality)
        
        # Check success
        if self.detected_movements >= self.required_movements:
            self.success = self._validate_overall_movement_pattern()
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
    
    def _calculate_head_pose(self, landmarks) -> Optional[Dict[str, float]]:
        """Calculate head pose angles (yaw, pitch, roll) from landmarks"""
        if len(landmarks) < 468:
            return None
        
        try:
            # Key facial landmarks for pose estimation
            # Nose tip: 1
            # Chin: 152
            # Left eye corner: 33
            # Right eye corner: 263
            # Left mouth corner: 61
            # Right mouth corner: 291
            
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            
            # Convert to numpy arrays
            nose = np.array([nose_tip.x, nose_tip.y, nose_tip.z])
            chin_point = np.array([chin.x, chin.y, chin.z])
            left_eye_point = np.array([left_eye.x, left_eye.y, left_eye.z])
            right_eye_point = np.array([right_eye.x, right_eye.y, right_eye.z])
            left_mouth_point = np.array([left_mouth.x, left_mouth.y, left_mouth.z])
            right_mouth_point = np.array([right_mouth.x, right_mouth.y, right_mouth.z])
            
            # Calculate eye center
            eye_center = (left_eye_point + right_eye_point) / 2
            
            # Calculate mouth center
            mouth_center = (left_mouth_point + right_mouth_point) / 2
            
            # Calculate yaw (left-right rotation)
            eye_vector = right_eye_point - left_eye_point
            yaw = np.arctan2(eye_vector[0], abs(eye_vector[2])) * 180 / np.pi
            
            # Calculate pitch (up-down rotation)
            face_vector = eye_center - mouth_center
            pitch = np.arctan2(face_vector[1], abs(face_vector[2])) * 180 / np.pi
            
            # Calculate roll (tilt)
            roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
            
            return {
                'yaw': float(yaw),      # Left-right: negative = left, positive = right
                'pitch': float(pitch),  # Up-down: negative = down, positive = up
                'roll': float(roll)     # Tilt: negative = left tilt, positive = right tilt
            }
            
        except (IndexError, AttributeError, ZeroDivisionError) as e:
            return None
    
    def _detect_movement_state(self, current_pose: Dict[str, float], current_time: float) -> bool:
        """Detect head movement state changes and validate natural movement"""
        
        if self.baseline_pose is None:
            return False
        
        # Calculate angle differences from baseline
        yaw_diff = abs(current_pose['yaw'] - self.baseline_pose['yaw'])
        pitch_diff = abs(current_pose['pitch'] - self.baseline_pose['pitch'])
        
        # Determine if significant movement is occurring
        is_moving_now = (yaw_diff > self.movement_threshold or 
                        pitch_diff > self.movement_threshold)
        
        # Detect movement start
        if is_moving_now and not self.is_currently_moving:
            self.current_movement_start = current_time
            self.is_currently_moving = True
            self.last_pose = current_pose.copy()
            return False  # Don't count until movement ends
        
        # Detect movement end
        elif not is_moving_now and self.is_currently_moving:
            if self.current_movement_start is not None and self.last_pose is not None:
                movement_duration = current_time - self.current_movement_start
                
                # Validate movement quality and determine direction
                movement_direction = self._determine_movement_direction(self.last_pose, current_pose)
                
                if (self._validate_movement_quality(movement_duration, current_time) and 
                    movement_direction is not None):
                    
                    self.last_movement_time = current_time
                    self.movement_history.append({
                        'direction': movement_direction,
                        'duration': movement_duration,
                        'timestamp': current_time
                    })
                    self.detected_directions.add(movement_direction)
                    self.is_currently_moving = False
                    return True  # Valid movement completed
            
            self.is_currently_moving = False
        
        return False
    
    def _determine_movement_direction(self, start_pose: Dict[str, float], 
                                    end_pose: Dict[str, float]) -> Optional[str]:
        """Determine the primary direction of head movement"""
        
        yaw_change = end_pose['yaw'] - start_pose['yaw']
        pitch_change = end_pose['pitch'] - start_pose['pitch']
        
        # Determine primary movement direction
        if abs(yaw_change) > abs(pitch_change):
            # Horizontal movement
            if yaw_change > self.movement_threshold:
                return 'right'
            elif yaw_change < -self.movement_threshold:
                return 'left'
        else:
            # Vertical movement
            if pitch_change > self.movement_threshold:
                return 'up'
            elif pitch_change < -self.movement_threshold:
                return 'down'
        
        return None
    
    def _validate_movement_quality(self, duration: float, current_time: float) -> bool:
        """Validate if the movement is natural and intentional"""
        
        # Check movement duration
        if duration < self.movement_duration_min or duration > self.movement_duration_max:
            return False
        
        # Check interval between movements
        if (self.last_movement_time is not None and 
            current_time - self.last_movement_time < self.min_movement_interval):
            return False
        
        return True
    
    def _validate_overall_movement_pattern(self) -> bool:
        """Validate overall movement pattern for anti-spoofing"""
        if len(self.movement_history) < 2:
            return True  # Not enough data to validate
        
        # Check for direction diversity (should have different directions)
        unique_directions = len(self.detected_directions)
        if unique_directions < min(2, self.required_movements):
            return False
        
        # Check for reasonable timing variation (not too mechanical)
        durations = [m['duration'] for m in self.movement_history]
        if len(durations) > 1:
            duration_variance = np.var(durations)
            if duration_variance < 0.1:  # Too regular
                return False
        
        # Check for reasonable average timing
        intervals = []
        for i in range(1, len(self.movement_history)):
            interval = (self.movement_history[i]['timestamp'] - 
                       self.movement_history[i-1]['timestamp'])
            intervals.append(interval)
        
        if intervals:
            avg_interval = np.mean(intervals)
            if avg_interval < self.min_movement_interval or avg_interval > self.max_movement_interval:
                return False
        
        return True
    
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        confidence_factors = []
        
        # Movement diversity quality
        direction_diversity = len(self.detected_directions) / len(self.movement_directions)
        confidence_factors.append(direction_diversity)
        
        # Movement timing naturalness
        if len(self.movement_history) > 1:
            durations = [m['duration'] for m in self.movement_history]
            duration_variance = np.var(durations)
            # Natural variance in timing
            timing_factor = min(1.0, duration_variance / 0.3)
            confidence_factors.append(timing_factor)
        
        # Overall detection quality
        if self.quality_metrics:
            avg_quality = np.mean(self.quality_metrics)
            confidence_factors.append(avg_quality)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0


class MouthOpenChallenge(Challenge):
    """Challenge untuk membuka mulut"""
    
    def __init__(self, challenge_id: str, min_open_time: float = 1.0,
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 15.0, open_threshold: float = 0.6):
        super().__init__(
            challenge_id,
            ChallengeType.MOUTH_OPEN,
            difficulty,
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


class SequenceChallenge(Challenge):
    """Enhanced sequence challenge dengan improved step validation"""
    
    def __init__(self, challenge_id: str, sequence: List[ChallengeType], 
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 20.0):
        super().__init__(
            challenge_id,
            ChallengeType.SEQUENCE,
            difficulty,
            duration,
            f"Lakukan berurutan: {' → '.join([c.value for c in sequence])}"
        )
        self.sequence = sequence
        self.current_step = 0
        self.step_results = []
        self.step_start_time = None
        self.step_timeout = 8.0  # 8 seconds per step
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        if self.current_step >= len(self.sequence):
            self.success = True
            self.completed = True
            return True
        
        current_time = time.time()
        current_challenge_type = self.sequence[self.current_step]
        
        # Initialize step timer
        if self.step_start_time is None:
            self.step_start_time = current_time
        
        # Check step timeout
        if current_time - self.step_start_time > self.step_timeout:
            # Step timeout - move to next step or fail
            print(f"Step {self.current_step + 1} timeout - moving to next step")
            self._advance_to_next_step(current_time, detection_results, success=False)
            return False
        
        # Check current step completion
        step_completed = self._validate_current_step(detection_results)
        
        if step_completed:
            print(f"Step {self.current_step + 1} completed: {current_challenge_type.value}")
            self._advance_to_next_step(current_time, detection_results, success=True)
        
        # Store response data
        self.response_data.append({
            'timestamp': current_time,
            'current_step': self.current_step,
            'challenge_type': current_challenge_type.value,
            'step_progress': (current_time - self.step_start_time) / self.step_timeout,
            'detection_data': detection_results.copy()
        })
        
        # Update quality metrics
        quality = self._assess_detection_quality(detection_results)
        self.quality_metrics.append(quality)
        
        # Check overall completion
        if self.current_step >= len(self.sequence):
            self.success = len(self.step_results) >= len(self.sequence) * 0.8  # 80% success rate
            self.completed = True
            return True
        
        # Check overall timeout
        if self.is_expired():
            self.completed = True
            return True
        
        return False
    
    def _validate_current_step(self, detection_results: Dict) -> bool:
        """Validate completion of current sequence step"""
        current_challenge_type = self.sequence[self.current_step]
        
        if current_challenge_type == ChallengeType.BLINK:
            # Enhanced blink detection
            blink_count = detection_results.get('blink_count', 0)
            return blink_count > len([r for r in self.step_results if r.get('success', False)])
            
        elif current_challenge_type == ChallengeType.HEAD_MOVEMENT:
            # Check for smile indicators (MAR or other facial features)
            landmarks = detection_results.get('landmarks', [])
            if landmarks and len(landmarks) >= 468:
                # Use mouth landmarks for smile detection
                try:
                    left_corner = landmarks[61]
                    right_corner = landmarks[291]
                    mouth_width = abs(right_corner.x - left_corner.x)
                    return mouth_width > 0.05  # Simple smile threshold
                except (IndexError, AttributeError):
                    return False
            return False
            
        elif current_challenge_type == ChallengeType.MOUTH_OPEN:
            return detection_results.get('mouth_open', False)
            
        elif current_challenge_type == ChallengeType.HEAD_MOVEMENT:
            head_pose = detection_results.get('head_pose', {})
            yaw = abs(head_pose.get('yaw', 0))
            pitch = abs(head_pose.get('pitch', 0))
            return yaw > 15 or pitch > 15  # Any significant head movement
            
        return False
    
    def _advance_to_next_step(self, current_time: float, detection_results: Dict, success: bool):
        """Advance to next step in sequence"""
        self.step_results.append({
            'step': self.current_step,
            'challenge_type': self.sequence[self.current_step],
            'success': success,
            'timestamp': current_time,
            'duration': current_time - (self.step_start_time or current_time),
            'detection_results': detection_results.copy()
        })
        
        self.current_step += 1
        self.step_start_time = current_time  # Reset timer for next step
    
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.step_results) == 0:
            return 0.0
            
        # Calculate confidence based on step completion rate and timing
        successful_steps = sum(1 for step in self.step_results if step.get('success', False))
        completion_rate = successful_steps / len(self.sequence)
        
        # Factor in timing consistency
        step_durations = [step.get('duration', 0) for step in self.step_results if step.get('success', False)]
        if step_durations:
            avg_duration = np.mean(step_durations)
            timing_consistency = 1.0 - min(1.0, np.std(step_durations) / max(avg_duration, 1.0))
        else:
            timing_consistency = 0.0
        
        # Combined confidence
        confidence = (completion_rate * 0.7) + (timing_consistency * 0.3)
        return confidence

class ChallengeResponseSystem:
    """
    Enhanced system untuk mengelola challenge-response dengan anti-spoofing
    Now includes audio feedback and retry logic
    """
    
    def __init__(self, anti_replay_window: float = 300.0, 
                 audio_enabled: bool = True, max_attempts: int = 3):  # 5 minutes
        self.current_challenge = None
        self.challenge_history = deque(maxlen=100)
        self.anti_replay_window = anti_replay_window
        self.used_challenges = set()
        
        # Audio feedback system
        self.audio_system = None
        if AUDIO_AVAILABLE and audio_enabled:
            self.audio_system = AudioFeedbackSystem(
                audio_enabled=audio_enabled,
                voice_enabled=audio_enabled
            )
        
        # Retry logic
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.failed_challenges = []
        self.session_timeout = 30.0  # 30 seconds total session time
        self.session_start_time = None
        
        # Security measures
        self.challenge_timestamps = deque(maxlen=50)
        self.last_success_time = None
        self.replay_detection_enabled = True
        
    def generate_random_challenge(self, difficulty: ChallengeDifficulty = None) -> Challenge:
        """Generate random challenge dengan enhanced types and audio feedback"""
        if difficulty is None:
            difficulty = random.choice(list(ChallengeDifficulty))
        
        # Enhanced challenge types with anti-spoofing focus
        challenge_types = [
            ChallengeType.BLINK,
            ChallengeType.HEAD_MOVEMENT,
            ChallengeType.MOUTH_OPEN,
        ]
        
        # Add distance challenges if available
        if DISTANCE_CHALLENGE_AVAILABLE:
            challenge_types.extend([
                ChallengeType.MOVE_CLOSER,
                ChallengeType.MOVE_FARTHER
            ])
        
        challenge_type = random.choice(challenge_types)
        challenge_id = f"{challenge_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Anti-replay: pastikan challenge tidak diulang dalam window tertentu
        while challenge_id in self.used_challenges:
            challenge_id = f"{challenge_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
            
        self.used_challenges.add(challenge_id)
        
        # Create specific challenge with enhanced parameters
        challenge = None
        
        if challenge_type == ChallengeType.BLINK:
            required_blinks = 2 if difficulty == ChallengeDifficulty.EASY else (3 if difficulty == ChallengeDifficulty.MEDIUM else 4)
            challenge = BlinkChallenge(challenge_id, required_blinks=required_blinks, difficulty=difficulty)
            
        elif challenge_type == ChallengeType.HEAD_MOVEMENT:
            required_movements = 2 if difficulty == ChallengeDifficulty.EASY else (3 if difficulty == ChallengeDifficulty.MEDIUM else 4)
            challenge = HeadMovementChallenge(challenge_id, required_movements=required_movements, difficulty=difficulty)
            
        elif challenge_type == ChallengeType.MOUTH_OPEN:
            min_open_time = 1.0 if difficulty == ChallengeDifficulty.EASY else (1.5 if difficulty == ChallengeDifficulty.MEDIUM else 2.0)
            challenge = MouthOpenChallenge(challenge_id, min_open_time=min_open_time, difficulty=difficulty)
            
        # Distance challenges
        elif challenge_type == ChallengeType.MOVE_CLOSER and DISTANCE_CHALLENGE_AVAILABLE:
            challenge = DistanceChallenge(challenge_id, direction="closer", difficulty=difficulty)
            
        elif challenge_type == ChallengeType.MOVE_FARTHER and DISTANCE_CHALLENGE_AVAILABLE:
            challenge = DistanceChallenge(challenge_id, direction="farther", difficulty=difficulty)
            
        # Fallback for unimplemented challenge types
        if challenge is None:
            challenge = self._create_placeholder_challenge(challenge_id, challenge_type, difficulty)
        
        return challenge
    
    def _create_placeholder_challenge(self, challenge_id: str, challenge_type: ChallengeType, 
                                    difficulty: ChallengeDifficulty) -> Challenge:
        """Create placeholder for unimplemented challenge types"""
        duration = 10.0 if difficulty == ChallengeDifficulty.EASY else (15.0 if difficulty == ChallengeDifficulty.MEDIUM else 20.0)
        
        # Return basic challenge that always succeeds (for testing)
        challenge = Challenge(challenge_id, challenge_type, difficulty, duration, f"Perform {challenge_type.value}")
        challenge.success = True
        challenge.completed = True
        return challenge
        
    def generate_sequence_challenge(self, num_steps: int = 3, 
                                  difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM) -> SequenceChallenge:
        """Generate enhanced sequence challenge"""
        available_types = [
            ChallengeType.BLINK,
            ChallengeType.HEAD_MOVEMENT,
            ChallengeType.MOUTH_OPEN
        ]
        
        # Select sequence based on difficulty
        if difficulty == ChallengeDifficulty.EASY:
            sequence = random.sample(available_types, min(2, num_steps))
        elif difficulty == ChallengeDifficulty.MEDIUM:
            sequence = random.sample(available_types, min(3, num_steps))
        else:  # HARD
            sequence = random.sample(available_types, min(4, num_steps))
        
        challenge_id = f"sequence_{int(time.time())}_{random.randint(1000, 9999)}"
        duration = len(sequence) * 8.0  # 8 seconds per step
        
        return SequenceChallenge(challenge_id, sequence, difficulty, duration)
        
    def start_challenge(self, challenge_type: str = 'random', 
                       difficulty: ChallengeDifficulty = None) -> Challenge:
        """Start new challenge dengan enhanced options and audio feedback"""
        
        # Check session timeout
        if self.session_start_time is None:
            self.session_start_time = time.time()
        elif time.time() - self.session_start_time > self.session_timeout:
            print("Session timeout reached. Resetting session.")
            self.reset_session()
            self.session_start_time = time.time()
        
        # Check maximum attempts
        if self.current_attempt >= self.max_attempts:
            print(f"Maximum attempts ({self.max_attempts}) reached. Please try again later.")
            if self.audio_system:
                self.audio_system.play_warning("Maximum attempts reached. Please try again later.")
            return None
        
        # Generate challenge
        if challenge_type == 'random':
            challenge = self.generate_random_challenge(difficulty)
        elif challenge_type == 'sequence':
            challenge = self.generate_sequence_challenge(difficulty=difficulty or ChallengeDifficulty.MEDIUM)
        elif challenge_type == 'distance_closer' and DISTANCE_CHALLENGE_AVAILABLE:
            challenge_id = f"distance_closer_{int(time.time())}_{random.randint(1000, 9999)}"
            challenge = DistanceChallenge(challenge_id, direction="closer", difficulty=difficulty or ChallengeDifficulty.MEDIUM)
        elif challenge_type == 'distance_farther' and DISTANCE_CHALLENGE_AVAILABLE:
            challenge_id = f"distance_farther_{int(time.time())}_{random.randint(1000, 9999)}"
            challenge = DistanceChallenge(challenge_id, direction="farther", difficulty=difficulty or ChallengeDifficulty.MEDIUM)
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
            
        if challenge is None:
            return None
            
        challenge.start()
        self.current_challenge = challenge
        self.current_attempt += 1
        
        # Record challenge timestamp
        self.challenge_timestamps.append(time.time())
        
        # Audio feedback
        if self.audio_system:
            self.audio_system.play_challenge_start(challenge.description)
        
        print(f"\n🎯 New Challenge Started: {challenge.description}")
        # Handle both enum and non-enum difficulty values
        difficulty_str = challenge.difficulty.value if hasattr(challenge.difficulty, 'value') else str(challenge.difficulty)
        print(f"   Difficulty: {difficulty_str}")
        # Handle both numeric and string duration values with robust error handling
        try:
            if isinstance(challenge.duration, (int, float)):
                duration_str = f"{challenge.duration:.1f}s"
            else:
                duration_str = f"{float(challenge.duration):.1f}s"
        except (ValueError, TypeError):
            duration_str = "15.0s"  # fallback default
        print(f"   Duration: {duration_str}")
        print(f"   Attempt: {self.current_attempt}/{self.max_attempts}")
        
        return challenge
        
    def process_frame(self, detection_results: Dict) -> Optional[ChallengeResult]:
        """
        Process frame dengan detection results
        Returns ChallengeResult jika challenge completed
        Includes enhanced audio feedback and retry logic
        """
        if self.current_challenge is None:
            return None
        
        # Check for replay attacks
        if self.replay_detection_enabled and self._detect_replay_attack(detection_results):
            print("⚠️ Potential replay attack detected!")
            if self.audio_system:
                self.audio_system.play_warning("Replay attack detected")
            self.current_challenge = None
            return None
        
        # Process response
        completed = self.current_challenge.process_response(detection_results)
        
        # Audio feedback for progress (every 5 seconds)
        if (self.audio_system and self.current_challenge and 
            hasattr(self.current_challenge, 'start_time') and
            self.current_challenge.start_time):
            
            elapsed = time.time() - self.current_challenge.start_time
            remaining = self.current_challenge.duration - elapsed
            
            # Countdown warnings
            if remaining <= 5 and remaining > 4:
                self.audio_system.play_countdown(int(remaining))
        
        if completed:
            result = self.current_challenge.get_result()
            self.challenge_history.append(result)
            
            # Audio feedback based on result
            if self.audio_system:
                if result.success:
                    self.audio_system.play_challenge_success(result.challenge_type.value)
                    self.last_success_time = time.time()
                    self.current_attempt = 0  # Reset attempts on success
                else:
                    failure_reason = self._get_failure_reason(result)
                    self.audio_system.play_challenge_failure(result.challenge_type.value, failure_reason)
                    self.failed_challenges.append(result.challenge_type)
            
            self.current_challenge = None
            
            # Check if retry is needed
            if not result.success and self.current_attempt < self.max_attempts:
                print(f"Challenge failed. Attempts remaining: {self.max_attempts - self.current_attempt}")
                self._provide_failure_guidance(result)
            
            logging.info(f"Challenge completed: {result.challenge_id}, Success: {result.success}, "
                        f"Confidence: {result.confidence_score:.3f}, Quality: {result.quality_score:.3f}")
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
            'type_statistics': type_stats,
            'current_attempt': self.current_attempt,
            'max_attempts': self.max_attempts,
            'session_active': self.session_start_time is not None,
            'session_time_remaining': max(0, self.session_timeout - (time.time() - (self.session_start_time or time.time())))
        }
    
    def _detect_replay_attack(self, detection_results: Dict) -> bool:
        """Detect potential replay attacks based on temporal patterns"""
        if not self.replay_detection_enabled:
            return False
        
        current_time = time.time()
        
        # Check for suspiciously regular timing
        if len(self.challenge_timestamps) >= 3:
            recent_intervals = []
            for i in range(1, min(4, len(self.challenge_timestamps))):
                interval = self.challenge_timestamps[-i] - self.challenge_timestamps[-i-1]
                recent_intervals.append(interval)
            
            if recent_intervals:
                interval_variance = np.var(recent_intervals)
                if interval_variance < 0.1:  # Too regular timing
                    return True
        
        # Check for identical frame patterns (simplified)
        landmarks = detection_results.get('landmark_coordinates', [])
        if landmarks and len(landmarks) > 10:
            # Check if landmarks are suspiciously static
            landmark_variance = np.var([point[0] for point in landmarks[:10]] + 
                                      [point[1] for point in landmarks[:10]])
            if landmark_variance < 0.001:  # Too static
                return True
        
        return False
    
    def _get_failure_reason(self, result: ChallengeResult) -> str:
        """Get human-readable failure reason"""
        if result.quality_score < 0.5:
            return "Poor detection quality"
        elif result.intentional_score < 0.5:
            return "Movement appears unnatural"
        elif result.response_time >= result.details.get('duration', 15.0):
            return "Time limit exceeded"
        else:
            return "Challenge requirements not met"
    
    def _provide_failure_guidance(self, result: ChallengeResult):
        """Provide guidance based on failure reason"""
        challenge_type = result.challenge_type.value
        guidance = ChallengeInstructions.get_guidance(challenge_type)
        
        print(f"\n💡 Guidance for {challenge_type} challenge:")
        for tip in guidance:
            print(f"   • {tip}")
        
        if self.audio_system:
            guidance_text = f"Tips for {challenge_type}: " + ". ".join(guidance[:2])  # Limit to first 2 tips
            self.audio_system.speak(guidance_text)
    
    def reset_session(self):
        """Reset challenge session"""
        self.current_attempt = 0
        self.failed_challenges.clear()
        self.session_start_time = None
        self.current_challenge = None
        
        if self.audio_system:
            self.audio_system.speak("Challenge session reset")
        
        print("🔄 Challenge session reset")
    
    def get_session_status(self) -> Dict:
        """Get current session status"""
        return {
            'session_active': self.session_start_time is not None,
            'current_attempt': self.current_attempt,
            'max_attempts': self.max_attempts,
            'attempts_remaining': max(0, self.max_attempts - self.current_attempt),
            'session_time_remaining': max(0, self.session_timeout - (time.time() - (self.session_start_time or time.time()))),
            'has_current_challenge': self.current_challenge is not None,
            'failed_challenge_types': [ct.value for ct in self.failed_challenges],
            'audio_enabled': self.audio_system is not None
        }
    
    def set_audio_enabled(self, enabled: bool):
        """Enable/disable audio feedback"""
        if self.audio_system:
            self.audio_system.set_audio_enabled(enabled)
            self.audio_system.set_voice_enabled(enabled)
    
    def shutdown(self):
        """Shutdown challenge system"""
        if self.audio_system:
            self.audio_system.shutdown()
        print("🔒 Challenge system shutdown")

def test_challenge_response_system():
    """
    Enhanced test function untuk challenge-response system
    Now includes audio feedback and retry logic testing
    """
    from src.detection.landmark_detection import LivenessVerifier
    
    # Initialize systems
    verifier = LivenessVerifier()
    challenge_system = ChallengeResponseSystem(audio_enabled=True, max_attempts=3)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("🎯 Testing Enhanced Challenge-Response System...")
    print("=" * 60)
    print("Instructions will appear on screen with audio feedback")
    print("📋 Controls:")
    print("   'n' - New random challenge")
    print("   'e' - Easy challenge")  
    print("   'm' - Medium challenge")
    print("   'h' - Hard challenge")
    print("   's' - Sequence challenge")
    print("   'c' - Distance closer challenge")
    print("   'f' - Distance farther challenge")
    print("   'r' - Reset session")
    print("   'a' - Toggle audio")
    print("   'q' - Quit")
    print("=" * 60)
    
    current_challenge = None
    audio_enabled = True
    
    try:
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
                print(f"\n🎯 Challenge Result:")
                print(f"   Success: {challenge_result.success}")
                print(f"   Response Time: {challenge_result.response_time:.2f}s")
                print(f"   Confidence: {challenge_result.confidence_score:.3f}")
                print(f"   Quality: {challenge_result.quality_score:.3f}")
                print(f"   Intentional: {challenge_result.intentional_score:.3f}")
                current_challenge = None
            
            # Draw enhanced challenge status
            challenge_status = challenge_system.get_current_challenge_status()
            session_status = challenge_system.get_session_status()
            
            # Display session info
            y_offset = 30
            cv2.putText(frame, f"Attempt: {session_status['current_attempt']}/{session_status['max_attempts']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            session_time = session_status['session_time_remaining']
            cv2.putText(frame, f"Session Time: {session_time:.1f}s", 
                       (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if challenge_status:
                # Draw challenge description
                cv2.putText(frame, challenge_status['description'], (10, y_offset + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Draw timer with color coding
                remaining = challenge_status['remaining_time']
                timer_color = (0, 255, 0) if remaining > 5 else (0, 165, 255) if remaining > 2 else (0, 0, 255)
                cv2.putText(frame, f"Time: {remaining:.1f}s", (10, y_offset + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)
                
                # Draw progress bar
                progress = challenge_status['progress']
                bar_width = 300
                bar_height = 20
                cv2.rectangle(frame, (10, y_offset + 110), (10 + bar_width, y_offset + 110 + bar_height), (255, 255, 255), 2)
                cv2.rectangle(frame, (10, y_offset + 110), (10 + int(bar_width * progress), y_offset + 110 + bar_height), (0, 255, 0), -1)
                
                # Distance challenge specific info
                if hasattr(challenge_system.current_challenge, 'get_progress_info'):
                    progress_info = challenge_system.current_challenge.get_progress_info()
                    cv2.putText(frame, progress_info.get('message', ''), (10, y_offset + 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
            else:
                cv2.putText(frame, "Press 'n' for new challenge", (10, y_offset + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw detection info
            if detection_results['landmarks_detected']:
                y_info = 200
                cv2.putText(frame, f"Blinks: {detection_results['blink_count']}", (10, y_info), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if detection_results['mouth_open']:
                    cv2.putText(frame, "MOUTH OPEN", (10, y_info + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if detection_results['head_pose']:
                    pose = detection_results['head_pose']
                    cv2.putText(frame, f"Head: Y:{pose['yaw']:.0f}° P:{pose['pitch']:.0f}°", 
                               (10, y_info + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Audio status indicator
            audio_status = "🔊 ON" if audio_enabled else "🔇 OFF"
            cv2.putText(frame, f"Audio: {audio_status}", (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Enhanced Challenge-Response Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and challenge_status is None:
                # Start new random challenge
                challenge_system.start_challenge('random')
            elif key == ord('e') and challenge_status is None:
                # Start easy challenge
                challenge_system.start_challenge('random', ChallengeDifficulty.EASY)
            elif key == ord('m') and challenge_status is None:
                # Start medium challenge
                challenge_system.start_challenge('random', ChallengeDifficulty.MEDIUM)
            elif key == ord('h') and challenge_status is None:
                # Start hard challenge
                challenge_system.start_challenge('random', ChallengeDifficulty.HARD)
            elif key == ord('s') and challenge_status is None:
                # Start sequence challenge
                challenge_system.start_challenge('sequence')
            elif key == ord('c') and challenge_status is None and DISTANCE_CHALLENGE_AVAILABLE:
                # Start distance closer challenge
                challenge_system.start_challenge('distance_closer')
            elif key == ord('f') and challenge_status is None and DISTANCE_CHALLENGE_AVAILABLE:
                # Start distance farther challenge
                challenge_system.start_challenge('distance_farther')
            elif key == ord('r'):
                # Reset session
                challenge_system.reset_session()
            elif key == ord('a'):
                # Toggle audio
                audio_enabled = not audio_enabled
                challenge_system.set_audio_enabled(audio_enabled)
                print(f"Audio {'enabled' if audio_enabled else 'disabled'}")
        
        # Print final statistics
        stats = challenge_system.get_challenge_statistics()
        print("\n" + "=" * 60)
        print("📊 FINAL CHALLENGE STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if key != 'type_statistics':
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        if 'type_statistics' in stats:
            print(f"\n📈 By Challenge Type:")
            for challenge_type, type_stats in stats['type_statistics'].items():
                success_rate = (type_stats['success'] / type_stats['total']) * 100 if type_stats['total'] > 0 else 0
                avg_time = np.mean(type_stats['response_times']) if type_stats['response_times'] else 0
                print(f"   {challenge_type}: {success_rate:.1f}% success, {avg_time:.2f}s avg time")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        challenge_system.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print("🔒 Test completed and resources cleaned up")

if __name__ == "__main__":
    test_challenge_response_system()

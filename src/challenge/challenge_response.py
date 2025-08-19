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

class ChallengeType(Enum):
    """Enhanced enum untuk tipe-tipe challenge anti-spoofing"""
    BLINK = "blink"
    MOUTH_OPEN = "mouth_open"
    SMILE = "smile"  # NEW: Proper smile detection
    HEAD_MOVEMENT = "head_movement"  # CONSOLIDATED: All head directions
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


class SmileChallenge(Challenge):
    """Enhanced smile challenge dengan proper smile detection dan anti-spoofing"""
    
    def __init__(self, challenge_id: str, required_smiles: int = 2,
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 15.0):
        super().__init__(
            challenge_id,
            ChallengeType.SMILE,
            difficulty,
            duration,
            f"Senyum natural {required_smiles} kali (dalam {duration:.0f} detik)"
        )
        self.required_smiles = required_smiles
        self.detected_smiles = 0
        self.last_smile_time = None
        self.smile_intervals = []
        self.mouth_ratio_history = deque(maxlen=30)  # Track mouth aspect ratio
        self.baseline_mouth_ratio = None
        
        # Anti-spoofing thresholds
        self.min_smile_interval = 1.0  # Minimum time between natural smiles
        self.max_smile_interval = 8.0  # Maximum time for responsive smiling
        self.smile_threshold = 0.15  # Minimum mouth ratio increase for smile
        self.smile_duration_min = 0.5  # Minimum duration for natural smile
        self.smile_duration_max = 3.0  # Maximum duration for natural smile
        
        # State tracking
        self.current_smile_start = None
        self.is_currently_smiling = False
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_time = time.time()
        landmarks = detection_results.get('landmarks', [])
        
        # Calculate mouth aspect ratio for smile detection
        mouth_ratio = self._calculate_mouth_aspect_ratio(landmarks)
        if mouth_ratio is None:
            return False
        
        self.mouth_ratio_history.append(mouth_ratio)
        
        # Establish baseline if not set
        if self.baseline_mouth_ratio is None and len(self.mouth_ratio_history) >= 10:
            # Use median of first samples as baseline (neutral expression)
            self.baseline_mouth_ratio = np.median(list(self.mouth_ratio_history)[:10])
        
        # Detect smile state
        if self.baseline_mouth_ratio is not None:
            smile_detected = self._detect_smile_state(mouth_ratio, current_time)
            
            if smile_detected and self.detected_smiles < self.required_smiles:
                self.detected_smiles += 1
                print(f"Natural smile detected! Count: {self.detected_smiles}/{self.required_smiles}")
        
        # Store response data
        self.response_data.append({
            'timestamp': current_time,
            'mouth_ratio': mouth_ratio,
            'baseline_ratio': self.baseline_mouth_ratio,
            'is_smiling': self.is_currently_smiling,
            'detected_smiles': self.detected_smiles
        })
        
        # Update quality metrics
        quality = self._assess_detection_quality(detection_results)
        self.quality_metrics.append(quality)
        
        # Check success
        if self.detected_smiles >= self.required_smiles:
            self.success = self._validate_overall_smile_pattern()
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
    
    def _calculate_mouth_aspect_ratio(self, landmarks) -> Optional[float]:
        """Calculate mouth aspect ratio for smile detection"""
        if len(landmarks) < 468:
            return None
        
        try:
            # MediaPipe mouth landmark indices
            # Mouth corners: 61 (left), 291 (right)  
            # Upper lip: 13, 14, 15
            # Lower lip: 17, 18, 19
            left_corner = landmarks[61]   # Left mouth corner
            right_corner = landmarks[291] # Right mouth corner
            upper_center = landmarks[13]  # Upper lip center
            lower_center = landmarks[17]  # Lower lip center
            
            # Calculate mouth width and height
            mouth_width = np.linalg.norm(
                np.array([right_corner.x, right_corner.y]) - 
                np.array([left_corner.x, left_corner.y])
            )
            
            mouth_height = np.linalg.norm(
                np.array([upper_center.x, upper_center.y]) - 
                np.array([lower_center.x, lower_center.y])
            )
            
            # Avoid division by zero
            if mouth_height == 0:
                return 0.0
                
            # Mouth aspect ratio (MAR) - width/height ratio increases when smiling
            mar = mouth_width / mouth_height
            return mar
            
        except (IndexError, AttributeError) as e:
            return None
    
    def _detect_smile_state(self, current_ratio: float, current_time: float) -> bool:
        """Detect smile state changes and validate natural smiling"""
        
        ratio_increase = current_ratio - self.baseline_mouth_ratio
        is_smiling_now = ratio_increase > self.smile_threshold
        
        # Detect smile start
        if is_smiling_now and not self.is_currently_smiling:
            self.current_smile_start = current_time
            self.is_currently_smiling = True
            return False  # Don't count until smile ends
        
        # Detect smile end
        elif not is_smiling_now and self.is_currently_smiling:
            if self.current_smile_start is not None:
                smile_duration = current_time - self.current_smile_start
                
                # Validate smile duration and interval
                if self._validate_smile_quality(smile_duration, current_time):
                    self.last_smile_time = current_time
                    self.smile_intervals.append(
                        current_time - (self.last_smile_time or current_time)
                    )
                    self.is_currently_smiling = False
                    return True  # Valid smile completed
            
            self.is_currently_smiling = False
        
        return False
    
    def _validate_smile_quality(self, duration: float, current_time: float) -> bool:
        """Validate if the smile is natural and intentional"""
        
        # Check smile duration
        if duration < self.smile_duration_min or duration > self.smile_duration_max:
            return False
        
        # Check interval between smiles
        if (self.last_smile_time is not None and 
            current_time - self.last_smile_time < self.min_smile_interval):
            return False
        
        return True
    
    def _validate_overall_smile_pattern(self) -> bool:
        """Validate overall smiling pattern for anti-spoofing"""
        if len(self.smile_intervals) < 2:
            return True  # Not enough data to validate
        
        # Check for reasonable timing variation (not too mechanical)
        interval_variance = np.var(self.smile_intervals)
        if interval_variance < 0.2:  # Too regular
            return False
        
        # Check for reasonable average interval
        avg_interval = np.mean(self.smile_intervals)
        if avg_interval < self.min_smile_interval or avg_interval > self.max_smile_interval:
            return False
        
        return True
    
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        confidence_factors = []
        
        # Mouth ratio variation quality
        if len(self.mouth_ratio_history) > 10:
            ratios = list(self.mouth_ratio_history)
            ratio_range = max(ratios) - min(ratios)
            variation_factor = min(1.0, ratio_range / 0.3)  # Normalize to expected range
            confidence_factors.append(variation_factor)
        
        # Smile timing naturalness
        if len(self.smile_intervals) > 1:
            interval_variance = np.var(self.smile_intervals)
            timing_factor = min(1.0, interval_variance / 1.0)  # Reward natural variation
            confidence_factors.append(timing_factor)
        
        # Response completeness
        completion_factor = min(1.0, self.detected_smiles / self.required_smiles)
        confidence_factors.append(completion_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0


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
    """Enhanced head movement challenge dengan direction classification"""
    
    def __init__(self, challenge_id: str, required_directions: List[str] = None,
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 20.0):
        if required_directions is None:
            required_directions = ['left', 'right', 'up', 'down']
        
        super().__init__(
            challenge_id,
            ChallengeType.HEAD_MOVEMENT,
            difficulty,
            duration,
            f"Gerakkan kepala ke: {', '.join(required_directions)} (dalam {duration:.0f} detik)"
        )
        self.required_directions = required_directions
        self.completed_directions = set()
        self.head_positions = deque(maxlen=50)  # Track head position history
        self.baseline_position = None
        self.current_direction = None
        self.direction_start_time = None
        
        # Movement thresholds
        self.movement_threshold = 15.0  # Degrees for direction classification
        self.min_movement_duration = 1.0  # Minimum time in direction
        self.max_return_time = 3.0  # Maximum time to return to center
        
        # Direction tracking
        self.direction_history = []
        self.last_direction_time = None
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_time = time.time()
        
        # Get head pose angles
        pitch = detection_results.get('head_pitch', 0)
        yaw = detection_results.get('head_yaw', 0)
        roll = detection_results.get('head_roll', 0)
        
        # Store current position
        position = {'pitch': pitch, 'yaw': yaw, 'roll': roll, 'time': current_time}
        self.head_positions.append(position)
        
        # Establish baseline if not set
        if self.baseline_position is None and len(self.head_positions) >= 10:
            positions = list(self.head_positions)[:10]
            self.baseline_position = {
                'pitch': np.mean([p['pitch'] for p in positions]),
                'yaw': np.mean([p['yaw'] for p in positions]),
                'roll': np.mean([p['roll'] for p in positions])
            }
        
        # Detect direction if baseline is established
        if self.baseline_position is not None:
            direction = self._classify_head_direction(pitch, yaw)
            self._process_direction_change(direction, current_time)
        
        # Store response data
        self.response_data.append({
            'timestamp': current_time,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'direction': self.current_direction,
            'completed_directions': list(self.completed_directions)
        })
        
        # Update quality metrics
        quality = self._assess_detection_quality(detection_results)
        self.quality_metrics.append(quality)
        
        # Check success
        if len(self.completed_directions) >= len(self.required_directions):
            self.success = self._validate_movement_pattern()
            self.completed = True
            return True
            
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
    
    def _classify_head_direction(self, pitch: float, yaw: float) -> Optional[str]:
        """Classify head direction based on pitch and yaw angles"""
        if self.baseline_position is None:
            return None
        
        pitch_diff = pitch - self.baseline_position['pitch']
        yaw_diff = yaw - self.baseline_position['yaw']
        
        # Determine primary movement direction
        if abs(pitch_diff) > self.movement_threshold or abs(yaw_diff) > self.movement_threshold:
            if abs(pitch_diff) > abs(yaw_diff):
                # Vertical movement dominant
                return 'up' if pitch_diff > 0 else 'down'
            else:
                # Horizontal movement dominant  
                return 'right' if yaw_diff > 0 else 'left'
        
        return 'center'  # Near baseline position
    
    def _process_direction_change(self, direction: str, current_time: float):
        """Process head direction changes and validate movements"""
        
        # Direction change detected
        if direction != self.current_direction:
            # Complete previous direction if it was valid
            if (self.current_direction is not None and 
                self.current_direction != 'center' and
                self.direction_start_time is not None):
                
                duration = current_time - self.direction_start_time
                if duration >= self.min_movement_duration:
                    # Valid movement completed
                    if self.current_direction in self.required_directions:
                        if self.current_direction not in self.completed_directions:
                            self.completed_directions.add(self.current_direction)
                            print(f"Direction completed: {self.current_direction}")
                            print(f"Progress: {len(self.completed_directions)}/{len(self.required_directions)}")
            
            # Start tracking new direction
            self.current_direction = direction
            self.direction_start_time = current_time
            self.direction_history.append({
                'direction': direction,
                'start_time': current_time
            })
    
    def _validate_movement_pattern(self) -> bool:
        """Validate movement pattern for anti-spoofing"""
        if len(self.direction_history) < len(self.required_directions):
            return False
        
        # Check for natural movement timing
        direction_durations = []
        for i, entry in enumerate(self.direction_history):
            if i < len(self.direction_history) - 1:
                duration = self.direction_history[i + 1]['start_time'] - entry['start_time']
                direction_durations.append(duration)
        
        if direction_durations:
            # Check for reasonable variation in timing (not too mechanical)
            duration_variance = np.var(direction_durations)
            if duration_variance < 0.3:  # Too regular
                return False
        
        # Ensure all required directions were completed
        return all(direction in self.completed_directions for direction in self.required_directions)
    
    def _calculate_confidence(self) -> float:
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        confidence_factors = []
        
        # Movement range quality
        if len(self.head_positions) > 20:
            positions = list(self.head_positions)
            pitch_range = max(p['pitch'] for p in positions) - min(p['pitch'] for p in positions)
            yaw_range = max(p['yaw'] for p in positions) - min(p['yaw'] for p in positions)
            
            # Reward sufficient movement range
            range_factor = min(1.0, (pitch_range + yaw_range) / 60.0)  # Expected ~30° range each
            confidence_factors.append(range_factor)
        
        # Direction completion
        completion_factor = len(self.completed_directions) / len(self.required_directions)
        confidence_factors.append(completion_factor)
        
        # Movement naturalness (if enough data)
        if len(self.direction_history) > 2:
            timing_variance = np.var([
                self.direction_history[i + 1]['start_time'] - entry['start_time']
                for i, entry in enumerate(self.direction_history[:-1])
            ])
            naturalness_factor = min(1.0, timing_variance / 2.0)  # Reward natural variation
            confidence_factors.append(naturalness_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0

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
            
        elif current_challenge_type == ChallengeType.SMILE:
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
    """
    
    def __init__(self, anti_replay_window: float = 300.0):  # 5 minutes
        self.current_challenge = None
        self.challenge_history = deque(maxlen=100)
        self.anti_replay_window = anti_replay_window
        self.used_challenges = set()
        
    def generate_random_challenge(self, difficulty: ChallengeDifficulty = None) -> Challenge:
        """Generate random challenge dengan enhanced types"""
        if difficulty is None:
            difficulty = random.choice(list(ChallengeDifficulty))
        
        # Enhanced challenge types with anti-spoofing focus
        challenge_types = [
            ChallengeType.BLINK,
            ChallengeType.SMILE,
            ChallengeType.HEAD_MOVEMENT,
            ChallengeType.MOUTH_OPEN,
            ChallengeType.COVER_EYE,
            ChallengeType.MOVE_CLOSER
        ]
        
        challenge_type = random.choice(challenge_types)
        challenge_id = f"{challenge_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Anti-replay: pastikan challenge tidak diulang dalam window tertentu
        while challenge_id in self.used_challenges:
            challenge_id = f"{challenge_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
            
        self.used_challenges.add(challenge_id)
        
        # Create specific challenge with enhanced parameters
        if challenge_type == ChallengeType.BLINK:
            required_blinks = 2 if difficulty == ChallengeDifficulty.EASY else (3 if difficulty == ChallengeDifficulty.MEDIUM else 4)
            return BlinkChallenge(challenge_id, required_blinks=required_blinks, difficulty=difficulty)
            
        elif challenge_type == ChallengeType.SMILE:
            required_smiles = 1 if difficulty == ChallengeDifficulty.EASY else (2 if difficulty == ChallengeDifficulty.MEDIUM else 3)
            return SmileChallenge(challenge_id, required_smiles=required_smiles, difficulty=difficulty)
            
        elif challenge_type == ChallengeType.HEAD_MOVEMENT:
            if difficulty == ChallengeDifficulty.EASY:
                directions = ['left', 'right']
            elif difficulty == ChallengeDifficulty.MEDIUM:
                directions = ['left', 'right', 'up', 'down']
            else:  # HARD
                directions = ['left', 'right', 'up', 'down']
            return HeadMovementChallenge(challenge_id, required_directions=directions, difficulty=difficulty)
            
        elif challenge_type == ChallengeType.MOUTH_OPEN:
            min_open_time = 1.0 if difficulty == ChallengeDifficulty.EASY else (1.5 if difficulty == ChallengeDifficulty.MEDIUM else 2.0)
            return MouthOpenChallenge(challenge_id, min_open_time=min_open_time)
            
        # For new challenge types, return basic implementation for now
        elif challenge_type == ChallengeType.COVER_EYE:
            return self._create_placeholder_challenge(challenge_id, challenge_type, difficulty)
        elif challenge_type == ChallengeType.MOVE_CLOSER:
            return self._create_placeholder_challenge(challenge_id, challenge_type, difficulty)
    
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
            ChallengeType.SMILE,
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
        """Start new challenge dengan enhanced options"""
        if challenge_type == 'random':
            challenge = self.generate_random_challenge(difficulty)
        elif challenge_type == 'sequence':
            challenge = self.generate_sequence_challenge(difficulty=difficulty or ChallengeDifficulty.MEDIUM)
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
            
        challenge.start()
        self.current_challenge = challenge
        
        print(f"\\n🎯 New Challenge Started: {challenge.description}")
        print(f"   Difficulty: {challenge.difficulty.value}")
        print(f"   Duration: {challenge.duration:.1f}s")
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

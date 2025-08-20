"""
Distance Challenge Implementation for Step 3
Implementing face proximity detection using bounding box size
"""

import time
import numpy as np
from typing import Dict, Optional
from collections import deque
from .challenge_response import Challenge, ChallengeType, ChallengeDifficulty


class DistanceChallenge(Challenge):
    """
    Enhanced distance challenge - ask user to move closer/farther
    Uses face bounding box size to measure distance changes
    """
    
    def __init__(self, challenge_id: str, direction: str = "closer",
                 difficulty: ChallengeDifficulty = ChallengeDifficulty.MEDIUM,
                 duration: float = 15.0):
        """
        Initialize distance challenge
        
        Args:
            direction: "closer" or "farther"
            difficulty: Challenge difficulty level
            duration: Time limit in seconds
        """
        challenge_type = ChallengeType.MOVE_CLOSER if direction == "closer" else ChallengeType.MOVE_FARTHER
        super().__init__(
            challenge_id,
            challenge_type,
            difficulty,
            duration,
            f"Gerakkan wajah {direction} ke kamera (dalam {duration:.0f} detik)"
        )
        
        self.direction = direction  # "closer" or "farther"
        self.baseline_size = None
        self.current_size = None
        self.size_history = deque(maxlen=30)  # Track face size history
        self.target_achieved = False
        self.achievement_start_time = None
        self.required_hold_time = 2.0  # Hold position for 2 seconds
        
        # Distance thresholds based on difficulty
        if difficulty == ChallengeDifficulty.EASY:
            self.size_change_threshold = 0.15  # 15% size change
            self.required_hold_time = 1.5
        elif difficulty == ChallengeDifficulty.MEDIUM:
            self.size_change_threshold = 0.25  # 25% size change
            self.required_hold_time = 2.0
        else:  # HARD
            self.size_change_threshold = 0.35  # 35% size change
            self.required_hold_time = 2.5
        
        # Movement validation
        self.movement_smoothness_threshold = 0.1  # For natural movement detection
        self.min_movement_duration = 1.0  # Minimum time to reach target
        
    def _process_specific_response(self, detection_results: Dict) -> bool:
        """Process distance challenge response"""
        if not detection_results.get('landmarks_detected', False):
            return False
            
        current_time = time.time()
        
        # Get face bounding box size
        face_size = self._calculate_face_size(detection_results)
        if face_size is None:
            return False
        
        self.current_size = face_size
        self.size_history.append(face_size)
        
        # Establish baseline if not set
        if self.baseline_size is None and len(self.size_history) >= 10:
            # Use median of first samples as baseline
            self.baseline_size = np.median(list(self.size_history)[:10])
            print(f"Distance challenge baseline set: {self.baseline_size:.3f}")
        
        # Check achievement if baseline is established
        if self.baseline_size is not None:
            achievement = self._check_distance_achievement(current_time)
            
            if achievement and not self.target_achieved:
                self.target_achieved = True
                self.achievement_start_time = current_time
                print(f"Target distance achieved! Hold for {self.required_hold_time}s")
            elif not achievement and self.target_achieved:
                # Lost target position
                self.target_achieved = False
                self.achievement_start_time = None
                print("Target position lost, continue moving...")
        
        # Store response data
        self.response_data.append({
            'timestamp': current_time,
            'face_size': face_size,
            'baseline_size': self.baseline_size,
            'size_ratio': face_size / self.baseline_size if self.baseline_size else 1.0,
            'target_achieved': self.target_achieved
        })
        
        # Update quality metrics
        quality = self._assess_detection_quality(detection_results)
        self.quality_metrics.append(quality)
        
        # Check success (hold target position for required time)
        if (self.target_achieved and self.achievement_start_time and
            current_time - self.achievement_start_time >= self.required_hold_time):
            self.success = self._validate_movement_pattern()
            self.completed = True
            return True
        
        # Check timeout
        if self.is_expired():
            self.completed = True
            return True
            
        return False
    
    def _calculate_face_size(self, detection_results: Dict) -> Optional[float]:
        """Calculate face size from bounding box or landmarks"""
        
        # Method 1: Use bounding box if available
        face_box = detection_results.get('face_box')
        if face_box:
            width = face_box.get('width', 0)
            height = face_box.get('height', 0)
            return math.sqrt(width * height)  # Use area as size metric
        
        # Method 2: Use landmarks to estimate face size
        landmarks = detection_results.get('landmark_coordinates', [])
        if len(landmarks) >= 468:  # MediaPipe full face landmarks
            try:
                # Use face outline landmarks to calculate approximate size
                # Face width: left cheek to right cheek
                left_cheek = landmarks[172]  # Left face boundary
                right_cheek = landmarks[397]  # Right face boundary
                
                # Face height: forehead to chin
                forehead = landmarks[10]     # Top of forehead
                chin = landmarks[152]        # Bottom of chin
                
                # Calculate face dimensions
                face_width = abs(right_cheek[0] - left_cheek[0])
                face_height = abs(chin[1] - forehead[1])
                
                # Return face area as size metric
                return face_width * face_height
                
            except (IndexError, TypeError):
                return None
        
        # Method 3: Use head pose confidence as fallback
        head_pose = detection_results.get('head_pose', {})
        if head_pose and 'confidence' in head_pose:
            # Higher confidence often correlates with larger/clearer face
            return head_pose['confidence']
        
        return None
    
    def _check_distance_achievement(self, current_time: float) -> bool:
        """Check if target distance has been achieved"""
        if self.baseline_size is None or self.current_size is None:
            return False
        
        size_ratio = self.current_size / self.baseline_size
        
        if self.direction == "closer":
            # Face should be larger (closer to camera)
            target_achieved = size_ratio >= (1.0 + self.size_change_threshold)
        else:  # "farther"
            # Face should be smaller (farther from camera)
            target_achieved = size_ratio <= (1.0 - self.size_change_threshold)
        
        return target_achieved
    
    def _validate_movement_pattern(self) -> bool:
        """Validate that movement was natural and intentional"""
        if len(self.size_history) < 10:
            return True  # Not enough data to validate
        
        size_values = list(self.size_history)
        
        # Check for smooth movement (not too abrupt)
        size_differences = [abs(size_values[i] - size_values[i-1]) 
                           for i in range(1, len(size_values))]
        
        if size_differences:
            max_change = max(size_differences)
            avg_change = np.mean(size_differences)
            
            # Movement should be relatively smooth
            smoothness_ratio = max_change / (avg_change + 0.001)  # Avoid division by zero
            if smoothness_ratio > 5.0:  # Too abrupt
                return False
        
        # Check movement direction consistency
        if len(size_values) >= 5:
            recent_trend = np.polyfit(range(len(size_values[-5:])), size_values[-5:], 1)[0]
            
            if self.direction == "closer" and recent_trend <= 0:
                return False  # Should be increasing
            elif self.direction == "farther" and recent_trend >= 0:
                return False  # Should be decreasing
        
        return True
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence score based on movement quality"""
        if not self.success or len(self.response_data) == 0:
            return 0.0
            
        confidence_factors = []
        
        # Movement achievement
        if self.target_achieved:
            confidence_factors.append(1.0)
        
        # Movement smoothness
        if len(self.size_history) > 5:
            size_values = list(self.size_history)
            size_variance = np.var(size_values)
            # Moderate variance indicates natural movement
            smoothness_factor = min(1.0, 1.0 - abs(size_variance - 0.1) / 0.1)
            confidence_factors.append(max(0.3, smoothness_factor))
        
        # Hold duration quality
        if self.achievement_start_time:
            hold_duration = time.time() - self.achievement_start_time
            hold_factor = min(1.0, hold_duration / self.required_hold_time)
            confidence_factors.append(hold_factor)
        
        # Movement direction consistency
        if len(self.response_data) > 10:
            size_ratios = [d['size_ratio'] for d in self.response_data if d['size_ratio']]
            if size_ratios:
                direction_trend = np.polyfit(range(len(size_ratios)), size_ratios, 1)[0]
                expected_trend = 1.0 if self.direction == "closer" else -1.0
                direction_factor = max(0.0, min(1.0, direction_trend * expected_trend))
                confidence_factors.append(direction_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def get_progress_info(self) -> Dict:
        """Get current progress information for UI display"""
        if self.baseline_size is None or self.current_size is None:
            return {
                'status': 'initializing',
                'message': 'Establishing baseline distance...',
                'progress': 0.0
            }
        
        size_ratio = self.current_size / self.baseline_size
        
        if self.direction == "closer":
            target_ratio = 1.0 + self.size_change_threshold
            progress = min(1.0, max(0.0, (size_ratio - 1.0) / self.size_change_threshold))
            message = f"Move closer to camera (current: {size_ratio:.1%})"
        else:  # "farther"
            target_ratio = 1.0 - self.size_change_threshold
            progress = min(1.0, max(0.0, (1.0 - size_ratio) / self.size_change_threshold))
            message = f"Move farther from camera (current: {size_ratio:.1%})"
        
        if self.target_achieved:
            if self.achievement_start_time:
                hold_time = time.time() - self.achievement_start_time
                remaining_hold = max(0, self.required_hold_time - hold_time)
                message = f"Hold position for {remaining_hold:.1f}s more"
                progress = min(1.0, hold_time / self.required_hold_time)
            else:
                message = "Target achieved! Hold position..."
                progress = 1.0
        
        return {
            'status': 'target_achieved' if self.target_achieved else 'moving',
            'message': message,
            'progress': progress,
            'size_ratio': size_ratio,
            'target_ratio': target_ratio
        }


# Import math for calculations
import math

"""
Enhanced Anti-Spoofing Integration
=================================

This module implements the Step 1 requirements from yangIni.md:
- Real-time face anti-spoofing detection that runs BEFORE attendance checking
- Multiple anti-spoofing techniques simultaneously
- Challenge-response system integration
- Progress indicators and confidence thresholds
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple
from collections import deque
import json

# Import our anti-spoofing components
from src.models.antispoofing_cnn_model import RealTimeAntiSpoofingDetector
from src.detection.landmark_detection import FacialLandmarkDetector
from src.challenge.challenge_response import ChallengeResponseSystem

logger = logging.getLogger(__name__)

class RealTimeAntiSpoofingSystem:
    """
    Real-time anti-spoofing system implementing Step 1 requirements.
    
    This system runs continuously while a face is detected and applies
    multiple anti-spoofing techniques simultaneously before any attendance checking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Load configuration
        self.config = config or self._get_default_config()
        
        # Initialize core components
        self.cnn_detector = RealTimeAntiSpoofingDetector(
            model_path=self.config.get('cnn_model_path'),
            device=self.config.get('device', 'cpu')
        )
        
        self.landmark_detector = FacialLandmarkDetector()
        self.challenge_system = ChallengeResponseSystem()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # System state
        self.detection_state = 'INIT'  # INIT -> DETECTING -> CHALLENGING -> VERIFIED -> FAILED
        self.session_start_time = None
        self.challenge_start_time = None
        self.detection_history = deque(maxlen=30)  # Store last 30 frames of results
        
        # Progress tracking
        self.progress = {
            'face_detected': False,
            'cnn_analysis': 0.0,
            'landmark_detection': 0.0,
            'challenge_completion': 0.0,
            'overall_progress': 0.0
        }
        
        # Challenge state
        self.current_challenge = None
        self.challenge_attempts = 0
        self.max_challenge_attempts = 3
        
        # Detection results aggregation
        self.aggregated_results = {
            'cnn_results': deque(maxlen=15),
            'landmark_results': deque(maxlen=15),
            'challenge_results': deque(maxlen=5)
        }
        
        logger.info("Real-time anti-spoofing system initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration based on Step 1 requirements"""
        return {
            'confidence_threshold': 0.95,  # 95% confidence as specified
            'session_timeout': 60,  # 60 seconds total session timeout
            'challenge_timeout': 15,  # 15 seconds per challenge
            'min_detection_frames': 10,  # Minimum frames for stable detection
            'cnn_weight': 0.6,  # CNN-based texture analysis weight
            'landmark_weight': 0.2,  # Landmark-based micro-movement weight
            'challenge_weight': 0.2,  # Challenge-response weight
            'face_detection_threshold': 0.8,
            'device': 'cpu',
            'cnn_model_path': None
        }
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Main processing function implementing Step 1 requirements.
        
        Processes each frame through the complete anti-spoofing pipeline:
        1. Face detection
        2. CNN-based texture analysis
        3. Landmark-based micro-movement detection  
        4. Challenge-response system
        5. Color space analysis
        
        Args:
            frame: Input BGR frame from webcam
            
        Returns:
            Dict with detection results and progress information
        """
        current_time = time.time()
        
        # Initialize session if needed
        if self.detection_state == 'INIT':
            self.session_start_time = current_time
            self.detection_state = 'DETECTING'
            logger.info("Anti-spoofing session started")
        
        # Check session timeout
        if self._is_session_timeout(current_time):
            return self._handle_timeout()
        
        try:
            # Step 1: Detect if face is present
            face_detection_result = self._detect_face_presence(frame)
            
            if not face_detection_result['face_detected']:
                return self._create_result(
                    is_real=False,
                    confidence=0.0,
                    status='no_face',
                    message='Posisikan wajah Anda di depan kamera',
                    progress=self.progress
                )
            
            # Step 2: Apply multiple anti-spoofing techniques simultaneously
            antispoofing_results = self._apply_antispoofing_techniques(frame)
            
            # Step 3: Update progress and state
            self._update_progress(antispoofing_results)
            
            # Step 4: Check if verification is complete
            verification_result = self._check_verification_completion(antispoofing_results)
            
            # Step 5: Handle challenge system
            if self.detection_state == 'CHALLENGING':
                challenge_result = self._process_challenge(frame, antispoofing_results)
                if challenge_result:
                    verification_result = challenge_result
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return self._create_error_result(str(e))
    
    def _detect_face_presence(self, frame: np.ndarray) -> Dict:
        """Detect if a face is present in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        face_detected = len(faces) > 0
        face_count = len(faces)
        
        # Update progress
        self.progress['face_detected'] = face_detected
        
        result = {
            'face_detected': face_detected,
            'face_count': face_count,
            'face_locations': faces.tolist() if face_detected else []
        }
        
        if face_count > 1:
            result['warning'] = 'Terdeteksi lebih dari satu wajah. Pastikan hanya Anda yang terlihat.'
        
        return result
    
    def _apply_antispoofing_techniques(self, frame: np.ndarray) -> Dict:
        """
        Apply multiple anti-spoofing techniques simultaneously as per Step 1:
        - CNN-based texture analysis to detect print/screen artifacts
        - Landmark-based micro-movement detection
        - Color space analysis for detecting unnatural skin tones
        """
        results = {}
        
        # 1. CNN-based texture analysis
        try:
            cnn_result = self.cnn_detector.detect_antispoofing(frame)
            results['cnn_analysis'] = cnn_result
            self.aggregated_results['cnn_results'].append(cnn_result)
        except Exception as e:
            logger.error(f"CNN analysis error: {e}")
            results['cnn_analysis'] = {'is_real_face': False, 'confidence': 0.0, 'error': str(e)}
        
        # 2. Landmark-based micro-movement detection
        try:
            landmark_result = self.landmark_detector.process_frame(frame)
            results['landmark_analysis'] = landmark_result
            self.aggregated_results['landmark_results'].append(landmark_result)
        except Exception as e:
            logger.error(f"Landmark analysis error: {e}")
            results['landmark_analysis'] = {'landmarks_detected': False, 'error': str(e)}
        
        # 3. Color space analysis (already integrated in CNN detector)
        # This is handled within the CNN detector's detailed analysis
        
        return results
    
    def _update_progress(self, antispoofing_results: Dict):
        """Update progress indicators based on detection results"""
        
        # CNN Analysis Progress
        cnn_result = antispoofing_results.get('cnn_analysis', {})
        cnn_confidence = cnn_result.get('confidence', 0.0)
        self.progress['cnn_analysis'] = min(1.0, cnn_confidence * 1.2)  # Boost for visual feedback
        
        # Landmark Detection Progress
        landmark_result = antispoofing_results.get('landmark_analysis', {})
        landmarks_detected = landmark_result.get('landmarks_detected', False)
        has_movement = (landmark_result.get('head_movement', False) or 
                       landmark_result.get('blink_count', 0) > 0)
        
        landmark_progress = 0.0
        if landmarks_detected:
            landmark_progress += 0.5
        if has_movement:
            landmark_progress += 0.5
        
        self.progress['landmark_detection'] = landmark_progress
        
        # Overall Progress (weighted average)
        weights = self.config
        self.progress['overall_progress'] = (
            weights['cnn_weight'] * self.progress['cnn_analysis'] +
            weights['landmark_weight'] * self.progress['landmark_detection'] +
            weights['challenge_weight'] * self.progress['challenge_completion']
        )
    
    def _check_verification_completion(self, antispoofing_results: Dict) -> Dict:
        """Check if anti-spoofing verification is complete"""
        
        # Get aggregated confidence from recent frames
        aggregated_confidence = self._calculate_aggregated_confidence(antispoofing_results)
        
        # Check if confidence threshold is met
        if aggregated_confidence >= self.config['confidence_threshold']:
            # Additional check: ensure we have enough stable frames
            if len(self.aggregated_results['cnn_results']) >= self.config['min_detection_frames']:
                self.detection_state = 'VERIFIED'
                
                return self._create_result(
                    is_real=True,
                    confidence=aggregated_confidence,
                    status='verified',
                    message='Verifikasi anti-spoofing berhasil! Wajah Anda telah terverifikasi sebagai asli.',
                    progress=self.progress,
                    details=antispoofing_results
                )
        
        # Check if we need to start challenging
        cnn_confidence = antispoofing_results.get('cnn_analysis', {}).get('confidence', 0.0)
        if (cnn_confidence > 0.7 and 
            self.detection_state == 'DETECTING' and 
            not self.current_challenge):
            
            return self._initiate_challenge()
        
        # Continue detection
        return self._create_result(
            is_real=False,
            confidence=aggregated_confidence,
            status='processing',
            message=self._get_processing_message(),
            progress=self.progress,
            details=antispoofing_results
        )
    
    def _calculate_aggregated_confidence(self, current_results: Dict) -> float:
        """Calculate aggregated confidence from recent frames using weighted voting"""
        
        # Get recent CNN results
        recent_cnn = list(self.aggregated_results['cnn_results'])[-5:]  # Last 5 frames
        cnn_confidences = [r.get('confidence', 0.0) for r in recent_cnn if 'error' not in r]
        avg_cnn_confidence = np.mean(cnn_confidences) if cnn_confidences else 0.0
        
        # Get recent landmark results
        recent_landmarks = list(self.aggregated_results['landmark_results'])[-5:]
        landmark_scores = []
        for lr in recent_landmarks:
            score = 0.0
            if lr.get('landmarks_detected', False):
                score += 0.5
            if lr.get('head_movement', False) or lr.get('blink_count', 0) > 0:
                score += 0.5
            landmark_scores.append(score)
        
        avg_landmark_confidence = np.mean(landmark_scores) if landmark_scores else 0.0
        
        # Challenge confidence (if completed)
        challenge_confidence = self.progress['challenge_completion']
        
        # Weighted combination as per Step 1 requirements
        weights = self.config
        aggregated = (
            weights['cnn_weight'] * avg_cnn_confidence +
            weights['landmark_weight'] * avg_landmark_confidence +
            weights['challenge_weight'] * challenge_confidence
        )
        
        return aggregated
    
    def _initiate_challenge(self) -> Dict:
        """Initiate challenge-response system"""
        self.detection_state = 'CHALLENGING'
        self.challenge_start_time = time.time()
        self.current_challenge = self.challenge_system.generate_random_challenge()
        
        logger.info(f"Challenge initiated: {self.current_challenge}")
        
        return self._create_result(
            is_real=False,
            confidence=self.progress['overall_progress'],
            status='challenging',
            message=f'Instruksi: {self.current_challenge.instruction}',
            progress=self.progress,
            challenge_info={
                'instruction': self.current_challenge.instruction,
                'timeout': self.config['challenge_timeout'],
                'attempts_remaining': self.max_challenge_attempts - self.challenge_attempts
            }
        )
    
    def _process_challenge(self, frame: np.ndarray, antispoofing_results: Dict) -> Optional[Dict]:
        """Process active challenge"""
        if not self.current_challenge:
            return None
        
        current_time = time.time()
        challenge_elapsed = current_time - self.challenge_start_time
        
        # Check challenge timeout
        if challenge_elapsed > self.config['challenge_timeout']:
            return self._handle_challenge_timeout()
        
        # Process challenge with landmark results
        landmark_result = antispoofing_results.get('landmark_analysis', {})
        challenge_result = self.challenge_system.process_frame(landmark_result)
        
        if challenge_result and challenge_result.success:
            # Challenge completed successfully
            self.progress['challenge_completion'] = 1.0
            self.current_challenge = None
            self.detection_state = 'DETECTING'  # Return to detection for final verification
            
            return self._create_result(
                is_real=False,  # Still need final verification
                confidence=self.progress['overall_progress'],
                status='challenge_completed',
                message='Tantangan berhasil diselesaikan! Melakukan verifikasi final...',
                progress=self.progress
            )
        elif challenge_result and not challenge_result.success:
            # Challenge failed
            return self._handle_challenge_failure()
        
        # Challenge in progress
        challenge_progress = min(1.0, challenge_elapsed / self.config['challenge_timeout'])
        self.progress['challenge_completion'] = challenge_progress * 0.5  # Partial progress
        
        time_remaining = self.config['challenge_timeout'] - challenge_elapsed
        
        return self._create_result(
            is_real=False,
            confidence=self.progress['overall_progress'],
            status='challenging',
            message=f'{self.current_challenge.instruction} (Sisa waktu: {int(time_remaining)}s)',
            progress=self.progress,
            challenge_info={
                'instruction': self.current_challenge.instruction,
                'time_remaining': time_remaining,
                'progress': challenge_progress
            }
        )
    
    def _handle_challenge_timeout(self) -> Dict:
        """Handle challenge timeout"""
        self.challenge_attempts += 1
        
        if self.challenge_attempts >= self.max_challenge_attempts:
            return self._handle_verification_failure('Gagal menyelesaikan tantangan dalam waktu yang ditentukan')
        
        # Retry challenge
        self.current_challenge = None
        self.challenge_start_time = None
        
        return self._create_result(
            is_real=False,
            confidence=0.0,
            status='challenge_retry',
            message=f'Waktu habis. Mencoba tantangan baru... (Percobaan {self.challenge_attempts + 1}/{self.max_challenge_attempts})',
            progress=self.progress
        )
    
    def _handle_challenge_failure(self) -> Dict:
        """Handle challenge failure"""
        self.challenge_attempts += 1
        
        if self.challenge_attempts >= self.max_challenge_attempts:
            return self._handle_verification_failure('Gagal menyelesaikan tantangan setelah beberapa percobaan')
        
        return self._create_result(
            is_real=False,
            confidence=0.0,
            status='challenge_failed',
            message=f'Tantangan gagal. Mencoba lagi... (Percobaan {self.challenge_attempts + 1}/{self.max_challenge_attempts})',
            progress=self.progress
        )
    
    def _handle_verification_failure(self, reason: str) -> Dict:
        """Handle verification failure"""
        self.detection_state = 'FAILED'
        
        return self._create_result(
            is_real=False,
            confidence=0.0,
            status='failed',
            message=f'Verifikasi anti-spoofing gagal: {reason}',
            progress=self.progress
        )
    
    def _is_session_timeout(self, current_time: float) -> bool:
        """Check if session has timed out"""
        if not self.session_start_time:
            return False
        
        elapsed = current_time - self.session_start_time
        return elapsed > self.config['session_timeout']
    
    def _handle_timeout(self) -> Dict:
        """Handle session timeout"""
        self.detection_state = 'FAILED'
        
        return self._create_result(
            is_real=False,
            confidence=0.0,
            status='timeout',
            message='Sesi verifikasi anti-spoofing telah berakhir. Silakan coba lagi.',
            progress=self.progress
        )
    
    def _get_processing_message(self) -> str:
        """Get appropriate processing message based on current state"""
        if self.progress['overall_progress'] < 0.3:
            return 'Menganalisis wajah Anda...'
        elif self.progress['overall_progress'] < 0.6:
            return 'Melakukan deteksi gerakan alami...'
        elif self.progress['overall_progress'] < 0.9:
            return 'Verifikasi keautentikan wajah...'
        else:
            return 'Menyelesaikan verifikasi anti-spoofing...'
    
    def _create_result(self, is_real: bool, confidence: float, status: str, 
                      message: str, progress: Dict, details: Optional[Dict] = None,
                      challenge_info: Optional[Dict] = None) -> Dict:
        """Create standardized result dictionary"""
        result = {
            'is_real_face': is_real,
            'confidence': confidence,
            'status': status,
            'message': message,
            'progress': progress.copy(),
            'timestamp': time.time(),
            'session_elapsed': time.time() - self.session_start_time if self.session_start_time else 0,
            'detection_state': self.detection_state
        }
        
        if details:
            result['detailed_analysis'] = details
        
        if challenge_info:
            result['challenge_info'] = challenge_info
        
        return result
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result"""
        return self._create_result(
            is_real=False,
            confidence=0.0,
            status='error',
            message=f'Error: {error_message}',
            progress=self.progress
        )
    
    def reset_session(self):
        """Reset the anti-spoofing session"""
        self.detection_state = 'INIT'
        self.session_start_time = None
        self.challenge_start_time = None
        self.current_challenge = None
        self.challenge_attempts = 0
        
        # Reset progress
        self.progress = {
            'face_detected': False,
            'cnn_analysis': 0.0,
            'landmark_detection': 0.0,
            'challenge_completion': 0.0,
            'overall_progress': 0.0
        }
        
        # Clear aggregated results
        for key in self.aggregated_results:
            self.aggregated_results[key].clear()
        
        logger.info("Anti-spoofing session reset")
    
    def get_session_stats(self) -> Dict:
        """Get session statistics"""
        return {
            'detection_state': self.detection_state,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'challenge_attempts': self.challenge_attempts,
            'frames_processed': len(self.aggregated_results['cnn_results']),
            'current_progress': self.progress.copy(),
            'aggregated_confidence': self._calculate_aggregated_confidence({}) if self.aggregated_results['cnn_results'] else 0.0
        }

"""
Step 5: Integration between Anti-Spoofing and Face Recognition
Implements seamless workflow with state management and performance optimization
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, Optional, Tuple, Any
from enum import Enum
from collections import deque
import uuid

# Import anti-spoofing components
from ..models.antispoofing_cnn_model import RealTimeAntiSpoofingDetector
from ..models.face_recognition_cnn import FaceRecognitionSystem
from ..database.attendance_db import AttendanceDatabase
from ..challenge.challenge_response import ChallengeResponseSystem
from ..detection.landmark_detection import LivenessVerifier

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System states for Step 5 integration"""
    INIT = "init"
    ANTI_SPOOFING = "anti_spoofing"
    RECOGNIZING = "recognizing"
    SUCCESS = "success"
    FAILED = "failed"


class Step5IntegratedSystem:
    """
    Step 5 implementation: Seamless integration between anti-spoofing and face recognition
    
    Workflow:
    1. Start webcam stream
    2. Run anti-spoofing detection (Phase 1)
    3. If passes, run face recognition (Phase 2)
    4. Display attendance confirmation or registration prompt
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self._initialize_components()
        
        # State management
        self.current_state = SystemState.INIT
        self.session_id = None
        self.state_start_time = None
        self.session_data = {}
        
        # Performance optimization
        self.frame_skip_counter = 0
        self.processing_thread = None
        self.result_queue = queue.Queue()
        self.frame_cache = deque(maxlen=10)
        
        # State timeouts (seconds)
        self.state_timeouts = {
            SystemState.ANTI_SPOOFING: 30,
            SystemState.RECOGNIZING: 15
        }
        
        self.logger.info("Step 5 Integrated System initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for Step 5 system"""
        return {
            'antispoofing_confidence_threshold': 0.85,
            'recognition_confidence_threshold': 0.85,
            'frame_skip_rate': 3,  # Process every 3rd frame
            'enable_threading': True,
            'enable_caching': True,
            'max_concurrent_sessions': 5,
            'log_state_changes': True,
            'performance_monitoring': True
        }
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Anti-spoofing detector
            self.antispoofing_detector = RealTimeAntiSpoofingDetector(
                device=self.config.get('device', 'cpu')
            )
            
            # Face recognition system
            self.face_recognition = FaceRecognitionSystem(
                similarity_threshold=self.config['recognition_confidence_threshold']
            )
            
            # Database
            self.database = AttendanceDatabase()
            
            # Challenge system
            self.challenge_system = ChallengeResponseSystem()
            
            # Liveness verifier for fallback
            self.liveness_verifier = LivenessVerifier()
            
            # Load face embeddings into memory cache
            embeddings = self.database.get_all_embeddings()
            self.face_recognition.load_embeddings_from_database(embeddings)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start_session(self) -> str:
        """Start a new attendance session"""
        self.session_id = str(uuid.uuid4())
        self.current_state = SystemState.INIT
        self.state_start_time = time.time()
        
        self.session_data = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'antispoofing_result': None,
            'recognition_result': None,
            'state_history': [],
            'performance_metrics': {}
        }
        
        self._log_state_change(SystemState.INIT, "Session started")
        return self.session_id
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main processing function implementing Step 5 workflow
        
        Args:
            frame: Input BGR frame from webcam
            
        Returns:
            Dict with current state and processing results
        """
        if not self.session_id:
            return self._create_error_result("No active session")
        
        # Performance optimization: frame skipping
        if self._should_skip_frame():
            return self._get_current_status()
        
        # Check for state timeout
        if self._is_state_timeout():
            return self._handle_timeout()
        
        try:
            # Cache frame for performance
            if self.config['enable_caching']:
                self.frame_cache.append(frame.copy())
            
            # Process based on current state
            if self.current_state == SystemState.INIT:
                return self._transition_to_antispoofing(frame)
            
            elif self.current_state == SystemState.ANTI_SPOOFING:
                return self._process_antispoofing_phase(frame)
            
            elif self.current_state == SystemState.RECOGNIZING:
                return self._process_recognition_phase(frame)
            
            elif self.current_state in [SystemState.SUCCESS, SystemState.FAILED]:
                return self._get_final_result()
            
            else:
                return self._create_error_result(f"Unknown state: {self.current_state}")
                
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return self._create_error_result(f"Processing failed: {str(e)}")
    
    def _should_skip_frame(self) -> bool:
        """Determine if frame should be skipped for performance"""
        if not self.config.get('frame_skip_rate'):
            return False
        
        self.frame_skip_counter += 1
        if self.frame_skip_counter >= self.config['frame_skip_rate']:
            self.frame_skip_counter = 0
            return False
        return True
    
    def _is_state_timeout(self) -> bool:
        """Check if current state has timed out"""
        if not self.state_start_time or self.current_state not in self.state_timeouts:
            return False
        
        elapsed = time.time() - self.state_start_time
        timeout = self.state_timeouts[self.current_state]
        return elapsed > timeout
    
    def _handle_timeout(self) -> Dict[str, Any]:
        """Handle state timeout"""
        self._transition_to_state(SystemState.FAILED)
        
        timeout_message = {
            SystemState.ANTI_SPOOFING: "Anti-spoofing verification timed out. Please try again.",
            SystemState.RECOGNIZING: "Face recognition timed out. Please try again."
        }.get(self.current_state, "Session timed out")
        
        return {
            'session_id': self.session_id,
            'state': self.current_state.value,
            'status': 'timeout',
            'message': timeout_message,
            'timestamp': time.time()
        }
    
    def _transition_to_antispoofing(self, frame: np.ndarray) -> Dict[str, Any]:
        """Transition from INIT to ANTI_SPOOFING state"""
        self._transition_to_state(SystemState.ANTI_SPOOFING)
        
        return {
            'session_id': self.session_id,
            'state': self.current_state.value,
            'status': 'started',
            'message': 'Verifying real person...',
            'phase': 'antispoofing',
            'progress': 0.0,
            'instructions': 'Look directly at the camera and follow the instructions',
            'timestamp': time.time()
        }
    
    def _process_antispoofing_phase(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process Phase 1: Anti-spoofing detection"""
        start_time = time.time()
        
        try:
            # Run anti-spoofing detection
            antispoofing_result = self.antispoofing_detector.detect_antispoofing(frame)
            
            # Store result for analysis
            self.session_data['antispoofing_result'] = antispoofing_result
            
            # Check if anti-spoofing passed
            is_real = antispoofing_result.get('is_real_face', False)
            confidence = antispoofing_result.get('confidence', 0.0)
            
            processing_time = time.time() - start_time
            self.session_data['performance_metrics']['antispoofing_time'] = processing_time
            
            if is_real and confidence >= self.config['antispoofing_confidence_threshold']:
                # Anti-spoofing passed - transition to recognition
                self._transition_to_state(SystemState.RECOGNIZING)
                
                return {
                    'session_id': self.session_id,
                    'state': self.current_state.value,
                    'status': 'verified',
                    'message': 'Face verified! Recognizing...',
                    'phase': 'recognition_starting',
                    'antispoofing_confidence': confidence,
                    'antispoofing_details': antispoofing_result.get('detailed_analysis', {}),
                    'processing_time': processing_time,
                    'timestamp': time.time()
                }
            else:
                # Still processing or failed
                status = 'processing' if confidence > 0.3 else 'warning'
                message = antispoofing_result.get('message', 'Verifying authenticity...')
                
                return {
                    'session_id': self.session_id,
                    'state': self.current_state.value,
                    'status': status,
                    'message': message,
                    'phase': 'antispoofing',
                    'confidence': confidence,
                    'progress': min(confidence / self.config['antispoofing_confidence_threshold'], 1.0),
                    'challenge_info': antispoofing_result.get('challenge_info'),
                    'processing_time': processing_time,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Anti-spoofing processing failed: {e}")
            self._transition_to_state(SystemState.FAILED)
            
            return {
                'session_id': self.session_id,
                'state': self.current_state.value,
                'status': 'error',
                'message': f'Anti-spoofing verification failed: {str(e)}',
                'timestamp': time.time()
            }
    
    def _process_recognition_phase(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process Phase 2: Face recognition"""
        start_time = time.time()
        
        try:
            # Extract face from frame for recognition
            face_image = self._extract_face_for_recognition(frame)
            if face_image is None:
                return {
                    'session_id': self.session_id,
                    'state': self.current_state.value,
                    'status': 'no_face',
                    'message': 'Face not detected for recognition. Please look directly at camera.',
                    'timestamp': time.time()
                }
            
            # Run face recognition
            recognition_result = self.face_recognition.recognize_face(face_image)
            
            # Store result
            self.session_data['recognition_result'] = recognition_result
            
            processing_time = time.time() - start_time
            self.session_data['performance_metrics']['recognition_time'] = processing_time
            
            if recognition_result['success']:
                # Recognition successful
                user_id = recognition_result['user_id']
                confidence = recognition_result['confidence']
                
                # Record attendance
                antispoofing_confidence = self.session_data['antispoofing_result']['confidence']
                attendance_recorded = self._record_attendance(
                    user_id, confidence, antispoofing_confidence, processing_time
                )
                
                if attendance_recorded:
                    self._transition_to_state(SystemState.SUCCESS)
                    
                    # Get user info
                    user_info = self.database.get_user_info(user_id)
                    
                    return {
                        'session_id': self.session_id,
                        'state': self.current_state.value,
                        'status': 'success',
                        'message': f'Welcome, {user_info.get("name", user_id)}! Attendance recorded.',
                        'user_id': user_id,
                        'user_info': user_info,
                        'recognition_confidence': confidence,
                        'antispoofing_confidence': antispoofing_confidence,
                        'processing_time': processing_time,
                        'total_time': time.time() - self.session_data['start_time'],
                        'timestamp': time.time()
                    }
                else:
                    self._transition_to_state(SystemState.FAILED)
                    return {
                        'session_id': self.session_id,
                        'state': self.current_state.value,
                        'status': 'error',
                        'message': 'Failed to record attendance. Please try again.',
                        'timestamp': time.time()
                    }
            else:
                # Recognition failed
                self._transition_to_state(SystemState.FAILED)
                
                return {
                    'session_id': self.session_id,
                    'state': self.current_state.value,
                    'status': 'unknown_user',
                    'message': 'Face not recognized. Please register first.',
                    'recognition_confidence': recognition_result.get('confidence', 0.0),
                    'processing_time': processing_time,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Face recognition processing failed: {e}")
            self._transition_to_state(SystemState.FAILED)
            
            return {
                'session_id': self.session_id,
                'state': self.current_state.value,
                'status': 'error',
                'message': f'Face recognition failed: {str(e)}',
                'timestamp': time.time()
            }
    
    def _extract_face_for_recognition(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region for recognition"""
        try:
            # Use OpenCV face detection for quick extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_image = frame[y1:y2, x1:x2]
            return face_image
            
        except Exception as e:
            self.logger.error(f"Face extraction failed: {e}")
            return None
    
    def _record_attendance(self, user_id: str, recognition_confidence: float,
                          antispoofing_confidence: float, processing_time: float) -> bool:
        """Record attendance in database"""
        try:
            success = self.database.record_attendance(
                user_id=user_id,
                confidence_score=recognition_confidence,
                antispoofing_score=antispoofing_confidence,
                recognition_time=processing_time,
                session_id=self.session_id,
                device_info="Step5_Integrated_System"
            )
            
            if success:
                self.logger.info(f"Attendance recorded for {user_id}")
            else:
                self.logger.error(f"Failed to record attendance for {user_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Attendance recording error: {e}")
            return False
    
    def _transition_to_state(self, new_state: SystemState):
        """Transition to new state with logging"""
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # Log state change
        self._log_state_change(new_state, f"Transitioned from {old_state.value}")
        
        # Update session data
        self.session_data['state_history'].append({
            'from_state': old_state.value,
            'to_state': new_state.value,
            'timestamp': time.time()
        })
    
    def _log_state_change(self, state: SystemState, message: str):
        """Log state changes for debugging"""
        if self.config.get('log_state_changes', True):
            self.logger.info(f"Session {self.session_id}: {state.value} - {message}")
    
    def _get_current_status(self) -> Dict[str, Any]:
        """Get current session status without processing"""
        return {
            'session_id': self.session_id,
            'state': self.current_state.value,
            'status': 'waiting',
            'message': 'Processing...',
            'timestamp': time.time()
        }
    
    def _get_final_result(self) -> Dict[str, Any]:
        """Get final result for completed session"""
        return {
            'session_id': self.session_id,
            'state': self.current_state.value,
            'status': 'completed',
            'final_result': self.session_data.get('recognition_result'),
            'performance_metrics': self.session_data.get('performance_metrics', {}),
            'timestamp': time.time()
        }
    
    def _create_error_result(self, message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'session_id': self.session_id,
            'state': 'error',
            'status': 'error',
            'message': message,
            'timestamp': time.time()
        }
    
    def reset_session(self):
        """Reset current session"""
        self.session_id = None
        self.current_state = SystemState.INIT
        self.state_start_time = None
        self.session_data.clear()
        self.frame_cache.clear()
        self.frame_skip_counter = 0
        
        self.logger.info("Session reset")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            'face_recognition_stats': self.face_recognition.get_system_info(),
            'database_stats': self.database.get_database_stats(),
            'current_state': self.current_state.value,
            'session_active': self.session_id is not None,
            'cache_size': len(self.frame_cache),
            'config': self.config
        }
    
    def shutdown(self):
        """Shutdown system and cleanup resources"""
        self.reset_session()
        
        # Stop any background threads
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("Step 5 Integrated System shutdown complete")


def create_step5_system(config: Optional[Dict] = None) -> Step5IntegratedSystem:
    """Factory function to create Step 5 integrated system"""
    return Step5IntegratedSystem(config)
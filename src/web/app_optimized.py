"""
OPTIMIZED Flask Web Application for Face Anti-Spoofing Attendance System
Implementasi yang dioptimalkan untuk real-time performance dengan frame processing pipeline
"""

import os
import sqlite3
import time
import json
import base64
import cv2
import numpy as np
import io
import secrets
import threading
import queue
import random
import sys
import logging
import traceback
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from collections import deque

# Face recognition import
try:
    import face_recognition
except ImportError:
    print("Warning: face_recognition not available, using fallback")
    face_recognition = None


def convert_to_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to JSON-compatible types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


class SequentialDetectionState:
    """
    Sequential detection state management for 2-phase process:
    Phase 1: Liveness & Landmark (Anti-Spoofing)
    Phase 2: Face Recognition (CNN)
    """
    def __init__(self):
        self.phase = 'liveness'  # 'liveness' -> 'recognition' -> 'complete'
        self.liveness_passed = False
        self.landmark_passed = False
        self.anti_spoofing_passed = False
        self.recognition_result = None
        self.start_time = time.time()
        self.phase_start_time = time.time()
        
        # Phase timeouts
        self.timeouts = {
            'liveness': 30,  # 30 seconds for anti-spoofing
            'recognition': 15  # 15 seconds for face recognition
        }
        
        # Challenge state for landmark detection
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_completed = False
        self.challenge_progress = 0.0
        
        # Available challenges for phase 1
        self.challenges = [
            {'type': 'blink', 'instruction': 'Kedipkan mata 3 kali', 'target_count': 3},
            {'type': 'head_left', 'instruction': 'Gerakkan kepala ke kiri', 'duration': 2.0},
            {'type': 'head_right', 'instruction': 'Gerakkan kepala ke kanan', 'duration': 2.0},
            {'type': 'smile', 'instruction': 'Senyum selama 2 detik', 'duration': 2.0},
        ]
        
        # Generate initial challenge
        self.generate_new_challenge()
    
    def can_proceed_to_recognition(self):
        """Check if both anti-spoofing methods have passed"""
        return self.liveness_passed and self.landmark_passed
    
    def transition_to_recognition(self):
        """Transition from phase 1 to phase 2"""
        if self.can_proceed_to_recognition():
            self.phase = 'recognition'
            self.phase_start_time = time.time()
            self.anti_spoofing_passed = True
            return True
        return False
    
    def is_phase_timeout(self):
        """Check if current phase has timed out"""
        elapsed = time.time() - self.phase_start_time
        return elapsed > self.timeouts.get(self.phase, 30)
    
    def generate_new_challenge(self):
        """Generate a new random challenge for landmark detection"""
        if not self.landmark_passed:
            self.current_challenge = random.choice(self.challenges).copy()
            self.challenge_start_time = time.time()
            self.challenge_completed = False
            self.challenge_progress = 0.0
            
            # Add challenge-specific state
            if self.current_challenge['type'] == 'blink':
                self.current_challenge['current_count'] = 0
    
    def update_challenge(self, landmark_results):
        """Update challenge progress"""
        if not self.current_challenge or self.landmark_passed:
            return True
        
        current_time = time.time()
        challenge_type = self.current_challenge['type']
        
        # Check challenge timeout (10 seconds per challenge)
        if current_time - self.challenge_start_time > 10.0:
            self.generate_new_challenge()
            return False
        
        # Update based on challenge type
        if challenge_type == 'blink':
            blink_count = landmark_results.get('blink_count', 0)
            if blink_count > self.current_challenge['current_count']:
                self.current_challenge['current_count'] = blink_count
            
            self.challenge_progress = min(1.0, self.current_challenge['current_count'] / self.current_challenge['target_count'])
            
            if self.current_challenge['current_count'] >= self.current_challenge['target_count']:
                self.landmark_passed = True
                self.challenge_completed = True
                
        elif challenge_type in ['head_left', 'head_right']:
            head_movement = landmark_results.get('head_movement', False)
            head_direction = landmark_results.get('head_direction', 'center')
            
            target_direction = 'left' if challenge_type == 'head_left' else 'right'
            
            if head_direction == target_direction:
                elapsed = current_time - self.challenge_start_time
                self.challenge_progress = min(1.0, elapsed / self.current_challenge['duration'])
                
                if elapsed >= self.current_challenge['duration']:
                    self.landmark_passed = True
                    self.challenge_completed = True
            else:
                self.challenge_progress = max(0.0, self.challenge_progress - 0.1)
                
        elif challenge_type == 'smile':
            mouth_open = landmark_results.get('mouth_open', False)
            if mouth_open:  # Simplified smile detection
                elapsed = current_time - self.challenge_start_time
                self.challenge_progress = min(1.0, elapsed / self.current_challenge['duration'])
                
                if elapsed >= self.current_challenge['duration']:
                    self.landmark_passed = True
                    self.challenge_completed = True
        
        return self.landmark_passed
    
    def get_challenge_info(self):
        """Get current challenge information"""
        if not self.current_challenge:
            return None
        
        return {
            'instruction': self.current_challenge['instruction'],
            'progress': self.challenge_progress,
            'time_remaining': max(0, 10.0 - (time.time() - self.challenge_start_time)),
            'completed': self.challenge_completed
        }
    
    def get_status(self):
        """Get current status for UI"""
        return {
            'phase': self.phase,
            'liveness_passed': self.liveness_passed,
            'landmark_passed': self.landmark_passed,
            'anti_spoofing_passed': self.anti_spoofing_passed,
            'can_proceed': self.can_proceed_to_recognition(),
            'challenge_info': self.get_challenge_info(),
            'is_timeout': self.is_phase_timeout(),
            'recognition_result': self.recognition_result
        }


class SecurityAssessmentState:
    """
    Persistent state management for security assessment
    Fixes issue where detection methods reset every frame
    """
    def __init__(self):
        # Movement Detection State
        self.movement_verified = False
        self.movement_last_verified = None
        self.movement_grace_period = 3.0  # seconds
        self.movement_history = deque(maxlen=10)  # Track recent movements
        
        # CNN Detection State
        self.cnn_verified = False
        self.cnn_confidence_history = deque(maxlen=30)  # 1 second at 30fps
        self.cnn_verification_threshold = 0.7
        self.cnn_consistency_required = 20  # frames
        
        # Landmark Detection & Challenge State
        self.landmark_verified = False
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_timeout = 10.0  # seconds
        self.challenge_completed = False
        self.challenge_progress = 0.0
        
        # Available challenges
        self.challenges = [
            {'type': 'blink', 'instruction': 'Kedipkan mata 3 kali', 'target_count': 3},
            {'type': 'head_left', 'instruction': 'Hadapkan kepala ke kiri', 'duration': 2.0},
            {'type': 'head_right', 'instruction': 'Hadapkan kepala ke kanan', 'duration': 2.0},
            {'type': 'smile', 'instruction': 'Senyum selama 2 detik', 'duration': 2.0},
            {'type': 'mouth_open', 'instruction': 'Buka mulut selama 2 detik', 'duration': 2.0}
        ]
        
        # Overall state
        self.last_update = time.time()
        
    def update_movement(self, has_movement, head_movement=False):
        """Update movement detection with grace period"""
        current_time = time.time()
        
        # Add to movement history
        self.movement_history.append({
            'time': current_time,
            'movement': has_movement or head_movement
        })
        
        # Check if there was recent movement
        recent_movement = any(
            entry['movement'] and (current_time - entry['time']) <= self.movement_grace_period
            for entry in self.movement_history
        )
        
        if recent_movement:
            self.movement_verified = True
            self.movement_last_verified = current_time
        elif self.movement_verified:
            # Keep verified status during grace period
            if current_time - self.movement_last_verified <= self.movement_grace_period:
                # Still in grace period
                pass
            else:
                # Grace period expired
                self.movement_verified = False
        
        return self.movement_verified
    
    def update_cnn(self, confidence, is_live):
        """Update CNN detection with consistency check"""
        self.cnn_confidence_history.append({
            'confidence': confidence,
            'is_live': is_live,
            'time': time.time()
        })
        
        # Check consistency over recent frames
        if len(self.cnn_confidence_history) >= self.cnn_consistency_required:
            recent_confidences = [entry['confidence'] for entry in list(self.cnn_confidence_history)[-self.cnn_consistency_required:]]
            recent_live_count = sum(1 for entry in list(self.cnn_confidence_history)[-self.cnn_consistency_required:] if entry['is_live'])
            
            avg_confidence = np.mean(recent_confidences)
            live_ratio = recent_live_count / self.cnn_consistency_required
            
            # Verify if consistently good
            if avg_confidence >= self.cnn_verification_threshold and live_ratio >= 0.7:
                self.cnn_verified = True
        
        return self.cnn_verified
    
    def generate_new_challenge(self):
        """Generate a new random challenge"""
        if not self.landmark_verified:
            self.current_challenge = random.choice(self.challenges).copy()
            self.challenge_start_time = time.time()
            self.challenge_completed = False
            self.challenge_progress = 0.0
            
            # Add challenge-specific state
            if self.current_challenge['type'] == 'blink':
                self.current_challenge['current_count'] = 0
                self.current_challenge['last_blink_time'] = 0
            
    def update_challenge(self, landmark_results):
        """Update challenge progress based on landmark results"""
        if not self.current_challenge or self.landmark_verified:
            return True
        
        current_time = time.time()
        challenge_type = self.current_challenge['type']
        
        # Check timeout
        if current_time - self.challenge_start_time > self.challenge_timeout:
            self.generate_new_challenge()  # Generate new challenge
            return False
        
        # Update challenge based on type
        if challenge_type == 'blink':
            blink_count = landmark_results.get('blink_count', 0)
            if blink_count > self.current_challenge['current_count']:
                self.current_challenge['current_count'] = blink_count
                self.current_challenge['last_blink_time'] = current_time
            
            self.challenge_progress = min(1.0, self.current_challenge['current_count'] / self.current_challenge['target_count'])
            
            if self.current_challenge['current_count'] >= self.current_challenge['target_count']:
                self.landmark_verified = True
                self.challenge_completed = True
                
        elif challenge_type in ['head_left', 'head_right']:
            head_movement = landmark_results.get('head_movement', False)
            head_direction = landmark_results.get('head_direction', 'center')
            
            target_direction = 'left' if challenge_type == 'head_left' else 'right'
            
            if head_direction == target_direction:
                elapsed = current_time - self.challenge_start_time
                self.challenge_progress = min(1.0, elapsed / self.current_challenge['duration'])
                
                if elapsed >= self.current_challenge['duration']:
                    self.landmark_verified = True
                    self.challenge_completed = True
            else:
                self.challenge_progress = 0.0
                
        elif challenge_type == 'smile':
            # Check for smile detection (simplified)
            mouth_open = landmark_results.get('mouth_open', False)
            # In real implementation, you'd check for smile landmarks
            if mouth_open:  # Temporary placeholder
                elapsed = current_time - self.challenge_start_time
                self.challenge_progress = min(1.0, elapsed / self.current_challenge['duration'])
                
                if elapsed >= self.current_challenge['duration']:
                    self.landmark_verified = True
                    self.challenge_completed = True
                    
        elif challenge_type == 'mouth_open':
            mouth_open = landmark_results.get('mouth_open', False)
            if mouth_open:
                elapsed = current_time - self.challenge_start_time
                self.challenge_progress = min(1.0, elapsed / self.current_challenge['duration'])
                
                if elapsed >= self.current_challenge['duration']:
                    self.landmark_verified = True
                    self.challenge_completed = True
            else:
                self.challenge_progress = 0.0
        
        return self.landmark_verified
    
    def get_challenge_info(self):
        """Get current challenge information for UI"""
        if not self.current_challenge:
            return None
            
        return {
            'instruction': self.current_challenge['instruction'],
            'progress': self.challenge_progress,
            'time_remaining': max(0, self.challenge_timeout - (time.time() - self.challenge_start_time)),
            'completed': self.challenge_completed
        }
    
    def get_security_status(self):
        """Get overall security status"""
        methods_passed = sum([
            self.movement_verified,
            self.cnn_verified, 
            self.landmark_verified
        ])
        
        return {
            'movement_verified': self.movement_verified,
            'cnn_verified': self.cnn_verified,
            'landmark_verified': self.landmark_verified,
            'methods_passed': methods_passed,
            'security_passed': methods_passed >= 2,
            'challenge_info': self.get_challenge_info()
        }
import gc

# Flexible imports that work both as module and standalone
try:
    from ..utils.logger import get_web_logger
    from ..utils.config import SystemConfig
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, project_root)
    
    try:
        from src.utils.logger import get_web_logger
        from src.utils.config import SystemConfig
    except ImportError:
        # Create minimal fallback classes
        import logging
        
        def get_web_logger():
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            return logger
        
        class SystemConfig:
            def __init__(self):
                self.web = type('obj', (object,), {
                    'secret_key': 'fallback-secret-key',
                    'database_url': 'sqlite:///attendance.db',
                    'debug': False
                })


class OptimizedFrameProcessor:
    """
    Optimized frame processor dengan pipeline processing dan caching
    """
    
    def __init__(self):
        # Lazy import untuk performa startup
        self.landmark_verifier = None
        self.cnn_predictor = None
        
        # Processing pipeline
        self.frame_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.result_cache = {}
        self.cache_times = {}
        self.cache_duration = 0.1  # 100ms cache
        
        # Performance monitoring
        self.processing_times = deque(maxlen=50)
        self.frame_count = 0
        self.skip_frame_count = 2  # Process every 2nd frame
        
        # Threading
        self.processing_thread = None
        self.stop_processing = False
        
        # Security Assessment State - UPDATED FOR SEQUENTIAL PROCESSING
        self.sequential_states = {}  # Per session sequential state
        self.security_states = {}  # Keep for backward compatibility
        
        print("[OK] OptimizedFrameProcessor initialized")
    
    def get_sequential_state(self, session_id):
        """Get or create sequential detection state for session"""
        if session_id not in self.sequential_states:
            self.sequential_states[session_id] = SequentialDetectionState()
        return self.sequential_states[session_id]
    
    def get_security_state(self, session_id):
        """Get or create security state for session"""
        if session_id not in self.security_states:
            self.security_states[session_id] = SecurityAssessmentState()
            # Generate initial challenge
            self.security_states[session_id].generate_new_challenge()
        return self.security_states[session_id]
    
    def _generate_user_message(self, security_status, security_level, methods_passed):
        """Generate user-friendly message with instructions"""
        challenge_info = security_status.get('challenge_info')
        
        if challenge_info and not security_status['landmark_verified']:
            # Show challenge instruction
            progress = int(challenge_info['progress'] * 100)
            time_left = int(challenge_info['time_remaining'])
            return f"{challenge_info['instruction']} ({progress}% - {time_left}s tersisa)"
        
        # Status messages
        if security_level == "SECURE":
            return "✅ Verifikasi Lengkap - Semua metode berhasil!"
        elif security_level == "GOOD":
            return f"✅ Verifikasi Berhasil - {methods_passed}/3 metode terverifikasi"
        elif security_level == "WARNING":
            remaining = 2 - methods_passed
            return f"⚠️ Butuh {remaining} metode lagi untuk verifikasi lengkap"
        else:
            return "❌ Belum terverifikasi - Ikuti instruksi yang diberikan"
    
    def _init_models_lazy(self):
        """
        Lazy initialization untuk model loading
        """
        if self.landmark_verifier is None:
            try:
                # Try relative import first
                try:
                    from ..detection.optimized_landmark_detection import OptimizedLivenessVerifier
                except ImportError:
                    # Fallback for standalone execution
                    try:
                        from src.detection.optimized_landmark_detection import OptimizedLivenessVerifier
                    except ImportError:
                        from detection.optimized_landmark_detection import OptimizedLivenessVerifier
                
                self.landmark_verifier = OptimizedLivenessVerifier(history_length=15)
                print("[OK] Landmark verifier loaded")
            except Exception as e:
                print(f"❌ Failed to load landmark verifier: {e}")
                # Fallback to original
                try:
                    try:
                        from ..detection.landmark_detection import LivenessVerifier
                    except ImportError:
                        try:
                            from src.detection.landmark_detection import LivenessVerifier
                        except ImportError:
                            from detection.landmark_detection import LivenessVerifier
                    self.landmark_verifier = LivenessVerifier()
                except Exception:
                    print("❌ No landmark verifier available")
                    self.landmark_verifier = None
        
        if self.cnn_predictor is None:
            try:
                # Try relative import first
                try:
                    from ..models.optimized_cnn_model import OptimizedLivenessPredictor
                except ImportError:
                    # Fallback for standalone execution
                    try:
                        from src.models.optimized_cnn_model import OptimizedLivenessPredictor
                    except ImportError:
                        from models.optimized_cnn_model import OptimizedLivenessPredictor
                
                self.cnn_predictor = OptimizedLivenessPredictor(use_quantization=True)
                print("[OK] CNN predictor loaded")
            except Exception as e:
                print(f"❌ Failed to load CNN predictor: {e}")
                self.cnn_predictor = None
    
    def get_cache_key(self, image_data):
        """
        Generate cache key from image data
        """
        # Use hash of first 1000 bytes for speed
        return hash(image_data[:1000])
    
    def process_frame_sequential(self, image, session_id="default", user_id=None):
        """
        Sequential frame processing: Phase 1 (Anti-Spoofing) → Phase 2 (Face Recognition)
        """
        start_time = time.time()
        
        try:
            # Get sequential state
            seq_state = self.get_sequential_state(session_id)
            
            # Check for timeout
            if seq_state.is_phase_timeout():
                # Reset to phase 1 on timeout
                self.sequential_states[session_id] = SequentialDetectionState()
                seq_state = self.sequential_states[session_id]
                
                return {
                    'session_id': str(session_id),
                    'phase': 'liveness',
                    'status': 'timeout',
                    'message': 'Waktu habis. Memulai ulang verifikasi.',
                    'processing_time': time.time() - start_time
                }
            
            # Initialize models if needed
            self._init_models_lazy()
            
            if image is None or image.size == 0:
                return self._create_empty_sequential_result(start_time, seq_state.phase)
            
            # Resize image for consistent processing
            if image.shape[1] > 640 or image.shape[0] > 480:
                image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
            
            # =========================================
            # PHASE 1: ANTI-SPOOFING (LIVENESS + LANDMARK)
            # =========================================
            if seq_state.phase == 'liveness':
                return self._process_anti_spoofing_phase(image, seq_state, session_id, start_time)
            
            # =========================================
            # PHASE 2: FACE RECOGNITION (CNN)
            # =========================================
            elif seq_state.phase == 'recognition':
                return self._process_recognition_phase(image, seq_state, session_id, user_id, start_time)
            
            # =========================================
            # PHASE 3: COMPLETE
            # =========================================
            elif seq_state.phase == 'complete':
                return {
                    'session_id': str(session_id),
                    'phase': 'complete',
                    'status': 'success',
                    'message': 'Verifikasi selesai.',
                    'recognition_result': seq_state.recognition_result,
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Sequential frame processing error: {e}")
            return {
                'session_id': str(session_id),
                'phase': 'error',
                'status': 'error',
                'error': str(e),
                'message': f'Processing error: {str(e)[:50]}',
                'processing_time': processing_time
            }
    
    def _process_anti_spoofing_phase(self, image, seq_state, session_id, start_time):
        """Process Phase 1: Liveness + Landmark Detection"""
        
        # ========= LIVENESS DETECTION =========
        liveness_start = time.time()
        liveness_results = {}
        
        if self.cnn_predictor:
            try:
                cnn_prediction = self.cnn_predictor.predict_optimized(image, use_cache=True)
                liveness_confidence = cnn_prediction.get('confidence', 0.0)
                is_live = cnn_prediction.get('is_live', False)
                
                # Update liveness status
                if liveness_confidence > 0.8 and is_live:
                    seq_state.liveness_passed = True
                
                liveness_results = {
                    'confidence': liveness_confidence,
                    'is_live': is_live,
                    'passed': seq_state.liveness_passed
                }
            except Exception as e:
                print(f"Liveness detection error: {e}")
                liveness_results = {
                    'confidence': 0.0,
                    'is_live': False,
                    'passed': False
                }
        else:
            # Fallback liveness simulation
            liveness_results = {
                'confidence': 0.85,
                'is_live': True,
                'passed': True
            }
            seq_state.liveness_passed = True
        
        liveness_time = time.time() - liveness_start
        
        # ========= LANDMARK DETECTION =========
        landmark_start = time.time()
        landmark_results = {}
        
        if self.landmark_verifier:
            try:
                if hasattr(self.landmark_verifier, 'process_frame_optimized'):
                    landmark_result = self.landmark_verifier.process_frame_optimized(image)
                else:
                    landmark_result = self.landmark_verifier.process_frame(image)
                
                # Update challenge progress
                seq_state.update_challenge(landmark_result)
                
                landmark_results = {
                    'landmarks_detected': landmark_result.get('landmarks_detected', False),
                    'blink_count': landmark_result.get('blink_count', 0),
                    'head_movement': landmark_result.get('head_movement', False),
                    'mouth_open': landmark_result.get('mouth_open', False),
                    'challenge_passed': seq_state.landmark_passed
                }
                
            except Exception as e:
                print(f"Landmark detection error: {e}")
                landmark_results = {
                    'landmarks_detected': False,
                    'blink_count': 0,
                    'head_movement': False,
                    'mouth_open': False,
                    'challenge_passed': False
                }
        else:
            # Fallback landmark simulation
            landmark_results = {
                'landmarks_detected': True,
                'blink_count': 1,
                'head_movement': False,
                'mouth_open': False,
                'challenge_passed': False
            }
        
        landmark_time = time.time() - landmark_start
        
        # ========= CHECK PHASE 1 COMPLETION =========
        if seq_state.can_proceed_to_recognition():
            seq_state.transition_to_recognition()
            
            return {
                'session_id': str(session_id),
                'phase': 'liveness',
                'status': 'completed',
                'message': 'Anti-spoofing berhasil! Hadap lurus ke kamera untuk verifikasi identitas.',
                'liveness_results': liveness_results,
                'landmark_results': landmark_results,
                'challenge_info': seq_state.get_challenge_info(),
                'next_phase': 'recognition',
                'processing_time': time.time() - start_time,
                'liveness_time': liveness_time,
                'landmark_time': landmark_time
            }
        else:
            # Still in phase 1
            challenge_info = seq_state.get_challenge_info()
            message = challenge_info['instruction'] if challenge_info else "Menunggu verifikasi anti-spoofing..."
            
            return {
                'session_id': str(session_id),
                'phase': 'liveness',
                'status': 'processing',
                'message': message,
                'liveness_results': liveness_results,
                'landmark_results': landmark_results,
                'challenge_info': challenge_info,
                'processing_time': time.time() - start_time,
                'liveness_time': liveness_time,
                'landmark_time': landmark_time
            }
    
    def _process_recognition_phase(self, image, seq_state, session_id, user_id, start_time):
        """Process Phase 2: Face Recognition using CNN and Database Matching"""
        
        recognition_start = time.time()
        
        try:
            # Extract face encodings for comparison
            import face_recognition
            
            # Convert image for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                return {
                    'session_id': str(session_id),
                    'phase': 'recognition',
                    'status': 'no_face',
                    'message': 'Wajah tidak terdeteksi. Hadap lurus ke kamera.',
                    'processing_time': time.time() - start_time
                }
            
            if len(face_locations) > 1:
                return {
                    'session_id': str(session_id),
                    'phase': 'recognition',
                    'status': 'multiple_faces',
                    'message': 'Terdeteksi lebih dari satu wajah. Pastikan hanya Anda di depan kamera.',
                    'processing_time': time.time() - start_time
                }
            
            # Extract face encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if len(face_encodings) == 0:
                return {
                    'session_id': str(session_id),
                    'phase': 'recognition',
                    'status': 'encoding_failed',
                    'message': 'Gagal mengekstrak fitur wajah. Coba lagi.',
                    'processing_time': time.time() - start_time
                }
            
            current_encoding = face_encodings[0]
            
            # Compare with database
            db_path = 'attendance.db'  # Use default path
            matched_user = None
            best_distance = float('inf')
            
            try:
                import sqlite3
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get all registered face data
                    cursor.execute('''
                        SELECT fd.user_id, fd.face_encoding, u.username, u.full_name 
                        FROM face_data fd 
                        JOIN users u ON fd.user_id = u.id 
                        WHERE u.is_active = 1
                    ''')
                    
                    registered_faces = cursor.fetchall()
                    
                    for user_id_db, encoding_json, username, full_name in registered_faces:
                        try:
                            stored_encoding = np.array(json.loads(encoding_json))
                            distance = face_recognition.face_distance([stored_encoding], current_encoding)[0]
                            
                            # Use threshold of 0.6 for matching
                            if distance < 0.6 and distance < best_distance:
                                best_distance = distance
                                matched_user = {
                                    'user_id': user_id_db,
                                    'username': username,
                                    'full_name': full_name,
                                    'distance': distance,
                                    'confidence': (1.0 - distance) * 100
                                }
                        except Exception as e:
                            print(f"Error processing stored encoding: {e}")
                            continue
            
            except Exception as e:
                print(f"Database error during recognition: {e}")
                return {
                    'session_id': str(session_id),
                    'phase': 'recognition',
                    'status': 'database_error',
                    'message': 'Error mengakses database.',
                    'processing_time': time.time() - start_time
                }
            
            recognition_time = time.time() - recognition_start
            
            # ========= PROCESS RECOGNITION RESULT =========
            if matched_user:
                # SUCCESSFUL RECOGNITION
                seq_state.recognition_result = matched_user
                seq_state.phase = 'complete'
                
                return {
                    'session_id': str(session_id),
                    'phase': 'recognition',
                    'status': 'success',
                    'message': f'Selamat datang, {matched_user["full_name"]}!',
                    'user': matched_user,
                    'processing_time': time.time() - start_time,
                    'recognition_time': recognition_time,
                    'attendance_ready': True
                }
            else:
                # UNKNOWN PERSON
                return {
                    'session_id': str(session_id),
                    'phase': 'recognition',
                    'status': 'unknown',
                    'message': 'Wajah tidak terdaftar dalam sistem. Silakan daftar terlebih dahulu.',
                    'processing_time': time.time() - start_time,
                    'recognition_time': recognition_time,
                    'attendance_ready': False
                }
                
        except Exception as e:
            print(f"Recognition phase error: {e}")
            return {
                'session_id': str(session_id),
                'phase': 'recognition',
                'status': 'error',
                'message': f'Error dalam pengenalan wajah: {str(e)[:50]}',
                'processing_time': time.time() - start_time
            }
    
    def _create_empty_sequential_result(self, start_time, phase):
        """Create empty result for sequential processing"""
        return {
            'session_id': 'unknown',
            'phase': phase,
            'status': 'no_face',
            'message': 'Wajah tidak terdeteksi',
            'processing_time': time.time() - start_time
        }

    def process_frame_optimized(self, image, session_id="default", use_cache=True):
        """
        Optimized frame processing dengan multi-method integration
        """
        start_time = time.time()
        
        try:
            # Frame skipping untuk performance
            self.frame_count += 1
            if self.frame_count % self.skip_frame_count != 0:
                # Return cached results for skipped frames
                if session_id in self.result_cache:
                    cached_result = self.result_cache[session_id].copy()
                    cached_result['frame_skipped'] = True
                    cached_result['processing_time'] = time.time() - start_time
                    return cached_result
            
            # Initialize models if needed
            self._init_models_lazy()
            
            if image is None or image.size == 0:
                return self._create_empty_result(start_time)
            
            # Check cache
            cache_key = None
            if use_cache:
                image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
                cache_key = self.get_cache_key(image_bytes)
                
                if (cache_key in self.result_cache and 
                    time.time() - self.cache_times.get(cache_key, 0) < self.cache_duration):
                    cached_result = self.result_cache[cache_key].copy()
                    cached_result['from_cache'] = True
                    cached_result['processing_time'] = time.time() - start_time
                    return cached_result
            
            # Resize image untuk consistent processing
            if image.shape[1] > 640 or image.shape[0] > 480:
                image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
            
            # =========================================
            # STEP 1: FACIAL LANDMARK DETECTION
            # =========================================
            landmark_start = time.time()
            landmark_results = {}
            
            if self.landmark_verifier:
                try:
                    # Set timeout for landmark detection
                    timeout_duration = 0.5  # 500ms timeout
                    
                    def landmark_detection_worker():
                        if hasattr(self.landmark_verifier, 'process_frame_optimized'):
                            return self.landmark_verifier.process_frame_optimized(image)
                        else:
                            return self.landmark_verifier.process_frame(image)
                    
                    # Use threading to prevent blocking
                    result_queue = queue.Queue()
                    
                    def run_detection():
                        try:
                            result = landmark_detection_worker()
                            result_queue.put(('success', result))
                        except Exception as e:
                            result_queue.put(('error', e))
                    
                    detection_thread = threading.Thread(target=run_detection)
                    detection_thread.daemon = True
                    detection_thread.start()
                    detection_thread.join(timeout=timeout_duration)
                    
                    if detection_thread.is_alive():
                        # Timeout occurred - use fallback
                        print(f"⚠️ Landmark detection timeout after {timeout_duration}s, using fallback")
                        landmark_results = self._create_empty_landmark_result()
                    else:
                        # Get result from queue
                        try:
                            status, result = result_queue.get_nowait()
                            if status == 'success':
                                landmark_results = result
                            else:
                                print(f"Landmark detection error: {result}")
                                landmark_results = self._create_empty_landmark_result()
                        except queue.Empty:
                            landmark_results = self._create_empty_landmark_result()
                            
                except Exception as e:
                    print(f"Landmark detection error: {e}")
                    landmark_results = self._create_empty_landmark_result()
            else:
                landmark_results = self._create_empty_landmark_result()
            
            landmark_time = time.time() - landmark_start
            
            # =========================================
            # STEP 2: CNN LIVENESS DETECTION
            # =========================================
            cnn_start = time.time()
            cnn_results = {}
            
            if self.cnn_predictor:
                try:
                    cnn_prediction = self.cnn_predictor.predict_optimized(image, use_cache=True)
                    cnn_results = {
                        'confidence': cnn_prediction.get('confidence', 0.0),
                        'is_live': cnn_prediction.get('is_live', False),
                        'probabilities': cnn_prediction.get('probabilities', {'fake': 1.0, 'live': 0.0})
                    }
                except Exception as e:
                    print(f"CNN prediction error: {e}")
                    cnn_results = {
                        'confidence': 0.0,
                        'is_live': False,
                        'probabilities': {'fake': 1.0, 'live': 0.0}
                    }
            else:
                # Fallback CNN simulation
                has_landmarks = landmark_results.get('landmarks_detected', False)
                cnn_confidence = 0.85 if has_landmarks else 0.1
                cnn_results = {
                    'confidence': cnn_confidence,
                    'is_live': cnn_confidence > 0.5,
                    'probabilities': {
                        'live': cnn_confidence,
                        'fake': 1.0 - cnn_confidence
                    }
                }
            
            cnn_time = time.time() - cnn_start
            
            # =========================================
            # STEP 3: ENHANCED SECURITY ASSESSMENT WITH STATE PERSISTENCE
            # =========================================
            fusion_start = time.time()
            
            # Get security state for this session
            security_state = self.get_security_state(session_id)
            
            # Update movement detection with grace period
            has_movement = landmark_results.get('head_movement', False)
            blink_detected = landmark_results.get('blink_count', 0) > 0
            mouth_movement = landmark_results.get('mouth_open', False)
            any_movement = has_movement or blink_detected or mouth_movement
            
            movement_verified = security_state.update_movement(any_movement, has_movement)
            
            # Update CNN with consistency check
            cnn_confidence = cnn_results.get('confidence', 0.0)
            cnn_is_live = cnn_results.get('is_live', False)
            cnn_verified = security_state.update_cnn(cnn_confidence, cnn_is_live)
            
            # Update challenge system for landmark detection
            if not security_state.current_challenge:
                security_state.generate_new_challenge()
            
            landmark_verified = security_state.update_challenge(landmark_results)
            
            # Get overall security status
            security_status = security_state.get_security_status()
            
            # Enhanced method details with persistent state
            method_details = {
                'movement': {
                    'verified': movement_verified,
                    'score': 1.0 if movement_verified else 0.0,
                    'status': 'VERIFIED' if movement_verified else 'CHECKING',
                    'description': 'Movement detected with 3s grace period'
                },
                'cnn': {
                    'verified': cnn_verified,
                    'score': cnn_confidence,
                    'status': 'VERIFIED' if cnn_verified else 'CHECKING',
                    'description': f'CNN confidence: {cnn_confidence:.2f}'
                },
                'landmark': {
                    'verified': landmark_verified,
                    'score': 1.0 if landmark_verified else security_state.challenge_progress,
                    'status': 'VERIFIED' if landmark_verified else 'CHALLENGE_ACTIVE',
                    'description': 'Challenge-based verification'
                }
            }
            
            methods_passed = security_status['methods_passed']
            security_passed = security_status['security_passed']
            
            # Security level classification with persistent state
            if security_passed and methods_passed == 3:
                security_level = "SECURE"
                security_color = "success"
            elif security_passed:
                security_level = "GOOD"
                security_color = "primary"
            elif methods_passed >= 1:
                security_level = "WARNING"
                security_color = "warning"
            else:
                security_level = "DANGER" 
                security_color = "danger"
            
            fusion_time = time.time() - fusion_start
            
            # =========================================
            # STEP 4: RESULT COMPILATION WITH ENHANCED STATE
            # =========================================
            total_processing_time = time.time() - start_time
            self.processing_times.append(total_processing_time)
            
            # Calculate overall confidence from verified methods
            verified_methods = [method_details[key]['verified'] for key in method_details.keys()]
            overall_confidence = sum(verified_methods) / len(verified_methods)
            
            # Estimate FPS
            if len(self.processing_times) > 0:
                avg_time = np.mean(list(self.processing_times))
                estimated_fps = 1.0 / avg_time if avg_time > 0 else 0
            else:
                estimated_fps = 0
            
            # Prepare landmark points for frontend
            landmark_points = []
            landmarks = landmark_results.get('landmark_coordinates', [])
            if landmarks:
                # Convert to frontend format with colors
                for i, landmark in enumerate(landmarks):
                    if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                        # Determine color based on landmark type
                        if i < 20:
                            color = '#FF0000'  # Red for eye area
                        elif i < 40:
                            color = '#00FF00'  # Green for nose area
                        elif i < 60:
                            color = '#0000FF'  # Blue for mouth area
                        else:
                            color = '#FFFFFF'  # White for other points
                        
                        landmark_points.append({
                            'x': float(landmark[0]),
                            'y': float(landmark[1]),
                            'color': color,
                            'index': i
                        })
            
            # Enhanced result with persistent state and challenge info
            result = {
                'session_id': str(session_id),
                'frame_processed': True,
                'timestamp': float(time.time()),
                
                # Landmark Detection Results
                'landmarks_detected': landmark_results.get('landmarks_detected', False),
                'landmark_count': len(landmark_points),
                'landmarks': landmark_points,
                'blink_count': landmark_results.get('blink_count', 0),
                'head_movement': landmark_results.get('head_movement', False),
                'mouth_open': landmark_results.get('mouth_open', False),
                'ear_left': landmark_results.get('ear_left', 0.0),
                'ear_right': landmark_results.get('ear_right', 0.0),
                'mar': landmark_results.get('mar', 0.0),
                
                # CNN Results
                'cnn_confidence': float(cnn_confidence),
                'cnn_probabilities': cnn_results.get('probabilities', {}),
                
                # Liveness Assessment (kept for compatibility)
                'liveness_score': float(method_details['landmark']['score'] * 100),
                'liveness_raw_score': landmark_results.get('liveness_score', 0.0),
                'is_live': security_passed,
                'liveness_status': security_level,
                
                # Enhanced Multi-Method Security Assessment
                'security_level': security_level,
                'security_color': security_color,
                'security_passed': security_passed,
                'overall_confidence': float(overall_confidence),
                'methods_passed': methods_passed,
                'method_details': method_details,
                
                # Challenge System Information
                'challenge_info': security_status.get('challenge_info'),
                'movement_verified': security_status['movement_verified'],
                'cnn_verified': security_status['cnn_verified'],
                'landmark_verified': security_status['landmark_verified'],
                
                # Performance Metrics
                'processing_time': total_processing_time,
                'landmark_time': landmark_time,
                'cnn_time': cnn_time,
                'fusion_time': fusion_time,
                'estimated_fps': estimated_fps,
                'frame_skipped': False,
                'from_cache': False,
                
                # Enhanced Message with Challenge Instructions
                'message': self._generate_user_message(security_status, security_level, methods_passed)
            }
            
            # Cache result
            if cache_key:
                # Manage cache size
                if len(self.result_cache) >= 50:  # Limit cache size
                    oldest_key = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
                    del self.result_cache[oldest_key]
                    del self.cache_times[oldest_key]
                
                self.result_cache[cache_key] = result.copy()
                self.cache_times[cache_key] = time.time()
            
            # Also cache by session_id for frame skipping
            self.result_cache[session_id] = result.copy()
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Frame processing error: {e}")
            return {
                'session_id': str(session_id),
                'frame_processed': False,
                'landmarks_detected': False,
                'landmark_count': 0,
                'landmarks': [],
                'security_level': 'ERROR',
                'security_color': 'danger',
                'processing_time': processing_time,
                'error': str(e),
                'message': f'Processing error: {str(e)[:50]}'
            }
    
    def _create_empty_landmark_result(self):
        """Create empty landmark result"""
        return {
            'landmarks_detected': False,
            'landmark_coordinates': [],
            'confidence': 0.0,
            'blink_count': 0,
            'mouth_open': False,
            'head_movement': False,
            'ear_left': 0.0,
            'ear_right': 0.0,
            'mar': 0.0,
            'liveness_score': 0.0,
            'is_live': False,
            'liveness_status': 'NO_FACE'
        }
    
    def _create_empty_result(self, start_time):
        """Create empty result for failed processing"""
        return {
            'frame_processed': False,
            'landmarks_detected': False,
            'landmark_count': 0,
            'landmarks': [],
            'security_level': 'NO_FACE',
            'security_color': 'secondary',
            'processing_time': time.time() - start_time,
            'message': 'No face detected'
        }
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {'avg_processing_time': 0, 'estimated_fps': 0}
        
        times = list(self.processing_times)
        return {
            'avg_processing_time': np.mean(times),
            'min_processing_time': np.min(times),
            'max_processing_time': np.max(times),
            'estimated_fps': 1.0 / np.mean(times),
            'cache_size': len(self.result_cache),
            'frames_processed': self.frame_count
        }
    
    def cleanup_cache(self):
        """Cleanup old cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, cache_time in self.cache_times.items()
            if current_time - cache_time > self.cache_duration * 10
        ]
        
        for key in expired_keys:
            self.result_cache.pop(key, None)
            self.cache_times.pop(key, None)
        
        # Force garbage collection every 100 cleanups
        if len(expired_keys) > 0 and self.frame_count % 100 == 0:
            gc.collect()


# Global frame processor instance
frame_processor = OptimizedFrameProcessor()


def create_optimized_app(config: SystemConfig = None):
    """
    Create optimized Flask app with SocketIO
    """
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    template_dir = project_root / "src" / "web" / "templates"
    static_dir = project_root / "src" / "web" / "static"
    
    # Create Flask app
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir)
    )
    
    # Configure app
    if config:
        app.config['SECRET_KEY'] = config.web.secret_key
        app.config['DATABASE_URL'] = config.web.database_url
        app.config['DEBUG'] = config.web.debug
    else:
        app.config['SECRET_KEY'] = secrets.token_hex(16)
        app.config['DATABASE_URL'] = 'sqlite:///attendance.db'
        app.config['DEBUG'] = False
    
    # Setup logger
    logger = get_web_logger()
    
    # Initialize SocketIO with optimized settings
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*",
        async_mode='threading',  # Use threading for better performance
        ping_timeout=60,
        ping_interval=25
    )
    
    # Initialize database
    init_db(app)
    
    # Register routes
    register_optimized_routes(app, logger)
    register_optimized_socketio_events(socketio, logger)
    register_error_handlers(app, logger)
    
    # Performance monitoring endpoint
    @app.route('/api/performance')
    def get_performance():
        stats = frame_processor.get_performance_stats()
        return jsonify(stats)
    
    # Cache cleanup endpoint
    @app.route('/api/cleanup-cache')
    def cleanup_cache():
        frame_processor.cleanup_cache()
        return jsonify({'status': 'Cache cleaned'})
    
    logger.info("Optimized Flask application with SocketIO created successfully")
    return app, socketio


def register_optimized_socketio_events(socketio, logger):
    """
    Register optimized SocketIO events dengan frame processing pipeline
    """
    
    @socketio.on('connect')
    def handle_connect():
        logger.info("Client connected to optimized landmark detection")
        emit('status', {'message': 'Connected to optimized server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info("Client disconnected from optimized landmark detection")
    
    @socketio.on('process_frame')
    def handle_optimized_frame_processing(data):
        """
        Sequential frame processing dengan 2-phase approach
        """
        try:
            # Get session info
            session_id = data.get('session_id', 'default')
            user_id = session.get('user_id', 'anonymous')
            processing_mode = data.get('mode', 'sequential')  # 'sequential' or 'parallel'
            
            # Decode image
            if 'image' not in data:
                emit('error', {'message': 'No image data'})
                return
            
            image_b64 = data['image']
            try:
                image_data = base64.b64decode(image_b64.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                emit('error', {'message': f'Image decode error: {str(e)}'})
                return
            
            if image is None:
                emit('error', {'message': 'Invalid image data'})
                return
            
            # Choose processing mode
            if processing_mode == 'sequential':
                # NEW: Sequential processing (2-phase)
                result = frame_processor.process_frame_sequential(
                    image, 
                    session_id=session_id,
                    user_id=user_id
                )
                
                # Add session info
                result['session_id'] = session_id
                result['user_id'] = user_id
                result['mode'] = 'sequential'
                
                # Convert result to JSON-serializable format
                serializable_result = convert_to_serializable(result)
                
                # Emit result with phase-specific event name
                if result['phase'] == 'liveness':
                    emit('liveness_result', serializable_result)
                elif result['phase'] == 'recognition':
                    emit('recognition_result', serializable_result)
                else:
                    emit('detection_result', serializable_result)
                    
            else:
                # LEGACY: Parallel processing (all methods together)
                result = frame_processor.process_frame_optimized(
                    image, 
                    session_id=session_id, 
                    use_cache=True
                )
                
                # Add session info
                result['session_id'] = session_id
                result['user_id'] = user_id
                result['mode'] = 'parallel'
                
                # Convert result to JSON-serializable format
                serializable_result = convert_to_serializable(result)
                
                # Emit result
                emit('landmark_result', serializable_result)
            
            # Emit performance stats periodically
            if frame_processor.frame_count % 30 == 0:  # Every 30 frames
                stats = frame_processor.get_performance_stats()
                serializable_stats = convert_to_serializable(stats)
                emit('performance_stats', serializable_stats)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            emit('error', {'message': f'Processing error: {str(e)}'})

    @socketio.on('reset_detection')
    def handle_reset_detection(data):
        """Reset detection state for new session"""
        try:
            session_id = data.get('session_id', 'default')
            
            # Clear sequential state
            if session_id in frame_processor.sequential_states:
                del frame_processor.sequential_states[session_id]
            
            # Clear security state
            if session_id in frame_processor.security_states:
                del frame_processor.security_states[session_id]
            
            emit('detection_reset', {
                'status': 'success',
                'message': 'Detection state reset',
                'session_id': session_id
            })
            
        except Exception as e:
            emit('detection_reset', {
                'status': 'error',
                'message': f'Reset error: {str(e)}',
                'session_id': session_id
            })

    @socketio.on('capture_face')
    def handle_capture_face(data):
        """Handle face capture event for registration with comprehensive debugging"""
        try:
            print("=== CAPTURE FACE DEBUG ===")
            
            # 1. Check basic data
            if not data:
                print("❌ No data received")
                emit('face_capture_result', {
                    'status': 'error',
                    'message': 'No data received'
                })
                return
            
            print(f"📥 Data keys: {list(data.keys()) if hasattr(data, 'keys') else 'Not a dict'}")
            
            user_id = data.get('user_id')
            position = data.get('position')  # 'front', 'left', 'right'
            image_data = data.get('image')
            
            print(f"👤 User ID: {user_id}")
            print(f"📍 Position: {position}")
            print(f"🖼️ Image data exists: {bool(image_data)}")
            
            if image_data:
                print(f"🖼️ Image data length: {len(image_data)}")
                print(f"🖼️ Image data prefix: {image_data[:50] if len(image_data) > 50 else image_data}")
            
            # 2. Check session (alternative way to get user_id)
            session_user_id = session.get('user_id') if 'session' in globals() else None
            print(f"🔐 Session user ID: {session_user_id}")
            
            # Use session user_id if not provided in data
            if not user_id and session_user_id:
                user_id = session_user_id
                print(f"✅ Using session user_id: {user_id}")
            
            # Validate required data
            if not all([user_id, position, image_data]):
                missing = []
                if not user_id: missing.append('user_id')
                if not position: missing.append('position')
                if not image_data: missing.append('image_data')
                
                error_msg = f'Data tidak lengkap: {", ".join(missing)}'
                print(f"❌ {error_msg}")
                emit('face_capture_result', {
                    'status': 'error', 
                    'message': error_msg
                })
                return
            
            # 3. Process base64 image
            print("🔄 Processing image data...")
            try:
                # Remove data URL prefix if present
                original_length = len(image_data)
                if image_data.startswith('data:'):
                    image_data = image_data.split(',')[1]
                    print(f"🔧 Removed data URL prefix, length: {original_length} -> {len(image_data)}")
                
                # Decode base64
                import base64
                image_bytes = base64.b64decode(image_data)
                print(f"✅ Decoded image size: {len(image_bytes)} bytes")
                
                # Convert to numpy array using PIL
                import numpy as np
                from PIL import Image
                import io
                
                image_pil = Image.open(io.BytesIO(image_bytes))
                print(f"📐 PIL Image size: {image_pil.size}, mode: {image_pil.mode}")
                
                image_array = np.array(image_pil)
                print(f"📐 Numpy array shape: {image_array.shape}")
                
                # Convert RGB to BGR for face_recognition (if needed)
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    # PIL loads as RGB, face_recognition expects RGB, so no conversion needed
                    rgb_image = image_array
                    print("✅ Image ready for face_recognition (RGB format)")
                else:
                    print(f"⚠️ Unexpected image shape: {image_array.shape}")
                    
            except Exception as e:
                error_msg = f"Failed to process image data: {str(e)}"
                print(f"❌ Image processing error: {error_msg}")
                import traceback
                traceback.print_exc()
                emit('face_capture_result', {
                    'status': 'error',
                    'message': 'Gagal memproses data gambar'
                })
                return
            
            # 4. Save image to local storage (skip face recognition for now)
            print("� Saving image to local storage...")
            try:
                # Create faces directory if it doesn't exist
                faces_dir = os.path.join('static', 'faces')
                os.makedirs(faces_dir, exist_ok=True)
                print(f"� Faces directory: {faces_dir}")
                
                # Generate filename with timestamp
                import time
                timestamp = int(time.time())
                image_filename = f"face_{user_id}_{position}_{timestamp}.jpg"
                image_path = os.path.join(faces_dir, image_filename)
                
                # Save image file
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                print(f"✅ Image saved to: {image_path}")
                print(f"📏 File size: {len(image_bytes)} bytes")
                
                # Optional: Verify image was saved correctly
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    print(f"✅ File verification: {file_size} bytes on disk")
                else:
                    raise FileNotFoundError("Image file not found after saving")
                
            except Exception as e:
                error_msg = f"Failed to save image: {str(e)}"
                print(f"❌ Image saving error: {error_msg}")
                import traceback
                traceback.print_exc()
                emit('face_capture_result', {
                    'status': 'error',
                    'message': 'Gagal menyimpan gambar ke disk'
                })
                return
            
            # 5. Save image path to database (instead of face encoding)
            print("💾 Saving image path to database...")
            try:
                # Get database path
                db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
                print(f"🗄️ Database path: {db_path}")
                
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check if user exists
                    cursor.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
                    user_record = cursor.fetchone()
                    
                    if not user_record:
                        print(f"❌ User {user_id} not found in database")
                        emit('face_capture_result', {
                            'status': 'error',
                            'message': 'User tidak ditemukan dalam sistem'
                        })
                        return
                    
                    print(f"✅ User found: {user_record[1]} (ID: {user_record[0]})")
                    
                    # Check if position already exists for this user
                    cursor.execute('SELECT id FROM face_data WHERE user_id = ? AND face_position = ?', (user_id, position))
                    existing = cursor.fetchone()
                    
                    if existing:
                        print(f"🔄 Updating existing face data for position {position}")
                        cursor.execute('''UPDATE face_data 
                                         SET image_path = ?, updated_at = CURRENT_TIMESTAMP 
                                         WHERE user_id = ? AND face_position = ?''',
                                      (image_path, user_id, position))
                    else:
                        print(f"➕ Inserting new face data for position {position}")
                        # Add image_path column if it doesn't exist
                        try:
                            cursor.execute('ALTER TABLE face_data ADD COLUMN image_path TEXT')
                            print("✅ Added image_path column to face_data table")
                        except sqlite3.OperationalError:
                            # Column already exists
                            pass
                        
                        cursor.execute('''INSERT INTO face_data (user_id, face_position, image_path) 
                                         VALUES (?, ?, ?)''',
                                      (user_id, position, image_path))
                    
                    conn.commit()
                    print(f"✅ Successfully saved image path for position: {position}")
                
                # Success response
                success_msg = f'Foto {position} berhasil disimpan ke {image_filename}'
                print(f"🎉 {success_msg}")
                emit('face_capture_result', {
                    'status': 'success', 
                    'message': success_msg,
                    'position': position,
                    'image_path': image_path,
                    'image_filename': image_filename
                })
                
            except Exception as e:
                error_msg = f"Database error: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                emit('face_capture_result', {
                    'status': 'error',
                    'message': 'Gagal menyimpan data wajah ke database'
                })
                return
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"💥 {error_msg}")
            import traceback
            traceback.print_exc()
            logger.error(f"Error in capture_face: {e}")
            emit('face_capture_result', {
                'status': 'error', 
                'message': 'Terjadi kesalahan saat memproses gambar'
            })
        
        print("=== END CAPTURE FACE DEBUG ===\n")


def register_optimized_routes(app, logger):
    """
    Register optimized routes
    """
    
    @app.route('/')
    def index():
        return render_template('index_clean.html')
    
    @app.route('/favicon.ico')
    def favicon():
        return '', 204  # No Content response for favicon
    
    @app.route('/face-detection')
    def face_detection():
        return render_template('face_detection_optimized.html')
    
    @app.route('/dashboard')
    def dashboard():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        # Get user data from session
        user_data = {
            'id': session.get('user_id'),
            'username': session.get('username'),
            'full_name': session.get('full_name'),
            'role': session.get('role')
        }
        
        # Check face registration status
        user_has_face_data = False
        if user_data['id'] != 0:  # Skip for admin
            try:
                db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT DISTINCT face_position FROM face_data WHERE user_id = ?', (user_data['id'],))
                    positions = [row[0] for row in cursor.fetchall()]
                    user_has_face_data = len(positions) >= 3 and all(pos in positions for pos in ['front', 'left', 'right'])
            except Exception as e:
                print(f"Error checking face data: {e}")
        else:
            user_has_face_data = True  # Admin doesn't need face registration
        
        return render_template('dashboard.html', user=user_data, user_has_face_data=user_has_face_data)
    
    @app.route('/attendance')
    def attendance():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        # Get user data from session
        user_data = {
            'id': session.get('user_id'),
            'username': session.get('username'),
            'full_name': session.get('full_name'),
            'role': session.get('role')
        }
        
        return render_template('attendance.html', user=user_data)
    
    @app.route('/attendance-sequential')
    def attendance_sequential():
        """Sequential attendance with 2-phase detection"""
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        # Get user data from session
        user_data = {
            'id': session.get('user_id'),
            'username': session.get('username'),
            'full_name': session.get('full_name'),
            'role': session.get('role')
        }
        
        return render_template('attendance_sequential.html', user=user_data)
    
    @app.route('/api/record-attendance', methods=['POST'])
    def record_attendance():
        """Record attendance after successful recognition"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not logged in'})
        
        try:
            data = request.get_json()
            recognized_user = data.get('user')
            detection_data = data.get('detection_data', {})
            
            if not recognized_user:
                return jsonify({'success': False, 'message': 'User data required'})
            
            # Record attendance in database
            db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO attendance_records 
                    (user_id, check_type, confidence_score, liveness_score, security_level, methods_passed, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    recognized_user['user_id'],
                    'in',  # Default to check-in
                    recognized_user.get('confidence', 0.0) / 100.0,  # Convert to 0-1 range
                    1.0,  # Liveness passed in sequential mode
                    'SEQUENTIAL_SUCCESS',
                    2,  # Both phases passed
                    detection_data.get('processing_time', 0.0)
                ))
                
                conn.commit()
                attendance_id = cursor.lastrowid
            
            return jsonify({
                'success': True,
                'message': f'Attendance recorded for {recognized_user["full_name"]}',
                'attendance_id': attendance_id,
                'user': recognized_user
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error recording attendance: {str(e)}'
            })
    
    @app.route('/api/check-face-registered')
    def check_face_registered():
        if 'user_id' not in session:
            return jsonify({'registered': False})
        
        user_id = session['user_id']
        
        # Skip check for admin
        if user_id == 0:
            return jsonify({'registered': True, 'is_admin': True})
        
        try:
            db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM face_data WHERE user_id = ?', (user_id,))
                count = cursor.fetchone()[0]
                
                # Check if user has all 3 positions registered
                cursor.execute('SELECT DISTINCT face_position FROM face_data WHERE user_id = ?', (user_id,))
                positions = [row[0] for row in cursor.fetchall()]
                
                has_all_positions = len(positions) >= 3 and all(pos in positions for pos in ['front', 'left', 'right'])
                
                return jsonify({
                    'registered': has_all_positions,
                    'positions_count': len(positions),
                    'positions': positions,
                    'is_admin': False
                })
        except Exception as e:
            print(f"Error checking face registration: {e}")
            return jsonify({'registered': False, 'error': str(e)})
    
    @app.route('/register-face')
    def register_face():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        user_id = session['user_id']
        
        # Skip for admin
        if user_id == 0:
            flash('Admin tidak perlu registrasi wajah.', 'info')
            return redirect(url_for('dashboard'))
        
        # Check if already registered
        try:
            db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT face_position FROM face_data WHERE user_id = ?', (user_id,))
                positions = [row[0] for row in cursor.fetchall()]
                
                if len(positions) >= 3:
                    flash('Wajah Anda sudah terdaftar!', 'info')
                    return redirect(url_for('dashboard'))
        except Exception as e:
            print(f"Error checking registration: {e}")
        
        # Get user data for template
        user_data = {
            'id': session.get('user_id'),
            'username': session.get('username'),
            'full_name': session.get('full_name'),
            'role': session.get('role')
        }
        
        return render_template('register_face.html', user=user_data)
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            # Special handling for admin
            if username == 'admin' and password == 'admin':
                session['user_id'] = 0  # Special admin ID
                session['username'] = 'admin'
                session['full_name'] = 'Administrator'
                session['role'] = 'admin'
                flash('Welcome Administrator!', 'success')
                return redirect(url_for('dashboard'))
            
            # Database authentication for regular users
            try:
                db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT id, username, full_name, password_hash, role FROM users WHERE username = ? AND is_active = 1', (username,))
                    user = cursor.fetchone()
                    
                    if user and check_password_hash(user[3], password):
                        # Login successful
                        session['user_id'] = user[0]
                        session['username'] = user[1]
                        session['full_name'] = user[2]
                        session['role'] = user[4]
                        flash(f'Welcome back, {user[2]}!', 'success')
                        return redirect(url_for('dashboard'))
                    else:
                        flash('Invalid username or password!', 'error')
            except Exception as e:
                flash(f'Login error: {str(e)}', 'error')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """User registration (placeholder for optimized app)"""
        if request.method == 'POST':
            flash('Registration feature coming soon!', 'info')
            return redirect(url_for('login'))
        return render_template('register.html')
    
    @app.route('/logout')
    def logout():
        session.clear()
        flash('Logged out successfully!', 'success')
        return redirect(url_for('index'))


def register_error_handlers(app, logger):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        try:
            return render_template('404.html'), 404
        except:
            # Fallback if template doesn't exist
            return "404 - Page not found", 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        try:
            return render_template('404.html', error='Internal server error'), 500
        except:
            # Fallback if template doesn't exist
            return "500 - Internal server error", 500


def init_db(app: Flask) -> None:
    """Initialize database"""
    with app.app_context():
        db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Attendance records
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_type TEXT DEFAULT 'in',
                    confidence_score REAL,
                    liveness_score REAL,
                    security_level TEXT,
                    methods_passed INTEGER,
                    processing_time REAL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Face data table for face registration
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    face_position TEXT NOT NULL,
                    face_encoding TEXT NOT NULL,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create default accounts if they don't exist
            try:
                # Check if admin account exists
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
                if cursor.fetchone()[0] == 0:
                    # Create admin account
                    admin_password_hash = generate_password_hash('admin')
                    cursor.execute('''
                        INSERT INTO users (username, full_name, email, password_hash, role)
                        VALUES (?, ?, ?, ?, ?)
                    ''', ('admin', 'Administrator', 'admin@example.com', admin_password_hash, 'admin'))
                    print("✅ Created admin account: username=admin, password=admin")
                
                # Check if user account exists
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('user',))
                if cursor.fetchone()[0] == 0:
                    # Create regular user account
                    user_password_hash = generate_password_hash('user123')
                    cursor.execute('''
                        INSERT INTO users (username, full_name, email, password_hash, role)
                        VALUES (?, ?, ?, ?, ?)
                    ''', ('user', 'Regular User', 'user@example.com', user_password_hash, 'user'))
                    print("✅ Created user account: username=user, password=user123")
                
                # Check if demo account exists
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('demo',))
                if cursor.fetchone()[0] == 0:
                    # Create demo account with simple password
                    demo_password_hash = generate_password_hash('demo')
                    cursor.execute('''
                        INSERT INTO users (username, full_name, email, password_hash, role)
                        VALUES (?, ?, ?, ?, ?)
                    ''', ('demo', 'Demo User', 'demo@example.com', demo_password_hash, 'user'))
                    print("✅ Created demo account: username=demo, password=demo")
                    
            except Exception as e:
                print(f"⚠️ Error creating default accounts: {e}")
            
            conn.commit()


# Create global app instance for imports
try:
    config = SystemConfig()
except:
    config = None

app, socketio = create_optimized_app(config)


if __name__ == "__main__":
    # Test the optimized application
    print("🚀 TESTING OPTIMIZED WEB APPLICATION")
    print("=" * 50)
    
    app, socketio = create_optimized_app()
    
    # Test frame processor
    print("Testing frame processor...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for i in range(3):
        result = frame_processor.process_frame_optimized(dummy_image, f"test_{i}")
    
    # Performance test
    times = []
    for i in range(10):
        start = time.time()
        result = frame_processor.process_frame_optimized(dummy_image, f"test_{i}")
        times.append(time.time() - start)
        print(f"Frame {i+1}: {times[-1]*1000:.1f}ms, Security: {result.get('security_level', 'N/A')}")
    
    avg_time = np.mean(times)
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"Average processing time: {avg_time*1000:.1f}ms")
    print(f"Estimated FPS: {1.0/avg_time:.1f}")
    
    stats = frame_processor.get_performance_stats()
    print(f"Detailed stats: {stats}")
    
    print("\n🌐 Starting optimized web server...")
    print("Visit: http://localhost:5000/face-detection")
    
    # Run the optimized app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

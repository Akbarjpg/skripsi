"""
Step 4 Web Application - CNN Face Recognition Integration
Integrates anti-spoofing detection with CNN face recognition
"""

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import uuid
import threading
import json
from typing import Dict, Optional, List
import os
from datetime import datetime

from ..models.face_recognition_cnn import FaceRecognitionSystem
from ..database.attendance_db import AttendanceDatabase
from ..challenge.challenge_response import ChallengeResponseSystem
from ..detection.landmark_detection import LivenessVerifier
from ..utils.logger import get_logger


class Step4AttendanceApp:
    """
    Step 4 application class integrating anti-spoofing with CNN face recognition
    """
    
    def __init__(self):
        """Initialize the Step 4 attendance application"""
        self.logger = get_logger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'step4_face_attendance_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.database = AttendanceDatabase()
        self.face_recognition = FaceRecognitionSystem(
            model_path="models/face_recognition_model.pth",
            similarity_threshold=0.85
        )
        self.challenge_system = ChallengeResponseSystem()
        self.liveness_verifier = LivenessVerifier()
        
        # Application state
        self.camera = None
        self.is_streaming = False
        self.current_sessions = {}  # session_id -> session_data
        
        # Load face embeddings from database
        self._load_embeddings()
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        self.logger.info("Step 4 Attendance application initialized")
    
    def _load_embeddings(self):
        """Load face embeddings from database into recognition system"""
        try:
            embeddings = self.database.get_all_embeddings()
            self.face_recognition.load_embeddings_from_database(embeddings)
            self.logger.info(f"Loaded {len(embeddings)} face embeddings")
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
    
    def _setup_routes(self):
        """Setup Flask routes for Step 4"""
        
        @self.app.route('/')
        def index():
            """Main Step 4 attendance page"""
            return render_template('attendance_step4.html')
        
        @self.app.route('/register')
        def register_page():
            """Step 4 user registration page"""
            return render_template('registration_step4.html')
        
        @self.app.route('/admin')
        def admin_page():
            """Step 4 admin dashboard"""
            stats = self.database.get_database_stats()
            users = self.database.get_all_users()
            return render_template('admin_step4.html', stats=stats, users=users)
        
        @self.app.route('/api/users', methods=['GET'])
        def get_users():
            """Get all registered users"""
            try:
                users = self.database.get_all_users()
                return jsonify({'success': True, 'users': users})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/user/<user_id>', methods=['GET'])
        def get_user(user_id):
            """Get specific user information"""
            try:
                user_info = self.database.get_user_info(user_id)
                if user_info:
                    return jsonify({'success': True, 'user': user_info})
                else:
                    return jsonify({'success': False, 'error': 'User not found'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/register_user', methods=['POST'])
        def register_user_api():
            """Register a new user via API"""
            try:
                data = request.json
                success = self.database.register_user(
                    user_id=data['user_id'],
                    name=data['name'],
                    email=data.get('email', ''),
                    role=data.get('role', 'employee'),
                    department=data.get('department', '')
                )
                
                if success:
                    return jsonify({'success': True, 'message': 'User registered successfully'})
                else:
                    return jsonify({'success': False, 'error': 'User already exists'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/attendance/today', methods=['GET'])
        def get_todays_attendance():
            """Get today's attendance records"""
            try:
                records = self.database.get_daily_attendance()
                return jsonify({'success': True, 'records': records})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/attendance/user/<user_id>')
        def get_user_attendance(user_id):
            """Get attendance history for specific user"""
            try:
                days = request.args.get('days', 30, type=int)
                records = self.database.get_user_attendance(user_id, days)
                return jsonify({'success': True, 'records': records})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get comprehensive system statistics"""
            try:
                stats = self.database.get_database_stats()
                face_recognition_info = self.face_recognition.get_system_info()
                
                combined_stats = {
                    'database': stats,
                    'face_recognition': face_recognition_info,
                    'system_status': {
                        'camera_available': self.camera is not None,
                        'streaming_active': self.is_streaming,
                        'active_sessions': len(self.current_sessions),
                        'recognition_ready': len(self.face_recognition.embedding_cache) > 0
                    }
                }
                
                return jsonify({'success': True, 'stats': combined_stats})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(self._generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers for real-time communication"""
        
        @self.socketio.on('start_attendance')
        def handle_start_attendance(data):
            """Start Step 4 attendance verification process"""
            try:
                session_id = str(uuid.uuid4())
                
                # Initialize session
                session_data = {
                    'session_id': session_id,
                    'start_time': time.time(),
                    'phase': 'antispoofing',  # antispoofing -> recognition -> complete
                    'antispoofing_result': None,
                    'recognition_result': None,
                    'client_id': request.sid
                }
                
                self.current_sessions[session_id] = session_data
                
                # Start camera if not already active
                if not self.is_streaming:
                    self._start_camera()
                
                emit('session_started', {
                    'session_id': session_id,
                    'message': 'Step 4: Starting anti-spoofing + face recognition',
                    'phase': 'antispoofing'
                })
                
                # Start processing in background
                threading.Thread(
                    target=self._process_step4_attendance,
                    args=(session_id,),
                    daemon=True
                ).start()
                
            except Exception as e:
                self.logger.error(f"Failed to start Step 4 attendance: {e}")
                emit('error', {'message': f'Failed to start attendance: {str(e)}'})
        
        @self.socketio.on('start_registration')
        def handle_start_registration(data):
            """Start Step 4 user registration with face capture"""
            try:
                user_id = data.get('user_id')
                name = data.get('name')
                email = data.get('email', '')
                role = data.get('role', 'employee')
                department = data.get('department', '')
                
                if not user_id or not name:
                    emit('error', {'message': 'User ID and name are required'})
                    return
                
                # Register user in database first
                success = self.database.register_user(user_id, name, email, role, department)
                
                if not success:
                    emit('error', {'message': 'User already exists or registration failed'})
                    return
                
                session_id = str(uuid.uuid4())
                
                # Initialize registration session
                session_data = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'user_name': name,
                    'start_time': time.time(),
                    'phase': 'registration',
                    'captured_images': [],
                    'target_images': 12,  # Capture more images for better accuracy
                    'client_id': request.sid
                }
                
                self.current_sessions[session_id] = session_data
                
                # Start camera if not already active
                if not self.is_streaming:
                    self._start_camera()
                
                emit('registration_started', {
                    'session_id': session_id,
                    'user_id': user_id,
                    'user_name': name,
                    'message': 'Step 4: Starting face registration with CNN embedding',
                    'target_images': 12
                })
                
                # Start registration processing
                threading.Thread(
                    target=self._process_step4_registration,
                    args=(session_id,),
                    daemon=True
                ).start()
                
            except Exception as e:
                self.logger.error(f"Failed to start Step 4 registration: {e}")
                emit('error', {'message': f'Failed to start registration: {str(e)}'})
        
        @self.socketio.on('cancel_session')
        def handle_cancel_session(data):
            """Cancel current session"""
            try:
                session_id = data.get('session_id')
                if session_id in self.current_sessions:
                    del self.current_sessions[session_id]
                    emit('session_cancelled', {'message': 'Session cancelled'})
                
            except Exception as e:
                self.logger.error(f"Failed to cancel session: {e}")
                emit('error', {'message': f'Failed to cancel session: {str(e)}'})
        
        @self.socketio.on('get_system_status')
        def handle_get_status():
            """Get current system status"""
            try:
                status = {
                    'camera_active': self.is_streaming,
                    'registered_users': len(self.face_recognition.embedding_cache),
                    'active_sessions': len(self.current_sessions),
                    'recognition_ready': len(self.face_recognition.embedding_cache) > 0
                }
                emit('system_status', status)
                
            except Exception as e:
                emit('error', {'message': f'Failed to get status: {str(e)}'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnect - cleanup sessions"""
            sessions_to_remove = []
            for session_id, session_data in self.current_sessions.items():
                if session_data.get('client_id') == request.sid:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.current_sessions[session_id]
                self.logger.info(f"Cleaned up session {session_id} on disconnect")
    
    def _start_camera(self):
        """Start camera capture with optimal settings"""
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
                # Set optimal camera settings
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_streaming = True
            self.logger.info("Camera started for Step 4")
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            raise
    
    def _stop_camera(self):
        """Stop camera capture"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            self.is_streaming = False
            self.logger.info("Camera stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop camera: {e}")
    
    def _generate_frames(self):
        """Generate frames for video streaming"""
        while self.is_streaming and self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.logger.error(f"Frame generation error: {e}")
                break
    
    def _process_step4_attendance(self, session_id: str):
        """Process Step 4 attendance: Anti-spoofing → CNN Recognition"""
        try:
            session_data = self.current_sessions.get(session_id)
            if not session_data:
                return
            
            client_id = session_data['client_id']
            
            # Phase 1: Anti-spoofing Detection
            self.socketio.emit('phase_update', {
                'phase': 'antispoofing',
                'message': 'Phase 1: Verifying real person with anti-spoofing...',
                'progress': 0
            }, room=client_id)
            
            antispoofing_result = self._run_antispoofing_verification(session_id)
            session_data['antispoofing_result'] = antispoofing_result
            
            # Log antispoofing attempt
            self.database.log_antispoofing_attempt(
                session_id=session_id,
                result='passed' if antispoofing_result['success'] else 'failed',
                confidence_score=antispoofing_result.get('confidence', 0.0),
                challenge_results=antispoofing_result.get('challenge_results'),
                failure_reason=antispoofing_result.get('message') if not antispoofing_result['success'] else None
            )
            
            if not antispoofing_result['success']:
                self.socketio.emit('attendance_failed', {
                    'phase': 'antispoofing',
                    'reason': 'Anti-spoofing verification failed',
                    'details': antispoofing_result['message'],
                    'confidence': antispoofing_result.get('confidence', 0.0)
                }, room=client_id)
                
                if session_id in self.current_sessions:
                    del self.current_sessions[session_id]
                return
            
            # Phase 2: CNN Face Recognition
            session_data['phase'] = 'recognition'
            self.socketio.emit('phase_update', {
                'phase': 'recognition',
                'message': 'Phase 2: Running CNN face recognition...',
                'progress': 50,
                'antispoofing_confidence': antispoofing_result['confidence']
            }, room=client_id)
            
            recognition_result = self._run_cnn_face_recognition(session_id)
            session_data['recognition_result'] = recognition_result
            
            if recognition_result['success']:
                # Record successful attendance
                self.database.record_attendance(
                    user_id=recognition_result['user_id'],
                    confidence_score=recognition_result['confidence'],
                    antispoofing_score=antispoofing_result['confidence'],
                    recognition_time=recognition_result.get('processing_time'),
                    session_id=session_id,
                    attendance_type='check_in',
                    notes=f"Step 4: Anti-spoofing + CNN recognition"
                )
                
                # Get user info
                user_info = self.database.get_user_info(recognition_result['user_id'])
                
                self.socketio.emit('attendance_success', {
                    'user_id': recognition_result['user_id'],
                    'user_name': user_info['name'] if user_info else 'Unknown',
                    'department': user_info.get('department', '') if user_info else '',
                    'confidence': recognition_result['confidence'],
                    'antispoofing_confidence': antispoofing_result['confidence'],
                    'processing_time': recognition_result.get('processing_time', 0),
                    'timestamp': datetime.now().isoformat(),
                    'step': 'Step 4: Anti-spoofing + CNN Recognition'
                }, room=client_id)
                
                self.logger.info(f"Step 4 attendance success: {recognition_result['user_id']}")
                
            else:
                self.socketio.emit('attendance_failed', {
                    'phase': 'recognition',
                    'reason': 'Face not recognized',
                    'details': recognition_result['message'],
                    'confidence': recognition_result.get('confidence', 0.0),
                    'antispoofing_confidence': antispoofing_result['confidence']
                }, room=client_id)
                
                self.logger.info(f"Step 4 attendance failed: {recognition_result['message']}")
            
            # Clean up session
            if session_id in self.current_sessions:
                del self.current_sessions[session_id]
                
        except Exception as e:
            self.logger.error(f"Failed to process Step 4 attendance session {session_id}: {e}")
            if session_id in self.current_sessions:
                client_id = self.current_sessions[session_id]['client_id']
                self.socketio.emit('error', {
                    'message': f'Step 4 attendance processing failed: {str(e)}'
                }, room=client_id)
                del self.current_sessions[session_id]
    
    def _process_step4_registration(self, session_id: str):
        """Process Step 4 face registration with CNN embedding generation"""
        try:
            session_data = self.current_sessions.get(session_id)
            if not session_data:
                return
            
            client_id = session_data['client_id']
            user_id = session_data['user_id']
            user_name = session_data['user_name']
            target_images = session_data['target_images']
            
            captured_images = []
            valid_images = []
            
            # Phase 1: Capture multiple face images
            self.socketio.emit('registration_phase', {
                'phase': 'capture',
                'message': f'Capturing {target_images} face images for {user_name}...'
            }, room=client_id)
            
            for i in range(target_images):
                if session_id not in self.current_sessions:
                    break  # Session cancelled
                
                self.socketio.emit('registration_progress', {
                    'phase': 'capture',
                    'current': i + 1,
                    'total': target_images,
                    'message': f'Capturing image {i + 1}/{target_images}...'
                }, room=client_id)
                
                # Capture face image with quality check
                face_image = self._capture_quality_face_image()
                if face_image is not None:
                    captured_images.append(face_image)
                    
                    # Run quality validation on captured image
                    if self._validate_face_quality(face_image):
                        valid_images.append(face_image)
                        self.socketio.emit('image_captured', {
                            'captured': len(captured_images),
                            'valid': len(valid_images),
                            'total': target_images
                        }, room=client_id)
                
                time.sleep(0.7)  # Pause between captures for better variety
            
            if len(valid_images) < 8:
                self.socketio.emit('registration_failed', {
                    'reason': 'Not enough high-quality face images captured',
                    'captured': len(captured_images),
                    'valid': len(valid_images),
                    'required': 8
                }, room=client_id)
                
                if session_id in self.current_sessions:
                    del self.current_sessions[session_id]
                return
            
            # Phase 2: Generate CNN face embeddings
            self.socketio.emit('registration_phase', {
                'phase': 'embedding',
                'message': f'Generating CNN face embeddings for {user_name}...'
            }, room=client_id)
            
            registration_result = self.face_recognition.register_face(valid_images, user_id)
            
            if registration_result['success']:
                # Phase 3: Store embedding in database
                self.socketio.emit('registration_phase', {
                    'phase': 'database',
                    'message': 'Storing face embedding in database...'
                }, room=client_id)
                
                storage_success = self.database.store_face_embedding(
                    user_id=user_id,
                    embedding=registration_result['embedding'],
                    quality_score=0.95,  # High quality since we validated images
                    num_images=registration_result['num_embeddings'],
                    metadata={
                        'registration_method': 'Step 4 CNN',
                        'images_captured': len(captured_images),
                        'valid_images': len(valid_images),
                        'embedding_dim': len(registration_result['embedding'])
                    }
                )
                
                if storage_success:
                    # Reload embeddings into recognition system
                    self._load_embeddings()
                    
                    self.socketio.emit('registration_success', {
                        'user_id': user_id,
                        'user_name': user_name,
                        'images_captured': len(captured_images),
                        'valid_images': len(valid_images),
                        'images_used': registration_result['num_embeddings'],
                        'embedding_dim': len(registration_result['embedding']),
                        'message': f'Step 4 CNN face registration completed for {user_name}',
                        'ready_for_recognition': True
                    }, room=client_id)
                    
                    self.logger.info(f"Step 4 registration success: {user_id}")
                else:
                    self.socketio.emit('registration_failed', {
                        'reason': 'Failed to store face embedding in database'
                    }, room=client_id)
            else:
                self.socketio.emit('registration_failed', {
                    'reason': f'CNN embedding generation failed: {registration_result["message"]}'
                }, room=client_id)
            
            # Clean up session
            if session_id in self.current_sessions:
                del self.current_sessions[session_id]
                
        except Exception as e:
            self.logger.error(f"Failed to process Step 4 registration session {session_id}: {e}")
            if session_id in self.current_sessions:
                client_id = self.current_sessions[session_id]['client_id']
                self.socketio.emit('error', {
                    'message': f'Step 4 registration failed: {str(e)}'
                }, room=client_id)
                del self.current_sessions[session_id]
    
    def _run_antispoofing_verification(self, session_id: str) -> Dict:
        """Run Step 3 anti-spoofing verification"""
        try:
            # Run the complete challenge-response system from Step 3
            result = self.challenge_system.run_sequential_challenges()
            
            self.logger.info(f"Anti-spoofing result for {session_id}: {result.get('success', False)}")
            
            return {
                'success': result.get('success', False),
                'confidence': result.get('confidence', 0.0),
                'message': result.get('message', 'Anti-spoofing verification completed'),
                'challenge_results': result.get('challenge_results', {}),
                'processing_time': result.get('processing_time', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Anti-spoofing verification failed: {e}")
            return {
                'success': False,
                'confidence': 0.0,
                'message': f'Anti-spoofing verification failed: {str(e)}'
            }
    
    def _run_cnn_face_recognition(self, session_id: str) -> Dict:
        """Run CNN-based face recognition"""
        try:
            # Capture high-quality face image
            face_image = self._capture_quality_face_image()
            if face_image is None:
                return {
                    'success': False,
                    'user_id': 'Unknown',
                    'confidence': 0.0,
                    'message': 'Could not capture high-quality face image'
                }
            
            # Run CNN face recognition
            result = self.face_recognition.recognize_face(face_image)
            
            self.logger.info(f"CNN recognition result for {session_id}: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"CNN face recognition failed: {e}")
            return {
                'success': False,
                'user_id': 'Unknown',
                'confidence': 0.0,
                'message': f'CNN face recognition failed: {str(e)}'
            }
    
    def _capture_quality_face_image(self) -> Optional[np.ndarray]:
        """Capture a high-quality face image from camera"""
        try:
            if self.camera is None or not self.is_streaming:
                return None
            
            # Capture multiple frames and select the best one
            best_frame = None
            best_score = 0
            
            for _ in range(5):  # Try 5 frames
                ret, frame = self.camera.read()
                if ret:
                    # Simple quality scoring (you could improve this)
                    score = cv2.Laplacian(frame, cv2.CV_64F).var()  # Variance of Laplacian for sharpness
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()
                
                time.sleep(0.1)  # Brief pause between frames
            
            return best_frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture quality face image: {e}")
            return None
    
    def _validate_face_quality(self, face_image: np.ndarray) -> bool:
        """Validate that the image is suitable for CNN recognition"""
        try:
            if face_image is None or face_image.size == 0:
                return False
            
            # Check image sharpness
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Require minimum sharpness
            if laplacian_var < 100:  # Threshold for sharpness
                return False
            
            # Check brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 50 or mean_brightness > 200:  # Too dark or too bright
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Face quality validation failed: {e}")
            return False
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the Step 4 Flask application"""
        try:
            self.logger.info(f"Starting Step 4 attendance application on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Failed to start Step 4 application: {e}")
            raise
        finally:
            self._stop_camera()


def create_step4_app():
    """Application factory function for Step 4"""
    app = Step4AttendanceApp()
    return app.app, app.socketio


if __name__ == '__main__':
    app = Step4AttendanceApp()
    app.run(debug=True)

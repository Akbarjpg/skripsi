"""
Flask Web Application for Face Anti-Spoofing Attendance System
Clean, organized implementation with application factory pattern
"""

import os
import sqlite3
import time
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

from ..utils.logger import get_web_logger
from ..utils.config import SystemConfig


def create_app(config: SystemConfig = None):
    """Application factory for Flask app with SocketIO"""
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    template_dir = project_root / "src" / "web" / "templates"
    static_dir = project_root / "src" / "web" / "static"
    
    # Create Flask app with proper paths
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
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize database
    init_db(app)
    
    # Register routes
    register_routes(app, logger)
    register_api_routes(app, logger)
    register_socketio_events(socketio, logger)
    register_error_handlers(app, logger)
    
    logger.info("✅ Flask application with SocketIO created successfully")
    return app, socketio


def init_db(app: Flask) -> None:
    """Initialize database with required tables"""
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
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Attendance records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_type TEXT DEFAULT 'in',
                    confidence_score REAL,
                    liveness_score REAL,
                    challenge_passed BOOLEAN DEFAULT 0,
                    image_path TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Verification sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS verification_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    challenges_completed INTEGER DEFAULT 0,
                    final_score REAL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()


def register_routes(app: Flask, logger) -> None:
    """Register main application routes"""
    
    @app.route('/')
    def index():
        """Home page - redirect to test dashboard"""
        return render_template('test_dashboard.html')
    
    @app.route('/home')
    def home():
        """Test dashboard page"""
        return render_template('test_dashboard.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """User login"""
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Username and password are required', 'error')
                return render_template('login.html')
            
            # Check user credentials
            db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
            
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT id, username, password_hash, full_name, role FROM users WHERE username = ? AND is_active = 1',
                        (username,)
                    )
                    user = cursor.fetchone()
                
                if user and check_password_hash(user[2], password):
                    session['user_id'] = user[0]
                    session['username'] = user[1]
                    session['full_name'] = user[3]
                    session['role'] = user[4]
                    
                    flash(f'Welcome back, {user[3]}!', 'success')
                    logger.info(f"User {username} logged in successfully")
                    
                    return redirect(url_for('attendance'))
                else:
                    flash('Invalid username or password', 'error')
                    logger.warning(f"Failed login attempt for username: {username}")
                    
            except Exception as e:
                logger.error(f"Login error: {e}")
                flash('An error occurred during login', 'error')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """User registration"""
        if request.method == 'POST':
            username = request.form.get('username')
            full_name = request.form.get('full_name')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            role = request.form.get('role', 'user')
            
            # Validation
            if not all([username, full_name, password, confirm_password]):
                flash('All required fields must be filled', 'error')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return render_template('register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long', 'error')
                return render_template('register.html')
            
            # Create user
            db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
            
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check if username already exists
                    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
                    if cursor.fetchone():
                        flash('Username already exists', 'error')
                        return render_template('register.html')
                    
                    # Create new user
                    password_hash = generate_password_hash(password)
                    cursor.execute('''
                        INSERT INTO users (username, full_name, email, password_hash, role)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (username, full_name, email, password_hash, role))
                    
                    conn.commit()
                    
                    flash('Registration successful! Please log in.', 'success')
                    logger.info(f"New user registered: {username}")
                    
                    return redirect(url_for('login'))
                    
            except Exception as e:
                logger.error(f"Registration error: {e}")
                flash('An error occurred during registration', 'error')
        
        return render_template('register.html')
    
    @app.route('/logout')
    def logout():
        """User logout"""
        username = session.get('username', 'Unknown')
        session.clear()
        flash('You have been logged out successfully', 'info')
        logger.info(f"User {username} logged out")
        return redirect(url_for('login'))
    
    @app.route('/attendance')
    def attendance():
        """Attendance verification page"""
        if 'user_id' not in session:
            flash('Please log in to access attendance system', 'warning')
            return redirect(url_for('login'))
        
        return render_template('attendance.html', user=session)
    
    @app.route('/socketio-test')
    def socketio_test():
        """Socket.IO connection test page"""
        return render_template('socketio_test.html')
    
    @app.route('/face-detection-test')
    def face_detection_test():
        """Face detection with landmark visualization test page"""
        return render_template('face_detection_test.html')
    
    @app.route('/face-detection-test-fixed')
    def face_detection_test_fixed():
        """Fixed face detection test page"""
        return render_template('face_detection_test_fixed.html')
    
    @app.route('/face-detection-clean')
    def face_detection_clean():
        """Clean face detection test page - fully validated"""
        return render_template('face_detection_clean.html')
    
    @app.route('/face_detection')
    def face_detection():
        """Main face detection page with integrated liveness detection"""
        return render_template('face_detection_clean.html')
    
    @app.route('/simple-camera-test')
    def simple_camera_test():
        """Simple camera test page for debugging"""
        return render_template('simple_camera_test.html')
    
    @app.route('/test-dashboard')
    def test_dashboard():
        """Test dashboard page with all testing features"""
        return render_template('test_dashboard.html')
    
    @app.route('/dashboard')
    def dashboard():
        """User dashboard"""
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        # Get user's attendance records
        db_path = app.config.get('DATABASE_URL', 'sqlite:///attendance.db').replace('sqlite:///', '')
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, check_type, confidence_score, liveness_score, challenge_passed
                    FROM attendance_records 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                ''', (session['user_id'],))
                
                records = cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            records = []
        
        return render_template('dashboard.html', records=records, user=session)


def register_api_routes(app: Flask, logger) -> None:
    """Register API routes for AJAX calls"""
    
    @app.route('/api/verify', methods=['POST'])
    def api_verify():
        """API endpoint for face verification"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        try:
            # This would integrate with your face verification system
            # For now, return a mock response
            
            return jsonify({
                'success': True,
                'confidence': 0.95,
                'liveness_score': 0.88,
                'challenge_passed': True,
                'message': 'Verification successful'
            })
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/enroll', methods=['POST'])
    def api_enroll():
        """API endpoint for face enrollment"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        try:
            # This would integrate with your face enrollment system
            # For now, return a mock response
            
            return jsonify({
                'success': True,
                'message': 'Face enrollment successful'
            })
            
        except Exception as e:
            logger.error(f"Enrollment error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/challenge', methods=['POST'])
    def api_challenge():
        """API endpoint for liveness challenges"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        try:
            challenge_type = request.json.get('type', 'blink')
            
            # This would integrate with your challenge system
            # For now, return a mock response
            
            return jsonify({
                'success': True,
                'challenge_type': challenge_type,
                'instructions': f'Please {challenge_type} for verification',
                'timeout': 10
            })
            
        except Exception as e:
            logger.error(f"Challenge error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/start_verification', methods=['POST'])
    def start_verification():
        """Start verification session"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        try:
            user_id = session['user_id']
            
            # Generate a session ID
            import uuid
            session_id = str(uuid.uuid4())
            
            # For now, create a simple challenge
            challenge_types = ['blink', 'mouth_open', 'head_left', 'head_right']
            import random
            challenge_type = random.choice(challenge_types)
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'challenge': {
                    'id': str(uuid.uuid4()),
                    'type': challenge_type,
                    'description': f'Please {challenge_type.replace("_", " ")} for verification',
                    'duration': 5.0
                }
            })
            
        except Exception as e:
            logger.error(f"Start verification error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/complete_attendance', methods=['POST'])
    def complete_attendance():
        """Complete attendance recording"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        try:
            data = request.json or {}
            session_id = data.get('session_id')
            check_type = data.get('check_type', 'check_in')
            verification_data = data.get('verification_data', {})
            
            # For now, just log the attendance (in real implementation, save to database)
            logger.info(f"Attendance recorded for user {session['user_id']}: {check_type}")
            
            return jsonify({
                'success': True, 
                'message': 'Attendance recorded successfully',
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Complete attendance error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500


def register_socketio_events(socketio, logger) -> None:
    """Register SocketIO event handlers for real-time communication"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info("Client connected to landmark detection")
        emit('status', {'message': 'Connected to server'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("Client disconnected from landmark detection")
    
    @socketio.on('process_frame')
    def handle_frame_processing(data):
        """Handle real-time frame processing via WebSocket with landmark detection"""
        import time  # Explicit import to avoid scoping issues
        print("=== DEBUG: process_frame received ===")
        print(f"=== DEBUG: Data keys: {list(data.keys())}")
        
        # JSON safety function
        def make_json_safe(obj):
            """Convert numpy types and other non-JSON types to JSON-safe types"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(item) for item in obj]
            elif obj is None:
                return None
            else:
                return obj
        
        try:
            # Allow testing mode without authentication - face detection page works without login
            test_mode = data.get('test_mode', False)
            
            # For face detection page, allow processing without authentication
            print(f"=== DEBUG: test_mode: {test_mode} ===")
            
            if 'user_id' not in session:
                print("=== DEBUG: No user session found - using anonymous mode ===")
                user_id = 'anonymous_user'  # Allow anonymous access for testing
            else:
                user_id = session.get('user_id')
                print(f"=== DEBUG: Using authenticated user: {user_id} ===")
            
            print(f"=== DEBUG: Processing frame for user: {user_id}")
            
            # Decode base64 image
            if 'image' not in data:
                print("=== DEBUG: No image data in request ===")
                emit('error', {'message': 'No image data'})
                return
                
            image_b64 = data['image']
            print(f"=== DEBUG: Image data length: {len(image_b64)}")
            
            try:
                image_data = base64.b64decode(image_b64.split(',')[1])
                print(f"=== DEBUG: Decoded image data length: {len(image_data)}")
            except Exception as decode_error:
                print(f"=== DEBUG: Base64 decode error: {decode_error}")
                emit('error', {'message': 'Invalid base64 data'})
                return
                
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("=== DEBUG: Failed to decode image ===")
                emit('error', {'message': 'Invalid image data'})
                return
            
            print(f"=== DEBUG: Image decoded successfully, shape: {image.shape}")
            
            session_id = data.get('session_id', 'default')
            print(f"=== DEBUG: Session ID: {session_id}")
            
            # Always try real detection first, use test mode only as fallback
            print("=== DEBUG: Attempting REAL MediaPipe landmark detection ===")
            
            # Initialize landmark detector for real detection
            try:
                print("=== DEBUG: Importing LivenessVerifier ===")
                from ..detection.landmark_detection import LivenessVerifier
                verifier = LivenessVerifier()
                print("=== DEBUG: LivenessVerifier initialized ===")
                
                # Process frame with landmark detection
                print("=== DEBUG: Processing frame with MediaPipe ===")
                landmark_results = verifier.process_frame(image)
                print(f"=== DEBUG: Landmark results: {landmark_results}")
                
                # Get actual landmark coordinates (normalized 0-1)
                landmarks = landmark_results.get('landmark_coordinates', [])
                print(f"=== DEBUG: {len(landmarks)} landmarks detected")
                
                # If we have real landmarks, use them instead of test mode
                if landmarks and len(landmarks) > 0:
                    print("=== DEBUG: Using REAL MediaPipe landmarks ===")
                    use_real_landmarks = True
                else:
                    print("=== DEBUG: No real landmarks detected, will use test mode as fallback ===")
                    use_real_landmarks = False
                    
            except Exception as detection_error:
                print(f"=== DEBUG: Real detection failed: {detection_error} ===")
                use_real_landmarks = False
                landmarks = []
            
            # If test_mode is requested AND real detection failed, create test landmarks
            if test_mode and not use_real_landmarks:
                print("=== DEBUG: Creating fallback test landmarks ===")
                
                # Create colorful test landmarks
                test_landmarks = []
                colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFFFFF']
                
                # Create a nice pattern of test points
                for i in range(50):  # 50 colorful test points
                    x = 0.25 + (i % 10) * 0.05  # 10 points across
                    y = 0.25 + (i // 10) * 0.08   # 5 points down
                    color = colors[i % len(colors)]
                    
                    test_landmarks.append({
                        'x': x,
                        'y': y, 
                        'color': color,
                        'index': i
                    })
                
                print(f"=== DEBUG: Created {len(test_landmarks)} test landmarks ===")
                
                # Send test result
                results = {
                    'session_id': str(session_id),
                    'frame_processed': True,
                    'timestamp': float(time.time()),
                    'landmarks_detected': True,
                    'landmark_count': len(test_landmarks),
                    'landmarks': test_landmarks,
                    'test_mode': True,
                    'blink_count': 3,
                    'head_movement': True,
                    'mouth_open': False,
                    'liveness_score': 0.95,
                    'cnn_confidence': 0.98,
                    'security_score': 0.96,
                    'security_level': 'SECURE',
                    'message': f'TEST MODE: {len(test_landmarks)} colored test points displayed'
                }
                
                print(f"=== DEBUG: Emitting TEST landmark result ===")
                emit('landmark_result', results)
                return
            
            # Continue with real landmark processing if we have them
            if use_real_landmarks:
                # Convert landmarks to frontend format with colors
                landmark_points = []
                print("=== DEBUG: Converting landmarks for frontend ===")
                print(f"=== DEBUG: Landmarks type: {type(landmarks)} ===")
                print(f"=== DEBUG: Landmarks length: {len(landmarks)} ===")
                if landmarks:
                    print(f"=== DEBUG: First landmark type: {type(landmarks[0])} ===")
                    print(f"=== DEBUG: First landmark: {landmarks[0]} ===")
                
                # MediaPipe 478/468 landmark indices for different face parts (adjusted for 478)
                face_contour = list(range(0, 17)) + list(range(267, 284)) + list(range(389, 397))
                left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                nose_tip = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305]
                mouth_outer = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
                left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                right_eyebrow = [296, 334, 293, 300, 276, 283, 282, 295, 285, 336]
                
                # Handle both 468 and 478 landmarks
                max_landmarks = min(478, len(landmarks))
                print(f"=== DEBUG: Processing {max_landmarks} landmarks ===")
                
                # Add all landmarks with appropriate colors
                for i, landmark in enumerate(landmarks):
                    if i < max_landmarks:  # Support both 468 and 478 landmarks
                        try:
                            # Handle different landmark formats
                            if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                                x = float(landmark[0])  # Already normalized
                                y = float(landmark[1])  # Already normalized
                            elif hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                                x = float(landmark.x)  # MediaPipe landmark object
                                y = float(landmark.y)
                            else:
                                print(f"=== DEBUG: Unknown landmark format at index {i}: {type(landmark)} ===")
                                continue
                            
                            # Determine color based on landmark region (bounds check)
                            if i < len(face_contour) and i in face_contour:
                                color = '#00FF00'  # Green for face outline
                            elif i in left_eye or i in right_eye:
                                color = '#FF0000'  # Red for eyes
                            elif i in nose_tip:
                                color = '#0000FF'  # Blue for nose
                            elif i in mouth_outer:
                                color = '#FFFF00'  # Yellow for mouth
                            elif i in left_eyebrow or i in right_eyebrow:
                                color = '#FF00FF'  # Magenta for eyebrows
                            else:
                                color = '#FFFFFF'  # White for other points
                            
                            landmark_points.append({
                                'x': float(x),  # Ensure JSON serializable
                                'y': float(y),  # Ensure JSON serializable
                                'color': str(color),  # Ensure JSON serializable
                                'index': int(i)  # Ensure JSON serializable
                            })
                            
                        except Exception as landmark_error:
                            print(f"=== DEBUG: Error processing landmark {i}: {landmark_error} ===")
                            print(f"=== DEBUG: Landmark data: {landmark} ===")
                            continue
            
                print(f"=== DEBUG: Converted {len(landmark_points)} landmark points for frontend ===")
                
                # Get liveness metrics from detection results
                has_landmarks = len(landmark_points) > 0
                blink_count = landmark_results.get('blink_count', 0)
                head_movement = landmark_results.get('head_movement', False)
                mouth_open = landmark_results.get('mouth_open', False)
                ear_left = landmark_results.get('ear_left', 0.0)
                ear_right = landmark_results.get('ear_right', 0.0)
                mar = landmark_results.get('mar', 0.0)
                head_pose = landmark_results.get('head_pose', None)
                
                # Get calculated liveness score from detection
                liveness_score = landmark_results.get('liveness_score', 0.0) / 100.0  # Convert to 0-1 range
                is_live = landmark_results.get('is_live', False)
                liveness_status = landmark_results.get('liveness_status', 'UNKNOWN')
                liveness_metrics = landmark_results.get('liveness_metrics', {})
                
                print(f"=== DEBUG: Liveness results - Score: {liveness_score:.3f}, Status: {liveness_status}, Is Live: {is_live}")
                
                # Simulate CNN score (you can integrate actual CNN model here)
                cnn_confidence = 0.92 if has_landmarks else 0.0
                
                # Calculate security score based on liveness
                security_score = (liveness_score + cnn_confidence) / 2.0
                
                # Security assessment
                if security_score > 0.8:
                    security_level = "SECURE" 
                elif security_score > 0.6:
                    security_level = "GOOD"
                elif security_score > 0.4:
                    security_level = "WARNING"
                else:
                    security_level = "DANGER"
                
                # Clean data to ensure JSON serialization
                safe_head_pose = make_json_safe(head_pose) if head_pose else None
                safe_liveness_metrics = make_json_safe(liveness_metrics) if liveness_metrics else {}

                results = {
                    'session_id': str(session_id),
                    'frame_processed': True,
                    'timestamp': float(time.time()),
                    'landmarks_detected': bool(has_landmarks),  # Based on actual landmark points
                    'landmark_count': int(len(landmark_points)),  # Use actual converted points
                    'landmarks': landmark_points,  # Send landmark coordinates to frontend
                    'test_mode': False,  # REAL landmarks
                    
                    # Liveness detection results
                    'blink_count': int(blink_count) if blink_count else 0,
                    'head_movement': bool(head_movement),
                    'mouth_open': bool(mouth_open),
                    'ear_left': float(ear_left),
                    'ear_right': float(ear_right),
                    'mar': float(mar),
                    'head_pose': safe_head_pose,
                    
                    # Liveness assessment
                    'liveness_score': float(liveness_score),
                    'liveness_raw_score': float(landmark_results.get('liveness_score', 0.0)),  # 0-100 scale
                    'is_live': bool(is_live),
                    'liveness_status': str(liveness_status),
                    'liveness_metrics': safe_liveness_metrics,
                    
                    # Overall assessment
                    'cnn_confidence': float(cnn_confidence),
                    'security_score': float(security_score),
                    'security_level': str(security_level),
                    'message': f'Liveness: {liveness_status} ({landmark_results.get("liveness_score", 0):.1f}/100)'
                }
                
                print(f"=== DEBUG: Emitting REAL landmark_result with liveness score {liveness_score:.3f} ===")
                emit('landmark_result', results)
                
            else:
                print("=== DEBUG: No landmarks detected - sending empty result ===")
                # Send empty result when no detection
                results = {
                    'session_id': str(session_id),
                    'frame_processed': True,
                    'timestamp': float(time.time()),
                    'landmarks_detected': False,
                    'landmark_count': 0,
                    'landmarks': [],
                    'test_mode': False,
                    'message': 'No face detected',
                    'liveness_score': 0.0,
                    'cnn_confidence': 0.0,
                    'security_score': 0.0,
                    'security_level': 'WARNING'
                }
                emit('landmark_result', results)
                
        except Exception as main_error:
            print(f"=== DEBUG: Main frame processing error: {main_error} ===")
            logger.error(f"Frame processing error: {main_error}")
            emit('error', {'message': f'Frame processing failed: {str(main_error)}'})
            
        print("=== DEBUG: process_frame handler completed ===")
    
    @socketio.on('start_challenge')
    def handle_start_challenge(data):
        """Handle challenge start request"""
        try:
            if 'user_id' not in session:
                emit('error', {'message': 'Not authenticated'})
                return
            
            challenge_type = data.get('type', 'blink')
            
            # Generate a simple challenge
            challenge = {
                'id': f"challenge_{int(time.time())}",
                'type': challenge_type,
                'description': f"Please perform: {challenge_type}",
                'duration': 5000,  # 5 seconds
                'status': 'active'
            }
            
            emit('challenge_started', challenge)
            logger.info(f"Challenge started for user: {session['user_id']}")
            
        except Exception as e:
            logger.error(f"Challenge start error: {e}")
            emit('error', {'message': f'Challenge start failed: {str(e)}'})


def register_error_handlers(app: Flask, logger) -> None:
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors"""
        logger.warning(f"404 error: {request.url}")
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"500 error: {error}")
        return render_template('500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        """Handle 403 errors"""
        logger.warning(f"403 error: {request.url}")
        return render_template('403.html'), 403


# For backward compatibility and direct running
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

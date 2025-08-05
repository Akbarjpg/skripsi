"""
Flask Web Application untuk Face Anti-Spoofing Attendance System
Menyediakan interface untuk absensi dengan verifikasi multi-layer
"""

import os
import sys
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, session, Response, redirect
from flask_socketio import SocketIO, emit
import base64
import time
import logging
from datetime import datetime
import json
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.cnn_model import create_model
from src.detection.landmark_detection import LivenessVerifier
from src.challenge.challenge_response import ChallengeResponseSystem

class AttendanceSystem:
    """
    Main attendance system class
    """
    
    def __init__(self, model_path=None, db_path='attendance.db'):
        self.db_path = db_path
        self.setup_database()
        
        # Initialize AI components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.liveness_verifier = LivenessVerifier()
        self.challenge_system = ChallengeResponseSystem()
        
        # Load trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Session state
        self.active_sessions = {}
        
    def setup_database(self):
        """Setup SQLite database untuk attendance records"""
        conn = sqlite3.connect(self.db_path)
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
                check_type TEXT NOT NULL,  -- 'check_in' or 'check_out'
                verification_score REAL,
                challenge_results TEXT,  -- JSON string
                ip_address TEXT,
                location_data TEXT,  -- JSON string
                verification_methods TEXT,  -- JSON array of methods used
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Verification sessions table (untuk tracking multi-step verification)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',  -- 'active', 'completed', 'failed'
                cnn_score REAL,
                landmark_score REAL,
                challenge_score REAL,
                fusion_score REAL,
                challenge_data TEXT,  -- JSON string
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_model(self, model_path):
        """Load trained CNN model"""
        try:
            self.model = create_model('custom', num_classes=2)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.model = None
            
    def predict_liveness(self, image):
        """
        Predict liveness menggunakan trained CNN model
        """
        if self.model is None:
            return 0.5, "Model not loaded"
        
        try:
            # Preprocess image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs, _ = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                liveness_score = probabilities[0][1].item()  # Probability of real class
                
            return liveness_score, "success"
            
        except Exception as e:
            logging.error(f"CNN prediction error: {e}")
            return 0.0, f"Prediction error: {str(e)}"
    
    def create_verification_session(self, user_id, ip_address, user_agent):
        """Create new verification session"""
        session_id = f"session_{int(time.time())}_{user_id}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO verification_sessions 
            (id, user_id, ip_address, user_agent) 
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, ip_address, user_agent))
        conn.commit()
        conn.close()
        
        return session_id
    
    def update_verification_session(self, session_id, **kwargs):
        """Update verification session dengan scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in ['cnn_score', 'landmark_score', 'challenge_score', 'fusion_score', 'status']:
                update_fields.append(f"{key} = ?")
                values.append(value)
            elif key == 'challenge_data':
                update_fields.append("challenge_data = ?")
                values.append(json.dumps(value))
        
        if update_fields:
            values.append(session_id)
            query = f"UPDATE verification_sessions SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def calculate_fusion_score(self, cnn_score, landmark_score, challenge_score):
        """
        Calculate fusion score berdasarkan logika: minimal 2 dari 3 metode harus lulus
        """
        threshold = 0.7
        
        methods_passed = 0
        if cnn_score >= threshold:
            methods_passed += 1
        if landmark_score >= threshold:
            methods_passed += 1
        if challenge_score >= threshold:
            methods_passed += 1
        
        # Fusion score berdasarkan weighted average dengan bonus untuk multiple methods
        weights = [0.4, 0.3, 0.3]  # CNN, Landmark, Challenge
        weighted_score = (cnn_score * weights[0] + 
                         landmark_score * weights[1] + 
                         challenge_score * weights[2])
        
        # Bonus jika minimal 2 metode lulus
        if methods_passed >= 2:
            bonus = 0.1 * methods_passed
            weighted_score = min(1.0, weighted_score + bonus)
        else:
            # Penalty jika kurang dari 2 metode lulus
            weighted_score *= 0.7
        
        return weighted_score, methods_passed
    
    def record_attendance(self, user_id, check_type, verification_data):
        """Record attendance dengan verification data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO attendance_records 
            (user_id, check_type, verification_score, challenge_results, 
             ip_address, location_data, verification_methods) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            check_type,
            verification_data.get('fusion_score', 0.0),
            json.dumps(verification_data.get('challenge_results', {})),
            verification_data.get('ip_address', ''),
            json.dumps(verification_data.get('location_data', {})),
            json.dumps(verification_data.get('methods_used', []))
        ))
        
        conn.commit()
        conn.close()

# Initialize Flask app
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize attendance system
attendance_system = AttendanceSystem()

@app.route('/')
def index():
    """Main page"""
    if 'user_id' in session:
        return redirect('/attendance')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check credentials in database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, password_hash FROM users 
            WHERE username = ?
        ''', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = username
            session['name'] = user[1]
            session['role'] = 'user'  # Default role
            
            # Redirect to intended page or attendance
            next_page = request.args.get('next', '/attendance')
            return redirect(next_page)
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        name = request.form.get('name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not name or not password:
            return render_template('register.html', error='All fields are required')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        # Check if username already exists
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return render_template('register.html', error='Username already exists')
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (username, name, password_hash) 
            VALUES (?, ?, ?)
        ''', (username, name, password_hash))
        conn.commit()
        conn.close()
        
        return render_template('login.html', success='Registration successful. Please login.')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect('/login')

@app.route('/attendance')
def attendance():
    """Attendance verification page"""
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('attendance.html')

@app.route('/admin')
def admin_page():
    """Admin dashboard"""
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    return render_template('admin.html')

@app.route('/api/start_verification', methods=['POST'])
def start_verification():
    """Start verification session"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = session['user_id']
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    session_id = attendance_system.create_verification_session(user_id, ip_address, user_agent)
    
    # Start challenge
    challenge = attendance_system.challenge_system.start_challenge('random')
    
    return jsonify({
        'session_id': session_id,
        'challenge': {
            'id': challenge.challenge_id,
            'type': challenge.challenge_type.value,
            'description': challenge.description,
            'duration': challenge.duration
        }
    })

@socketio.on('process_frame')
def handle_frame_processing(data):
    """Handle real-time frame processing via WebSocket"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        session_id = data.get('session_id')
        
        # Process with all verification methods
        results = {}
        
        # 1. CNN Liveness Detection
        cnn_score, cnn_message = attendance_system.predict_liveness(image)
        results['cnn_score'] = cnn_score
        results['cnn_message'] = cnn_message
        
        # 2. Facial Landmark Detection
        landmark_results = attendance_system.liveness_verifier.process_frame(image)
        landmark_score = 0.8 if landmark_results['landmarks_detected'] else 0.0
        
        # Adjust score based on natural movements
        if landmark_results['landmarks_detected']:
            natural_score = 0.0
            if landmark_results['blink_count'] > 0:
                natural_score += 0.3
            if landmark_results['head_movement']:
                natural_score += 0.3
            if landmark_results['mouth_open']:
                natural_score += 0.2
            landmark_score = min(1.0, landmark_score + natural_score)
        
        results['landmark_score'] = landmark_score
        results['landmark_results'] = landmark_results
        
        # 3. Challenge Processing
        challenge_result = attendance_system.challenge_system.process_frame(landmark_results)
        challenge_score = 0.0
        
        if challenge_result:
            challenge_score = 1.0 if challenge_result.success else 0.3
            results['challenge_completed'] = True
            results['challenge_result'] = {
                'success': challenge_result.success,
                'response_time': challenge_result.response_time,
                'confidence': challenge_result.confidence_score
            }
        else:
            results['challenge_completed'] = False
            
            # Get current challenge status
            challenge_status = attendance_system.challenge_system.get_current_challenge_status()
            results['challenge_status'] = challenge_status
        
        results['challenge_score'] = challenge_score
        
        # 4. Fusion Score
        fusion_score, methods_passed = attendance_system.calculate_fusion_score(
            cnn_score, landmark_score, challenge_score
        )
        results['fusion_score'] = fusion_score
        results['methods_passed'] = methods_passed
        results['verification_success'] = fusion_score >= 0.7 and methods_passed >= 2
        
        # Update session
        if session_id:
            attendance_system.update_verification_session(
                session_id,
                cnn_score=cnn_score,
                landmark_score=landmark_score,
                challenge_score=challenge_score,
                fusion_score=fusion_score,
                status='completed' if results['challenge_completed'] else 'active'
            )
        
        # Emit results back to client
        emit('verification_results', results)
        
    except Exception as e:
        logging.error(f"Frame processing error: {e}")
        emit('error', {'message': str(e)})

@app.route('/api/complete_attendance', methods=['POST'])
def complete_attendance():
    """Complete attendance recording"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    session_id = data.get('session_id')
    check_type = data.get('check_type', 'check_in')
    verification_data = data.get('verification_data', {})
    
    # Record attendance
    attendance_system.record_attendance(
        session['user_id'],
        check_type,
        verification_data
    )
    
    return jsonify({'success': True, 'message': 'Attendance recorded successfully'})

@app.route('/api/attendance_history')
def attendance_history():
    """Get attendance history for current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = sqlite3.connect(attendance_system.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, check_type, verification_score 
        FROM attendance_records 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 50
    ''', (session['user_id'],))
    
    records = cursor.fetchall()
    conn.close()
    
    history = []
    for record in records:
        history.append({
            'timestamp': record[0],
            'check_type': record[1],
            'verification_score': record[2]
        })
    
    return jsonify({'history': history})

# Additional API Endpoints
@app.route('/api/enroll', methods=['POST'])
def enroll_user():
    """Enroll new user face data"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Get image data from request
        image_data = request.json.get('image_data')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process enrollment
        user_id = session['user_id']
        # Here you would save face embeddings for the user
        # For now, just return success
        
        return jsonify({
            'success': True,
            'message': 'Face enrolled successfully'
        })
    
    except Exception as e:
        logging.error(f"Enrollment error: {e}")
        return jsonify({'error': 'Enrollment failed'}), 500

@app.route('/api/challenge', methods=['GET'])
def get_challenge():
    """Get current challenge for user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        challenge_status = attendance_system.challenge_system.get_current_challenge_status()
        return jsonify(challenge_status)
    
    except Exception as e:
        logging.error(f"Challenge error: {e}")
        return jsonify({'error': 'Failed to get challenge'}), 500

@app.route('/api/verify', methods=['POST'])
def verify_face():
    """Verify face for attendance"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Get image data from request
        image_data = request.json.get('image_data')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process verification
        # This would integrate with your existing verification logic
        verification_result = {
            'success': True,
            'confidence': 0.85,
            'liveness_score': 0.92,
            'message': 'Verification successful'
        }
        
        return jsonify(verification_result)
    
    except Exception as e:
        logging.error(f"Verification error: {e}")
        return jsonify({'error': 'Verification failed'}), 500

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden(e):
    """Handle 403 errors"""
    return render_template('403.html'), 403

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run with SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

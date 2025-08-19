"""
FOCUSED ANTI-SPOOFING ATTENDANCE SYSTEM
=======================================
Core focus: Advanced anti-spoofing detection for attendance
- Users: Just scan face (no login required)
- Admin: Login required for management
- Main goal: Detect and prevent spoofing attacks
"""

import cv2
import numpy as np
import sqlite3
import json
import time
import os
from datetime import datetime
from threading import Lock

# Import our advanced anti-spoofing system
import sys
sys.path.append('.')
from src.detection.advanced_antispoofing import AdvancedAntiSpoofingDetector
from src.detection.landmark_detection import FacialLandmarkDetector

class AntiSpoofingAttendanceSystem:
    """
    Core Anti-Spoofing Attendance System
    Focus: Maximum spoofing detection accuracy
    """
    
    def __init__(self):
        print("🔒 Initializing Anti-Spoofing Attendance System")
        
        # Core components
        self.face_detector = FacialLandmarkDetector()
        self.antispoofing_detector = AdvancedAntiSpoofingDetector()
        
        # Database
        self.db_lock = Lock()
        self.init_database()
        
        # Anti-spoofing configuration
        self.spoofing_config = {
            'min_confidence': 0.7,      # Minimum face confidence
            'spoofing_threshold': 0.5,   # Anti-spoofing threshold
            'required_frames': 5,        # Frames needed for verification
            'max_attempts': 3,           # Max failed attempts
            'cooldown_seconds': 30       # Cooldown after failed attempts
        }
        
        # Session tracking
        self.session_data = {}
        self.failed_attempts = {}
        
        print("✅ Anti-Spoofing System Ready")
    
    def init_database(self):
        """Initialize attendance database"""
        with sqlite3.connect('attendance.db') as conn:
            cursor = conn.cursor()
            
            # Create users table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    spoofing_score REAL,
                    confidence REAL,
                    detection_method TEXT,
                    status TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create spoofing_logs table for security analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spoofing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    spoofing_score REAL,
                    attack_type TEXT,
                    confidence REAL,
                    ip_address TEXT,
                    status TEXT
                )
            ''')
            
            conn.commit()
    
    def process_attendance(self, frame, session_id=None):
        """
        Core anti-spoofing attendance processing
        Returns: dict with attendance result and spoofing analysis
        """
        start_time = time.time()
        
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Initialize session if new
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                'frames_processed': 0,
                'spoofing_scores': [],
                'face_confidences': [],
                'start_time': start_time,
                'status': 'processing'
            }
        
        session = self.session_data[session_id]
        
        # Check cooldown for failed attempts
        if self._is_in_cooldown(session_id):
            return {
                'status': 'cooldown',
                'message': f'Too many failed attempts. Wait {self.spoofing_config["cooldown_seconds"]} seconds.',
                'session_id': session_id
            }
        
        try:
            # Step 1: Face Detection using landmarks
            landmark_result = self.face_detector.detect_landmarks(frame)
            
            if not landmark_result or not landmark_result.get('landmarks'):
                return {
                    'status': 'no_face',
                    'message': 'No face detected. Please position your face clearly.',
                    'session_id': session_id
                }
            
            # Extract face information from landmarks
            landmarks = landmark_result['landmarks']
            face_confidence = landmark_result.get('confidence', 0.0)
            
            # Create face bbox from landmarks for anti-spoofing
            if len(landmarks) > 0:
                # Calculate bounding box from landmarks
                x_coords = [p[0] for p in landmarks]
                y_coords = [p[1] for p in landmarks]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add some padding
                padding = 20
                h, w = frame.shape[:2]
                face_bbox = (
                    max(0, x_min - padding),
                    max(0, y_min - padding),
                    min(w, x_max + padding),
                    min(h, y_max + padding)
                )
            else:
                return {
                    'status': 'no_face',
                    'message': 'Face landmarks not detected properly.',
                    'session_id': session_id
                }
            
            if face_confidence < self.spoofing_config['min_confidence']:
                return {
                    'status': 'low_confidence',
                    'message': 'Face detection confidence too low. Please improve lighting.',
                    'confidence': face_confidence,
                    'session_id': session_id
                }
            
            # Step 2: Advanced Anti-Spoofing Detection
            spoofing_result = self.antispoofing_detector.detect_spoofing(
                frame, 
                face_bbox
            )
            
            spoofing_score = spoofing_result.get('spoofing_probability', 1.0)
            is_live = spoofing_score < self.spoofing_config['spoofing_threshold']
            
            # Update session data
            session['frames_processed'] += 1
            session['spoofing_scores'].append(spoofing_score)
            session['face_confidences'].append(face_confidence)
            
            # Step 3: Multi-frame verification
            if session['frames_processed'] < self.spoofing_config['required_frames']:
                avg_spoofing = np.mean(session['spoofing_scores'])
                progress = session['frames_processed'] / self.spoofing_config['required_frames']
                
                return {
                    'status': 'processing',
                    'message': f'Verifying authenticity... ({session["frames_processed"]}/{self.spoofing_config["required_frames"]})',
                    'progress': progress,
                    'spoofing_score': spoofing_score,
                    'avg_spoofing': avg_spoofing,
                    'confidence': face_confidence,
                    'session_id': session_id,
                    'is_live_frame': is_live
                }
            
            # Step 4: Final verification
            avg_spoofing_score = np.mean(session['spoofing_scores'])
            avg_confidence = np.mean(session['face_confidences'])
            
            # Determine if this is a live person
            final_is_live = avg_spoofing_score < self.spoofing_config['spoofing_threshold']
            
            if final_is_live:
                # Success - Record attendance
                attendance_result = self._record_attendance(
                    landmarks, 
                    avg_spoofing_score, 
                    avg_confidence,
                    spoofing_result
                )
                
                # Clean up session
                del self.session_data[session_id]
                
                return {
                    'status': 'success',
                    'message': 'Attendance recorded successfully!',
                    'user_name': attendance_result.get('user_name', 'Unknown'),
                    'timestamp': attendance_result['timestamp'],
                    'spoofing_score': avg_spoofing_score,
                    'confidence': avg_confidence,
                    'detection_details': spoofing_result,
                    'session_id': session_id,
                    'processing_time': time.time() - session['start_time']
                }
            else:
                # Spoofing detected - Log security event
                self._log_spoofing_attempt(avg_spoofing_score, spoofing_result, session_id)
                self._increment_failed_attempts(session_id)
                
                # Clean up session
                del self.session_data[session_id]
                
                return {
                    'status': 'spoofing_detected',
                    'message': 'Spoofing attack detected! Access denied.',
                    'spoofing_score': avg_spoofing_score,
                    'confidence': avg_confidence,
                    'attack_type': spoofing_result.get('dominant_attack_type', 'unknown'),
                    'detection_details': spoofing_result,
                    'session_id': session_id,
                    'security_alert': True
                }
                
        except Exception as e:
            print(f"❌ Error in attendance processing: {e}")
            return {
                'status': 'error',
                'message': f'Processing error: {str(e)}',
                'session_id': session_id
            }
    
    def _record_attendance(self, face_data, spoofing_score, confidence, spoofing_details):
        """Record successful attendance"""
        with self.db_lock:
            with sqlite3.connect('attendance.db') as conn:
                cursor = conn.cursor()
                
                # For now, create a dummy user (in real system, do face recognition)
                user_name = f"User_{int(time.time())}"  # Replace with actual face recognition
                
                # Insert attendance record
                cursor.execute('''
                    INSERT INTO attendance 
                    (user_id, spoofing_score, confidence, detection_method, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    1,  # Dummy user ID
                    spoofing_score,
                    confidence,
                    json.dumps(spoofing_details.get('method_scores', {})),
                    'success'
                ))
                
                conn.commit()
                
                return {
                    'user_name': user_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
    
    def _log_spoofing_attempt(self, spoofing_score, spoofing_details, session_id):
        """Log spoofing attempt for security analysis"""
        with self.db_lock:
            with sqlite3.connect('attendance.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO spoofing_logs 
                    (spoofing_score, attack_type, confidence, ip_address, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    spoofing_score,
                    spoofing_details.get('dominant_attack_type', 'unknown'),
                    spoofing_details.get('confidence', 0),
                    'localhost',  # In real system, get actual IP
                    'blocked'
                ))
                
                conn.commit()
    
    def _is_in_cooldown(self, session_id):
        """Check if session is in cooldown due to failed attempts"""
        if session_id not in self.failed_attempts:
            return False
        
        attempts_data = self.failed_attempts[session_id]
        if attempts_data['count'] >= self.spoofing_config['max_attempts']:
            time_since_last = time.time() - attempts_data['last_attempt']
            return time_since_last < self.spoofing_config['cooldown_seconds']
        
        return False
    
    def _increment_failed_attempts(self, session_id):
        """Increment failed attempts counter"""
        if session_id not in self.failed_attempts:
            self.failed_attempts[session_id] = {'count': 0, 'last_attempt': 0}
        
        self.failed_attempts[session_id]['count'] += 1
        self.failed_attempts[session_id]['last_attempt'] = time.time()
    
    def get_spoofing_statistics(self):
        """Get anti-spoofing statistics for admin"""
        with sqlite3.connect('attendance.db') as conn:
            cursor = conn.cursor()
            
            # Get recent attendance
            cursor.execute('''
                SELECT COUNT(*) as total_attempts,
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                       AVG(spoofing_score) as avg_spoofing_score
                FROM attendance 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            
            attendance_stats = cursor.fetchone()
            
            # Get spoofing attempts
            cursor.execute('''
                SELECT COUNT(*) as spoofing_attempts,
                       AVG(spoofing_score) as avg_attack_score,
                       attack_type,
                       COUNT(*) as count
                FROM spoofing_logs 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY attack_type
            ''')
            
            spoofing_stats = cursor.fetchall()
            
            return {
                'attendance': {
                    'total_attempts': attendance_stats[0] or 0,
                    'successful': attendance_stats[1] or 0,
                    'avg_spoofing_score': attendance_stats[2] or 0
                },
                'security': {
                    'spoofing_attempts': len(spoofing_stats),
                    'attack_types': spoofing_stats
                }
            }

def main():
    """Main function to run the anti-spoofing attendance system"""
    print("🚀 Starting Anti-Spoofing Attendance System")
    print("Focus: Advanced spoofing detection for secure attendance")
    print("=" * 60)
    
    # Initialize system
    attendance_system = AntiSpoofingAttendanceSystem()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("📷 Camera initialized")
    print("🔒 Anti-spoofing detection active")
    print("\nInstructions:")
    print("- Position your face clearly in front of camera")
    print("- System will verify you are a real person")
    print("- Press 'q' to quit, 's' to show statistics")
    print("=" * 60)
    
    session_id = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame for attendance
        result = attendance_system.process_attendance(frame, session_id)
        
        # Update session ID
        session_id = result.get('session_id')
        
        # Display result on frame
        display_frame = frame.copy()
        _draw_attendance_info(display_frame, result)
        
        # Show frame
        cv2.imshow('Anti-Spoofing Attendance System', display_frame)
        
        # Handle result
        if result['status'] == 'success':
            print(f"✅ Attendance: {result['user_name']} at {result['timestamp']}")
            print(f"   Spoofing Score: {result['spoofing_score']:.3f}")
            session_id = None  # Reset for next person
            
        elif result['status'] == 'spoofing_detected':
            print(f"🚨 SPOOFING DETECTED: {result['attack_type']}")
            print(f"   Score: {result['spoofing_score']:.3f}")
            session_id = None  # Reset for next attempt
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stats = attendance_system.get_spoofing_statistics()
            print("\n📊 Anti-Spoofing Statistics (Last 24h):")
            print(f"   Total Attempts: {stats['attendance']['total_attempts']}")
            print(f"   Successful: {stats['attendance']['successful']}")
            print(f"   Spoofing Attempts: {stats['security']['spoofing_attempts']}")
            print(f"   Avg Spoofing Score: {stats['attendance']['avg_spoofing_score']:.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Anti-Spoofing Attendance System stopped")

def _draw_attendance_info(frame, result):
    """Draw attendance information on frame"""
    height, width = frame.shape[:2]
    
    # Status color
    status_colors = {
        'no_face': (0, 0, 255),        # Red
        'low_confidence': (0, 165, 255),  # Orange
        'processing': (0, 255, 255),    # Yellow
        'success': (0, 255, 0),         # Green
        'spoofing_detected': (0, 0, 255),  # Red
        'cooldown': (128, 0, 128),      # Purple
        'error': (0, 0, 255)            # Red
    }
    
    status = result['status']
    color = status_colors.get(status, (255, 255, 255))
    
    # Draw status box
    cv2.rectangle(frame, (10, 10), (width - 10, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (width - 10, 120), color, 2)
    
    # Status text
    cv2.putText(frame, f"Status: {status.upper()}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Message
    message = result.get('message', '')
    cv2.putText(frame, message, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Additional info
    if 'spoofing_score' in result:
        score_text = f"Spoofing Score: {result['spoofing_score']:.3f}"
        cv2.putText(frame, score_text, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if 'progress' in result:
        # Progress bar
        progress = result['progress']
        bar_width = int((width - 40) * progress)
        cv2.rectangle(frame, (20, 130), (20 + bar_width, 150), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 130), (width - 20, 150), (255, 255, 255), 1)

if __name__ == "__main__":
    main()

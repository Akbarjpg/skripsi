"""
MINIMAL ANTI-SPOOFING ATTENDANCE SYSTEM
======================================
Focus: Core anti-spoofing without complex dependencies
Uses basic OpenCV face detection + simple spoofing checks
"""

import cv2
import numpy as np
import sqlite3
import json
import time
from datetime import datetime
from threading import Lock

class MinimalAntiSpoofingSystem:
    """
    Minimal Anti-Spoofing System for Attendance
    Focus: Simple but effective spoofing detection
    """
    
    def __init__(self):
        print("🔒 Initializing Minimal Anti-Spoofing System")
        
        # OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Database
        self.db_lock = Lock()
        self.init_database()
        
        # Anti-spoofing configuration
        self.config = {
            'min_face_size': (50, 50),
            'max_faces': 1,
            'spoofing_threshold': 0.6,
            'required_frames': 3,
            'movement_threshold': 10,
            'texture_threshold': 50
        }
        
        # Session tracking
        self.sessions = {}
        
        print("✅ Minimal Anti-Spoofing System Ready")
    
    def init_database(self):
        """Initialize attendance database"""
        with sqlite3.connect('attendance.db') as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS simple_attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    spoofing_score REAL,
                    face_size INTEGER,
                    movement_score REAL,
                    texture_score REAL,
                    status TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS spoofing_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    spoofing_score REAL,
                    detection_reason TEXT,
                    status TEXT
                )
            ''')
            
            conn.commit()
    
    def detect_face(self, frame):
        """Simple face detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=self.config['min_face_size']
        )
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        return {
            'bbox': (x, y, w, h),
            'confidence': min(1.0, (w * h) / 10000),  # Simple confidence based on size
            'face_roi': gray[y:y+h, x:x+w]
        }
    
    def detect_spoofing(self, frame, face_data, session_id):
        """
        Simple but effective spoofing detection
        Combines multiple basic techniques
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'frames': [],
                'face_positions': [],
                'texture_scores': [],
                'movement_scores': []
            }
        
        session = self.sessions[session_id]
        face_roi = face_data['face_roi']
        x, y, w, h = face_data['bbox']
        
        # 1. Texture Analysis (detect printed photos)
        texture_score = self._analyze_texture(face_roi)
        session['texture_scores'].append(texture_score)
        
        # 2. Movement Detection (detect static images)
        movement_score = self._detect_movement(face_data['bbox'], session)
        session['movement_scores'].append(movement_score)
        
        # 3. Size Consistency (detect screen displays)
        size_score = self._check_size_consistency(w, h, session)
        
        # 4. Edge Analysis (detect screen bezels/frames)
        edge_score = self._analyze_edges(face_roi)
        
        # Combine scores
        spoofing_probability = self._calculate_spoofing_probability(
            texture_score, movement_score, size_score, edge_score
        )
        
        # Store frame data
        session['frames'].append(frame.copy())
        session['face_positions'].append((x, y, w, h))
        
        # Keep only recent frames
        if len(session['frames']) > self.config['required_frames']:
            session['frames'].pop(0)
            session['face_positions'].pop(0)
            session['texture_scores'].pop(0)
            session['movement_scores'].pop(0)
        
        is_live = spoofing_probability < self.config['spoofing_threshold']
        
        return {
            'is_live': is_live,
            'spoofing_probability': spoofing_probability,
            'scores': {
                'texture': texture_score,
                'movement': movement_score,
                'size': size_score,
                'edge': edge_score
            },
            'frames_processed': len(session['frames'])
        }
    
    def _analyze_texture(self, face_roi):
        """Analyze texture to detect printed photos"""
        if face_roi.size == 0:
            return 1.0  # High spoofing score for invalid ROI
        
        # Calculate texture variance (real faces have more texture variation)
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize variance (higher variance = more likely real)
        texture_score = max(0, 1.0 - (variance / 1000))
        return min(1.0, texture_score)
    
    def _detect_movement(self, current_bbox, session):
        """Detect natural head movement"""
        if len(session['face_positions']) < 2:
            return 0.5  # Neutral score for insufficient data
        
        # Calculate movement between frames
        prev_x, prev_y, prev_w, prev_h = session['face_positions'][-1]
        curr_x, curr_y, curr_w, curr_h = current_bbox
        
        movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        # Too little movement suggests static image
        if movement < 2:
            return 0.8  # High spoofing score
        # Too much movement suggests unnatural behavior
        elif movement > 50:
            return 0.7  # Moderate spoofing score
        else:
            return max(0, 0.6 - (movement / 100))  # Lower score for natural movement
    
    def _check_size_consistency(self, w, h, session):
        """Check if face size changes naturally"""
        if len(session['face_positions']) < 2:
            return 0.3  # Low spoofing score for insufficient data
        
        # Calculate size variation
        sizes = [(fw * fh) for fx, fy, fw, fh in session['face_positions']]
        current_size = w * h
        sizes.append(current_size)
        
        # Real faces have some size variation due to natural movement
        if len(sizes) > 1:
            size_var = np.var(sizes) / np.mean(sizes)
            # Too consistent size suggests screen display
            if size_var < 0.01:
                return 0.7  # High spoofing score
            else:
                return max(0, 0.4 - size_var)
        
        return 0.3
    
    def _analyze_edges(self, face_roi):
        """Analyze edges to detect screen displays"""
        if face_roi.size == 0:
            return 1.0
        
        # Apply edge detection
        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Very high edge density might indicate screen pixels
        if edge_density > 0.3:
            return 0.8  # High spoofing score
        # Very low edge density might indicate poor quality photo
        elif edge_density < 0.05:
            return 0.6  # Moderate spoofing score
        else:
            return max(0, 0.4 - edge_density)
    
    def _calculate_spoofing_probability(self, texture, movement, size, edge):
        """Calculate final spoofing probability"""
        # Weighted combination of scores
        weights = {
            'texture': 0.4,    # Most important
            'movement': 0.3,   # Second most important
            'size': 0.2,       # Less important
            'edge': 0.1        # Supporting evidence
        }
        
        spoofing_score = (
            weights['texture'] * texture +
            weights['movement'] * movement +
            weights['size'] * size +
            weights['edge'] * edge
        )
        
        return min(1.0, max(0.0, spoofing_score))
    
    def process_attendance(self, frame, session_id=None):
        """Process attendance with anti-spoofing"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Detect face
        face_data = self.detect_face(frame)
        
        if face_data is None:
            return {
                'status': 'no_face',
                'message': 'No face detected',
                'session_id': session_id
            }
        
        # Check spoofing
        spoofing_result = self.detect_spoofing(frame, face_data, session_id)
        
        frames_needed = self.config['required_frames']
        frames_processed = spoofing_result['frames_processed']
        
        if frames_processed < frames_needed:
            return {
                'status': 'processing',
                'message': f'Analyzing... ({frames_processed}/{frames_needed})',
                'progress': frames_processed / frames_needed,
                'spoofing_score': spoofing_result['spoofing_probability'],
                'scores': spoofing_result['scores'],
                'session_id': session_id
            }
        
        # Final decision
        if spoofing_result['is_live']:
            # Record attendance
            self._record_attendance(spoofing_result)
            
            # Clear session
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            return {
                'status': 'success',
                'message': 'Attendance recorded successfully!',
                'spoofing_score': spoofing_result['spoofing_probability'],
                'scores': spoofing_result['scores'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'session_id': session_id
            }
        else:
            # Spoofing detected
            self._log_spoofing_attempt(spoofing_result)
            
            # Clear session
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            return {
                'status': 'spoofing_detected',
                'message': 'Spoofing detected! Access denied.',
                'spoofing_score': spoofing_result['spoofing_probability'],
                'scores': spoofing_result['scores'],
                'session_id': session_id
            }
    
    def _record_attendance(self, spoofing_result):
        """Record successful attendance"""
        with self.db_lock:
            with sqlite3.connect('attendance.db') as conn:
                cursor = conn.cursor()
                
                scores = spoofing_result['scores']
                cursor.execute('''
                    INSERT INTO simple_attendance 
                    (spoofing_score, face_size, movement_score, texture_score, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    spoofing_result['spoofing_probability'],
                    scores.get('size', 0),
                    scores.get('movement', 0),
                    scores.get('texture', 0),
                    'success'
                ))
                
                conn.commit()
    
    def _log_spoofing_attempt(self, spoofing_result):
        """Log spoofing attempt"""
        with self.db_lock:
            with sqlite3.connect('attendance.db') as conn:
                cursor = conn.cursor()
                
                scores = spoofing_result['scores']
                reason = max(scores.items(), key=lambda x: x[1])
                
                cursor.execute('''
                    INSERT INTO spoofing_alerts 
                    (spoofing_score, detection_reason, status)
                    VALUES (?, ?, ?)
                ''', (
                    spoofing_result['spoofing_probability'],
                    f"{reason[0]}: {reason[1]:.3f}",
                    'blocked'
                ))
                
                conn.commit()
    
    def get_statistics(self):
        """Get system statistics"""
        with sqlite3.connect('attendance.db') as conn:
            cursor = conn.cursor()
            
            # Attendance stats
            cursor.execute('''
                SELECT COUNT(*) as total,
                       AVG(spoofing_score) as avg_score
                FROM simple_attendance 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            attendance_stats = cursor.fetchone()
            
            # Spoofing stats
            cursor.execute('''
                SELECT COUNT(*) as attempts,
                       AVG(spoofing_score) as avg_score
                FROM spoofing_alerts 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            spoofing_stats = cursor.fetchone()
            
            return {
                'attendance': {
                    'total': attendance_stats[0] or 0,
                    'avg_score': attendance_stats[1] or 0
                },
                'security': {
                    'spoofing_attempts': spoofing_stats[0] or 0,
                    'avg_attack_score': spoofing_stats[1] or 0
                }
            }

def main():
    """Main function for minimal anti-spoofing system"""
    print("🚀 Starting Minimal Anti-Spoofing Attendance System")
    print("Focus: Simple but effective spoofing detection")
    print("=" * 60)
    
    # Initialize system
    system = MinimalAntiSpoofingSystem()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("📷 Camera ready")
    print("🔒 Anti-spoofing detection active")
    print("\nControls:")
    print("- Press SPACE to start new session")
    print("- Press 's' to show statistics")
    print("- Press 'q' to quit")
    print("=" * 60)
    
    session_id = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame if session active
        if session_id:
            result = system.process_attendance(frame, session_id)
            
            # Draw results on frame
            _draw_results(frame, result)
            
            # Handle results
            if result['status'] in ['success', 'spoofing_detected']:
                print(f"\n{'✅' if result['status'] == 'success' else '🚨'} {result['message']}")
                print(f"   Spoofing Score: {result['spoofing_score']:.3f}")
                print(f"   Scores: {result['scores']}")
                session_id = None  # Reset session
        else:
            # Draw instructions
            cv2.putText(frame, "Press SPACE to start detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Minimal Anti-Spoofing System', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and session_id is None:
            session_id = f"session_{int(time.time())}"
            print(f"\n🔍 Starting new detection session: {session_id}")
        elif key == ord('s'):
            stats = system.get_statistics()
            print(f"\n📊 Statistics (24h):")
            print(f"   Attendance: {stats['attendance']['total']}")
            print(f"   Spoofing Attempts: {stats['security']['spoofing_attempts']}")
            print(f"   Avg Security Score: {stats['attendance']['avg_score']:.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 System stopped")

def _draw_results(frame, result):
    """Draw detection results on frame"""
    status = result['status']
    
    # Status colors
    colors = {
        'no_face': (0, 0, 255),           # Red
        'processing': (0, 255, 255),      # Yellow
        'success': (0, 255, 0),           # Green
        'spoofing_detected': (0, 0, 255)  # Red
    }
    
    color = colors.get(status, (255, 255, 255))
    
    # Draw status
    cv2.putText(frame, f"Status: {status.upper()}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw message
    message = result.get('message', '')
    cv2.putText(frame, message, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw spoofing score
    if 'spoofing_score' in result:
        score_text = f"Spoofing: {result['spoofing_score']:.3f}"
        cv2.putText(frame, score_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw progress bar
    if 'progress' in result:
        progress = result['progress']
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (10, 110), (10 + bar_width, 130), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 110), (410, 130), (255, 255, 255), 1)

if __name__ == "__main__":
    main()

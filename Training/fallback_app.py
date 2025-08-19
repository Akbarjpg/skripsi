"""
Fallback Web Application - No Heavy Dependencies
Simplified version that avoids numpy conflicts
"""

from flask import Flask, render_template, request, jsonify, redirect
import os
import base64
import time
import json
import sqlite3
from datetime import datetime

# Configure template and static folders
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'web', 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'web', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'your-secret-key-change-this'

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verification_method TEXT,
            confidence_score REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/attendance')
def attendance_page():
    """Attendance verification page"""
    return render_template('attendance.html')

@app.route('/api/register_user', methods=['POST'])
def register_user():
    """Register new user"""
    try:
        data = request.get_json()
        username = data.get('username')
        name = data.get('name')
        
        if not username or not name:
            return jsonify({'error': 'Username and name required'}), 400
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO users (username, name) VALUES (?, ?)',
            (username, name)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User registered successfully'})
    
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify_attendance', methods=['POST'])
def verify_attendance():
    """Simplified attendance verification"""
    try:
        data = request.get_json()
        username = data.get('username')
        
        if not username:
            return jsonify({'error': 'Username required'}), 400
        
        # Simplified verification (no AI processing)
        # In production, this would integrate with the AI models
        verification_result = {
            'success': True,
            'confidence': 0.85,
            'method': 'manual_verification',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        user_result = cursor.fetchone()
        
        if not user_result:
            return jsonify({'error': 'User not found'}), 404
        
        user_id = user_result[0]
        
        # Save attendance
        cursor.execute(
            'INSERT INTO attendance (user_id, verification_method, confidence_score) VALUES (?, ?, ?)',
            (user_id, verification_result['method'], verification_result['confidence'])
        )
        
        conn.commit()
        conn.close()
        
        return jsonify(verification_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance_history')
def attendance_history():
    """Get attendance history"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.name, u.username, a.timestamp, a.verification_method, a.confidence_score
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.timestamp DESC
            LIMIT 50
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        history = []
        for row in results:
            history.append({
                'name': row[0],
                'username': row[1],
                'timestamp': row[2],
                'method': row[3],
                'confidence': row[4]
            })
        
        return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin')
def admin_page():
    """Admin dashboard"""
    return render_template('admin.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    print("="*50)
    print("FALLBACK WEB APPLICATION STARTING")
    print("="*50)
    print("✓ No heavy AI dependencies")
    print("✓ Basic attendance tracking")
    print("✓ SQLite database")
    print("✓ User registration")
    print("✓ Manual verification")
    print()
    print("🌐 Starting server...")
    print("📍 Access: http://localhost:5000")
    print("⚠️  Note: AI features disabled due to numpy conflicts")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

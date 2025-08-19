"""
SIMPLE ANTI-SPOOFING WEB INTERFACE
==================================
Focus: Core anti-spoofing attendance (no user login required)
Admin panel: Separate login for management
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import json
from attendance_antispoofing import AntiSpoofingAttendanceSystem
import os

app = Flask(__name__)
app.secret_key = 'antispoofing_secret_key_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize anti-spoofing system
attendance_system = AntiSpoofingAttendanceSystem()

# Simple admin credentials (in production, use proper authentication)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "antispoofing123"

@app.route('/')
def index():
    """Main attendance page - no login required"""
    return render_template('attendance_main.html')

@app.route('/admin')
def admin_login():
    """Admin login page"""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login_post():
    """Handle admin login"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        return redirect(url_for('admin_dashboard'))
    else:
        return render_template('admin_login.html', error="Invalid credentials")

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard - requires login"""
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    # Get anti-spoofing statistics
    stats = attendance_system.get_spoofing_statistics()
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    return redirect(url_for('index'))

@socketio.on('frame')
def handle_frame(data):
    """Handle camera frame for anti-spoofing detection"""
    try:
        # Decode frame
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('result', {'status': 'error', 'message': 'Invalid frame data'})
            return
        
        # Get session ID
        session_id = data.get('session_id')
        
        # Process with anti-spoofing system
        result = attendance_system.process_attendance(frame, session_id)
        
        # Send result back to client
        emit('result', result)
        
    except Exception as e:
        emit('result', {
            'status': 'error', 
            'message': f'Processing error: {str(e)}'
        })

@socketio.on('reset_session')
def handle_reset_session():
    """Reset the current session"""
    emit('result', {
        'status': 'reset',
        'message': 'Session reset. Ready for next person.'
    })

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Simple Anti-Spoofing Web Interface")
    print("=" * 50)
    print("📱 User Interface: http://127.0.0.1:5000 (No login required)")
    print("🔒 Admin Panel: http://127.0.0.1:5000/admin")
    print(f"   Username: {ADMIN_USERNAME}")
    print(f"   Password: {ADMIN_PASSWORD}")
    print("=" * 50)
    
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)

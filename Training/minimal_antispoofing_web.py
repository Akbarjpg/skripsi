"""
MINIMAL ANTI-SPOOFING WEB INTERFACE
==================================
Focus: Core anti-spoofing with simple web interface
No complex dependencies, just effective spoofing detection
"""

from flask import Flask, render_template, request, jsonify, session as flask_session, redirect
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import json
from minimal_antispoofing import MinimalAntiSpoofingSystem
import os

app = Flask(__name__)
app.secret_key = 'minimal_antispoofing_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize minimal anti-spoofing system
antispoofing_system = MinimalAntiSpoofingSystem()

# Simple admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "antispoofing123"

@app.route('/')
def index():
    """Main attendance page - no login required for users"""
    return render_template('minimal_attendance.html')

@app.route('/admin')
def admin_login():
    """Admin login page"""
    if flask_session.get('admin_logged_in'):
        return redirect('/admin/dashboard')
    return render_template('minimal_admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login_post():
    """Handle admin login"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        flask_session['admin_logged_in'] = True
        return redirect('/admin/dashboard')
    else:
        return render_template('minimal_admin_login.html', error="Invalid credentials")

@app.route('/admin/dashboard')
def admin_dashboard():
    """Admin dashboard"""
    if not flask_session.get('admin_logged_in'):
        return redirect('/admin')
    
    stats = antispoofing_system.get_statistics()
    return render_template('minimal_admin_dashboard.html', stats=stats)

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    flask_session.pop('admin_logged_in', None)
    return redirect('/')

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
        
        # Process with minimal anti-spoofing system
        result = antispoofing_system.process_attendance(frame, session_id)
        
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
    
    print("🚀 Starting Minimal Anti-Spoofing Web Interface")
    print("=" * 50)
    print("📱 User Interface: http://127.0.0.1:5000")
    print("🔒 Admin Panel: http://127.0.0.1:5000/admin")
    print(f"   Username: {ADMIN_USERNAME}")
    print(f"   Password: {ADMIN_PASSWORD}")
    print("\n🎯 FOCUSED FEATURES:")
    print("   ✅ Simple face detection (OpenCV)")
    print("   ✅ Texture analysis (photo detection)")
    print("   ✅ Movement detection (static image)")
    print("   ✅ Size consistency (screen display)")
    print("   ✅ Edge analysis (screen bezels)")
    print("=" * 50)
    
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)

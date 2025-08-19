#!/usr/bin/env python3
"""
Simplified Face Registration System Test
"""

import sqlite3
import json
import sys
import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO, emit
import base64
import numpy as np
from PIL import Image
import io

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition imported successfully")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("❌ face_recognition not available")

def create_simple_face_app():
    """Create a simplified face registration app for testing"""
    app = Flask(__name__, template_folder='src/web/templates')
    app.secret_key = 'face-registration-test-key'
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize database
    def init_db():
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Create users table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user'
        )
        ''')
        
        # Create face_data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            face_position TEXT NOT NULL,
            face_encoding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, face_position),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Insert admin user if not exists
        cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if not cursor.fetchone():
            cursor.execute('''
            INSERT INTO users (username, password, full_name, role) 
            VALUES (?, ?, ?, ?)
            ''', ('admin', 'admin', 'Administrator', 'admin'))
        
        # Insert test user if not exists
        cursor.execute('SELECT id FROM users WHERE username = ?', ('testuser',))
        if not cursor.fetchone():
            cursor.execute('''
            INSERT INTO users (username, password, full_name, role) 
            VALUES (?, ?, ?, ?)
            ''', ('testuser', 'password', 'Test User', 'user'))
        
        conn.commit()
        conn.close()
    
    init_db()
    
    @app.route('/')
    def index():
        return '''
        <html>
        <head><title>Face Registration Test</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>🧪 Face Registration System Test</h1>
            <p>This is a simplified test version of the face registration system.</p>
            
            <h3>Test Accounts:</h3>
            <ul>
                <li><strong>Admin:</strong> username=admin, password=admin</li>
                <li><strong>User:</strong> username=testuser, password=password</li>
            </ul>
            
            <h3>Available Routes:</h3>
            <ul>
                <li><a href="/login">Login</a></li>
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/register-face">Face Registration</a></li>
                <li><a href="/check-face-registered">Check Face Status API</a></li>
            </ul>
            
            <h3>System Status:</h3>
            <ul>
                <li>Face Recognition: ''' + ('✅ Available' if FACE_RECOGNITION_AVAILABLE else '❌ Not Available') + '''</li>
                <li>Database: ✅ Connected</li>
                <li>SocketIO: ✅ Active</li>
            </ul>
        </body>
        </html>
        '''
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            # Special handling for admin
            if username == 'admin' and password == 'admin':
                session['user_id'] = 0
                session['username'] = 'admin'
                session['full_name'] = 'Administrator'
                session['role'] = 'admin'
                conn.close()
                return redirect(url_for('dashboard'))
            
            # Regular user login
            cursor.execute('SELECT id, username, full_name, role FROM users WHERE username = ? AND password = ?', 
                         (username, password))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['full_name'] = user[2]
                session['role'] = user[3]
                return redirect(url_for('dashboard'))
            else:
                return '<p>Invalid credentials. <a href="/login">Try again</a></p>'
        
        return '''
        <html>
        <head><title>Login - Face Registration Test</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>Login</h2>
            <form method="post">
                <p>Username: <input type="text" name="username" value="testuser"></p>
                <p>Password: <input type="password" name="password" value="password"></p>
                <p><input type="submit" value="Login"></p>
            </form>
            <p><small>Test accounts: admin/admin or testuser/password</small></p>
        </body>
        </html>
        '''
    
    @app.route('/dashboard')
    def dashboard():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
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
                conn = sqlite3.connect('attendance.db')
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT face_position FROM face_data WHERE user_id = ?', (user_data['id'],))
                positions = [row[0] for row in cursor.fetchall()]
                user_has_face_data = len(positions) >= 3 and all(pos in positions for pos in ['front', 'left', 'right'])
                conn.close()
            except Exception as e:
                print(f"Error checking face data: {e}")
        else:
            user_has_face_data = True  # Admin doesn't need face registration
        
        return f'''
        <html>
        <head><title>Dashboard - Face Registration Test</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>Dashboard</h2>
            <p>Welcome, <strong>{user_data['full_name'] or user_data['username']}</strong></p>
            <p>Role: {user_data['role']}</p>
            
            <h3>Face Registration Status:</h3>
            <p>Status: ''' + ('✅ Complete' if user_has_face_data else '❌ Not Registered') + '''</p>
            
            <h3>Actions:</h3>
            <ul>
                <li><a href="/register-face">''' + ('Update Face Data' if user_has_face_data else 'Register Face') + '''</a></li>
                <li><a href="/check-face-registered">Check Face Status (API)</a></li>
                <li><a href="/">Back to Home</a></li>
            </ul>
        </body>
        </html>
        '''
    
    @app.route('/register-face')
    def register_face():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        return render_template('register_face.html', user={'id': session.get('user_id')})
    
    @app.route('/check-face-registered')
    def check_face_registered():
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        
        user_id = session.get('user_id')
        if user_id == 0:  # Admin
            return jsonify({'registered': True, 'message': 'Admin account - face registration not required'})
        
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT face_position FROM face_data WHERE user_id = ?', (user_id,))
            positions = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            required_positions = ['front', 'left', 'right']
            registered = len(positions) >= 3 and all(pos in positions for pos in required_positions)
            
            return jsonify({
                'registered': registered,
                'positions': positions,
                'missing': [pos for pos in required_positions if pos not in positions]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @socketio.on('capture_face')
    def handle_capture_face(data):
        """Handle face capture event"""
        if not FACE_RECOGNITION_AVAILABLE:
            emit('face_capture_result', {'status': 'error', 'message': 'face_recognition library not available'})
            return
        
        try:
            user_id = data.get('user_id')
            position = data.get('position')
            image_data = data.get('image')
            
            if not all([user_id, position, image_data]):
                emit('face_capture_result', {'status': 'error', 'message': 'Data tidak lengkap'})
                return
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Convert RGB to BGR for face_recognition
            if len(image_array.shape) == 3:
                image_array = image_array[:, :, ::-1]
            
            # Extract face encodings
            face_locations = face_recognition.face_locations(image_array)
            
            if len(face_locations) == 0:
                emit('face_capture_result', {'status': 'error', 'message': 'Wajah tidak terdeteksi'})
                return
            
            if len(face_locations) > 1:
                emit('face_capture_result', {'status': 'error', 'message': 'Terdeteksi lebih dari satu wajah'})
                return
            
            face_encodings = face_recognition.face_encodings(image_array, face_locations)
            
            if len(face_encodings) == 0:
                emit('face_capture_result', {'status': 'error', 'message': 'Gagal mengekstrak fitur wajah'})
                return
            
            face_encoding = face_encodings[0]
            
            # Save to database
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            # Check if position already exists
            cursor.execute('SELECT id FROM face_data WHERE user_id = ? AND face_position = ?', (user_id, position))
            existing = cursor.fetchone()
            
            # Convert encoding to JSON string
            encoding_json = json.dumps(face_encoding.tolist())
            
            if existing:
                cursor.execute('''UPDATE face_data 
                                 SET face_encoding = ?, updated_at = CURRENT_TIMESTAMP 
                                 WHERE user_id = ? AND face_position = ?''',
                              (encoding_json, user_id, position))
            else:
                cursor.execute('''INSERT INTO face_data (user_id, face_position, face_encoding) 
                                 VALUES (?, ?, ?)''',
                              (user_id, position, encoding_json))
            
            conn.commit()
            conn.close()
            
            emit('face_capture_result', {'status': 'success', 'message': f'Wajah posisi {position} berhasil direkam'})
            
        except Exception as e:
            print(f"Error in capture_face: {e}")
            emit('face_capture_result', {'status': 'error', 'message': 'Terjadi kesalahan saat memproses gambar'})
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected to face registration test')
    
    return app, socketio

if __name__ == '__main__':
    print("🚀 Starting Face Registration Test System")
    app, socketio = create_simple_face_app()
    print("✅ System ready at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)

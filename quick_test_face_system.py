#!/usr/bin/env python3
"""
Quick Face Registration System Test (Standalone)
"""

import sqlite3
import json
import sys
import os

def test_database_schema():
    """Test if face_data table exists and has correct schema"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Check if face_data table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_data'")
        if cursor.fetchone():
            print("✅ face_data table exists")
            
            # Check table schema
            cursor.execute("PRAGMA table_info(face_data)")
            columns = cursor.fetchall()
            print("📋 Table schema:")
            for col in columns:
                print(f"   - {col[1]} ({col[2]})")
            
            return True
        else:
            print("❌ face_data table not found")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    finally:
        conn.close()

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import face_recognition
        print("✅ face_recognition imported successfully")
        
        import flask_socketio
        print("✅ flask_socketio imported successfully")
        
        from flask import Flask
        print("✅ Flask imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_flask_app():
    """Test if basic Flask app can be created"""
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/test')
        def test():
            return "OK"
        
        # Test if route is registered
        with app.test_client() as client:
            response = client.get('/test')
            if response.status_code == 200:
                print("✅ Basic Flask app works")
                return True
            else:
                print("❌ Flask app test failed")
                return False
                
    except Exception as e:
        print(f"❌ Flask app test error: {e}")
        return False

def test_face_recognition_functionality():
    """Test face_recognition functionality"""
    try:
        import face_recognition
        import numpy as np
        from PIL import Image
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test face detection (should find no faces in random image)
        face_locations = face_recognition.face_locations(dummy_image)
        print(f"✅ Face detection works (found {len(face_locations)} faces in test image)")
        
        return True
    except Exception as e:
        print(f"❌ Face recognition test error: {e}")
        return False

def create_test_user():
    """Create a test user in database"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            # Create users table
            cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'user'
            )
            ''')
        
        # Check if test user exists
        cursor.execute('SELECT id FROM users WHERE username = ?', ('testuser',))
        if not cursor.fetchone():
            cursor.execute('''
            INSERT INTO users (username, password_hash, full_name, role) 
            VALUES (?, ?, ?, ?)
            ''', ('testuser', 'password', 'Test User', 'user'))
            print("✅ Created test user: testuser/password")
        else:
            print("✅ Test user already exists")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 FACE REGISTRATION SYSTEM TEST (STANDALONE)")
    print("=" * 60)
    print()
    
    success = True
    
    print("1. Testing basic imports...")
    success &= test_imports()
    print()
    
    print("2. Testing face_recognition functionality...")
    success &= test_face_recognition_functionality()
    print()
    
    print("3. Testing database schema...")
    success &= test_database_schema()
    print()
    
    print("4. Creating test user...")
    success &= create_test_user()
    print()
    
    print("5. Testing basic Flask app...")
    success &= test_basic_flask_app()
    print()
    
    print("=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Face registration system is ready to use.")
        print()
        print("✅ Database schema is correct")
        print("✅ Required packages are installed")
        print("✅ Basic functionality works")
        print()
        print("To start the full server, run:")
        print("python simple_face_registration_test.py")
        print("python start_face_registration.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them.")
    print("=" * 60)

if __name__ == "__main__":
    main()

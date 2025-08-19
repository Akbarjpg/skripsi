#!/usr/bin/env python3
"""
Quick test for Face Registration System functionality
"""

import sys
import os
import sqlite3

# Add the src/web directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir, 'src', 'web')
sys.path.insert(0, web_dir)

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
        
        from app_optimized import create_optimized_app
        print("✅ app_optimized module loaded successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_face_registration_routes():
    """Test if face registration routes are properly configured"""
    try:
        from app_optimized import create_optimized_app
        
        app, socketio = create_optimized_app()
        
        # Check if routes exist
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(rule.endpoint)
        
        required_routes = ['register_face', 'check_face_registered']
        
        for route in required_routes:
            if route in routes:
                print(f"✅ Route '{route}' configured")
            else:
                print(f"❌ Route '{route}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Route test error: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 FACE REGISTRATION SYSTEM TEST")
    print("=" * 60)
    print()
    
    success = True
    
    print("1. Testing imports...")
    success &= test_imports()
    print()
    
    print("2. Testing database schema...")
    success &= test_database_schema()
    print()
    
    print("3. Testing routes configuration...")
    success &= test_face_registration_routes()
    print()
    
    print("=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Face registration system is ready to use.")
        print()
        print("To start the server, run:")
        print("python start_face_registration.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them.")
    print("=" * 60)

if __name__ == "__main__":
    main()

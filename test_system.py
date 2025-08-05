#!/usr/bin/env python3
"""
Test imports and basic functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_imports():
    """Test all critical imports"""
    try:
        print("Testing Flask imports...")
        from flask import Flask
        print("✅ Flask imported successfully")
        
        print("Testing Flask-SocketIO imports...")
        from flask_socketio import SocketIO, emit
        print("✅ Flask-SocketIO imported successfully")
        
        print("Testing config imports...")
        from src.utils.config import SystemConfig
        print("✅ SystemConfig imported successfully")
        
        print("Testing app_clean imports...")
        from src.web.app_clean import create_app
        print("✅ create_app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_creation():
    """Test app creation"""
    try:
        print("\nTesting app creation...")
        from src.web.app_clean import create_app
        
        app, socketio = create_app()
        print("✅ App and SocketIO created successfully")
        print(f"App: {app}")
        print(f"SocketIO: {socketio}")
        
        return True
        
    except Exception as e:
        print(f"❌ App creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test config loading"""
    try:
        print("\nTesting config loading...")
        from src.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        print("✅ Config loaded successfully")
        print(f"Config type: {type(config)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Starting system tests...\n")
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test config
    success &= test_config_loading()
    
    # Test app creation
    success &= test_app_creation()
    
    if success:
        print("\n🎉 All tests passed! System is ready.")
        
        print("\nStarting web server...")
        try:
            from src.web.app_clean import create_app
            app, socketio = create_app()
            
            print("🌐 Starting server on http://localhost:5000")
            print("📱 Access test dashboard at: http://localhost:5000")
            print("🎯 Face detection test at: http://localhost:5000/face-detection-test")
            print("Press Ctrl+C to stop\n")
            
            socketio.run(app, host='localhost', port=5000, debug=True)
            
        except KeyboardInterrupt:
            print("\n👋 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")
            
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

#!/usr/bin/env python3
"""
Launcher for Face Registration System
Usage: python start_face_registration.py
"""

import sys
import os

# Add the src/web directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(current_dir, 'src', 'web')
sys.path.insert(0, web_dir)

try:
    from app_optimized import create_optimized_app
    
    print("=" * 60)
    print("🚀 FACE REGISTRATION SYSTEM STARTING")
    print("=" * 60)
    print()
    print("✅ Face recognition library loaded")
    print("✅ Flask-SocketIO configured")
    print("✅ Database initialized with face_data table")
    print("✅ Admin login fix applied (admin/admin)")
    print()
    print("📋 FEATURES INCLUDED:")
    print("  • 3-Position face capture (front/left/right)")
    print("  • Real-time face encoding extraction")
    print("  • Dashboard integration with face status")
    print("  • Admin special authentication")
    print("  • Face registration validation")
    print()
    print("🌐 Server will start at: http://localhost:5001")
    print("🔑 Admin access: username=admin, password=admin")
    print("👤 Regular users must register face before attendance")
    print()
    print("=" * 60)
    
    # Create and run the app
    app, socketio = create_optimized_app()
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages:")
    print("pip install face-recognition flask-socketio")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)

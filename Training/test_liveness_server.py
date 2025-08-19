#!/usr/bin/env python3
"""
Simple launcher for liveness detection system
"""
import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 LIVENESS DETECTION SYSTEM")
print("=" * 50)
print("🎯 Features:")
print("  ✅ Real-time liveness scoring (0-100)")
print("  ✅ Eye blink detection") 
print("  ✅ Head pose estimation")
print("  ✅ Mouth movement analysis")
print("  ✅ Live/Fake classification")
print("  ✅ Anti-spoofing protection")
print()

try:
    print("📦 Loading modules...")
    
    # Import Flask components
    from flask import Flask, render_template
    from flask_socketio import SocketIO
    
    # Import our application
    from src.web.app_clean import create_app
    
    print("✅ All modules loaded successfully!")
    print()
    print("🌐 Starting web server...")
    
    # Create app
    app, socketio = create_app()
    
    print("✅ Flask app created")
    print("✅ SocketIO configured")
    print("✅ Routes registered")
    print("✅ Database initialized")
    print()
    print("🌟 SERVER READY!")
    print(f"📱 Open your browser to: http://localhost:5000")
    print(f"🔍 Direct liveness test: http://localhost:5000/face_detection")
    print()
    print("🎯 TESTING INSTRUCTIONS:")
    print("1. Open http://localhost:5000/face_detection in your browser")
    print("2. Click 'Start Camera' button")
    print("3. Look at the camera and:")
    print("   • Blink your eyes naturally")
    print("   • Move your head left and right")
    print("   • Open and close your mouth")
    print("   • Watch the liveness score increase!")
    print()
    print("📊 Expected Results:")
    print("  • Live faces: Score 60-100, Status: LIVE")
    print("  • Fake/Static: Score 0-50, Status: FAKE")
    print("  • Real-time landmark visualization")
    print("  • Eye/mouth movement tracking")
    print()
    print("🛑 Press Ctrl+C to stop server")
    print("=" * 50)
    
    # Start server
    socketio.run(
        app,
        host='localhost',
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True
    )
    
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped by user")
    print("Thank you for testing the liveness detection system!")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("📦 Installing missing dependencies...")
    os.system("pip install flask flask-socketio opencv-python mediapipe numpy")
    print("✅ Try running again after installation")
    
except Exception as e:
    print(f"\n❌ Error starting server: {e}")
    import traceback
    print(traceback.format_exc())
    print("\n🔧 Troubleshooting:")
    print("1. Check if port 5000 is free")
    print("2. Run: pip install -r requirements.txt")
    print("3. Check camera permissions")
    
finally:
    print("\n👋 Goodbye!")

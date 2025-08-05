#!/usr/bin/env python3
"""
Start the Flask server for liveness detection demo
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.web.app_clean import create_app
from src.utils.config import get_default_config

def start_server():
    print("🚀 STARTING LIVENESS DETECTION SERVER")
    print("=" * 50)
    
    try:
        # Create default config
        config = get_default_config()
        config.web.debug = True
        config.web.host = "localhost"
        config.web.port = 5000
        
        # Create app
        app, socketio = create_app(config)
        
        print("✅ Flask app created successfully")
        print(f"🌐 Server will start at: http://{config.web.host}:{config.web.port}")
        print("📱 Navigate to: http://localhost:5000/face_detection")
        print()
        print("🎯 Test Instructions:")
        print("  1. Click 'Start Camera'")
        print("  2. Look at camera")
        print("  3. Blink naturally")
        print("  4. Move head left/right") 
        print("  5. Open/close mouth")
        print("  6. Try holding photo for spoofing test")
        print()
        print("🔍 Watch for:")
        print("  • Liveness Score: 0-100")
        print("  • Status: LIVE/FAKE/UNCERTAIN")
        print("  • Blink count increasing")
        print("  • EAR/MAR values changing")
        print()
        print("🛑 Press Ctrl+C to stop server")
        print("=" * 50)
        
        # Start server
        socketio.run(
            app, 
            host=config.web.host, 
            port=config.web.port, 
            debug=config.web.debug,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_server()

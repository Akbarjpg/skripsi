#!/usr/bin/env python3
"""
Simple Flask server for liveness detection demo (without complex config)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def start_simple_server():
    print("🚀 STARTING SIMPLE LIVENESS DETECTION SERVER")
    print("=" * 50)
    
    try:
        from flask import Flask, render_template
        from flask_socketio import SocketIO
        from pathlib import Path
        
        # Get project paths
        project_root = Path(__file__).parent
        template_dir = project_root / "src" / "web" / "templates"
        static_dir = project_root / "src" / "web" / "static"
        
        # Create simple Flask app
        app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir)
        )
        
        app.config['SECRET_KEY'] = 'simple_demo_key_for_liveness_test'
        
        # Initialize SocketIO
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        # Import the socket events from the original app
        from src.web.app_clean import register_socketio_events
        from src.utils.logger import get_web_logger
        
        logger = get_web_logger()
        register_socketio_events(socketio, logger)
        
        @app.route('/')
        def index():
            return "<h1>Liveness Detection Demo</h1><p><a href='/face_detection'>Go to Face Detection Test</a></p>"
        
        @app.route('/face_detection')
        def face_detection():
            return render_template('face_detection_clean.html')
        
        print("✅ Simple Flask app created successfully")
        print("🌐 Server starting at: http://localhost:5000")
        print("📱 Navigate to: http://localhost:5000/face_detection")
        print()
        print("🎯 Test Instructions:")
        print("  1. Click 'Start Camera'")
        print("  2. Look at camera")
        print("  3. Blink naturally")
        print("  4. Move head left/right") 
        print("  5. Open/close mouth")
        print("  6. Watch liveness score increase!")
        print()
        print("🔍 Expected Results:")
        print("  • Liveness Score: 0-100 (aim for >70)")
        print("  • Status: LIVE/FAKE/UNCERTAIN")
        print("  • Blink count should increase")
        print("  • EAR/MAR values should change with movement")
        print()
        print("🛑 Press Ctrl+C to stop server")
        print("=" * 50)
        
        # Start server
        socketio.run(
            app, 
            host="localhost", 
            port=5000, 
            debug=True,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_simple_server()

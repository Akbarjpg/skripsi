"""
Quick test script to start the face detection server
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.web.app_clean import create_app
    
    print("🚀 Starting Face Detection Server...")
    print("📍 URL: http://localhost:5000")
    print("🎯 Test Landmark Detection: http://localhost:5000/face-detection-clean")
    print("📊 Dashboard: http://localhost:5000/test-dashboard")
    print("")
    
    # create_app() returns (app, socketio) tuple
    app, socketio = create_app()
    
    print("✅ Server starting...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()

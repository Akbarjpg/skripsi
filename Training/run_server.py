#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run
from src.web.app_clean import create_app

if __name__ == '__main__':
    print("🚀 Starting Face Detection Server...")
    app, socketio = create_app()
    print("✅ Server starting on http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

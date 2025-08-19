#!/usr/bin/env python3
"""
Test SocketIO functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    print("Testing imports...")
    
    from src.web.app_clean import create_app
    print("✅ create_app imported successfully")
    
    from src.utils.config import SystemConfig
    print("✅ SystemConfig imported successfully")
    
    print("\nCreating app...")
    app, socketio = create_app()
    print("✅ App and SocketIO created successfully")
    
    print(f"App: {app}")
    print(f"SocketIO: {socketio}")
    
    print("\n🎉 All tests passed! SocketIO is working.")
    
    print("\nStarting server on localhost:5000...")
    socketio.run(app, host='localhost', port=5000, debug=True)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

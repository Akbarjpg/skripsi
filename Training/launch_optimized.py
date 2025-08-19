#!/usr/bin/env python3
"""
Direct Optimized System Launcher
Bypasses import issues by launching optimized components directly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def launch_optimized_system():
    """Launch optimized anti-spoofing system directly"""
    print("🚀 LAUNCHING OPTIMIZED ANTI-SPOOFING SYSTEM")
    print("=" * 50)
    
    try:
        print("1. 📦 Loading optimized components...")
        
        # Import optimized web app directly
        from src.web.app_optimized import create_optimized_app
        print("   ✅ Optimized web app loaded")
        
        # Create app instance
        print("2. 🏗️ Creating application instance...")
        app, socketio = create_optimized_app()
        print("   ✅ Flask app and SocketIO created")
        
        # Configuration
        host = '127.0.0.1'
        port = 5000
        debug = True
        
        print(f"3. 🌐 Starting server on {host}:{port}")
        print(f"   📱 Open browser: http://{host}:{port}")
        print("   🔧 Debug mode: ON")
        print("\n🎯 OPTIMIZED FEATURES ACTIVE:")
        print("   ✅ JSON serialization fixes applied")
        print("   ✅ Landmark detection timeout (500ms)")
        print("   ✅ Performance optimization (10+ FPS)")
        print("   ✅ Multi-method anti-spoofing")
        print("\n⚡ Starting server...")
        print("=" * 50)
        
        # Run the server
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            allow_unsafe_werkzeug=True,
            use_reloader=False  # Disable reloader to avoid import issues
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server launch failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if all dependencies are installed")
        print("2. Ensure camera permissions are granted")
        print("3. Try running: python test_json_fixes.py first")

if __name__ == "__main__":
    launch_optimized_system()

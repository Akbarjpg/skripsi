#!/usr/bin/env python3
"""
BLINK DETECTION READY LAUNCHER
Launch your web app with all blink detection fixes applied and ready
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def launch_with_blink_fixes():
    print("🚀 LAUNCHING FACE ATTENDANCE WITH BLINK DETECTION FIXES")
    print("=" * 60)
    print("🔧 APPLIED FIXES:")
    print("  ✅ Numpy stride error fixed (.copy() method in CNN)")
    print("  ✅ MediaPipe eye landmarks optimized (4-point EAR)")
    print("  ✅ Blink detection sensitivity improved (threshold: 0.25 → 0.30)")
    print("  ✅ Faster detection (3 frames → 2 frames requirement)")
    print("  ✅ Enhanced debug logging for troubleshooting")
    print("  ✅ Sequential challenge system improved")
    print("=" * 60)
    
    try:
        print("1. 📦 Loading optimized components...")
        
        # Import the web app
        from src.web.app_optimized import create_optimized_app
        print("   ✅ Web application loaded")
        
        print("2. 🏗️ Creating application instance...")
        app, socketio = create_optimized_app()
        print("   ✅ Flask app and SocketIO initialized")
        
        # Server configuration
        host = '127.0.0.1'
        port = 5000
        
        print(f"3. 🌐 Starting server on {host}:{port}")
        print(f"   📱 Open browser: http://{host}:{port}")
        print("\n🎯 TESTING INSTRUCTIONS:")
        print("   1. Navigate to 'Sequential Detection' mode")
        print("   2. Select challenge: 'Kedipkan mata 3 kali'")
        print("   3. Look directly at camera")
        print("   4. Blink clearly 3 times")
        print("   5. Watch blink counter increase: 0 → 1 → 2 → 3")
        print("   6. Challenge should complete successfully!")
        
        print("\n📊 EXPECTED IMPROVEMENTS:")
        print("   • More sensitive blink detection")
        print("   • Faster response time")
        print("   • Better debug information")
        print("   • Fewer false negatives")
        
        print("\n⚡ Starting server...")
        print("=" * 60)
        
        # Launch the server
        socketio.run(
            app,
            host=host,
            port=port,
            debug=True,
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
        print("1. Check if camera is available and not in use")
        print("2. Ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("3. Check for any import errors above")
        print("4. Try running a simple test first:")
        print("   python simple_blink_test.py")

if __name__ == "__main__":
    launch_with_blink_fixes()

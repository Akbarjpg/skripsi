#!/usr/bin/env python3
"""
Launch the main Face Anti-Spoofing system with integrated liveness detection
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def launch_main_system():
    print("🚀 LAUNCHING FACE ANTI-SPOOFING SYSTEM")
    print("💫 WITH INTEGRATED LIVENESS DETECTION")
    print("=" * 60)
    
    try:
        from src.web.app_clean import create_app
        from src.utils.config import get_default_config
        
        print("✅ Modules imported successfully")
        
        # Create app with default config
        config = get_default_config()
        app, socketio = create_app(config)
        
        print("✅ App initialized with liveness detection")
        
        print("\n🌐 SYSTEM FEATURES:")
        print("  🎯 Real-time liveness detection (0-100 score)")
        print("  👁️  Eye blink detection and counting")
        print("  👄 Mouth movement analysis")
        print("  🔄 Head pose estimation") 
        print("  🛡️ Anti-spoofing protection")
        print("  📊 Live/Fake classification")
        print("  🔍 Facial landmark visualization")
        
        print("\n📱 WEB INTERFACE:")
        print("  🏠 Main: http://localhost:5000")
        print("  � Liveness Test: http://localhost:5000/face_detection")
        print("  📋 Dashboard: http://localhost:5000/dashboard")
        
        print("\n🎯 TESTING INSTRUCTIONS:")
        print("  1. Navigate to http://localhost:5000/face_detection")
        print("  2. Click 'Start Camera'")
        print("  3. Look directly at camera")
        print("  4. Blink naturally (watch counter increase)")
        print("  5. Turn head left/right slowly")
        print("  6. Open/close mouth")
        print("  7. Watch liveness score increase!")
        print("  8. Try holding a photo to test anti-spoofing")
        
        print("\n🔍 EXPECTED RESULTS:")
        print("  • Live faces: Score 70-100 (GREEN status)")
        print("  • Uncertain: Score 40-69 (YELLOW status)")
        print("  • Fake/Photos: Score 0-39 (RED status)")
        print("  • Blink count should increase with natural blinks")
        print("  • EAR values should change (0.2-0.4 normal)")
        print("  • MAR values should change with mouth movement")
        
        print("\n🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        print("🔴 Starting web server...")
        
        # Launch web application
        socketio.run(
            app,
            host="localhost", 
            port=5000,
            debug=True,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("✅ Shutdown complete")
    except Exception as e:
        print(f"\n❌ Error launching system: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("  • Make sure all dependencies are installed")
        print("  • Check if port 5000 is available")
        print("  • Try: pip install -r requirements.txt")
        print("  • For camera issues, check webcam permissions")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_main_system()

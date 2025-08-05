#!/usr/bin/env python3
"""
Simple launcher for testing local image saving in web interface
"""

import sys
import os
import time
import webbrowser
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_prerequisites():
    """Check if everything is ready for testing"""
    print("🔍 Checking Prerequisites...")
    
    # Check app file
    app_path = Path("src/web/app_optimized.py")
    if not app_path.exists():
        print(f"❌ {app_path} not found!")
        return False
    print(f"✅ Found {app_path}")
    
    # Check faces directory
    faces_dir = Path("static/faces")
    faces_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Faces directory ready: {faces_dir}")
    
    # Check database
    db_path = Path("attendance.db")
    if db_path.exists():
        print(f"✅ Database exists: {db_path}")
    else:
        print(f"⚠️ Database will be created on first use")
    
    return True

def start_test_server():
    """Start the server for testing"""
    print("\n🚀 Starting Test Server...")
    
    try:
        # Import the Flask app
        from src.web.app_optimized import app, socketio
        
        print("✅ App imported successfully")
        print("🌐 Starting server on http://localhost:5000")
        print("📱 Face registration at: http://localhost:5000/register-face")
        print("🔍 Watch the console for debug messages!")
        
        print("\n" + "="*50)
        print("🎯 TESTING INSTRUCTIONS")
        print("="*50)
        print("1. 📷 Open face registration page")
        print("2. 🤖 Wait for automatic face detection")
        print("3. ⏱️ Watch for 3-2-1 countdown")
        print("4. 📸 Image will be captured automatically")
        print("5. 💾 Check console for 'Image saved to: static/faces/...'")
        print("6. 📂 Check static/faces/ folder for saved images")
        print("7. ✅ Should see success message instead of 'terjadi kesalahan'")
        print("="*50)
        
        # Try to open browser
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open('http://localhost:5000/register-face')
                print("🌐 Browser opened automatically")
            except:
                print("⚠️ Could not open browser automatically")
                print("   Please open: http://localhost:5000/register-face")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.start()
        
        # Start server
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("🧪 Local Image Saving - Web Interface Test")
    print("=" * 50)
    
    if not check_prerequisites():
        print("❌ Prerequisites check failed")
        return False
    
    print("\n✨ Local Image Saving Changes:")
    print("   • Face recognition processing temporarily disabled")
    print("   • Images saved directly to static/faces/ directory")
    print("   • Database stores image file paths instead of encodings")
    print("   • Enhanced debugging to track every step")
    
    print("\n🎯 Expected Result:")
    print("   • No more 'terjadi kesalahan saat memproses wajah' error")
    print("   • Images saved successfully to disk")
    print("   • Clear debug messages showing exactly what happens")
    
    print("\nPress Enter to start the server...")
    try:
        input()
    except:
        pass
    
    return start_test_server()

if __name__ == "__main__":
    main()

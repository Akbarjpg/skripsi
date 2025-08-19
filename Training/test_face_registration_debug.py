#!/usr/bin/env python3
"""
Test script for the face registration debugging implementation
Tests the comprehensive error handling and debugging features
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

def test_debugging_implementation():
    """Test if the debugging implementation is correctly added"""
    print("=== Face Registration Debugging Test ===\n")
    
    # Check Python backend changes
    app_path = Path("src/web/app_optimized.py")
    if not app_path.exists():
        print("❌ app_optimized.py not found!")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        app_content = f.read()
    
    # Check for debugging features in backend
    backend_checks = [
        '=== CAPTURE FACE DEBUG ===',
        'print(f"📥 Data keys:',
        'print(f"👤 User ID:',
        'print(f"📍 Position:',
        'print(f"🖼️ Image data exists:',
        'print(f"✅ Decoded image size:',
        'print(f"👥 Faces found:',
        'print(f"✅ Face encoding extracted',
        'print(f"✅ Successfully saved face data',
        'import traceback',
        'traceback.print_exc()'
    ]
    
    missing_backend = []
    for check in backend_checks:
        if check not in app_content:
            missing_backend.append(check)
    
    if missing_backend:
        print(f"❌ Missing backend debugging features: {missing_backend}")
        return False
    
    print("✅ Backend debugging implementation found")
    
    # Check JavaScript frontend changes
    template_path = Path("src/web/templates/register_face.html")
    if not template_path.exists():
        print("❌ register_face.html not found!")
        return False
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Check for debugging features in frontend
    frontend_checks = [
        'console.log(\'=== CAPTURE PHOTO DEBUG ===\')',
        'console.log(\'🔌 Setting up SocketIO debugging\')',
        'console.log(\'✅ SocketIO connected successfully\')',
        'console.log(\'📨 Received face_capture_result event:\')',
        'console.log(\'📐 Canvas dimensions:\')',
        'console.log(\'🖼️ Image data type:\')',
        'console.log(\'📤 Sending capture_face event\')',
        'console.log(\'✅ Capture successful:\')',
        'console.error(\'❌ Capture failed:\')',
        'setupSocketDebug()'
    ]
    
    missing_frontend = []
    for check in frontend_checks:
        if check not in template_content:
            missing_frontend.append(check)
    
    if missing_frontend:
        print(f"❌ Missing frontend debugging features: {missing_frontend}")
        return False
    
    print("✅ Frontend debugging implementation found")
    
    return True

def test_error_handling_improvements():
    """Test if error handling improvements are implemented"""
    print("\n🔍 Testing Error Handling Improvements...")
    
    app_path = Path("src/web/app_optimized.py")
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for improved error messages
    error_improvements = [
        'Data tidak lengkap:',
        'Gagal memproses data gambar',
        'Wajah tidak terdeteksi. Pastikan wajah Anda terlihat jelas',
        'Terdeteksi lebih dari satu wajah. Pastikan hanya Anda',
        'Gagal mengekstrak fitur wajah. Pastikan wajah terlihat jelas',
        'User tidak ditemukan dalam sistem',
        'Gagal menyimpan data wajah ke database'
    ]
    
    missing_improvements = []
    for improvement in error_improvements:
        if improvement not in content:
            missing_improvements.append(improvement)
    
    if missing_improvements:
        print(f"❌ Missing error handling improvements: {missing_improvements}")
        return False
    
    print("✅ Error handling improvements found")
    return True

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n📦 Testing Dependencies...")
    
    dependencies = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'face_recognition': 'face-recognition'
    }
    
    missing_deps = []
    available_deps = []
    
    for dep, package in dependencies.items():
        try:
            if dep == 'PIL':
                from PIL import Image
            elif dep == 'cv2':
                import cv2
            elif dep == 'numpy':
                import numpy
            elif dep == 'face_recognition':
                import face_recognition
            
            available_deps.append(dep)
            print(f"✅ {dep} available")
        except ImportError:
            missing_deps.append(package)
            print(f"❌ {dep} not available - install with: pip install {package}")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {missing_deps}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All dependencies available")
    return True

def generate_test_instructions():
    """Generate testing instructions"""
    print("\n📋 Testing Instructions:")
    print("=" * 50)
    
    print("\n1. Start the server:")
    print("   python src/web/app_optimized.py")
    
    print("\n2. Open browser and navigate to:")
    print("   http://localhost:5000/register-face")
    
    print("\n3. Open browser developer tools (F12)")
    print("   - Check Console tab for debug messages")
    print("   - Look for SocketIO connection messages")
    print("   - Monitor capture_face events")
    
    print("\n4. Test face registration:")
    print("   - Position face in camera")
    print("   - Wait for automatic countdown")
    print("   - Check console for detailed debug info")
    
    print("\n5. Check server console for:")
    print("   - === CAPTURE FACE DEBUG === messages")
    print("   - Detailed step-by-step processing")
    print("   - Specific error messages if issues occur")
    
    print("\n6. Expected debug output:")
    print("   📥 Data received and validated")
    print("   🖼️ Image processing steps")
    print("   👥 Face detection results")
    print("   💾 Database operations")
    print("   ✅ Success confirmation")

def main():
    print("🧪 Face Registration Debug Implementation Test\n")
    
    # Test implementation
    impl_ok = test_debugging_implementation()
    
    # Test error handling
    error_ok = test_error_handling_improvements()
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    if impl_ok and error_ok and deps_ok:
        print("\n🎉 All tests passed! Debug implementation is ready.")
        
        # Generate instructions
        generate_test_instructions()
        
        # Try to open browser for testing
        try:
            print("\n🌐 Opening browser for manual testing...")
            time.sleep(2)
            webbrowser.open('http://localhost:5000/register-face')
        except:
            print("Could not open browser automatically")
        
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

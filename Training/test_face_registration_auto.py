#!/usr/bin/env python3
"""
Test script for automatic face registration system
Tests the HTML template and automatic capture functionality
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_template_validation():
    """Test if the register_face.html template is valid"""
    template_path = current_dir / "src" / "web" / "templates" / "register_face.html"
    
    print("🔍 Testing Face Registration Template...")
    print(f"Template path: {template_path}")
    
    if not template_path.exists():
        print("❌ Template file not found!")
        return False
    
    # Read and validate template content
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key automatic capture features
    required_features = [
        'countdown-display',
        'startFaceDetection',
        'detectFace',
        'startCountdown',
        'simpleFaceDetection',
        'face-detected',
        'stableFrames'
    ]
    
    missing_features = []
    for feature in required_features:
        if feature not in content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing automatic capture features: {missing_features}")
        return False
    
    print("✅ All automatic capture features found in template")
    
    # Check HTML structure
    html_checks = [
        '<div id="countdown-display"',
        'class="countdown-circle"',
        'id="status-message"',
        'startFaceDetection()',
        'simpleFaceDetection(imageData)'
    ]
    
    for check in html_checks:
        if check not in content:
            print(f"❌ Missing HTML structure: {check}")
            return False
    
    print("✅ HTML structure validation passed")
    return True

def test_server_compatibility():
    """Test if the server has the required face capture endpoint"""
    app_path = current_dir / "src" / "web" / "app_optimized.py"
    
    print("\n🔍 Testing Server Compatibility...")
    
    if not app_path.exists():
        print("❌ app_optimized.py not found!")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_endpoints = [
        '@socketio.on(\'capture_face\')',
        'handle_capture_face',
        'face_capture_result',
        'face_recognition',
        'face_locations'
    ]
    
    for endpoint in required_endpoints:
        if endpoint not in content:
            print(f"❌ Missing server endpoint: {endpoint}")
            return False
    
    print("✅ Server compatibility check passed")
    return True

def start_test_server():
    """Start the Flask server for testing"""
    print("\n🚀 Starting Test Server...")
    print("Note: Please start the server manually with:")
    print("python src/web/app_optimized.py")
    print("\nThen navigate to: http://localhost:5000/register_face")
    print("\nAutomatic capture features to test:")
    print("1. Camera should start automatically")
    print("2. Status should show 'Mencari wajah...'")
    print("3. When face is detected: 'Wajah terdeteksi! Tetap dalam posisi...'")
    print("4. After 10 stable frames: 3-2-1 countdown should appear")
    print("5. Photo should be captured automatically")
    print("6. Process should repeat for left and right positions")

def main():
    print("=== Automatic Face Registration Test ===\n")
    
    # Test template
    template_ok = test_template_validation()
    
    # Test server compatibility
    server_ok = test_server_compatibility()
    
    if template_ok and server_ok:
        print("\n✅ All tests passed! Automatic face registration is ready.")
        start_test_server()
        
        # Try to open browser (optional)
        try:
            time.sleep(2)
            print("\n🌐 Opening browser for manual testing...")
            webbrowser.open('http://localhost:5000/register_face')
        except:
            print("Could not open browser automatically")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False
    
    return True

if __name__ == "__main__":
    main()

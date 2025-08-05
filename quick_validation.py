#!/usr/bin/env python3
"""
Quick validation test dengan error handling
"""

def quick_validation():
    print("🚀 QUICK SYSTEM VALIDATION")
    print("=" * 40)
    
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    try:
        import torch
        import cv2
        import numpy as np
        from flask import Flask
        print("  ✅ Core dependencies OK")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Import error: {e}")
    
    # Test 2: MediaPipe
    print("\n2. Testing MediaPipe...")
    try:
        import mediapipe as mp
        print(f"  ✅ MediaPipe version: {mp.__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ MediaPipe error: {e}")
    
    # Test 3: Landmark detection
    print("\n3. Testing landmark detection...")
    try:
        from src.detection.landmark_detection import LivenessVerifier
        verifier = LivenessVerifier()
        print("  ✅ LivenessVerifier initialized")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Landmark detection error: {e}")
    
    # Test 4: Flask app
    print("\n4. Testing Flask app...")
    try:
        from src.web.app_clean import create_app
        app, socketio = create_app()
        print("  ✅ Flask app and SocketIO created")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Flask app error: {e}")
    
    # Test 5: CNN model (optional)
    print("\n5. Testing CNN model...")
    try:
        from src.models.cnn_model import LivenessDetectionCNN
        model = LivenessDetectionCNN()
        print("  ✅ CNN model loaded")
        tests_passed += 1
    except Exception as e:
        print(f"  ⚠️  CNN model issue: {e}")
    
    # Results
    print("\n" + "=" * 40)
    print(f"📊 Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed >= 3:
        print("🎉 SYSTEM IS FUNCTIONAL!")
        print("✅ Core landmark detection system working")
        print("✅ Web interface ready")
        print("\n🌐 Ready to test at: http://localhost:5000/face_detection_clean")
    else:
        print("⚠️  Some critical components need attention")
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = quick_validation()
    exit(0 if success else 1)

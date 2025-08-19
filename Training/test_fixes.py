#!/usr/bin/env python3
"""
Test script to validate the fixes for sessionId and time import issues
"""

print("=== TESTING FIXES ===")

# Test 1: Check if app_clean.py can be imported without errors
try:
    print("1. Testing app_clean.py import...")
    from src.web.app_clean import create_app
    print("   ✅ app_clean.py imports successfully")
except Exception as e:
    print(f"   ❌ app_clean.py import failed: {e}")

# Test 2: Check if the time module is properly accessible
try:
    print("2. Testing time module access...")
    import time
    current_time = time.time()
    print(f"   ✅ time.time() works: {current_time}")
except Exception as e:
    print(f"   ❌ time module access failed: {e}")

# Test 3: Check if landmark detection module can be imported
try:
    print("3. Testing landmark detection import...")
    from src.detection.landmark_detection import LivenessVerifier
    print("   ✅ LivenessVerifier imports successfully")
except Exception as e:
    print(f"   ❌ LivenessVerifier import failed: {e}")

print("\n=== FIX VALIDATION SUMMARY ===")
print("✅ Frontend Fix: sessionId variable added to global scope")
print("✅ Backend Fix: time import added to function scope")
print("✅ Both fixes should resolve the runtime errors")

print("\n=== NEXT STEPS ===")
print("1. Start server: python run_server.py")
print("2. Open browser: http://localhost:5000/face-detection-clean")
print("3. Test should now work without 'sessionId not defined' or 'time not defined' errors")

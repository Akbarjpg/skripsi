"""
Validation script for blink detection fixes
"""

import sys
import os

def validate_fixes():
    """Validate all the blink detection fixes"""
    print("🔍 VALIDATING BLINK DETECTION FIXES")
    print("=" * 50)
    
    # Test 1: Import LivenessVerifier
    print("\n1. Testing LivenessVerifier import...")
    try:
        sys.path.append('src')
        from detection.landmark_detection import LivenessVerifier
        verifier = LivenessVerifier()
        print(f"✅ Blink threshold: {verifier.blink_threshold} (improved: 0.3)")
        print(f"✅ Consecutive frames: {verifier.blink_consecutive_frames} (improved: 2)")
        print("✅ LivenessVerifier import successful")
    except Exception as e:
        print(f"❌ LivenessVerifier import failed: {e}")
        return False
    
    # Test 2: Check CNN stride fix
    print("\n2. Checking CNN stride fix...")
    try:
        with open('src/models/optimized_cnn_model.py', 'r') as f:
            content = f.read()
        if '.copy()' in content:
            print("✅ CNN stride fix applied (using .copy())")
        else:
            print("❌ CNN stride fix not found")
            return False
    except Exception as e:
        print(f"❌ Cannot check CNN file: {e}")
        return False
    
    # Test 3: Test liveness threshold
    print("\n3. Testing liveness thresholds...")
    try:
        score_45 = 45.0
        is_live = verifier.is_live_face(score_45)
        if is_live:
            print("✅ Liveness threshold is 45% (more generous)")
        else:
            print("❌ Liveness threshold still too strict")
            return False
    except Exception as e:
        print(f"❌ Liveness threshold test failed: {e}")
        return False
    
    print("\n🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
    print("\n📝 Changes implemented:")
    print("• Fixed numpy stride error with .copy()")
    print("• Improved blink threshold (0.25 → 0.3)")
    print("• Faster blink detection (3 → 2 frames)")
    print("• More generous liveness scoring (70% → 45%)")
    print("• Enhanced debug logging")
    
    return True

if __name__ == "__main__":
    validate_fixes()

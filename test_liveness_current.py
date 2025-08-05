#!/usr/bin/env python3
"""
Test current liveness detection capabilities
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.landmark_detection import LivenessVerifier
import cv2
import numpy as np

def test_liveness_detection():
    print("🔍 TESTING CURRENT LIVENESS DETECTION STATE")
    print("=" * 50)
    
    try:
        # Initialize verifier
        verifier = LivenessVerifier()
        print("✅ LivenessVerifier imported successfully")
        
        # Check available methods
        print("\n📋 Available methods:")
        methods = [method for method in dir(verifier) if not method.startswith('_')]
        for method in methods:
            print(f"  - {method}")
        
        # Test with dummy image (no face)
        print("\n🖼️ Testing with empty image (no face):")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = verifier.process_frame(dummy_image)
        
        print("Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # Check what liveness metrics are available
        print("\n🎯 LIVENESS DETECTION ANALYSIS:")
        
        # Check for blink detection
        if 'blink_count' in result:
            print(f"✅ Blink detection: {result['blink_count']} (available)")
        else:
            print("❌ Blink detection: NOT AVAILABLE")
        
        # Check for EAR values
        if 'ear_left' in result and 'ear_right' in result:
            print(f"✅ Eye Aspect Ratio: L={result['ear_left']:.3f}, R={result['ear_right']:.3f}")
        else:
            print("❌ Eye Aspect Ratio: NOT AVAILABLE")
        
        # Check for mouth movement
        if 'mar' in result:
            print(f"✅ Mouth Aspect Ratio: {result['mar']:.3f}")
        else:
            print("❌ Mouth Aspect Ratio: NOT AVAILABLE")
            
        if 'mouth_open' in result:
            print(f"✅ Mouth open detection: {result['mouth_open']}")
        else:
            print("❌ Mouth open detection: NOT AVAILABLE")
        
        # Check for head movement
        if 'head_movement' in result:
            print(f"✅ Head movement: {result['head_movement']}")
        else:
            print("❌ Head movement: NOT AVAILABLE")
            
        if 'head_pose' in result:
            print(f"✅ Head pose: {result['head_pose']}")
        else:
            print("❌ Head pose: NOT AVAILABLE")
        
        # Check for liveness score
        liveness_score_found = False
        for key in result.keys():
            if 'liveness' in key.lower():
                print(f"✅ Liveness metric found: {key} = {result[key]}")
                liveness_score_found = True
        
        if not liveness_score_found:
            print("❌ No liveness score calculated")
        
        print("\n📊 SUMMARY:")
        print(f"Total metrics available: {len(result)}")
        
        # Determine if liveness detection is working
        has_blink = 'blink_count' in result
        has_ear = 'ear_left' in result and 'ear_right' in result
        has_mouth = 'mar' in result or 'mouth_open' in result
        has_head = 'head_movement' in result or 'head_pose' in result
        
        working_features = sum([has_blink, has_ear, has_mouth, has_head])
        
        if working_features >= 3:
            print("🟢 LIVENESS DETECTION: MOSTLY WORKING")
        elif working_features >= 2:
            print("🟡 LIVENESS DETECTION: PARTIALLY WORKING")
        elif working_features >= 1:
            print("🟠 LIVENESS DETECTION: BASIC FUNCTIONALITY")
        else:
            print("🔴 LIVENESS DETECTION: NOT WORKING")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing liveness detection: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_liveness_detection()
    os.system('python test_liveness_server.py')

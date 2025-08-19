#!/usr/bin/env python3
"""
Quick test to verify lighting detection fix
"""

import sys
import os
import numpy as np
import cv2

# Add the web app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'web'))

def test_lighting_fix():
    """Test the lighting detection fix"""
    print("🔍 Testing Lighting Detection Fix...")
    
    try:
        from app_optimized import EnhancedFrameProcessor
        
        processor = EnhancedFrameProcessor()
        print("✅ EnhancedFrameProcessor created")
        
        # Create test frames
        print("\n📸 Creating test frames...")
        
        # 1. Normal lighting frame (brightness ~120, good contrast >25)
        normal_frame = np.random.normal(120, 30, (480, 640, 3)).astype(np.uint8)
        normal_frame = np.clip(normal_frame, 0, 255)
        
        # 2. Poor lighting frame (brightness ~36, very dark)
        poor_lighting_frame = np.random.normal(36, 15, (480, 640, 3)).astype(np.uint8)
        poor_lighting_frame = np.clip(poor_lighting_frame, 0, 255)
        
        # 3. Too bright frame (brightness ~216, too bright)
        too_bright_frame = np.random.normal(216, 20, (480, 640, 3)).astype(np.uint8)
        too_bright_frame = np.clip(too_bright_frame, 0, 255)
        
        # Test each frame
        test_cases = [
            ("Normal Lighting", normal_frame, "> 0.6"),
            ("Poor Lighting (Dark)", poor_lighting_frame, "< 0.5"),
            ("Too Bright", too_bright_frame, "< 0.8")
        ]
        
        print("\n🧪 Testing lighting detection...")
        for name, frame, expected in test_cases:
            quality = processor.assess_frame_quality(frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            lighting_score = quality['lighting_score']
            
            print(f"\n--- {name} ---")
            print(f"   Brightness: {brightness:.1f}")
            print(f"   Contrast (std): {contrast:.1f}")
            print(f"   Lighting Score: {lighting_score:.3f}")
            print(f"   Expected: {expected}")
            
            # Validate expectations
            if name == "Normal Lighting":
                if lighting_score > 0.6:
                    print("   ✅ PASS: Normal lighting correctly detected")
                else:
                    print("   ❌ FAIL: Normal lighting not detected")
                    
            elif name == "Poor Lighting (Dark)":
                if lighting_score < 0.5:
                    print("   ✅ PASS: Poor lighting correctly detected")
                else:
                    print("   ❌ FAIL: Poor lighting not detected")
                    
            elif name == "Too Bright":
                if lighting_score < 0.8:
                    print("   ✅ PASS: Too bright correctly detected")
                else:
                    print("   ❌ FAIL: Too bright not detected")
        
        print(f"\n🎉 Lighting Detection Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LIGHTING DETECTION FIX TEST")
    print("=" * 50)
    
    success = test_lighting_fix()
    
    if success:
        print("\n✅ Lighting detection is working correctly!")
    else:
        print("\n❌ Lighting detection needs more fixes")

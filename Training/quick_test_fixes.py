#!/usr/bin/env python3
"""
Quick test untuk verified fixes
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def quick_test():
    print("🧪 Quick Optimized System Test")
    print("=" * 40)
    
    # Test 1: JSON Conversion
    print("\n1. Testing JSON conversion...")
    try:
        from src.web.app_optimized import convert_to_serializable
        import numpy as np
        import json
        
        # Test problematic types
        test_data = {
            'confidence': np.float32(0.85),
            'is_live': np.bool_(True),
            'landmarks_detected': np.bool_(False)
        }
        
        converted = convert_to_serializable(test_data)
        json_str = json.dumps(converted)
        print("✅ JSON conversion working")
        
    except Exception as e:
        print(f"❌ JSON conversion failed: {e}")
        return False
    
    # Test 2: Frame Processor Import
    print("\n2. Testing frame processor import...")
    try:
        from src.web.app_optimized import OptimizedFrameProcessor
        processor = OptimizedFrameProcessor()
        print("✅ Frame processor imported")
        
    except Exception as e:
        print(f"❌ Frame processor import failed: {e}")
        return False
    
    # Test 3: Quick Processing Test
    print("\n3. Testing quick processing...")
    try:
        import numpy as np
        
        # Very small test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        start = time.time()
        result = processor.process_frame_optimized(test_image, session_id="quick_test")
        end = time.time()
        
        print(f"✅ Processing completed in {end-start:.3f}s")
        
        # Test serialization of result
        serializable_result = convert_to_serializable(result)
        json.dumps(serializable_result)
        print("✅ Result serialization working")
        
    except Exception as e:
        print(f"❌ Processing test failed: {e}")
        return False
    
    print("\n🎉 All quick tests passed!")
    print("✅ System fixes are working correctly")
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Ready to start optimized server")
    else:
        print("\n❌ Issues found - check logs")

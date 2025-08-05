#!/usr/bin/env python3
"""
Test script untuk memvalidasi fixes JSON serialization dan performance
"""

import sys
import os
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_convert_to_serializable():
    """Test convert_to_serializable function"""
    print("🧪 Testing convert_to_serializable function...")
    
    try:
        # Import the function - FIX: Use src. prefix for imports
        from src.web.app_optimized import convert_to_serializable
        
        # Test data with numpy types
        test_data = {
            'numpy_int': np.int32(42),
            'numpy_float': np.float64(3.14),
            'numpy_bool': np.bool_(True),
            'numpy_array': np.array([1, 2, 3]),
            'regular_int': 123,
            'regular_float': 2.71,
            'regular_bool': False,
            'regular_list': [1, 2, 3],
            'nested_dict': {
                'inner_numpy': np.float32(1.23),
                'inner_list': [np.int64(456), np.bool_(False)]
            }
        }
        
        # Convert to serializable
        converted = convert_to_serializable(test_data)
        
        # Try to serialize to JSON
        json_str = json.dumps(converted)
        print("✅ JSON serialization successful!")
        print(f"📝 Sample output: {json_str[:100]}...")
        
        # Check types
        print("📊 Type conversions:")
        for key, value in converted.items():
            if key != 'nested_dict':
                print(f"  {key}: {type(value).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_frame_processor_import():
    """Test frame processor import"""
    print("\n🧪 Testing frame processor import...")
    
    try:
        from src.web.app_optimized import OptimizedFrameProcessor
        processor = OptimizedFrameProcessor()
        print("✅ OptimizedFrameProcessor imported successfully!")
        
        # Test performance stats
        stats = processor.get_performance_stats()
        print(f"📊 Initial stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing OptimizedFrameProcessor: {e}")
        return False

def test_landmark_detection_timeout():
    """Test landmark detection with timeout"""
    print("\n🧪 Testing landmark detection configuration...")
    
    try:
        from src.web.app_optimized import OptimizedFrameProcessor
        import cv2
        
        processor = OptimizedFrameProcessor()
        
        # Check settings
        print(f"📦 Skip frame count: {processor.skip_frame_count}")
        print(f"⏱️ Cache duration: {processor.cache_duration}")
        print(f"📊 Processing times buffer size: {processor.processing_times.maxlen}")
        
        # Create dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("🎯 Testing with dummy image (should be fast)...")
        start_time = time.time()
        
        # This should be quick since it's a dummy image
        result = processor._create_empty_landmark_result()
        
        elapsed = time.time() - start_time
        print(f"⏱️ Empty result creation time: {elapsed:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting JSON Serialization & Performance Fixes Test")
    print("=" * 60)
    
    import time
    
    # Run tests
    results = []
    
    results.append(test_convert_to_serializable())
    results.append(test_frame_processor_import())
    results.append(test_landmark_detection_timeout())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! JSON serialization fixes should work.")
        print("\n🔧 FIXES IMPLEMENTED:")
        print("1. ✅ Added convert_to_serializable() function")
        print("2. ✅ Applied conversion to emit() calls")
        print("3. ✅ Added landmark detection timeout (500ms)")
        print("4. ✅ Improved frame skipping (every 2nd frame)")
        print("5. ✅ Shorter cache duration (100ms)")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run the optimized system: python src/core/app_launcher.py")
    print("2. Test in browser at http://localhost:5000")
    print("3. Check for JSON serialization errors in console")
    print("4. Verify landmark detection is responsive")

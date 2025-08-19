#!/usr/bin/env python3
"""
Test JSON serialization fix untuk numpy arrays
"""

import json
import numpy as np
from src.utils import json_serializable, safe_jsonify
from src.detection.landmark_detection import FacialLandmarkDetector, LivenessVerifier

def test_json_serialization():
    """Test JSON serialization dengan numpy arrays"""
    print("Testing JSON serialization fix...")
    
    # Test 1: Basic numpy array conversion
    print("\n1. Testing numpy array conversion:")
    test_array = np.array([[1, 2], [3, 4]])
    converted = json_serializable(test_array)
    print(f"   Original: {type(test_array)} - {test_array.shape}")
    print(f"   Converted: {type(converted)} - {converted}")
    
    # Test 2: Complex dictionary with numpy arrays
    print("\n2. Testing complex dictionary:")
    test_dict = {
        'landmarks': np.random.rand(5, 2),
        'confidence': np.float64(0.95),
        'count': np.int32(5),
        'nested': {
            'array': np.array([1, 2, 3]),
            'number': np.float32(1.5)
        }
    }
    
    safe_dict = safe_jsonify(test_dict)
    json_string = json.dumps(safe_dict)  # This should not raise an error
    print(f"   Successfully converted to JSON string (length: {len(json_string)})")
    
    # Test 3: Mock detection result
    print("\n3. Testing mock detection result:")
    mock_result = {
        'landmarks_detected': True,
        'confidence': 0.95,
        'landmarks': np.random.rand(10, 2),  # This would cause the error
        'blink_count': 2,
        'ear_left': np.float64(0.25),
        'ear_right': np.float64(0.27)
    }
    
    try:
        # This should work now
        safe_result = safe_jsonify(mock_result)
        json_str = json.dumps(safe_result)
        print(f"   ✅ Mock detection result serialized successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ JSON serialization test completed!")

if __name__ == "__main__":
    test_json_serialization()

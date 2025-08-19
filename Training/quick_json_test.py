#!/usr/bin/env python3
"""
Quick test untuk JSON serialization fix
"""

import numpy as np
import json

def convert_to_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to JSON-compatible types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

if __name__ == "__main__":
    print("🧪 Testing JSON Serialization Fix")
    
    # Test problematic data types
    test_data = {
        'numpy_bool': np.bool_(True),
        'numpy_int': np.int32(42),
        'numpy_float': np.float64(3.14),
        'confidence': np.float32(0.85),
        'is_live': np.bool_(False),
        'probabilities': {
            'live': np.float64(0.7),
            'fake': np.float64(0.3)
        }
    }
    
    print("Original data types:")
    for key, value in test_data.items():
        print(f"  {key}: {type(value)}")
    
    # Convert
    converted = convert_to_serializable(test_data)
    
    print("\nConverted data types:")
    for key, value in converted.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with types {[type(v) for v in value.values()]}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Test JSON serialization
    try:
        json_str = json.dumps(converted)
        print("\n✅ JSON serialization successful!")
        print(f"JSON: {json_str}")
    except Exception as e:
        print(f"\n❌ JSON serialization failed: {e}")

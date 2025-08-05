#!/usr/bin/env python3
"""
Test dengan debug output langsung ke console
"""

from src.detection.landmark_detection import LivenessVerifier
import numpy as np

def test_direct():
    print("=== DIRECT LANDMARK TEST ===")
    
    # Create dummy image 
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_image[200:280, 300:340] = [255, 255, 255]  # White rectangle sebagai "wajah"
    
    verifier = LivenessVerifier()
    
    print("Testing with dummy image...")
    results = verifier.process_frame(dummy_image)
    
    print("\n=== RESULTS ===")
    for key, value in results.items():
        if key == 'landmark_coordinates' and isinstance(value, list):
            print(f"{key}: {len(value)} coordinates")
            if value:
                print(f"  First coordinate: {value[0]}")
                print(f"  Last coordinate: {value[-1]}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_direct()

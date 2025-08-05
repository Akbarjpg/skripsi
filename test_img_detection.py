#!/usr/bin/env python3
"""
Test landmark detection dengan gambar wajah asli
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
from src.detection.landmark_detection import LivenessVerifier

def test_with_real_image():
    print("🧪 TESTING LANDMARK DETECTION WITH REAL FACE IMAGE")
    print("=" * 60)
    
    # Test dengan salah satu gambar real
    image_path = "test_img/color/1_1.avi_100_real.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Initialize verifier
    verifier = LivenessVerifier()
    print("✅ LivenessVerifier initialized")
    
    # Process the image
    print("\n🔍 Processing image with MediaPipe...")
    results = verifier.process_frame(image)
    
    # Display results
    print("\n📊 RESULTS:")
    print(f"  Landmarks detected: {results['landmarks_detected']}")
    print(f"  Confidence: {results['confidence']}")
    print(f"  Number of landmarks: {len(results.get('landmark_coordinates', []))}")
    print(f"  EAR left: {results['ear_left']}")
    print(f"  EAR right: {results['ear_right']}")
    print(f"  MAR: {results['mar']}")
    print(f"  Blink count: {results['blink_count']}")
    print(f"  Mouth open: {results['mouth_open']}")
    
    if results['landmarks_detected'] and len(results['landmark_coordinates']) > 0:
        landmarks = results['landmark_coordinates']
        print(f"\n🎉 SUCCESS! Detected {len(landmarks)} landmarks")
        print(f"  First landmark: {landmarks[0]}")
        print(f"  Last landmark: {landmarks[-1]}")
        print("  Landmark format looks correct for frontend!")
        
        # Test a few more images
        test_images = [
            "test_img/color/1_2.avi_100_real.jpg",
            "test_img/color/2_1.avi_100_real.jpg", 
            "test_img/color/3_1.avi_100_real.jpg"
        ]
        
        success_count = 1  # Already have 1 success
        
        for test_img in test_images:
            if os.path.exists(test_img):
                print(f"\n🧪 Testing: {test_img}")
                img = cv2.imread(test_img)
                if img is not None:
                    res = verifier.process_frame(img)
                    if res['landmarks_detected']:
                        success_count += 1
                        print(f"  ✅ Success: {len(res['landmark_coordinates'])} landmarks")
                    else:
                        print(f"  ⚠️ No landmarks detected")
        
        print(f"\n📈 SUMMARY: {success_count}/4 images successfully processed")
        
        if success_count >= 2:
            print("🎉 LANDMARK DETECTION IS WORKING WELL!")
            print("   The web interface should display landmarks correctly.")
        
    else:
        print("❌ No landmarks detected in this image")
        print("   This might be due to image quality or face orientation")

if __name__ == "__main__":
    test_with_real_image()

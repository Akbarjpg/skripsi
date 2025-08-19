"""
Test imports for anti-spoofing system
"""
import sys
sys.path.append('.')

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing Anti-Spoofing System Imports")
    print("=" * 40)
    
    try:
        print("1. Testing FacialLandmarkDetector...")
        from src.detection.landmark_detection import FacialLandmarkDetector
        detector = FacialLandmarkDetector()
        print("   ✅ FacialLandmarkDetector imported successfully")
        
        print("2. Testing AdvancedAntiSpoofingDetector...")
        from src.detection.advanced_antispoofing import AdvancedAntiSpoofingDetector
        antispoofing = AdvancedAntiSpoofingDetector()
        print("   ✅ AdvancedAntiSpoofingDetector imported successfully")
        
        print("3. Testing basic components...")
        import cv2
        import numpy as np
        import sqlite3
        print("   ✅ Basic components imported successfully")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        print("✅ Anti-spoofing system ready to use")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()

"""
Quick test to verify the time variable fix
"""
import sys
import os
import numpy as np
import cv2

# Add project to path
sys.path.append('.')

def test_sequential_processing():
    """Test the sequential processing fix"""
    print("🧪 Testing Sequential Processing Fix")
    print("=" * 40)
    
    try:
        # Import the optimized frame processor
        from src.web.app_optimized import OptimizedFrameProcessor
        
        print("✅ Import successful")
        
        # Create processor
        processor = OptimizedFrameProcessor()
        print("✅ Processor created")
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✅ Test image created")
        
        # Test sequential processing
        print("🔍 Testing sequential processing...")
        result = processor.process_frame_sequential(test_image, session_id="test_session")
        
        print("✅ Sequential processing completed!")
        print(f"   Result phase: {result.get('phase', 'unknown')}")
        print(f"   Result status: {result.get('status', 'unknown')}")
        print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequential_processing()
    if success:
        print("\n🎉 FIX VERIFIED - Sequential processing is working!")
    else:
        print("\n❌ FIX FAILED - Still has issues")

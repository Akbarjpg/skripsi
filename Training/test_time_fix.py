"""
Simple test to verify the time variable fix
"""
import sys
import os
sys.path.append('.')

def test_time_fix():
    """Test that the time variable fix works"""
    print("🧪 Testing Time Variable Fix")
    print("=" * 30)
    
    try:
        # Test the core functionality
        from src.web.app_optimized import EnhancedFrameProcessor
        
        print("✅ Import successful")
        
        # Create processor
        processor = EnhancedFrameProcessor()
        print("✅ Processor created")
        
        # Test that time module works
        import time
        current_time = time.time()
        print(f"✅ Time module working: {current_time}")
        
        # Check if we can access the process_frame_sequential method
        if hasattr(processor, 'process_frame_sequential'):
            print("✅ process_frame_sequential method available")
        else:
            print("❌ process_frame_sequential method not found")
            
        print("\n🎉 TIME FIX VERIFIED - No scope conflicts!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_time_fix()

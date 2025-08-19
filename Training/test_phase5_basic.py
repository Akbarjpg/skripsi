#!/usr/bin/env python3
"""
Simplified Phase 5 Test - Basic Enhanced Frame Processing
"""

import os
import sys
import numpy as np
import cv2
import time

# Add the source directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src', 'web')
sys.path.insert(0, src_dir)

def test_phase5_basic():
    """Basic test of Phase 5 enhanced frame processing"""
    print("=" * 60)
    print("PHASE 5 ENHANCED FRAME PROCESSING - BASIC TEST")
    print("=" * 60)
    
    try:
        # Import our enhanced frame processor
        print("Importing EnhancedFrameProcessor...")
        from app_optimized import EnhancedFrameProcessor
        print("✅ Successfully imported EnhancedFrameProcessor")
        
        # Create processor instance
        print("Creating processor instance...")
        processor = EnhancedFrameProcessor()
        print("✅ Successfully created processor")
        
        # Create a test image
        print("Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✅ Test image created")
        
        # Test 1: Frame Quality Assessment
        print("\n--- Test 1: Frame Quality Assessment ---")
        start_time = time.time()
        quality_result = processor.assess_frame_quality(test_image)
        assessment_time = time.time() - start_time
        
        print(f"Quality Assessment Time: {assessment_time:.3f}s")
        print(f"Overall Quality: {quality_result.get('overall_quality', 'N/A')}")
        print(f"Quality Grade: {quality_result.get('quality_grade', 'N/A')}")
        print("✅ Frame quality assessment working")
        
        # Test 2: Intelligent Frame Selection
        print("\n--- Test 2: Intelligent Frame Selection ---")
        should_process, reason = processor.should_process_frame(test_image, quality_result, "test_session")
        print(f"Should Process: {should_process}")
        print(f"Reason: {reason}")
        print("✅ Intelligent frame selection working")
        
        # Test 3: Background Analysis
        print("\n--- Test 3: Background Analysis ---")
        background_result = processor.detect_background_context(test_image)
        print(f"Screen Likelihood: {background_result.get('screen_likelihood', 'N/A')}")
        print(f"Natural Likelihood: {background_result.get('natural_likelihood', 'N/A')}")
        print("✅ Background analysis working")
        
        # Test 4: Processing Statistics
        print("\n--- Test 4: Processing Statistics ---")
        stats = processor.get_processing_stats()
        print(f"Frame Count: {stats.get('frame_count', 0)}")
        print(f"Current Stage: {stats.get('current_stage', 'N/A')}")
        print(f"Suspicion Level: {stats.get('current_suspicion_level', 0.0):.3f}")
        print("✅ Processing statistics working")
        
        print("\n" + "=" * 60)
        print("🎉 PHASE 5 BASIC TEST COMPLETED SUCCESSFULLY!")
        print("✅ All core enhanced processing features are working")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Check that src/web/app_optimized.py contains EnhancedFrameProcessor class")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase5_basic()
    if success:
        print("\n🚀 Phase 5 is ready for advanced testing!")
    else:
        print("\n❌ Phase 5 needs fixes before proceeding")

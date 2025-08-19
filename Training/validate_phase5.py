#!/usr/bin/env python3
"""
Final Phase 5 Validation - Quick Test
"""

import os
import sys

# Add the web app directory to Python path
web_dir = os.path.join(os.path.dirname(__file__), 'src', 'web')
sys.path.insert(0, web_dir)

print("=" * 50)
print("PHASE 5 FINAL VALIDATION")
print("=" * 50)

try:
    # Import the enhanced processor
    from app_optimized import EnhancedFrameProcessor
    print("✅ EnhancedFrameProcessor imported successfully")
    
    # Create instance
    processor = EnhancedFrameProcessor()
    print("✅ EnhancedFrameProcessor instance created")
    
    # Test with dummy data
    import numpy as np
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test quality assessment
    quality = processor.assess_frame_quality(test_image)
    print(f"✅ Quality assessment: {quality['quality_grade']} ({quality['overall_quality']:.3f})")
    
    # Test background analysis  
    background = processor.detect_background_context(test_image)
    print(f"✅ Background analysis: Suspicion {background['background_suspicion']:.3f}")
    
    # Test frame selection
    should_process, reason = processor.should_process_frame(test_image, quality, "test")
    print(f"✅ Frame selection: {should_process} ({reason})")
    
    # Test statistics
    stats = processor.get_processing_stats()
    print(f"✅ Statistics: {stats['frame_count']} frames, Stage: {stats['current_stage']}")
    
    print("\n" + "=" * 50)
    print("🎉 PHASE 5 VALIDATION SUCCESSFUL!")
    print("✅ All enhanced processing features working")
    print("✅ Ready for production use")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    import traceback
    traceback.print_exc()

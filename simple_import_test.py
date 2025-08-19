#!/usr/bin/env python3
"""
Simple test to check if EnhancedFrameProcessor can be imported
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'web'))

try:
    from app_optimized import EnhancedFrameProcessor
    print("✅ Successfully imported EnhancedFrameProcessor")
    
    # Try to create an instance
    processor = EnhancedFrameProcessor()
    print("✅ Successfully created EnhancedFrameProcessor instance")
    
    # Test basic functionality
    import numpy as np
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("✅ Testing frame quality assessment...")
    quality = processor.assess_frame_quality(test_image)
    print(f"   Quality result: {quality.get('overall_quality', 'N/A')}")
    
    print("✅ All basic tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

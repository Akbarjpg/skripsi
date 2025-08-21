"""Simple test to debug the comprehensive test suite"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

try:
    print("Testing import...")
    from src.testing.comprehensive_test_suite import ComprehensiveTestSuite
    print("✅ Import successful")
    
    print("Testing initialization...")
    test_suite = ComprehensiveTestSuite()
    print("✅ Initialization successful")
    
    print("Testing system info...")
    print(test_suite.test_metadata['system_info'])
    print("✅ System info retrieved successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

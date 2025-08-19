#!/usr/bin/env python3
"""
Test import fixes for app_launcher.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_app_launcher_imports():
    """Test that app_launcher can be imported without relative import errors"""
    print("🧪 Testing app_launcher.py import fixes...")
    
    try:
        # Test import of app_launcher
        from src.core.app_launcher import AppLauncher
        print("✅ AppLauncher imported successfully")
        
        # Test creating instance
        launcher = AppLauncher(debug=True)
        print("✅ AppLauncher instance created")
        
        # Test that the critical methods exist
        methods_to_check = ['launch_web_app', 'train_model', 'run_tests']
        for method in methods_to_check:
            if hasattr(launcher, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_optimized_app_import():
    """Test that optimized app can be imported"""
    print("\n🧪 Testing optimized app import...")
    
    try:
        from src.web.app_optimized import create_optimized_app, OptimizedFrameProcessor
        print("✅ Optimized app components imported successfully")
        
        # Test creating optimized app
        app, socketio = create_optimized_app()
        print("✅ Optimized app created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimized app import error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 TESTING IMPORT FIXES")
    print("=" * 40)
    
    results = []
    results.append(test_app_launcher_imports())
    results.append(test_optimized_app_import())
    
    print("\n" + "=" * 40)
    print("📋 RESULTS SUMMARY")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ Import fixes successful!")
        print("🚀 Ready to run: python src/core/app_launcher.py")
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("🔧 Check the errors above")

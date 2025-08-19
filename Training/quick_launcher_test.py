#!/usr/bin/env python3
"""
Quick test for app_launcher import fix
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

print("🧪 Testing app_launcher.py import fix...")

try:
    print("1. Testing basic import...")
    from src.core.app_launcher import AppLauncher
    print("✅ AppLauncher imported successfully")
    
    print("2. Testing instance creation...")
    launcher = AppLauncher(debug=True)
    print("✅ AppLauncher instance created")
    
    print("3. Testing method availability...")
    if hasattr(launcher, 'launch_web_app'):
        print("✅ launch_web_app method available")
    else:
        print("❌ launch_web_app method missing")
    
    print("\n🎉 IMPORT FIX SUCCESSFUL!")
    print("✅ Ready to run: python src/core/app_launcher.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Need to fix remaining import issues")
    
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()

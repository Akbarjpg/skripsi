"""
NumPy Fix Solution untuk Face Anti-Spoofing System
"""

import os
import sys
import subprocess

def check_numpy_installation():
    """Check current numpy status"""
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ NumPy location: {np.__file__}")
        return True
    except ImportError as e:
        print(f"✗ NumPy import error: {e}")
        return False
    except Exception as e:
        print(f"✗ NumPy error: {e}")
        return False

def fix_numpy_issue():
    """Attempt to fix numpy import issues"""
    print("🔧 ATTEMPTING NUMPY FIX...")
    
    # Method 1: Force reinstall with specific version
    try:
        print("Method 1: Installing numpy 1.24.4 (stable version)")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy==1.24.4", "--force-reinstall", "--no-cache-dir"
        ])
        return check_numpy_installation()
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Install with user flag
    try:
        print("Method 2: Installing with --user flag")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy", "--user", "--force-reinstall"
        ])
        return check_numpy_installation()
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    return False

def create_minimal_test():
    """Create minimal test without heavy dependencies"""
    test_code = '''
import sys
import os

def test_minimal_system():
    """Test sistem tanpa numpy dependencies berat"""
    print("🧪 TESTING MINIMAL SYSTEM...")
    
    # Test basic Python imports
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        print("✗ OpenCV not available")
    
    # Test Flask
    try:
        from flask import Flask, redirect
        print("✓ Flask with redirect available")
    except ImportError:
        print("✗ Flask not available")
    
    # Test basic file structure
    required_files = [
        "src/web/app.py",
        "src/models/simple_model.py",
        "launch.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
    
    print("✅ Minimal system test complete")

if __name__ == "__main__":
    test_minimal_system()
'''
    
    with open("minimal_test.py", "w") as f:
        f.write(test_code)
    
    print("✓ Created minimal_test.py")

def main():
    print("=" * 60)
    print("NUMPY IMPORT ISSUE - TROUBLESHOOTING")
    print("=" * 60)
    
    # Check current status
    print("\n1. CHECKING CURRENT NUMPY STATUS:")
    numpy_works = check_numpy_installation()
    
    if not numpy_works:
        print("\n2. ATTEMPTING FIXES:")
        numpy_works = fix_numpy_issue()
    
    if numpy_works:
        print("\n🎉 NUMPY FIXED! System should work now.")
        print("\nTry running: python launch.py --mode web")
    else:
        print("\n⚠️ NUMPY STILL HAS ISSUES")
        print("\n🔧 ALTERNATIVE SOLUTIONS:")
        print("1. Restart your Python interpreter/terminal")
        print("2. Use virtual environment:")
        print("   python -m venv venv")
        print("   venv\\Scripts\\activate")
        print("   pip install numpy opencv-python flask")
        print("3. Try conda instead of pip:")
        print("   conda install numpy opencv flask")
        
        # Create minimal test
        create_minimal_test()
        print("\n4. Or use minimal test: python minimal_test.py")

if __name__ == "__main__":
    main()

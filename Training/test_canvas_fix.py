#!/usr/bin/env python3
"""
Quick test for the automatic face registration fix
Tests the JavaScript canvas fix specifically
"""

import webbrowser
import time
import sys
from pathlib import Path

def test_canvas_fix():
    """Test the canvas.getImageData fix"""
    print("=== Canvas Fix Test ===\n")
    
    template_path = Path("src/web/templates/register_face.html")
    
    if not template_path.exists():
        print("❌ Template file not found!")
        return False
    
    # Read template content
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the error
    if "this.canvas.getImageData" in content:
        print("❌ ERROR: Found 'this.canvas.getImageData' - this will cause JavaScript error!")
        print("   Should be 'this.context.getImageData' instead")
        return False
    
    # Check for the correct implementation
    if "this.context.getImageData" in content:
        print("✅ Fixed: Found correct 'this.context.getImageData' implementation")
    else:
        print("⚠️  Warning: No getImageData calls found")
    
    # Check for other potential canvas-related errors
    error_patterns = [
        "canvas.drawImage",
        "canvas.fillRect", 
        "canvas.strokeRect"
    ]
    
    errors_found = []
    for pattern in error_patterns:
        if f"this.{pattern}" in content:
            errors_found.append(pattern)
    
    if errors_found:
        print(f"⚠️  Warning: Found potential canvas errors: {errors_found}")
        print("   These should probably use 'this.context' instead")
    
    # Check for required face detection components
    required_components = [
        "detectFace()",
        "simpleFaceDetection(imageData)",
        "startFaceDetection()",
        "this.context.drawImage"
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing components: {missing_components}")
        return False
    
    print("✅ All required face detection components found")
    print("\n=== Test Instructions ===")
    print("1. Start the server:")
    print("   python src/web/app_optimized.py")
    print("2. Open browser:")
    print("   http://localhost:5000/register_face")
    print("3. Check browser console (F12) for JavaScript errors")
    print("4. Verify automatic face detection works without errors")
    
    return True

def main():
    print("Testing Canvas getImageData Fix...")
    
    if test_canvas_fix():
        print("\n✅ Canvas fix test passed!")
        print("The JavaScript error should now be resolved.")
        
        # Try to open browser for manual testing
        try:
            print("\n🌐 Opening browser for manual testing...")
            time.sleep(1)
            webbrowser.open('http://localhost:5000/register_face')
        except:
            print("Could not open browser automatically")
        
        return True
    else:
        print("\n❌ Canvas fix test failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Simple test to verify face registration debugging is working
"""

def test_basic_implementation():
    """Basic test of debugging implementation"""
    print("🔍 Checking Face Registration Debug Implementation...")
    
    # Check backend
    try:
        with open('src/web/app_optimized.py', 'r', encoding='utf-8') as f:
            backend_content = f.read()
        
        if '=== CAPTURE FACE DEBUG ===' in backend_content:
            print("✅ Backend debugging found")
        else:
            print("❌ Backend debugging missing")
            return False
    except Exception as e:
        print(f"❌ Error reading backend: {e}")
        return False
    
    # Check frontend
    try:
        with open('src/web/templates/register_face.html', 'r', encoding='utf-8') as f:
            frontend_content = f.read()
        
        if 'setupSocketDebug()' in frontend_content:
            print("✅ Frontend debugging found")
        else:
            print("❌ Frontend debugging missing")
            return False
    except Exception as e:
        print(f"❌ Error reading frontend: {e}")
        return False
    
    print("\n✅ Debug implementation looks good!")
    print("\n📋 To test the fix:")
    print("1. Start server: python src/web/app_optimized.py")
    print("2. Open: http://localhost:5000/register-face")
    print("3. Open browser console (F12)")
    print("4. Try face registration and check console output")
    
    return True

if __name__ == "__main__":
    test_basic_implementation()

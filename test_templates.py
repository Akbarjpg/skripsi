"""
Test template loading fix
"""

def test_template_loading():
    """Test if Flask can find templates correctly"""
    try:
        import os
        from flask import Flask, render_template
        
        # Test fallback app configuration
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'web', 'templates')
        
        print(f"Template directory: {template_dir}")
        print(f"Template directory exists: {os.path.exists(template_dir)}")
        
        if os.path.exists(template_dir):
            files = os.listdir(template_dir)
            print(f"Templates found: {files}")
        
        # Test app creation
        app = Flask(__name__, template_folder=template_dir)
        
        @app.route('/')
        def home():
            return render_template('index.html')
        
        # Test template rendering
        with app.test_client() as client:
            response = client.get('/')
            print(f"Response status: {response.status_code}")
            if response.status_code == 200:
                print("✓ Template loading successful!")
                return True
            else:
                print(f"✗ Template loading failed with status {response.status_code}")
                return False
        
    except Exception as e:
        print(f"✗ Template test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("TESTING TEMPLATE LOADING FIX")
    print("="*50)
    
    success = test_template_loading()
    
    if success:
        print("\n🎉 TEMPLATE ISSUE FIXED!")
        print("✅ Flask can now find index.html and other templates")
        print("✅ Web application should work properly")
    else:
        print("\n❌ Still has template issues")
    
    print("="*50)

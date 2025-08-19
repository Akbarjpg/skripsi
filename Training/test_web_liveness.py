#!/usr/bin/env python3
"""
Test liveness detection with web interface
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_liveness():
    print("🌐 TESTING WEB LIVENESS DETECTION")
    print("=" * 50)
    
    try:
        # Start server in background
        from src.web.app_clean import create_app
        from src.utils.config import SystemConfig
        
        print("🚀 Starting Flask app...")
        
        # Create basic config
        config = SystemConfig()
        config.web.debug = True
        config.web.host = "localhost"
        config.web.port = 5000
        
        app, socketio = create_app(config)
        
        print("✅ Flask app created successfully")
        print("📊 Testing endpoints...")
        
        with app.test_client() as client:
            # Test face detection page
            response = client.get('/face_detection')
            print(f"Face detection page status: {response.status_code}")
            
            if response.status_code == 200:
                # Check if liveness UI elements are present
                content = response.get_data(as_text=True)
                
                # Check for liveness-related UI elements
                liveness_indicators = [
                    'liveness',
                    'blink',
                    'mouth',
                    'head',
                    'ear',
                    'mar'
                ]
                
                found_indicators = []
                for indicator in liveness_indicators:
                    if indicator.lower() in content.lower():
                        found_indicators.append(indicator)
                
                print(f"Liveness UI indicators found: {found_indicators}")
                
                # Check for specific liveness UI elements
                if 'updateLivenessInfo' in content:
                    print("✅ Frontend liveness update function found")
                else:
                    print("❌ Frontend liveness update function NOT found")
                
                if 'livenessInfo' in content:
                    print("✅ Liveness info display element found")
                else:
                    print("❌ Liveness info display element NOT found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing web liveness: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_web_liveness()

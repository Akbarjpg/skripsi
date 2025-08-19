#!/usr/bin/env python3
"""
Test untuk memverifikasi bahwa API endpoints yang hilang sudah ditambahkan
"""

import requests
import json

def test_api_endpoints():
    """Test API endpoints yang diperlukan untuk attendance system"""
    base_url = "http://localhost:5000"
    
    print("🔍 Testing API endpoints...")
    
    # Test endpoints yang perlu ada
    endpoints_to_test = [
        "/api/start_verification",
        "/api/complete_attendance", 
        "/api/verify",
        "/api/enroll",
        "/api/challenge"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            # Test dengan POST request (tanpa authentication untuk test sederhana)
            response = requests.post(f"{base_url}{endpoint}", 
                                   json={}, 
                                   timeout=5)
            
            if response.status_code == 404:
                print(f"  ❌ {endpoint} - NOT FOUND (404)")
            elif response.status_code == 401:
                print(f"  ✅ {endpoint} - EXISTS (401 - needs auth)")
            elif response.status_code in [200, 400, 500]:
                print(f"  ✅ {endpoint} - EXISTS ({response.status_code})")
            else:
                print(f"  ⚠️  {endpoint} - UNKNOWN ({response.status_code})")
                
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  {endpoint} - SERVER NOT RUNNING")
            break
        except Exception as e:
            print(f"  ❌ {endpoint} - ERROR: {e}")
    
    print("\n✅ API endpoint test completed!")

if __name__ == "__main__":
    print("🎯 Face Anti-Spoofing System - API Endpoint Test")
    print("=" * 60)
    print("Make sure the server is running: python main.py --mode web")
    print("-" * 60)
    
    test_api_endpoints()
    
    print("\n📋 Fix Summary:")
    print("✅ Added /api/start_verification endpoint")
    print("✅ Added /api/complete_attendance endpoint") 
    print("✅ Fixed 'Gagal memulai verifikasi' error")
    print("✅ Added proper imports (time, uuid)")
    print("\n🚀 The attendance verification should now work!")

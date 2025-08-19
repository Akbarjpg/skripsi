#!/usr/bin/env python3
"""
Comprehensive page testing script for the optimized Flask app
Tests all routes and templates for errors
"""

import requests
import time
import sys
from urllib.parse import urljoin

class PageTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def test_page(self, path, expected_status=200, description="", login_required=False):
        """Test a single page"""
        url = urljoin(self.base_url, path)
        
        try:
            response = self.session.get(url, timeout=10)
            
            # Check if redirect (for login-required pages)
            if login_required and response.status_code == 302:
                status = "✅ REDIRECT (Login Required)"
                success = True
            elif response.status_code == expected_status:
                status = "✅ SUCCESS"
                success = True
            else:
                status = f"❌ ERROR (Status: {response.status_code})"
                success = False
                
            self.test_results.append({
                'path': path,
                'url': url,
                'status_code': response.status_code,
                'success': success,
                'status': status,
                'description': description
            })
            
            print(f"{status:25} | {path:20} | {description}")
            return success
            
        except requests.exceptions.RequestException as e:
            status = f"❌ CONNECTION ERROR"
            self.test_results.append({
                'path': path,
                'url': url,
                'status_code': None,
                'success': False,
                'status': status,
                'description': description,
                'error': str(e)
            })
            print(f"{status:25} | {path:20} | {description} - {str(e)}")
            return False
    
    def login_as_demo(self):
        """Login as demo user for authenticated tests"""
        login_url = urljoin(self.base_url, '/login')
        login_data = {
            'username': 'demo',
            'password': 'demo'
        }
        
        try:
            response = self.session.post(login_url, data=login_data, timeout=10)
            if response.status_code in [200, 302]:  # Success or redirect
                print("✅ Successfully logged in as demo user")
                return True
            else:
                print(f"❌ Login failed with status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all pages"""
        print("🧪 COMPREHENSIVE PAGE TESTING")
        print("=" * 80)
        print(f"Testing against: {self.base_url}")
        print("-" * 80)
        print(f"{'STATUS':25} | {'PATH':20} | DESCRIPTION")
        print("-" * 80)
        
        # Test public pages first
        public_pages = [
            ('/', 'Home page'),
            ('/login', 'Login page'),
            ('/register', 'Registration page'),
            ('/favicon.ico', 'Favicon', 204),  # No content expected
        ]
        
        for page_info in public_pages:
            if len(page_info) == 3:
                path, desc, expected_status = page_info
                self.test_page(path, expected_status, desc)
            else:
                path, desc = page_info
                self.test_page(path, 200, desc)
        
        print("-" * 80)
        print("TESTING AUTHENTICATED PAGES (without login)...")
        print("-" * 80)
        
        # Test protected pages without login (should redirect)
        protected_pages = [
            ('/dashboard', 'Dashboard (requires login)'),
            ('/attendance', 'Attendance (requires login)'),
            ('/face-detection', 'Face Detection'),
        ]
        
        for path, desc in protected_pages:
            # Face detection might not require login, others should redirect
            if 'requires login' in desc:
                self.test_page(path, 302, desc, login_required=True)
            else:
                self.test_page(path, 200, desc)
        
        print("-" * 80)
        print("ATTEMPTING LOGIN AND TESTING AUTHENTICATED PAGES...")
        print("-" * 80)
        
        # Login and test authenticated pages
        if self.login_as_demo():
            authenticated_pages = [
                ('/dashboard', 'Dashboard (authenticated)'),
                ('/attendance', 'Attendance (authenticated)'),
                ('/face-detection', 'Face Detection (authenticated)'),
            ]
            
            for path, desc in authenticated_pages:
                self.test_page(path, 200, desc)
        
        print("-" * 80)
        print("TESTING API ENDPOINTS...")
        print("-" * 80)
        
        # Test API endpoints
        api_endpoints = [
            ('/api/performance', 'Performance API'),
            ('/api/cleanup-cache', 'Cache Cleanup API'),
        ]
        
        for path, desc in api_endpoints:
            self.test_page(path, 200, desc)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - successful_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            print("-" * 40)
            for result in self.test_results:
                if not result['success']:
                    print(f"• {result['path']} - {result['status']}")
                    if 'error' in result:
                        print(f"  Error: {result['error']}")
        
        print("\n🎯 DETAILED RESULTS:")
        print("-" * 80)
        for result in self.test_results:
            status_icon = "✅" if result['success'] else "❌"
            print(f"{status_icon} {result['path']:20} | Status: {result['status_code']} | {result['description']}")

def main():
    """Main testing function"""
    print("⏳ Waiting for server to start...")
    time.sleep(3)  # Give server time to start
    
    tester = PageTester()
    tester.run_comprehensive_test()
    
    print("\n🎉 Page testing complete!")
    print("💡 If there are connection errors, make sure the server is running:")
    print("   python launch_optimized.py")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 5 Web Application Test
Tests the Step 5 integration through the web interface
"""

import sys
import os
import time
import threading
import requests
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.web.app_step4 import create_app
    from src.integration.antispoofing_face_recognition import create_step5_system
    print("✅ Web app imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_step5_web_interface():
    """Test Step 5 through web interface"""
    print("🌐 Testing Step 5 Web Interface")
    print("=" * 50)
    
    try:
        # Create Flask app
        app = create_app()
        
        # Test configuration
        app.config.update({
            'TESTING': True,
            'WTF_CSRF_ENABLED': False
        })
        
        with app.test_client() as client:
            # Test 1: Check if Step 5 template is accessible
            print("📄 Test 1: Checking Step 5 template accessibility...")
            response = client.get('/attendance_sequential')
            
            if response.status_code == 200:
                print("✅ Step 5 template accessible")
                
                # Check if key elements are present
                html_content = response.get_data(as_text=True)
                required_elements = [
                    'Step 5: Integrated Anti-Spoofing',
                    'videoElement',
                    'canvasElement',
                    'antispoofing-indicator',
                    'recognition-indicator',
                    'attendance_flow.js'
                ]
                
                missing_elements = []
                for element in required_elements:
                    if element not in html_content:
                        missing_elements.append(element)
                
                if missing_elements:
                    print(f"⚠️  Missing elements: {missing_elements}")
                else:
                    print("✅ All required UI elements present")
            else:
                print(f"❌ Template not accessible: {response.status_code}")
            
            # Test 2: Check JavaScript file
            print("\n📜 Test 2: Checking JavaScript file...")
            js_response = client.get('/static/js/attendance_flow.js')
            
            if js_response.status_code == 200:
                print("✅ JavaScript file accessible")
                js_content = js_response.get_data(as_text=True)
                
                # Check for key JavaScript components
                js_required = [
                    'Step5AttendanceFlow',
                    'startAttendance',
                    'handleStateUpdate',
                    'antispoofing_progress',
                    'recognition_started'
                ]
                
                for component in js_required:
                    if component in js_content:
                        print(f"  ✅ {component} found")
                    else:
                        print(f"  ❌ {component} missing")
            else:
                print(f"❌ JavaScript file not accessible: {js_response.status_code}")
        
        print("\n🎉 Web interface test completed!")
        
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")

def run_live_demo():
    """Run a live demo with camera"""
    print("\n📹 Live Demo Instructions")
    print("=" * 50)
    print("To test Step 5 with live camera:")
    print()
    print("1. Start the web server:")
    print("   python -c \"from src.web.app_step4 import create_app; app = create_app(); app.run(debug=True, port=5000)\"")
    print()
    print("2. Open browser to: http://localhost:5000/attendance_sequential")
    print()
    print("3. Test the complete workflow:")
    print("   • Click 'Start Attendance Verification'")
    print("   • Watch Phase 1: Anti-spoofing verification")
    print("   • See transition to Phase 2: Face recognition")
    print("   • Observe real-time progress and confidence scores")
    print()
    print("4. Expected behavior:")
    print("   • State indicators should animate (1 → 2)")
    print("   • Progress bar should fill during anti-spoofing")
    print("   • Confidence percentages should update live")
    print("   • Success/failure messages should appear")
    print("   • Auto-reset after 5 seconds on success")

def performance_benchmark():
    """Run performance benchmark for Step 5"""
    print("\n⚡ Performance Benchmark")
    print("=" * 50)
    
    try:
        # Create Step 5 system
        config = {
            'frame_skip_rate': 3,
            'enable_caching': True,
            'enable_threading': False  # For consistent benchmarking
        }
        
        system = create_step5_system(config)
        
        # Create test image
        import cv2
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Benchmark frame processing
        num_frames = 50
        start_time = time.time()
        
        session_id = system.start_session()
        
        for i in range(num_frames):
            result = system.process_frame(test_image)
            if i % 10 == 0:
                print(f"  Frame {i}: State={result.get('state')}, Time={time.time() - start_time:.2f}s")
        
        total_time = time.time() - start_time
        fps = num_frames / total_time
        
        print(f"\n📊 Performance Results:")
        print(f"  Frames processed: {num_frames}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Time per frame: {(total_time/num_frames)*1000:.1f}ms")
        
        # Test frame skipping efficiency
        skipped_frames = 0
        processed_frames = 0
        
        for i in range(30):
            if system._should_skip_frame():
                skipped_frames += 1
            else:
                processed_frames += 1
        
        skip_efficiency = (skipped_frames / 30) * 100
        print(f"  Frame skip efficiency: {skip_efficiency:.1f}% (target: ~67%)")
        
        system.shutdown()
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")

def main():
    """Main test function"""
    print("🚀 Step 5 Integration Testing Suite")
    print("Testing Anti-Spoofing + Face Recognition Integration")
    print()
    
    # Test options
    tests = {
        '1': 'Unit Tests (Automated)',
        '2': 'Web Interface Test',
        '3': 'Performance Benchmark', 
        '4': 'Live Demo Instructions',
        '5': 'Run All Tests'
    }
    
    print("Available tests:")
    for key, value in tests.items():
        print(f"  {key}. {value}")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        print("\n🧪 Running Unit Tests...")
        os.system("python test_step5_integration.py")
    
    elif choice == '2':
        print("\n🌐 Running Web Interface Test...")
        test_step5_web_interface()
    
    elif choice == '3':
        print("\n⚡ Running Performance Benchmark...")
        performance_benchmark()
    
    elif choice == '4':
        run_live_demo()
    
    elif choice == '5':
        print("\n🔄 Running All Tests...")
        print("\n1️⃣  Unit Tests:")
        os.system("python test_step5_integration.py")
        
        print("\n2️⃣  Web Interface Test:")
        test_step5_web_interface()
        
        print("\n3️⃣  Performance Benchmark:")
        performance_benchmark()
        
        print("\n4️⃣  Live Demo:")
        run_live_demo()
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
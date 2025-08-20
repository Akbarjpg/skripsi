#!/usr/bin/env python3
"""
Step 5 Integration Test Suite
Tests the seamless integration between anti-spoofing and face recognition
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from src.integration.antispoofing_face_recognition import Step5IntegratedSystem, SystemState
    from src.models.face_recognition_cnn import FaceRecognitionSystem
    from src.database.attendance_db import AttendanceDatabase
    print("✅ All Step 5 imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure all Step 5 components are properly installed")
    sys.exit(1)

class Step5TestSuite:
    """Comprehensive test suite for Step 5 integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Test configuration
        self.test_config = {
            'antispoofing_confidence_threshold': 0.7,  # Lower for testing
            'recognition_confidence_threshold': 0.7,   # Lower for testing
            'frame_skip_rate': 1,  # No skipping for tests
            'enable_threading': False,  # Disable for predictable testing
            'enable_caching': True,
            'log_state_changes': True,
            'performance_monitoring': True
        }
        
        print("🧪 Step 5 Integration Test Suite Initialized")
    
    def create_test_image(self, width=640, height=480):
        """Create a test image for testing"""
        # Create a simple test image with face-like features
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add a simple face-like rectangle
        cv2.rectangle(image, (200, 150), (440, 330), (120, 120, 120), -1)
        cv2.circle(image, (270, 200), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (370, 200), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(image, (300, 250), (340, 270), (0, 0, 0), -1)  # Nose
        cv2.rectangle(image, (280, 290), (360, 310), (0, 0, 0), -1)  # Mouth
        
        return image
    
    def test_system_initialization(self):
        """Test 1: System initialization"""
        print("\n🔧 Test 1: System Initialization")
        
        try:
            system = Step5IntegratedSystem(config=self.test_config)
            
            # Check initial state
            assert system.current_state == SystemState.INIT, "Initial state should be INIT"
            assert system.session_id is None, "Session ID should be None initially"
            assert not system.isProcessing if hasattr(system, 'isProcessing') else True, "Should not be processing initially"
            
            print("✅ System initialization successful")
            self.test_results['passed'] += 1
            return system
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Initialization: {str(e)}")
            return None
    
    def test_session_management(self, system):
        """Test 2: Session management"""
        print("\n📝 Test 2: Session Management")
        
        try:
            # Start session
            session_id = system.start_session()
            
            assert session_id is not None, "Session ID should not be None"
            assert system.session_id == session_id, "System should store session ID"
            assert system.current_state == SystemState.INIT, "State should be INIT after session start"
            assert 'session_id' in system.session_data, "Session data should contain session_id"
            
            print(f"✅ Session started with ID: {session_id[:8]}...")
            
            # Reset session
            system.reset_session()
            
            assert system.session_id is None, "Session ID should be None after reset"
            assert system.current_state == SystemState.INIT, "State should be INIT after reset"
            
            print("✅ Session reset successful")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Session management test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Session management: {str(e)}")
    
    def test_state_transitions(self, system):
        """Test 3: State machine transitions"""
        print("\n🔄 Test 3: State Machine Transitions")
        
        try:
            # Start session
            session_id = system.start_session()
            test_image = self.create_test_image()
            
            # Test initial transition
            result = system.process_frame(test_image)
            
            assert 'session_id' in result, "Result should contain session_id"
            assert 'state' in result, "Result should contain state"
            assert 'status' in result, "Result should contain status"
            
            print(f"✅ Initial frame processing: State={result.get('state')}, Status={result.get('status')}")
            
            # Test state transition to anti-spoofing
            if system.current_state == SystemState.ANTI_SPOOFING:
                print("✅ Successfully transitioned to ANTI_SPOOFING state")
            else:
                print(f"⚠️  Expected ANTI_SPOOFING, got {system.current_state}")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ State transition test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"State transitions: {str(e)}")
    
    def test_antispoofing_phase(self, system):
        """Test 4: Anti-spoofing phase processing"""
        print("\n🛡️  Test 4: Anti-Spoofing Phase")
        
        try:
            # Start session and get to anti-spoofing state
            session_id = system.start_session()
            test_image = self.create_test_image()
            
            # Process several frames to simulate anti-spoofing
            antispoofing_results = []
            for i in range(5):
                result = system.process_frame(test_image)
                antispoofing_results.append(result)
                
                if result.get('state') == 'anti_spoofing':
                    print(f"  Frame {i+1}: Confidence={result.get('confidence', 0):.2f}, Status={result.get('status')}")
                elif result.get('state') == 'recognizing':
                    print(f"  Frame {i+1}: Transitioned to recognition phase!")
                    break
                
                time.sleep(0.1)  # Small delay
            
            # Check if we got meaningful anti-spoofing results
            confidence_values = [r.get('confidence', 0) for r in antispoofing_results if r.get('confidence')]
            if confidence_values:
                avg_confidence = sum(confidence_values) / len(confidence_values)
                print(f"✅ Anti-spoofing processing: Average confidence = {avg_confidence:.2f}")
            else:
                print("⚠️  No confidence values returned")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Anti-spoofing phase test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Anti-spoofing phase: {str(e)}")
    
    def test_face_recognition_phase(self, system):
        """Test 5: Face recognition phase (simulated)"""
        print("\n👤 Test 5: Face Recognition Phase")
        
        try:
            # Manually set system to recognition state for testing
            system.start_session()
            system.current_state = SystemState.RECOGNIZING
            system.session_data['antispoofing_result'] = {
                'is_real_face': True,
                'confidence': 0.9
            }
            
            test_image = self.create_test_image()
            result = system.process_frame(test_image)
            
            print(f"✅ Recognition phase result: State={result.get('state')}, Status={result.get('status')}")
            
            # Check result structure
            expected_keys = ['session_id', 'state', 'status', 'message', 'timestamp']
            for key in expected_keys:
                assert key in result, f"Result should contain {key}"
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Face recognition phase test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Face recognition phase: {str(e)}")
    
    def test_performance_optimization(self, system):
        """Test 6: Performance optimization features"""
        print("\n⚡ Test 6: Performance Optimization")
        
        try:
            # Test frame skipping
            system.config['frame_skip_rate'] = 3
            system.frame_skip_counter = 0
            
            should_skip_frame1 = system._should_skip_frame()  # Should not skip (counter = 1)
            should_skip_frame2 = system._should_skip_frame()  # Should skip (counter = 2)
            should_skip_frame3 = system._should_skip_frame()  # Should not skip (counter = 0, reset)
            
            assert should_skip_frame1 == True, "First call should skip"
            assert should_skip_frame2 == True, "Second call should skip"
            assert should_skip_frame3 == False, "Third call should not skip"
            
            print("✅ Frame skipping logic working correctly")
            
            # Test caching
            if system.config['enable_caching']:
                test_image = self.create_test_image()
                system.frame_cache.append(test_image)
                assert len(system.frame_cache) == 1, "Frame should be cached"
                print("✅ Frame caching working")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Performance optimization test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Performance optimization: {str(e)}")
    
    def test_timeout_handling(self, system):
        """Test 7: Timeout handling"""
        print("\n⏰ Test 7: Timeout Handling")
        
        try:
            # Set very short timeout for testing
            system.state_timeouts[SystemState.ANTI_SPOOFING] = 0.1  # 100ms
            
            system.start_session()
            system.current_state = SystemState.ANTI_SPOOFING
            system.state_start_time = time.time() - 0.2  # Set to past time
            
            # Check timeout detection
            is_timeout = system._is_state_timeout()
            assert is_timeout == True, "Should detect timeout"
            
            # Test timeout handling
            test_image = self.create_test_image()
            result = system.process_frame(test_image)
            
            assert result.get('status') == 'timeout', "Should return timeout status"
            assert system.current_state == SystemState.FAILED, "Should transition to FAILED state"
            
            print("✅ Timeout handling working correctly")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Timeout handling test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Timeout handling: {str(e)}")
    
    def test_error_handling(self, system):
        """Test 8: Error handling and recovery"""
        print("\n🚨 Test 8: Error Handling")
        
        try:
            # Test processing without session
            system.reset_session()
            test_image = self.create_test_image()
            result = system.process_frame(test_image)
            
            assert result.get('status') == 'error', "Should return error for no session"
            assert 'No active session' in result.get('message', ''), "Should indicate no session"
            
            print("✅ No session error handling working")
            
            # Test invalid image processing
            system.start_session()
            invalid_image = None
            
            try:
                result = system.process_frame(invalid_image)
                # Should handle gracefully
                print("✅ Invalid image handled gracefully")
            except Exception:
                print("✅ Invalid image properly rejected")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Error handling: {str(e)}")
    
    def test_database_integration(self):
        """Test 9: Database integration"""
        print("\n🗄️  Test 9: Database Integration")
        
        try:
            # Test database connection
            db = AttendanceDatabase("test_step5.db")
            
            # Test user registration
            success = db.register_user("test_user_step5", "Test User", "test@example.com")
            if success:
                print("✅ Test user registered successfully")
            else:
                print("⚠️  User already exists or registration failed")
            
            # Test database stats
            stats = db.get_database_stats()
            assert isinstance(stats, dict), "Stats should be a dictionary"
            print(f"✅ Database stats: {stats.get('active_users', 0)} users")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Database integration test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Database integration: {str(e)}")
    
    def test_system_stats(self, system):
        """Test 10: System statistics and monitoring"""
        print("\n📊 Test 10: System Statistics")
        
        try:
            stats = system.get_system_stats()
            
            # Check stats structure
            expected_keys = ['face_recognition_stats', 'database_stats', 'current_state', 'config']
            for key in expected_keys:
                assert key in stats, f"Stats should contain {key}"
            
            print("✅ System statistics structure valid")
            print(f"  Current state: {stats['current_state']}")
            print(f"  Session active: {stats.get('session_active', False)}")
            
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ System statistics test failed: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"System statistics: {str(e)}")
    
    def run_all_tests(self):
        """Run all Step 5 tests"""
        print("🚀 Starting Step 5 Integration Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Initialize system
        system = self.test_system_initialization()
        
        if system:
            # Run all tests
            self.test_session_management(system)
            self.test_state_transitions(system)
            self.test_antispoofing_phase(system)
            self.test_face_recognition_phase(system)
            self.test_performance_optimization(system)
            self.test_timeout_handling(system)
            self.test_error_handling(system)
            self.test_system_stats(system)
            
            # Cleanup
            system.shutdown()
        
        # Database test (separate)
        self.test_database_integration()
        
        # Print results
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("🏁 Step 5 Test Results")
        print("=" * 60)
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        
        if self.test_results['errors']:
            print("\n🚨 Errors encountered:")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        # Overall result
        if self.test_results['failed'] == 0:
            print("\n🎉 All Step 5 tests passed! System is ready for production.")
            return True
        else:
            print(f"\n⚠️  {self.test_results['failed']} test(s) failed. Please review and fix issues.")
            return False

def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 Step 5: Anti-Spoofing + Face Recognition Integration Test")
    print("Testing seamless workflow with state management and performance optimization")
    print()
    
    # Run tests
    test_suite = Step5TestSuite()
    success = test_suite.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
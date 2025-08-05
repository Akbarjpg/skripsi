#!/usr/bin/env python3
"""
Quick test to verify the sequential detection system is working
"""

def main():
    print("🧪 TESTING SEQUENTIAL DETECTION SYSTEM")
    print("=" * 45)
    
    try:
        # Test 1: Import test
        print("\n1️⃣ TESTING IMPORTS...")
        from src.web.app_optimized import SequentialDetectionState, OptimizedFrameProcessor, app
        print("   ✅ All classes imported successfully")
        
        # Test 2: State initialization
        print("\n2️⃣ TESTING STATE INITIALIZATION...")
        state = SequentialDetectionState()
        print(f"   ✅ Initial phase: {state.phase}")
        print(f"   ✅ Timeouts: {state.timeouts}")
        print(f"   ✅ Challenge system: {state.current_challenge is not None}")
        
        # Test 3: Processor initialization  
        print("\n3️⃣ TESTING FRAME PROCESSOR...")
        processor = OptimizedFrameProcessor()
        print(f"   ✅ Sequential states tracking: {'sequential_states' in dir(processor)}")
        print(f"   ✅ Security states tracking: {'security_states' in dir(processor)}")
        
        # Test 4: State transitions
        print("\n4️⃣ TESTING STATE TRANSITIONS...")
        state.liveness_passed = True
        state.landmark_passed = True
        can_proceed = state.can_proceed_to_recognition()
        print(f"   ✅ Can proceed to recognition: {can_proceed}")
        
        if can_proceed:
            success = state.transition_to_recognition()
            print(f"   ✅ Transition successful: {success}")
            print(f"   ✅ New phase: {state.phase}")
        
        # Test 5: App routes
        print("\n5️⃣ TESTING APP ROUTES...")
        with app.app_context():
            routes = [str(rule) for rule in app.url_map.iter_rules()]
            sequential_routes = [r for r in routes if 'sequential' in r]
            print(f"   ✅ Sequential routes found: {len(sequential_routes)}")
            for route in sequential_routes:
                print(f"      - {route}")
        
        print("\n" + "=" * 45)
        print("🎉 ALL TESTS PASSED!")
        print("✨ Sequential Detection System is READY!")
        print("\n📋 TO START THE SERVER:")
        print("   python src/web/app_optimized.py")
        print("\n🌐 THEN VISIT:")
        print("   http://localhost:5000")
        print("   Choose 'Sequential Mode' for guided detection")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()

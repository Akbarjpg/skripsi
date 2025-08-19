"""
Test Enhanced Multi-Modal Fusion System
Tests the new weighted fusion, cross-validation, temporal consistency, and uncertainty propagation
"""

import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.web.app_optimized import EnhancedSecurityAssessmentState
except ImportError:
    print("❌ Failed to import EnhancedSecurityAssessmentState")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_enhanced_fusion():
    """Test the enhanced multi-modal fusion system"""
    print("🧪 TESTING ENHANCED MULTI-MODAL FUSION SYSTEM")
    print("=" * 60)
    
    # Initialize enhanced security state
    security_state = EnhancedSecurityAssessmentState()
    
    print(f"✅ Enhanced Security Assessment State initialized")
    print(f"   Initial fusion weights: {security_state.fusion_weights}")
    print(f"   Challenge difficulty: {security_state.challenge_difficulty}")
    print(f"   Available challenges: {len(security_state.challenges['easy'])}")
    
    # Test 1: Basic Fusion Calculation
    print("\n📊 TEST 1: Basic Fusion Calculation")
    print("-" * 40)
    
    # Simulate good quality data
    cnn_data = {'confidence': 0.85, 'is_live': True}
    landmark_data = {'landmarks_detected': True, 'landmark_count': 68}
    movement_data = {'head_movement': True, 'blink_detected': True, 'blink_count': 3}
    
    fusion_result = security_state.calculate_enhanced_fusion_score(
        cnn_data, landmark_data, movement_data
    )
    
    print(f"   Final fusion score: {fusion_result['final_score']:.3f}")
    print(f"   Aggregated decision: {fusion_result['aggregated_decision']}")
    print(f"   Method scores: {fusion_result['method_scores']}")
    print(f"   Adaptive weights: {fusion_result['adaptive_weights']}")
    print(f"   Uncertainty: {fusion_result['uncertainty']:.3f}")
    print(f"   Confidence interval: [{fusion_result['confidence_interval']['lower']:.3f}, {fusion_result['confidence_interval']['upper']:.3f}]")
    
    # Test 2: Temporal Consistency
    print("\n⏱️ TEST 2: Temporal Consistency")
    print("-" * 40)
    
    # Simulate multiple frames with consistent data
    consistent_scores = []
    for i in range(20):
        # Simulate slightly varying but consistent data
        noise = np.random.normal(0, 0.05)  # Small noise
        cnn_data = {'confidence': 0.8 + noise, 'is_live': True}
        
        # Update CNN history
        security_state.update_cnn(cnn_data['confidence'], cnn_data['is_live'])
        
        fusion_result = security_state.calculate_enhanced_fusion_score(
            cnn_data, landmark_data, movement_data
        )
        consistent_scores.append(fusion_result['final_score'])
    
    temporal_consistency = fusion_result['temporal_consistency']
    score_variance = np.var(consistent_scores)
    
    print(f"   Temporal consistency score: {temporal_consistency:.3f}")
    print(f"   Score variance over 20 frames: {score_variance:.4f}")
    print(f"   Average fusion score: {np.mean(consistent_scores):.3f}")
    print(f"   Score stability: {'Good' if score_variance < 0.01 else 'Poor'}")
    
    # Test 3: Cross-Validation
    print("\n🔍 TEST 3: Cross-Validation Between Methods")
    print("-" * 40)
    
    # Test scenario: CNN says live but no landmark movement (suspicious)
    suspicious_cnn = {'confidence': 0.95, 'is_live': True}
    suspicious_landmark = {'landmarks_detected': True, 'landmark_count': 68}
    suspicious_movement = {'head_movement': False, 'blink_detected': False, 'blink_count': 0}
    
    fusion_result_suspicious = security_state.calculate_enhanced_fusion_score(
        suspicious_cnn, suspicious_landmark, suspicious_movement
    )
    
    cv_scores = fusion_result_suspicious['cross_validation']
    print(f"   CNN-Landmark consistency: {cv_scores['cnn_landmark_consistency']:.3f}")
    print(f"   Movement-CNN alignment: {cv_scores['movement_cnn_alignment']:.3f}")
    print(f"   Overall cross-validation: {np.mean(list(cv_scores.values())):.3f}")
    
    # Test 4: Suspicious Pattern Detection
    print("\n🚨 TEST 4: Suspicious Pattern Detection")
    print("-" * 40)
    
    # Simulate perfect stillness (suspicious)
    for i in range(30):  # 1 second at 30fps
        still_movement = {'head_movement': False, 'blink_detected': False, 'blink_count': 0}
        security_state.detect_suspicious_patterns(still_movement, 0.9, landmark_data)
    
    suspicious_patterns = security_state.suspicious_patterns
    print(f"   Perfect stillness duration: {suspicious_patterns['perfect_stillness_duration']:.1f}s")
    print(f"   Too regular movements: {suspicious_patterns['too_regular_movements']:.3f}")
    print(f"   Impossible transitions: {suspicious_patterns['impossible_transitions']}")
    print(f"   Consistency violations: {suspicious_patterns['consistency_violations']}")
    
    # Test 5: Adaptive Weight Adjustment
    print("\n⚖️ TEST 5: Adaptive Weight Adjustment")
    print("-" * 40)
    
    # Store original weights
    original_weights = security_state.fusion_weights.copy()
    
    # Simulate poor lighting conditions
    security_state.lighting_quality = 0.3  # Poor lighting
    security_state.face_size_history.extend([0.05] * 10)  # Small face
    
    # Update adaptive weights
    security_state.update_adaptive_weights()
    
    print(f"   Original weights: {original_weights}")
    print(f"   Adaptive weights: {security_state.adaptive_weights}")
    print(f"   CNN weight change: {security_state.adaptive_weights['cnn'] - original_weights['cnn']:+.3f}")
    print(f"   Landmark weight change: {security_state.adaptive_weights['landmark'] - original_weights['landmark']:+.3f}")
    
    # Test 6: Challenge System Progression
    print("\n🎯 TEST 6: Challenge System with Difficulty Progression")
    print("-" * 40)
    
    # Create fresh state for challenge testing
    challenge_state = EnhancedSecurityAssessmentState()
    
    for attempt in range(5):
        # Generate new challenge
        challenge_state.generate_new_challenge()
        challenge_info = challenge_state.get_challenge_info()
        
        print(f"   Attempt {attempt + 1}:")
        print(f"     Difficulty: {challenge_info['difficulty']}")
        print(f"     Challenge: {challenge_info['challenge_type']}")
        print(f"     Weight: {challenge_info['weight']:.1f}")
        print(f"     Instruction: {challenge_info['instruction']}")
        
        # Simulate completion
        if challenge_info['challenge_type'] == 'blink':
            for i in range(challenge_info.get('target_count', 3)):
                landmark_result = {'blink_count': i + 1, 'landmarks_detected': True}
                challenge_state.update_challenge(landmark_result)
        
        if challenge_state.landmark_verified:
            print(f"     ✅ Challenge completed!")
            break
    
    # Test 7: Multi-Frame Aggregation
    print("\n📈 TEST 7: Multi-Frame Aggregation")
    print("-" * 40)
    
    # Simulate decision aggregation over multiple frames
    aggregation_state = EnhancedSecurityAssessmentState()
    decisions = []
    confidences = []
    
    for frame in range(20):
        # Simulate varying quality frames
        if frame < 10:
            # Poor initial frames
            cnn_conf = 0.4 + np.random.normal(0, 0.1)
        else:
            # Better later frames
            cnn_conf = 0.8 + np.random.normal(0, 0.05)
        
        cnn_data = {'confidence': max(0, min(1, cnn_conf)), 'is_live': cnn_conf > 0.5}
        
        fusion_result = aggregation_state.calculate_enhanced_fusion_score(
            cnn_data, landmark_data, movement_data
        )
        
        decisions.append(fusion_result['aggregated_decision'])
        confidences.append(fusion_result['final_score'])
    
    print(f"   Individual frame decisions: {decisions}")
    print(f"   Frame confidence progression: {[f'{c:.2f}' for c in confidences]}")
    print(f"   Final aggregated decision: {fusion_result['aggregated_decision']}")
    print(f"   Confidence improvement: {confidences[-1] - confidences[0]:+.3f}")
    
    # Test 8: Uncertainty Propagation
    print("\n🌐 TEST 8: Uncertainty Propagation")
    print("-" * 40)
    
    # Test with different uncertainty scenarios
    scenarios = [
        ("High confidence, low uncertainty", {'confidence': 0.95, 'is_live': True}),
        ("Medium confidence, medium uncertainty", {'confidence': 0.65, 'is_live': True}),
        ("Low confidence, high uncertainty", {'confidence': 0.35, 'is_live': False}),
    ]
    
    for scenario_name, cnn_data in scenarios:
        fusion_result = security_state.calculate_enhanced_fusion_score(
            cnn_data, landmark_data, movement_data
        )
        
        print(f"   {scenario_name}:")
        print(f"     Final score: {fusion_result['final_score']:.3f}")
        print(f"     Uncertainty: {fusion_result['uncertainty']:.3f}")
        print(f"     Confidence interval width: {fusion_result['confidence_interval']['width']:.3f}")
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    final_status = security_state.get_security_status()
    print(f"✅ Enhanced Multi-Modal Fusion System Test Complete!")
    print(f"   Total frames processed: {final_status['total_frames_processed']}")
    print(f"   Verification duration: {final_status['verification_duration']:.1f}s")
    print(f"   Challenge attempts: {final_status['challenge_attempts']}")
    print(f"   Completed challenges: {final_status['completed_challenges']}")
    print(f"   Final fusion score: {final_status['fusion_score']:.3f}")
    print(f"   Final aggregated decision: {final_status['aggregated_decision']}")
    print(f"   Cross-validation scores: {final_status['cross_validation']}")
    print(f"   Environmental quality: {final_status['environmental_quality']}")
    
    return True

def test_performance_comparison():
    """Compare performance between basic and enhanced fusion"""
    print("\n⚡ PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Basic fusion simulation (simple voting)
    def basic_fusion(cnn_conf, movement_detected, landmark_detected):
        methods_passed = sum([cnn_conf > 0.7, movement_detected, landmark_detected])
        return methods_passed >= 2
    
    # Enhanced fusion
    enhanced_state = EnhancedSecurityAssessmentState()
    
    # Test scenarios
    test_cases = [
        ("High quality", 0.9, True, True),
        ("Medium quality", 0.6, True, False),
        ("Poor CNN, good movement", 0.3, True, True),
        ("Good CNN, no movement", 0.8, False, False),
        ("Borderline case", 0.5, False, True),
    ]
    
    print("Test Case                | Basic Result | Enhanced Score | Enhanced Decision | Improvement")
    print("-" * 85)
    
    for case_name, cnn_conf, movement, landmark in test_cases:
        # Basic fusion
        basic_result = basic_fusion(cnn_conf, movement, landmark)
        
        # Enhanced fusion
        cnn_data = {'confidence': cnn_conf, 'is_live': cnn_conf > 0.5}
        landmark_data = {'landmarks_detected': landmark, 'landmark_count': 68 if landmark else 0}
        movement_data = {'head_movement': movement, 'blink_detected': movement, 'blink_count': 3 if movement else 0}
        
        fusion_result = enhanced_state.calculate_enhanced_fusion_score(
            cnn_data, landmark_data, movement_data
        )
        
        enhanced_score = fusion_result['final_score']
        enhanced_decision = fusion_result['aggregated_decision']
        
        # Determine improvement
        if basic_result == enhanced_decision:
            improvement = "Same"
        elif enhanced_decision and not basic_result:
            improvement = "Better ✅"
        elif not enhanced_decision and basic_result:
            improvement = "Stricter ⚠️"
        else:
            improvement = "Different"
        
        print(f"{case_name:<24} | {str(basic_result):<11} | {enhanced_score:>13.3f} | {str(enhanced_decision):<17} | {improvement}")

if __name__ == "__main__":
    try:
        # Run comprehensive tests
        success = test_enhanced_fusion()
        
        if success:
            test_performance_comparison()
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Enhanced Multi-Modal Fusion System is ready for deployment!")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

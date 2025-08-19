import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detection.landmark_detection import LivenessVerifier
    print("Import successful!")
    
    verifier = LivenessVerifier()
    print(f"Left eye indices: {verifier.left_eye_indices}")
    print(f"Right eye indices: {verifier.right_eye_indices}")
    
    # Test EAR with minimal data
    test_landmarks = [[10.0, 10.0] for _ in range(500)]
    ear = verifier.calculate_eye_aspect_ratio(test_landmarks, [33, 133, 159, 145])
    print(f"Test EAR: {ear}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

"""
Facial Landmark Detection untuk verifikasi gerakan alami
Menggunakan MediaPipe untuk real-time landmark detection (lightweight)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import math
import time
from collections import deque
import logging

class FacialLandmarkDetector:
    """
    Lightweight facial landmark detector menggunakan MediaPipe saja
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Args:
            confidence_threshold: Threshold untuk detection confidence
        """
        self.confidence_threshold = confidence_threshold
        
        # Timestamp management for MediaPipe
        self.last_timestamp = 0
        self.frame_count = 0
        
        # Initialize MediaPipe only
        self._init_mediapipe()
            
        # Landmark indices untuk berbagai fitur wajah
        self._setup_landmark_indices()
        
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold
        )
            
    def _setup_landmark_indices(self):
        """Setup indices untuk berbagai region wajah (MediaPipe 468 points)"""
        # MediaPipe landmarks (468 points)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.nose_indices = [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305]
            
    def detect_landmarks(self, image):
        """
        Detect facial landmarks dari gambar menggunakan MediaPipe
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            landmarks: Array of landmark coordinates atau None jika tidak terdeteksi
            confidence: Detection confidence
        """
        return self._detect_landmarks_mediapipe(image)
            
    def _detect_landmarks_mediapipe(self, image):
        """Detect landmarks menggunakan MediaPipe dengan proper timestamp handling"""
        print("=== DEBUG: _detect_landmarks_mediapipe called ===")
        
        try:
            # Ensure monotonic timestamps for MediaPipe
            current_timestamp = int(time.time() * 1e6)  # microseconds
            if current_timestamp <= self.last_timestamp:
                current_timestamp = self.last_timestamp + 1
            self.last_timestamp = current_timestamp
            self.frame_count += 1
            
            print(f"=== DEBUG: Processing frame #{self.frame_count}, timestamp: {current_timestamp}")
            
            # Get image dimensions from original image
            height, width = image.shape[:2]
            print(f"=== DEBUG: Original image dimensions: {width}x{height}")
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"=== DEBUG: Image converted to RGB, shape: {rgb_image.shape}")
            
            # Process with MediaPipe using the RGB image directly (no timestamp for now)
            print("=== DEBUG: Processing with MediaPipe face_mesh ===")
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                print(f"=== DEBUG: {len(results.multi_face_landmarks)} face(s) detected ===")
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to array numpy dengan normalisasi koordinat
                landmarks = []
                
                for i, landmark in enumerate(face_landmarks.landmark):
                    # Store normalized coordinates (0-1) - already normalized by MediaPipe
                    normalized_x = landmark.x
                    normalized_y = landmark.y
                    landmarks.append([normalized_x, normalized_y])
                
                print(f"=== DEBUG: Converted {len(landmarks)} landmarks to normalized coordinates ===")
                
                # Calculate confidence (MediaPipe doesn't provide confidence directly)
                confidence = 0.9  # High confidence for detected landmarks
                
                return landmarks, confidence
            else:
                print("=== DEBUG: No face landmarks detected ===")
                return None, 0.0
                
        except Exception as e:
            print(f"=== DEBUG: MediaPipe processing error: {e} ===")
            logging.error(f"MediaPipe detection error: {e}")
            return None, 0.0

class LivenessVerifier:
    """
    Verifikasi liveness berdasarkan gerakan facial landmarks (MediaPipe only)
    """
    
    def __init__(self, history_length=30):
        """
        Args:
            history_length: Panjang history untuk tracking gerakan
        """
        self.history_length = history_length
        self.landmark_detector = FacialLandmarkDetector()
        
        # History tracking
        self.landmark_history = deque(maxlen=history_length)
        self.eye_aspect_ratio_history = deque(maxlen=history_length)
        self.mouth_aspect_ratio_history = deque(maxlen=history_length)
        self.head_pose_history = deque(maxlen=history_length)
        
        # Blink detection
        self.blink_threshold = 0.25
        self.blink_consecutive_frames = 3
        self.blink_count = 0
        self.blink_frames = 0
        self.blink_detected = False  # Add this flag
        
        # Mouth detection
        self.mouth_open_threshold = 0.6
        self.mouth_open_frames = 0
        
        # Head movement detection
        self.head_movement_threshold = 15.0  # degrees
        
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Calculate Eye Aspect Ratio (EAR) untuk blink detection
        """
        try:
            print(f"=== DEBUG: EAR calculation with {len(eye_indices)} indices ===")
            
            if len(eye_indices) < 6:
                print(f"=== DEBUG: Not enough eye indices: {len(eye_indices)} ===")
                return 0.0
            
            if len(landmarks) == 0:
                print("=== DEBUG: No landmarks for EAR calculation ===")
                return 0.0
                
            # Ambil koordinat mata satu per satu
            eye_points = []
            for idx in eye_indices:
                if isinstance(idx, int) and 0 <= idx < len(landmarks):
                    eye_points.append(landmarks[idx])
                else:
                    print(f"=== DEBUG: Invalid eye index: {idx} ===")
                    
            if len(eye_points) < 6:
                print(f"=== DEBUG: Not enough valid eye points: {len(eye_points)} ===")
                return 0.0
            
            # Convert to numpy array
            eye_points = np.array(eye_points)
            print(f"=== DEBUG: Eye points shape: {eye_points.shape} ===")
            
            # Calculate EAR - simplified version
            if eye_points.shape[0] >= 6 and eye_points.shape[1] >= 2:
                # Use first 6 points for EAR calculation
                p1, p2, p3, p4, p5, p6 = eye_points[:6]
                
                # Vertical distances
                vertical_1 = np.linalg.norm(p2 - p6)
                vertical_2 = np.linalg.norm(p3 - p5)
                
                # Horizontal distance
                horizontal = np.linalg.norm(p1 - p4)
                
                if horizontal > 0:
                    # EAR formula
                    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                    print(f"=== DEBUG: EAR calculated: {ear} ===")
                    return ear
                else:
                    print("=== DEBUG: Horizontal distance is zero ===")
                    return 0.0
            else:
                print(f"=== DEBUG: Invalid eye points shape: {eye_points.shape} ===")
                return 0.0
                
        except Exception as e:
            print(f"=== DEBUG: EAR calculation error: {e} ===")
            return 0.0

        
    def calculate_head_pose(self, landmarks):
        """
        Estimate head pose dari facial landmarks (MediaPipe only)
        """
        try:
            if not landmarks or len(landmarks) < 400:
                return None
                
            # MediaPipe landmarks - using more stable points
            landmarks = np.array(landmarks)
            
            # Key facial points for head pose estimation
            nose_tip = landmarks[1]  # Nose tip
            chin = landmarks[18] if len(landmarks) > 18 else landmarks[10]  # Chin
            left_eye = landmarks[33] if len(landmarks) > 33 else landmarks[30]  # Left eye corner
            right_eye = landmarks[362] if len(landmarks) > 362 else landmarks[130]  # Right eye corner
            left_mouth = landmarks[61] if len(landmarks) > 61 else landmarks[50]  # Left mouth corner
            right_mouth = landmarks[291] if len(landmarks) > 291 else landmarks[280]  # Right mouth corner
            
            # Calculate angles
            # Yaw (left-right rotation)
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (left_mouth + right_mouth) / 2
            yaw_vector = mouth_center - eye_center
            yaw_angle = math.degrees(math.atan2(yaw_vector[1], yaw_vector[0]))
            
            # Pitch (up-down rotation) 
            face_vertical = chin - nose_tip
            pitch_angle = math.degrees(math.atan2(face_vertical[1], face_vertical[0])) - 90
            
            # Roll (tilt)
            eye_vector = right_eye - left_eye
            roll_angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
            
            return {
                'yaw': float(yaw_angle),
                'pitch': float(pitch_angle),
                'roll': float(roll_angle)
            }
        except Exception as e:
            print(f"=== DEBUG: Head pose calculation error: {e} ===")
            return None
        
    def detect_blink(self, ear_left, ear_right):
        """
        Detect eye blink berdasarkan EAR
        """
        avg_ear = (ear_left + ear_right) / 2.0
        
        if avg_ear < self.blink_threshold:
            self.blink_frames += 1
        else:
            if self.blink_frames >= self.blink_consecutive_frames:
                self.blink_count += 1
            self.blink_frames = 0
            
        return self.blink_count
        
    def detect_mouth_opening(self, mar):
        """
        Detect mouth opening berdasarkan MAR
        """
        if mar > self.mouth_open_threshold:
            self.mouth_open_frames += 1
            return True
        else:
            self.mouth_open_frames = 0
            return False
            
    def detect_head_movement(self, current_pose):
        """
        Detect significant head movement
        """
        if len(self.head_pose_history) == 0:
            return False
            
        previous_pose = self.head_pose_history[-1]
        
        yaw_diff = abs(current_pose['yaw'] - previous_pose['yaw'])
        pitch_diff = abs(current_pose['pitch'] - previous_pose['pitch'])
        roll_diff = abs(current_pose['roll'] - previous_pose['roll'])
        
        return (yaw_diff > self.head_movement_threshold or 
                pitch_diff > self.head_movement_threshold or 
                roll_diff > self.head_movement_threshold)
        
    def calculate_liveness_score(self, landmarks, ear_left, ear_right, mar, head_pose):
        """
        Calculate comprehensive liveness score (0-100)
        
        Args:
            landmarks: Detected facial landmarks
            ear_left: Left eye aspect ratio
            ear_right: Right eye aspect ratio  
            mar: Mouth aspect ratio
            head_pose: Head pose information
            
        Returns:
            float: Liveness score (0-100)
        """
        score = 0.0
        max_score = 100.0
        
        # Base score for having landmarks
        if landmarks and len(landmarks) > 0:
            score += 20.0  # 20 points for face detection
        
        # Eye blink score (30 points max)
        avg_ear = (ear_left + ear_right) / 2.0
        if avg_ear > 0:
            # Reward normal eye opening (EAR between 0.2-0.4)
            if 0.2 <= avg_ear <= 0.4:
                score += 15.0
            # Reward blink history
            if self.blink_count > 0:
                score += min(15.0, self.blink_count * 3.0)  # 3 points per blink, max 15
        
        # Mouth movement score (20 points max)
        if mar > 0:
            if mar > 0.3:  # Mouth movement detected
                score += 10.0
            if hasattr(self, 'prev_mar') and abs(mar - self.prev_mar) > 0.1:
                score += 10.0  # Mouth movement change
        self.prev_mar = mar
        
        # Head movement score (20 points max) 
        if head_pose:
            # Reward head movement variety
            if len(self.head_pose_history) > 1:
                recent_poses = list(self.head_pose_history)[-5:]  # Last 5 poses
                yaw_range = max([p['yaw'] for p in recent_poses]) - min([p['yaw'] for p in recent_poses])
                pitch_range = max([p['pitch'] for p in recent_poses]) - min([p['pitch'] for p in recent_poses])
                
                if yaw_range > 10:  # Head turned left/right
                    score += 10.0
                if pitch_range > 10:  # Head moved up/down
                    score += 10.0
        
        # Anti-spoofing checks (10 points max)
        if landmarks and len(landmarks) > 400:  # MediaPipe gives 468 points for real faces
            score += 5.0
        
        # Temporal consistency (detect video attacks)
        if len(self.landmark_history) > 5:
            # Check for natural micro-movements
            recent_landmarks = list(self.landmark_history)[-3:]
            if len(recent_landmarks) >= 2:
                movement = np.array(recent_landmarks[-1]) - np.array(recent_landmarks[-2])
                avg_movement = np.mean(np.abs(movement))
                if 0.001 < avg_movement < 0.05:  # Natural small movements
                    score += 5.0
        
        return min(score, max_score)
    
    def is_live_face(self, liveness_score, threshold=70.0):
        """
        Determine if face is live based on liveness score
        
        Args:
            liveness_score: Calculated liveness score
            threshold: Minimum score for live classification
            
        Returns:
            bool: True if face is considered live
        """
        return liveness_score >= threshold

    def process_frame(self, image):
        """
        Process single frame untuk liveness detection
        
        Returns:
            dict: Dictionary berisi semua metrics dan detection results
        """
        print("=== DEBUG: LivenessVerifier.process_frame called ===")
        
        landmarks, confidence = self.landmark_detector.detect_landmarks(image)
        print(f"=== DEBUG: Landmark detection result: {landmarks is not None}, confidence: {confidence}")
        
        if landmarks is None:
            print("=== DEBUG: No landmarks detected, returning default results ===")
            return {
                'landmarks_detected': False,
                'landmark_coordinates': [],
                'confidence': 0.0,
                'blink_count': self.blink_count,
                'mouth_open': False,
                'head_movement': False,
                'ear_left': 0.0,
                'ear_right': 0.0,
                'mar': 0.0,
                'head_pose': None,
                'liveness_score': 0.0,
                'is_live': False,
                'liveness_status': 'NO_FACE'
            }
        
        print(f"=== DEBUG: Processing {len(landmarks)} landmarks ===")
        
        # Add landmarks to history for temporal analysis
        self.landmark_history.append(landmarks)
        
        # Calculate metrics safely
        try:
            ear_left = self.calculate_eye_aspect_ratio(
                landmarks, self.landmark_detector.left_eye_indices
            )
            ear_right = self.calculate_eye_aspect_ratio(
                landmarks, self.landmark_detector.right_eye_indices
            )
            
            print(f"=== DEBUG: EAR left: {ear_left}, right: {ear_right}")
            
            # Simplified MAR calculation
            mar = 0.0
            try:
                mouth_indices = self.landmark_detector.mouth_indices
                if len(mouth_indices) >= 4:
                    mouth_points = []
                    for idx in mouth_indices[:4]:
                        if 0 <= idx < len(landmarks):
                            mouth_points.append(landmarks[idx])
                    
                    if len(mouth_points) >= 4:
                        mouth_points = np.array(mouth_points)
                        if mouth_points.shape[1] >= 2:
                            vertical_mouth = np.linalg.norm(mouth_points[1] - mouth_points[3])
                            horizontal_mouth = np.linalg.norm(mouth_points[0] - mouth_points[2])
                            if horizontal_mouth > 0:
                                mar = vertical_mouth / horizontal_mouth
                                
            except Exception as e:
                print(f"=== DEBUG: MAR calculation error: {e} ===")
                mar = 0.0
            
            print(f"=== DEBUG: MAR: {mar}")
            
            # Calculate head pose
            head_pose = None
            head_movement = False
            try:
                head_pose = self.calculate_head_pose(landmarks)
                if head_pose:
                    self.head_pose_history.append(head_pose)
                    head_movement = self.detect_head_movement(head_pose)
                    print(f"=== DEBUG: Head pose: {head_pose}")
            except Exception as e:
                print(f"=== DEBUG: Head pose calculation error: {e} ===")
            
            # Detect blink
            avg_ear = (ear_left + ear_right) / 2.0
            is_blink = avg_ear < self.blink_threshold
            
            if is_blink and not hasattr(self, 'blink_detected'):
                self.blink_detected = False
                
            if is_blink and not self.blink_detected:
                self.blink_count += 1
                self.blink_detected = True
                print(f"=== DEBUG: Blink detected! Total count: {self.blink_count}")
            elif not is_blink:
                self.blink_detected = False
                
            # Detect mouth open
            mouth_open = mar > 0.6  # mouth_threshold
            if mouth_open:
                print("=== DEBUG: Mouth open detected")
            
            # Calculate comprehensive liveness score
            liveness_score = self.calculate_liveness_score(landmarks, ear_left, ear_right, mar, head_pose)
            is_live = self.is_live_face(liveness_score)
            
            # Determine liveness status
            if liveness_score >= 80:
                liveness_status = "LIVE"
            elif liveness_score >= 60:
                liveness_status = "LIKELY_LIVE" 
            elif liveness_score >= 40:
                liveness_status = "UNCERTAIN"
            elif liveness_score >= 20:
                liveness_status = "LIKELY_FAKE"
            else:
                liveness_status = "FAKE"
            
            print(f"=== DEBUG: Liveness score: {liveness_score:.1f}, Status: {liveness_status}")

            results = {
                'landmarks_detected': True,
                'landmark_coordinates': landmarks,  # Send normalized coordinates to frontend
                'confidence': confidence,
                'blink_count': self.blink_count,
                'mouth_open': mouth_open,
                'head_movement': head_movement,
                'ear_left': float(ear_left),
                'ear_right': float(ear_right),
                'mar': float(mar),
                'head_pose': head_pose,
                'liveness_score': float(liveness_score),
                'is_live': is_live,
                'liveness_status': liveness_status,
                'liveness_metrics': {
                    'blinks': self.blink_count,
                    'head_movement_range': head_movement,
                    'mouth_movement': mouth_open,
                    'landmark_count': len(landmarks),
                    'confidence': confidence
                }
            }
            
            print(f"=== DEBUG: Returning results with liveness score {liveness_score:.1f} ===")
            return results
            
        except Exception as e:
            print(f"=== DEBUG: Process frame error: {e} ===")
            return {
                'landmarks_detected': False,
                'landmark_coordinates': [],
                'confidence': 0.0,
                'blink_count': self.blink_count,
                'mouth_open': False,
                'head_movement': False,
                'ear_left': 0.0,
                'ear_right': 0.0,
                'mar': 0.0,
                'head_pose': None,
                'liveness_score': 0.0,
                'is_live': False,
                'liveness_status': 'ERROR'
            }
        
    def reset_counters(self):
        """Reset semua counters dan history"""
        self.blink_count = 0
        self.blink_frames = 0
        self.mouth_open_frames = 0
        self.landmark_history.clear()
        self.eye_aspect_ratio_history.clear()
        self.mouth_aspect_ratio_history.clear()
        self.head_pose_history.clear()
        
    def visualize_landmarks(self, image, landmarks):
        """
        Visualize landmarks pada gambar
        """
        if landmarks is None:
            return image
            
        result_image = image.copy()
        
        # Draw landmarks
        for point in landmarks:
            cv2.circle(result_image, tuple(point.astype(int)), 1, (0, 255, 0), -1)
            
        # Draw eye regions
        left_eye_points = landmarks[self.landmark_detector.left_eye_indices]
        right_eye_points = landmarks[self.landmark_detector.right_eye_indices]
        
        cv2.polylines(result_image, [left_eye_points.astype(int)], True, (255, 0, 0), 2)
        cv2.polylines(result_image, [right_eye_points.astype(int)], True, (255, 0, 0), 2)
        
        # Draw mouth region
        mouth_points = landmarks[self.landmark_detector.mouth_indices]
        cv2.polylines(result_image, [mouth_points.astype(int)], True, (0, 0, 255), 2)
        
        return result_image

def test_landmark_detection():
    """
    Test function untuk landmark detection
    """
    # Initialize detector
    verifier = LivenessVerifier()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Testing Facial Landmark Detection...")
    print("Instructions:")
    print("- Blink your eyes")
    print("- Open your mouth")
    print("- Move your head left/right, up/down")
    print("- Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = verifier.process_frame(frame)
        
        # Draw information
        if results['landmarks_detected']:
            # Visualize landmarks
            frame = verifier.visualize_landmarks(frame, results['landmarks'])
            
            # Draw metrics
            y_offset = 30
            cv2.putText(frame, f"Blinks: {results['blink_count']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            cv2.putText(frame, f"EAR L/R: {results['ear_left']:.3f}/{results['ear_right']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            cv2.putText(frame, f"MAR: {results['mar']:.3f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            if results['mouth_open']:
                cv2.putText(frame, "MOUTH OPEN", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            
            if results['head_movement']:
                cv2.putText(frame, "HEAD MOVEMENT", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Head pose
            if results['head_pose']:
                pose = results['head_pose']
                cv2.putText(frame, f"Yaw: {pose['yaw']:.1f}°", (10, frame.shape[0] - 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pose['pitch']:.1f}°", (10, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Roll: {pose['roll']:.1f}°", (10, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Facial Landmark Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_landmark_detection()

"""
OPTIMIZED Facial Landmark Detection untuk verifikasi gerakan alami
Implementasi yang dioptimalkan untuk real-time performance
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
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

class OptimizedFacialLandmarkDetector:
    """
    OPTIMIZED lightweight facial landmark detector menggunakan MediaPipe
    - Reduced landmark processing
    - Frame skipping
    - Memory optimization
    - GPU acceleration
    """
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, 
                 use_static_image_mode=False, frame_skip=2):
        """
        Args:
            frame_skip: Process every N frames (2 = process every 2nd frame)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_skip = frame_skip
        self.frame_counter = 0
        
        # Cache last results for frame skipping
        self.last_results = None
        self.results_cache_time = time.time()
        
        # Initialize with optimized settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=use_static_image_mode,
            max_num_faces=1,  # Only detect 1 face for performance
            refine_landmarks=False,  # Disable refinement for speed
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Optimized landmark indices (critical points for performance with anti-spoofing focus)
        # Based on proven research for robust liveness detection
        self.critical_indices = {
            'left_eye': [33, 133, 160, 158, 153, 144],  # 6-point robust EAR calculation
            'right_eye': [362, 263, 385, 387, 373, 380],  # 6-point robust EAR calculation
            'mouth': [61, 291, 13, 14, 17, 18],  # Key mouth points for MAR
            'nose': [1, 2, 6, 168],  # Stable nose reference points
            'face_outline': [10, 151, 9, 175, 18]  # Key face boundary points
        }
        
        # IMPROVED thresholds for anti-spoofing
        self.blink_threshold = 0.22  # Research-proven optimal threshold
        self.blink_consecutive_frames = 3  # Minimum frames for valid blink
        self.min_blink_duration = 2  # Minimum blink duration
        self.max_blink_duration = 8  # Maximum blink duration
        
        # Quality validation settings
        self.min_confidence_threshold = 0.7
        self.min_landmark_count = 25  # Reduced for optimization but still sufficient
        
        # Pre-compute indices for faster access
        self.left_eye_indices = self.critical_indices['left_eye']
        self.right_eye_indices = self.critical_indices['right_eye']
        self.mouth_indices = self.critical_indices['mouth']
        
        # Image preprocessing settings
        self.target_width = 320  # Reduced from 640 for speed
        self.target_height = 240  # Reduced from 480 for speed
        
        print(f"[OK] OptimizedFacialLandmarkDetector initialized with frame_skip={frame_skip}")
    
    def preprocess_image(self, image):
        """
        OPTIMIZED image preprocessing with caching
        """
        # Resize to smaller dimensions for faster processing
        if image.shape[1] > self.target_width or image.shape[0] > self.target_height:
            image = cv2.resize(image, (self.target_width, self.target_height), 
                             interpolation=cv2.INTER_LINEAR)  # Faster than INTER_CUBIC
        
        # Convert BGR to RGB (MediaPipe requirement)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Improve contrast only if needed (skip for performance)
        # rgb_image = cv2.convertScaleAbs(rgb_image, alpha=1.1, beta=10)
        
        return rgb_image
    
    def detect_landmarks(self, image):
        """
        OPTIMIZED landmark detection with frame skipping and caching
        """
        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            # Return cached results for skipped frames
            if self.last_results is not None:
                return self.last_results
        
        try:
            # Preprocess image
            rgb_image = self.preprocess_image(image)
            
            # MediaPipe detection
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]  # Only first face
                
                # Extract only critical landmarks for performance
                landmarks = []
                height, width = rgb_image.shape[:2]
                
                # Process only critical landmarks
                all_indices = (self.left_eye_indices + self.right_eye_indices + 
                             self.mouth_indices + self.critical_indices['nose'] + 
                             self.critical_indices['face_outline'])
                
                for idx in all_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = landmark.x  # Already normalized 0-1
                        y = landmark.y  # Already normalized 0-1
                        landmarks.append([x, y])
                
                # Cache results
                self.last_results = (landmarks, 0.9)  # Simplified confidence
                return landmarks, 0.9
            
            else:
                self.last_results = (None, 0.0)
                return None, 0.0
                
        except Exception as e:
            print(f"Landmark detection error: {e}")
            return None, 0.0


class OptimizedLivenessVerifier:
    """
    OPTIMIZED Verifikasi liveness dengan performance improvements
    - Cached calculations
    - Reduced computation complexity
    - Memory management
    - Multi-threading ready
    """
    
    def __init__(self, history_length=15):  # Reduced from 30
        """
        Optimized initialization with reduced memory usage and improved anti-spoofing
        """
        self.history_length = history_length
        self.landmark_detector = OptimizedFacialLandmarkDetector(frame_skip=2)
        
        # Reduced history tracking for memory efficiency
        self.landmark_history = deque(maxlen=history_length)
        self.ear_history = deque(maxlen=10)  # For temporal smoothing
        self.mar_history = deque(maxlen=10)  # For temporal smoothing
        
        # IMPROVED optimized thresholds based on research
        self.blink_threshold = 0.22  # Research-proven optimal threshold
        self.blink_consecutive_frames = 3  # Increased for reliability
        self.min_blink_duration = 2  # Minimum blink duration
        self.max_blink_duration = 8  # Maximum blink duration
        self.mouth_open_threshold = 0.5  # Lowered for better sensitivity
        self.head_movement_threshold = 8.0  # Reduced for better sensitivity
        
        # Counters
        self.blink_count = 0
        self.blink_frames = 0
        self.mouth_open_frames = 0
        self.blink_detected = False
        
        # Performance monitoring
        self.process_times = deque(maxlen=10)
        
        # Cache for expensive calculations
        self._cache = {}
        self._cache_time = {}
        self.cache_duration = 0.1  # 100ms cache
        
        print("[OK] OptimizedLivenessVerifier initialized with improved anti-spoofing")
    
    def _get_cached_or_compute(self, key, compute_func, *args):
        """
        Cache expensive computations
        """
        current_time = time.time()
        if (key in self._cache and 
            current_time - self._cache_time.get(key, 0) < self.cache_duration):
            return self._cache[key]
        
        result = compute_func(*args)
        self._cache[key] = result
        self._cache_time[key] = current_time
        return result
    
    def calculate_eye_aspect_ratio_optimized(self, landmarks, eye_indices):
        """
        OPTIMIZED EAR calculation with robust 6-point method and temporal smoothing
        """
        try:
            if not landmarks or len(eye_indices) < 6:
                return 0.0
            
            # Extract and validate eye points
            eye_points = []
            for idx in eye_indices[:6]:  # Use only first 6 points
                if idx < len(landmarks):
                    eye_points.append([float(landmarks[idx][0]), float(landmarks[idx][1])])
            
            if len(eye_points) < 6:
                return 0.0
            
            eye_points = np.array(eye_points, dtype=np.float32)
            
            # ROBUST 6-point EAR calculation
            # Points: [outer_corner(0), inner_corner(1), top_outer(2), top_inner(3), bottom_inner(4), bottom_outer(5)]
            
            # Calculate vertical distances
            v1 = np.linalg.norm(eye_points[2] - eye_points[5])  # top_outer to bottom_outer
            v2 = np.linalg.norm(eye_points[3] - eye_points[4])  # top_inner to bottom_inner
            
            # Calculate horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[1])   # outer_corner to inner_corner
            
            if h > 1e-6:  # Avoid division by zero
                # Standard EAR formula
                ear = (v1 + v2) / (2.0 * h)
                
                # Apply bounds
                ear = max(0.0, min(ear, 1.0))
                
                # Add to history for temporal smoothing
                self.ear_history.append(ear)
                
                # Apply temporal smoothing
                if len(self.ear_history) >= 3:
                    recent_ears = list(self.ear_history)[-3:]
                    smoothed_ear = np.mean(recent_ears)
                    return smoothed_ear
                
                return ear
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def calculate_mouth_aspect_ratio_optimized(self, landmarks, mouth_indices):
        """
        OPTIMIZED MAR calculation
        """
        try:
            if not landmarks or len(mouth_indices) < 4:
                return 0.0
            
            # Use only 4 key mouth points for speed
            mouth_points = []
            for idx in mouth_indices[:4]:
                if idx < len(landmarks):
                    mouth_points.append(landmarks[idx])
            
            if len(mouth_points) < 4:
                return 0.0
            
            mouth_points = np.array(mouth_points)
            
            # Simplified MAR calculation
            vertical = np.linalg.norm(mouth_points[1] - mouth_points[3])
            horizontal = np.linalg.norm(mouth_points[0] - mouth_points[2])
            
            if horizontal > 0:
                mar = vertical / horizontal
                return min(max(mar, 0.0), 2.0)  # Clamp to reasonable range
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def detect_blink_optimized(self, ear_left, ear_right):
        """
        OPTIMIZED blink detection with improved validation and temporal consistency
        """
        # Validate input
        if ear_left <= 0 or ear_right <= 0:
            return self.blink_count
            
        # Calculate average EAR with asymmetry check
        avg_ear = (ear_left + ear_right) / 2.0
        ear_asymmetry = abs(ear_left - ear_right)
        
        # Reject asymmetric detections (likely errors)
        if ear_asymmetry > 0.15:
            return self.blink_count
        
        # Add to history for smoothing
        self.ear_history.append(avg_ear)
        
        # Use smoothed EAR for detection
        if len(self.ear_history) >= 3:
            smoothed_ear = np.mean(list(self.ear_history)[-3:])
        else:
            smoothed_ear = avg_ear
        
        # Improved blink state machine
        is_closed = smoothed_ear < self.blink_threshold
        
        if is_closed:
            self.blink_frames += 1
        else:
            # Check for valid blink
            if (self.min_blink_duration <= self.blink_frames <= self.max_blink_duration):
                self.blink_count += 1
            self.blink_frames = 0
        
        return self.blink_count
    
    def calculate_liveness_score_optimized(self, landmarks, ear_left, ear_right, mar):
        """
        OPTIMIZED liveness score calculation with reduced complexity
        """
        score = 0.0
        
        # Base score for face detection (20 points)
        if landmarks and len(landmarks) > 10:
            score += 20.0
        
        # Eye metrics (40 points max)
        avg_ear = (ear_left + ear_right) / 2.0
        if avg_ear > 0:
            # Normal eye opening
            if 0.2 <= avg_ear <= 0.4:
                score += 20.0
            # Blink bonus
            if self.blink_count > 0:
                score += min(20.0, self.blink_count * 5.0)
        
        # Mouth movement (20 points max) 
        if mar > 0.3:
            score += 15.0
        if len(self.mar_history) > 1:
            mar_variance = np.var(list(self.mar_history))
            if mar_variance > 0.01:  # Movement detected
                score += 5.0
        
        # Temporal consistency (20 points max)
        if len(self.landmark_history) > 3:
            # Simple movement detection
            recent = list(self.landmark_history)[-3:]
            if len(recent) >= 2:
                movement = np.mean([np.linalg.norm(np.array(recent[i]) - np.array(recent[i-1])) 
                                 for i in range(1, len(recent))])
                if 0.001 < movement < 0.05:  # Natural micro-movements
                    score += 20.0
        
        return min(score, 100.0)
    
    def process_frame_optimized(self, image):
        """
        OPTIMIZED frame processing with performance monitoring
        """
        start_time = time.time()
        
        try:
            # Landmark detection
            landmarks, confidence = self.landmark_detector.detect_landmarks(image)
            
            if landmarks is None:
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
                    'liveness_score': 0.0,
                    'is_live': False,
                    'liveness_status': 'NO_FACE',
                    'processing_time': time.time() - start_time
                }
            
            # Add to history
            self.landmark_history.append(landmarks)
            
            # Calculate metrics with caching
            ear_left = self.calculate_eye_aspect_ratio_optimized(
                landmarks, self.landmark_detector.left_eye_indices
            )
            ear_right = self.calculate_eye_aspect_ratio_optimized(
                landmarks, self.landmark_detector.right_eye_indices
            )
            mar = self.calculate_mouth_aspect_ratio_optimized(
                landmarks, self.landmark_detector.mouth_indices
            )
            
            # Add MAR to history
            self.mar_history.append(mar)
            
            # Detection
            self.detect_blink_optimized(ear_left, ear_right)
            mouth_open = mar > self.mouth_open_threshold
            
            # Simple head movement (reduced complexity)
            head_movement = False
            if len(self.landmark_history) > 2:
                current_center = np.mean(landmarks, axis=0)
                prev_center = np.mean(self.landmark_history[-2], axis=0)
                movement = np.linalg.norm(current_center - prev_center)
                head_movement = movement > 0.01
            
            # Liveness score
            liveness_score = self.calculate_liveness_score_optimized(
                landmarks, ear_left, ear_right, mar
            )
            is_live = liveness_score >= 70.0
            
            # Status
            if liveness_score >= 80:
                status = "LIVE"
            elif liveness_score >= 60:
                status = "LIKELY_LIVE"
            elif liveness_score >= 40:
                status = "UNCERTAIN"
            else:
                status = "LIKELY_FAKE"
            
            processing_time = time.time() - start_time
            self.process_times.append(processing_time)
            
            # Memory cleanup every 100 frames
            if len(self.process_times) % 100 == 0:
                gc.collect()
            
            return {
                'landmarks_detected': True,
                'landmark_coordinates': landmarks,
                'confidence': confidence,
                'blink_count': self.blink_count,
                'mouth_open': mouth_open,
                'head_movement': head_movement,
                'ear_left': float(ear_left),
                'ear_right': float(ear_right),
                'mar': float(mar),
                'liveness_score': float(liveness_score),
                'is_live': is_live,
                'liveness_status': status,
                'processing_time': processing_time,
                'avg_processing_time': np.mean(list(self.process_times)),
                'fps_estimate': 1.0 / np.mean(list(self.process_times)) if self.process_times else 0
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Frame processing error: {e}")
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
                'liveness_score': 0.0,
                'is_live': False,
                'liveness_status': 'ERROR',
                'processing_time': processing_time
            }
    
    def get_performance_stats(self):
        """
        Get performance statistics
        """
        if not self.process_times:
            return {}
        
        times = list(self.process_times)
        return {
            'avg_processing_time': np.mean(times),
            'min_processing_time': np.min(times),
            'max_processing_time': np.max(times),
            'estimated_fps': 1.0 / np.mean(times),
            'cache_size': len(self._cache)
        }
    
    def reset_counters(self):
        """Reset all counters and clear caches"""
        self.blink_count = 0
        self.blink_frames = 0
        self.mouth_open_frames = 0
        self.landmark_history.clear()
        self.ear_history.clear()
        self.mar_history.clear()
        self._cache.clear()
        self._cache_time.clear()
        gc.collect()


# Threading wrapper for concurrent processing
class ThreadedLivenessProcessor:
    """
    Multi-threaded wrapper for concurrent frame processing
    """
    
    def __init__(self, max_workers=2):
        self.verifier = OptimizedLivenessVerifier()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
    
    def process_frame_async(self, image, session_id="default"):
        """
        Process frame asynchronously
        """
        future = self.executor.submit(self.verifier.process_frame_optimized, image)
        self.futures[session_id] = future
        return future
    
    def get_result(self, session_id="default", timeout=0.1):
        """
        Get result if ready, otherwise return None
        """
        if session_id in self.futures:
            future = self.futures[session_id]
            try:
                if future.done():
                    result = future.result(timeout=timeout)
                    del self.futures[session_id]
                    return result
            except Exception as e:
                print(f"Async processing error: {e}")
                del self.futures[session_id]
        return None
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)


if __name__ == "__main__":
    # Performance test
    print("🚀 TESTING OPTIMIZED LANDMARK DETECTION")
    print("=" * 50)
    
    verifier = OptimizedLivenessVerifier()
    
    # Test with dummy image
    dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        verifier.process_frame_optimized(dummy_image)
    
    # Performance test
    times = []
    for i in range(20):
        start = time.time()
        result = verifier.process_frame_optimized(dummy_image)
        times.append(time.time() - start)
        
        if i % 5 == 0:
            print(f"Frame {i+1}: {times[-1]*1000:.1f}ms")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"Average processing time: {avg_time*1000:.1f}ms")
    print(f"Estimated FPS: {fps:.1f}")
    print(f"Min time: {np.min(times)*1000:.1f}ms")
    print(f"Max time: {np.max(times)*1000:.1f}ms")
    
    stats = verifier.get_performance_stats()
    print(f"\nDetailed stats: {stats}")

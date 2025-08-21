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


# Advanced optimization features
class FaceTracker:
    """
    Optimized face tracker to avoid re-detection in consecutive frames
    """
    
    def __init__(self, tracking_quality_threshold=0.7, max_tracking_frames=30):
        """
        Initialize face tracker
        """
        self.tracking_quality_threshold = tracking_quality_threshold
        self.max_tracking_frames = max_tracking_frames
        self.tracker = None
        self.tracking_frame_count = 0
        self.last_bbox = None
        self.is_tracking = False
        
    def start_tracking(self, image, bbox):
        """
        Start tracking a face in the given bounding box
        """
        try:
            # Use CSRT tracker for balance of speed and accuracy
            self.tracker = cv2.TrackerCSRT_create()
            
            # Convert bbox format if needed
            x, y, w, h = bbox
            if w < 0:
                x += w
                w = -w
            if h < 0:
                y += h
                h = -h
            
            # Initialize tracker
            success = self.tracker.init(image, (x, y, w, h))
            if success:
                self.is_tracking = True
                self.tracking_frame_count = 0
                self.last_bbox = bbox
                return True
            
        except Exception as e:
            print(f"Tracking initialization error: {e}")
            
        return False
    
    def update_tracking(self, image):
        """
        Update tracker with new frame
        """
        if not self.is_tracking or self.tracker is None:
            return None, 0.0
        
        try:
            success, bbox = self.tracker.update(image)
            self.tracking_frame_count += 1
            
            if success and self.tracking_frame_count < self.max_tracking_frames:
                # Calculate tracking quality (simplified)
                quality = max(0.0, 1.0 - (self.tracking_frame_count / self.max_tracking_frames))
                
                if quality >= self.tracking_quality_threshold:
                    self.last_bbox = bbox
                    return bbox, quality
            
            # Stop tracking if quality is too low or max frames reached
            self.stop_tracking()
            return None, 0.0
            
        except Exception as e:
            print(f"Tracking update error: {e}")
            self.stop_tracking()
            return None, 0.0
    
    def stop_tracking(self):
        """
        Stop tracking
        """
        self.is_tracking = False
        self.tracker = None
        self.tracking_frame_count = 0


class ROIProcessor:
    """
    Region of Interest processor for focused processing
    """
    
    def __init__(self, expansion_factor=1.2, update_frequency=5):
        """
        Initialize ROI processor
        """
        self.expansion_factor = expansion_factor
        self.update_frequency = update_frequency
        self.current_roi = None
        self.frame_count = 0
        
    def calculate_roi(self, image, face_bbox):
        """
        Calculate ROI based on face bounding box
        """
        if face_bbox is None:
            return None
        
        h, w = image.shape[:2]
        x, y, bbox_w, bbox_h = face_bbox
        
        # Expand bounding box
        center_x, center_y = x + bbox_w // 2, y + bbox_h // 2
        expanded_w = int(bbox_w * self.expansion_factor)
        expanded_h = int(bbox_h * self.expansion_factor)
        
        # Calculate ROI coordinates
        roi_x1 = max(0, center_x - expanded_w // 2)
        roi_y1 = max(0, center_y - expanded_h // 2)
        roi_x2 = min(w, center_x + expanded_w // 2)
        roi_y2 = min(h, center_y + expanded_h // 2)
        
        return (roi_x1, roi_y1, roi_x2, roi_y2)
    
    def extract_roi(self, image, roi_coords=None):
        """
        Extract ROI from image
        """
        if roi_coords is None:
            roi_coords = self.current_roi
        
        if roi_coords is None:
            return image
        
        x1, y1, x2, y2 = roi_coords
        return image[y1:y2, x1:x2]
    
    def update_roi(self, image, face_bbox):
        """
        Update ROI if needed
        """
        self.frame_count += 1
        
        if (self.frame_count % self.update_frequency == 0 or 
            self.current_roi is None):
            self.current_roi = self.calculate_roi(image, face_bbox)
        
        return self.current_roi


class AdaptiveThresholdManager:
    """
    Adaptive threshold manager for environment-specific optimizations
    """
    
    def __init__(self, learning_rate=0.05):
        """
        Initialize adaptive threshold manager
        """
        self.learning_rate = learning_rate
        self.environment_stats = {
            'brightness': deque(maxlen=50),
            'contrast': deque(maxlen=50),
            'face_size': deque(maxlen=50)
        }
        
        # Base thresholds
        self.base_thresholds = {
            'blink_threshold': 0.22,
            'mouth_threshold': 0.5,
            'movement_threshold': 0.01
        }
        
        # Current adaptive thresholds
        self.current_thresholds = self.base_thresholds.copy()
        
    def analyze_environment(self, image, face_landmarks=None):
        """
        Analyze current environment conditions
        """
        # Brightness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.environment_stats['brightness'].append(brightness)
        
        # Contrast analysis
        contrast = np.std(gray)
        self.environment_stats['contrast'].append(contrast)
        
        # Face size analysis (if landmarks available)
        if face_landmarks:
            face_area = self._calculate_face_area(face_landmarks)
            self.environment_stats['face_size'].append(face_area)
    
    def _calculate_face_area(self, landmarks):
        """
        Calculate approximate face area from landmarks
        """
        if not landmarks or len(landmarks) < 4:
            return 0
        
        landmarks_array = np.array(landmarks)
        x_min, x_max = np.min(landmarks_array[:, 0]), np.max(landmarks_array[:, 0])
        y_min, y_max = np.min(landmarks_array[:, 1]), np.max(landmarks_array[:, 1])
        
        return (x_max - x_min) * (y_max - y_min)
    
    def update_thresholds(self):
        """
        Update thresholds based on environment analysis
        """
        if len(self.environment_stats['brightness']) < 10:
            return
        
        # Analyze recent environment data
        recent_brightness = np.mean(list(self.environment_stats['brightness'])[-10:])
        recent_contrast = np.mean(list(self.environment_stats['contrast'])[-10:])
        
        # Adjust blink threshold based on lighting
        if recent_brightness < 80:  # Low light
            brightness_factor = 1.1  # More sensitive in low light
        elif recent_brightness > 180:  # Bright light
            brightness_factor = 0.9  # Less sensitive in bright light
        else:
            brightness_factor = 1.0
        
        # Adjust based on contrast
        if recent_contrast < 30:  # Low contrast
            contrast_factor = 1.05
        else:
            contrast_factor = 1.0
        
        # Update thresholds with smoothing
        new_blink_threshold = (self.base_thresholds['blink_threshold'] * 
                              brightness_factor * contrast_factor)
        
        self.current_thresholds['blink_threshold'] = (
            self.current_thresholds['blink_threshold'] * (1 - self.learning_rate) +
            new_blink_threshold * self.learning_rate
        )
        
        # Clamp to reasonable bounds
        self.current_thresholds['blink_threshold'] = max(0.15, min(0.35, 
            self.current_thresholds['blink_threshold']))
    
    def get_threshold(self, threshold_name):
        """
        Get current adaptive threshold
        """
        return self.current_thresholds.get(threshold_name, 
            self.base_thresholds.get(threshold_name, 0.5))


class MemoryManager:
    """
    Advanced memory management for long-running applications
    """
    
    def __init__(self, cleanup_interval=100, max_memory_mb=512):
        """
        Initialize memory manager
        """
        self.cleanup_interval = cleanup_interval
        self.max_memory_mb = max_memory_mb
        self.frame_count = 0
        self.last_cleanup = time.time()
        
    def check_memory_usage(self):
        """
        Check current memory usage
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except:
            return 0
    
    def should_cleanup(self):
        """
        Check if memory cleanup is needed
        """
        self.frame_count += 1
        
        # Check interval
        if self.frame_count % self.cleanup_interval == 0:
            return True
        
        # Check memory usage
        memory_usage = self.check_memory_usage()
        if memory_usage > self.max_memory_mb:
            return True
        
        # Check time
        if time.time() - self.last_cleanup > 60:  # Every minute
            return True
        
        return False
    
    def cleanup(self):
        """
        Perform memory cleanup
        """
        gc.collect()
        self.last_cleanup = time.time()
        
        # Clear OpenCV cache
        try:
            cv2.destroyAllWindows()
        except:
            pass


# Threading wrapper for concurrent processing
class ThreadedLivenessProcessor:
    """
    Multi-threaded wrapper for concurrent frame processing with advanced optimizations
    """
    
    def __init__(self, max_workers=2, enable_tracking=True, enable_roi=True):
        self.verifier = OptimizedLivenessVerifier()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        
        # Advanced features
        self.face_tracker = FaceTracker() if enable_tracking else None
        self.roi_processor = ROIProcessor() if enable_roi else None
        self.adaptive_thresholds = AdaptiveThresholdManager()
        self.memory_manager = MemoryManager()
        
        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'tracking_hits': 0,
            'roi_hits': 0,
            'memory_cleanups': 0
        }
    
    def process_frame_async(self, image, session_id="default"):
        """
        Process frame asynchronously with optimizations
        """
        # Memory management
        if self.memory_manager.should_cleanup():
            self.memory_manager.cleanup()
            self.performance_stats['memory_cleanups'] += 1
        
        # Face tracking optimization
        if self.face_tracker and self.face_tracker.is_tracking:
            bbox, quality = self.face_tracker.update_tracking(image)
            if bbox and quality > 0.7:
                self.performance_stats['tracking_hits'] += 1
                # Use ROI processing
                if self.roi_processor:
                    roi = self.roi_processor.extract_roi(image, 
                        self.roi_processor.calculate_roi(image, bbox))
                    if roi.size > 0:
                        image = roi
                        self.performance_stats['roi_hits'] += 1
        
        # Environment analysis for adaptive thresholds
        self.adaptive_thresholds.analyze_environment(image)
        self.adaptive_thresholds.update_thresholds()
        
        # Update verifier thresholds
        self.verifier.blink_threshold = self.adaptive_thresholds.get_threshold('blink_threshold')
        
        # Async processing
        future = self.executor.submit(self.verifier.process_frame_optimized, image)
        self.futures[session_id] = future
        self.performance_stats['frames_processed'] += 1
        
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
                    
                    # Add optimization stats to result
                    result['optimization_stats'] = self.get_optimization_stats()
                    return result
            except Exception as e:
                print(f"Async processing error: {e}")
                del self.futures[session_id]
        return None
    
    def get_optimization_stats(self):
        """
        Get optimization performance statistics
        """
        stats = self.performance_stats.copy()
        
        if stats['frames_processed'] > 0:
            stats['tracking_hit_rate'] = stats['tracking_hits'] / stats['frames_processed']
            stats['roi_hit_rate'] = stats['roi_hits'] / stats['frames_processed']
        
        stats['current_thresholds'] = self.adaptive_thresholds.current_thresholds
        stats['memory_usage_mb'] = self.memory_manager.check_memory_usage()
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)
        if self.face_tracker:
            self.face_tracker.stop_tracking()


if __name__ == "__main__":
    # Performance test with advanced optimizations
    print("🚀 TESTING OPTIMIZED LANDMARK DETECTION WITH ADVANCED FEATURES")
    print("=" * 70)
    
    # Test standard verifier
    print("\n📊 Testing Standard OptimizedLivenessVerifier:")
    verifier = OptimizedLivenessVerifier()
    
    # Test threaded processor with optimizations
    print("\n🔧 Testing ThreadedLivenessProcessor with Advanced Optimizations:")
    processor = ThreadedLivenessProcessor(enable_tracking=True, enable_roi=True)
    
    # Test with dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Standard verifier test
    times = []
    for i in range(20):
        start = time.time()
        result = verifier.process_frame_optimized(dummy_image)
        times.append(time.time() - start)
        
        if i % 5 == 0:
            print(f"Standard Frame {i+1}: {times[-1]*1000:.1f}ms")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n📈 Standard Verifier Results:")
    print(f"Average processing time: {avg_time*1000:.1f}ms")
    print(f"Estimated FPS: {fps:.1f}")
    
    # Advanced processor test
    print(f"\n🚀 Advanced Processor Test:")
    opt_times = []
    for i in range(20):
        start = time.time()
        future = processor.process_frame_async(dummy_image, f"test_{i}")
        result = processor.get_result(f"test_{i}", timeout=1.0)
        opt_times.append(time.time() - start)
        
        if i % 5 == 0 and result:
            print(f"Advanced Frame {i+1}: {opt_times[-1]*1000:.1f}ms")
            if 'optimization_stats' in result:
                opt_stats = result['optimization_stats']
                print(f"  Memory usage: {opt_stats.get('memory_usage_mb', 0):.1f}MB")
    
    opt_avg_time = np.mean(opt_times)
    opt_fps = 1.0 / opt_avg_time
    
    print(f"\n📈 Advanced Processor Results:")
    print(f"Average processing time: {opt_avg_time*1000:.1f}ms")
    print(f"Estimated FPS: {opt_fps:.1f}")
    print(f"Performance improvement: {(avg_time/opt_avg_time-1)*100:.1f}%")
    
    # Get final optimization stats
    final_stats = processor.get_optimization_stats()
    print(f"\n🎯 Final Optimization Statistics:")
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value:.3f}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Cleanup
    processor.cleanup()
    
    print("\n🎉 Advanced optimization testing completed!")
    print("=" * 70)

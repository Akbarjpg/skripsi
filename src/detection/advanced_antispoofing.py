"""
Advanced Anti-Spoofing Techniques Implementation

This module provides state-of-the-art anti-spoofing classes for robust face verification.
Implements cutting-edge techniques for detecting sophisticated spoofing attacks.
"""
import numpy as np
import cv2
import time
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import logging

logger = logging.getLogger(__name__)

class TextureAnalyzer:
    """
    Advanced texture analysis for detecting print artifacts, screen door effects, and moire patterns.
    Uses multi-scale texture analysis and frequency domain processing.
    """
    
    def __init__(self):
        self.gabor_kernels = self._create_gabor_bank()
        self.history = deque(maxlen=30)  # Keep last 30 frames for temporal analysis
        
    def _create_gabor_bank(self):
        """Create a bank of Gabor filters for texture analysis"""
        kernels = []
        for theta in [0, 45, 90, 135]:  # 4 orientations
            for frequency in [0.1, 0.3, 0.5]:  # 3 frequencies
                kernel = cv2.getGaborKernel((31, 31), 5, np.radians(theta), 
                                          2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kernel)
        return kernels
    
    def analyze(self, image: np.ndarray, face_landmarks=None) -> dict:
        """
        Comprehensive texture analysis for spoofing detection
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            h, w = gray.shape
            
            # 1. Frequency domain analysis for moire patterns
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
            
            # Detect periodic patterns (moire, screen door effects)
            high_freq_energy = np.sum(magnitude_spectrum[h//4:3*h//4, w//4:3*w//4])
            total_energy = np.sum(magnitude_spectrum)
            moire_score = high_freq_energy / (total_energy + 1e-6)
            
            # 2. Gabor filter responses for texture analysis
            gabor_responses = []
            for kernel in self.gabor_kernels:
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                response = np.var(filtered)
                gabor_responses.append(response)
            
            texture_uniformity = np.std(gabor_responses) / (np.mean(gabor_responses) + 1e-6)
            
            # 3. Local Binary Pattern (LBP) analysis
            lbp = self._calculate_lbp(gray)
            lbp_uniformity = self._lbp_uniformity(lbp)
            
            # 4. Print artifact detection
            print_score = self._detect_print_artifacts(gray)
            
            # 5. Screen reflection detection
            reflection_score = self._detect_screen_reflections(gray, face_landmarks)
            
            # 6. Edge quality analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_quality = self._analyze_edge_quality(edges)
            
            # Combine all texture metrics
            texture_analysis = {
                'moire_score': float(moire_score),
                'texture_uniformity': float(texture_uniformity),
                'lbp_uniformity': float(lbp_uniformity),
                'print_artifacts': float(print_score),
                'screen_reflection': float(reflection_score),
                'edge_quality': float(edge_quality),
                'is_authentic': self._classify_texture_authenticity(
                    moire_score, texture_uniformity, lbp_uniformity, 
                    print_score, reflection_score, edge_quality
                ),
                'confidence': 0.8  # Base confidence
            }
            
            self.history.append(texture_analysis)
            
            # Add temporal consistency
            if len(self.history) > 5:
                texture_analysis['temporal_consistency'] = self._analyze_temporal_consistency()
            
            return texture_analysis
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return {'error': str(e), 'is_authentic': False, 'confidence': 0.0}
    
    def _calculate_lbp(self, image):
        """Calculate Local Binary Pattern"""
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                            (1,1), (1,0), (1,-1), (0,-1)]):
                    if image[i+di, j+dj] >= center:
                        code += 2**k
                lbp[i-1, j-1] = code
        return lbp
    
    def _lbp_uniformity(self, lbp):
        """Calculate LBP uniformity measure"""
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return np.std(hist) / (np.mean(hist) + 1e-6)
    
    def _detect_print_artifacts(self, image):
        """Detect print artifacts like halftone patterns"""
        # Use morphological operations to detect regular patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        diff = cv2.absdiff(image, closed)
        return np.mean(diff) / 255.0
    
    def _detect_screen_reflections(self, image, landmarks=None):
        """Detect screen reflections and unnatural lighting"""
        if landmarks is None:
            return 0.0
            
        # Focus on eye regions if landmarks available
        try:
            # Assuming MediaPipe landmarks format
            left_eye_points = landmarks[33:42] if len(landmarks) > 42 else None
            right_eye_points = landmarks[362:371] if len(landmarks) > 371 else None
            
            if left_eye_points is not None and right_eye_points is not None:
                # Extract eye regions and analyze for unnatural highlights
                return self._analyze_eye_reflections(image, left_eye_points, right_eye_points)
        except:
            pass
            
        # Fallback: general specular highlight detection
        gray = cv2.GaussianBlur(image, (5, 5), 0)
        _, highlights = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        return np.sum(highlights) / (image.shape[0] * image.shape[1] * 255.0)
    
    def _analyze_eye_reflections(self, image, left_eye, right_eye):
        """Analyze eye regions for unnatural reflections"""
        h, w = image.shape
        reflection_score = 0.0
        
        for eye_points in [left_eye, right_eye]:
            # Create eye mask
            eye_pts = np.array([(int(p.x * w), int(p.y * h)) for p in eye_points], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [eye_pts], 255)
            
            # Extract eye region
            eye_region = cv2.bitwise_and(image, image, mask=mask)
            eye_pixels = eye_region[mask > 0]
            
            if len(eye_pixels) > 0:
                # Check for unnatural bright spots
                bright_threshold = np.percentile(eye_pixels, 95)
                very_bright_pixels = np.sum(eye_pixels > bright_threshold)
                reflection_score += very_bright_pixels / len(eye_pixels)
        
        return reflection_score / 2.0  # Average of both eyes
    
    def _analyze_edge_quality(self, edges):
        """Analyze edge quality - printed images often have softer edges"""
        if np.sum(edges) == 0:
            return 0.0
            
        # Calculate edge strength distribution
        edge_pixels = edges[edges > 0]
        return np.std(edge_pixels.astype(float)) / 255.0
    
    def _classify_texture_authenticity(self, moire, uniformity, lbp, print_artifacts, 
                                     reflection, edge_quality):
        """Classify if texture indicates authentic face"""
        # Weighted scoring system
        score = 0.0
        
        # High moire indicates screen/digital display
        if moire > 0.3:
            score -= 0.3
        
        # Very uniform texture indicates printed/artificial
        if uniformity < 0.1:
            score -= 0.2
            
        # Print artifacts
        if print_artifacts > 0.2:
            score -= 0.3
            
        # Unnatural reflections
        if reflection > 0.4:
            score -= 0.2
            
        # Poor edge quality
        if edge_quality < 0.1:
            score -= 0.1
            
        # Natural texture variations indicate authentic
        if 0.1 < uniformity < 0.8 and lbp > 0.5:
            score += 0.4
            
        return score > 0.0
    
    def _analyze_temporal_consistency(self):
        """Analyze texture consistency over time"""
        if len(self.history) < 5:
            return 0.5
            
        recent_scores = [h.get('moire_score', 0.5) for h in list(self.history)[-5:]]
        return 1.0 - np.std(recent_scores)  # Higher consistency = more authentic


class DepthEstimator:
    """
    Advanced facial depth estimation to detect flat photos using multiple techniques.
    Combines geometric analysis, shadow detection, and learned depth features.
    """
    
    def __init__(self):
        self.prev_landmarks = None
        self.depth_history = deque(maxlen=20)
        
    def estimate(self, image: np.ndarray, landmarks=None) -> dict:
        """
        Comprehensive depth analysis for spoofing detection
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # 1. Geometric depth estimation using facial landmarks
            geometric_depth = self._estimate_geometric_depth(landmarks) if landmarks else 0.0
            
            # 2. Shadow-based depth analysis
            shadow_depth = self._analyze_shadows(gray, landmarks)
            
            # 3. Gradient-based depth cues
            gradient_depth = self._analyze_gradients(gray)
            
            # 4. Perspective distortion analysis
            perspective_score = self._analyze_perspective(landmarks) if landmarks else 0.5
            
            # 5. Facial structure consistency
            structure_consistency = self._analyze_facial_structure(landmarks) if landmarks else 0.5
            
            # 6. Temporal depth consistency
            temporal_consistency = self._analyze_temporal_depth_consistency()
            
            depth_analysis = {
                'geometric_depth': float(geometric_depth),
                'shadow_depth': float(shadow_depth),
                'gradient_depth': float(gradient_depth),
                'perspective_score': float(perspective_score),
                'structure_consistency': float(structure_consistency),
                'temporal_consistency': float(temporal_consistency),
                'is_3d': self._classify_3d_authenticity(
                    geometric_depth, shadow_depth, gradient_depth,
                    perspective_score, structure_consistency, temporal_consistency
                ),
                'confidence': 0.85
            }
            
            self.depth_history.append(depth_analysis)
            return depth_analysis
            
        except Exception as e:
            logger.error(f"Error in depth estimation: {e}")
            return {'error': str(e), 'is_3d': False, 'confidence': 0.0}
    
    def _estimate_geometric_depth(self, landmarks):
        """Estimate depth using facial landmark geometry"""
        if landmarks is None or len(landmarks) < 10:
            return 0.0
            
        try:
            # Extract key points for depth estimation
            nose_tip = landmarks[1]  # Nose tip
            left_eye = landmarks[33]  # Left eye corner
            right_eye = landmarks[362]  # Right eye corner
            mouth_center = landmarks[13]  # Mouth center
            
            # Calculate relative distances
            eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
            nose_to_eye_distance = np.sqrt((nose_tip.x - (left_eye.x + right_eye.x)/2)**2 + 
                                         (nose_tip.y - (left_eye.y + right_eye.y)/2)**2)
            
            # Depth indicator: ratio should be consistent for real 3D faces
            depth_ratio = nose_to_eye_distance / (eye_distance + 1e-6)
            
            # Real faces typically have depth_ratio between 0.3-0.8
            if 0.3 <= depth_ratio <= 0.8:
                return min(1.0, depth_ratio * 1.5)
            else:
                return max(0.0, 0.5 - abs(depth_ratio - 0.55))
                
        except Exception:
            return 0.0
    
    def _analyze_shadows(self, image, landmarks):
        """Analyze shadow patterns for depth cues"""
        if landmarks is None:
            return 0.0
            
        try:
            h, w = image.shape
            
            # Focus on nose and cheek areas where shadows are prominent
            nose_region = self._extract_nose_region(image, landmarks, h, w)
            cheek_regions = self._extract_cheek_regions(image, landmarks, h, w)
            
            shadow_score = 0.0
            regions_analyzed = 0
            
            # Analyze nose shadow
            if nose_region is not None:
                nose_shadow = self._detect_shadow_in_region(nose_region)
                shadow_score += nose_shadow
                regions_analyzed += 1
            
            # Analyze cheek shadows
            for cheek in cheek_regions:
                if cheek is not None:
                    cheek_shadow = self._detect_shadow_in_region(cheek)
                    shadow_score += cheek_shadow
                    regions_analyzed += 1
            
            return shadow_score / max(1, regions_analyzed)
            
        except Exception:
            return 0.0
    
    def _extract_nose_region(self, image, landmarks, h, w):
        """Extract nose region for shadow analysis"""
        try:
            nose_tip = landmarks[1]
            nose_bridge = landmarks[6]
            
            # Define nose region
            x_center = int((nose_tip.x + nose_bridge.x) * w / 2)
            y_center = int((nose_tip.y + nose_bridge.y) * h / 2)
            
            region_size = min(w, h) // 20
            x1 = max(0, x_center - region_size)
            x2 = min(w, x_center + region_size)
            y1 = max(0, y_center - region_size)
            y2 = min(h, y_center + region_size)
            
            return image[y1:y2, x1:x2]
        except:
            return None
    
    def _extract_cheek_regions(self, image, landmarks, h, w):
        """Extract cheek regions for shadow analysis"""
        try:
            left_cheek = landmarks[116]  # Left cheek
            right_cheek = landmarks[345]  # Right cheek
            
            regions = []
            for cheek in [left_cheek, right_cheek]:
                x_center = int(cheek.x * w)
                y_center = int(cheek.y * h)
                
                region_size = min(w, h) // 25
                x1 = max(0, x_center - region_size)
                x2 = min(w, x_center + region_size)
                y1 = max(0, y_center - region_size)
                y2 = min(h, y_center + region_size)
                
                regions.append(image[y1:y2, x1:x2])
            
            return regions
        except:
            return [None, None]
    
    def _detect_shadow_in_region(self, region):
        """Detect shadow patterns in a facial region"""
        if region is None or region.size == 0:
            return 0.0
            
        # Calculate local contrast and gradient patterns
        blurred = cv2.GaussianBlur(region, (5, 5), 0)
        gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Shadow areas have specific gradient patterns
        shadow_score = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)
        return min(1.0, shadow_score / 10.0)
    
    def _analyze_gradients(self, image):
        """Analyze image gradients for depth information"""
        # Calculate Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Real faces have more varied gradient patterns than flat images
        gradient_variance = np.var(magnitude)
        gradient_mean = np.mean(magnitude)
        
        # Normalize the depth score
        depth_score = gradient_variance / (gradient_mean + 1e-6)
        return min(1.0, depth_score / 1000.0)
    
    def _analyze_perspective(self, landmarks):
        """Analyze perspective distortion patterns"""
        if landmarks is None or len(landmarks) < 10:
            return 0.5
            
        try:
            # Analyze facial symmetry and perspective consistency
            left_eye = landmarks[33]
            right_eye = landmarks[362]
            nose = landmarks[1]
            mouth = landmarks[13]
            
            # Check if facial features follow perspective rules
            eye_y_diff = abs(left_eye.y - right_eye.y)
            face_width = abs(left_eye.x - right_eye.x)
            
            # Real faces should have minimal eye height difference relative to width
            symmetry_score = 1.0 - min(1.0, eye_y_diff / (face_width + 1e-6) * 10)
            
            return symmetry_score
            
        except Exception:
            return 0.5
    
    def _analyze_facial_structure(self, landmarks):
        """Analyze facial structure consistency with 3D face"""
        if landmarks is None:
            return 0.5
            
        try:
            # Check facial proportions that are consistent with real 3D faces
            left_eye = landmarks[33]
            right_eye = landmarks[362]
            nose = landmarks[1]
            chin = landmarks[18]
            
            # Calculate key ratios
            eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
            face_height = abs(chin.y - ((left_eye.y + right_eye.y) / 2))
            
            # Typical face ratio (golden ratio approximation)
            face_ratio = face_height / (eye_distance + 1e-6)
            
            # Real faces typically have ratio between 1.2-1.8
            if 1.2 <= face_ratio <= 1.8:
                return 1.0
            else:
                return max(0.0, 1.0 - abs(face_ratio - 1.5) / 1.5)
                
        except Exception:
            return 0.5
    
    def _analyze_temporal_depth_consistency(self):
        """Analyze depth consistency over time"""
        if len(self.depth_history) < 3:
            return 0.5
            
        recent_depths = [h.get('geometric_depth', 0.5) for h in list(self.depth_history)[-5:]]
        return 1.0 - min(1.0, np.std(recent_depths) * 2)
    
    def _classify_3d_authenticity(self, geometric, shadow, gradient, perspective, 
                                 structure, temporal):
        """Classify if depth analysis indicates real 3D face"""
        # Weighted combination of all depth indicators
        weights = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]  # Sum = 1.0
        scores = [geometric, shadow, gradient, perspective, structure, temporal]
        
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        return weighted_score > 0.6

class MicroExpressionDetector:
    """
    Advanced micro-expression detection using optical flow and landmark tracking.
    Detects involuntary facial micro-movements that distinguish real faces from static images.
    """
    
    def __init__(self):
        self.prev_landmarks = None
        self.landmark_history = deque(maxlen=60)  # 2 seconds at 30fps
        self.optical_flow_history = deque(maxlen=30)
        self.micro_movements = deque(maxlen=100)
        
    def detect(self, frames: list, landmarks_sequence: list = None) -> dict:
        """
        Detect micro-expressions and involuntary movements
        """
        try:
            if len(frames) < 2:
                return {'micro_expression_score': 0.0, 'confidence': 0.0}
            
            # 1. Optical flow analysis
            optical_flow_score = self._analyze_optical_flow(frames)
            
            # 2. Landmark micro-movements
            landmark_micro_score = 0.0
            if landmarks_sequence and len(landmarks_sequence) >= 2:
                landmark_micro_score = self._analyze_landmark_micro_movements(landmarks_sequence)
            
            # 3. Facial region micro-analysis
            facial_micro_score = self._analyze_facial_micro_regions(frames)
            
            # 4. Blink micro-patterns
            blink_micro_score = self._analyze_blink_micro_patterns(frames, landmarks_sequence)
            
            # 5. Muscle tension detection
            muscle_tension_score = self._detect_muscle_tension(frames, landmarks_sequence)
            
            # 6. Temporal pattern analysis
            temporal_pattern_score = self._analyze_temporal_patterns()
            
            micro_expression_analysis = {
                'optical_flow_score': float(optical_flow_score),
                'landmark_micro_score': float(landmark_micro_score),
                'facial_micro_score': float(facial_micro_score),
                'blink_micro_score': float(blink_micro_score),
                'muscle_tension_score': float(muscle_tension_score),
                'temporal_pattern_score': float(temporal_pattern_score),
                'micro_expression_score': self._combine_micro_scores(
                    optical_flow_score, landmark_micro_score, facial_micro_score,
                    blink_micro_score, muscle_tension_score, temporal_pattern_score
                ),
                'is_live': self._classify_micro_expression_authenticity(
                    optical_flow_score, landmark_micro_score, facial_micro_score,
                    blink_micro_score, muscle_tension_score, temporal_pattern_score
                ),
                'confidence': 0.8
            }
            
            self.micro_movements.append(micro_expression_analysis)
            return micro_expression_analysis
            
        except Exception as e:
            logger.error(f"Error in micro-expression detection: {e}")
            return {'error': str(e), 'micro_expression_score': 0.0, 'confidence': 0.0}
    
    def _analyze_optical_flow(self, frames):
        """Analyze optical flow for micro-movements"""
        if len(frames) < 2:
            return 0.0
            
        try:
            # Convert frames to grayscale
            gray_frames = []
            for frame in frames[-5:]:  # Analyze last 5 frames
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                gray_frames.append(gray)
            
            flow_magnitudes = []
            for i in range(1, len(gray_frames)):
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i],
                    None, None,
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                if flow[0] is not None:
                    # Calculate flow magnitude
                    flow_vectors = flow[0][flow[1] == 1]  # Only good features
                    if len(flow_vectors) > 0:
                        magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
                        flow_magnitudes.extend(magnitudes)
            
            if len(flow_magnitudes) == 0:
                return 0.0
            
            # Micro-movements have small but non-zero magnitudes
            micro_flow = [m for m in flow_magnitudes if 0.1 < m < 2.0]
            micro_flow_ratio = len(micro_flow) / len(flow_magnitudes)
            
            self.optical_flow_history.append(micro_flow_ratio)
            return micro_flow_ratio
            
        except Exception:
            return 0.0
    
    def _analyze_landmark_micro_movements(self, landmarks_sequence):
        """Analyze subtle movements in facial landmarks"""
        if len(landmarks_sequence) < 2:
            return 0.0
            
        try:
            # Focus on key landmarks that show micro-expressions
            key_points = [33, 362, 1, 13, 14, 15, 16, 17]  # Eyes, nose, mouth corners
            
            micro_movements = []
            for i in range(1, min(len(landmarks_sequence), 10)):
                prev_landmarks = landmarks_sequence[i-1]
                curr_landmarks = landmarks_sequence[i]
                
                if prev_landmarks and curr_landmarks:
                    for point_idx in key_points:
                        if point_idx < len(prev_landmarks) and point_idx < len(curr_landmarks):
                            prev_pt = prev_landmarks[point_idx]
                            curr_pt = curr_landmarks[point_idx]
                            
                            # Calculate micro-movement
                            movement = np.sqrt(
                                (curr_pt.x - prev_pt.x)**2 + 
                                (curr_pt.y - prev_pt.y)**2
                            )
                            
                            # Micro-movements are typically 0.001-0.01 in normalized coordinates
                            if 0.0005 < movement < 0.02:
                                micro_movements.append(movement)
            
            if len(micro_movements) == 0:
                return 0.0
            
            # Score based on frequency and consistency of micro-movements
            movement_score = min(1.0, len(micro_movements) / 50.0)
            movement_consistency = 1.0 - min(1.0, np.std(micro_movements) * 100)
            
            self.landmark_history.extend(micro_movements)
            return (movement_score + movement_consistency) / 2.0
            
        except Exception:
            return 0.0
    
    def _analyze_facial_micro_regions(self, frames):
        """Analyze micro-movements in specific facial regions"""
        if len(frames) < 2:
            return 0.0
            
        try:
            # Define regions of interest (relative coordinates)
            regions = {
                'eye_region': (0.3, 0.25, 0.7, 0.5),
                'mouth_region': (0.35, 0.6, 0.65, 0.85),
                'forehead_region': (0.3, 0.1, 0.7, 0.3)
            }
            
            micro_scores = []
            
            for region_name, (x1, y1, x2, y2) in regions.items():
                region_movements = []
                
                for i in range(1, min(len(frames), 5)):
                    prev_frame = frames[i-1]
                    curr_frame = frames[i]
                    
                    h, w = prev_frame.shape[:2]
                    
                    # Extract region
                    region_prev = prev_frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
                    region_curr = curr_frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
                    
                    if region_prev.size > 0 and region_curr.size > 0:
                        # Calculate structural similarity
                        diff = cv2.absdiff(region_prev, region_curr)
                        movement = np.mean(diff) / 255.0
                        
                        # Micro-movements are small but detectable
                        if 0.005 < movement < 0.05:
                            region_movements.append(movement)
                
                if len(region_movements) > 0:
                    micro_scores.append(np.mean(region_movements))
            
            return np.mean(micro_scores) if micro_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_blink_micro_patterns(self, frames, landmarks_sequence):
        """Analyze micro-patterns in blinking behavior"""
        if not landmarks_sequence or len(landmarks_sequence) < 10:
            return 0.0
            
        try:
            # Calculate Eye Aspect Ratio (EAR) for micro-blink detection
            ear_values = []
            
            for landmarks in landmarks_sequence[-20:]:  # Last 20 frames
                if landmarks and len(landmarks) > 42:
                    # Left eye landmarks
                    left_eye = [landmarks[i] for i in [33, 7, 163, 144, 145, 153]]
                    # Right eye landmarks  
                    right_eye = [landmarks[i] for i in [362, 382, 381, 380, 374, 373]]
                    
                    left_ear = self._calculate_ear(left_eye)
                    right_ear = self._calculate_ear(right_eye)
                    
                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_values.append(avg_ear)
            
            if len(ear_values) < 5:
                return 0.0
            
            # Analyze micro-variations in EAR
            ear_array = np.array(ear_values)
            
            # Detect micro-blinks (small rapid changes)
            ear_diff = np.diff(ear_array)
            micro_blinks = np.sum((np.abs(ear_diff) > 0.01) & (np.abs(ear_diff) < 0.1))
            
            # Natural micro-blink patterns
            micro_blink_score = min(1.0, micro_blinks / 10.0)
            
            # EAR variability indicates natural eye movement
            ear_variability = np.std(ear_values)
            variability_score = min(1.0, ear_variability * 20.0)
            
            return (micro_blink_score + variability_score) / 2.0
            
        except Exception:
            return 0.0
    
    def _calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        try:
            # Vertical eye landmarks
            A = np.sqrt((eye_landmarks[1].x - eye_landmarks[5].x)**2 + 
                       (eye_landmarks[1].y - eye_landmarks[5].y)**2)
            B = np.sqrt((eye_landmarks[2].x - eye_landmarks[4].x)**2 + 
                       (eye_landmarks[2].y - eye_landmarks[4].y)**2)
            
            # Horizontal eye landmark
            C = np.sqrt((eye_landmarks[0].x - eye_landmarks[3].x)**2 + 
                       (eye_landmarks[0].y - eye_landmarks[3].y)**2)
            
            # Calculate EAR
            ear = (A + B) / (2.0 * C + 1e-6)
            return ear
        except:
            return 0.2  # Default EAR value
    
    def _detect_muscle_tension(self, frames, landmarks_sequence):
        """Detect subtle muscle tension changes"""
        if not landmarks_sequence or len(landmarks_sequence) < 5:
            return 0.0
            
        try:
            # Focus on facial muscles that show tension
            muscle_regions = {
                'forehead': [10, 151, 9, 10],  # Forehead muscle points
                'cheek': [116, 117, 118, 119],  # Cheek muscle points
                'jaw': [172, 136, 150, 149]     # Jaw muscle points
            }
            
            tension_scores = []
            
            for region_name, point_indices in muscle_regions.items():
                region_tensions = []
                
                for i in range(1, min(len(landmarks_sequence), 10)):
                    prev_landmarks = landmarks_sequence[i-1]
                    curr_landmarks = landmarks_sequence[i]
                    
                    if prev_landmarks and curr_landmarks:
                        # Calculate micro-distance changes between muscle points
                        for j in range(len(point_indices)-1):
                            if (point_indices[j] < len(prev_landmarks) and 
                                point_indices[j+1] < len(prev_landmarks) and
                                point_indices[j] < len(curr_landmarks) and 
                                point_indices[j+1] < len(curr_landmarks)):
                                
                                # Previous distance
                                prev_dist = np.sqrt(
                                    (prev_landmarks[point_indices[j]].x - 
                                     prev_landmarks[point_indices[j+1]].x)**2 +
                                    (prev_landmarks[point_indices[j]].y - 
                                     prev_landmarks[point_indices[j+1]].y)**2
                                )
                                
                                # Current distance
                                curr_dist = np.sqrt(
                                    (curr_landmarks[point_indices[j]].x - 
                                     curr_landmarks[point_indices[j+1]].x)**2 +
                                    (curr_landmarks[point_indices[j]].y - 
                                     curr_landmarks[point_indices[j+1]].y)**2
                                )
                                
                                # Micro-tension change
                                tension_change = abs(curr_dist - prev_dist)
                                if 0.0001 < tension_change < 0.01:  # Micro-level changes
                                    region_tensions.append(tension_change)
                
                if len(region_tensions) > 0:
                    tension_scores.append(np.mean(region_tensions))
            
            return np.mean(tension_scores) if tension_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in micro-movements"""
        if len(self.micro_movements) < 10:
            return 0.5
            
        try:
            # Extract scores from history
            optical_scores = [m.get('optical_flow_score', 0) for m in self.micro_movements]
            landmark_scores = [m.get('landmark_micro_score', 0) for m in self.micro_movements]
            
            # Natural faces show temporal variability
            optical_variability = np.std(optical_scores)
            landmark_variability = np.std(landmark_scores)
            
            # Score based on natural variability patterns
            temporal_score = min(1.0, (optical_variability + landmark_variability) * 5.0)
            return temporal_score
            
        except Exception:
            return 0.5
    
    def _combine_micro_scores(self, optical, landmark, facial, blink, muscle, temporal):
        """Combine all micro-expression scores"""
        weights = [0.2, 0.25, 0.15, 0.15, 0.15, 0.1]
        scores = [optical, landmark, facial, blink, muscle, temporal]
        
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        return min(1.0, weighted_score)
    
    def _classify_micro_expression_authenticity(self, optical, landmark, facial, 
                                               blink, muscle, temporal):
        """Classify if micro-expressions indicate live face"""
        combined_score = self._combine_micro_scores(
            optical, landmark, facial, blink, muscle, temporal
        )
        return combined_score > 0.4  # Threshold for live detection

class EyeTracker:
    """
    Advanced eye tracking for analyzing natural gaze patterns vs artificial ones.
    Includes saccade detection, fixation analysis, and gaze pattern validation.
    """
    
    def __init__(self):
        self.gaze_history = deque(maxlen=100)  # Store gaze points
        self.saccade_history = deque(maxlen=50)
        self.fixation_history = deque(maxlen=30)
        
    def track(self, frames: list, landmarks_sequence: list = None) -> dict:
        """
        Comprehensive eye tracking and gaze pattern analysis
        """
        try:
            if not landmarks_sequence or len(landmarks_sequence) < 5:
                return {'eye_movement_score': 0.0, 'confidence': 0.0}
            
            # 1. Gaze direction analysis
            gaze_score = self._analyze_gaze_patterns(landmarks_sequence)
            
            # 2. Saccade detection and analysis
            saccade_score = self._analyze_saccades(landmarks_sequence)
            
            # 3. Fixation pattern analysis
            fixation_score = self._analyze_fixations(landmarks_sequence)
            
            # 4. Eye movement smoothness
            smoothness_score = self._analyze_movement_smoothness(landmarks_sequence)
            
            # 5. Pupil behavior analysis
            pupil_score = self._analyze_pupil_behavior(frames, landmarks_sequence)
            
            # 6. Binocular coordination
            coordination_score = self._analyze_binocular_coordination(landmarks_sequence)
            
            # 7. Natural eye movement patterns
            natural_pattern_score = self._analyze_natural_patterns()
            
            eye_tracking_analysis = {
                'gaze_score': float(gaze_score),
                'saccade_score': float(saccade_score),
                'fixation_score': float(fixation_score),
                'smoothness_score': float(smoothness_score),
                'pupil_score': float(pupil_score),
                'coordination_score': float(coordination_score),
                'natural_pattern_score': float(natural_pattern_score),
                'eye_movement_score': self._combine_eye_scores(
                    gaze_score, saccade_score, fixation_score, smoothness_score,
                    pupil_score, coordination_score, natural_pattern_score
                ),
                'is_natural': self._classify_eye_movement_authenticity(
                    gaze_score, saccade_score, fixation_score, smoothness_score,
                    pupil_score, coordination_score, natural_pattern_score
                ),
                'confidence': 0.85
            }
            
            return eye_tracking_analysis
            
        except Exception as e:
            logger.error(f"Error in eye tracking: {e}")
            return {'error': str(e), 'eye_movement_score': 0.0, 'confidence': 0.0}
    
    def _analyze_gaze_patterns(self, landmarks_sequence):
        """Analyze gaze direction patterns"""
        try:
            gaze_points = []
            
            for landmarks in landmarks_sequence[-20:]:  # Last 20 frames
                if landmarks and len(landmarks) > 42:
                    # Calculate gaze direction using eye landmarks
                    left_eye_center = self._get_eye_center(landmarks, 'left')
                    right_eye_center = self._get_eye_center(landmarks, 'right')
                    
                    if left_eye_center and right_eye_center:
                        # Average gaze point
                        gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
                        gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
                        gaze_points.append((gaze_x, gaze_y))
            
            if len(gaze_points) < 3:
                return 0.0
            
            self.gaze_history.extend(gaze_points)
            
            # Analyze gaze variability (natural eyes move)
            gaze_array = np.array(gaze_points)
            gaze_variance = np.var(gaze_array, axis=0)
            
            # Natural gaze shows moderate variability
            variability_score = min(1.0, (gaze_variance[0] + gaze_variance[1]) * 1000)
            
            # Analyze gaze smoothness
            if len(gaze_points) > 1:
                gaze_velocities = []
                for i in range(1, len(gaze_points)):
                    velocity = np.sqrt(
                        (gaze_points[i][0] - gaze_points[i-1][0])**2 +
                        (gaze_points[i][1] - gaze_points[i-1][1])**2
                    )
                    gaze_velocities.append(velocity)
                
                # Natural gaze has varied velocities
                velocity_variance = np.var(gaze_velocities) if gaze_velocities else 0
                velocity_score = min(1.0, velocity_variance * 10000)
                
                return (variability_score + velocity_score) / 2.0
            
            return variability_score
            
        except Exception:
            return 0.0
    
    def _get_eye_center(self, landmarks, eye):
        """Calculate eye center from landmarks"""
        try:
            if eye == 'left':
                # Left eye landmark indices
                eye_points = [33, 7, 163, 144, 145, 153]
            else:
                # Right eye landmark indices
                eye_points = [362, 382, 381, 380, 374, 373]
            
            x_coords = [landmarks[i].x for i in eye_points if i < len(landmarks)]
            y_coords = [landmarks[i].y for i in eye_points if i < len(landmarks)]
            
            if len(x_coords) >= 6:
                return (np.mean(x_coords), np.mean(y_coords))
            
            return None
        except:
            return None
    
    def _analyze_saccades(self, landmarks_sequence):
        """Detect and analyze saccadic eye movements"""
        try:
            if len(landmarks_sequence) < 5:
                return 0.0
            
            # Calculate gaze velocities
            gaze_velocities = []
            prev_gaze = None
            
            for landmarks in landmarks_sequence[-15:]:
                if landmarks and len(landmarks) > 42:
                    left_center = self._get_eye_center(landmarks, 'left')
                    right_center = self._get_eye_center(landmarks, 'right')
                    
                    if left_center and right_center:
                        current_gaze = (
                            (left_center[0] + right_center[0]) / 2,
                            (left_center[1] + right_center[1]) / 2
                        )
                        
                        if prev_gaze:
                            velocity = np.sqrt(
                                (current_gaze[0] - prev_gaze[0])**2 +
                                (current_gaze[1] - prev_gaze[1])**2
                            )
                            gaze_velocities.append(velocity)
                        
                        prev_gaze = current_gaze
            
            if len(gaze_velocities) < 3:
                return 0.0
            
            # Detect saccades (rapid eye movements)
            velocity_threshold = np.mean(gaze_velocities) + 2 * np.std(gaze_velocities)
            saccades = [v for v in gaze_velocities if v > velocity_threshold]
            
            # Natural eyes have occasional saccades
            saccade_frequency = len(saccades) / len(gaze_velocities)
            saccade_score = min(1.0, saccade_frequency * 5.0)  # Normalize
            
            self.saccade_history.append(saccade_score)
            return saccade_score
            
        except Exception:
            return 0.0
    
    def _analyze_fixations(self, landmarks_sequence):
        """Analyze eye fixation patterns"""
        try:
            if len(self.gaze_history) < 10:
                return 0.5
            
            # Group nearby gaze points as fixations
            fixations = []
            current_fixation = []
            
            gaze_points = list(self.gaze_history)[-20:]  # Last 20 points
            
            for i, point in enumerate(gaze_points):
                if not current_fixation:
                    current_fixation = [point]
                else:
                    # Check if point is close to current fixation
                    fixation_center = np.mean(current_fixation, axis=0)
                    distance = np.sqrt(
                        (point[0] - fixation_center[0])**2 +
                        (point[1] - fixation_center[1])**2
                    )
                    
                    if distance < 0.01:  # Threshold for fixation
                        current_fixation.append(point)
                    else:
                        if len(current_fixation) >= 3:  # Minimum fixation duration
                            fixations.append(current_fixation)
                        current_fixation = [point]
            
            # Add last fixation
            if len(current_fixation) >= 3:
                fixations.append(current_fixation)
            
            # Analyze fixation patterns
            if len(fixations) == 0:
                return 0.0
            
            # Natural eyes have varied fixation durations
            fixation_durations = [len(f) for f in fixations]
            duration_variance = np.var(fixation_durations)
            
            # Score based on fixation pattern naturalness
            fixation_score = min(1.0, duration_variance / 10.0)
            
            self.fixation_history.append(fixation_score)
            return fixation_score
            
        except Exception:
            return 0.5
    
    def _analyze_movement_smoothness(self, landmarks_sequence):
        """Analyze smoothness of eye movements"""
        try:
            if len(landmarks_sequence) < 5:
                return 0.0
            
            # Calculate acceleration of gaze points
            gaze_positions = []
            
            for landmarks in landmarks_sequence[-10:]:
                if landmarks and len(landmarks) > 42:
                    left_center = self._get_eye_center(landmarks, 'left')
                    right_center = self._get_eye_center(landmarks, 'right')
                    
                    if left_center and right_center:
                        avg_position = (
                            (left_center[0] + right_center[0]) / 2,
                            (left_center[1] + right_center[1]) / 2
                        )
                        gaze_positions.append(avg_position)
            
            if len(gaze_positions) < 3:
                return 0.0
            
            # Calculate velocity and acceleration
            velocities = []
            for i in range(1, len(gaze_positions)):
                vel = np.sqrt(
                    (gaze_positions[i][0] - gaze_positions[i-1][0])**2 +
                    (gaze_positions[i][1] - gaze_positions[i-1][1])**2
                )
                velocities.append(vel)
            
            accelerations = []
            for i in range(1, len(velocities)):
                acc = abs(velocities[i] - velocities[i-1])
                accelerations.append(acc)
            
            if len(accelerations) == 0:
                return 0.0
            
            # Natural eye movements have smooth acceleration patterns
            acceleration_smoothness = 1.0 - min(1.0, np.std(accelerations) * 1000)
            return max(0.0, acceleration_smoothness)
            
        except Exception:
            return 0.0
    
    def _analyze_pupil_behavior(self, frames, landmarks_sequence):
        """Analyze pupil behavior and responses"""
        try:
            # Simplified pupil analysis based on eye region brightness
            pupil_responses = []
            
            for i, landmarks in enumerate(landmarks_sequence[-10:]):
                if landmarks and i < len(frames) and len(landmarks) > 42:
                    frame = frames[i] if i < len(frames) else frames[-1]
                    
                    # Extract eye regions
                    left_eye_region = self._extract_eye_region(frame, landmarks, 'left')
                    right_eye_region = self._extract_eye_region(frame, landmarks, 'right')
                    
                    if left_eye_region is not None and right_eye_region is not None:
                        # Estimate pupil size based on dark regions
                        left_pupil = self._estimate_pupil_size(left_eye_region)
                        right_pupil = self._estimate_pupil_size(right_eye_region)
                        
                        avg_pupil = (left_pupil + right_pupil) / 2.0
                        pupil_responses.append(avg_pupil)
            
            if len(pupil_responses) < 3:
                return 0.5
            
            # Natural pupils show slight variations
            pupil_variance = np.var(pupil_responses)
            pupil_score = min(1.0, pupil_variance * 100)
            
            return pupil_score
            
        except Exception:
            return 0.5
    
    def _extract_eye_region(self, frame, landmarks, eye):
        """Extract eye region from frame"""
        try:
            h, w = frame.shape[:2]
            
            if eye == 'left':
                eye_points = [33, 7, 163, 144, 145, 153]
            else:
                eye_points = [362, 382, 381, 380, 374, 373]
            
            # Get eye bounding box
            x_coords = [int(landmarks[i].x * w) for i in eye_points if i < len(landmarks)]
            y_coords = [int(landmarks[i].y * h) for i in eye_points if i < len(landmarks)]
            
            if len(x_coords) >= 6 and len(y_coords) >= 6:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding
                padding = 5
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                eye_region = frame[y_min:y_max, x_min:x_max]
                return eye_region if eye_region.size > 0 else None
            
            return None
        except:
            return None
    
    def _estimate_pupil_size(self, eye_region):
        """Estimate pupil size from eye region"""
        try:
            if len(eye_region.shape) == 3:
                gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_eye = eye_region
            
            # Find darkest regions (pupil approximation)
            _, threshold = cv2.threshold(gray_eye, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_pixels = np.sum(threshold == 0)
            total_pixels = threshold.size
            
            pupil_ratio = dark_pixels / total_pixels
            return pupil_ratio
            
        except:
            return 0.1  # Default pupil size
    
    def _analyze_binocular_coordination(self, landmarks_sequence):
        """Analyze coordination between both eyes"""
        try:
            coordination_scores = []
            
            for landmarks in landmarks_sequence[-10:]:
                if landmarks and len(landmarks) > 42:
                    # Get eye centers
                    left_center = self._get_eye_center(landmarks, 'left')
                    right_center = self._get_eye_center(landmarks, 'right')
                    
                    if left_center and right_center:
                        # Calculate eye alignment
                        y_difference = abs(left_center[1] - right_center[1])
                        x_distance = abs(left_center[0] - right_center[0])
                        
                        # Good coordination means eyes are aligned horizontally
                        coordination = 1.0 - min(1.0, y_difference / (x_distance + 1e-6) * 10)
                        coordination_scores.append(coordination)
            
            return np.mean(coordination_scores) if coordination_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_natural_patterns(self):
        """Analyze overall natural eye movement patterns"""
        try:
            if len(self.gaze_history) < 20:
                return 0.5
            
            # Analyze temporal patterns in gaze
            recent_gaze = list(self.gaze_history)[-20:]
            gaze_array = np.array(recent_gaze)
            
            # Natural eyes show complex movement patterns
            x_movement = gaze_array[:, 0]
            y_movement = gaze_array[:, 1]
            
            # Check for movement complexity
            x_complexity = len(np.where(np.diff(x_movement) != 0)[0]) / len(x_movement)
            y_complexity = len(np.where(np.diff(y_movement) != 0)[0]) / len(y_movement)
            
            complexity_score = (x_complexity + y_complexity) / 2.0
            
            # Check for natural frequency patterns
            if len(recent_gaze) > 10:
                x_fft = np.abs(fft(x_movement))
                y_fft = np.abs(fft(y_movement))
                
                # Natural eye movements have specific frequency characteristics
                frequency_score = min(1.0, (np.std(x_fft) + np.std(y_fft)) / 2.0)
            else:
                frequency_score = 0.5
            
            return (complexity_score + frequency_score) / 2.0
            
        except Exception:
            return 0.5
    
    def _combine_eye_scores(self, gaze, saccade, fixation, smoothness, 
                           pupil, coordination, natural):
        """Combine all eye tracking scores"""
        weights = [0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15]
        scores = [gaze, saccade, fixation, smoothness, pupil, coordination, natural]
        
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        return min(1.0, weighted_score)
    
    def _classify_eye_movement_authenticity(self, gaze, saccade, fixation, smoothness,
                                          pupil, coordination, natural):
        """Classify if eye movements indicate live person"""
        combined_score = self._combine_eye_scores(
            gaze, saccade, fixation, smoothness, pupil, coordination, natural
        )
        return combined_score > 0.5  # Threshold for natural eye movement

class RemotePPGDetector:
    """
    Advanced remote photoplethysmography (PPG) for heartbeat detection.
    Uses subtle color changes to detect blood flow and cardiac rhythm.
    """
    
    def __init__(self):
        self.ppg_history = deque(maxlen=150)  # 5 seconds at 30fps
        self.heart_rate_history = deque(maxlen=20)
        self.signal_quality_history = deque(maxlen=50)
        
    def detect(self, frames: list, landmarks_sequence: list = None) -> dict:
        """
        Comprehensive remote PPG analysis for liveness detection
        """
        try:
            if len(frames) < 30:  # Need at least 1 second of data
                return {'heartbeat_score': 0.0, 'confidence': 0.0}
            
            # 1. Extract facial regions for PPG analysis
            ppg_signals = self._extract_ppg_signals(frames, landmarks_sequence)
            
            # 2. Signal preprocessing and filtering
            filtered_signals = self._preprocess_ppg_signals(ppg_signals)
            
            # 3. Heart rate estimation
            heart_rate = self._estimate_heart_rate(filtered_signals)
            
            # 4. Signal quality assessment
            signal_quality = self._assess_signal_quality(filtered_signals)
            
            # 5. Cardiac rhythm analysis
            rhythm_score = self._analyze_cardiac_rhythm(filtered_signals)
            
            # 6. Temporal consistency analysis
            temporal_consistency = self._analyze_temporal_consistency()
            
            # 7. PPG authenticity validation
            authenticity_score = self._validate_ppg_authenticity(filtered_signals)
            
            ppg_analysis = {
                'heart_rate': float(heart_rate) if heart_rate else 0.0,
                'signal_quality': float(signal_quality),
                'rhythm_score': float(rhythm_score),
                'temporal_consistency': float(temporal_consistency),
                'authenticity_score': float(authenticity_score),
                'heartbeat_score': self._combine_ppg_scores(
                    signal_quality, rhythm_score, temporal_consistency, authenticity_score
                ),
                'is_live': self._classify_ppg_liveness(
                    heart_rate, signal_quality, rhythm_score, 
                    temporal_consistency, authenticity_score
                ),
                'confidence': 0.75
            }
            
            self.ppg_history.extend(filtered_signals)
            if heart_rate:
                self.heart_rate_history.append(heart_rate)
            self.signal_quality_history.append(signal_quality)
            
            return ppg_analysis
            
        except Exception as e:
            logger.error(f"Error in PPG detection: {e}")
            return {'error': str(e), 'heartbeat_score': 0.0, 'confidence': 0.0}
    
    def _extract_ppg_signals(self, frames, landmarks_sequence):
        """Extract PPG signals from facial regions"""
        try:
            ppg_signals = []
            
            # Define regions of interest for PPG
            roi_regions = ['forehead', 'left_cheek', 'right_cheek', 'nose']
            
            for i, frame in enumerate(frames[-60:]):  # Last 2 seconds
                frame_signals = {}
                
                if (landmarks_sequence and i < len(landmarks_sequence) and 
                    landmarks_sequence[i]):
                    landmarks = landmarks_sequence[i]
                    
                    # Extract each ROI
                    for region in roi_regions:
                        roi_signal = self._extract_roi_signal(frame, landmarks, region)
                        if roi_signal is not None:
                            frame_signals[region] = roi_signal
                
                # If no landmarks, use default regions
                if not frame_signals:
                    frame_signals = self._extract_default_roi_signals(frame)
                
                ppg_signals.append(frame_signals)
            
            return ppg_signals
            
        except Exception:
            return []
    
    def _extract_roi_signal(self, frame, landmarks, region):
        """Extract PPG signal from specific facial region"""
        try:
            h, w = frame.shape[:2]
            
            # Define region coordinates based on landmarks
            if region == 'forehead':
                # Forehead region
                center_x = int(landmarks[10].x * w)  # Forehead center
                center_y = int(landmarks[10].y * h)
                roi_size = min(w, h) // 15
                
            elif region == 'left_cheek':
                # Left cheek region
                center_x = int(landmarks[116].x * w)
                center_y = int(landmarks[116].y * h)
                roi_size = min(w, h) // 20
                
            elif region == 'right_cheek':
                # Right cheek region
                center_x = int(landmarks[345].x * w)
                center_y = int(landmarks[345].y * h)
                roi_size = min(w, h) // 20
                
            elif region == 'nose':
                # Nose bridge region
                center_x = int(landmarks[6].x * w)
                center_y = int(landmarks[6].y * h)
                roi_size = min(w, h) // 25
                
            else:
                return None
            
            # Extract ROI
            x1 = max(0, center_x - roi_size)
            x2 = min(w, center_x + roi_size)
            y1 = max(0, center_y - roi_size)
            y2 = min(h, center_y + roi_size)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Extract green channel (most sensitive to blood flow)
            if len(roi.shape) == 3:
                green_channel = roi[:, :, 1]  # Green channel
            else:
                green_channel = roi
            
            # Calculate mean intensity
            mean_intensity = np.mean(green_channel)
            return mean_intensity
            
        except Exception:
            return None
    
    def _extract_default_roi_signals(self, frame):
        """Extract ROI signals when landmarks are not available"""
        try:
            h, w = frame.shape[:2]
            signals = {}
            
            # Default regions (relative coordinates)
            default_regions = {
                'forehead': (0.4, 0.15, 0.6, 0.3),
                'left_cheek': (0.25, 0.4, 0.4, 0.6),
                'right_cheek': (0.6, 0.4, 0.75, 0.6),
                'center': (0.4, 0.3, 0.6, 0.7)
            }
            
            for region, (x1_rel, y1_rel, x2_rel, y2_rel) in default_regions.items():
                x1 = int(x1_rel * w)
                x2 = int(x2_rel * w)
                y1 = int(y1_rel * h)
                y2 = int(y2_rel * h)
                
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    if len(roi.shape) == 3:
                        green_channel = roi[:, :, 1]
                    else:
                        green_channel = roi
                    
                    signals[region] = np.mean(green_channel)
            
            return signals
            
        except Exception:
            return {}
    
    def _preprocess_ppg_signals(self, ppg_signals):
        """Preprocess and filter PPG signals"""
        try:
            if len(ppg_signals) < 10:
                return []
            
            # Extract time series for each region
            region_series = {}
            
            # Get all unique regions
            all_regions = set()
            for frame_signals in ppg_signals:
                all_regions.update(frame_signals.keys())
            
            # Build time series for each region
            for region in all_regions:
                series = []
                for frame_signals in ppg_signals:
                    value = frame_signals.get(region, 0)
                    series.append(value)
                region_series[region] = series
            
            # Filter each time series
            filtered_series = {}
            for region, series in region_series.items():
                if len(series) >= 10:
                    filtered = self._apply_ppg_filter(series)
                    filtered_series[region] = filtered
            
            return filtered_series
            
        except Exception:
            return {}
    
    def _apply_ppg_filter(self, signal):
        """Apply bandpass filter for PPG signal"""
        try:
            signal_array = np.array(signal, dtype=np.float64)
            
            # Remove DC component
            signal_array = signal_array - np.mean(signal_array)
            
            # Simple moving average filter
            if len(signal_array) >= 5:
                kernel_size = min(5, len(signal_array))
                kernel = np.ones(kernel_size) / kernel_size
                
                # Pad signal for convolution
                padded_signal = np.pad(signal_array, kernel_size//2, mode='edge')
                filtered = np.convolve(padded_signal, kernel, mode='valid')
                
                # Ensure output length matches input
                if len(filtered) != len(signal_array):
                    filtered = filtered[:len(signal_array)]
                
                return filtered.tolist()
            
            return signal_array.tolist()
            
        except Exception:
            return signal
    
    def _estimate_heart_rate(self, filtered_signals):
        """Estimate heart rate from PPG signals"""
        try:
            if not filtered_signals:
                return None
            
            # Use the signal with best quality
            best_signal = None
            best_quality = 0
            
            for region, signal in filtered_signals.items():
                if len(signal) >= 30:  # Need at least 1 second
                    quality = self._calculate_signal_quality(signal)
                    if quality > best_quality:
                        best_quality = quality
                        best_signal = signal
            
            if best_signal is None:
                return None
            
            # Estimate heart rate using FFT
            signal_array = np.array(best_signal)
            
            # Apply window function
            windowed_signal = signal_array * np.hamming(len(signal_array))
            
            # Calculate FFT
            fft_result = np.abs(fft(windowed_signal))
            
            # Frequency range for heart rate (0.5-3 Hz, 30-180 BPM)
            fs = 30  # Assuming 30 fps
            freqs = fftfreq(len(fft_result), 1/fs)
            
            # Find peak in heart rate frequency range
            hr_indices = np.where((freqs >= 0.5) & (freqs <= 3.0))[0]
            
            if len(hr_indices) > 0:
                hr_fft = fft_result[hr_indices]
                hr_freqs = freqs[hr_indices]
                
                # Find dominant frequency
                peak_idx = np.argmax(hr_fft)
                dominant_freq = hr_freqs[peak_idx]
                
                # Convert to BPM
                heart_rate = dominant_freq * 60
                
                # Validate heart rate range
                if 30 <= heart_rate <= 180:
                    return heart_rate
            
            return None
            
        except Exception:
            return None
    
    def _calculate_signal_quality(self, signal):
        """Calculate PPG signal quality"""
        try:
            signal_array = np.array(signal)
            
            if len(signal_array) < 5:
                return 0.0
            
            # Signal-to-noise ratio estimation
            signal_power = np.var(signal_array)
            
            # Estimate noise using high-frequency components
            if len(signal_array) > 10:
                diff_signal = np.diff(signal_array)
                noise_power = np.var(diff_signal)
                
                snr = signal_power / (noise_power + 1e-6)
                quality = min(1.0, snr / 10.0)
            else:
                quality = signal_power / 1000.0  # Rough estimate
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.0
    
    def _assess_signal_quality(self, filtered_signals):
        """Assess overall signal quality"""
        try:
            if not filtered_signals:
                return 0.0
            
            qualities = []
            for region, signal in filtered_signals.items():
                quality = self._calculate_signal_quality(signal)
                qualities.append(quality)
            
            return np.mean(qualities) if qualities else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_cardiac_rhythm(self, filtered_signals):
        """Analyze cardiac rhythm patterns"""
        try:
            if not filtered_signals:
                return 0.0
            
            rhythm_scores = []
            
            for region, signal in filtered_signals.items():
                if len(signal) >= 30:
                    # Analyze rhythm regularity
                    signal_array = np.array(signal)
                    
                    # Find peaks (heartbeats)
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(signal_array, height=np.mean(signal_array))
                    
                    if len(peaks) >= 3:
                        # Calculate inter-beat intervals
                        intervals = np.diff(peaks)
                        
                        # Regular rhythm has consistent intervals
                        if len(intervals) > 1:
                            interval_std = np.std(intervals)
                            interval_mean = np.mean(intervals)
                            
                            rhythm_regularity = 1.0 - min(1.0, interval_std / (interval_mean + 1e-6))
                            rhythm_scores.append(rhythm_regularity)
            
            return np.mean(rhythm_scores) if rhythm_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_temporal_consistency(self):
        """Analyze temporal consistency of PPG signals"""
        try:
            if len(self.heart_rate_history) < 3:
                return 0.5
            
            # Heart rate should be relatively stable
            recent_hr = list(self.heart_rate_history)[-5:]
            hr_std = np.std(recent_hr)
            hr_mean = np.mean(recent_hr)
            
            # Consistent heart rate indicates live person
            consistency = 1.0 - min(1.0, hr_std / (hr_mean + 1e-6) * 5)
            return max(0.0, consistency)
            
        except Exception:
            return 0.5
    
    def _validate_ppg_authenticity(self, filtered_signals):
        """Validate if PPG signals are authentic"""
        try:
            if not filtered_signals:
                return 0.0
            
            authenticity_scores = []
            
            for region, signal in filtered_signals.items():
                if len(signal) >= 20:
                    # Check for natural PPG characteristics
                    signal_array = np.array(signal)
                    
                    # 1. Amplitude variability (natural hearts have some variation)
                    amplitude_var = np.std(signal_array) / (np.mean(np.abs(signal_array)) + 1e-6)
                    amplitude_score = min(1.0, amplitude_var * 5)
                    
                    # 2. Frequency content (should be in physiological range)
                    fft_result = np.abs(fft(signal_array))
                    freqs = fftfreq(len(fft_result), 1/30)  # 30 fps
                    
                    # Energy in heart rate frequency band
                    hr_indices = np.where((freqs >= 0.5) & (freqs <= 3.0))[0]
                    total_energy = np.sum(fft_result)
                    hr_energy = np.sum(fft_result[hr_indices])
                    
                    frequency_score = hr_energy / (total_energy + 1e-6)
                    
                    # 3. Signal complexity (natural signals have some complexity)
                    complexity = len(np.where(np.diff(signal_array) != 0)[0]) / len(signal_array)
                    complexity_score = min(1.0, complexity * 2)
                    
                    # Combine scores
                    region_authenticity = (amplitude_score + frequency_score + complexity_score) / 3.0
                    authenticity_scores.append(region_authenticity)
            
            return np.mean(authenticity_scores) if authenticity_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _combine_ppg_scores(self, signal_quality, rhythm, temporal, authenticity):
        """Combine all PPG analysis scores"""
        weights = [0.3, 0.25, 0.25, 0.2]
        scores = [signal_quality, rhythm, temporal, authenticity]
        
        weighted_score = sum(w * s for w, s in zip(weights, scores))
        return min(1.0, weighted_score)
    
    def _classify_ppg_liveness(self, heart_rate, signal_quality, rhythm, 
                              temporal, authenticity):
        """Classify if PPG indicates live person"""
        # Heart rate must be in reasonable range
        if heart_rate is None or not (40 <= heart_rate <= 200):
            return False
        
        # Combine all factors
        combined_score = self._combine_ppg_scores(signal_quality, rhythm, temporal, authenticity)
        
        # Additional checks
        quality_threshold = signal_quality > 0.3
        rhythm_threshold = rhythm > 0.4
        
        return combined_score > 0.5 and quality_threshold and rhythm_threshold


# Advanced Anti-Spoofing Integration Class
class AdvancedAntiSpoofingProcessor:
    """
    Main integration class that combines all advanced anti-spoofing techniques.
    Provides unified interface for comprehensive spoofing detection.
    """
    
    def __init__(self):
        self.texture_analyzer = TextureAnalyzer()
        self.depth_estimator = DepthEstimator()
        self.micro_expression_detector = MicroExpressionDetector()
        self.eye_tracker = EyeTracker()
        self.ppg_detector = RemotePPGDetector()
        
        self.detection_history = deque(maxlen=100)
        
    def process_comprehensive_antispoofing(self, frames: list, landmarks_sequence: list = None) -> dict:
        """
        Comprehensive anti-spoofing analysis using all advanced techniques
        """
        try:
            if not frames:
                return {'error': 'No frames provided', 'is_live': False, 'confidence': 0.0}
            
            current_frame = frames[-1]
            current_landmarks = landmarks_sequence[-1] if landmarks_sequence else None
            
            # 1. Texture Analysis
            texture_result = self.texture_analyzer.analyze(current_frame, current_landmarks)
            
            # 2. Depth Estimation
            depth_result = self.depth_estimator.estimate(current_frame, current_landmarks)
            
            # 3. Micro-Expression Detection (needs multiple frames)
            micro_result = self.micro_expression_detector.detect(frames, landmarks_sequence)
            
            # 4. Eye Tracking
            eye_result = self.eye_tracker.track(frames, landmarks_sequence)
            
            # 5. PPG Detection
            ppg_result = self.ppg_detector.detect(frames, landmarks_sequence)
            
            # 6. Advanced Fusion Logic
            fusion_result = self._advanced_fusion(
                texture_result, depth_result, micro_result, eye_result, ppg_result
            )
            
            comprehensive_result = {
                'texture_analysis': texture_result,
                'depth_analysis': depth_result,
                'micro_expression_analysis': micro_result,
                'eye_tracking_analysis': eye_result,
                'ppg_analysis': ppg_result,
                'fusion_result': fusion_result,
                'is_live': fusion_result['is_live'],
                'confidence': fusion_result['confidence'],
                'spoofing_type': fusion_result.get('spoofing_type', 'unknown'),
                'timestamp': time.time()
            }
            
            self.detection_history.append(comprehensive_result)
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive anti-spoofing: {e}")
            return {
                'error': str(e),
                'is_live': False,
                'confidence': 0.0,
                'spoofing_type': 'error'
            }
    
    def _advanced_fusion(self, texture, depth, micro, eye, ppg):
        """
        Advanced fusion logic combining all anti-spoofing methods
        """
        try:
            # Extract individual authenticity scores
            texture_authentic = texture.get('is_authentic', False)
            texture_conf = texture.get('confidence', 0.0)
            
            depth_authentic = depth.get('is_3d', False)
            depth_conf = depth.get('confidence', 0.0)
            
            micro_authentic = micro.get('is_live', False)
            micro_conf = micro.get('confidence', 0.0)
            
            eye_authentic = eye.get('is_natural', False)
            eye_conf = eye.get('confidence', 0.0)
            
            ppg_authentic = ppg.get('is_live', False)
            ppg_conf = ppg.get('confidence', 0.0)
            
            # Weighted voting system
            methods = [
                ('texture', texture_authentic, texture_conf, 0.2),
                ('depth', depth_authentic, depth_conf, 0.25),
                ('micro_expression', micro_authentic, micro_conf, 0.2),
                ('eye_tracking', eye_authentic, eye_conf, 0.2),
                ('ppg', ppg_authentic, ppg_conf, 0.15)
            ]
            
            # Calculate weighted score
            total_weight = 0
            weighted_score = 0
            failed_methods = []
            
            for method_name, authentic, confidence, weight in methods:
                if confidence > 0.1:  # Only consider methods with reasonable confidence
                    score = 1.0 if authentic else 0.0
                    weighted_score += score * weight * confidence
                    total_weight += weight * confidence
                    
                    if not authentic:
                        failed_methods.append({
                            'method': method_name,
                            'confidence': confidence,
                            'reason': self._get_failure_reason(method_name, texture, depth, micro, eye, ppg)
                        })
            
            # Normalize score
            final_score = weighted_score / (total_weight + 1e-6) if total_weight > 0 else 0.0
            
            # Determine spoofing type based on failed methods
            spoofing_type = self._determine_spoofing_type(failed_methods, texture, depth, micro, eye, ppg)
            
            # Final decision
            is_live = final_score > 0.6 and len(failed_methods) <= 2
            
            # Confidence calculation
            method_confidences = [conf for _, _, conf, _ in methods if conf > 0.1]
            avg_confidence = np.mean(method_confidences) if method_confidences else 0.0
            
            # Adjust confidence based on agreement
            agreement_factor = 1.0 - (len(failed_methods) / 5.0)  # 5 total methods
            final_confidence = avg_confidence * agreement_factor
            
            return {
                'is_live': is_live,
                'confidence': final_confidence,
                'weighted_score': final_score,
                'failed_methods': failed_methods,
                'spoofing_type': spoofing_type,
                'method_scores': {
                    'texture': texture_authentic,
                    'depth': depth_authentic,
                    'micro_expression': micro_authentic,
                    'eye_tracking': eye_authentic,
                    'ppg': ppg_authentic
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced fusion: {e}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'error': str(e),
                'spoofing_type': 'unknown'
            }
    
    def _get_failure_reason(self, method_name, texture, depth, micro, eye, ppg):
        """Get specific failure reason for each method"""
        try:
            if method_name == 'texture':
                if texture.get('moire_score', 0) > 0.5:
                    return "Screen/digital display detected"
                elif texture.get('print_artifacts', 0) > 0.3:
                    return "Print artifacts detected"
                elif texture.get('screen_reflection', 0) > 0.4:
                    return "Unnatural screen reflections"
                else:
                    return "Artificial texture patterns"
                    
            elif method_name == 'depth':
                if depth.get('geometric_depth', 0) < 0.3:
                    return "Flat surface detected"
                elif depth.get('shadow_depth', 0) < 0.2:
                    return "Lack of natural shadows"
                else:
                    return "2D image characteristics"
                    
            elif method_name == 'micro_expression':
                if micro.get('optical_flow_score', 0) < 0.2:
                    return "No natural micro-movements"
                elif micro.get('blink_micro_score', 0) < 0.3:
                    return "Unnatural blinking patterns"
                else:
                    return "Static facial expression"
                    
            elif method_name == 'eye_tracking':
                if eye.get('gaze_score', 0) < 0.3:
                    return "Unnatural gaze patterns"
                elif eye.get('saccade_score', 0) < 0.2:
                    return "Lack of natural eye movements"
                else:
                    return "Artificial eye behavior"
                    
            elif method_name == 'ppg':
                if ppg.get('heart_rate', 0) == 0:
                    return "No heartbeat detected"
                elif ppg.get('signal_quality', 0) < 0.3:
                    return "Poor blood flow signal"
                else:
                    return "Unnatural cardiac patterns"
                    
            return "Method-specific analysis failed"
            
        except Exception:
            return "Analysis error"
    
    def _determine_spoofing_type(self, failed_methods, texture, depth, micro, eye, ppg):
        """Determine the type of spoofing attack"""
        try:
            failed_method_names = [m['method'] for m in failed_methods]
            
            # Photo attack
            if 'depth' in failed_method_names and 'micro_expression' in failed_method_names:
                if texture.get('print_artifacts', 0) > 0.3:
                    return "printed_photo"
                else:
                    return "digital_photo"
            
            # Video attack
            elif 'ppg' in failed_method_names and 'eye_tracking' in failed_method_names:
                return "video_replay"
            
            # Screen attack
            elif 'texture' in failed_method_names:
                if texture.get('moire_score', 0) > 0.5:
                    return "screen_display"
                elif texture.get('screen_reflection', 0) > 0.4:
                    return "digital_screen"
            
            # Mask attack
            elif 'micro_expression' in failed_method_names and 'ppg' in failed_method_names:
                return "mask_attack"
            
            # Multiple failures indicate sophisticated attack
            elif len(failed_methods) >= 3:
                return "sophisticated_attack"
            
            # Single method failure
            elif len(failed_methods) == 1:
                return f"possible_{failed_methods[0]['method']}_issue"
            
            return "unknown_spoofing"
            
        except Exception:
            return "unknown"


# Integration example and utility functions
def integrate_with_existing_pipeline(frame, landmarks=None, frame_history=None, landmark_history=None):
    """
    Integration function for existing anti-spoofing pipeline
    """
    try:
        # Initialize processor (in practice, this would be a singleton)
        processor = AdvancedAntiSpoofingProcessor()
        
        # Prepare frame and landmark sequences
        frames = frame_history if frame_history else [frame]
        landmarks_seq = landmark_history if landmark_history else ([landmarks] if landmarks else None)
        
        # Process with advanced anti-spoofing
        result = processor.process_comprehensive_antispoofing(frames, landmarks_seq)
        
        # Return results in format compatible with existing fusion logic
        return {
            'advanced_antispoofing': {
                'is_live': result['is_live'],
                'confidence': result['confidence'],
                'spoofing_type': result.get('spoofing_type', 'unknown'),
                'detailed_analysis': {
                    'texture_score': result['texture_analysis'].get('is_authentic', False),
                    'depth_score': result['depth_analysis'].get('is_3d', False),
                    'micro_expression_score': result['micro_expression_analysis'].get('is_live', False),
                    'eye_tracking_score': result['eye_tracking_analysis'].get('is_natural', False),
                    'ppg_score': result['ppg_analysis'].get('is_live', False)
                },
                'failed_methods': result['fusion_result'].get('failed_methods', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Error in advanced anti-spoofing integration: {e}")
        return {
            'advanced_antispoofing': {
                'is_live': False,
                'confidence': 0.0,
                'error': str(e)
            }
        }

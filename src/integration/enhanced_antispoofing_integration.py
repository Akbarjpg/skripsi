"""
Advanced Anti-Spoofing Integration Module

Integrates the advanced anti-spoofing techniques with the existing
face recognition and liveness detection system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time

# Import existing modules
try:
    from src.detection.advanced_antispoofing import (
        AdvancedAntiSpoofingProcessor,
        TextureAnalyzer,
        DepthEstimator,
        MicroExpressionDetector,
        EyeTracker,
        RemotePPGDetector
    )
except ImportError as e:
    print(f"Warning: Could not import advanced anti-spoofing: {e}")
    AdvancedAntiSpoofingProcessor = None

# Import validation framework
try:
    from src.testing.antispoofing_validator import AntiSpoofingValidator
except ImportError as e:
    print(f"Warning: Could not import validation framework: {e}")
    AntiSpoofingValidator = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSecurityResult:
    """Enhanced security assessment result with advanced anti-spoofing"""
    is_live: bool
    confidence: float
    
    # Basic detection scores
    basic_liveness_score: float
    blink_detection_score: float
    face_quality_score: float
    
    # Advanced anti-spoofing scores
    texture_analysis_score: float
    depth_estimation_score: float
    micro_expression_score: float
    eye_tracking_score: float
    ppg_detection_score: float
    
    # Fusion results
    fusion_confidence: float
    spoofing_type_detected: str
    risk_level: str
    
    # Processing metrics
    processing_time: float
    detection_methods_used: List[str]
    
    # Detailed analysis
    detailed_scores: Dict[str, Any]
    recommendations: List[str]

class IntegratedAntiSpoofingSystem:
    """
    Integrated anti-spoofing system combining basic and advanced techniques
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize integrated anti-spoofing system"""
        self.config = config or self._get_default_config()
        
        # Initialize processors
        self.advanced_processor = None
        self.validator = None
        
        # Initialize components if available
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_processing_time': 0.0,
            'detection_accuracy': 0.0
        }
        
        logger.info("Integrated Anti-Spoofing System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_advanced_detection': True,
            'enable_basic_detection': True,
            'fusion_weights': {
                'basic_liveness': 0.25,
                'texture_analysis': 0.20,
                'depth_estimation': 0.15,
                'micro_expression': 0.15,
                'eye_tracking': 0.15,
                'ppg_detection': 0.10
            },
            'confidence_thresholds': {
                'high_confidence': 0.9,
                'medium_confidence': 0.7,
                'low_confidence': 0.5
            },
            'processing_timeout': 5.0,  # seconds
            'fallback_to_basic': True,
            'enable_performance_tracking': True
        }
    
    def _initialize_components(self):
        """Initialize system components"""
        try:
            if AdvancedAntiSpoofingProcessor and self.config.get('enable_advanced_detection', True):
                self.advanced_processor = AdvancedAntiSpoofingProcessor()
                logger.info("Advanced anti-spoofing processor initialized")
            
            if AntiSpoofingValidator:
                self.validator = AntiSpoofingValidator()
                logger.info("Anti-spoofing validator initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            if self.config.get('fallback_to_basic', True):
                logger.info("Falling back to basic detection only")
            else:
                raise
    
    def process_comprehensive_detection(self, image: np.ndarray, 
                                      face_landmarks: np.ndarray = None,
                                      face_bbox: Tuple[int, int, int, int] = None) -> EnhancedSecurityResult:
        """
        Process comprehensive anti-spoofing detection
        
        Args:
            image: Input image
            face_landmarks: Optional face landmarks
            face_bbox: Optional face bounding box
            
        Returns:
            EnhancedSecurityResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Initialize scores
            scores = {
                'basic_liveness': 0.5,
                'blink_detection': 0.5,
                'face_quality': 0.5,
                'texture_analysis': 0.5,
                'depth_estimation': 0.5,
                'micro_expression': 0.5,
                'eye_tracking': 0.5,
                'ppg_detection': 0.5
            }
            
            detection_methods = []
            detailed_scores = {}
            recommendations = []
            
            # 1. Basic liveness detection (existing system)
            if self.config.get('enable_basic_detection', True):
                basic_result = self._run_basic_detection(image, face_landmarks, face_bbox)
                scores.update(basic_result['scores'])
                detailed_scores.update(basic_result['detailed'])
                detection_methods.extend(basic_result['methods'])
                recommendations.extend(basic_result['recommendations'])
            
            # 2. Advanced anti-spoofing detection
            if self.advanced_processor and self.config.get('enable_advanced_detection', True):
                advanced_result = self._run_advanced_detection(image, face_landmarks, face_bbox)
                scores.update(advanced_result['scores'])
                detailed_scores.update(advanced_result['detailed'])
                detection_methods.extend(advanced_result['methods'])
                recommendations.extend(advanced_result['recommendations'])
            
            # 3. Fusion and final decision
            fusion_result = self._perform_fusion(scores, detailed_scores)
            
            # 4. Risk assessment
            risk_assessment = self._assess_risk_level(fusion_result, detailed_scores)
            
            # 5. Generate final result
            processing_time = time.time() - start_time
            
            result = EnhancedSecurityResult(
                is_live=fusion_result['is_live'],
                confidence=fusion_result['confidence'],
                basic_liveness_score=scores['basic_liveness'],
                blink_detection_score=scores['blink_detection'],
                face_quality_score=scores['face_quality'],
                texture_analysis_score=scores['texture_analysis'],
                depth_estimation_score=scores['depth_estimation'],
                micro_expression_score=scores['micro_expression'],
                eye_tracking_score=scores['eye_tracking'],
                ppg_detection_score=scores['ppg_detection'],
                fusion_confidence=fusion_result['fusion_confidence'],
                spoofing_type_detected=fusion_result['spoofing_type'],
                risk_level=risk_assessment['risk_level'],
                processing_time=processing_time,
                detection_methods_used=detection_methods,
                detailed_scores=detailed_scores,
                recommendations=recommendations
            )
            
            # Update performance statistics
            self._update_performance_stats(result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive detection: {e}")
            
            # Fallback result
            processing_time = time.time() - start_time
            return EnhancedSecurityResult(
                is_live=False,
                confidence=0.0,
                basic_liveness_score=0.0,
                blink_detection_score=0.0,
                face_quality_score=0.0,
                texture_analysis_score=0.0,
                depth_estimation_score=0.0,
                micro_expression_score=0.0,
                eye_tracking_score=0.0,
                ppg_detection_score=0.0,
                fusion_confidence=0.0,
                spoofing_type_detected="error",
                risk_level="high",
                processing_time=processing_time,
                detection_methods_used=["error_fallback"],
                detailed_scores={"error": str(e)},
                recommendations=["System error occurred, manual verification required"]
            )
    
    def _run_basic_detection(self, image: np.ndarray, face_landmarks: np.ndarray = None,
                           face_bbox: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
        """Run basic liveness detection methods"""
        try:
            scores = {}
            detailed = {}
            methods = []
            recommendations = []
            
            # Simulate basic detection (replace with actual implementation)
            # This would call existing methods like:
            # - Blink detection
            # - Face quality assessment
            # - Basic movement detection
            
            # Basic liveness score (placeholder)
            basic_score = self._calculate_basic_liveness_score(image)
            scores['basic_liveness'] = basic_score
            detailed['basic_liveness_details'] = {
                'face_detected': face_bbox is not None,
                'landmarks_available': face_landmarks is not None,
                'image_quality': self._assess_image_quality(image)
            }
            methods.append('basic_liveness')
            
            # Blink detection score (placeholder)
            blink_score = self._calculate_blink_score(image, face_landmarks)
            scores['blink_detection'] = blink_score
            detailed['blink_detection_details'] = {
                'blink_detected': blink_score > 0.5,
                'eye_aspect_ratio': 0.25  # Placeholder
            }
            methods.append('blink_detection')
            
            # Face quality score
            quality_score = self._assess_face_quality(image, face_bbox)
            scores['face_quality'] = quality_score
            detailed['face_quality_details'] = {
                'resolution': f"{image.shape[1]}x{image.shape[0]}",
                'brightness': np.mean(image),
                'sharpness': self._calculate_sharpness(image)
            }
            methods.append('face_quality')
            
            # Generate recommendations
            if basic_score < 0.5:
                recommendations.append("Basic liveness detection indicates potential spoofing")
            if blink_score < 0.3:
                recommendations.append("No blink detected - consider requesting user to blink")
            if quality_score < 0.6:
                recommendations.append("Image quality is poor - improve lighting or camera position")
            
            return {
                'scores': scores,
                'detailed': detailed,
                'methods': methods,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in basic detection: {e}")
            return {
                'scores': {'basic_liveness': 0.0, 'blink_detection': 0.0, 'face_quality': 0.0},
                'detailed': {'error': str(e)},
                'methods': ['basic_error'],
                'recommendations': ['Basic detection failed']
            }
    
    def _run_advanced_detection(self, image: np.ndarray, face_landmarks: np.ndarray = None,
                              face_bbox: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
        """Run advanced anti-spoofing detection methods"""
        try:
            scores = {}
            detailed = {}
            methods = []
            recommendations = []
            
            if not self.advanced_processor:
                return {
                    'scores': {},
                    'detailed': {'advanced_not_available': True},
                    'methods': [],
                    'recommendations': ['Advanced detection not available']
                }
            
            # Run comprehensive advanced detection
            advanced_result = self.advanced_processor.process_comprehensive_antispoofing([image])
            
            if advanced_result and len(advanced_result) > 0:
                result = advanced_result[0]
                
                # Extract scores
                scores['texture_analysis'] = result.get('texture_analysis', {}).get('confidence', 0.5)
                scores['depth_estimation'] = result.get('depth_analysis', {}).get('confidence', 0.5)
                scores['micro_expression'] = result.get('micro_expression', {}).get('confidence', 0.5)
                scores['eye_tracking'] = result.get('eye_tracking', {}).get('confidence', 0.5)
                scores['ppg_detection'] = result.get('ppg_analysis', {}).get('confidence', 0.5)
                
                # Extract detailed information
                detailed['advanced_antispoofing'] = result
                
                # Methods used
                methods.extend(['texture_analysis', 'depth_estimation', 'micro_expression', 
                              'eye_tracking', 'ppg_detection'])
                
                # Generate advanced recommendations
                if scores['texture_analysis'] < 0.5:
                    recommendations.append("Texture analysis indicates potential print/screen attack")
                if scores['depth_estimation'] < 0.5:
                    recommendations.append("Depth analysis suggests flat/2D presentation")
                if scores['micro_expression'] < 0.4:
                    recommendations.append("No natural micro-expressions detected")
                if scores['eye_tracking'] < 0.4:
                    recommendations.append("Unnatural eye movement patterns detected")
                if scores['ppg_detection'] < 0.3:
                    recommendations.append("No vital signs detected - potential non-living presentation")
            
            return {
                'scores': scores,
                'detailed': detailed,
                'methods': methods,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in advanced detection: {e}")
            return {
                'scores': {
                    'texture_analysis': 0.0,
                    'depth_estimation': 0.0,
                    'micro_expression': 0.0,
                    'eye_tracking': 0.0,
                    'ppg_detection': 0.0
                },
                'detailed': {'advanced_error': str(e)},
                'methods': ['advanced_error'],
                'recommendations': ['Advanced detection failed']
            }
    
    def _perform_fusion(self, scores: Dict[str, float], detailed_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Perform score fusion and make final decision"""
        try:
            weights = self.config['fusion_weights']
            
            # Calculate weighted fusion score
            weighted_score = 0.0
            total_weight = 0.0
            
            for method, weight in weights.items():
                if method in scores:
                    weighted_score += scores[method] * weight
                    total_weight += weight
            
            if total_weight > 0:
                fusion_confidence = weighted_score / total_weight
            else:
                fusion_confidence = 0.5
            
            # Determine liveness decision
            thresholds = self.config['confidence_thresholds']
            is_live = fusion_confidence >= thresholds['low_confidence']
            
            # Determine spoofing type based on lowest scores
            spoofing_type = self._determine_spoofing_type(scores, detailed_scores)
            
            # Final confidence (may be adjusted based on consistency)
            final_confidence = self._adjust_confidence_based_on_consistency(fusion_confidence, scores)
            
            return {
                'is_live': is_live,
                'confidence': final_confidence,
                'fusion_confidence': fusion_confidence,
                'spoofing_type': spoofing_type
            }
            
        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'fusion_confidence': 0.0,
                'spoofing_type': 'error'
            }
    
    def _determine_spoofing_type(self, scores: Dict[str, float], 
                                detailed_scores: Dict[str, Any]) -> str:
        """Determine the type of spoofing attack detected"""
        try:
            # Find the method with lowest confidence
            min_score = float('inf')
            suspicious_method = None
            
            for method, score in scores.items():
                if score < min_score:
                    min_score = score
                    suspicious_method = method
            
            # Map suspicious methods to spoofing types
            spoofing_mapping = {
                'texture_analysis': 'print_attack',
                'depth_estimation': 'flat_display',
                'micro_expression': 'static_image',
                'eye_tracking': 'artificial_gaze',
                'ppg_detection': 'non_living',
                'blink_detection': 'static_image',
                'basic_liveness': 'general_spoofing'
            }
            
            if suspicious_method and min_score < 0.5:
                return spoofing_mapping.get(suspicious_method, 'unknown_attack')
            
            # Check for specific patterns in detailed scores
            if 'advanced_antispoofing' in detailed_scores:
                advanced_result = detailed_scores['advanced_antispoofing']
                
                # Check for specific spoofing indicators
                if advanced_result.get('spoofing_type_probabilities'):
                    spoofing_probs = advanced_result['spoofing_type_probabilities']
                    max_prob_type = max(spoofing_probs.items(), key=lambda x: x[1])
                    if max_prob_type[1] > 0.6:
                        return max_prob_type[0]
            
            return 'potential_spoofing' if min_score < 0.7 else 'live'
            
        except Exception as e:
            logger.error(f"Error determining spoofing type: {e}")
            return 'unknown'
    
    def _assess_risk_level(self, fusion_result: Dict[str, Any], 
                          detailed_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level"""
        try:
            confidence = fusion_result['confidence']
            is_live = fusion_result['is_live']
            spoofing_type = fusion_result['spoofing_type']
            
            # Base risk assessment on confidence and decision
            if is_live and confidence >= 0.9:
                risk_level = 'low'
            elif is_live and confidence >= 0.7:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            # Adjust risk based on spoofing type severity
            high_risk_types = ['deepfake', 'sophisticated_mask', 'advanced_replay']
            if spoofing_type in high_risk_types:
                risk_level = 'high'
            
            # Calculate risk score
            risk_score = 1.0 - confidence if not is_live else max(0.1, 1.0 - confidence)
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': self._identify_risk_factors(detailed_scores)
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                'risk_level': 'high',
                'risk_score': 1.0,
                'risk_factors': ['assessment_error']
            }
    
    def _identify_risk_factors(self, detailed_scores: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        try:
            # Check image quality factors
            if 'face_quality_details' in detailed_scores:
                quality_details = detailed_scores['face_quality_details']
                
                if quality_details.get('brightness', 128) < 50:
                    risk_factors.append('low_lighting')
                elif quality_details.get('brightness', 128) > 200:
                    risk_factors.append('overexposure')
                
                if quality_details.get('sharpness', 1.0) < 0.5:
                    risk_factors.append('blurry_image')
            
            # Check for specific attack indicators
            if 'advanced_antispoofing' in detailed_scores:
                advanced = detailed_scores['advanced_antispoofing']
                
                if advanced.get('texture_analysis', {}).get('print_artifacts_detected', False):
                    risk_factors.append('print_artifacts')
                
                if advanced.get('depth_analysis', {}).get('flat_surface_detected', False):
                    risk_factors.append('flat_presentation')
                
                if advanced.get('micro_expression', {}).get('natural_expressions', 0) < 0.3:
                    risk_factors.append('unnatural_expressions')
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            risk_factors.append('analysis_error')
        
        return risk_factors
    
    def _adjust_confidence_based_on_consistency(self, base_confidence: float, 
                                               scores: Dict[str, float]) -> float:
        """Adjust confidence based on consistency across methods"""
        try:
            # Calculate standard deviation of scores
            score_values = list(scores.values())
            if len(score_values) > 1:
                std_dev = np.std(score_values)
                
                # If scores are very inconsistent, reduce confidence
                if std_dev > 0.3:
                    adjustment = -0.1 * std_dev
                    return max(0.0, base_confidence + adjustment)
            
            return base_confidence
            
        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
            return base_confidence
    
    def _calculate_basic_liveness_score(self, image: np.ndarray) -> float:
        """Calculate basic liveness score (placeholder implementation)"""
        try:
            # This would implement actual basic liveness detection
            # For now, return a score based on image properties
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Simple measures
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Normalize to 0-1 range
            brightness_norm = min(1.0, brightness / 128.0)
            contrast_norm = min(1.0, contrast / 64.0)
            
            # Combine factors
            score = (brightness_norm + contrast_norm) / 2.0
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating basic liveness score: {e}")
            return 0.0
    
    def _calculate_blink_score(self, image: np.ndarray, face_landmarks: np.ndarray = None) -> float:
        """Calculate blink detection score (placeholder implementation)"""
        try:
            # This would implement actual blink detection
            # For now, return random score for demonstration
            return np.random.random()  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating blink score: {e}")
            return 0.0
    
    def _assess_face_quality(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int] = None) -> float:
        """Assess face image quality"""
        try:
            # Extract face region if bbox provided
            if face_bbox:
                x, y, w, h = face_bbox
                face_img = image[y:y+h, x:x+w]
            else:
                face_img = image
            
            # Calculate quality metrics
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_norm = min(1.0, sharpness / 500.0)
            
            # Brightness
            brightness = np.mean(gray)
            brightness_norm = 1.0 - abs(brightness - 128) / 128.0
            
            # Contrast
            contrast = np.std(gray)
            contrast_norm = min(1.0, contrast / 64.0)
            
            # Combine metrics
            quality_score = (sharpness_norm + brightness_norm + contrast_norm) / 3.0
            return np.clip(quality_score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return 0.0
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess overall image quality"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate quality metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Normalize and combine
            quality = min(1.0, (sharpness / 500.0 + contrast / 64.0) / 2.0)
            return np.clip(quality, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return 0.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0
    
    def _update_performance_stats(self, result: EnhancedSecurityResult, processing_time: float):
        """Update performance statistics"""
        try:
            if self.config.get('enable_performance_tracking', True):
                self.performance_stats['total_detections'] += 1
                
                if result.confidence > 0.5:  # Consider it successful if confidence > 0.5
                    self.performance_stats['successful_detections'] += 1
                
                # Update rolling average of processing time
                total = self.performance_stats['total_detections']
                current_avg = self.performance_stats['average_processing_time']
                self.performance_stats['average_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )
                
                # Update accuracy estimate
                self.performance_stats['detection_accuracy'] = (
                    self.performance_stats['successful_detections'] / total
                )
                
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics report"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'system_status': {
                'advanced_available': self.advanced_processor is not None,
                'validator_available': self.validator is not None,
                'last_updated': time.time()
            },
            'configuration': self.config.copy()
        }
    
    def run_validation_test(self, test_images: List[np.ndarray], 
                           test_labels: List[bool]) -> Dict[str, Any]:
        """Run validation test using the integrated system"""
        if not self.validator:
            return {'error': 'Validator not available'}
        
        def detection_wrapper(image):
            """Wrapper function for validator"""
            result = self.process_comprehensive_detection(image)
            return {
                'is_live': result.is_live,
                'confidence': result.confidence
            }
        
        return self.validator.run_comprehensive_validation(
            detection_wrapper, test_images, test_labels
        )


# Example integration function
def create_enhanced_detection_system(config_path: str = None) -> IntegratedAntiSpoofingSystem:
    """
    Create an enhanced detection system with configuration
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        IntegratedAntiSpoofingSystem instance
    """
    config = None
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    return IntegratedAntiSpoofingSystem(config)


# Example usage
if __name__ == "__main__":
    # Create integrated system
    system = create_enhanced_detection_system()
    
    # Example detection
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = system.process_comprehensive_detection(test_image)
    
    print(f"Detection Result:")
    print(f"Is Live: {result.is_live}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Methods Used: {', '.join(result.detection_methods_used)}")
    
    if result.recommendations:
        print(f"Recommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
    
    # Get performance report
    performance = system.get_performance_report()
    print(f"\nSystem Performance:")
    print(f"Total Detections: {performance['performance_stats']['total_detections']}")
    print(f"Success Rate: {performance['performance_stats']['detection_accuracy']:.1%}")
    print(f"Avg Processing Time: {performance['performance_stats']['average_processing_time']:.3f}s")

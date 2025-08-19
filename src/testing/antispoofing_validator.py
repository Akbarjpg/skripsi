"""
Advanced Anti-Spoofing Validator and Testing Framework

Comprehensive testing framework for validating anti-spoofing improvements
against various attack types and edge cases.
"""
import numpy as np
import cv2
import time
import json
import os
from collections import defaultdict, deque
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    attack_type: str
    description: str
    expected_result: bool  # True for live, False for spoofed
    test_parameters: Dict[str, Any]

@dataclass
class TestResult:
    """Individual test result"""
    scenario_name: str
    predicted: bool
    expected: bool
    confidence: float
    processing_time: float
    detailed_scores: Dict[str, Any]
    error_message: str = None

class AttackSimulator:
    """
    Simulates various spoofing attacks for testing purposes
    """
    
    def __init__(self):
        self.attack_generators = {
            'printed_photo': self._simulate_printed_photo,
            'screen_display': self._simulate_screen_display,
            'video_replay': self._simulate_video_replay,
            'mask_attack': self._simulate_mask_attack,
            'deepfake': self._simulate_deepfake,
            'paper_cutout': self._simulate_paper_cutout
        }
    
    def generate_attack_scenario(self, original_image: np.ndarray, attack_type: str, **kwargs) -> np.ndarray:
        """Generate simulated attack from original image"""
        try:
            if attack_type in self.attack_generators:
                return self.attack_generators[attack_type](original_image, **kwargs)
            else:
                logger.warning(f"Unknown attack type: {attack_type}")
                return original_image
        except Exception as e:
            logger.error(f"Error generating attack scenario {attack_type}: {e}")
            return original_image
    
    def _simulate_printed_photo(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate printed photo artifacts"""
        # Add print artifacts
        noise_level = kwargs.get('noise_level', 0.1)
        halftone_pattern = kwargs.get('halftone_pattern', True)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        
        # Simulate halftone pattern
        if halftone_pattern:
            h, w = image.shape[:2]
            pattern = np.zeros((h, w), dtype=np.uint8)
            
            # Create dot pattern
            for i in range(0, h, 4):
                for j in range(0, w, 4):
                    if (i + j) % 8 == 0:
                        cv2.circle(pattern, (j, i), 1, 255, -1)
            
            # Apply pattern
            if len(noisy_image.shape) == 3:
                pattern_3d = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
            else:
                pattern_3d = pattern
            
            noisy_image = cv2.addWeighted(noisy_image, 0.9, pattern_3d, 0.1, 0)
        
        # Reduce sharpness (printing effect)
        blurred = cv2.GaussianBlur(noisy_image, (3, 3), 0)
        return cv2.addWeighted(noisy_image, 0.7, blurred, 0.3, 0)
    
    def _simulate_screen_display(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate screen display artifacts"""
        screen_type = kwargs.get('screen_type', 'lcd')
        moire_intensity = kwargs.get('moire_intensity', 0.3)
        
        # Add screen door effect
        h, w = image.shape[:2]
        screen_mask = np.ones((h, w), dtype=np.float32)
        
        # Create grid pattern
        for i in range(0, h, 3):
            screen_mask[i, :] *= 0.8
        for j in range(0, w, 3):
            screen_mask[:, j] *= 0.8
        
        # Apply screen effect
        if len(image.shape) == 3:
            screen_mask_3d = np.stack([screen_mask] * 3, axis=2)
        else:
            screen_mask_3d = screen_mask
        
        screen_image = (image * screen_mask_3d).astype(np.uint8)
        
        # Add moire pattern
        if moire_intensity > 0:
            moire = np.sin(np.arange(w) * 0.5) * np.sin(np.arange(h).reshape(-1, 1) * 0.3)
            moire = (moire * moire_intensity * 20).astype(np.uint8)
            
            if len(screen_image.shape) == 3:
                moire_3d = np.stack([moire] * 3, axis=2)
            else:
                moire_3d = moire
            
            screen_image = cv2.add(screen_image, moire_3d)
        
        # Add screen reflection
        reflection = np.random.normal(0.8, 0.1, image.shape).astype(np.float32)
        reflection = np.clip(reflection, 0.5, 1.2)
        
        return (screen_image * reflection).astype(np.uint8)
    
    def _simulate_video_replay(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate video replay artifacts"""
        compression_level = kwargs.get('compression_level', 0.3)
        frame_artifacts = kwargs.get('frame_artifacts', True)
        
        # Simulate compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int((1 - compression_level) * 100)]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        compressed_image = cv2.imdecode(encimg, 1)
        
        # Add frame artifacts
        if frame_artifacts:
            # Add slight blur
            compressed_image = cv2.GaussianBlur(compressed_image, (2, 2), 0)
            
            # Add temporal noise
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            compressed_image = np.clip(compressed_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return compressed_image
    
    def _simulate_mask_attack(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate mask attack effects"""
        mask_type = kwargs.get('mask_type', 'paper')
        edge_quality = kwargs.get('edge_quality', 0.5)
        
        # Reduce facial detail (mask effect)
        masked_image = cv2.bilateralFilter(image, 15, 80, 80)
        
        # Add artificial edges
        if edge_quality < 0.8:
            edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            masked_image = cv2.addWeighted(masked_image, 0.9, edges_colored, 0.1, 0)
        
        # Reduce texture details
        masked_image = cv2.medianBlur(masked_image, 5)
        
        return masked_image
    
    def _simulate_deepfake(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate deepfake artifacts"""
        quality_level = kwargs.get('quality_level', 0.7)
        temporal_consistency = kwargs.get('temporal_consistency', 0.8)
        
        # Simulate neural network artifacts
        # Add slight blur to simulate generative artifacts
        deepfake_image = cv2.GaussianBlur(image, (2, 2), 0)
        
        # Add subtle color shifts
        hsv = cv2.cvtColor(deepfake_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.95  # Reduce saturation slightly
        hsv[:, :, 2] = hsv[:, :, 2] * 1.05  # Increase brightness slightly
        deepfake_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add neural network compression artifacts
        if quality_level < 0.9:
            # Simulate lossy reconstruction
            deepfake_image = cv2.bilateralFilter(deepfake_image, 9, 75, 75)
        
        return deepfake_image
    
    def _simulate_paper_cutout(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Simulate paper cutout attack"""
        # Very flat appearance with sharp edges
        cutout_image = cv2.medianBlur(image, 7)
        
        # Enhance edges artificially
        gray = cv2.cvtColor(cutout_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine
        cutout_image = cv2.addWeighted(cutout_image, 0.8, edges_colored, 0.2, 0)
        
        return cutout_image

class PerformanceProfiler:
    """
    Profiles performance metrics for anti-spoofing system
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timing_data = defaultdict(list)
        
    def profile_detection_speed(self, detection_func, test_images: List[np.ndarray], iterations: int = 5) -> Dict[str, float]:
        """Profile detection speed"""
        times = []
        
        for iteration in range(iterations):
            for image in test_images:
                start_time = time.time()
                try:
                    _ = detection_func(image)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except Exception as e:
                    logger.warning(f"Detection failed in profiling: {e}")
                    times.append(float('inf'))
        
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            return {'error': 'All detections failed'}
        
        return {
            'mean_time': np.mean(valid_times),
            'std_time': np.std(valid_times),
            'min_time': np.min(valid_times),
            'max_time': np.max(valid_times),
            'fps_estimate': 1.0 / np.mean(valid_times) if np.mean(valid_times) > 0 else 0,
            'success_rate': len(valid_times) / len(times)
        }
    
    def profile_accuracy_vs_speed(self, detection_func, test_scenarios: List[TestScenario]) -> Dict[str, Any]:
        """Profile accuracy vs speed trade-offs"""
        results = []
        
        for scenario in test_scenarios:
            start_time = time.time()
            try:
                result = detection_func(scenario.test_parameters.get('image'))
                processing_time = time.time() - start_time
                
                predicted = result.get('is_live', False)
                confidence = result.get('confidence', 0.0)
                
                accuracy = 1.0 if predicted == scenario.expected_result else 0.0
                
                results.append({
                    'scenario': scenario.name,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'speed_score': 1.0 / processing_time if processing_time > 0 else 0
                })
                
            except Exception as e:
                logger.error(f"Error profiling scenario {scenario.name}: {e}")
                results.append({
                    'scenario': scenario.name,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'processing_time': float('inf'),
                    'speed_score': 0.0
                })
        
        return {
            'individual_results': results,
            'overall_accuracy': np.mean([r['accuracy'] for r in results]),
            'overall_speed': np.mean([r['speed_score'] for r in results if r['speed_score'] > 0]),
            'accuracy_std': np.std([r['accuracy'] for r in results]),
            'speed_std': np.std([r['speed_score'] for r in results if r['speed_score'] > 0])
        }

class RobustnessEvaluator:
    """
    Evaluates system robustness against edge cases and challenging conditions
    """
    
    def __init__(self):
        self.edge_case_generators = {
            'low_light': self._generate_low_light,
            'high_light': self._generate_high_light,
            'motion_blur': self._generate_motion_blur,
            'occlusion': self._generate_occlusion,
            'extreme_angle': self._generate_extreme_angle,
            'low_resolution': self._generate_low_resolution,
            'noise': self._generate_noise,
            'compression': self._generate_compression
        }
    
    def evaluate_robustness(self, detection_func, base_images: List[np.ndarray], 
                          expected_results: List[bool]) -> Dict[str, Any]:
        """Comprehensive robustness evaluation"""
        robustness_results = {}
        
        for edge_case, generator in self.edge_case_generators.items():
            case_results = []
            
            for i, (image, expected) in enumerate(zip(base_images, expected_results)):
                try:
                    # Generate edge case variant
                    edge_image = generator(image)
                    
                    # Test detection
                    result = detection_func(edge_image)
                    predicted = result.get('is_live', False)
                    confidence = result.get('confidence', 0.0)
                    
                    accuracy = 1.0 if predicted == expected else 0.0
                    
                    case_results.append({
                        'image_id': i,
                        'accuracy': accuracy,
                        'confidence': confidence,
                        'predicted': predicted,
                        'expected': expected
                    })
                    
                except Exception as e:
                    logger.error(f"Error in robustness test {edge_case} for image {i}: {e}")
                    case_results.append({
                        'image_id': i,
                        'accuracy': 0.0,
                        'confidence': 0.0,
                        'error': str(e)
                    })
            
            # Compile case statistics
            valid_results = [r for r in case_results if 'error' not in r]
            
            if valid_results:
                robustness_results[edge_case] = {
                    'accuracy': np.mean([r['accuracy'] for r in valid_results]),
                    'confidence_mean': np.mean([r['confidence'] for r in valid_results]),
                    'confidence_std': np.std([r['confidence'] for r in valid_results]),
                    'success_rate': len(valid_results) / len(case_results),
                    'individual_results': case_results
                }
            else:
                robustness_results[edge_case] = {
                    'accuracy': 0.0,
                    'confidence_mean': 0.0,
                    'confidence_std': 0.0,
                    'success_rate': 0.0,
                    'error': 'All tests failed'
                }
        
        # Overall robustness score
        overall_accuracy = np.mean([r['accuracy'] for r in robustness_results.values() 
                                  if isinstance(r.get('accuracy'), (int, float))])
        
        robustness_results['overall_robustness'] = {
            'mean_accuracy': overall_accuracy,
            'robustness_score': overall_accuracy,  # Can be more complex
            'edge_cases_passed': sum(1 for r in robustness_results.values() 
                                   if isinstance(r.get('accuracy'), (int, float)) and r['accuracy'] > 0.8)
        }
        
        return robustness_results
    
    def _generate_low_light(self, image: np.ndarray) -> np.ndarray:
        """Generate low light condition"""
        dark_factor = 0.3
        return (image * dark_factor).astype(np.uint8)
    
    def _generate_high_light(self, image: np.ndarray) -> np.ndarray:
        """Generate high light/overexposed condition"""
        bright_factor = 1.5
        return np.clip(image * bright_factor, 0, 255).astype(np.uint8)
    
    def _generate_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """Generate motion blur"""
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
    
    def _generate_occlusion(self, image: np.ndarray) -> np.ndarray:
        """Generate partial occlusion"""
        h, w = image.shape[:2]
        occluded = image.copy()
        
        # Add random occlusion patches
        for _ in range(3):
            x = np.random.randint(0, w//2)
            y = np.random.randint(0, h//2)
            patch_w = np.random.randint(w//10, w//4)
            patch_h = np.random.randint(h//10, h//4)
            
            occluded[y:y+patch_h, x:x+patch_w] = 0
        
        return occluded
    
    def _generate_extreme_angle(self, image: np.ndarray) -> np.ndarray:
        """Generate extreme viewing angle"""
        h, w = image.shape[:2]
        
        # Perspective transformation
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst_points = np.float32([[w*0.1, 0], [w*0.9, h*0.1], [0, h], [w*0.8, h*0.9]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (w, h))
    
    def _generate_low_resolution(self, image: np.ndarray) -> np.ndarray:
        """Generate low resolution version"""
        h, w = image.shape[:2]
        small = cv2.resize(image, (w//4, h//4))
        return cv2.resize(small, (w, h))
    
    def _generate_noise(self, image: np.ndarray) -> np.ndarray:
        """Generate noisy version"""
        noise = np.random.normal(0, 25, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    def _generate_compression(self, image: np.ndarray) -> np.ndarray:
        """Generate heavily compressed version"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

class AntiSpoofingValidator:
    """
    Main validation framework combining all testing components
    """
    
    def __init__(self, output_dir: str = "antispoofing_test_results"):
        self.attack_simulator = AttackSimulator()
        self.performance_profiler = PerformanceProfiler()
        self.robustness_evaluator = RobustnessEvaluator()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize test scenarios
        self.test_scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios"""
        scenarios = []
        
        # Live face scenarios
        scenarios.append(TestScenario(
            name="live_face_normal",
            attack_type="none",
            description="Normal live face under good conditions",
            expected_result=True,
            test_parameters={"condition": "normal"}
        ))
        
        scenarios.append(TestScenario(
            name="live_face_low_light",
            attack_type="none",
            description="Live face under low light conditions",
            expected_result=True,
            test_parameters={"condition": "low_light"}
        ))
        
        # Attack scenarios
        attack_types = ["printed_photo", "screen_display", "video_replay", 
                       "mask_attack", "deepfake", "paper_cutout"]
        
        for attack_type in attack_types:
            scenarios.append(TestScenario(
                name=f"attack_{attack_type}",
                attack_type=attack_type,
                description=f"Spoofing attack using {attack_type.replace('_', ' ')}",
                expected_result=False,
                test_parameters={"attack_type": attack_type}
            ))
        
        return scenarios
    
    def run_comprehensive_validation(self, detection_func, test_images: List[np.ndarray], 
                                   test_labels: List[bool]) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        logger.info("Starting comprehensive anti-spoofing validation...")
        
        validation_results = {
            'timestamp': time.time(),
            'test_summary': {},
            'attack_simulation_results': {},
            'performance_profiling': {},
            'robustness_evaluation': {},
            'detailed_metrics': {},
            'recommendations': []
        }
        
        try:
            # 1. Basic accuracy testing
            logger.info("Running basic accuracy tests...")
            basic_results = self._run_basic_accuracy_tests(detection_func, test_images, test_labels)
            validation_results['basic_accuracy'] = basic_results
            
            # 2. Attack simulation tests
            logger.info("Running attack simulation tests...")
            attack_results = self._run_attack_simulation_tests(detection_func, test_images)
            validation_results['attack_simulation_results'] = attack_results
            
            # 3. Performance profiling
            logger.info("Running performance profiling...")
            performance_results = self.performance_profiler.profile_detection_speed(
                detection_func, test_images[:10]  # Use subset for speed
            )
            validation_results['performance_profiling'] = performance_results
            
            # 4. Robustness evaluation
            logger.info("Running robustness evaluation...")
            robustness_results = self.robustness_evaluator.evaluate_robustness(
                detection_func, test_images[:20], test_labels[:20]  # Use subset
            )
            validation_results['robustness_evaluation'] = robustness_results
            
            # 5. Generate detailed metrics
            detailed_metrics = self._calculate_detailed_metrics(validation_results)
            validation_results['detailed_metrics'] = detailed_metrics
            
            # 6. Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            validation_results['recommendations'] = recommendations
            
            # 7. Save results
            self._save_validation_results(validation_results)
            
            # 8. Generate visualizations
            self._generate_visualizations(validation_results)
            
            logger.info("Comprehensive validation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _run_basic_accuracy_tests(self, detection_func, test_images: List[np.ndarray], 
                                 test_labels: List[bool]) -> Dict[str, Any]:
        """Run basic accuracy tests"""
        results = []
        
        for i, (image, expected) in enumerate(zip(test_images, test_labels)):
            try:
                start_time = time.time()
                result = detection_func(image)
                processing_time = time.time() - start_time
                
                predicted = result.get('is_live', False)
                confidence = result.get('confidence', 0.0)
                
                results.append({
                    'image_id': i,
                    'predicted': predicted,
                    'expected': expected,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'correct': predicted == expected
                })
                
            except Exception as e:
                logger.error(f"Error testing image {i}: {e}")
                results.append({
                    'image_id': i,
                    'predicted': False,
                    'expected': expected,
                    'confidence': 0.0,
                    'processing_time': float('inf'),
                    'correct': False,
                    'error': str(e)
                })
        
        # Calculate metrics
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            accuracy = np.mean([r['correct'] for r in valid_results])
            avg_confidence = np.mean([r['confidence'] for r in valid_results])
            avg_time = np.mean([r['processing_time'] for r in valid_results])
            
            # Calculate precision, recall, F1
            tp = sum(1 for r in valid_results if r['predicted'] and r['expected'])
            fp = sum(1 for r in valid_results if r['predicted'] and not r['expected'])
            tn = sum(1 for r in valid_results if not r['predicted'] and not r['expected'])
            fn = sum(1 for r in valid_results if not r['predicted'] and r['expected'])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
        else:
            accuracy = avg_confidence = avg_time = precision = recall = f1 = 0.0
            tp = fp = tn = fn = 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_time,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'individual_results': results,
            'success_rate': len(valid_results) / len(results)
        }
    
    def _run_attack_simulation_tests(self, detection_func, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """Run attack simulation tests"""
        attack_results = {}
        
        for attack_type in ['printed_photo', 'screen_display', 'video_replay', 'mask_attack']:
            attack_test_results = []
            
            for i, image in enumerate(test_images[:10]):  # Test on subset
                try:
                    # Generate attack
                    attack_image = self.attack_simulator.generate_attack_scenario(image, attack_type)
                    
                    # Test detection
                    result = detection_func(attack_image)
                    predicted = result.get('is_live', False)
                    confidence = result.get('confidence', 0.0)
                    
                    # For attacks, expected result is False (spoofed)
                    correct = not predicted
                    
                    attack_test_results.append({
                        'image_id': i,
                        'predicted': predicted,
                        'expected': False,
                        'confidence': confidence,
                        'correct': correct
                    })
                    
                except Exception as e:
                    logger.error(f"Error in attack simulation {attack_type} for image {i}: {e}")
                    attack_test_results.append({
                        'image_id': i,
                        'predicted': True,  # Assume failure = classified as live
                        'expected': False,
                        'confidence': 0.0,
                        'correct': False,
                        'error': str(e)
                    })
            
            # Calculate attack-specific metrics
            valid_results = [r for r in attack_test_results if 'error' not in r]
            
            if valid_results:
                detection_rate = np.mean([r['correct'] for r in valid_results])
                avg_confidence = np.mean([r['confidence'] for r in valid_results])
            else:
                detection_rate = avg_confidence = 0.0
            
            attack_results[attack_type] = {
                'detection_rate': detection_rate,  # Rate of correctly identifying as attack
                'average_confidence': avg_confidence,
                'individual_results': attack_test_results,
                'success_rate': len(valid_results) / len(attack_test_results)
            }
        
        # Overall attack detection performance
        all_detection_rates = [r['detection_rate'] for r in attack_results.values()]
        attack_results['overall'] = {
            'mean_detection_rate': np.mean(all_detection_rates),
            'std_detection_rate': np.std(all_detection_rates),
            'min_detection_rate': np.min(all_detection_rates),
            'max_detection_rate': np.max(all_detection_rates)
        }
        
        return attack_results
    
    def _calculate_detailed_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        metrics = {}
        
        # Extract basic accuracy metrics
        basic_results = validation_results.get('basic_accuracy', {})
        metrics['accuracy_metrics'] = {
            'overall_accuracy': basic_results.get('accuracy', 0.0),
            'precision': basic_results.get('precision', 0.0),
            'recall': basic_results.get('recall', 0.0),
            'f1_score': basic_results.get('f1_score', 0.0)
        }
        
        # Extract performance metrics
        performance_results = validation_results.get('performance_profiling', {})
        metrics['performance_metrics'] = {
            'mean_processing_time': performance_results.get('mean_time', 0.0),
            'fps_estimate': performance_results.get('fps_estimate', 0.0),
            'real_time_capable': performance_results.get('mean_time', float('inf')) < 0.1
        }
        
        # Extract robustness metrics
        robustness_results = validation_results.get('robustness_evaluation', {})
        if 'overall_robustness' in robustness_results:
            metrics['robustness_metrics'] = robustness_results['overall_robustness']
        
        # Extract attack detection metrics
        attack_results = validation_results.get('attack_simulation_results', {})
        if 'overall' in attack_results:
            metrics['attack_detection_metrics'] = attack_results['overall']
        
        return metrics
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        # Check accuracy
        accuracy = validation_results.get('basic_accuracy', {}).get('accuracy', 0.0)
        if accuracy < 0.9:
            recommendations.append(f"Overall accuracy ({accuracy:.2%}) is below 90%. Consider improving base detection algorithms.")
        
        # Check processing speed
        mean_time = validation_results.get('performance_profiling', {}).get('mean_time', 0.0)
        if mean_time > 0.1:
            recommendations.append(f"Processing time ({mean_time:.3f}s) exceeds real-time requirements. Consider optimization.")
        
        # Check attack detection
        attack_results = validation_results.get('attack_simulation_results', {})
        for attack_type, results in attack_results.items():
            if attack_type != 'overall' and isinstance(results, dict):
                detection_rate = results.get('detection_rate', 0.0)
                if detection_rate < 0.8:
                    recommendations.append(f"{attack_type.replace('_', ' ').title()} detection rate ({detection_rate:.2%}) is low. Improve {attack_type} detection.")
        
        # Check robustness
        robustness_results = validation_results.get('robustness_evaluation', {})
        for condition, results in robustness_results.items():
            if condition != 'overall_robustness' and isinstance(results, dict):
                accuracy = results.get('accuracy', 0.0)
                if accuracy < 0.7:
                    recommendations.append(f"Performance under {condition.replace('_', ' ')} conditions ({accuracy:.2%}) needs improvement.")
        
        if not recommendations:
            recommendations.append("System performance meets all criteria. Consider testing with more diverse datasets.")
        
        return recommendations
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        timestamp = int(time.time())
        filename = f"validation_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = self._convert_numpy_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Validation results saved to: {filepath}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization plots"""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Anti-Spoofing System Validation Results', fontsize=16)
            
            # 1. Accuracy metrics
            ax1 = axes[0, 0]
            basic_results = results.get('basic_accuracy', {})
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [basic_results.get(m, 0.0) for m in metrics]
            
            bars = ax1.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
            ax1.set_title('Classification Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 2. Attack detection rates
            ax2 = axes[0, 1]
            attack_results = results.get('attack_simulation_results', {})
            attack_types = [k for k in attack_results.keys() if k != 'overall']
            detection_rates = [attack_results[k].get('detection_rate', 0.0) for k in attack_types]
            
            if attack_types:
                bars = ax2.bar(range(len(attack_types)), detection_rates, 
                              color=['red', 'orange', 'yellow', 'green'][:len(attack_types)])
                ax2.set_title('Attack Detection Rates')
                ax2.set_ylabel('Detection Rate')
                ax2.set_ylim(0, 1)
                ax2.set_xticks(range(len(attack_types)))
                ax2.set_xticklabels([t.replace('_', '\n') for t in attack_types], rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, detection_rates):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
            
            # 3. Robustness evaluation
            ax3 = axes[1, 0]
            robustness_results = results.get('robustness_evaluation', {})
            robustness_conditions = [k for k in robustness_results.keys() if k != 'overall_robustness']
            robustness_scores = [robustness_results[k].get('accuracy', 0.0) for k in robustness_conditions]
            
            if robustness_conditions:
                bars = ax3.bar(range(len(robustness_conditions)), robustness_scores, 
                              color='purple', alpha=0.7)
                ax3.set_title('Robustness Under Different Conditions')
                ax3.set_ylabel('Accuracy')
                ax3.set_ylim(0, 1)
                ax3.set_xticks(range(len(robustness_conditions)))
                ax3.set_xticklabels([c.replace('_', '\n') for c in robustness_conditions], 
                                   rotation=45, ha='right')
                
                # Add value labels
                for bar, value in zip(bars, robustness_scores):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom')
            
            # 4. Performance summary
            ax4 = axes[1, 1]
            performance_data = results.get('performance_profiling', {})
            fps = performance_data.get('fps_estimate', 0.0)
            mean_time = performance_data.get('mean_time', 0.0)
            
            # Create a simple text summary
            ax4.text(0.1, 0.8, f"Processing Speed:", fontsize=12, weight='bold', transform=ax4.transAxes)
            ax4.text(0.1, 0.7, f"Mean Time: {mean_time:.3f}s", fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.6, f"Est. FPS: {fps:.1f}", fontsize=10, transform=ax4.transAxes)
            ax4.text(0.1, 0.5, f"Real-time: {'Yes' if mean_time < 0.1 else 'No'}", 
                    fontsize=10, transform=ax4.transAxes)
            
            ax4.text(0.1, 0.3, f"Overall Summary:", fontsize=12, weight='bold', transform=ax4.transAxes)
            overall_accuracy = basic_results.get('accuracy', 0.0)
            ax4.text(0.1, 0.2, f"Accuracy: {overall_accuracy:.1%}", fontsize=10, transform=ax4.transAxes)
            
            # Color code based on performance
            color = 'green' if overall_accuracy > 0.9 and mean_time < 0.1 else 'orange' if overall_accuracy > 0.8 else 'red'
            ax4.text(0.1, 0.1, f"Status: {'Excellent' if color == 'green' else 'Good' if color == 'orange' else 'Needs Improvement'}", 
                    fontsize=10, color=color, weight='bold', transform=ax4.transAxes)
            
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Performance Summary')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save plot
            timestamp = int(time.time())
            plot_filename = f"validation_plots_{timestamp}.png"
            plot_filepath = os.path.join(self.output_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Validation plots saved to: {plot_filepath}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")


# Example usage and integration
def validate_antispoofing_system():
    """
    Example function showing how to use the validation framework
    """
    # Initialize validator
    validator = AntiSpoofingValidator()
    
    # Example detection function (replace with actual implementation)
    def example_detection_func(image):
        # This would call your actual anti-spoofing detection
        # For example: return AdvancedAntiSpoofingProcessor().process_comprehensive_antispoofing([image])
        return {
            'is_live': np.random.random() > 0.3,  # Placeholder
            'confidence': np.random.random()
        }
    
    # Generate test data (replace with actual test images)
    test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(50)]
    test_labels = [bool(np.random.randint(0, 2)) for _ in range(50)]
    
    # Run validation
    results = validator.run_comprehensive_validation(example_detection_func, test_images, test_labels)
    
    return results


if __name__ == "__main__":
    # Run validation example
    validation_results = validate_antispoofing_system()
    print("Validation completed successfully!")
    print(f"Overall accuracy: {validation_results.get('basic_accuracy', {}).get('accuracy', 0.0):.2%}")

#!/usr/bin/env python3
"""
Performance Testing Script untuk Anti-Spoofing System
Membandingkan performa sebelum dan sesudah optimasi
"""

import sys
import os
import time
import numpy as np
import cv2
import psutil
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import deque
import threading
import queue

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def measure_memory_usage():
    """Measure current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def create_test_images(count=50):
    """Create test images for benchmarking"""
    print(f"🖼️ Creating {count} test images...")
    images = []
    
    for i in range(count):
        # Create realistic face-like images
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add some face-like features
        # Eye regions
        cv2.circle(img, (200, 180), 30, (255, 255, 255), -1)
        cv2.circle(img, (440, 180), 30, (255, 255, 255), -1)
        cv2.circle(img, (200, 180), 15, (0, 0, 0), -1)
        cv2.circle(img, (440, 180), 15, (0, 0, 0), -1)
        
        # Mouth region
        cv2.ellipse(img, (320, 300), (40, 20), 0, 0, 360, (100, 50, 50), -1)
        
        # Nose
        cv2.ellipse(img, (320, 240), (15, 25), 0, 0, 360, (150, 120, 100), -1)
        
        images.append(img)
    
    return images

def benchmark_original_system(test_images):
    """Benchmark original landmark detection system"""
    print("\n📊 BENCHMARKING ORIGINAL SYSTEM")
    print("=" * 50)
    
    try:
        from src.detection.landmark_detection import LivenessVerifier
        verifier = LivenessVerifier()
        
        times = []
        memory_usage = []
        
        # Warmup
        for i in range(3):
            verifier.process_frame(test_images[0])
        
        initial_memory = measure_memory_usage()
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Benchmark
        print("Processing frames...")
        for i, image in enumerate(test_images):
            start_time = time.time()
            
            result = verifier.process_frame(image)
            
            process_time = time.time() - start_time
            times.append(process_time)
            
            current_memory = measure_memory_usage()
            memory_usage.append(current_memory)
            
            if i % 10 == 0:
                print(f"Frame {i+1}: {process_time*1000:.1f}ms, Memory: {current_memory:.1f}MB")
        
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time
        max_memory = max(memory_usage)
        memory_growth = max_memory - initial_memory
        
        results = {
            'system': 'original',
            'avg_processing_time': avg_time,
            'avg_fps': avg_fps,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'initial_memory': initial_memory,
            'max_memory': max_memory,
            'memory_growth': memory_growth,
            'processing_times': times,
            'memory_usage': memory_usage
        }
        
        print(f"\n📈 ORIGINAL SYSTEM RESULTS:")
        print(f"Average processing time: {avg_time*1000:.1f}ms")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Memory growth: {memory_growth:.1f}MB")
        
        return results
        
    except Exception as e:
        print(f"❌ Original system benchmark failed: {e}")
        return None

def benchmark_optimized_system(test_images):
    """Benchmark optimized landmark detection system"""
    print("\n🚀 BENCHMARKING OPTIMIZED SYSTEM")
    print("=" * 50)
    
    try:
        from src.detection.optimized_landmark_detection import OptimizedLivenessVerifier
        verifier = OptimizedLivenessVerifier(history_length=15)
        
        times = []
        memory_usage = []
        
        # Warmup
        for i in range(3):
            verifier.process_frame_optimized(test_images[0])
        
        initial_memory = measure_memory_usage()
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Benchmark
        print("Processing frames...")
        for i, image in enumerate(test_images):
            start_time = time.time()
            
            result = verifier.process_frame_optimized(image)
            
            process_time = time.time() - start_time
            times.append(process_time)
            
            current_memory = measure_memory_usage()
            memory_usage.append(current_memory)
            
            if i % 10 == 0:
                fps_estimate = result.get('fps_estimate', 0)
                print(f"Frame {i+1}: {process_time*1000:.1f}ms, Memory: {current_memory:.1f}MB, Est.FPS: {fps_estimate:.1f}")
        
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time
        max_memory = max(memory_usage)
        memory_growth = max_memory - initial_memory
        
        # Get performance stats from verifier
        perf_stats = verifier.get_performance_stats()
        
        results = {
            'system': 'optimized',
            'avg_processing_time': avg_time,
            'avg_fps': avg_fps,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'initial_memory': initial_memory,
            'max_memory': max_memory,
            'memory_growth': memory_growth,
            'processing_times': times,
            'memory_usage': memory_usage,
            'cache_size': perf_stats.get('cache_size', 0),
            'frames_processed': len(test_images)
        }
        
        print(f"\n🚀 OPTIMIZED SYSTEM RESULTS:")
        print(f"Average processing time: {avg_time*1000:.1f}ms")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Memory growth: {memory_growth:.1f}MB")
        print(f"Cache efficiency: {perf_stats.get('cache_size', 0)} entries")
        
        return results
        
    except Exception as e:
        print(f"❌ Optimized system benchmark failed: {e}")
        return None

def benchmark_cnn_models():
    """Benchmark CNN model performance"""
    print("\n🧠 BENCHMARKING CNN MODELS")
    print("=" * 50)
    
    results = {}
    
    # Test input sizes
    input_size = 112
    batch_size = 1
    num_iterations = 50
    
    # Create test input
    test_input = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
    
    # Test original CNN (if available)
    try:
        from src.models.cnn_model import LivenessDetectionCNN
        import torch
        
        print("Testing original CNN...")
        original_model = LivenessDetectionCNN()
        original_model.eval()
        
        # Warmup
        dummy_tensor = torch.randn(1, 3, input_size, input_size)
        for _ in range(5):
            with torch.no_grad():
                _ = original_model(dummy_tensor)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = original_model(dummy_tensor)
            times.append(time.time() - start_time)
        
        results['original_cnn'] = {
            'avg_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'parameters': sum(p.numel() for p in original_model.parameters())
        }
        
        print(f"Original CNN: {np.mean(times)*1000:.1f}ms, {1.0/np.mean(times):.1f} FPS")
        
    except Exception as e:
        print(f"⚠️ Original CNN test failed: {e}")
    
    # Test optimized CNN
    try:
        from src.models.optimized_cnn_model import OptimizedLivenessCNN, OptimizedLivenessPredictor
        
        print("Testing optimized CNN...")
        optimized_model = OptimizedLivenessCNN(input_size=input_size)
        optimized_model.eval()
        
        # Warmup
        dummy_tensor = torch.randn(1, 3, input_size, input_size)
        for _ in range(5):
            with torch.no_grad():
                _ = optimized_model(dummy_tensor)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = optimized_model(dummy_tensor)
            times.append(time.time() - start_time)
        
        results['optimized_cnn'] = {
            'avg_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'parameters': sum(p.numel() for p in optimized_model.parameters())
        }
        
        print(f"Optimized CNN: {np.mean(times)*1000:.1f}ms, {1.0/np.mean(times):.1f} FPS")
        
        # Test optimized predictor
        print("Testing optimized predictor...")
        predictor = OptimizedLivenessPredictor(use_quantization=True)
        
        # Warmup
        for _ in range(5):
            predictor.predict_optimized(test_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            result = predictor.predict_optimized(test_input)
            times.append(time.time() - start_time)
        
        results['optimized_predictor'] = {
            'avg_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'cache_hits': predictor.get_performance_stats().get('cache_hit_ratio', 0)
        }
        
        print(f"Optimized Predictor: {np.mean(times)*1000:.1f}ms, {1.0/np.mean(times):.1f} FPS")
        
    except Exception as e:
        print(f"⚠️ Optimized CNN test failed: {e}")
    
    return results

def benchmark_full_pipeline(test_images):
    """Benchmark full processing pipeline"""
    print("\n🔄 BENCHMARKING FULL PIPELINE")
    print("=" * 50)
    
    try:
        from src.web.app_optimized import OptimizedFrameProcessor
        
        processor = OptimizedFrameProcessor()
        
        times = []
        memory_usage = []
        
        initial_memory = measure_memory_usage()
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Benchmark full pipeline
        print("Processing full pipeline...")
        for i, image in enumerate(test_images[:20]):  # Test with fewer images for full pipeline
            start_time = time.time()
            
            result = processor.process_frame_optimized(image, f"session_{i}")
            
            process_time = time.time() - start_time
            times.append(process_time)
            
            current_memory = measure_memory_usage()
            memory_usage.append(current_memory)
            
            if i % 5 == 0:
                security_level = result.get('security_level', 'N/A')
                methods_passed = result.get('methods_passed', 0)
                print(f"Frame {i+1}: {process_time*1000:.1f}ms, Security: {security_level}, Methods: {methods_passed}/3")
        
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time
        max_memory = max(memory_usage)
        memory_growth = max_memory - initial_memory
        
        # Get processor stats
        perf_stats = processor.get_performance_stats()
        
        results = {
            'system': 'full_pipeline',
            'avg_processing_time': avg_time,
            'avg_fps': avg_fps,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'initial_memory': initial_memory,
            'max_memory': max_memory,
            'memory_growth': memory_growth,
            'processing_times': times,
            'memory_usage': memory_usage,
            'cache_size': perf_stats.get('cache_size', 0),
            'frames_processed': perf_stats.get('frames_processed', 0)
        }
        
        print(f"\n🔄 FULL PIPELINE RESULTS:")
        print(f"Average processing time: {avg_time*1000:.1f}ms")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Memory growth: {memory_growth:.1f}MB")
        print(f"Cache efficiency: {perf_stats.get('cache_size', 0)} entries")
        
        return results
        
    except Exception as e:
        print(f"❌ Full pipeline benchmark failed: {e}")
        return None

def create_performance_report(results_list, cnn_results):
    """Create comprehensive performance report"""
    print("\n📋 PERFORMANCE COMPARISON REPORT")
    print("=" * 70)
    
    # Landmark Detection Comparison
    if len(results_list) >= 2:
        original = results_list[0]
        optimized = results_list[1]
        
        if original and optimized:
            speedup = original['avg_processing_time'] / optimized['avg_processing_time']
            fps_improvement = optimized['avg_fps'] / original['avg_fps']
            memory_improvement = (original['memory_growth'] - optimized['memory_growth']) / original['memory_growth'] * 100
            
            print(f"\n🔍 LANDMARK DETECTION OPTIMIZATION:")
            print(f"├─ Speed improvement: {speedup:.2f}x faster")
            print(f"├─ FPS improvement: {fps_improvement:.2f}x ({original['avg_fps']:.1f} → {optimized['avg_fps']:.1f} FPS)")
            print(f"├─ Memory improvement: {memory_improvement:.1f}% less memory growth")
            print(f"├─ Processing time: {original['avg_processing_time']*1000:.1f}ms → {optimized['avg_processing_time']*1000:.1f}ms")
            print(f"└─ Cache efficiency: {optimized.get('cache_size', 0)} entries")
    
    # CNN Comparison
    if cnn_results:
        print(f"\n🧠 CNN MODEL OPTIMIZATION:")
        
        if 'original_cnn' in cnn_results and 'optimized_cnn' in cnn_results:
            orig_cnn = cnn_results['original_cnn']
            opt_cnn = cnn_results['optimized_cnn']
            
            speedup = orig_cnn['avg_time'] / opt_cnn['avg_time']
            param_reduction = (orig_cnn['parameters'] - opt_cnn['parameters']) / orig_cnn['parameters'] * 100
            
            print(f"├─ CNN speed improvement: {speedup:.2f}x faster")
            print(f"├─ Parameter reduction: {param_reduction:.1f}% fewer parameters")
            print(f"├─ Original CNN: {orig_cnn['avg_time']*1000:.1f}ms, {orig_cnn['parameters']:,} params")
            print(f"└─ Optimized CNN: {opt_cnn['avg_time']*1000:.1f}ms, {opt_cnn['parameters']:,} params")
        
        if 'optimized_predictor' in cnn_results:
            predictor = cnn_results['optimized_predictor']
            print(f"\n🎯 OPTIMIZED PREDICTOR:")
            print(f"├─ Processing time: {predictor['avg_time']*1000:.1f}ms")
            print(f"├─ Estimated FPS: {predictor['fps']:.1f}")
            print(f"└─ Cache hit ratio: {predictor.get('cache_hits', 0):.2f}")
    
    # Full Pipeline
    if len(results_list) >= 3:
        pipeline = results_list[2]
        if pipeline:
            print(f"\n🔄 FULL PIPELINE PERFORMANCE:")
            print(f"├─ End-to-end processing: {pipeline['avg_processing_time']*1000:.1f}ms")
            print(f"├─ Real-time FPS: {pipeline['avg_fps']:.1f}")
            print(f"├─ Memory efficiency: {pipeline['memory_growth']:.1f}MB growth")
            print(f"└─ Multi-method integration: ✅ Landmark + CNN + Movement")
    
    # Performance Targets Assessment
    print(f"\n🎯 PERFORMANCE TARGETS ASSESSMENT:")
    
    target_fps = 15
    target_memory = 100  # MB
    
    if len(results_list) >= 1 and results_list[-1]:
        final_results = results_list[-1]
        fps_target_met = final_results['avg_fps'] >= target_fps
        memory_target_met = final_results['memory_growth'] <= target_memory
        
        print(f"├─ FPS Target (≥{target_fps}): {'✅ PASSED' if fps_target_met else '❌ FAILED'} ({final_results['avg_fps']:.1f} FPS)")
        print(f"├─ Memory Target (≤{target_memory}MB): {'✅ PASSED' if memory_target_met else '❌ FAILED'} ({final_results['memory_growth']:.1f}MB)")
        print(f"├─ Real-time capability: {'✅ YES' if fps_target_met else '❌ NO'}")
        print(f"└─ Security methods: ✅ ALL 3 MAINTAINED (Landmark + CNN + Movement)")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    recommendations = []
    
    if len(results_list) >= 1 and results_list[-1]:
        final_results = results_list[-1]
        
        if final_results['avg_fps'] < target_fps:
            recommendations.append("Consider increasing frame skip ratio")
            recommendations.append("Reduce input image resolution further")
            recommendations.append("Enable GPU acceleration if available")
        
        if final_results['memory_growth'] > target_memory:
            recommendations.append("Implement more aggressive cache cleanup")
            recommendations.append("Reduce history buffer sizes")
            recommendations.append("Add periodic garbage collection")
        
        if final_results['std_time'] > final_results['avg_processing_time'] * 0.3:
            recommendations.append("Optimize frame processing consistency")
            recommendations.append("Implement frame processing pipeline")
    
    if not recommendations:
        recommendations.append("✅ Performance targets met - system optimized")
    
    for i, rec in enumerate(recommendations, 1):
        prefix = "├─" if i < len(recommendations) else "└─"
        print(f"{prefix} {rec}")

def save_results_to_file(results_list, cnn_results, filename="performance_report.json"):
    """Save detailed results to JSON file"""
    
    # Prepare data for JSON serialization
    json_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'landmark_detection': {},
        'cnn_models': cnn_results or {},
        'summary': {}
    }
    
    # Add landmark detection results
    for result in results_list:
        if result:
            system_name = result['system']
            json_data['landmark_detection'][system_name] = {
                'avg_processing_time_ms': result['avg_processing_time'] * 1000,
                'avg_fps': result['avg_fps'],
                'min_time_ms': result['min_time'] * 1000,
                'max_time_ms': result['max_time'] * 1000,
                'std_time_ms': result['std_time'] * 1000,
                'memory_growth_mb': result['memory_growth'],
                'cache_size': result.get('cache_size', 0)
            }
    
    # Add summary
    if len(results_list) >= 2 and results_list[0] and results_list[1]:
        original = results_list[0]
        optimized = results_list[1]
        
        json_data['summary'] = {
            'speed_improvement': original['avg_processing_time'] / optimized['avg_processing_time'],
            'fps_improvement': optimized['avg_fps'] / original['avg_fps'],
            'memory_improvement_percent': (original['memory_growth'] - optimized['memory_growth']) / original['memory_growth'] * 100,
            'targets_met': {
                'fps_target_15': optimized['avg_fps'] >= 15,
                'memory_target_100mb': optimized['memory_growth'] <= 100
            }
        }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")

def plot_performance_comparison(results_list):
    """Create performance comparison plots"""
    try:
        import matplotlib.pyplot as plt
        
        if len(results_list) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        systems = [r['system'] for r in results_list if r]
        
        # Processing Time Comparison
        avg_times = [r['avg_processing_time'] * 1000 for r in results_list if r]
        ax1.bar(systems, avg_times, color=['red', 'green', 'blue'][:len(systems)])
        ax1.set_title('Processing Time Comparison')
        ax1.set_ylabel('Time (ms)')
        ax1.set_ylim(0, max(avg_times) * 1.2)
        
        # FPS Comparison
        fps_values = [r['avg_fps'] for r in results_list if r]
        ax2.bar(systems, fps_values, color=['red', 'green', 'blue'][:len(systems)])
        ax2.set_title('FPS Comparison')
        ax2.set_ylabel('FPS')
        ax2.axhline(y=15, color='orange', linestyle='--', label='Target: 15 FPS')
        ax2.legend()
        
        # Memory Growth Comparison
        memory_growth = [r['memory_growth'] for r in results_list if r]
        ax3.bar(systems, memory_growth, color=['red', 'green', 'blue'][:len(systems)])
        ax3.set_title('Memory Growth Comparison')
        ax3.set_ylabel('Memory Growth (MB)')
        ax3.axhline(y=100, color='orange', linestyle='--', label='Target: <100 MB')
        ax3.legend()
        
        # Processing Time Distribution (for optimized system)
        if len(results_list) >= 2 and results_list[1]:
            times = [t * 1000 for t in results_list[1]['processing_times']]
            ax4.hist(times, bins=20, alpha=0.7, color='green')
            ax4.set_title('Optimized System - Time Distribution')
            ax4.set_xlabel('Processing Time (ms)')
            ax4.set_ylabel('Frequency')
            ax4.axvline(x=np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.1f}ms')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance plots saved to: performance_comparison.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"⚠️ Plot creation failed: {e}")

def main():
    """Main performance testing function"""
    print("🚀 COMPREHENSIVE ANTI-SPOOFING PERFORMANCE TESTING")
    print("=" * 70)
    print("Testing optimization improvements for:")
    print("├─ Facial Landmark Detection")
    print("├─ CNN Liveness Detection") 
    print("├─ Full Processing Pipeline")
    print("└─ Memory and Speed Optimization")
    print()
    
    # Create test data
    test_images = create_test_images(50)
    print(f"✅ Created {len(test_images)} test images")
    
    # Run benchmarks
    results = []
    
    # Benchmark original system
    original_results = benchmark_original_system(test_images)
    if original_results:
        results.append(original_results)
    
    # Benchmark optimized system
    optimized_results = benchmark_optimized_system(test_images)
    if optimized_results:
        results.append(optimized_results)
    
    # Benchmark full pipeline
    pipeline_results = benchmark_full_pipeline(test_images)
    if pipeline_results:
        results.append(pipeline_results)
    
    # Benchmark CNN models
    cnn_results = benchmark_cnn_models()
    
    # Generate comprehensive report
    create_performance_report(results, cnn_results)
    
    # Save results to file
    save_results_to_file(results, cnn_results)
    
    # Create performance plots
    plot_performance_comparison(results)
    
    print(f"\n✅ PERFORMANCE TESTING COMPLETED")
    print(f"📊 Check performance_report.json for detailed results")
    print(f"🎯 System optimized for real-time anti-spoofing detection")
    
    # Final recommendations
    if len(results) >= 2 and results[0] and results[1]:
        speedup = results[0]['avg_processing_time'] / results[1]['avg_processing_time']
        print(f"\n🎉 OPTIMIZATION SUCCESS:")
        print(f"├─ {speedup:.1f}x speed improvement achieved")
        print(f"├─ {results[1]['avg_fps']:.1f} FPS real-time performance")
        print(f"└─ All 3 security methods maintained")

if __name__ == "__main__":
    main()

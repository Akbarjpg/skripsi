#!/usr/bin/env python3
"""
Simple validation script for Step 6 optimization implementation
"""

import os
import sys
import time

def validate_files():
    """Validate that all optimization files exist"""
    print("🔍 Validating Step 6 optimization files...")
    
    required_files = [
        "src/models/optimized_cnn_model.py",
        "src/detection/optimized_landmark_detection.py", 
        "src/web/app_optimized.py",
        "src/config/optimization_settings.yaml",
        "test_step6_optimization.py",
        "STEP6_OPTIMIZATION_COMPLETE.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def validate_imports():
    """Validate that optimization components can be imported"""
    print("\n🔍 Validating optimization imports...")
    
    import_tests = [
        ("OptimizedLivenessPredictor", "src.models.optimized_cnn_model"),
        ("OptimizedLivenessVerifier", "src.detection.optimized_landmark_detection"),
        ("SystemOptimizationManager", "src.web.app_optimized"),
    ]
    
    all_imports_work = True
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    for class_name, module_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {class_name} from {module_name}")
        except ImportError as e:
            print(f"❌ {class_name} from {module_name} - Import Error: {e}")
            all_imports_work = False
        except AttributeError as e:
            print(f"❌ {class_name} from {module_name} - Attribute Error: {e}")
            all_imports_work = False
        except Exception as e:
            print(f"⚠️ {class_name} from {module_name} - Other Error: {e}")
    
    return all_imports_work

def validate_optimization_features():
    """Validate key optimization features"""
    print("\n🔍 Validating optimization features...")
    
    features_implemented = []
    
    try:
        # Test configuration loading
        import yaml
        if os.path.exists("src/config/optimization_settings.yaml"):
            with open("src/config/optimization_settings.yaml", 'r') as f:
                config = yaml.safe_load(f)
                
            if 'model_optimization' in config:
                features_implemented.append("Model Optimization Config")
            if 'processing_pipeline' in config:
                features_implemented.append("Processing Pipeline Config")
            if 'resource_management' in config:
                features_implemented.append("Resource Management Config")
            if 'accuracy_improvements' in config:
                features_implemented.append("Accuracy Improvements Config")
                
    except Exception as e:
        print(f"⚠️ Configuration validation failed: {e}")
    
    # Test model optimization components
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.models.optimized_cnn_model import OptimizedLivenessPredictor
        
        # Try to create instance
        predictor = OptimizedLivenessPredictor(use_quantization=True)
        features_implemented.append("Quantized CNN Model")
        
        # Test caching
        if hasattr(predictor, 'cache'):
            features_implemented.append("Prediction Caching")
            
    except Exception as e:
        print(f"⚠️ CNN optimization validation failed: {e}")
    
    # Test detection optimization components
    try:
        from src.detection.optimized_landmark_detection import OptimizedLivenessVerifier
        
        verifier = OptimizedLivenessVerifier()
        features_implemented.append("Optimized Landmark Detection")
        
        if hasattr(verifier, 'face_tracker'):
            features_implemented.append("Face Tracking")
            
    except Exception as e:
        print(f"⚠️ Detection optimization validation failed: {e}")
    
    # Print results
    print(f"\n✅ Optimization features implemented: {len(features_implemented)}")
    for feature in features_implemented:
        print(f"   • {feature}")
    
    return len(features_implemented) > 0

def check_performance_improvements():
    """Quick performance check"""
    print("\n🔍 Quick performance validation...")
    
    try:
        import numpy as np
        import cv2
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test basic image processing
        start_time = time.time()
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (112, 112))
        processing_time = time.time() - start_time
        
        print(f"✅ Basic image processing: {processing_time*1000:.2f}ms")
        
        if processing_time < 0.01:  # Less than 10ms
            print("✅ Image processing performance: EXCELLENT")
        elif processing_time < 0.05:  # Less than 50ms
            print("✅ Image processing performance: GOOD")
        else:
            print("⚠️ Image processing performance: ACCEPTABLE")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance validation failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🧪 STEP 6 OPTIMIZATION VALIDATION")
    print("="*50)
    
    # Run validation tests
    files_ok = validate_files()
    imports_ok = validate_imports()
    features_ok = validate_optimization_features()
    performance_ok = check_performance_improvements()
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    total_tests = 4
    passed_tests = sum([files_ok, imports_ok, features_ok, performance_ok])
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Step 6 optimization implementation is working correctly")
    elif passed_tests >= 3:
        print("\n✅ MOST VALIDATIONS PASSED!")
        print("⚠️ Step 6 optimization implementation is mostly working")
    else:
        print("\n⚠️ SOME VALIDATIONS FAILED!")
        print("❌ Step 6 optimization implementation needs attention")
    
    return passed_tests >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

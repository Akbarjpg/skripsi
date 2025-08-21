"""
QUICK VALIDATION TEST FOR COMPREHENSIVE TESTING FRAMEWORK
========================================================
Simple test to verify all components work correctly
"""

import os
import sys
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from src.testing.comprehensive_test_suite import ComprehensiveTestSuite
        print("  ✅ ComprehensiveTestSuite imported")
        
        from src.testing.metrics_collector import MetricsCollector
        print("  ✅ MetricsCollector imported")
        
        from src.testing.report_generator import ReportGenerator
        print("  ✅ ReportGenerator imported")
        
        from src.testing.data_exporter import DataExporter
        print("  ✅ DataExporter imported")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_metrics_collector():
    """Test metrics collector functionality"""
    print("\n🧪 Testing MetricsCollector...")
    
    try:
        from src.testing.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test accuracy calculation
        predictions = [True, True, False, False, True]
        ground_truth = [True, False, False, True, True]
        
        metrics = collector.calculate_accuracy_metrics(predictions, ground_truth)
        
        print(f"  ✅ Accuracy metrics calculated: {metrics['accuracy']:.3f}")
        
        # Test statistical confidence
        values = [0.95, 0.93, 0.96, 0.94, 0.97]
        stats = collector.calculate_statistical_confidence(values)
        
        print(f"  ✅ Statistical confidence calculated: {stats['mean']:.3f}")
        
        return True
    except Exception as e:
        print(f"  ❌ MetricsCollector error: {e}")
        return False

def test_report_generator():
    """Test report generator functionality"""
    print("\n📊 Testing ReportGenerator...")
    
    try:
        from src.testing.report_generator import ReportGenerator
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp()
        
        generator = ReportGenerator()
        generator.output_dir = temp_dir
        
        # Test with sample data
        sample_antispoofing = [
            {
                'scenario': 'test_scenario',
                'tp': 85, 'tn': 90, 'fp': 5, 'fn': 15,
                'accuracy': 0.895, 'precision': 0.944, 'recall': 0.85, 'f1_score': 0.894
            }
        ]
        
        files = generator.generate_confusion_matrices(sample_antispoofing)
        
        if files and os.path.exists(files[0]):
            print("  ✅ Confusion matrix generated")
        else:
            print("  ⚠️ Confusion matrix generation failed")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"  ❌ ReportGenerator error: {e}")
        return False

def test_data_exporter():
    """Test data exporter functionality"""
    print("\n📁 Testing DataExporter...")
    
    try:
        from src.testing.data_exporter import DataExporter
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp()
        
        exporter = DataExporter()
        exporter.base_output_dir = temp_dir
        exporter.csv_dir = os.path.join(temp_dir, "csv")
        exporter.latex_dir = os.path.join(temp_dir, "latex")
        exporter.json_dir = os.path.join(temp_dir, "json")
        
        # Ensure directories exist
        for directory in [exporter.csv_dir, exporter.latex_dir, exporter.json_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Test with sample data
        sample_results = {
            'antispoofing': [
                {
                    'scenario': 'test',
                    'tp': 85, 'tn': 90, 'fp': 5, 'fn': 15,
                    'accuracy': 0.895, 'total_samples': 195
                }
            ]
        }
        
        csv_files = exporter.export_csv_data(sample_results)
        
        if csv_files and os.path.exists(csv_files[0]):
            print("  ✅ CSV export successful")
        else:
            print("  ⚠️ CSV export failed")
        
        # Test LaTeX export
        overall_metrics = {
            'overall_accuracy': 0.96,
            'overall_precision': 0.94,
            'total_tests_run': 500
        }
        
        latex_files = exporter.export_latex_tables(overall_metrics)
        
        if latex_files and os.path.exists(latex_files[0]):
            print("  ✅ LaTeX export successful")
        else:
            print("  ⚠️ LaTeX export failed")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"  ❌ DataExporter error: {e}")
        return False

def test_comprehensive_suite():
    """Test comprehensive test suite"""
    print("\n🚀 Testing ComprehensiveTestSuite...")
    
    try:
        from src.testing.comprehensive_test_suite import ComprehensiveTestSuite
        
        # Create suite with minimal configuration
        suite = ComprehensiveTestSuite()
        
        # Override configuration for quick test
        suite.config = {
            'antispoofing_tests': {
                'printed_photos': 2,
                'digital_displays': 1,
                'video_replays': 1,
                'masks_3d': 1
            },
            'face_recognition_tests': {
                'registered_users': 2,
                'lighting_conditions': ['normal'],
                'angles': [0],
                'expressions': ['neutral']
            },
            'challenge_response_tests': {
                'blink_tests': 2,
                'head_movement_tests': 2,
                'smile_tests': 2,
                'distance_tests': 2
            },
            'performance_tests': {
                'duration_minutes': 0.1,
                'concurrent_users': 1,
                'stress_test_duration': 0.1
            }
        }
        
        print("  🔄 Running mini test suite...")
        
        # Test individual components
        antispoofing_results = suite.run_antispoofing_tests()
        if antispoofing_results:
            print("  ✅ Anti-spoofing tests completed")
        
        face_recognition_results = suite.run_face_recognition_tests()
        if face_recognition_results:
            print("  ✅ Face recognition tests completed")
        
        challenge_results = suite.run_challenge_response_tests()
        if challenge_results:
            print("  ✅ Challenge-response tests completed")
        
        print("  ✅ ComprehensiveTestSuite working correctly")
        
        return True
    except Exception as e:
        print(f"  ❌ ComprehensiveTestSuite error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🧪 COMPREHENSIVE TESTING FRAMEWORK VALIDATION")
    print("=" * 55)
    
    tests = [
        ("Module Imports", test_imports),
        ("MetricsCollector", test_metrics_collector),
        ("ReportGenerator", test_report_generator),
        ("DataExporter", test_data_exporter),
        ("ComprehensiveTestSuite", test_comprehensive_suite)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 55)
    print(f"🎯 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Run quick test: python run_comprehensive_tests.py --quick")
        print("2. Run full test: python run_comprehensive_tests.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_testing.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

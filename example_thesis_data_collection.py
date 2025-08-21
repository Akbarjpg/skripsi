"""
Example: How to integrate the structured test format with your actual system testing
"""

import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from testing.comprehensive_test_data_collector import TestDataCollector
from testing.test_result_formatter import DatasetInfo

def run_thesis_data_collection_example():
    """
    Example of how to use the structured format system for thesis data collection
    """
    print("THESIS DATA COLLECTION EXAMPLE")
    print("=" * 50)
    
    # Initialize the comprehensive test data collector
    collector = TestDataCollector("thesis_test_results")
    
    # Start a test session for thesis evaluation
    session_id = collector.start_test_session("Thesis_Chapter4_Evaluation")
    print(f"Started session: {session_id}")
    
    # Example dataset information (replace with your actual dataset)
    dataset_info = DatasetInfo(
        total_samples=1000,
        real_samples=500,
        fake_samples=500,
        unique_individuals=100
    )
    
    print("\n1. ANTI-SPOOFING TEST RESULTS")
    print("-" * 30)
    
    # Simulate anti-spoofing test results (replace with actual test results)
    antispoofing_results = [
        {
            'name': 'Print_Photo_Attack_Test',
            'tp': 485, 'tn': 475, 'fp': 25, 'fn': 15,
            'times': [0.12, 0.15, 0.11, 0.14, 0.13]
        },
        {
            'name': 'Digital_Display_Attack_Test', 
            'tp': 490, 'tn': 470, 'fp': 30, 'fn': 10,
            'times': [0.13, 0.16, 0.12, 0.15, 0.14]
        },
        {
            'name': 'Video_Replay_Attack_Test',
            'tp': 480, 'tn': 480, 'fp': 20, 'fn': 20,
            'times': [0.14, 0.17, 0.13, 0.16, 0.15]
        }
    ]
    
    for test in antispoofing_results:
        json_path = collector.collect_antispoofing_test_data(
            test_name=test['name'],
            true_positives=test['tp'],
            true_negatives=test['tn'],
            false_positives=test['fp'],
            false_negatives=test['fn'],
            detection_times=test['times'],
            dataset_info=dataset_info,
            detailed_log_path=f"logs/{test['name']}_detailed.log"
        )
        
        # Calculate and display key metrics
        accuracy = (test['tp'] + test['tn']) / (test['tp'] + test['tn'] + test['fp'] + test['fn'])
        far = test['fp'] / (test['fp'] + test['tn'])
        frr = test['fn'] / (test['fn'] + test['tp'])
        
        print(f"  {test['name']}:")
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    FAR: {far:.4f}")
        print(f"    FRR: {frr:.4f}")
        print(f"    Saved to: {json_path}")
        print()
    
    print("2. FACE RECOGNITION TEST RESULTS")
    print("-" * 33)
    
    # Simulate face recognition test results (replace with actual test results)
    face_recognition_results = [
        {
            'name': 'Frontal_Face_Recognition_Test',
            'rank1': 92, 'rank5': 98, 'queries': 100,
            'verification_correct': 95, 'verification_total': 100,
            'times': [0.08, 0.09, 0.07, 0.10, 0.08],
            'false_matches': 2, 'false_non_matches': 3
        },
        {
            'name': 'Multi_Angle_Recognition_Test',
            'rank1': 88, 'rank5': 95, 'queries': 100,
            'verification_correct': 90, 'verification_total': 100,
            'times': [0.09, 0.11, 0.08, 0.12, 0.10],
            'false_matches': 3, 'false_non_matches': 7
        }
    ]
    
    for test in face_recognition_results:
        json_path = collector.collect_face_recognition_test_data(
            test_name=test['name'],
            rank1_correct=test['rank1'],
            rank5_correct=test['rank5'],
            total_queries=test['queries'],
            verification_correct=test['verification_correct'],
            verification_total=test['verification_total'],
            recognition_times=test['times'],
            false_matches=test['false_matches'],
            false_non_matches=test['false_non_matches'],
            dataset_info=dataset_info,
            detailed_log_path=f"logs/{test['name']}_detailed.log"
        )
        
        # Calculate key metrics
        rank1_accuracy = test['rank1'] / test['queries']
        verification_accuracy = test['verification_correct'] / test['verification_total']
        
        print(f"  {test['name']}:")
        print(f"    Rank-1 Accuracy: {rank1_accuracy:.2%}")
        print(f"    Verification Accuracy: {verification_accuracy:.2%}")
        print(f"    Saved to: {json_path}")
        print()
    
    print("3. INTEGRATION TEST RESULTS")
    print("-" * 27)
    
    # Simulate integration test (complete system workflow)
    integration_json = collector.collect_integration_test_data(
        test_name="End_to_End_System_Test",
        antispoofing_metrics={
            'tp': 475, 'tn': 465, 'fp': 35, 'fn': 25,
            'detection_times': [0.15, 0.16, 0.14, 0.17, 0.15]
        },
        face_recognition_metrics={
            'rank1_correct': 85, 'rank5_correct': 92, 'total_queries': 100,
            'verification_correct': 87, 'verification_total': 100,
            'recognition_times': [0.10, 0.11, 0.09, 0.12, 0.10],
            'false_matches': 4, 'false_non_matches': 9
        },
        end_to_end_times=[2.5, 2.8, 2.3, 2.7, 2.6, 2.4],
        dataset_info=dataset_info,
        detailed_log_path="logs/integration_test_detailed.log"
    )
    
    print(f"  End-to-End System Test saved to: {integration_json}")
    
    print("\n4. GENERATING COMPREHENSIVE REPORTS")
    print("-" * 39)
    
    # End session and generate all outputs
    generated_files = collector.end_test_session()
    
    print("Generated files for thesis documentation:")
    for file_type, file_info in generated_files.items():
        if isinstance(file_info, dict):
            print(f"  {file_type}: {len(file_info)} files")
            for subtype, path in file_info.items():
                if isinstance(path, str) and len(path) < 100:  # Show shorter paths
                    print(f"    - {subtype}")
                elif isinstance(path, dict):
                    print(f"    - {subtype}: {len(path)} files")
        else:
            print(f"  {file_type}")
    
    print("\n" + "="*60)
    print("THESIS DATA COLLECTION COMPLETE!")
    print("="*60)
    
    print("\nYour test data is now organized for thesis Chapter 4:")
    print("\n📊 Section 4.1: Anti-Spoofing Test Results")
    print("   - Statistical analysis of detection accuracy")
    print("   - Confusion matrix analysis")
    print("   - Security risk assessment")
    
    print("\n🔍 Section 4.2: Face Recognition Test Results")
    print("   - Rank-1 and Rank-5 accuracy analysis") 
    print("   - CMC curve data")
    print("   - Error rate analysis")
    
    print("\n⚡ Section 4.3: System Performance Analysis")
    print("   - Resource utilization metrics")
    print("   - Processing time analysis")
    print("   - Bottleneck identification")
    
    print("\n📈 Section 4.4: Comparative Analysis")
    print("   - Cross-system performance comparison")
    print("   - Accuracy vs performance trade-offs")
    
    print("\n💭 Section 4.5: Discussion of Results")
    print("   - Key findings summary")
    print("   - Limitations analysis")
    print("   - Future work suggestions")
    
    print(f"\n📁 All files saved in: thesis_test_results/")
    print(f"📄 LaTeX tables ready for inclusion in: thesis_test_results/latex/")
    print(f"📋 Thesis chapter organization in: Thesis/Chapter4/")
    
    print("\n✅ Your thesis documentation is ready!")

if __name__ == "__main__":
    run_thesis_data_collection_example()

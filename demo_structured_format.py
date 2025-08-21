"""
Simplified example of the structured test result format system
"""

import sys
import os
import json

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from testing.test_result_formatter import TestResultFormatter, DatasetInfo
from testing.latex_table_generator import LaTeXTableGenerator
from testing.thesis_data_organizer import ThesisDataOrganizer

def demonstrate_structured_format():
    """
    Demonstrate the complete structured test result format system
    """
    print("STRUCTURED TEST RESULT FORMAT DEMONSTRATION")
    print("=" * 55)
    
    # Initialize components
    formatter = TestResultFormatter("demo_output")
    latex_generator = LaTeXTableGenerator("demo_output/latex")
    organizer = ThesisDataOrganizer("demo_output/Thesis/Chapter4")
    
    print("\n📊 CREATING STRUCTURED TEST RESULTS...")
    
    # Example dataset
    dataset_info = DatasetInfo(
        total_samples=1000,
        real_samples=500,
        fake_samples=500,
        unique_individuals=100
    )
    
    # Create multiple test results to demonstrate the system
    test_results = []
    
    # Anti-spoofing test 1
    antispoofing_result1 = formatter.create_antispoofing_result(
        tp=485, tn=475, fp=25, fn=15,
        detection_times=[0.12, 0.15, 0.11, 0.14, 0.13]
    )
    
    complete_result1 = formatter.create_complete_test_result(
        test_type="antispoofing",
        dataset_info=dataset_info,
        antispoofing_results=antispoofing_result1,
        detailed_logs_path="logs/antispoofing_test1.log"
    )
    
    json_path1 = formatter.save_test_result_json(complete_result1)
    test_results.append(complete_result1)
    
    print(f"✓ Anti-spoofing Test 1 - Accuracy: {antispoofing_result1.accuracy:.2%}")
    
    # Anti-spoofing test 2
    antispoofing_result2 = formatter.create_antispoofing_result(
        tp=490, tn=470, fp=30, fn=10,
        detection_times=[0.13, 0.16, 0.12, 0.15, 0.14]
    )
    
    complete_result2 = formatter.create_complete_test_result(
        test_type="antispoofing",
        dataset_info=dataset_info,
        antispoofing_results=antispoofing_result2,
        detailed_logs_path="logs/antispoofing_test2.log"
    )
    
    json_path2 = formatter.save_test_result_json(complete_result2)
    test_results.append(complete_result2)
    
    print(f"✓ Anti-spoofing Test 2 - Accuracy: {antispoofing_result2.accuracy:.2%}")
    
    # Face recognition test
    face_recognition_result = formatter.create_face_recognition_result(
        rank1_correct=92, rank5_correct=98, total_queries=100,
        verification_correct=95, verification_total=100,
        recognition_times=[0.08, 0.09, 0.07, 0.10, 0.08],
        false_matches=2, false_non_matches=3
    )
    
    complete_result3 = formatter.create_complete_test_result(
        test_type="face_recognition",
        dataset_info=dataset_info,
        face_recognition_results=face_recognition_result,
        detailed_logs_path="logs/face_recognition_test.log"
    )
    
    json_path3 = formatter.save_test_result_json(complete_result3)
    test_results.append(complete_result3)
    
    print(f"✓ Face Recognition Test - Rank-1 Accuracy: {face_recognition_result.rank1_accuracy:.2%}")
    
    print(f"\n📁 Saved {len(test_results)} test results in JSON format")
    
    # Convert to dict format for LaTeX and organizer
    from dataclasses import asdict
    results_dicts = [asdict(result) for result in test_results]
    
    print(f"\n📄 GENERATING LATEX TABLES...")
    
    # Generate LaTeX tables
    json_files = [json_path1, json_path2, json_path3]
    latex_tables = latex_generator.generate_complete_thesis_tables(json_files)
    
    print(f"✓ Generated {len(latex_tables)} LaTeX table types:")
    for table_type, file_path in latex_tables.items():
        print(f"  - {table_type}: {file_path}")
    
    print(f"\n📋 ORGANIZING FOR THESIS CHAPTER 4...")
    
    # Organize for thesis
    organized_files = organizer.organize_complete_thesis_data(json_files)
    
    print(f"✓ Organized data for {len(organized_files)} thesis sections:")
    for section, files in organized_files.items():
        if isinstance(files, dict):
            print(f"  - {section}: {len(files)} file types")
        else:
            print(f"  - {section}: 1 file")
    
    print(f"\n📊 DISPLAYING SAMPLE STRUCTURED RESULT...")
    
    # Show the structured format
    sample_result = asdict(complete_result1)
    
    print("\nSample JSON Structure (Anti-spoofing Test):")
    print("-" * 45)
    print(f"Test ID: {sample_result['test_info']['test_id']}")
    print(f"Test Type: {sample_result['test_info']['test_type']}")
    print(f"Test Date: {sample_result['test_info']['test_date']}")
    print(f"Total Samples: {sample_result['test_info']['dataset_info']['total_samples']}")
    
    print("\nAnti-spoofing Results:")
    as_results = sample_result['results']['antispoofing']
    print(f"  Accuracy: {as_results['accuracy']:.2%}")
    print(f"  Precision: {as_results['precision']:.2%}")
    print(f"  Recall: {as_results['recall']:.2%}")
    print(f"  F1-Score: {as_results['f1_score']:.2%}")
    print(f"  FAR: {as_results['far']:.4f}")
    print(f"  FRR: {as_results['frr']:.4f}")
    print(f"  Avg Detection Time: {as_results['average_detection_time']:.3f}s")
    print(f"  Confusion Matrix: {as_results['confusion_matrix']}")
    
    print(f"\n📈 SAMPLE LATEX TABLE PREVIEW...")
    
    # Show LaTeX table preview
    if 'antispoofing_results' in latex_tables:
        with open(latex_tables['antispoofing_results'], 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        print("\nLaTeX Table (first 10 lines):")
        print("-" * 30)
        for i, line in enumerate(latex_content.split('\n')[:10]):
            print(f"{i+1:2d}: {line}")
        if len(latex_content.split('\n')) > 10:
            print("    ... (more lines)")
    
    print(f"\n📁 OUTPUT DIRECTORY STRUCTURE:")
    print("-" * 30)
    print("demo_output/")
    print("├── json/                    # Raw JSON test results")
    print("├── csv/                     # CSV format for analysis")
    print("├── latex/                   # LaTeX tables")
    print("└── Thesis/Chapter4/         # Organized thesis data")
    print("    ├── Section_4.1_Anti_Spoofing_Results/")
    print("    ├── Section_4.2_Face_Recognition_Results/")
    print("    ├── Section_4.3_System_Performance_Analysis/")
    print("    ├── Section_4.4_Comparative_Analysis/")
    print("    └── Section_4.5_Discussion_Results/")
    
    print(f"\n" + "="*60)
    print("✅ STRUCTURED TEST RESULT FORMAT DEMONSTRATION COMPLETE!")
    print("="*60)
    
    print("\n🎯 KEY BENEFITS:")
    print("  ✓ Standardized format for all test results")
    print("  ✓ Automatic LaTeX table generation")
    print("  ✓ Organized structure for thesis chapters")
    print("  ✓ Statistical analysis included")
    print("  ✓ Ready for academic publication")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Replace example data with your actual test results")
    print("  2. Run your anti-spoofing and face recognition tests")
    print("  3. Use the collector to automatically format results")
    print("  4. Include generated LaTeX tables in your thesis")
    print("  5. Use organized data for Chapter 4 writing")
    
    print(f"\n📄 Generated Files:")
    print(f"  - JSON results: {len(test_results)} files")
    print(f"  - LaTeX tables: {len(latex_tables)} files")
    print(f"  - Thesis sections: {len(organized_files)} sections")
    
    return {
        'test_results': test_results,
        'latex_tables': latex_tables,
        'organized_files': organized_files
    }

if __name__ == "__main__":
    results = demonstrate_structured_format()

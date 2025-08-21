"""
Simple test to demonstrate the structured test result format system
"""

import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

try:
    from testing.test_result_formatter import TestResultFormatter, DatasetInfo
    from testing.latex_table_generator import LaTeXTableGenerator
    from testing.thesis_data_organizer import ThesisDataOrganizer
    print("✓ All modules imported successfully!")
    
    # Test basic functionality
    print("\n1. Testing Test Result Formatter...")
    formatter = TestResultFormatter("test_output")
    
    # Create example dataset info
    dataset_info = DatasetInfo(
        total_samples=100,
        real_samples=50,
        fake_samples=50,
        unique_individuals=25
    )
    
    # Create example antispoofing result
    antispoofing_result = formatter.create_antispoofing_result(
        tp=48, tn=47, fp=3, fn=2,
        detection_times=[0.12, 0.15, 0.11, 0.14]
    )
    
    print(f"   Antispoofing accuracy: {antispoofing_result.accuracy:.2%}")
    print(f"   FAR: {antispoofing_result.far:.4f}")
    print(f"   FRR: {antispoofing_result.frr:.4f}")
    
    # Create complete test result
    complete_result = formatter.create_complete_test_result(
        test_type="antispoofing",
        dataset_info=dataset_info,
        antispoofing_results=antispoofing_result,
        detailed_logs_path="test.log"
    )
    
    print(f"   Test ID: {complete_result.test_info.test_id}")
    
    # Save results
    json_path = formatter.save_test_result_json(complete_result)
    csv_path = formatter.save_test_result_csv(complete_result)
    
    print(f"   Saved JSON: {json_path}")
    print(f"   Saved CSV: {csv_path}")
    
    print("\n2. Testing LaTeX Table Generator...")
    latex_generator = LaTeXTableGenerator("test_output/latex")
    
    # Convert to dict for LaTeX generator
    from dataclasses import asdict
    result_dict = asdict(complete_result)
    
    # Generate LaTeX table
    latex_table = latex_generator.generate_antispoofing_results_table([result_dict])
    latex_file = latex_generator.save_latex_table(latex_table, "test_antispoofing.tex")
    
    print(f"   Generated LaTeX table: {latex_file}")
    print(f"   Table preview (first 3 lines):")
    for i, line in enumerate(latex_table.split('\n')[:3]):
        print(f"     {line}")
    
    print("\n3. Testing Thesis Data Organizer...")
    organizer = ThesisDataOrganizer("test_output/Thesis/Chapter4")
    
    # Organize antispoofing results
    organized_files = organizer.organize_antispoofing_results([result_dict])
    
    print(f"   Organized {len(organized_files)} file types:")
    for file_type, file_path in organized_files.items():
        print(f"     {file_type}: {file_path}")
    
    print("\n✓ All tests completed successfully!")
    print("\nThe structured test result format system is working correctly.")
    print("You can now use this system to collect and organize all test data for your thesis.")
    
    print("\nGenerated test files:")
    print("- test_output/json/: JSON test results")
    print("- test_output/csv/: CSV format data")
    print("- test_output/latex/: LaTeX tables")
    print("- test_output/Thesis/Chapter4/: Organized thesis data")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please check that all required modules are available.")
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

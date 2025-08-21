"""
LaTeX Table Generator for Thesis Documentation
Generates publication-ready LaTeX tables from test results
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass

class LaTeXTableGenerator:
    """
    Generates LaTeX tables for thesis documentation
    """
    
    def __init__(self, output_dir: str = "test_results/latex"):
        self.output_dir = output_dir
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def format_number(self, value: float, format_type: str = "percentage") -> str:
        """
        Format numbers according to thesis standards
        
        Args:
            value: Number to format
            format_type: "percentage", "decimal", "time", "integer"
        """
        if format_type == "percentage":
            return f"{value * 100:.2f}\\%"
        elif format_type == "decimal":
            return f"{value:.4f}"
        elif format_type == "time":
            return f"{value:.3f}s"
        elif format_type == "integer":
            return f"{int(value)}"
        else:
            return f"{value:.3f}"
    
    def generate_antispoofing_results_table(self, test_results: List[Dict], 
                                            caption: str = "Anti-Spoofing Detection Results",
                                            label: str = "tab:antispoofing_results") -> str:
        """
        Generate LaTeX table for anti-spoofing results
        """
        latex_content = []
        
        # Table header
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append(f"\\caption{{{caption}}}")
        latex_content.append(f"\\label{{{label}}}")
        latex_content.append("\\begin{tabular}{|l|c|c|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("\\textbf{Test ID} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{FAR} & \\textbf{FRR} \\\\")
        latex_content.append("\\hline")
        
        # Table rows
        for result in test_results:
            if result.get('results', {}).get('antispoofing'):
                as_result = result['results']['antispoofing']
                test_id = result['test_info']['test_id'].split('_')[-1]  # Short ID
                
                row = f"{test_id} & " \
                      f"{self.format_number(as_result['accuracy'])} & " \
                      f"{self.format_number(as_result['precision'])} & " \
                      f"{self.format_number(as_result['recall'])} & " \
                      f"{self.format_number(as_result['f1_score'])} & " \
                      f"{self.format_number(as_result['far'])} & " \
                      f"{self.format_number(as_result['frr'])} \\\\"
                
                latex_content.append(row)
                latex_content.append("\\hline")
        
        # Calculate averages
        if test_results:
            avg_accuracy = np.mean([r['results']['antispoofing']['accuracy'] 
                                   for r in test_results if r.get('results', {}).get('antispoofing')])
            avg_precision = np.mean([r['results']['antispoofing']['precision'] 
                                    for r in test_results if r.get('results', {}).get('antispoofing')])
            avg_recall = np.mean([r['results']['antispoofing']['recall'] 
                                 for r in test_results if r.get('results', {}).get('antispoofing')])
            avg_f1 = np.mean([r['results']['antispoofing']['f1_score'] 
                             for r in test_results if r.get('results', {}).get('antispoofing')])
            avg_far = np.mean([r['results']['antispoofing']['far'] 
                              for r in test_results if r.get('results', {}).get('antispoofing')])
            avg_frr = np.mean([r['results']['antispoofing']['frr'] 
                              for r in test_results if r.get('results', {}).get('antispoofing')])
            
            latex_content.append("\\hline")
            avg_row = f"\\textbf{{Average}} & " \
                     f"\\textbf{{{self.format_number(avg_accuracy)}}} & " \
                     f"\\textbf{{{self.format_number(avg_precision)}}} & " \
                     f"\\textbf{{{self.format_number(avg_recall)}}} & " \
                     f"\\textbf{{{self.format_number(avg_f1)}}} & " \
                     f"\\textbf{{{self.format_number(avg_far)}}} & " \
                     f"\\textbf{{{self.format_number(avg_frr)}}} \\\\"
            latex_content.append(avg_row)
            latex_content.append("\\hline")
        
        # Table footer
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_face_recognition_results_table(self, test_results: List[Dict],
                                                caption: str = "Face Recognition Accuracy Results",
                                                label: str = "tab:face_recognition_results") -> str:
        """
        Generate LaTeX table for face recognition results
        """
        latex_content = []
        
        # Table header
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append(f"\\caption{{{caption}}}")
        latex_content.append(f"\\label{{{label}}}")
        latex_content.append("\\begin{tabular}{|l|c|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("\\textbf{Test ID} & \\textbf{Rank-1} & \\textbf{Rank-5} & \\textbf{Verification} & \\textbf{FMR} & \\textbf{FNMR} \\\\")
        latex_content.append("\\textbf{} & \\textbf{Accuracy} & \\textbf{Accuracy} & \\textbf{Accuracy} & \\textbf{} & \\textbf{} \\\\")
        latex_content.append("\\hline")
        
        # Table rows
        for result in test_results:
            if result.get('results', {}).get('face_recognition'):
                fr_result = result['results']['face_recognition']
                test_id = result['test_info']['test_id'].split('_')[-1]  # Short ID
                
                row = f"{test_id} & " \
                      f"{self.format_number(fr_result['rank1_accuracy'])} & " \
                      f"{self.format_number(fr_result['rank5_accuracy'])} & " \
                      f"{self.format_number(fr_result['verification_accuracy'])} & " \
                      f"{self.format_number(fr_result['false_match_rate'])} & " \
                      f"{self.format_number(fr_result['false_non_match_rate'])} \\\\"
                
                latex_content.append(row)
                latex_content.append("\\hline")
        
        # Calculate averages
        if test_results:
            fr_results = [r['results']['face_recognition'] for r in test_results 
                         if r.get('results', {}).get('face_recognition')]
            
            if fr_results:
                avg_rank1 = np.mean([r['rank1_accuracy'] for r in fr_results])
                avg_rank5 = np.mean([r['rank5_accuracy'] for r in fr_results])
                avg_verification = np.mean([r['verification_accuracy'] for r in fr_results])
                avg_fmr = np.mean([r['false_match_rate'] for r in fr_results])
                avg_fnmr = np.mean([r['false_non_match_rate'] for r in fr_results])
                
                latex_content.append("\\hline")
                avg_row = f"\\textbf{{Average}} & " \
                         f"\\textbf{{{self.format_number(avg_rank1)}}} & " \
                         f"\\textbf{{{self.format_number(avg_rank5)}}} & " \
                         f"\\textbf{{{self.format_number(avg_verification)}}} & " \
                         f"\\textbf{{{self.format_number(avg_fmr)}}} & " \
                         f"\\textbf{{{self.format_number(avg_fnmr)}}} \\\\"
                latex_content.append(avg_row)
                latex_content.append("\\hline")
        
        # Table footer
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_performance_comparison_table(self, test_results: List[Dict],
                                              caption: str = "System Performance Metrics",
                                              label: str = "tab:performance_metrics") -> str:
        """
        Generate LaTeX table for performance comparison
        """
        latex_content = []
        
        # Table header
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append(f"\\caption{{{caption}}}")
        latex_content.append(f"\\label{{{label}}}")
        latex_content.append("\\begin{tabular}{|l|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("\\textbf{Test Type} & \\textbf{CPU Usage} & \\textbf{Memory Usage} & \\textbf{FPS} & \\textbf{Processing Time} \\\\")
        latex_content.append("\\textbf{} & \\textbf{(\\%)} & \\textbf{(MB)} & \\textbf{} & \\textbf{(seconds)} \\\\")
        latex_content.append("\\hline")
        
        # Group results by test type
        test_types = {}
        for result in test_results:
            if result.get('results', {}).get('performance'):
                test_type = result['test_info']['test_type']
                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(result['results']['performance'])
        
        # Generate rows for each test type
        for test_type, perf_results in test_types.items():
            avg_cpu = np.mean([r['cpu_usage_avg'] for r in perf_results])
            avg_memory = np.mean([r['memory_usage_avg'] for r in perf_results])
            avg_fps = np.mean([r['fps_avg'] for r in perf_results])
            avg_time = np.mean([r['total_processing_time'] for r in perf_results])
            
            row = f"{test_type.title()} & " \
                  f"{self.format_number(avg_cpu/100, 'percentage')} & " \
                  f"{self.format_number(avg_memory, 'decimal')} & " \
                  f"{self.format_number(avg_fps, 'decimal')} & " \
                  f"{self.format_number(avg_time, 'time')} \\\\"
            
            latex_content.append(row)
            latex_content.append("\\hline")
        
        # Table footer
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_confusion_matrix_table(self, confusion_matrix: List[List[int]],
                                        caption: str = "Confusion Matrix for Anti-Spoofing Detection",
                                        label: str = "tab:confusion_matrix") -> str:
        """
        Generate LaTeX table for confusion matrix
        """
        latex_content = []
        
        # Table header
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append(f"\\caption{{{caption}}}")
        latex_content.append(f"\\label{{{label}}}")
        latex_content.append("\\begin{tabular}{|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("\\multirow{2}{*}{\\textbf{Actual}} & \\multicolumn{2}{c|}{\\textbf{Predicted}} \\\\")
        latex_content.append("\\cline{2-3}")
        latex_content.append(" & \\textbf{Real} & \\textbf{Fake} \\\\")
        latex_content.append("\\hline")
        
        # Matrix values: [[TP, FP], [FN, TN]]
        tp, fp = confusion_matrix[0]
        fn, tn = confusion_matrix[1]
        
        latex_content.append(f"\\textbf{{Real}} & {tp} & {fn} \\\\")
        latex_content.append("\\hline")
        latex_content.append(f"\\textbf{{Fake}} & {fp} & {tn} \\\\")
        latex_content.append("\\hline")
        
        # Table footer
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def generate_dataset_info_table(self, test_results: List[Dict],
                                    caption: str = "Dataset Information Summary",
                                    label: str = "tab:dataset_info") -> str:
        """
        Generate LaTeX table for dataset information
        """
        latex_content = []
        
        # Table header
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append(f"\\caption{{{caption}}}")
        latex_content.append(f"\\label{{{label}}}")
        latex_content.append("\\begin{tabular}{|l|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("\\textbf{Test Type} & \\textbf{Total Samples} & \\textbf{Real Samples} & \\textbf{Fake Samples} & \\textbf{Unique Individuals} \\\\")
        latex_content.append("\\hline")
        
        # Group by test type
        test_types = {}
        for result in test_results:
            test_type = result['test_info']['test_type']
            dataset_info = result['test_info']['dataset_info']
            
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(dataset_info)
        
        # Generate rows
        for test_type, datasets in test_types.items():
            total_samples = sum([d['total_samples'] for d in datasets])
            total_real = sum([d['real_samples'] for d in datasets])
            total_fake = sum([d['fake_samples'] for d in datasets])
            unique_individuals = max([d['unique_individuals'] for d in datasets])  # Take max for unique count
            
            row = f"{test_type.title()} & " \
                  f"{self.format_number(total_samples, 'integer')} & " \
                  f"{self.format_number(total_real, 'integer')} & " \
                  f"{self.format_number(total_fake, 'integer')} & " \
                  f"{self.format_number(unique_individuals, 'integer')} \\\\"
            
            latex_content.append(row)
            latex_content.append("\\hline")
        
        # Table footer
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        return "\n".join(latex_content)
    
    def save_latex_table(self, latex_content: str, filename: str) -> str:
        """
        Save LaTeX table to file
        
        Returns:
            Path to saved file
        """
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        return filepath
    
    def generate_complete_thesis_tables(self, test_results_json_files: List[str]) -> Dict[str, str]:
        """
        Generate all LaTeX tables needed for thesis Chapter 4
        
        Args:
            test_results_json_files: List of paths to JSON test result files
            
        Returns:
            Dictionary mapping table type to file path
        """
        # Load all test results
        all_results = []
        for json_file in test_results_json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                all_results.append(json.load(f))
        
        generated_files = {}
        
        # Generate antispoofing results table
        antispoofing_results = [r for r in all_results if r.get('results', {}).get('antispoofing')]
        if antispoofing_results:
            table_content = self.generate_antispoofing_results_table(antispoofing_results)
            filepath = self.save_latex_table(table_content, "antispoofing_results.tex")
            generated_files['antispoofing_results'] = filepath
        
        # Generate face recognition results table
        face_recognition_results = [r for r in all_results if r.get('results', {}).get('face_recognition')]
        if face_recognition_results:
            table_content = self.generate_face_recognition_results_table(face_recognition_results)
            filepath = self.save_latex_table(table_content, "face_recognition_results.tex")
            generated_files['face_recognition_results'] = filepath
        
        # Generate performance comparison table
        performance_results = [r for r in all_results if r.get('results', {}).get('performance')]
        if performance_results:
            table_content = self.generate_performance_comparison_table(performance_results)
            filepath = self.save_latex_table(table_content, "performance_comparison.tex")
            generated_files['performance_comparison'] = filepath
        
        # Generate dataset info table
        if all_results:
            table_content = self.generate_dataset_info_table(all_results)
            filepath = self.save_latex_table(table_content, "dataset_info.tex")
            generated_files['dataset_info'] = filepath
        
        # Generate confusion matrix table (using first antispoofing result as example)
        if antispoofing_results:
            confusion_matrix = antispoofing_results[0]['results']['antispoofing']['confusion_matrix']
            table_content = self.generate_confusion_matrix_table(confusion_matrix)
            filepath = self.save_latex_table(table_content, "confusion_matrix.tex")
            generated_files['confusion_matrix'] = filepath
        
        return generated_files

# Example usage
if __name__ == "__main__":
    generator = LaTeXTableGenerator()
    
    # Example test result (replace with actual data)
    example_result = {
        "test_info": {
            "test_id": "TEST_20250821_123456_abc123",
            "test_date": "2025-08-21 12:34:56",
            "test_type": "antispoofing",
            "dataset_info": {
                "total_samples": 1000,
                "real_samples": 500,
                "fake_samples": 500,
                "unique_individuals": 100
            }
        },
        "results": {
            "antispoofing": {
                "accuracy": 0.9750,
                "precision": 0.9680,
                "recall": 0.9820,
                "f1_score": 0.9749,
                "far": 0.0240,
                "frr": 0.0180,
                "average_detection_time": 0.1520,
                "confusion_matrix": [[491, 12], [9, 488]]
            },
            "performance": {
                "cpu_usage_avg": 45.2,
                "memory_usage_avg": 512.5,
                "fps_avg": 28.5,
                "total_processing_time": 150.25
            }
        }
    }
    
    # Generate sample tables
    antispoofing_table = generator.generate_antispoofing_results_table([example_result])
    performance_table = generator.generate_performance_comparison_table([example_result])
    confusion_matrix_table = generator.generate_confusion_matrix_table(example_result['results']['antispoofing']['confusion_matrix'])
    
    # Save tables
    generator.save_latex_table(antispoofing_table, "sample_antispoofing.tex")
    generator.save_latex_table(performance_table, "sample_performance.tex")
    generator.save_latex_table(confusion_matrix_table, "sample_confusion_matrix.tex")
    
    print("Sample LaTeX tables generated successfully!")

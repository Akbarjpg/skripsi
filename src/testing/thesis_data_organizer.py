"""
Thesis Data Organizer for Chapter 4
Organizes test results according to thesis structure requirements
"""

import os
import json
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class ThesisDataOrganizer:
    """
    Organizes test data according to thesis Chapter 4 structure:
    - Section 4.1: Anti-Spoofing Test Results
    - Section 4.2: Face Recognition Test Results  
    - Section 4.3: System Performance Analysis
    - Section 4.4: Comparative Analysis
    - Section 4.5: Discussion of Results
    """
    
    def __init__(self, base_output_dir: str = "Thesis/Chapter4"):
        self.base_dir = base_output_dir
        self.sections = {
            "4.1": "Anti_Spoofing_Results",
            "4.2": "Face_Recognition_Results", 
            "4.3": "System_Performance_Analysis",
            "4.4": "Comparative_Analysis",
            "4.5": "Discussion_Results"
        }
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """Create organized directory structure for thesis chapter 4"""
        # Main chapter directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Section directories
        for section_num, section_name in self.sections.items():
            section_dir = f"{self.base_dir}/Section_{section_num}_{section_name}"
            os.makedirs(section_dir, exist_ok=True)
            
            # Subdirectories for each section
            subdirs = ["data", "tables", "figures", "analysis"]
            for subdir in subdirs:
                os.makedirs(f"{section_dir}/{subdir}", exist_ok=True)
        
        # Common directories
        os.makedirs(f"{self.base_dir}/Raw_Data", exist_ok=True)
        os.makedirs(f"{self.base_dir}/Summary_Reports", exist_ok=True)
        os.makedirs(f"{self.base_dir}/LaTeX_Tables", exist_ok=True)
        os.makedirs(f"{self.base_dir}/Figures", exist_ok=True)
    
    def organize_antispoofing_results(self, test_results: List[Dict]) -> Dict[str, str]:
        """
        Organize anti-spoofing test results for Section 4.1
        
        Returns:
            Dictionary with paths to organized files
        """
        section_dir = f"{self.base_dir}/Section_4.1_Anti_Spoofing_Results"
        organized_files = {}
        
        # Filter antispoofing results
        antispoofing_results = [r for r in test_results 
                               if r.get('results', {}).get('antispoofing')]
        
        if not antispoofing_results:
            return organized_files
        
        # 1. Raw data organization
        raw_data_file = f"{section_dir}/data/antispoofing_raw_data.json"
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(antispoofing_results, f, indent=2, ensure_ascii=False)
        organized_files['raw_data'] = raw_data_file
        
        # 2. Statistical summary
        summary_file = f"{section_dir}/analysis/antispoofing_statistical_summary.txt"
        self._generate_antispoofing_summary(antispoofing_results, summary_file)
        organized_files['statistical_summary'] = summary_file
        
        # 3. Performance metrics CSV
        metrics_file = f"{section_dir}/data/antispoofing_metrics.csv"
        self._generate_antispoofing_csv(antispoofing_results, metrics_file)
        organized_files['metrics_csv'] = metrics_file
        
        # 4. Confusion matrices
        confusion_file = f"{section_dir}/analysis/confusion_matrices_analysis.txt"
        self._analyze_confusion_matrices(antispoofing_results, confusion_file)
        organized_files['confusion_analysis'] = confusion_file
        
        return organized_files
    
    def organize_face_recognition_results(self, test_results: List[Dict]) -> Dict[str, str]:
        """
        Organize face recognition test results for Section 4.2
        
        Returns:
            Dictionary with paths to organized files
        """
        section_dir = f"{self.base_dir}/Section_4.2_Face_Recognition_Results"
        organized_files = {}
        
        # Filter face recognition results
        face_recognition_results = [r for r in test_results 
                                   if r.get('results', {}).get('face_recognition')]
        
        if not face_recognition_results:
            return organized_files
        
        # 1. Raw data organization
        raw_data_file = f"{section_dir}/data/face_recognition_raw_data.json"
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(face_recognition_results, f, indent=2, ensure_ascii=False)
        organized_files['raw_data'] = raw_data_file
        
        # 2. Accuracy analysis
        accuracy_file = f"{section_dir}/analysis/accuracy_analysis.txt"
        self._generate_face_recognition_analysis(face_recognition_results, accuracy_file)
        organized_files['accuracy_analysis'] = accuracy_file
        
        # 3. CMC curve data
        cmc_file = f"{section_dir}/data/cmc_curve_data.csv"
        self._generate_cmc_data(face_recognition_results, cmc_file)
        organized_files['cmc_data'] = cmc_file
        
        # 4. Error rate analysis
        error_file = f"{section_dir}/analysis/error_rate_analysis.txt"
        self._analyze_error_rates(face_recognition_results, error_file)
        organized_files['error_analysis'] = error_file
        
        return organized_files
    
    def organize_performance_analysis(self, test_results: List[Dict]) -> Dict[str, str]:
        """
        Organize system performance analysis for Section 4.3
        
        Returns:
            Dictionary with paths to organized files
        """
        section_dir = f"{self.base_dir}/Section_4.3_System_Performance_Analysis"
        organized_files = {}
        
        # Filter performance results
        performance_results = [r for r in test_results 
                              if r.get('results', {}).get('performance')]
        
        if not performance_results:
            return organized_files
        
        # 1. Performance metrics summary
        summary_file = f"{section_dir}/analysis/performance_summary.txt"
        self._generate_performance_summary(performance_results, summary_file)
        organized_files['performance_summary'] = summary_file
        
        # 2. Resource utilization data
        resource_file = f"{section_dir}/data/resource_utilization.csv"
        self._generate_resource_utilization_csv(performance_results, resource_file)
        organized_files['resource_data'] = resource_file
        
        # 3. Processing time analysis
        timing_file = f"{section_dir}/analysis/timing_analysis.txt"
        self._analyze_processing_times(performance_results, timing_file)
        organized_files['timing_analysis'] = timing_file
        
        # 4. Bottleneck identification
        bottleneck_file = f"{section_dir}/analysis/bottleneck_analysis.txt"
        self._identify_bottlenecks(performance_results, bottleneck_file)
        organized_files['bottleneck_analysis'] = bottleneck_file
        
        return organized_files
    
    def generate_comparative_analysis(self, test_results: List[Dict]) -> Dict[str, str]:
        """
        Generate comparative analysis for Section 4.4
        
        Returns:
            Dictionary with paths to generated files
        """
        section_dir = f"{self.base_dir}/Section_4.4_Comparative_Analysis"
        organized_files = {}
        
        # 1. Cross-system comparison
        comparison_file = f"{section_dir}/analysis/system_comparison.txt"
        self._generate_system_comparison(test_results, comparison_file)
        organized_files['system_comparison'] = comparison_file
        
        # 2. Accuracy vs Performance trade-offs
        tradeoff_file = f"{section_dir}/analysis/accuracy_performance_tradeoffs.txt"
        self._analyze_accuracy_performance_tradeoffs(test_results, tradeoff_file)
        organized_files['tradeoff_analysis'] = tradeoff_file
        
        # 3. Best practices recommendations
        recommendations_file = f"{section_dir}/analysis/recommendations.txt"
        self._generate_recommendations(test_results, recommendations_file)
        organized_files['recommendations'] = recommendations_file
        
        return organized_files
    
    def generate_discussion_materials(self, test_results: List[Dict]) -> Dict[str, str]:
        """
        Generate discussion materials for Section 4.5
        
        Returns:
            Dictionary with paths to generated files
        """
        section_dir = f"{self.base_dir}/Section_4.5_Discussion_Results"
        organized_files = {}
        
        # 1. Key findings summary
        findings_file = f"{section_dir}/analysis/key_findings.txt"
        self._generate_key_findings(test_results, findings_file)
        organized_files['key_findings'] = findings_file
        
        # 2. Limitations analysis
        limitations_file = f"{section_dir}/analysis/limitations.txt"
        self._analyze_limitations(test_results, limitations_file)
        organized_files['limitations'] = limitations_file
        
        # 3. Future work suggestions
        future_work_file = f"{section_dir}/analysis/future_work.txt"
        self._suggest_future_work(test_results, future_work_file)
        organized_files['future_work'] = future_work_file
        
        return organized_files
    
    def _generate_antispoofing_summary(self, results: List[Dict], output_file: str):
        """Generate statistical summary for antispoofing results"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ANTI-SPOOFING DETECTION RESULTS - STATISTICAL SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Extract metrics
            accuracies = [r['results']['antispoofing']['accuracy'] for r in results]
            precisions = [r['results']['antispoofing']['precision'] for r in results]
            recalls = [r['results']['antispoofing']['recall'] for r in results]
            f1_scores = [r['results']['antispoofing']['f1_score'] for r in results]
            fars = [r['results']['antispoofing']['far'] for r in results]
            frrs = [r['results']['antispoofing']['frr'] for r in results]
            detection_times = [r['results']['antispoofing']['average_detection_time'] for r in results]
            
            # Statistical analysis
            metrics = {
                'Accuracy': accuracies,
                'Precision': precisions,
                'Recall': recalls,
                'F1-Score': f1_scores,
                'FAR': fars,
                'FRR': frrs,
                'Detection Time (s)': detection_times
            }
            
            for metric_name, values in metrics.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  Mean: {np.mean(values):.4f}\n")
                f.write(f"  Std Dev: {np.std(values):.4f}\n")
                f.write(f"  Min: {np.min(values):.4f}\n")
                f.write(f"  Max: {np.max(values):.4f}\n")
                f.write(f"  Median: {np.median(values):.4f}\n\n")
            
            # Best performing test
            best_accuracy_idx = np.argmax(accuracies)
            f.write(f"Best Performing Test:\n")
            f.write(f"  Test ID: {results[best_accuracy_idx]['test_info']['test_id']}\n")
            f.write(f"  Accuracy: {accuracies[best_accuracy_idx]:.4f}\n")
            f.write(f"  FAR: {fars[best_accuracy_idx]:.4f}\n")
            f.write(f"  FRR: {frrs[best_accuracy_idx]:.4f}\n")
    
    def _generate_antispoofing_csv(self, results: List[Dict], output_file: str):
        """Generate CSV file with antispoofing metrics"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Test_ID', 'Test_Date', 'Accuracy', 'Precision', 'Recall', 
                'F1_Score', 'FAR', 'FRR', 'Detection_Time', 'Total_Samples',
                'Real_Samples', 'Fake_Samples', 'TP', 'TN', 'FP', 'FN'
            ])
            
            # Data rows
            for result in results:
                as_result = result['results']['antispoofing']
                test_info = result['test_info']
                dataset_info = test_info['dataset_info']
                confusion_matrix = as_result['confusion_matrix']
                
                writer.writerow([
                    test_info['test_id'],
                    test_info['test_date'],
                    as_result['accuracy'],
                    as_result['precision'],
                    as_result['recall'],
                    as_result['f1_score'],
                    as_result['far'],
                    as_result['frr'],
                    as_result['average_detection_time'],
                    dataset_info['total_samples'],
                    dataset_info['real_samples'],
                    dataset_info['fake_samples'],
                    confusion_matrix[0][0],  # TP
                    confusion_matrix[1][1],  # TN
                    confusion_matrix[0][1],  # FP
                    confusion_matrix[1][0]   # FN
                ])
    
    def _analyze_confusion_matrices(self, results: List[Dict], output_file: str):
        """Analyze confusion matrices for insights"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("CONFUSION MATRIX ANALYSIS\n")
            f.write("=" * 30 + "\n\n")
            
            total_tp = total_tn = total_fp = total_fn = 0
            
            for i, result in enumerate(results):
                confusion_matrix = result['results']['antispoofing']['confusion_matrix']
                tp, fp = confusion_matrix[0]
                fn, tn = confusion_matrix[1]
                
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                
                f.write(f"Test {i+1} ({result['test_info']['test_id']}):\n")
                f.write(f"  True Positives (Real correctly identified): {tp}\n")
                f.write(f"  True Negatives (Fake correctly rejected): {tn}\n")
                f.write(f"  False Positives (Fake wrongly accepted): {fp}\n")
                f.write(f"  False Negatives (Real wrongly rejected): {fn}\n")
                f.write(f"  Security Risk Score (FP rate): {fp/(fp+tn):.4f}\n")
                f.write(f"  User Experience Score (1-FN rate): {1-fn/(fn+tp):.4f}\n\n")
            
            # Overall analysis
            f.write("OVERALL ANALYSIS:\n")
            f.write(f"Total True Positives: {total_tp}\n")
            f.write(f"Total True Negatives: {total_tn}\n")
            f.write(f"Total False Positives: {total_fp}\n")
            f.write(f"Total False Negatives: {total_fn}\n")
            f.write(f"Overall Accuracy: {(total_tp + total_tn)/(total_tp + total_tn + total_fp + total_fn):.4f}\n")
            f.write(f"Overall Security Risk: {total_fp/(total_fp + total_tn):.4f}\n")
    
    def _generate_face_recognition_analysis(self, results: List[Dict], output_file: str):
        """Generate face recognition accuracy analysis"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FACE RECOGNITION ACCURACY ANALYSIS\n")
            f.write("=" * 40 + "\n\n")
            
            # Extract metrics
            rank1_accuracies = [r['results']['face_recognition']['rank1_accuracy'] for r in results]
            rank5_accuracies = [r['results']['face_recognition']['rank5_accuracy'] for r in results]
            verification_accuracies = [r['results']['face_recognition']['verification_accuracy'] for r in results]
            recognition_times = [r['results']['face_recognition']['average_recognition_time'] for r in results]
            
            f.write("RANK-1 IDENTIFICATION ACCURACY:\n")
            f.write(f"  Mean: {np.mean(rank1_accuracies):.4f} ± {np.std(rank1_accuracies):.4f}\n")
            f.write(f"  Range: {np.min(rank1_accuracies):.4f} - {np.max(rank1_accuracies):.4f}\n\n")
            
            f.write("RANK-5 IDENTIFICATION ACCURACY:\n")
            f.write(f"  Mean: {np.mean(rank5_accuracies):.4f} ± {np.std(rank5_accuracies):.4f}\n")
            f.write(f"  Range: {np.min(rank5_accuracies):.4f} - {np.max(rank5_accuracies):.4f}\n\n")
            
            f.write("VERIFICATION ACCURACY:\n")
            f.write(f"  Mean: {np.mean(verification_accuracies):.4f} ± {np.std(verification_accuracies):.4f}\n")
            f.write(f"  Range: {np.min(verification_accuracies):.4f} - {np.max(verification_accuracies):.4f}\n\n")
            
            f.write("RECOGNITION TIMING:\n")
            f.write(f"  Mean: {np.mean(recognition_times):.4f}s ± {np.std(recognition_times):.4f}s\n")
            f.write(f"  Range: {np.min(recognition_times):.4f}s - {np.max(recognition_times):.4f}s\n\n")
            
            # Performance classification
            excellent_threshold = 0.95
            good_threshold = 0.90
            
            excellent_tests = sum(1 for acc in rank1_accuracies if acc >= excellent_threshold)
            good_tests = sum(1 for acc in rank1_accuracies if good_threshold <= acc < excellent_threshold)
            
            f.write(f"PERFORMANCE CLASSIFICATION (Rank-1):\n")
            f.write(f"  Excellent (≥95%): {excellent_tests}/{len(results)} tests\n")
            f.write(f"  Good (90-95%): {good_tests}/{len(results)} tests\n")
            f.write(f"  Needs Improvement (<90%): {len(results) - excellent_tests - good_tests}/{len(results)} tests\n")
    
    def _generate_cmc_data(self, results: List[Dict], output_file: str):
        """Generate CMC (Cumulative Match Characteristic) curve data"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Test_ID', 'Rank_1', 'Rank_5', 'Verification_Accuracy', 'Recognition_Time'])
            
            # Data rows
            for result in results:
                fr_result = result['results']['face_recognition']
                writer.writerow([
                    result['test_info']['test_id'],
                    fr_result['rank1_accuracy'],
                    fr_result['rank5_accuracy'],
                    fr_result['verification_accuracy'],
                    fr_result['average_recognition_time']
                ])
    
    def _analyze_error_rates(self, results: List[Dict], output_file: str):
        """Analyze false match and false non-match rates"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ERROR RATE ANALYSIS\n")
            f.write("=" * 25 + "\n\n")
            
            fmrs = [r['results']['face_recognition']['false_match_rate'] for r in results]
            fnmrs = [r['results']['face_recognition']['false_non_match_rate'] for r in results]
            
            f.write("FALSE MATCH RATE (FMR) - Security Metric:\n")
            f.write(f"  Mean: {np.mean(fmrs):.6f}\n")
            f.write(f"  Std Dev: {np.std(fmrs):.6f}\n")
            f.write(f"  Maximum: {np.max(fmrs):.6f}\n\n")
            
            f.write("FALSE NON-MATCH RATE (FNMR) - Convenience Metric:\n")
            f.write(f"  Mean: {np.mean(fnmrs):.6f}\n")
            f.write(f"  Std Dev: {np.std(fnmrs):.6f}\n")
            f.write(f"  Maximum: {np.max(fnmrs):.6f}\n\n")
            
            # Security assessment
            high_security_fmr_threshold = 0.001  # 0.1%
            high_security_tests = sum(1 for fmr in fmrs if fmr <= high_security_fmr_threshold)
            
            f.write("SECURITY ASSESSMENT:\n")
            f.write(f"  High Security Tests (FMR ≤ 0.1%): {high_security_tests}/{len(results)}\n")
            f.write(f"  Average Security Level: {'High' if np.mean(fmrs) <= high_security_fmr_threshold else 'Medium'}\n")
    
    def _generate_performance_summary(self, results: List[Dict], output_file: str):
        """Generate system performance summary"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SYSTEM PERFORMANCE ANALYSIS\n")
            f.write("=" * 35 + "\n\n")
            
            # Extract performance metrics
            cpu_usages = [r['results']['performance']['cpu_usage_avg'] for r in results]
            memory_usages = [r['results']['performance']['memory_usage_avg'] for r in results]
            fps_values = [r['results']['performance']['fps_avg'] for r in results]
            processing_times = [r['results']['performance']['total_processing_time'] for r in results]
            
            f.write("CPU UTILIZATION:\n")
            f.write(f"  Mean: {np.mean(cpu_usages):.2f}%\n")
            f.write(f"  Peak: {np.max(cpu_usages):.2f}%\n")
            f.write(f"  Efficiency: {'High' if np.mean(cpu_usages) < 60 else 'Medium' if np.mean(cpu_usages) < 80 else 'Low'}\n\n")
            
            f.write("MEMORY UTILIZATION:\n")
            f.write(f"  Mean: {np.mean(memory_usages):.2f} MB\n")
            f.write(f"  Peak: {np.max(memory_usages):.2f} MB\n")
            f.write(f"  Stability: {'Stable' if np.std(memory_usages) < 50 else 'Variable'}\n\n")
            
            f.write("FRAME PROCESSING RATE:\n")
            f.write(f"  Mean FPS: {np.mean(fps_values):.2f}\n")
            f.write(f"  Real-time Capability: {'Yes' if np.mean(fps_values) >= 24 else 'Limited'}\n")
            f.write(f"  Consistency: {'High' if np.std(fps_values) < 2 else 'Medium'}\n\n")
            
            f.write("PROCESSING TIME:\n")
            f.write(f"  Mean: {np.mean(processing_times):.3f}s\n")
            f.write(f"  User Experience: {'Excellent' if np.mean(processing_times) < 2 else 'Good' if np.mean(processing_times) < 5 else 'Needs Improvement'}\n")
    
    def _generate_resource_utilization_csv(self, results: List[Dict], output_file: str):
        """Generate resource utilization CSV"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Test_ID', 'Test_Type', 'CPU_Usage_Avg', 'Memory_Usage_Avg', 'FPS_Avg', 'Processing_Time'])
            
            # Data rows
            for result in results:
                perf_result = result['results']['performance']
                writer.writerow([
                    result['test_info']['test_id'],
                    result['test_info']['test_type'],
                    perf_result['cpu_usage_avg'],
                    perf_result['memory_usage_avg'],
                    perf_result['fps_avg'],
                    perf_result['total_processing_time']
                ])
    
    def _analyze_processing_times(self, results: List[Dict], output_file: str):
        """Analyze processing time patterns"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PROCESSING TIME ANALYSIS\n")
            f.write("=" * 30 + "\n\n")
            
            # Group by test type
            test_types = {}
            for result in results:
                test_type = result['test_info']['test_type']
                if test_type not in test_types:
                    test_types[test_type] = []
                test_types[test_type].append(result['results']['performance']['total_processing_time'])
            
            for test_type, times in test_types.items():
                f.write(f"{test_type.upper()} PROCESSING TIMES:\n")
                f.write(f"  Mean: {np.mean(times):.3f}s\n")
                f.write(f"  Std Dev: {np.std(times):.3f}s\n")
                f.write(f"  Range: {np.min(times):.3f}s - {np.max(times):.3f}s\n")
                f.write(f"  Real-time Suitable: {'Yes' if np.mean(times) < 3 else 'No'}\n\n")
    
    def _identify_bottlenecks(self, results: List[Dict], output_file: str):
        """Identify system bottlenecks"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("BOTTLENECK IDENTIFICATION\n")
            f.write("=" * 30 + "\n\n")
            
            # Analyze resource usage patterns
            high_cpu_tests = []
            high_memory_tests = []
            low_fps_tests = []
            
            for result in results:
                perf = result['results']['performance']
                if perf['cpu_usage_avg'] > 80:
                    high_cpu_tests.append(result['test_info']['test_id'])
                if perf['memory_usage_avg'] > 1000:  # > 1GB
                    high_memory_tests.append(result['test_info']['test_id'])
                if perf['fps_avg'] < 20:
                    low_fps_tests.append(result['test_info']['test_id'])
            
            f.write("IDENTIFIED BOTTLENECKS:\n\n")
            
            if high_cpu_tests:
                f.write(f"CPU Bottleneck detected in {len(high_cpu_tests)} tests:\n")
                for test_id in high_cpu_tests:
                    f.write(f"  - {test_id}\n")
                f.write("Recommendation: Optimize algorithms, consider GPU acceleration\n\n")
            
            if high_memory_tests:
                f.write(f"Memory Bottleneck detected in {len(high_memory_tests)} tests:\n")
                for test_id in high_memory_tests:
                    f.write(f"  - {test_id}\n")
                f.write("Recommendation: Implement memory pooling, reduce model size\n\n")
            
            if low_fps_tests:
                f.write(f"Processing Speed Bottleneck detected in {len(low_fps_tests)} tests:\n")
                for test_id in low_fps_tests:
                    f.write(f"  - {test_id}\n")
                f.write("Recommendation: Optimize frame processing pipeline\n\n")
            
            if not (high_cpu_tests or high_memory_tests or low_fps_tests):
                f.write("No significant bottlenecks detected. System performance is optimal.\n")
    
    def _generate_system_comparison(self, results: List[Dict], output_file: str):
        """Generate system comparison analysis"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SYSTEM COMPARISON ANALYSIS\n")
            f.write("=" * 35 + "\n\n")
            
            # Compare different test types
            test_types = {}
            for result in results:
                test_type = result['test_info']['test_type']
                if test_type not in test_types:
                    test_types[test_type] = {'results': [], 'performance': []}
                
                test_types[test_type]['results'].append(result)
                if result.get('results', {}).get('performance'):
                    test_types[test_type]['performance'].append(result['results']['performance'])
            
            f.write("PERFORMANCE COMPARISON BY TEST TYPE:\n\n")
            
            for test_type, data in test_types.items():
                if data['performance']:
                    avg_cpu = np.mean([p['cpu_usage_avg'] for p in data['performance']])
                    avg_memory = np.mean([p['memory_usage_avg'] for p in data['performance']])
                    avg_fps = np.mean([p['fps_avg'] for p in data['performance']])
                    avg_time = np.mean([p['total_processing_time'] for p in data['performance']])
                    
                    f.write(f"{test_type.upper()}:\n")
                    f.write(f"  CPU Usage: {avg_cpu:.2f}%\n")
                    f.write(f"  Memory Usage: {avg_memory:.2f} MB\n")
                    f.write(f"  Processing Speed: {avg_fps:.2f} FPS\n")
                    f.write(f"  Total Time: {avg_time:.3f}s\n")
                    f.write(f"  Efficiency Score: {self._calculate_efficiency_score(avg_cpu, avg_memory, avg_fps):.2f}/10\n\n")
    
    def _calculate_efficiency_score(self, cpu: float, memory: float, fps: float) -> float:
        """Calculate efficiency score (0-10)"""
        cpu_score = max(0, 10 - (cpu / 10))  # Lower CPU usage is better
        memory_score = max(0, 10 - (memory / 100))  # Lower memory usage is better
        fps_score = min(10, fps / 3)  # Higher FPS is better, cap at 10
        return (cpu_score + memory_score + fps_score) / 3
    
    def _analyze_accuracy_performance_tradeoffs(self, results: List[Dict], output_file: str):
        """Analyze accuracy vs performance trade-offs"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ACCURACY VS PERFORMANCE TRADE-OFFS\n")
            f.write("=" * 40 + "\n\n")
            
            # Find tests with both accuracy and performance data
            complete_tests = []
            for result in results:
                if (result.get('results', {}).get('antispoofing') or 
                    result.get('results', {}).get('face_recognition')) and \
                   result.get('results', {}).get('performance'):
                    complete_tests.append(result)
            
            if not complete_tests:
                f.write("No tests found with both accuracy and performance data.\n")
                return
            
            f.write("TRADE-OFF ANALYSIS:\n\n")
            
            for result in complete_tests:
                test_id = result['test_info']['test_id']
                perf = result['results']['performance']
                
                f.write(f"Test: {test_id}\n")
                
                if result.get('results', {}).get('antispoofing'):
                    accuracy = result['results']['antispoofing']['accuracy']
                    f.write(f"  Anti-spoofing Accuracy: {accuracy:.4f}\n")
                
                if result.get('results', {}).get('face_recognition'):
                    accuracy = result['results']['face_recognition']['rank1_accuracy']
                    f.write(f"  Face Recognition Accuracy: {accuracy:.4f}\n")
                
                f.write(f"  CPU Usage: {perf['cpu_usage_avg']:.2f}%\n")
                f.write(f"  Processing Time: {perf['total_processing_time']:.3f}s\n")
                f.write(f"  FPS: {perf['fps_avg']:.2f}\n")
                
                # Calculate trade-off score
                if result.get('results', {}).get('antispoofing'):
                    accuracy_score = result['results']['antispoofing']['accuracy'] * 10
                elif result.get('results', {}).get('face_recognition'):
                    accuracy_score = result['results']['face_recognition']['rank1_accuracy'] * 10
                else:
                    accuracy_score = 0
                
                performance_score = self._calculate_efficiency_score(
                    perf['cpu_usage_avg'], 
                    perf['memory_usage_avg'], 
                    perf['fps_avg']
                )
                
                trade_off_score = (accuracy_score + performance_score) / 2
                f.write(f"  Trade-off Score: {trade_off_score:.2f}/10\n\n")
    
    def _generate_recommendations(self, results: List[Dict], output_file: str):
        """Generate system optimization recommendations"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SYSTEM OPTIMIZATION RECOMMENDATIONS\n")
            f.write("=" * 45 + "\n\n")
            
            # Analyze overall performance
            total_tests = len(results)
            antispoofing_tests = len([r for r in results if r.get('results', {}).get('antispoofing')])
            face_recognition_tests = len([r for r in results if r.get('results', {}).get('face_recognition')])
            
            f.write(f"Based on analysis of {total_tests} tests:\n")
            f.write(f"  - {antispoofing_tests} anti-spoofing tests\n")
            f.write(f"  - {face_recognition_tests} face recognition tests\n\n")
            
            f.write("RECOMMENDATIONS:\n\n")
            
            # Performance recommendations
            performance_results = [r for r in results if r.get('results', {}).get('performance')]
            if performance_results:
                avg_cpu = np.mean([r['results']['performance']['cpu_usage_avg'] for r in performance_results])
                avg_fps = np.mean([r['results']['performance']['fps_avg'] for r in performance_results])
                
                if avg_cpu > 70:
                    f.write("1. HIGH CPU USAGE DETECTED:\n")
                    f.write("   - Implement model quantization to reduce computational load\n")
                    f.write("   - Consider GPU acceleration for CNN inference\n")
                    f.write("   - Optimize frame processing pipeline\n\n")
                
                if avg_fps < 25:
                    f.write("2. LOW FRAME RATE DETECTED:\n")
                    f.write("   - Implement frame skipping during processing\n")
                    f.write("   - Use multithreading for parallel processing\n")
                    f.write("   - Reduce input image resolution\n\n")
            
            # Accuracy recommendations
            antispoofing_results = [r for r in results if r.get('results', {}).get('antispoofing')]
            if antispoofing_results:
                avg_accuracy = np.mean([r['results']['antispoofing']['accuracy'] for r in antispoofing_results])
                avg_far = np.mean([r['results']['antispoofing']['far'] for r in antispoofing_results])
                
                if avg_accuracy < 0.95:
                    f.write("3. ANTI-SPOOFING ACCURACY IMPROVEMENT:\n")
                    f.write("   - Collect more diverse training data\n")
                    f.write("   - Implement ensemble methods\n")
                    f.write("   - Add temporal consistency checks\n\n")
                
                if avg_far > 0.01:
                    f.write("4. SECURITY ENHANCEMENT:\n")
                    f.write("   - Increase anti-spoofing detection threshold\n")
                    f.write("   - Add additional liveness detection methods\n")
                    f.write("   - Implement multi-modal verification\n\n")
            
            f.write("5. GENERAL OPTIMIZATIONS:\n")
            f.write("   - Implement adaptive quality settings based on hardware\n")
            f.write("   - Add progressive loading for better user experience\n")
            f.write("   - Implement caching mechanisms for frequently accessed data\n")
            f.write("   - Add comprehensive error handling and recovery\n")
    
    def _generate_key_findings(self, results: List[Dict], output_file: str):
        """Generate key findings summary"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("KEY FINDINGS SUMMARY\n")
            f.write("=" * 25 + "\n\n")
            
            f.write("1. ANTI-SPOOFING PERFORMANCE:\n")
            antispoofing_results = [r for r in results if r.get('results', {}).get('antispoofing')]
            if antispoofing_results:
                avg_accuracy = np.mean([r['results']['antispoofing']['accuracy'] for r in antispoofing_results])
                avg_far = np.mean([r['results']['antispoofing']['far'] for r in antispoofing_results])
                f.write(f"   - Average detection accuracy: {avg_accuracy:.2%}\n")
                f.write(f"   - False acceptance rate: {avg_far:.4f}\n")
                f.write(f"   - System security level: {'High' if avg_far < 0.01 else 'Medium'}\n\n")
            
            f.write("2. FACE RECOGNITION PERFORMANCE:\n")
            face_recognition_results = [r for r in results if r.get('results', {}).get('face_recognition')]
            if face_recognition_results:
                avg_rank1 = np.mean([r['results']['face_recognition']['rank1_accuracy'] for r in face_recognition_results])
                f.write(f"   - Average identification accuracy: {avg_rank1:.2%}\n")
                f.write(f"   - Suitable for attendance system: {'Yes' if avg_rank1 > 0.90 else 'Needs improvement'}\n\n")
            
            f.write("3. SYSTEM PERFORMANCE:\n")
            performance_results = [r for r in results if r.get('results', {}).get('performance')]
            if performance_results:
                avg_fps = np.mean([r['results']['performance']['fps_avg'] for r in performance_results])
                avg_time = np.mean([r['results']['performance']['total_processing_time'] for r in performance_results])
                f.write(f"   - Real-time processing capability: {'Yes' if avg_fps >= 24 else 'Limited'}\n")
                f.write(f"   - User experience quality: {'Good' if avg_time < 3 else 'Needs optimization'}\n\n")
            
            f.write("4. OVERALL SYSTEM READINESS:\n")
            f.write(f"   - Production readiness: {'Ready' if len(results) >= 10 else 'Needs more testing'}\n")
            f.write(f"   - Deployment recommendation: {'Recommended' if len(results) >= 10 else 'Continue testing'}\n")
    
    def _analyze_limitations(self, results: List[Dict], output_file: str):
        """Analyze system limitations"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SYSTEM LIMITATIONS ANALYSIS\n")
            f.write("=" * 35 + "\n\n")
            
            f.write("IDENTIFIED LIMITATIONS:\n\n")
            
            f.write("1. HARDWARE DEPENDENCIES:\n")
            f.write("   - High CPU usage may limit deployment on low-end devices\n")
            f.write("   - Memory requirements may exceed mobile device capabilities\n")
            f.write("   - Camera quality affects detection accuracy\n\n")
            
            f.write("2. ENVIRONMENTAL CONSTRAINTS:\n")
            f.write("   - Lighting conditions impact performance\n")
            f.write("   - Background clutter may interfere with face detection\n")
            f.write("   - Network latency affects real-time processing\n\n")
            
            f.write("3. SCALABILITY CONCERNS:\n")
            f.write("   - Processing time increases with database size\n")
            f.write("   - Memory usage scales with number of registered users\n")
            f.write("   - Concurrent user handling needs optimization\n\n")
            
            f.write("4. ACCURACY LIMITATIONS:\n")
            # Analyze actual test results for limitations
            antispoofing_results = [r for r in results if r.get('results', {}).get('antispoofing')]
            if antispoofing_results:
                min_accuracy = min([r['results']['antispoofing']['accuracy'] for r in antispoofing_results])
                max_far = max([r['results']['antispoofing']['far'] for r in antispoofing_results])
                f.write(f"   - Lowest anti-spoofing accuracy: {min_accuracy:.2%}\n")
                f.write(f"   - Highest false acceptance rate: {max_far:.4f}\n")
            
            face_recognition_results = [r for r in results if r.get('results', {}).get('face_recognition')]
            if face_recognition_results:
                min_rank1 = min([r['results']['face_recognition']['rank1_accuracy'] for r in face_recognition_results])
                f.write(f"   - Lowest face recognition accuracy: {min_rank1:.2%}\n")
    
    def _suggest_future_work(self, results: List[Dict], output_file: str):
        """Suggest future work directions"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FUTURE WORK SUGGESTIONS\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("IMMEDIATE IMPROVEMENTS:\n\n")
            
            f.write("1. ALGORITHM ENHANCEMENTS:\n")
            f.write("   - Implement attention mechanisms in CNN models\n")
            f.write("   - Explore transformer-based architectures\n")
            f.write("   - Add federated learning capabilities\n")
            f.write("   - Implement continual learning for adaptation\n\n")
            
            f.write("2. MULTI-MODAL INTEGRATION:\n")
            f.write("   - Add voice recognition for enhanced security\n")
            f.write("   - Integrate gait analysis for identification\n")
            f.write("   - Implement behavioral biometrics\n")
            f.write("   - Add iris recognition as backup method\n\n")
            
            f.write("3. EDGE COMPUTING OPTIMIZATION:\n")
            f.write("   - Develop lightweight model variants\n")
            f.write("   - Implement edge-cloud hybrid processing\n")
            f.write("   - Add model compression techniques\n")
            f.write("   - Optimize for ARM processors\n\n")
            
            f.write("LONG-TERM RESEARCH DIRECTIONS:\n\n")
            
            f.write("1. PRIVACY-PRESERVING METHODS:\n")
            f.write("   - Implement homomorphic encryption\n")
            f.write("   - Add differential privacy mechanisms\n")
            f.write("   - Develop secure multi-party computation\n")
            f.write("   - Create template protection schemes\n\n")
            
            f.write("2. ADVERSARIAL ROBUSTNESS:\n")
            f.write("   - Defend against adversarial attacks\n")
            f.write("   - Improve deepfake detection capabilities\n")
            f.write("   - Add uncertainty quantification\n")
            f.write("   - Implement robust training methods\n\n")
            
            f.write("3. EXPLAINABLE AI:\n")
            f.write("   - Add decision explanation mechanisms\n")
            f.write("   - Implement attention visualization\n")
            f.write("   - Create interpretable model variants\n")
            f.write("   - Add confidence calibration\n")
    
    def organize_complete_thesis_data(self, test_results_json_files: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Organize all test data for complete thesis Chapter 4
        
        Args:
            test_results_json_files: List of paths to JSON test result files
            
        Returns:
            Dictionary with organized files for each section
        """
        # Load all test results
        all_results = []
        for json_file in test_results_json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                all_results.append(json.load(f))
        
        organized_data = {}
        
        # Section 4.1: Anti-Spoofing Test Results
        organized_data['Section_4.1'] = self.organize_antispoofing_results(all_results)
        
        # Section 4.2: Face Recognition Test Results
        organized_data['Section_4.2'] = self.organize_face_recognition_results(all_results)
        
        # Section 4.3: System Performance Analysis
        organized_data['Section_4.3'] = self.organize_performance_analysis(all_results)
        
        # Section 4.4: Comparative Analysis
        organized_data['Section_4.4'] = self.generate_comparative_analysis(all_results)
        
        # Section 4.5: Discussion of Results
        organized_data['Section_4.5'] = self.generate_discussion_materials(all_results)
        
        # Generate master summary
        summary_file = f"{self.base_dir}/THESIS_CHAPTER4_SUMMARY.txt"
        self._generate_master_summary(all_results, summary_file)
        organized_data['master_summary'] = summary_file
        
        return organized_data
    
    def _generate_master_summary(self, results: List[Dict], output_file: str):
        """Generate master summary for entire Chapter 4"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("THESIS CHAPTER 4 - MASTER SUMMARY\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total test results analyzed: {len(results)}\n\n")
            
            f.write("DIRECTORY STRUCTURE:\n")
            for section_num, section_name in self.sections.items():
                f.write(f"  Section_{section_num}_{section_name}/\n")
                f.write(f"    ├── data/           (Raw data and CSV files)\n")
                f.write(f"    ├── tables/         (LaTeX tables)\n")
                f.write(f"    ├── figures/        (Graphs and visualizations)\n")
                f.write(f"    └── analysis/       (Statistical analysis)\n\n")
            
            f.write("DATA ORGANIZATION COMPLETE.\n")
            f.write("All files are ready for thesis documentation.\n")

# Example usage
if __name__ == "__main__":
    organizer = ThesisDataOrganizer()
    
    # Example: Organize test results (replace with actual data)
    example_results = [{
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
    }]
    
    # Organize all data
    organized_files = organizer.organize_complete_thesis_data(["test_results/json/example.json"])
    
    print("Thesis data organization complete!")
    print("Organized files:")
    for section, files in organized_files.items():
        print(f"  {section}: {len(files) if isinstance(files, dict) else 1} files")

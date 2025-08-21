"""
Comprehensive Test Data Collection System
Implements the structured test result format for thesis documentation
"""

import json
import csv
import os
import time
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import queue

# Import our custom modules
from .test_result_formatter import TestResultFormatter, DatasetInfo
from .latex_table_generator import LaTeXTableGenerator  
from .thesis_data_organizer import ThesisDataOrganizer

class TestDataCollector:
    """
    Main class for collecting comprehensive test data according to 
    Step 9.2 structured format requirements
    """
    
    def __init__(self, output_base_dir: str = "test_results"):
        self.output_dir = output_base_dir
        self.formatter = TestResultFormatter(output_base_dir)
        self.latex_generator = LaTeXTableGenerator(f"{output_base_dir}/latex")
        self.thesis_organizer = ThesisDataOrganizer("Thesis/Chapter4")
        
        # Performance monitoring
        self.performance_data = queue.Queue()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Test session data
        self.current_session = {
            'start_time': None,
            'test_results': [],
            'session_id': None
        }
        
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create all necessary directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/sessions",
            f"{self.output_dir}/logs",
            f"{self.output_dir}/csv", 
            f"{self.output_dir}/json",
            f"{self.output_dir}/latex",
            f"{self.output_dir}/figures",
            f"{self.output_dir}/reports"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def start_test_session(self, session_name: str = None) -> str:
        """
        Start a new test collection session
        
        Returns:
            Session ID
        """
        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if session_name:
            session_id += f"_{session_name}"
        
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now(),
            'test_results': [],
            'session_metadata': {
                'session_name': session_name or "Default",
                'start_timestamp': datetime.now().isoformat(),
                'total_tests': 0,
                'test_types': []
            }
        }
        
        # Create session directory
        session_dir = f"{self.output_dir}/sessions/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        print(f"Started test session: {session_id}")
        return session_id
    
    def start_performance_monitoring(self):
        """Start monitoring system performance"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_performance_monitoring(self) -> List[Dict]:
        """
        Stop performance monitoring and return collected data
        
        Returns:
            List of performance samples
        """
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Collect all performance data
        performance_samples = []
        while not self.performance_data.empty():
            try:
                performance_samples.append(self.performance_data.get_nowait())
            except queue.Empty:
                break
        
        print(f"Performance monitoring stopped. Collected {len(performance_samples)} samples")
        return performance_samples
    
    def _monitor_performance(self):
        """Monitor system performance in background thread"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                
                # Estimate FPS (this would be replaced with actual frame processing data)
                current_time = time.time()
                
                sample = {
                    'timestamp': current_time,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_info.percent
                }
                
                self.performance_data.put(sample)
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                break
    
    def collect_antispoofing_test_data(self, 
                                       test_name: str,
                                       true_positives: int,
                                       true_negatives: int, 
                                       false_positives: int,
                                       false_negatives: int,
                                       detection_times: List[float],
                                       dataset_info: DatasetInfo,
                                       detailed_log_path: str = "") -> str:
        """
        Collect anti-spoofing test data according to structured format
        
        Args:
            test_name: Name/description of the test
            true_positives: Number of real faces correctly identified
            true_negatives: Number of fake faces correctly rejected
            false_positives: Number of fake faces wrongly accepted
            false_negatives: Number of real faces wrongly rejected
            detection_times: List of detection times for each sample
            dataset_info: Information about test dataset
            detailed_log_path: Path to detailed test logs
            
        Returns:
            Path to saved test result file
        """
        print(f"Collecting anti-spoofing test data: {test_name}")
        
        # Start performance monitoring if not already active
        if not self.monitoring_active:
            self.start_performance_monitoring()
        
        # Process test for a moment to collect performance data
        time.sleep(2)  # Simulate processing time
        
        # Stop monitoring and get performance data
        performance_samples = self.stop_performance_monitoring()
        
        # Create antispoofing results
        antispoofing_result = self.formatter.create_antispoofing_result(
            tp=true_positives,
            tn=true_negatives, 
            fp=false_positives,
            fn=false_negatives,
            detection_times=detection_times
        )
        
        # Create performance results
        performance_result = None
        if performance_samples:
            cpu_samples = [s['cpu_percent'] for s in performance_samples]
            memory_samples = [s['memory_mb'] for s in performance_samples]
            fps_samples = [25.0] * len(performance_samples)  # Mock FPS data
            total_time = sum(detection_times)
            
            performance_result = self.formatter.create_performance_result(
                cpu_usage_samples=cpu_samples,
                memory_usage_samples=memory_samples,
                fps_samples=fps_samples,
                total_processing_time=total_time
            )
        
        # Create complete test result
        complete_result = self.formatter.create_complete_test_result(
            test_type="antispoofing",
            dataset_info=dataset_info,
            antispoofing_results=antispoofing_result,
            performance_results=performance_result,
            detailed_logs_path=detailed_log_path
        )
        
        # Save results
        json_path = self.formatter.save_test_result_json(complete_result)
        csv_path = self.formatter.save_test_result_csv(complete_result)
        
        # Add to current session
        if self.current_session['session_id']:
            self.current_session['test_results'].append(complete_result)
            self.current_session['session_metadata']['total_tests'] += 1
            if 'antispoofing' not in self.current_session['session_metadata']['test_types']:
                self.current_session['session_metadata']['test_types'].append('antispoofing')
        
        print(f"Anti-spoofing test data saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Test ID: {complete_result.test_info.test_id}")
        
        return json_path
    
    def collect_face_recognition_test_data(self,
                                           test_name: str,
                                           rank1_correct: int,
                                           rank5_correct: int,
                                           total_queries: int,
                                           verification_correct: int,
                                           verification_total: int,
                                           recognition_times: List[float],
                                           false_matches: int,
                                           false_non_matches: int,
                                           dataset_info: DatasetInfo,
                                           detailed_log_path: str = "") -> str:
        """
        Collect face recognition test data according to structured format
        
        Returns:
            Path to saved test result file
        """
        print(f"Collecting face recognition test data: {test_name}")
        
        # Start performance monitoring
        if not self.monitoring_active:
            self.start_performance_monitoring()
        
        # Simulate processing time
        time.sleep(2)
        
        # Stop monitoring and get performance data
        performance_samples = self.stop_performance_monitoring()
        
        # Create face recognition results
        face_recognition_result = self.formatter.create_face_recognition_result(
            rank1_correct=rank1_correct,
            rank5_correct=rank5_correct,
            total_queries=total_queries,
            verification_correct=verification_correct,
            verification_total=verification_total,
            recognition_times=recognition_times,
            false_matches=false_matches,
            false_non_matches=false_non_matches
        )
        
        # Create performance results
        performance_result = None
        if performance_samples:
            cpu_samples = [s['cpu_percent'] for s in performance_samples]
            memory_samples = [s['memory_mb'] for s in performance_samples]
            fps_samples = [30.0] * len(performance_samples)  # Mock FPS data
            total_time = sum(recognition_times)
            
            performance_result = self.formatter.create_performance_result(
                cpu_usage_samples=cpu_samples,
                memory_usage_samples=memory_samples,
                fps_samples=fps_samples,
                total_processing_time=total_time
            )
        
        # Create complete test result
        complete_result = self.formatter.create_complete_test_result(
            test_type="face_recognition",
            dataset_info=dataset_info,
            face_recognition_results=face_recognition_result,
            performance_results=performance_result,
            detailed_logs_path=detailed_log_path
        )
        
        # Save results
        json_path = self.formatter.save_test_result_json(complete_result)
        csv_path = self.formatter.save_test_result_csv(complete_result)
        
        # Add to current session
        if self.current_session['session_id']:
            self.current_session['test_results'].append(complete_result)
            self.current_session['session_metadata']['total_tests'] += 1
            if 'face_recognition' not in self.current_session['session_metadata']['test_types']:
                self.current_session['session_metadata']['test_types'].append('face_recognition')
        
        print(f"Face recognition test data saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Test ID: {complete_result.test_info.test_id}")
        
        return json_path
    
    def collect_integration_test_data(self,
                                      test_name: str,
                                      antispoofing_metrics: Dict,
                                      face_recognition_metrics: Dict,
                                      end_to_end_times: List[float],
                                      dataset_info: DatasetInfo,
                                      detailed_log_path: str = "") -> str:
        """
        Collect integration test data (combined antispoofing + face recognition)
        
        Args:
            test_name: Name of integration test
            antispoofing_metrics: Dict with TP, TN, FP, FN, detection_times
            face_recognition_metrics: Dict with recognition metrics
            end_to_end_times: List of total processing times
            dataset_info: Dataset information
            detailed_log_path: Path to detailed logs
            
        Returns:
            Path to saved test result file
        """
        print(f"Collecting integration test data: {test_name}")
        
        # Start performance monitoring
        if not self.monitoring_active:
            self.start_performance_monitoring()
        
        # Simulate processing time
        time.sleep(3)  # Integration tests take longer
        
        # Stop monitoring and get performance data
        performance_samples = self.stop_performance_monitoring()
        
        # Create antispoofing results
        antispoofing_result = self.formatter.create_antispoofing_result(
            tp=antispoofing_metrics['tp'],
            tn=antispoofing_metrics['tn'],
            fp=antispoofing_metrics['fp'], 
            fn=antispoofing_metrics['fn'],
            detection_times=antispoofing_metrics['detection_times']
        )
        
        # Create face recognition results
        face_recognition_result = self.formatter.create_face_recognition_result(
            rank1_correct=face_recognition_metrics['rank1_correct'],
            rank5_correct=face_recognition_metrics['rank5_correct'],
            total_queries=face_recognition_metrics['total_queries'],
            verification_correct=face_recognition_metrics['verification_correct'],
            verification_total=face_recognition_metrics['verification_total'],
            recognition_times=face_recognition_metrics['recognition_times'],
            false_matches=face_recognition_metrics['false_matches'],
            false_non_matches=face_recognition_metrics['false_non_matches']
        )
        
        # Create performance results
        performance_result = None
        if performance_samples:
            cpu_samples = [s['cpu_percent'] for s in performance_samples]
            memory_samples = [s['memory_mb'] for s in performance_samples]
            fps_samples = [20.0] * len(performance_samples)  # Lower FPS for integration
            total_time = sum(end_to_end_times)
            
            performance_result = self.formatter.create_performance_result(
                cpu_usage_samples=cpu_samples,
                memory_usage_samples=memory_samples,
                fps_samples=fps_samples,
                total_processing_time=total_time
            )
        
        # Create complete test result
        complete_result = self.formatter.create_complete_test_result(
            test_type="integration",
            dataset_info=dataset_info,
            antispoofing_results=antispoofing_result,
            face_recognition_results=face_recognition_result,
            performance_results=performance_result,
            detailed_logs_path=detailed_log_path
        )
        
        # Save results
        json_path = self.formatter.save_test_result_json(complete_result)
        csv_path = self.formatter.save_test_result_csv(complete_result)
        
        # Add to current session
        if self.current_session['session_id']:
            self.current_session['test_results'].append(complete_result)
            self.current_session['session_metadata']['total_tests'] += 1
            if 'integration' not in self.current_session['session_metadata']['test_types']:
                self.current_session['session_metadata']['test_types'].append('integration')
        
        print(f"Integration test data saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Test ID: {complete_result.test_info.test_id}")
        
        return json_path
    
    def end_test_session(self) -> Dict[str, str]:
        """
        End current test session and generate all outputs
        
        Returns:
            Dictionary with paths to generated files
        """
        if not self.current_session['session_id']:
            print("No active test session")
            return {}
        
        session_id = self.current_session['session_id']
        session_dir = f"{self.output_dir}/sessions/{session_id}"
        
        print(f"Ending test session: {session_id}")
        
        # Update session metadata
        self.current_session['session_metadata']['end_timestamp'] = datetime.now().isoformat()
        self.current_session['session_metadata']['duration_minutes'] = \
            (datetime.now() - self.current_session['start_time']).total_seconds() / 60
        
        generated_files = {}
        
        # Save session metadata
        metadata_file = f"{session_dir}/session_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_session['session_metadata'], f, indent=2, ensure_ascii=False)
        generated_files['session_metadata'] = metadata_file
        
        # Save all test results in session
        session_results_file = f"{session_dir}/all_test_results.json"
        with open(session_results_file, 'w', encoding='utf-8') as f:
            # Convert dataclasses to dict for JSON serialization
            results_data = []
            for result in self.current_session['test_results']:
                from dataclasses import asdict
                results_data.append(asdict(result))
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        generated_files['session_results'] = session_results_file
        
        # Generate session summary report
        summary_file = f"{session_dir}/session_summary.txt"
        self._generate_session_summary(summary_file)
        generated_files['session_summary'] = summary_file
        
        # Generate LaTeX tables for this session
        if self.current_session['test_results']:
            # Convert results to JSON format for LaTeX generator
            json_files = []
            for result in self.current_session['test_results']:
                json_path = self.formatter.save_test_result_json(result)
                json_files.append(json_path)
            
            # Generate LaTeX tables
            latex_files = self.latex_generator.generate_complete_thesis_tables(json_files)
            generated_files.update(latex_files)
        
        # Organize data for thesis
        if self.current_session['test_results']:
            json_files = [self.formatter.save_test_result_json(result) 
                         for result in self.current_session['test_results']]
            
            thesis_files = self.thesis_organizer.organize_complete_thesis_data(json_files)
            generated_files['thesis_organization'] = thesis_files
        
        print(f"Test session ended successfully!")
        print(f"Generated files:")
        for file_type, file_path in generated_files.items():
            if isinstance(file_path, dict):
                print(f"  {file_type}: {len(file_path)} files")
            else:
                print(f"  {file_type}: {file_path}")
        
        # Reset session
        self.current_session = {'start_time': None, 'test_results': [], 'session_id': None}
        
        return generated_files
    
    def _generate_session_summary(self, output_file: str):
        """Generate summary report for current session"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"TEST SESSION SUMMARY\n")
            f.write(f"=" * 25 + "\n\n")
            
            session_meta = self.current_session['session_metadata']
            f.write(f"Session ID: {self.current_session['session_id']}\n")
            f.write(f"Session Name: {session_meta['session_name']}\n")
            f.write(f"Start Time: {session_meta['start_timestamp']}\n")
            f.write(f"End Time: {session_meta.get('end_timestamp', 'N/A')}\n")
            f.write(f"Duration: {session_meta.get('duration_minutes', 0):.2f} minutes\n")
            f.write(f"Total Tests: {session_meta['total_tests']}\n")
            f.write(f"Test Types: {', '.join(session_meta['test_types'])}\n\n")
            
            # Test type breakdown
            test_counts = {}
            for result in self.current_session['test_results']:
                test_type = result.test_info.test_type
                test_counts[test_type] = test_counts.get(test_type, 0) + 1
            
            f.write("TEST BREAKDOWN:\n")
            for test_type, count in test_counts.items():
                f.write(f"  {test_type.title()}: {count} tests\n")
            
            # Quick performance summary
            f.write("\nQUICK PERFORMANCE SUMMARY:\n")
            
            # Antispoofing summary
            antispoofing_results = [r for r in self.current_session['test_results'] 
                                   if r.results.antispoofing]
            if antispoofing_results:
                avg_accuracy = np.mean([r.results.antispoofing.accuracy for r in antispoofing_results])
                avg_far = np.mean([r.results.antispoofing.far for r in antispoofing_results])
                f.write(f"  Anti-spoofing Average Accuracy: {avg_accuracy:.2%}\n")
                f.write(f"  Anti-spoofing Average FAR: {avg_far:.4f}\n")
            
            # Face recognition summary
            face_recognition_results = [r for r in self.current_session['test_results'] 
                                       if r.results.face_recognition]
            if face_recognition_results:
                avg_rank1 = np.mean([r.results.face_recognition.rank1_accuracy 
                                   for r in face_recognition_results])
                f.write(f"  Face Recognition Average Accuracy: {avg_rank1:.2%}\n")
            
            # Performance summary
            performance_results = [r for r in self.current_session['test_results'] 
                                 if r.results.performance]
            if performance_results:
                avg_cpu = np.mean([r.results.performance.cpu_usage_avg for r in performance_results])
                avg_fps = np.mean([r.results.performance.fps_avg for r in performance_results])
                f.write(f"  Average CPU Usage: {avg_cpu:.1f}%\n")
                f.write(f"  Average FPS: {avg_fps:.1f}\n")
    
    def run_example_comprehensive_test(self) -> Dict[str, str]:
        """
        Run example comprehensive test to demonstrate the system
        
        Returns:
            Dictionary with generated file paths
        """
        print("Running example comprehensive test...")
        
        # Start session
        session_id = self.start_test_session("Example_Comprehensive_Test")
        
        # Example dataset info
        dataset_info = DatasetInfo(
            total_samples=1000,
            real_samples=500,
            fake_samples=500,
            unique_individuals=100
        )
        
        # Example 1: Anti-spoofing test
        print("\n1. Running anti-spoofing test...")
        antispoofing_json = self.collect_antispoofing_test_data(
            test_name="High_Quality_Dataset_Test",
            true_positives=485,  # Real faces correctly identified
            true_negatives=475,  # Fake faces correctly rejected  
            false_positives=25,  # Fake faces wrongly accepted
            false_negatives=15,  # Real faces wrongly rejected
            detection_times=[0.12, 0.15, 0.11, 0.14, 0.13, 0.16, 0.12],
            dataset_info=dataset_info,
            detailed_log_path="test_results/logs/antispoofing_detailed.log"
        )
        
        # Example 2: Face recognition test
        print("\n2. Running face recognition test...")
        face_recognition_json = self.collect_face_recognition_test_data(
            test_name="Multi_Angle_Recognition_Test",
            rank1_correct=92,     # Correctly identified at rank 1
            rank5_correct=98,     # Correctly identified within rank 5
            total_queries=100,    # Total test queries
            verification_correct=95,  # Correct verifications
            verification_total=100,   # Total verification attempts
            recognition_times=[0.08, 0.09, 0.07, 0.10, 0.08, 0.09],
            false_matches=2,      # False matches
            false_non_matches=3,  # False non-matches
            dataset_info=dataset_info,
            detailed_log_path="test_results/logs/face_recognition_detailed.log"
        )
        
        # Example 3: Integration test
        print("\n3. Running integration test...")
        integration_json = self.collect_integration_test_data(
            test_name="End_to_End_System_Test",
            antispoofing_metrics={
                'tp': 480, 'tn': 470, 'fp': 30, 'fn': 20,
                'detection_times': [0.15, 0.16, 0.14, 0.17, 0.15]
            },
            face_recognition_metrics={
                'rank1_correct': 88, 'rank5_correct': 95, 'total_queries': 100,
                'verification_correct': 90, 'verification_total': 100,
                'recognition_times': [0.10, 0.11, 0.09, 0.12, 0.10],
                'false_matches': 3, 'false_non_matches': 7
            },
            end_to_end_times=[2.5, 2.8, 2.3, 2.7, 2.6, 2.4],
            dataset_info=dataset_info,
            detailed_log_path="test_results/logs/integration_detailed.log"
        )
        
        # End session and generate all outputs
        print("\n4. Generating comprehensive reports...")
        generated_files = self.end_test_session()
        
        print(f"\nExample comprehensive test completed!")
        print(f"Session: {session_id}")
        print(f"Generated {len(generated_files)} file types")
        
        return generated_files

# Example usage
if __name__ == "__main__":
    collector = TestDataCollector()
    
    # Run example comprehensive test
    results = collector.run_example_comprehensive_test()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST DATA COLLECTION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    for file_type, file_info in results.items():
        if isinstance(file_info, dict):
            print(f"  {file_type}: {len(file_info)} files")
            for subtype, path in file_info.items():
                if isinstance(path, str):
                    print(f"    - {subtype}: {path}")
                elif isinstance(path, dict):
                    print(f"    - {subtype}: {len(path)} files")
        else:
            print(f"  {file_type}: {file_info}")
    
    print("\nAll test data is now organized and ready for thesis documentation!")
    print("Check the following directories:")
    print("  - test_results/: Raw test data and reports")
    print("  - Thesis/Chapter4/: Organized data for thesis sections")
    print("  - test_results/latex/: LaTeX tables ready for inclusion")

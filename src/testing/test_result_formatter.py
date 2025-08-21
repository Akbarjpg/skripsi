"""
Test Result Formatter for Thesis Documentation
Implements structured output format for comprehensive test data collection
"""

import json
import datetime
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class DatasetInfo:
    """Dataset information structure"""
    total_samples: int
    real_samples: int
    fake_samples: int
    unique_individuals: int

@dataclass
class AntispoofingResults:
    """Anti-spoofing test results structure"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    average_detection_time: float
    confusion_matrix: List[List[int]]  # [[TP, FP], [FN, TN]]

@dataclass
class FaceRecognitionResults:
    """Face recognition test results structure"""
    rank1_accuracy: float
    rank5_accuracy: float
    verification_accuracy: float
    average_recognition_time: float
    false_match_rate: float
    false_non_match_rate: float

@dataclass
class PerformanceResults:
    """System performance metrics structure"""
    cpu_usage_avg: float
    memory_usage_avg: float
    fps_avg: float
    total_processing_time: float

@dataclass
class TestInfo:
    """Test information metadata"""
    test_id: str
    test_date: str
    test_type: str  # "antispoofing|face_recognition|integration"
    dataset_info: DatasetInfo

@dataclass
class TestResults:
    """Combined test results structure"""
    antispoofing: Optional[AntispoofingResults] = None
    face_recognition: Optional[FaceRecognitionResults] = None
    performance: Optional[PerformanceResults] = None

@dataclass
class CompleteTestResult:
    """Complete test result structure for thesis documentation"""
    test_info: TestInfo
    results: TestResults
    detailed_logs: str

class TestResultFormatter:
    """
    Formats test results according to standardized structure for thesis documentation
    """
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        self.ensure_output_directory()
    
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/json", exist_ok=True)
        os.makedirs(f"{self.output_dir}/latex", exist_ok=True)
        os.makedirs(f"{self.output_dir}/csv", exist_ok=True)
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
    
    def generate_test_id(self) -> str:
        """Generate unique test identifier"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"TEST_{timestamp}_{unique_id}"
    
    def create_antispoofing_result(self, 
                                   tp: int, tn: int, fp: int, fn: int,
                                   detection_times: List[float]) -> AntispoofingResults:
        """
        Create antispoofing results from confusion matrix values
        
        Args:
            tp: True Positives (real faces correctly identified)
            tn: True Negatives (fake faces correctly rejected)
            fp: False Positives (fake faces wrongly accepted)
            fn: False Negatives (real faces wrongly rejected)
            detection_times: List of detection times for each sample
        """
        total = tp + tn + fp + fn
        
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Rejection Rate
        avg_detection_time = np.mean(detection_times) if detection_times else 0.0
        
        return AntispoofingResults(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1_score, 4),
            far=round(far, 4),
            frr=round(frr, 4),
            average_detection_time=round(avg_detection_time, 4),
            confusion_matrix=[[tp, fp], [fn, tn]]
        )
    
    def create_face_recognition_result(self,
                                       rank1_correct: int,
                                       rank5_correct: int,
                                       total_queries: int,
                                       verification_correct: int,
                                       verification_total: int,
                                       recognition_times: List[float],
                                       false_matches: int,
                                       false_non_matches: int) -> FaceRecognitionResults:
        """
        Create face recognition results from test data
        
        Args:
            rank1_correct: Number of correct rank-1 identifications
            rank5_correct: Number of correct rank-5 identifications
            total_queries: Total number of query images
            verification_correct: Number of correct verifications
            verification_total: Total number of verification attempts
            recognition_times: List of recognition times
            false_matches: Number of false matches
            false_non_matches: Number of false non-matches
        """
        rank1_accuracy = rank1_correct / total_queries if total_queries > 0 else 0.0
        rank5_accuracy = rank5_correct / total_queries if total_queries > 0 else 0.0
        verification_accuracy = verification_correct / verification_total if verification_total > 0 else 0.0
        avg_recognition_time = np.mean(recognition_times) if recognition_times else 0.0
        
        total_attempts = false_matches + false_non_matches + verification_correct
        false_match_rate = false_matches / total_attempts if total_attempts > 0 else 0.0
        false_non_match_rate = false_non_matches / total_attempts if total_attempts > 0 else 0.0
        
        return FaceRecognitionResults(
            rank1_accuracy=round(rank1_accuracy, 4),
            rank5_accuracy=round(rank5_accuracy, 4),
            verification_accuracy=round(verification_accuracy, 4),
            average_recognition_time=round(avg_recognition_time, 4),
            false_match_rate=round(false_match_rate, 4),
            false_non_match_rate=round(false_non_match_rate, 4)
        )
    
    def create_performance_result(self,
                                  cpu_usage_samples: List[float],
                                  memory_usage_samples: List[float],
                                  fps_samples: List[float],
                                  total_processing_time: float) -> PerformanceResults:
        """
        Create performance results from monitoring data
        """
        return PerformanceResults(
            cpu_usage_avg=round(np.mean(cpu_usage_samples), 2) if cpu_usage_samples else 0.0,
            memory_usage_avg=round(np.mean(memory_usage_samples), 2) if memory_usage_samples else 0.0,
            fps_avg=round(np.mean(fps_samples), 2) if fps_samples else 0.0,
            total_processing_time=round(total_processing_time, 4)
        )
    
    def create_complete_test_result(self,
                                    test_type: str,
                                    dataset_info: DatasetInfo,
                                    antispoofing_results: Optional[AntispoofingResults] = None,
                                    face_recognition_results: Optional[FaceRecognitionResults] = None,
                                    performance_results: Optional[PerformanceResults] = None,
                                    detailed_logs_path: str = "") -> CompleteTestResult:
        """
        Create complete test result structure
        """
        test_info = TestInfo(
            test_id=self.generate_test_id(),
            test_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            test_type=test_type,
            dataset_info=dataset_info
        )
        
        results = TestResults(
            antispoofing=antispoofing_results,
            face_recognition=face_recognition_results,
            performance=performance_results
        )
        
        return CompleteTestResult(
            test_info=test_info,
            results=results,
            detailed_logs=detailed_logs_path
        )
    
    def save_test_result_json(self, test_result: CompleteTestResult) -> str:
        """
        Save test result as JSON file
        
        Returns:
            Path to saved JSON file
        """
        filename = f"{self.output_dir}/json/{test_result.test_info.test_id}.json"
        
        # Convert dataclass to dict for JSON serialization
        result_dict = asdict(test_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def save_test_result_csv(self, test_result: CompleteTestResult) -> str:
        """
        Save test result as CSV file for easy analysis
        
        Returns:
            Path to saved CSV file
        """
        import csv
        
        filename = f"{self.output_dir}/csv/{test_result.test_info.test_id}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Metric', 'Value', 'Category'])
            
            # Write test info
            writer.writerow(['Test ID', test_result.test_info.test_id, 'Test Info'])
            writer.writerow(['Test Date', test_result.test_info.test_date, 'Test Info'])
            writer.writerow(['Test Type', test_result.test_info.test_type, 'Test Info'])
            writer.writerow(['Total Samples', test_result.test_info.dataset_info.total_samples, 'Dataset'])
            writer.writerow(['Real Samples', test_result.test_info.dataset_info.real_samples, 'Dataset'])
            writer.writerow(['Fake Samples', test_result.test_info.dataset_info.fake_samples, 'Dataset'])
            writer.writerow(['Unique Individuals', test_result.test_info.dataset_info.unique_individuals, 'Dataset'])
            
            # Write antispoofing results if available
            if test_result.results.antispoofing:
                as_result = test_result.results.antispoofing
                writer.writerow(['Accuracy', as_result.accuracy, 'Anti-Spoofing'])
                writer.writerow(['Precision', as_result.precision, 'Anti-Spoofing'])
                writer.writerow(['Recall', as_result.recall, 'Anti-Spoofing'])
                writer.writerow(['F1 Score', as_result.f1_score, 'Anti-Spoofing'])
                writer.writerow(['FAR', as_result.far, 'Anti-Spoofing'])
                writer.writerow(['FRR', as_result.frr, 'Anti-Spoofing'])
                writer.writerow(['Avg Detection Time', as_result.average_detection_time, 'Anti-Spoofing'])
            
            # Write face recognition results if available
            if test_result.results.face_recognition:
                fr_result = test_result.results.face_recognition
                writer.writerow(['Rank-1 Accuracy', fr_result.rank1_accuracy, 'Face Recognition'])
                writer.writerow(['Rank-5 Accuracy', fr_result.rank5_accuracy, 'Face Recognition'])
                writer.writerow(['Verification Accuracy', fr_result.verification_accuracy, 'Face Recognition'])
                writer.writerow(['Avg Recognition Time', fr_result.average_recognition_time, 'Face Recognition'])
                writer.writerow(['False Match Rate', fr_result.false_match_rate, 'Face Recognition'])
                writer.writerow(['False Non-Match Rate', fr_result.false_non_match_rate, 'Face Recognition'])
            
            # Write performance results if available
            if test_result.results.performance:
                perf_result = test_result.results.performance
                writer.writerow(['CPU Usage Avg', perf_result.cpu_usage_avg, 'Performance'])
                writer.writerow(['Memory Usage Avg', perf_result.memory_usage_avg, 'Performance'])
                writer.writerow(['FPS Avg', perf_result.fps_avg, 'Performance'])
                writer.writerow(['Total Processing Time', perf_result.total_processing_time, 'Performance'])
        
        return filename
    
    def generate_summary_report(self, test_results: List[CompleteTestResult]) -> str:
        """
        Generate summary report from multiple test results
        
        Returns:
            Path to summary report file
        """
        summary_filename = f"{self.output_dir}/summary_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE TEST RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total test runs: {len(test_results)}\n\n")
            
            # Categorize test results
            antispoofing_tests = [tr for tr in test_results if tr.results.antispoofing]
            face_recognition_tests = [tr for tr in test_results if tr.results.face_recognition]
            integration_tests = [tr for tr in test_results if tr.test_info.test_type == "integration"]
            
            f.write(f"Anti-spoofing tests: {len(antispoofing_tests)}\n")
            f.write(f"Face recognition tests: {len(face_recognition_tests)}\n")
            f.write(f"Integration tests: {len(integration_tests)}\n\n")
            
            # Anti-spoofing summary
            if antispoofing_tests:
                f.write("ANTI-SPOOFING RESULTS SUMMARY\n")
                f.write("-" * 30 + "\n")
                accuracies = [tr.results.antispoofing.accuracy for tr in antispoofing_tests]
                f.write(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
                fars = [tr.results.antispoofing.far for tr in antispoofing_tests]
                f.write(f"Average FAR: {np.mean(fars):.4f} ± {np.std(fars):.4f}\n")
                frrs = [tr.results.antispoofing.frr for tr in antispoofing_tests]
                f.write(f"Average FRR: {np.mean(frrs):.4f} ± {np.std(frrs):.4f}\n\n")
            
            # Face recognition summary
            if face_recognition_tests:
                f.write("FACE RECOGNITION RESULTS SUMMARY\n")
                f.write("-" * 30 + "\n")
                rank1_accs = [tr.results.face_recognition.rank1_accuracy for tr in face_recognition_tests]
                f.write(f"Average Rank-1 Accuracy: {np.mean(rank1_accs):.4f} ± {np.std(rank1_accs):.4f}\n")
                ver_accs = [tr.results.face_recognition.verification_accuracy for tr in face_recognition_tests]
                f.write(f"Average Verification Accuracy: {np.mean(ver_accs):.4f} ± {np.std(ver_accs):.4f}\n\n")
        
        return summary_filename

# Example usage and testing
if __name__ == "__main__":
    formatter = TestResultFormatter()
    
    # Example: Create antispoofing test result
    dataset_info = DatasetInfo(
        total_samples=1000,
        real_samples=500,
        fake_samples=500,
        unique_individuals=100
    )
    
    # Example metrics (replace with actual test results)
    antispoofing_result = formatter.create_antispoofing_result(
        tp=480, tn=470, fp=30, fn=20,
        detection_times=[0.15, 0.12, 0.18, 0.14, 0.16]
    )
    
    performance_result = formatter.create_performance_result(
        cpu_usage_samples=[45.2, 48.1, 42.8, 50.3],
        memory_usage_samples=[512.5, 520.1, 508.9, 515.2],
        fps_samples=[28.5, 29.1, 27.8, 28.9],
        total_processing_time=150.25
    )
    
    # Create complete test result
    complete_result = formatter.create_complete_test_result(
        test_type="antispoofing",
        dataset_info=dataset_info,
        antispoofing_results=antispoofing_result,
        performance_results=performance_result,
        detailed_logs_path="test_results/logs/detailed_antispoofing_log.txt"
    )
    
    # Save results
    json_path = formatter.save_test_result_json(complete_result)
    csv_path = formatter.save_test_result_csv(complete_result)
    
    print(f"Test result saved to:")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")

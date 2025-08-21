"""
METRICS COLLECTOR FOR COMPREHENSIVE TESTING
==========================================
Collects detailed performance metrics for thesis documentation

Features:
- Real-time performance monitoring
- Resource utilization tracking
- Accuracy metrics calculation
- Statistical analysis
- Time-series data collection
"""

import time
import psutil
import numpy as np
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import sqlite3
import os

class MetricsCollector:
    """
    Comprehensive metrics collection system for testing framework
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'network_stats': [],
            'disk_io': [],
            'timestamps': []
        }
        
        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'memory_warning': 85.0,
            'response_time_warning': 5.0
        }
        
        # Database for metrics storage
        self.metrics_db_path = "tests/test_results/metrics.db"
        self._init_metrics_database()
    
    def _init_metrics_database(self):
        """Initialize metrics database"""
        os.makedirs(os.path.dirname(self.metrics_db_path), exist_ok=True)
        
        with sqlite3.connect(self.metrics_db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_id TEXT,
                    metric_type TEXT,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    additional_data TEXT
                )
            ''')
            
            # Test results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    test_name TEXT,
                    timestamp TEXT,
                    result TEXT,
                    metrics TEXT,
                    success BOOLEAN
                )
            ''')
            
            conn.commit()
    
    def start_performance_monitoring(self, interval: float = 1.0) -> str:
        """
        Start performance monitoring
        
        Args:
            interval: Monitoring interval in seconds
            
        Returns:
            Monitoring session ID
        """
        session_id = f"monitoring_{int(time.time())}"
        
        if self.monitoring_active:
            print("⚠️ Performance monitoring already active")
            return session_id
        
        self.monitoring_active = True
        self.metrics_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'network_stats': [],
            'disk_io': [],
            'timestamps': []
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance,
            args=(interval, session_id),
            daemon=True
        )
        self.monitoring_thread.start()
        
        print(f"📊 Performance monitoring started (Session: {session_id})")
        return session_id
    
    def stop_performance_monitoring(self, session_id: str) -> Dict[str, Any]:
        """
        Stop performance monitoring and return collected data
        
        Args:
            session_id: Monitoring session ID
            
        Returns:
            Collected metrics data
        """
        if not self.monitoring_active:
            print("⚠️ Performance monitoring not active")
            return {}
        
        self.monitoring_active = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Calculate statistics
        stats = self._calculate_performance_statistics()
        
        # Save to database
        self._save_performance_data(session_id, stats)
        
        print(f"📊 Performance monitoring stopped (Session: {session_id})")
        
        return {
            'session_id': session_id,
            'raw_data': self.metrics_data.copy(),
            'statistics': stats,
            'duration': len(self.metrics_data['timestamps']),
            'warnings': self._check_performance_warnings(stats)
        }
    
    def _monitor_performance(self, interval: float, session_id: str):
        """Background performance monitoring"""
        print(f"🔄 Performance monitoring active (interval: {interval}s)")
        
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Network metrics
                network = psutil.net_io_counters()
                
                # Disk I/O metrics
                disk_io = psutil.disk_io_counters()
                
                # GPU metrics (if available)
                gpu_usage = self._get_gpu_usage()
                
                # Store metrics
                self.metrics_data['timestamps'].append(timestamp)
                self.metrics_data['cpu_usage'].append({
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else None
                })
                self.metrics_data['memory_usage'].append({
                    'percent': memory.percent,
                    'available': memory.available,
                    'used': memory.used,
                    'total': memory.total,
                    'swap_percent': swap.percent if swap else None
                })
                self.metrics_data['network_stats'].append({
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                })
                self.metrics_data['disk_io'].append({
                    'read_bytes': disk_io.read_bytes if disk_io else None,
                    'write_bytes': disk_io.write_bytes if disk_io else None,
                    'read_count': disk_io.read_count if disk_io else None,
                    'write_count': disk_io.write_count if disk_io else None
                })
                self.metrics_data['gpu_usage'].append(gpu_usage)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️ Error in performance monitoring: {e}")
                time.sleep(interval)
    
    def _get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage metrics (if available)"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                }
        except ImportError:
            pass
        except Exception as e:
            pass
        
        return None
    
    def _calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics from collected data"""
        if not self.metrics_data['timestamps']:
            return {}
        
        stats = {}
        
        # CPU statistics
        cpu_values = [data['percent'] for data in self.metrics_data['cpu_usage']]
        if cpu_values:
            stats['cpu'] = {
                'mean': np.mean(cpu_values),
                'std': np.std(cpu_values),
                'min': np.min(cpu_values),
                'max': np.max(cpu_values),
                'median': np.median(cpu_values),
                'p95': np.percentile(cpu_values, 95),
                'p99': np.percentile(cpu_values, 99)
            }
        
        # Memory statistics
        memory_values = [data['percent'] for data in self.metrics_data['memory_usage']]
        if memory_values:
            stats['memory'] = {
                'mean': np.mean(memory_values),
                'std': np.std(memory_values),
                'min': np.min(memory_values),
                'max': np.max(memory_values),
                'median': np.median(memory_values),
                'p95': np.percentile(memory_values, 95),
                'p99': np.percentile(memory_values, 99)
            }
        
        # GPU statistics (if available)
        gpu_values = [data['utilization'] for data in self.metrics_data['gpu_usage'] 
                     if data is not None and 'utilization' in data]
        if gpu_values:
            stats['gpu'] = {
                'mean': np.mean(gpu_values),
                'std': np.std(gpu_values),
                'min': np.min(gpu_values),
                'max': np.max(gpu_values),
                'median': np.median(gpu_values)
            }
        
        # Network statistics
        if len(self.metrics_data['network_stats']) > 1:
            first_net = self.metrics_data['network_stats'][0]
            last_net = self.metrics_data['network_stats'][-1]
            duration = len(self.metrics_data['timestamps'])
            
            stats['network'] = {
                'bytes_sent_rate': (last_net['bytes_sent'] - first_net['bytes_sent']) / duration,
                'bytes_recv_rate': (last_net['bytes_recv'] - first_net['bytes_recv']) / duration,
                'total_bytes_sent': last_net['bytes_sent'] - first_net['bytes_sent'],
                'total_bytes_recv': last_net['bytes_recv'] - first_net['bytes_recv']
            }
        
        return stats
    
    def _check_performance_warnings(self, stats: Dict[str, Any]) -> List[str]:
        """Check for performance warnings based on thresholds"""
        warnings = []
        
        # CPU warnings
        if 'cpu' in stats:
            if stats['cpu']['mean'] > self.thresholds['cpu_warning']:
                warnings.append(f"High CPU usage: {stats['cpu']['mean']:.1f}%")
            if stats['cpu']['max'] > 95:
                warnings.append(f"CPU spike detected: {stats['cpu']['max']:.1f}%")
        
        # Memory warnings
        if 'memory' in stats:
            if stats['memory']['mean'] > self.thresholds['memory_warning']:
                warnings.append(f"High memory usage: {stats['memory']['mean']:.1f}%")
            if stats['memory']['max'] > 95:
                warnings.append(f"Memory spike detected: {stats['memory']['max']:.1f}%")
        
        return warnings
    
    def _save_performance_data(self, session_id: str, stats: Dict[str, Any]):
        """Save performance data to database"""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                
                # Save statistics
                for metric_type, data in stats.items():
                    if isinstance(data, dict):
                        for metric_name, value in data.items():
                            cursor.execute('''
                                INSERT INTO performance_metrics 
                                (timestamp, test_id, metric_type, metric_name, value, unit, additional_data)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                timestamp,
                                session_id,
                                metric_type,
                                metric_name,
                                float(value) if isinstance(value, (int, float)) else None,
                                '%' if metric_type in ['cpu', 'memory', 'gpu'] else 'bytes/s',
                                json.dumps({'session_id': session_id})
                            ))
                
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ Error saving performance data: {e}")
    
    def collect_test_metrics(self, test_id: str, test_type: str, test_name: str, 
                           result: Any, metrics: Dict[str, Any], success: bool):
        """
        Collect metrics for individual test
        
        Args:
            test_id: Unique test identifier
            test_type: Type of test (antispoofing, face_recognition, etc.)
            test_name: Name of specific test
            result: Test result
            metrics: Test metrics
            success: Whether test was successful
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO test_results 
                    (test_id, test_type, test_name, timestamp, result, metrics, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    test_id,
                    test_type,
                    test_name,
                    datetime.now().isoformat(),
                    json.dumps(result, default=str),
                    json.dumps(metrics, default=str),
                    success
                ))
                
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ Error collecting test metrics: {e}")
    
    def calculate_accuracy_metrics(self, predictions: List[bool], 
                                 ground_truth: List[bool]) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics
        
        Args:
            predictions: List of predicted values
            ground_truth: List of ground truth values
            
        Returns:
            Dictionary of accuracy metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Convert to numpy arrays
        pred = np.array(predictions)
        truth = np.array(ground_truth)
        
        # Calculate confusion matrix components
        tp = np.sum((pred == True) & (truth == True))
        tn = np.sum((pred == False) & (truth == False))
        fp = np.sum((pred == True) & (truth == False))
        fn = np.sum((pred == False) & (truth == True))
        
        # Calculate metrics
        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False rates
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'far': far,
            'frr': frr,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'total_samples': total
        }
    
    def calculate_statistical_confidence(self, values: List[float], 
                                       confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate statistical confidence intervals
        
        Args:
            values: List of values
            confidence_level: Confidence level (default 0.95 for 95%)
            
        Returns:
            Statistical measures with confidence intervals
        """
        if not values:
            return {}
        
        values = np.array(values)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        
        # Calculate confidence interval
        from scipy import stats
        confidence_interval = stats.t.interval(
            confidence_level, n-1, loc=mean, scale=stats.sem(values)
        )
        
        return {
            'mean': mean,
            'std': std,
            'sem': stats.sem(values),  # Standard error of mean
            'confidence_level': confidence_level,
            'confidence_interval_lower': confidence_interval[0],
            'confidence_interval_upper': confidence_interval[1],
            'sample_size': n,
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
    
    def export_metrics_summary(self, output_path: str) -> Dict[str, Any]:
        """
        Export comprehensive metrics summary
        
        Args:
            output_path: Path to save summary
            
        Returns:
            Summary data
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                # Get all test results
                cursor.execute('''
                    SELECT test_type, test_name, success, metrics, timestamp
                    FROM test_results
                    ORDER BY timestamp DESC
                ''')
                test_results = cursor.fetchall()
                
                # Get performance metrics
                cursor.execute('''
                    SELECT metric_type, metric_name, AVG(value), MIN(value), MAX(value), COUNT(*)
                    FROM performance_metrics
                    GROUP BY metric_type, metric_name
                ''')
                perf_metrics = cursor.fetchall()
                
                # Compile summary
                summary = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_tests': len(test_results),
                    'test_results_by_type': {},
                    'performance_summary': {},
                    'overall_statistics': {}
                }
                
                # Process test results by type
                for test_type, test_name, success, metrics_json, timestamp in test_results:
                    if test_type not in summary['test_results_by_type']:
                        summary['test_results_by_type'][test_type] = {
                            'total': 0,
                            'successful': 0,
                            'failed': 0
                        }
                    
                    summary['test_results_by_type'][test_type]['total'] += 1
                    if success:
                        summary['test_results_by_type'][test_type]['successful'] += 1
                    else:
                        summary['test_results_by_type'][test_type]['failed'] += 1
                
                # Process performance metrics
                for metric_type, metric_name, avg_val, min_val, max_val, count in perf_metrics:
                    if metric_type not in summary['performance_summary']:
                        summary['performance_summary'][metric_type] = {}
                    
                    summary['performance_summary'][metric_type][metric_name] = {
                        'average': avg_val,
                        'minimum': min_val,
                        'maximum': max_val,
                        'sample_count': count
                    }
                
                # Calculate overall success rate
                total_tests = sum(data['total'] for data in summary['test_results_by_type'].values())
                total_successful = sum(data['successful'] for data in summary['test_results_by_type'].values())
                
                summary['overall_statistics'] = {
                    'overall_success_rate': total_successful / total_tests if total_tests > 0 else 0,
                    'total_test_types': len(summary['test_results_by_type']),
                    'database_records': len(test_results) + len(perf_metrics)
                }
                
                # Save summary
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                print(f"📊 Metrics summary exported to: {output_path}")
                return summary
                
        except Exception as e:
            print(f"❌ Error exporting metrics summary: {e}")
            return {}


def main():
    """Test the metrics collector"""
    print("🧪 Testing Metrics Collector")
    
    collector = MetricsCollector()
    
    # Start monitoring
    session_id = collector.start_performance_monitoring(interval=0.5)
    
    # Simulate some work
    time.sleep(5)
    
    # Stop monitoring
    results = collector.stop_performance_monitoring(session_id)
    
    # Print results
    print(f"\n📊 Monitoring Results:")
    print(f"Session ID: {results['session_id']}")
    print(f"Duration: {results['duration']} samples")
    
    if 'statistics' in results and 'cpu' in results['statistics']:
        cpu_stats = results['statistics']['cpu']
        print(f"CPU - Mean: {cpu_stats['mean']:.1f}%, Max: {cpu_stats['max']:.1f}%")
    
    if results['warnings']:
        print(f"⚠️ Warnings: {', '.join(results['warnings'])}")
    
    print("✅ Metrics collector test completed")


if __name__ == "__main__":
    main()

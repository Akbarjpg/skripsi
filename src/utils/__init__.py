"""
Utility functions untuk Face Anti-Spoofing system
"""

import os
import cv2
import numpy as np
import torch
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Optional imports for visualization (for lightweight deployment)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

def json_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays and other non-JSON-serializable objects to JSON-serializable formats
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serializable(item) for item in obj]
    else:
        return obj

def safe_jsonify(data: Dict) -> Dict:
    """
    Safely convert data to JSON-serializable format
    """
    return json_serializable(data)

def setup_directories(base_path: str, directories: List[str]):
    """
    Setup direktori yang diperlukan untuk sistem
    """
    created_dirs = []
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            created_dirs.append(dir_path)
            logging.info(f"Created directory: {dir_path}")
    
    return created_dirs

def load_config(config_path: str) -> Dict:
    """
    Load konfigurasi dari file JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file: {e}")
        return {}

def save_config(config: Dict, config_path: str):
    """
    Save konfigurasi ke file JSON
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Config saved to: {config_path}")
    except Exception as e:
        logging.error(f"Error saving config: {e}")

def calculate_model_size(model_path: str) -> Dict:
    """
    Calculate ukuran model file
    """
    if not os.path.exists(model_path):
        return {'error': 'Model file not found'}
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    
    return {
        'bytes': size_bytes,
        'mb': round(size_mb, 2),
        'gb': round(size_gb, 3),
        'human_readable': f"{size_mb:.1f} MB" if size_mb < 1000 else f"{size_gb:.2f} GB"
    }

def preprocess_image_for_model(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess gambar untuk model inference
    """
    # Convert BGR to RGB jika perlu
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def draw_detection_overlay(image: np.ndarray, detection_results: Dict) -> np.ndarray:
    """
    Draw overlay informasi deteksi pada gambar
    """
    overlay = image.copy()
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Background untuk text
    def draw_text_with_background(img, text, position, font_color=(255, 255, 255), bg_color=(0, 0, 0)):
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, 
                     (position[0] - 5, position[1] - text_height - 10),
                     (position[0] + text_width + 5, position[1] + 5),
                     bg_color, -1)
        cv2.putText(img, text, position, font, font_scale, font_color, thickness)
    
    y_offset = 30
    
    # CNN Score
    if 'cnn_score' in detection_results:
        score = detection_results['cnn_score']
        color = (0, 255, 0) if score > 0.7 else (0, 0, 255)
        text = f"CNN: {score:.3f}"
        draw_text_with_background(overlay, text, (10, y_offset), color)
        y_offset += 40
    
    # Landmark Detection
    if 'landmark_results' in detection_results:
        landmarks = detection_results['landmark_results']
        if landmarks.get('landmarks_detected', False):
            draw_text_with_background(overlay, "Face: DETECTED", (10, y_offset), (0, 255, 0))
        else:
            draw_text_with_background(overlay, "Face: NOT DETECTED", (10, y_offset), (0, 0, 255))
        y_offset += 40
        
        # Blink count
        blinks = landmarks.get('blink_count', 0)
        draw_text_with_background(overlay, f"Blinks: {blinks}", (10, y_offset), (255, 255, 0))
        y_offset += 40
        
        # Head pose
        if 'head_pose' in landmarks:
            pose = landmarks['head_pose']
            if pose:
                pose_text = f"Head: Y:{pose['yaw']:.0f}° P:{pose['pitch']:.0f}°"
                draw_text_with_background(overlay, pose_text, (10, y_offset), (255, 255, 0))
                y_offset += 40
    
    # Challenge Status
    if 'challenge_status' in detection_results:
        status = detection_results['challenge_status']
        if status:
            challenge_text = f"Challenge: {status['description'][:30]}..."
            draw_text_with_background(overlay, challenge_text, (10, y_offset), (0, 255, 255))
            y_offset += 40
            
            time_text = f"Time: {status['remaining_time']:.1f}s"
            draw_text_with_background(overlay, time_text, (10, y_offset), (0, 255, 255))
    
    # Overall Score
    if 'fusion_score' in detection_results:
        fusion_score = detection_results['fusion_score']
        score_color = (0, 255, 0) if fusion_score > 0.7 else (0, 165, 255) if fusion_score > 0.5 else (0, 0, 255)
        score_text = f"Overall: {fusion_score:.3f}"
        draw_text_with_background(overlay, score_text, (10, image.shape[0] - 30), score_color)
    
    return overlay

def create_comparison_plot(real_scores: List[float], fake_scores: List[float], 
                          title: str = "Score Distribution", save_path: str = None):
    """
    Create comparison plot untuk real vs fake scores
    Note: Requires matplotlib to be installed
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"📊 Plotting skipped: matplotlib not available")
        print(f"Real scores: count={len(real_scores)}, mean={np.mean(real_scores):.3f}")
        print(f"Fake scores: count={len(fake_scores)}, mean={np.mean(fake_scores):.3f}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(real_scores, bins=30, alpha=0.7, label='Real', color='green', density=True)
    plt.hist(fake_scores, bins=30, alpha=0.7, label='Fake', color='red', density=True)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title(f'{title} - Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    data = [real_scores, fake_scores]
    labels = ['Real', 'Fake']
    colors = ['green', 'red']
    
    box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Score')
    plt.title(f'{title} - Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(predictions: List[int], labels: List[int], 
                     probabilities: List[float] = None) -> Dict:
    """
    Calculate comprehensive metrics
    """
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                                confusion_matrix, roc_auc_score, classification_report)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
    
    metrics['precision_real'] = precision[1] if len(precision) > 1 else 0
    metrics['precision_fake'] = precision[0] if len(precision) > 0 else 0
    metrics['recall_real'] = recall[1] if len(recall) > 1 else 0
    metrics['recall_fake'] = recall[0] if len(recall) > 0 else 0
    metrics['f1_real'] = f1[1] if len(f1) > 1 else 0
    metrics['f1_fake'] = f1[0] if len(f1) > 0 else 0
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Calculate TPR, FPR, TNR, FNR
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # AUC if probabilities provided
    if probabilities is not None:
        try:
            metrics['auc'] = roc_auc_score(labels, probabilities)
        except ValueError:
            metrics['auc'] = 0.5
    
    # Classification report
    metrics['classification_report'] = classification_report(labels, predictions, output_dict=True)
    
    return metrics

def benchmark_inference_speed(model, input_size: Tuple[int, int, int] = (3, 224, 224), 
                            device: str = 'cpu', num_iterations: int = 100) -> Dict:
    """
    Benchmark inference speed model
    """
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times),
        'times': times
    }

def create_system_report(results: Dict, save_path: str = None) -> str:
    """
    Create comprehensive system performance report
    """
    report = []
    report.append("=" * 60)
    report.append("FACE ANTI-SPOOFING SYSTEM PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model Performance
    if 'model_metrics' in results:
        metrics = results['model_metrics']
        report.append("MODEL PERFORMANCE:")
        report.append("-" * 30)
        report.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        report.append(f"AUC Score: {metrics.get('auc', 0):.4f}")
        report.append(f"F1 Score (Real): {metrics.get('f1_real', 0):.4f}")
        report.append(f"F1 Score (Fake): {metrics.get('f1_fake', 0):.4f}")
        report.append(f"Precision (Real): {metrics.get('precision_real', 0):.4f}")
        report.append(f"Recall (Real): {metrics.get('recall_real', 0):.4f}")
        report.append("")
    
    # Speed Performance
    if 'speed_benchmark' in results:
        speed = results['speed_benchmark']
        report.append("INFERENCE SPEED:")
        report.append("-" * 30)
        report.append(f"Mean Inference Time: {speed.get('mean_time', 0)*1000:.2f} ms")
        report.append(f"FPS: {speed.get('fps', 0):.1f}")
        report.append(f"Min Time: {speed.get('min_time', 0)*1000:.2f} ms")
        report.append(f"Max Time: {speed.get('max_time', 0)*1000:.2f} ms")
        report.append("")
    
    # System Resources
    if 'system_info' in results:
        system = results['system_info']
        report.append("SYSTEM INFORMATION:")
        report.append("-" * 30)
        for key, value in system.items():
            report.append(f"{key}: {value}")
        report.append("")
    
    # Challenge Statistics
    if 'challenge_stats' in results:
        stats = results['challenge_stats']
        report.append("CHALLENGE-RESPONSE STATISTICS:")
        report.append("-" * 30)
        report.append(f"Total Challenges: {stats.get('total_challenges', 0)}")
        report.append(f"Success Rate: {stats.get('success_rate', 0):.2%}")
        report.append(f"Average Response Time: {stats.get('average_response_time', 0):.2f}s")
        report.append("")
    
    # Security Assessment
    if 'security_assessment' in results:
        security = results['security_assessment']
        report.append("SECURITY ASSESSMENT:")
        report.append("-" * 30)
        for attack_type, success_rate in security.items():
            report.append(f"{attack_type}: {(1-success_rate)*100:.1f}% protection")
        report.append("")
    
    report.append("=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        logging.info(f"Report saved to: {save_path}")
    
    return report_text

def validate_installation():
    """
    Validate installation dan dependency
    """
    issues = []
    warnings = []
    
    # Check Python packages
    required_packages = [
        'torch', 'torchvision', 'cv2', 'mediapipe', 
        'flask', 'numpy', 'sklearn', 'matplotlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        warnings.append(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        warnings.append("CUDA not available - using CPU")
    
    # Check directory structure
    required_dirs = ['models', 'logs', 'data', 'src']
    for directory in required_dirs:
        if not os.path.exists(directory):
            issues.append(f"Missing directory: {directory}")
    
    return {
        'issues': issues,
        'warnings': warnings,
        'status': 'ok' if len(issues) == 0 else 'error'
    }

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test validation
    validation = validate_installation()
    print(f"Validation status: {validation['status']}")
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print("Utility functions loaded successfully!")

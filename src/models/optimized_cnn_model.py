"""
OPTIMIZED CNN Model untuk Face Anti-Spoofing Liveness Detection
Arsitektur yang dioptimalkan untuk real-time inference dengan performance tinggi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
from typing import Tuple, Optional
import threading
import queue


class MobileNetV3Block(nn.Module):
    """
    Optimized MobileNetV3 block untuk real-time processing
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=True):
        super().__init__()
        
        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, 
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Squeeze-and-Excitation (optional for speed)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(in_channels, reduction=4)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.Hardswish(inplace=True)  # Faster than ReLU6
        
        # Skip connection
        self.use_skip = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        identity = x
        
        # Depthwise
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # SE block
        if self.use_se:
            out = self.se(out)
        
        # Pointwise
        out = self.pointwise(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.use_skip:
            out = out + identity
            
        return out


class SEBlock(nn.Module):
    """
    Optimized Squeeze-and-Excitation block
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OptimizedLivenessCNN(nn.Module):
    """
    Highly optimized CNN model for real-time liveness detection
    Designed for 15-20+ FPS performance
    """
    
    def __init__(self, num_classes=2, input_size=112, dropout_rate=0.2):
        super().__init__()
        
        self.input_size = input_size
        
        # Optimized stem - faster initial processing
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        
        # Lightweight backbone with MobileNetV3 blocks
        self.features = nn.Sequential(
            # Stage 1: 56x56
            MobileNetV3Block(16, 24, stride=1, use_se=False),  # Skip SE for speed
            MobileNetV3Block(24, 24, stride=1, use_se=False),
            
            # Stage 2: 28x28  
            MobileNetV3Block(24, 32, stride=2, use_se=False),
            MobileNetV3Block(32, 32, stride=1, use_se=False),
            
            # Stage 3: 14x14
            MobileNetV3Block(32, 64, stride=2, use_se=True),   # Use SE only in deeper layers
            MobileNetV3Block(64, 64, stride=1, use_se=True),
            
            # Stage 4: 7x7
            MobileNetV3Block(64, 96, stride=2, use_se=True),
        )
        
        # Efficient global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(96, 32),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"✅ OptimizedLivenessCNN initialized - Input size: {input_size}x{input_size}")
    
    def _initialize_weights(self):
        """Optimized weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Ensure correct input size
        if x.size(2) != self.input_size or x.size(3) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        # Feature extraction
        x = self.stem(x)
        x = self.features(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class QuantizedLivenessCNN(nn.Module):
    """
    Quantized version for even faster inference
    """
    
    def __init__(self, model_path=None):
        super().__init__()
        self.base_model = OptimizedLivenessCNN()
        
        if model_path:
            self.base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Quantize model for faster inference
        self.base_model.eval()
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.base_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
    
    def forward(self, x):
        return self.quantized_model(x)


class OptimizedLivenessPredictor:
    """
    Optimized predictor with caching, batching, and threading
    """
    
    def __init__(self, model_path=None, device='cpu', use_quantization=True, 
                 batch_size=1, cache_size=100):
        self.device = device
        self.batch_size = batch_size
        self.use_quantization = use_quantization
        
        # Load model
        if use_quantization:
            self.model = QuantizedLivenessCNN(model_path)
        else:
            self.model = OptimizedLivenessCNN()
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
        
        # Prediction cache
        self.cache = {}
        self.cache_times = {}
        self.cache_size = cache_size
        self.cache_duration = 0.1  # 100ms cache
        
        # Performance monitoring
        self.prediction_times = []
        
        # Threading for async predictions
        self.prediction_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.prediction_thread = None
        self.stop_thread = False
        
        # Image preprocessing
        self.input_size = 112
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        print(f"✅ OptimizedLivenessPredictor initialized - Device: {device}, Quantized: {use_quantization}")
    
    def preprocess_image_optimized(self, image):
        """
        Optimized image preprocessing
        """
        try:
            # Convert numpy to tensor efficiently
            if isinstance(image, np.ndarray):
                # Resize if needed
                if image.shape[:2] != (self.input_size, self.input_size):
                    import cv2
                    image = cv2.resize(image, (self.input_size, self.input_size), 
                                     interpolation=cv2.INTER_LINEAR)
                
                # Convert BGR to RGB and normalize
                image = image[:, :, ::-1]  # BGR to RGB
                tensor = torch.from_numpy(image).float() / 255.0
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to NCHW
            else:
                tensor = image
            
            # Normalize with pre-computed values
            if self.device == 'cpu':
                tensor = (tensor - self.mean) / self.std
            else:
                tensor = tensor.to(self.device)
                tensor = (tensor - self.mean.to(self.device)) / self.std.to(self.device)
            
            return tensor
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def get_cache_key(self, image):
        """
        Generate cache key from image
        """
        # Use image hash for caching (simplified)
        if isinstance(image, np.ndarray):
            return hash(image.tobytes()[:1000])  # Hash first 1000 bytes for speed
        return hash(str(image)[:100])
    
    def predict_optimized(self, image, use_cache=True):
        """
        Optimized prediction with caching
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = None
            if use_cache:
                cache_key = self.get_cache_key(image)
                if (cache_key in self.cache and 
                    time.time() - self.cache_times.get(cache_key, 0) < self.cache_duration):
                    return self.cache[cache_key]
            
            # Preprocess
            tensor = self.preprocess_image_optimized(image)
            if tensor is None:
                return {'confidence': 0.0, 'is_live': False, 'processing_time': time.time() - start_time}
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence = float(probabilities[0][1])  # Probability of "live"
                is_live = confidence > 0.5
            
            processing_time = time.time() - start_time
            self.prediction_times.append(processing_time)
            
            # Keep only last 100 times for memory efficiency
            if len(self.prediction_times) > 100:
                self.prediction_times = self.prediction_times[-50:]
            
            result = {
                'confidence': confidence,
                'is_live': is_live,
                'processing_time': processing_time,
                'probabilities': {
                    'fake': float(probabilities[0][0]),
                    'live': float(probabilities[0][1])
                }
            }
            
            # Cache result
            if use_cache and cache_key:
                # Manage cache size
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entries
                    oldest_key = min(self.cache_times.keys(), key=lambda k: self.cache_times[k])
                    del self.cache[oldest_key]
                    del self.cache_times[oldest_key]
                
                self.cache[cache_key] = result
                self.cache_times[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Prediction error: {e}")
            return {
                'confidence': 0.0,
                'is_live': False,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def predict_batch(self, images):
        """
        Batch prediction for multiple images
        """
        if not images:
            return []
        
        try:
            # Preprocess all images
            tensors = []
            for image in images:
                tensor = self.preprocess_image_optimized(image)
                if tensor is not None:
                    tensors.append(tensor)
            
            if not tensors:
                return [{'confidence': 0.0, 'is_live': False} for _ in images]
            
            # Batch processing
            batch_tensor = torch.cat(tensors, dim=0)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Process results
            results = []
            for i in range(len(tensors)):
                confidence = float(probabilities[i][1])
                results.append({
                    'confidence': confidence,
                    'is_live': confidence > 0.5,
                    'probabilities': {
                        'fake': float(probabilities[i][0]),
                        'live': float(probabilities[i][1])
                    }
                })
            
            return results
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return [{'confidence': 0.0, 'is_live': False, 'error': str(e)} for _ in images]
    
    def start_async_processing(self):
        """
        Start async prediction thread
        """
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.stop_thread = False
            self.prediction_thread = threading.Thread(target=self._prediction_worker)
            self.prediction_thread.start()
    
    def _prediction_worker(self):
        """
        Background thread for predictions
        """
        while not self.stop_thread:
            try:
                image, request_id = self.prediction_queue.get(timeout=0.1)
                result = self.predict_optimized(image)
                self.result_queue.put((request_id, result))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async prediction error: {e}")
    
    def predict_async(self, image, request_id=None):
        """
        Submit async prediction request
        """
        if request_id is None:
            request_id = time.time()
        
        try:
            self.prediction_queue.put_nowait((image, request_id))
            return request_id
        except queue.Full:
            # Queue full, process synchronously
            return self.predict_optimized(image)
    
    def get_async_result(self, request_id=None, timeout=0.01):
        """
        Get async prediction result
        """
        try:
            result_id, result = self.result_queue.get(timeout=timeout)
            if request_id is None or result_id == request_id:
                return result
            else:
                # Put back if not the requested ID
                self.result_queue.put((result_id, result))
                return None
        except queue.Empty:
            return None
    
    def stop_async_processing(self):
        """
        Stop async processing
        """
        self.stop_thread = True
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)
    
    def get_performance_stats(self):
        """
        Get performance statistics
        """
        if not self.prediction_times:
            return {}
        
        times = self.prediction_times[-50:]  # Last 50 predictions
        return {
            'avg_prediction_time': np.mean(times),
            'min_prediction_time': np.min(times),
            'max_prediction_time': np.max(times),
            'estimated_fps': 1.0 / np.mean(times),
            'cache_hit_ratio': len(self.cache) / max(len(self.prediction_times), 1),
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.cache.clear()
        self.cache_times.clear()


def create_optimized_model(model_type='mobile', input_size=112, **kwargs):
    """
    Factory function untuk create optimized models
    """
    if model_type == 'mobile':
        return OptimizedLivenessCNN(input_size=input_size, **kwargs)
    elif model_type == 'quantized':
        return QuantizedLivenessCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def benchmark_model(model, input_size=112, num_iterations=100):
    """
    Benchmark model performance
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    return {
        'avg_inference_time': avg_time,
        'estimated_fps': fps,
        'min_time': np.min(times),
        'max_time': np.max(times),
        'device': device
    }


if __name__ == "__main__":
    print("🚀 TESTING OPTIMIZED CNN MODEL")
    print("=" * 50)
    
    # Test different model configurations
    models_to_test = [
        ('OptimizedLivenessCNN', OptimizedLivenessCNN()),
        ('QuantizedLivenessCNN', QuantizedLivenessCNN())
    ]
    
    for name, model in models_to_test:
        print(f"\n📊 Testing {name}:")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Benchmark
        stats = benchmark_model(model, num_iterations=50)
        print(f"Avg inference time: {stats['avg_inference_time']*1000:.1f}ms")
        print(f"Estimated FPS: {stats['estimated_fps']:.1f}")
        print(f"Device: {stats['device']}")
    
    # Test predictor
    print(f"\n🎯 Testing OptimizedLivenessPredictor:")
    predictor = OptimizedLivenessPredictor()
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        predictor.predict_optimized(dummy_image)
    
    # Performance test
    times = []
    for _ in range(20):
        start = time.time()
        result = predictor.predict_optimized(dummy_image)
        times.append(time.time() - start)
    
    print(f"Predictor avg time: {np.mean(times)*1000:.1f}ms")
    print(f"Predictor estimated FPS: {1.0/np.mean(times):.1f}")
    
    stats = predictor.get_performance_stats()
    print(f"Predictor stats: {stats}")

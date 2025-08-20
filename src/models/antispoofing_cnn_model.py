"""
Enhanced Anti-Spoofing CNN Model Implementation - Step 2
========================================================

This module implements Step 2 requirements from yangIni.md:
- Specialized CNN model for real vs fake face detection  
- Binary classification with 224x224x3 RGB input
- Training capabilities on real/fake datasets
- Integration with existing anti-spoofing checks
- Weighted voting: CNN (60%), Landmarks (20%), Challenges (20%)
- 85% combined confidence threshold

Model Architecture:
- Input layer: 224x224x3 RGB images
- Multiple convolutional layers with batch normalization
- Dropout layers to prevent overfitting
- Binary output: real (1) or fake (0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import time
import os
import json
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class EnhancedAntiSpoofingCNN(nn.Module):
    """
    Enhanced CNN for binary anti-spoofing classification (Step 2)
    Optimized architecture for real vs fake face detection
    """
    
    def __init__(self, input_size=(224, 224), dropout_rate=0.5):
        super(EnhancedAntiSpoofingCNN, self).__init__()
        
        self.input_size = input_size
        
        # Enhanced feature extraction with residual connections
        # First block - Edge and basic texture detection
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        
        # Second block - Texture pattern analysis
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Third block - Complex pattern detection
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )
        
        # Fourth block - High-level feature extraction
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )
        
        # Fifth block - Deep feature extraction
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Binary: [fake, real]
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Confidence score 0-1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        features = self.conv5(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Binary classification
        classification_logits = self.classifier(features)
        
        # Confidence estimation
        confidence = self.confidence_head(features)
        
        return {
            'logits': classification_logits,
            'confidence': confidence,
            'features': features
        }


class AntiSpoofingDataset(Dataset):
    """
    Dataset class for training anti-spoofing CNN
    Handles real and fake face images as specified in Step 2
    """
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, use_depth: bool = False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_depth = use_depth
        
        # Updated for test_img structure:
        # data_dir/  (test_img)
        #   color/
        #     *_real.jpg
        #     *_fake.jpg  
        #   depth/
        #     *_real.jpg
        #     *_fake.jpg
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all samples with labels from test_img dataset"""
        # Choose between color and depth images
        image_dir = self.data_dir / ('depth' if self.use_depth else 'color')
        
        if not image_dir.exists():
            print(f"Warning: {image_dir} does not exist")
            return
        
        # Load all images and their labels
        all_samples = []
        
        # Load real images (label = 1) - files ending with _real.jpg
        real_files = list(image_dir.glob('*_real.jpg'))
        for img_path in real_files:
            all_samples.append((str(img_path), 1))
        
        # Load fake images (label = 0) - files ending with _fake.jpg
        fake_files = list(image_dir.glob('*_fake.jpg'))
        for img_path in fake_files:
            all_samples.append((str(img_path), 0))
        
        # Shuffle for random splitting
        import random
        random.seed(42)  # For reproducible splits
        random.shuffle(all_samples)
        
        # Split ratios: 70% train, 20% val, 10% test
        total_samples = len(all_samples)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.9)
        
        # Select samples based on split
        if self.split == 'train':
            self.samples = all_samples[:train_end]
        elif self.split == 'val':
            self.samples = all_samples[train_end:val_end]
        elif self.split == 'test':
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Count real and fake samples in this split
        real_count = sum(1 for _, label in self.samples if label == 1)
        fake_count = len(self.samples) - real_count
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")
        print(f"  Real images: {real_count}, Fake images: {fake_count}")
        print(f"  Using {'depth' if self.use_depth else 'color'} images from {image_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, torch.tensor(label, dtype=torch.long)
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)  # Quality score 0-1
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Get predictions from different heads
        classification = self.classifier(pooled_features)
        texture_analysis = self.texture_head(pooled_features)
        quality_score = torch.sigmoid(self.quality_head(pooled_features))
        
        return {
            'classification': classification,
            'texture_analysis': texture_analysis,
            'quality_score': quality_score,
            'features': pooled_features
        }


class RealTimeAntiSpoofingDetector:
    """
    Real-time anti-spoofing detector implementing Step 1 requirements
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = EnhancedAntiSpoofingCNN()  # Use enhanced model for Step 2
        
        # Load pre-trained weights if available
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded anti-spoofing model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Detection configuration based on Step 1 requirements
        self.config = {
            'confidence_threshold': 0.95,  # 95% confidence as specified
            'input_size': (224, 224),
            'color_channels': ['rgb', 'hsv', 'lab'],  # Multi-channel analysis
            'texture_analysis_enabled': True,
            'temporal_consistency_frames': 5,
            'quality_threshold': 0.6
        }
        
        # Frame history for temporal analysis
        self.frame_history = []
        self.prediction_history = []
        self.max_history = 10
        
        # Color space analyzers
        self.color_analyzers = {
            'rgb': self._analyze_rgb_channels,
            'hsv': self._analyze_hsv_channels,
            'lab': self._analyze_lab_channels
        }
        
    def detect_antispoofing(self, image: np.ndarray) -> Dict:
        """
        Main anti-spoofing detection function implementing Step 1 requirements
        
        Args:
            image: Input BGR image from webcam
            
        Returns:
            Dict containing:
            - is_real_face: Boolean indicating if face is real
            - confidence: Float confidence score (0-1)
            - detailed_analysis: Dict with breakdown of detection methods
            - processing_time: Float processing time in seconds
        """
        start_time = time.time()
        
        try:
            # 1. Preprocess image
            preprocessed = self._preprocess_image(image)
            if preprocessed is None:
                return self._create_error_result("Failed to preprocess image")
            
            # 2. CNN-based texture analysis
            cnn_result = self._run_cnn_analysis(preprocessed)
            
            # 3. Color space analysis for unnatural skin tones
            color_result = self._analyze_color_spaces(image)
            
            # 4. Temporal consistency check
            temporal_result = self._check_temporal_consistency(cnn_result)
            
            # 5. Quality assessment
            quality_result = self._assess_image_quality(image)
            
            # 6. Combine all analysis methods
            combined_result = self._combine_analysis_results(
                cnn_result, color_result, temporal_result, quality_result
            )
            
            # 7. Update history
            self._update_history(image, combined_result)
            
            processing_time = time.time() - start_time
            
            return {
                'is_real_face': combined_result['is_real'],
                'confidence': combined_result['confidence'],
                'detailed_analysis': {
                    'cnn_analysis': cnn_result,
                    'color_analysis': color_result,
                    'temporal_analysis': temporal_result,
                    'quality_analysis': quality_result
                },
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Anti-spoofing detection error: {e}")
            return self._create_error_result(f"Detection error: {str(e)}")
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess image for CNN input"""
        try:
            # Resize to model input size
            resized = cv2.resize(image, self.config['input_size'])
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None
    
    def _run_cnn_analysis(self, preprocessed_image: torch.Tensor) -> Dict:
        """Run CNN-based texture analysis"""
        try:
            with torch.no_grad():
                outputs = self.model(preprocessed_image)
                
                # Get classification probabilities
                classification_probs = F.softmax(outputs['classification'], dim=1)
                live_prob = classification_probs[0, 1].item()  # Assuming class 1 is "live"
                
                # Get texture analysis
                texture_probs = F.softmax(outputs['texture_analysis'], dim=1)
                texture_scores = {
                    'print_probability': texture_probs[0, 0].item(),
                    'screen_probability': texture_probs[0, 1].item(),
                    'mask_probability': texture_probs[0, 2].item(),
                    'deepfake_probability': texture_probs[0, 3].item()
                }
                
                # Get quality score
                quality_score = outputs['quality_score'][0, 0].item()
                
                # Determine if real based on CNN
                is_real_cnn = (live_prob > 0.7 and 
                              max(texture_scores.values()) < 0.3 and
                              quality_score > self.config['quality_threshold'])
                
                return {
                    'is_real': is_real_cnn,
                    'live_probability': live_prob,
                    'texture_scores': texture_scores,
                    'quality_score': quality_score,
                    'confidence': live_prob if is_real_cnn else (1 - live_prob)
                }
                
        except Exception as e:
            logger.error(f"CNN analysis error: {e}")
            return {
                'is_real': False,
                'live_probability': 0.0,
                'texture_scores': {},
                'quality_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_color_spaces(self, image: np.ndarray) -> Dict:
        """Analyze color spaces to detect unnatural skin tones"""
        try:
            results = {}
            
            for channel_name, analyzer in self.color_analyzers.items():
                results[channel_name] = analyzer(image)
            
            # Combine color space analysis
            skin_naturalness = np.mean([r.get('skin_naturalness', 0.5) for r in results.values()])
            
            is_natural = skin_naturalness > 0.6
            
            return {
                'is_natural_skin': is_natural,
                'skin_naturalness_score': skin_naturalness,
                'color_space_results': results,
                'confidence': skin_naturalness if is_natural else (1 - skin_naturalness)
            }
            
        except Exception as e:
            logger.error(f"Color space analysis error: {e}")
            return {
                'is_natural_skin': True,  # Default to natural if analysis fails
                'skin_naturalness_score': 0.5,
                'color_space_results': {},
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _analyze_rgb_channels(self, image: np.ndarray) -> Dict:
        """Analyze RGB channels for skin tone naturalness"""
        # Extract skin regions (simplified approach)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {'skin_naturalness': 0.5}
        
        # Extract face region
        x, y, w, h = faces[0]
        face_region = image[y:y+h, x:x+w]
        
        # Analyze RGB distribution
        b, g, r = cv2.split(face_region)
        
        # Check for natural skin tone ranges
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        
        # Natural skin typically has R > G > B
        naturalness = 1.0 if r_mean > g_mean > b_mean else 0.3
        
        return {'skin_naturalness': naturalness}
    
    def _analyze_hsv_channels(self, image: np.ndarray) -> Dict:
        """Analyze HSV channels for color consistency"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Check saturation and value consistency
        s_std = np.std(s)
        v_std = np.std(v)
        
        # Natural faces have moderate saturation variation
        naturalness = 1.0 - min(1.0, (s_std + v_std) / 100.0)
        
        return {'skin_naturalness': naturalness}
    
    def _analyze_lab_channels(self, image: np.ndarray) -> Dict:
        """Analyze LAB channels for luminance consistency"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Check luminance distribution
        l_std = np.std(l)
        
        # Natural lighting has moderate luminance variation
        naturalness = 1.0 - min(1.0, l_std / 50.0)
        
        return {'skin_naturalness': naturalness}
    
    def _check_temporal_consistency(self, current_result: Dict) -> Dict:
        """Check temporal consistency across frames"""
        if len(self.prediction_history) < 3:
            return {
                'is_consistent': True,
                'consistency_score': 0.8,
                'confidence': 0.8
            }
        
        # Check if recent predictions are consistent
        recent_predictions = self.prediction_history[-self.config['temporal_consistency_frames']:]
        live_predictions = [p.get('live_probability', 0.5) for p in recent_predictions]
        
        consistency_score = 1.0 - np.std(live_predictions)
        is_consistent = consistency_score > 0.7
        
        return {
            'is_consistent': is_consistent,
            'consistency_score': consistency_score,
            'confidence': consistency_score
        }
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Assess image quality for anti-spoofing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
        
        # Calculate contrast
        contrast = np.std(gray) / 128.0
        contrast_score = min(1.0, contrast)
        
        overall_quality = (sharpness_score + brightness_score + contrast_score) / 3.0
        
        return {
            'overall_quality': overall_quality,
            'sharpness_score': sharpness_score,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'is_good_quality': overall_quality > self.config['quality_threshold']
        }
    
    def _combine_analysis_results(self, cnn_result: Dict, color_result: Dict, 
                                temporal_result: Dict, quality_result: Dict) -> Dict:
        """Combine all analysis results with weighted voting"""
        
        # Weights as specified in Step 1 requirements
        weights = {
            'cnn': 0.6,      # CNN gets highest weight
            'color': 0.15,   # Color analysis
            'temporal': 0.15, # Temporal consistency
            'quality': 0.1   # Quality assessment
        }
        
        # Individual confidences
        cnn_confidence = cnn_result.get('confidence', 0.0)
        color_confidence = color_result.get('confidence', 0.0)
        temporal_confidence = temporal_result.get('confidence', 0.0)
        quality_confidence = quality_result.get('overall_quality', 0.0)
        
        # Individual decisions
        cnn_real = cnn_result.get('is_real', False)
        color_real = color_result.get('is_natural_skin', False)
        temporal_real = temporal_result.get('is_consistent', False)
        quality_real = quality_result.get('is_good_quality', False)
        
        # Weighted combination
        combined_confidence = (
            weights['cnn'] * cnn_confidence +
            weights['color'] * color_confidence +
            weights['temporal'] * temporal_confidence +
            weights['quality'] * quality_confidence
        )
        
        # Decision based on confidence threshold
        is_real = combined_confidence >= self.config['confidence_threshold']
        
        # Additional safety check: if CNN strongly disagrees, be cautious
        if cnn_confidence < 0.3 and cnn_real == False:
            is_real = False
            combined_confidence = min(combined_confidence, 0.3)
        
        return {
            'is_real': is_real,
            'confidence': combined_confidence,
            'individual_results': {
                'cnn': {'decision': cnn_real, 'confidence': cnn_confidence},
                'color': {'decision': color_real, 'confidence': color_confidence},
                'temporal': {'decision': temporal_real, 'confidence': temporal_confidence},
                'quality': {'decision': quality_real, 'confidence': quality_confidence}
            }
        }
    
    def _update_history(self, image: np.ndarray, result: Dict):
        """Update frame and prediction history"""
        # Update frame history
        self.frame_history.append(image.copy())
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Update prediction history
        self.prediction_history.append(result.copy())
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result"""
        return {
            'is_real_face': False,
            'confidence': 0.0,
            'detailed_analysis': {},
            'processing_time': 0.0,
            'timestamp': time.time(),
            'error': error_message
        }
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        if not self.prediction_history:
            return {'total_frames': 0, 'real_detections': 0, 'fake_detections': 0}
        
        real_count = sum(1 for p in self.prediction_history if p.get('is_real', False))
        fake_count = len(self.prediction_history) - real_count
        
        return {
            'total_frames': len(self.prediction_history),
            'real_detections': real_count,
            'fake_detections': fake_count,
            'average_confidence': np.mean([p.get('confidence', 0.0) for p in self.prediction_history])
        }


# Step 2 Training Components
class AntiSpoofingTrainer:
    """
    Training class for Step 2 anti-spoofing CNN model
    Handles training, validation, and model saving
    """
    
    def __init__(self, model: EnhancedAntiSpoofingCNN, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Training configuration
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Training initialized on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs['logits'], target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs['logits'].data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs['logits'], target)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, save_path: str = 'antispoofing_model.pth'):
        """
        Full training loop with validation and model saving
        """
        best_val_acc = 0.0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_history': {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'train_accuracies': self.train_accuracies,
                        'val_accuracies': self.val_accuracies
                    }
                }, save_path)
                print(f'New best model saved with val accuracy: {val_acc:.2f}%')
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc


# Step 2 Enhanced Integration
class EnhancedAntiSpoofingDetector:
    """
    Enhanced Real-time Anti-Spoofing Detector with Step 2 CNN Integration
    
    Implements weighted voting system:
    - CNN (60%), Landmarks (20%), Challenges (20%)
    - 85% combined confidence threshold
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Initialize enhanced CNN model
        self.cnn_model = EnhancedAntiSpoofingCNN()
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded pre-trained anti-spoofing model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}. Using untrained model.")
        else:
            print("No pre-trained model found. Using randomly initialized model.")
        
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # Detection configuration based on Step 2 requirements
        self.config = {
            'combined_confidence_threshold': 0.85,  # 85% as specified
            'cnn_weight': 0.6,  # 60% weight for CNN
            'landmark_weight': 0.2,  # 20% weight for landmarks
            'challenge_weight': 0.2,  # 20% weight for challenges
            'input_size': (224, 224),
        }
        
        # Frame history for temporal analysis
        self.frame_history = []
        self.prediction_history = []
        self.max_history = 10
        
        print("Enhanced Anti-Spoofing Detector initialized with Step 2 configuration")
    
    def preprocess_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess image for CNN input"""
        try:
            # Resize to model input size
            resized = cv2.resize(image, self.config['input_size'])
            
            # Convert BGR to RGB
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = resized
            
            # Normalize to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None
    
    def run_cnn_inference(self, image: np.ndarray) -> Dict:
        """
        Run CNN inference for binary classification (Step 2)
        Returns: real vs fake classification with confidence
        """
        try:
            preprocessed = self.preprocess_image(image)
            if preprocessed is None:
                return {'is_real': False, 'cnn_confidence': 0.0, 'error': 'Preprocessing failed'}
            
            with torch.no_grad():
                outputs = self.cnn_model(preprocessed)
                
                # Get classification probabilities
                probs = F.softmax(outputs['logits'], dim=1)
                fake_prob = probs[0, 0].item()  # Class 0: fake
                real_prob = probs[0, 1].item()  # Class 1: real
                
                # Get confidence from model
                model_confidence = outputs['confidence'][0, 0].item()
                
                # Determine if real based on CNN
                is_real_cnn = real_prob > fake_prob and real_prob > 0.5
                
                # Use the higher probability as confidence
                cnn_confidence = max(real_prob, fake_prob)
                
                return {
                    'is_real': is_real_cnn,
                    'cnn_confidence': cnn_confidence,
                    'real_probability': real_prob,
                    'fake_probability': fake_prob,
                    'model_confidence': model_confidence,
                    'features': outputs['features'][0].cpu().numpy() if 'features' in outputs else None
                }
                
        except Exception as e:
            logger.error(f"CNN inference error: {e}")
            return {
                'is_real': False,
                'cnn_confidence': 0.0,
                'error': str(e)
            }
    
    def combine_detection_results(self, cnn_result: Dict, landmark_result: Dict, 
                                challenge_result: Dict) -> Dict:
        """
        Combine detection results using weighted voting (Step 2)
        CNN (60%), Landmarks (20%), Challenges (20%)
        Minimum 85% combined confidence to pass
        """
        
        # Extract individual confidences
        cnn_confidence = cnn_result.get('cnn_confidence', 0.0)
        cnn_is_real = cnn_result.get('is_real', False)
        
        # Landmark confidence based on detection quality
        landmark_confidence = 0.0
        landmark_is_real = False
        if landmark_result.get('landmarks_detected', False):
            landmark_confidence = 0.5
            if (landmark_result.get('head_movement', False) or 
                landmark_result.get('blink_count', 0) > 0):
                landmark_confidence = 0.8
                landmark_is_real = True
        
        # Challenge confidence
        challenge_confidence = challenge_result.get('completion_confidence', 0.0)
        challenge_is_real = challenge_result.get('completed', False)
        
        # Weighted combination as per Step 2 requirements
        weights = self.config
        combined_confidence = (
            weights['cnn_weight'] * (cnn_confidence if cnn_is_real else 0.0) +
            weights['landmark_weight'] * (landmark_confidence if landmark_is_real else 0.0) +
            weights['challenge_weight'] * (challenge_confidence if challenge_is_real else 0.0)
        )
        
        # Decision based on 85% threshold
        is_real_combined = combined_confidence >= self.config['combined_confidence_threshold']
        
        return {
            'is_real_face': is_real_combined,
            'combined_confidence': combined_confidence,
            'individual_results': {
                'cnn': {
                    'confidence': cnn_confidence,
                    'is_real': cnn_is_real,
                    'weight': weights['cnn_weight']
                },
                'landmarks': {
                    'confidence': landmark_confidence,
                    'is_real': landmark_is_real,
                    'weight': weights['landmark_weight']
                },
                'challenges': {
                    'confidence': challenge_confidence,
                    'is_real': challenge_is_real,
                    'weight': weights['challenge_weight']
                }
            },
            'threshold_met': combined_confidence >= self.config['combined_confidence_threshold'],
            'threshold': self.config['combined_confidence_threshold']
        }
    
    def detect_antispoofing_step2(self, image: np.ndarray, landmark_result: Optional[Dict] = None,
                                 challenge_result: Optional[Dict] = None) -> Dict:
        """
        Main detection function implementing Step 2 requirements
        """
        start_time = time.time()
        
        try:
            # Run CNN inference (primary detection method)
            cnn_result = self.run_cnn_inference(image)
            
            # Use provided landmark and challenge results, or defaults
            if landmark_result is None:
                landmark_result = {'landmarks_detected': False}
            
            if challenge_result is None:
                challenge_result = {'completed': False, 'completion_confidence': 0.0}
            
            # Combine all results using weighted voting
            combined_result = self.combine_detection_results(
                cnn_result, landmark_result, challenge_result
            )
            
            # Update history
            self._update_history(image, combined_result)
            
            processing_time = time.time() - start_time
            
            result = {
                'is_real_face': combined_result['is_real_face'],
                'confidence': combined_result['combined_confidence'],
                'cnn_result': cnn_result,
                'combined_analysis': combined_result,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'step2_implementation': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Step 2 anti-spoofing detection error: {e}")
            return {
                'is_real_face': False,
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'step2_implementation': True
            }
    
    def _update_history(self, image: np.ndarray, result: Dict):
        """Update frame and prediction history"""
        # Update frame history
        self.frame_history.append(image.copy())
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        # Update prediction history
        self.prediction_history.append(result.copy())
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        return {
            'model_type': 'EnhancedAntiSpoofingCNN',
            'step2_implementation': True,
            'configuration': self.config,
            'device': str(self.device),
            'frames_processed': len(self.prediction_history),
            'architecture': {
                'input_size': self.config['input_size'],
                'output_classes': 2,
                'binary_classification': True
            }
        }


if __name__ == "__main__":
    # Step 2 Testing and Training Example
    print("=== Step 2 Enhanced Anti-Spoofing CNN ===")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model and detector
    model = EnhancedAntiSpoofingCNN()
    detector = EnhancedAntiSpoofingDetector(device=device)
    
    print(f"Model info: {detector.get_model_info()}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = detector.detect_antispoofing_step2(dummy_image)
    print(f"Test result: {result}")
    
    print("Step 2 implementation ready for training and deployment!")

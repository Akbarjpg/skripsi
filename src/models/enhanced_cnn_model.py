"""
Enhanced CNN Model with Advanced Anti-Spoofing Capabilities
Phase 3: Texture Analysis, Frequency Domain Features, and Attention Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
from typing import Tuple, Optional, Dict, List
import math
import cv2

class TextureAnalysisBlock(nn.Module):
    """
    Texture analysis block for detecting print artifacts and screen pixelation
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Local Binary Pattern inspired convolutions
        self.lbp_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        
        # Gradient-based texture detection
        self.sobel_x = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
        
        # High-frequency detail detection
        self.detail_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        
        # Initialize Sobel filters
        self._init_sobel_filters()
        
        # Fusion layer
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def _init_sobel_filters(self):
        """Initialize Sobel edge detection filters"""
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Expand for all input/output channels
        in_channels = self.sobel_x.in_channels
        out_channels = self.sobel_x.out_channels
        
        sobel_x_weight = sobel_x_kernel.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1)
        sobel_y_weight = sobel_y_kernel.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1)
        
        self.sobel_x.weight.data = sobel_x_weight
        self.sobel_y.weight.data = sobel_y_weight
        
        # Freeze Sobel filters
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
    
    def forward(self, x):
        # LBP-inspired features
        lbp_features = self.lbp_conv(x)
        
        # Edge detection features
        sobel_x_features = self.sobel_x(x)
        sobel_y_features = self.sobel_y(x)
        
        # High-frequency details
        detail_features = self.detail_conv(x)
        
        # Combine all texture features
        combined = torch.cat([lbp_features, sobel_x_features, sobel_y_features, detail_features], dim=1)
        
        # Fusion and normalization
        output = self.fusion(combined)
        output = self.bn(output)
        output = self.activation(output)
        
        return output

class FrequencyDomainBlock(nn.Module):
    """
    Frequency domain analysis block for detecting digital artifacts
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # FFT feature extraction layers
        self.fft_conv = nn.Conv2d(in_channels * 2, out_channels // 2, kernel_size=3, padding=1)  # *2 for real+imag
        self.high_freq_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.low_freq_conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Apply FFT to detect frequency domain artifacts
        fft_features = []
        
        for i in range(channels):
            # Extract single channel
            channel = x[:, i:i+1, :, :]  # [B, 1, H, W]
            
            # Apply 2D FFT
            channel_complex = torch.fft.fft2(channel.squeeze(1))  # [B, H, W]
            
            # Extract real and imaginary parts
            fft_real = channel_complex.real.unsqueeze(1)  # [B, 1, H, W]
            fft_imag = channel_complex.imag.unsqueeze(1)  # [B, 1, H, W]
            
            fft_features.append(torch.cat([fft_real, fft_imag], dim=1))  # [B, 2, H, W]
        
        # Concatenate FFT features from all channels
        fft_combined = torch.cat(fft_features, dim=1)  # [B, channels*2, H, W]
        
        # Process FFT features
        fft_processed = self.fft_conv(fft_combined)
        
        # High and low frequency component analysis
        # Create frequency masks
        center_h, center_w = height // 2, width // 2
        y, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        y, x_coords = y.to(x.device), x_coords.to(x.device)
        
        # Distance from center (DC component)
        distances = torch.sqrt((y - center_h) ** 2 + (x_coords - center_w) ** 2)
        
        # High frequency mask (outer regions)
        high_freq_mask = (distances > min(height, width) * 0.3).float()
        low_freq_mask = (distances <= min(height, width) * 0.3).float()
        
        # Apply frequency masks
        high_freq_features = self.high_freq_conv(x * high_freq_mask.unsqueeze(0).unsqueeze(0))
        low_freq_features = self.low_freq_conv(x * low_freq_mask.unsqueeze(0).unsqueeze(0))
        
        # Combine all frequency domain features
        combined = torch.cat([fft_processed, high_freq_features, low_freq_features], dim=1)
        
        output = self.bn(combined)
        output = self.activation(output)
        
        return output

class SpatialAttentionModule(nn.Module):
    """
    Spatial attention focusing on eyes, nose, mouth regions
    """
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable region weights for face parts
        self.register_parameter('eye_weight', nn.Parameter(torch.ones(1)))
        self.register_parameter('nose_weight', nn.Parameter(torch.ones(1)))
        self.register_parameter('mouth_weight', nn.Parameter(torch.ones(1)))
        
    def create_face_region_mask(self, feature_map):
        """Create attention mask focusing on key facial regions"""
        batch_size, _, height, width = feature_map.shape
        
        # Create region masks based on typical face proportions
        mask = torch.zeros(height, width, device=feature_map.device)
        
        # Eye regions (upper third of face)
        eye_region_top = int(height * 0.25)
        eye_region_bottom = int(height * 0.45)
        eye_region_left = int(width * 0.15)
        eye_region_right = int(width * 0.85)
        
        mask[eye_region_top:eye_region_bottom, eye_region_left:eye_region_right] += self.eye_weight
        
        # Nose region (center of face)
        nose_region_top = int(height * 0.4)
        nose_region_bottom = int(height * 0.65)
        nose_region_left = int(width * 0.35)
        nose_region_right = int(width * 0.65)
        
        mask[nose_region_top:nose_region_bottom, nose_region_left:nose_region_right] += self.nose_weight
        
        # Mouth region (lower face)
        mouth_region_top = int(height * 0.6)
        mouth_region_bottom = int(height * 0.8)
        mouth_region_left = int(width * 0.25)
        mouth_region_right = int(width * 0.75)
        
        mask[mouth_region_top:mouth_region_bottom, mouth_region_left:mouth_region_right] += self.mouth_weight
        
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    def get_attention_map(self, x):
        """Get attention map for visualization"""
        attention_map = self.attention_conv(x)
        face_mask = self.create_face_region_mask(x)
        attention_map = attention_map * (1 + face_mask)
        return attention_map
    
    def forward(self, x):
        # Generate attention map
        attention_map = self.attention_conv(x)
        
        # Apply face region bias
        face_mask = self.create_face_region_mask(x)
        attention_map = attention_map * (1 + face_mask)
        
        # Apply attention
        attended_features = x * attention_map
        
        return attended_features, attention_map

class ChannelAttentionModule(nn.Module):
    """
    Channel attention for feature importance weighting
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def get_channel_weights(self, x):
        """Get channel attention weights for analysis"""
        batch_size, channels, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))
        return self.sigmoid(avg_out + max_out)
        
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_out = self.fc(self.max_pool(x).view(batch_size, channels))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        
        return x * attention

class EnhancedAntiSpoofingCNN(nn.Module):
    """
    Enhanced CNN with advanced anti-spoofing capabilities
    """
    
    def __init__(self, num_classes=2, input_channels=3, dropout_rate=0.5, 
                 use_texture_analysis=True, use_frequency_analysis=True, 
                 use_attention=True, use_uncertainty=True, backbone='resnet18'):
        super().__init__()
        
        self.use_texture_analysis = use_texture_analysis
        self.use_frequency_analysis = use_frequency_analysis
        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty
        
        # Enhanced input preprocessing
        self.input_norm = nn.BatchNorm2d(input_channels)
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Texture analysis branch
        if use_texture_analysis:
            self.texture_branch = nn.ModuleList([
                TextureAnalysisBlock(128, 128),  # Changed from 64 to 128
                TextureAnalysisBlock(256, 256),  # Already correct
                TextureAnalysisBlock(512, 512)   # Changed from 256 to 512
            ])
        
        # Frequency domain analysis branch
        if use_frequency_analysis:
            self.frequency_branch = nn.ModuleList([
                FrequencyDomainBlock(128, 128),  # Changed from 64 to 128
                FrequencyDomainBlock(256, 256),  # Already correct
                FrequencyDomainBlock(512, 512)   # Changed from 256 to 512
            ])
        
        # Main backbone layers
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # Attention modules
        if use_attention:
            self.spatial_attention1 = SpatialAttentionModule(128)
            self.spatial_attention2 = SpatialAttentionModule(256)
            self.spatial_attention3 = SpatialAttentionModule(512)
            
            self.channel_attention1 = ChannelAttentionModule(128)
            self.channel_attention2 = ChannelAttentionModule(256)
            self.channel_attention3 = ChannelAttentionModule(512)
        
        # Feature fusion
        fusion_channels = 512  # Base channels from layer3
        if use_texture_analysis:
            fusion_channels += 512  # Only from final texture layer (layer3)
        if use_frequency_analysis:
            fusion_channels += 512  # Only from final frequency layer (layer3)
        
        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(fusion_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Uncertainty quantification branch
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
        # Main classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create residual layer"""
        layers = []
        
        # First block with stride=2 for downsampling
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        # Input normalization
        x = self.input_norm(x)
        
        # Stem processing
        x = self.stem(x)  # [B, 64, 56, 56]
        
        # Store features for multi-branch processing
        features = []
        attention_maps = []
        
        # Layer 1 processing
        x = self.layer1(x)  # [B, 128, 28, 28]
        
        # Apply texture analysis
        if self.use_texture_analysis:
            texture_features = self.texture_branch[0](x)
            features.append(texture_features)
        
        # Apply frequency analysis
        if self.use_frequency_analysis:
            freq_features = self.frequency_branch[0](x)
            features.append(freq_features)
        
        # Apply attention
        if self.use_attention:
            x = self.channel_attention1(x)
            x, att_map = self.spatial_attention1(x)
            attention_maps.append(att_map)
        
        # Layer 2 processing
        x = self.layer2(x)  # [B, 256, 14, 14]
        
        # Apply texture analysis
        if self.use_texture_analysis:
            texture_features = self.texture_branch[1](x)
            features.append(texture_features)
        
        # Apply frequency analysis
        if self.use_frequency_analysis:
            freq_features = self.frequency_branch[1](x)
            features.append(freq_features)
        
        # Apply attention
        if self.use_attention:
            x = self.channel_attention2(x)
            x, att_map = self.spatial_attention2(x)
            attention_maps.append(att_map)
        
        # Layer 3 processing
        x = self.layer3(x)  # [B, 512, 7, 7]
        
        # Apply texture analysis
        if self.use_texture_analysis:
            texture_features = self.texture_branch[2](x)
            features.append(texture_features)
        
        # Apply frequency analysis
        if self.use_frequency_analysis:
            freq_features = self.frequency_branch[2](x)
            features.append(freq_features)
        
        # Apply attention
        if self.use_attention:
            x = self.channel_attention3(x)
            x, att_map = self.spatial_attention3(x)
            attention_maps.append(att_map)
        
        # Feature fusion
        # Combine main features with texture and frequency features
        fusion_features = [x]  # Main features [B, 512, 7, 7]
        
        if features:
            # Only use the LAST feature from each branch (most discriminative)
            # Texture features: use only the last one (from layer 3)
            # Frequency features: use only the last one (from layer 3)
            
            # Get the last texture and frequency features
            if self.use_texture_analysis and self.use_frequency_analysis:
                # Both enabled: features has 6 elements, take last 2 (texture + freq from layer 3)
                last_features = features[-2:]
            elif self.use_texture_analysis or self.use_frequency_analysis:
                # Only one enabled: features has 3 elements, take last 1
                last_features = features[-1:]
            else:
                last_features = []
            
            # Pool and add the last features
            for feat in last_features:
                pooled_feat = F.adaptive_avg_pool2d(feat, (x.size(2), x.size(3)))
                fusion_features.append(pooled_feat)
        
        # Concatenate all features
        if len(fusion_features) > 1:
            fused = torch.cat(fusion_features, dim=1)
        else:
            fused = x
        
        # Final feature processing
        final_features = self.feature_fusion(fused)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(final_features)
        
        # Classification
        logits = self.classifier(final_features)
        
        if return_attention:
            return logits, final_features, uncertainty, attention_maps
        else:
            return logits, final_features, uncertainty

class EnsembleAntiSpoofingModel(nn.Module):
    """
    Ensemble model with multiple architectures
    """
    
    def __init__(self, num_models=3, num_classes=2, use_uncertainty=True, 
                 voting_strategy='soft', voting='soft'):
        super().__init__()
        
        self.num_models = num_models
        self.use_uncertainty = use_uncertainty
        self.voting = voting_strategy if voting_strategy else voting
        
        # Create diverse models
        self.models = nn.ModuleList([
            EnhancedAntiSpoofingCNN(num_classes=num_classes, 
                                  use_texture_analysis=True, 
                                  use_frequency_analysis=True,
                                  use_attention=True,
                                  use_uncertainty=use_uncertainty),
            EnhancedAntiSpoofingCNN(num_classes=num_classes,
                                  use_texture_analysis=True,
                                  use_frequency_analysis=False,
                                  use_attention=True,
                                  use_uncertainty=use_uncertainty),
            EnhancedAntiSpoofingCNN(num_classes=num_classes,
                                  use_texture_analysis=False,
                                  use_frequency_analysis=True,
                                  use_attention=True,
                                  use_uncertainty=use_uncertainty)
        ])
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x, return_individual=False):
        outputs = []
        features = []
        uncertainties = []
        
        for model in self.models:
            logits, feat, uncertainty = model(x)
            outputs.append(logits)
            features.append(feat)
            uncertainties.append(uncertainty)
        
        # Ensemble prediction
        if self.voting == 'soft':
            # Weighted average of probabilities
            probs = [F.softmax(logits, dim=1) for logits in outputs]
            weights = F.softmax(self.ensemble_weights, dim=0)
            
            ensemble_prob = sum(w * prob for w, prob in zip(weights, probs))
            ensemble_logits = torch.log(ensemble_prob + 1e-8)  # Convert back to logits
        else:
            # Hard voting
            predictions = [torch.argmax(logits, dim=1) for logits in outputs]
            ensemble_prediction = torch.mode(torch.stack(predictions), dim=0)[0]
            ensemble_logits = F.one_hot(ensemble_prediction, num_classes=outputs[0].size(1)).float()
        
        # Combine uncertainties
        ensemble_uncertainty = torch.mean(torch.stack(uncertainties), dim=0)
        
        # Combine features
        ensemble_features = torch.mean(torch.stack(features), dim=0)
        
        if return_individual:
            return ensemble_logits, ensemble_features, ensemble_uncertainty, outputs, features, uncertainties
        else:
            return ensemble_logits, ensemble_features, ensemble_uncertainty

def create_enhanced_model(model_type='enhanced', **kwargs):
    """
    Factory function for creating enhanced anti-spoofing models
    """
    if model_type == 'enhanced':
        return EnhancedAntiSpoofingCNN(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleAntiSpoofingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test enhanced model
    print("🔬 Testing Enhanced Anti-Spoofing CNN Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test different configurations
    configs = [
        {'name': 'Full Enhanced', 'use_texture_analysis': True, 'use_frequency_analysis': True, 'use_attention': True},
        {'name': 'Texture + Attention', 'use_texture_analysis': True, 'use_frequency_analysis': False, 'use_attention': True},
        {'name': 'Frequency + Attention', 'use_texture_analysis': False, 'use_frequency_analysis': True, 'use_attention': True},
    ]
    
    for config in configs:
        print(f"\\n📊 Testing {config['name']} Configuration:")
        
        model = EnhancedAntiSpoofingCNN(
            use_texture_analysis=config['use_texture_analysis'],
            use_frequency_analysis=config['use_frequency_analysis'],
            use_attention=config['use_attention']
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            start_time = time.time()
            logits, features, uncertainty = model(dummy_input)
            inference_time = time.time() - start_time
        
        print(f"   Output shape: {logits.shape}")
        print(f"   Feature shape: {features.shape}")
        print(f"   Uncertainty shape: {uncertainty.shape}")
        print(f"   Inference time: {inference_time*1000:.1f}ms")
    
    # Test ensemble model
    print(f"\\n🎯 Testing Ensemble Model:")
    ensemble_model = EnsembleAntiSpoofingModel(num_models=3).to(device)
    
    total_params = sum(p.numel() for p in ensemble_model.parameters())
    print(f"   Ensemble parameters: {total_params:,}")
    
    with torch.no_grad():
        start_time = time.time()
        logits, features, uncertainty = ensemble_model(dummy_input)
        inference_time = time.time() - start_time
    
    print(f"   Ensemble output shape: {logits.shape}")
    print(f"   Ensemble inference time: {inference_time*1000:.1f}ms")
    
    print("\\n✅ Enhanced CNN Model Testing Complete!")

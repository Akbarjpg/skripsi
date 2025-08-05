"""
CNN Model untuk Face Anti-Spoofing Liveness Detection
Menggunakan arsitektur yang dioptimalkan untuk real-time inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution untuk efisiensi komputasi
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block untuk channel attention
    """
    def __init__(self, channels, reduction=16):
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

class ResidualBlock(nn.Module):
    """
    Residual Block dengan SE attention
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        if self.stride != 1:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
            residual = F.avg_pool2d(residual, kernel_size=2, stride=2)
            
        out = self.se(out)
        out += self.skip(residual)
        out = F.relu(out, inplace=True)
        
        return out

class LivenessDetectionCNN(nn.Module):
    """
    CNN Model untuk Face Liveness Detection
    Arsitektur yang dioptimalkan untuk real-time inference
    """
    
    def __init__(self, num_classes=2, input_channels=3, dropout_rate=0.5):
        super().__init__()
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 64, 2)    # 56x56
        self.layer2 = self._make_layer(64, 128, 2)   # 28x28
        self.layer3 = self._make_layer(128, 256, 2)  # 14x14
        self.layer4 = self._make_layer(256, 512, 2)  # 7x7
        
        # Global pooling dan classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
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
    
    def forward(self, x):
        # Extract features
        x = self.stem(x)        # [B, 32, 56, 56]
        x = self.layer1(x)      # [B, 64, 28, 28]
        x = self.layer2(x)      # [B, 128, 14, 14]
        x = self.layer3(x)      # [B, 256, 7, 7]
        x = self.layer4(x)      # [B, 512, 4, 4]
        
        # Global pooling
        x = self.global_pool(x) # [B, 512, 1, 1]
        x = torch.flatten(x, 1) # [B, 512]
        
        # Feature fusion
        features = self.feature_fusion(x)  # [B, 128]
        
        # Classification
        logits = self.classifier(features)  # [B, 2]
        
        return logits, features

class PretrainedLivenessModel(nn.Module):
    """
    Model menggunakan pretrained backbone (ResNet, EfficientNet, dll)
    """
    
    def __init__(self, backbone='resnet18', num_classes=2, pretrained=True, dropout_rate=0.5):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = self.backbone.classifier[3].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Backbone {backbone} tidak didukung")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

class MultiScaleLivenessModel(nn.Module):
    """
    Multi-scale model yang menggunakan berbagai ukuran input
    untuk meningkatkan robustness
    """
    
    def __init__(self, base_model_class=LivenessDetectionCNN, num_classes=2):
        super().__init__()
        
        # Models untuk berbagai scale
        self.model_224 = base_model_class(num_classes=num_classes)
        self.model_112 = base_model_class(num_classes=num_classes)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # 128 features from each model
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Original scale (224x224)
        logits_224, features_224 = self.model_224(x)
        
        # Downsampled scale (112x112)
        x_112 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        logits_112, features_112 = self.model_112(x_112)
        
        # Combine features
        combined_features = torch.cat([features_224, features_112], dim=1)
        final_logits = self.fusion(combined_features)
        
        return final_logits, combined_features

def create_model(model_type='custom', **kwargs):
    """
    Factory function untuk membuat model
    
    Args:
        model_type: Tipe model ('custom', 'pretrained', 'multiscale')
        **kwargs: Parameter tambahan
    
    Returns:
        model: PyTorch model
    """
    
    if model_type == 'custom':
        return LivenessDetectionCNN(**kwargs)
    
    elif model_type == 'pretrained':
        return PretrainedLivenessModel(**kwargs)
    
    elif model_type == 'multiscale':
        return MultiScaleLivenessModel(**kwargs)
    
    else:
        raise ValueError(f"Model type {model_type} tidak didukung")

def count_parameters(model):
    """
    Hitung jumlah parameter model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def model_summary(model, input_size=(3, 224, 224)):
    """
    Print model summary
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    
    # Count parameters
    count_parameters(model)
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        output, features = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Feature shape: {features.shape}")

if __name__ == "__main__":
    # Test models
    print("=== Testing Custom CNN ===")
    model_custom = create_model('custom', num_classes=2)
    model_summary(model_custom)
    
    print("\\n=== Testing Pretrained ResNet18 ===")
    model_pretrained = create_model('pretrained', backbone='resnet18', num_classes=2)
    model_summary(model_pretrained)
    
    print("\\n=== Testing Multi-scale Model ===")
    model_multiscale = create_model('multiscale', num_classes=2)
    model_summary(model_multiscale)

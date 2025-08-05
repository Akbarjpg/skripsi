"""
Simple test model untuk mengatasi dimension issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLivenessModel(nn.Module):
    """
    Model CNN sederhana untuk testing
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = torch.flatten(pooled, 1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# Monkey patch untuk mengatasi model complex yang bermasalah
def create_simple_model():
    return SimpleLivenessModel()

if __name__ == "__main__":
    # Test model
    model = SimpleLivenessModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Simple model test passed!")

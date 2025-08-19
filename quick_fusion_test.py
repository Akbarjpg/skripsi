#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.enhanced_cnn_model import EnhancedAntiSpoofingCNN

print("Testing Enhanced CNN Feature Fusion...")

# Test with default parameters (both texture and frequency enabled)
model = EnhancedAntiSpoofingCNN()
print(f"Model fusion_channels setting: {model.feature_fusion[2].in_features}")

# Test with test parameters (uncertainty only)
test_model = EnhancedAntiSpoofingCNN(
    num_classes=2,
    use_uncertainty=True,
    backbone='resnet18'
)
print(f"Test model fusion_channels setting: {test_model.feature_fusion[2].in_features}")

# Try a simple forward pass with test model
try:
    x = torch.randn(1, 3, 224, 224)
    out = test_model(x)
    print("✅ Test model working!")
except Exception as e:
    print(f"❌ Test model error: {e}")

# Try with default model
try:
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print("✅ Default model working!")
except Exception as e:
    print(f"❌ Default model error: {e}")

#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.enhanced_cnn_model import EnhancedAntiSpoofingCNN

print("🧪 Testing Enhanced CNN Model...")

try:
    model = EnhancedAntiSpoofingCNN()
    x = torch.randn(1, 3, 224, 224)
    print("Forward pass...")
    out = model(x)
    print("✅ Enhanced CNN working!")
    print(f"Output shapes: {[o.shape if isinstance(o, torch.Tensor) else type(o) for o in out]}")
except Exception as e:
    print(f"❌ Enhanced CNN error: {e}")

# Test with the same parameters as test
try:
    test_model = EnhancedAntiSpoofingCNN(
        num_classes=2,
        use_uncertainty=True,
        backbone='resnet18'
    )
    x = torch.randn(1, 3, 224, 224)
    print("Test model forward pass...")
    out = test_model(x)
    print("✅ Test model working!")
    print(f"Output shapes: {[o.shape if isinstance(o, torch.Tensor) else type(o) for o in out]}")
except Exception as e:
    print(f"❌ Test model error: {e}")

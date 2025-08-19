"""
Quick validation for Phase 3 critical fixes
"""
import sys
import os
sys.path.append('src')

print("🔧 Testing Critical Phase 3 Fixes...")

# Test 1: Enhanced CNN with correct channel sizes
try:
    from models.enhanced_cnn_model import EnhancedAntiSpoofingCNN
    import torch
    
    model = EnhancedAntiSpoofingCNN(num_classes=2, use_uncertainty=True)
    test_input = torch.randn(1, 3, 224, 224)
    
    # Test forward pass
    logits, features, uncertainty = model(test_input)
    print(f"✅ Enhanced CNN forward pass successful: {logits.shape}, {features.shape}, {uncertainty.shape}")
    
except Exception as e:
    print(f"❌ Enhanced CNN test failed: {e}")

# Test 2: Individual components
try:
    from models.enhanced_cnn_model import TextureAnalysisBlock, FrequencyDomainBlock, SpatialAttentionModule
    
    # Test texture block with correct channels
    texture_block = TextureAnalysisBlock(128, 128)
    test_128 = torch.randn(1, 128, 28, 28)
    texture_out = texture_block(test_128)
    print(f"✅ TextureAnalysisBlock (128 channels): {texture_out.shape}")
    
    # Test spatial attention tuple return
    spatial_attn = SpatialAttentionModule(64)
    test_64 = torch.randn(1, 64, 56, 56)
    attended, attention_map = spatial_attn(test_64)
    print(f"✅ SpatialAttention tuple return: {attended.shape}, {attention_map.shape}")
    
except Exception as e:
    print(f"❌ Component test failed: {e}")

# Test 3: NumPy compatibility
try:
    from models.training import EnhancedEarlyStopping
    import numpy as np
    
    # Test that np.inf works
    early_stop = EnhancedEarlyStopping(patience=5, mode='min')
    print(f"✅ NumPy 2.0 compatibility: best_score = {early_stop.best_score}")
    
except Exception as e:
    print(f"❌ NumPy compatibility test failed: {e}")

print("\n🎯 Critical fixes validation complete!")

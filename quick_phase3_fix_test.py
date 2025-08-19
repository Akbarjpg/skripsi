"""
Quick validation test for Phase 3 fixes
"""
import sys
import os
sys.path.append('src')

print("🔧 Testing Phase 3 Fixes...")

# Test 1: Enhanced CNN import
try:
    from models.enhanced_cnn_model import EnhancedAntiSpoofingCNN
    print("✅ Enhanced CNN import successful")
    
    # Test model creation
    model = EnhancedAntiSpoofingCNN(num_classes=2, use_uncertainty=True)
    print("✅ Enhanced CNN creation successful")
    
    # Test forward pass
    import torch
    test_input = torch.randn(1, 3, 224, 224)
    logits, features, uncertainty = model(test_input)
    print(f"✅ Forward pass successful: logits{logits.shape}, features{features.shape}, uncertainty{uncertainty.shape}")
    
except Exception as e:
    print(f"❌ Enhanced CNN test failed: {e}")

# Test 2: Ensemble model
try:
    from models.enhanced_cnn_model import EnsembleAntiSpoofingModel
    ensemble = EnsembleAntiSpoofingModel(num_models=3, use_uncertainty=True, voting_strategy='soft')
    print("✅ Ensemble model creation successful")
    
    logits, features, uncertainty = ensemble(test_input)
    print(f"✅ Ensemble forward pass successful: {logits.shape}")
    
except Exception as e:
    print(f"❌ Ensemble model test failed: {e}")

# Test 3: Minimal augmentation
try:
    from models.minimal_antispoofing_augmentation import PrintAttackAugmentation
    aug = PrintAttackAugmentation()
    print("✅ Minimal augmentation import successful")
    
    import numpy as np
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = aug.apply_paper_texture_effect(test_image)
    print(f"✅ Augmentation application successful: {result.shape}")
    
except Exception as e:
    print(f"❌ Augmentation test failed: {e}")

# Test 4: Training components
try:
    from models.training import UncertaintyLoss, EnhancedEarlyStopping
    loss_fn = UncertaintyLoss()
    early_stop = EnhancedEarlyStopping(patience=5)
    print("✅ Training components successful")
    
except Exception as e:
    print(f"❌ Training components test failed: {e}")

print("\n🎯 Phase 3 fixes validation complete!")

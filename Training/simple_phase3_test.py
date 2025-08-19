"""
Simple Phase 3 validation test
"""
import sys
import os

# Add src to path
sys.path.append('src')

print("="*50)
print("🚀 PHASE 3 ENHANCED CNN VALIDATION")
print("="*50)

try:
    print("1. Testing basic imports...")
    import torch
    import numpy as np
    print(f"   ✅ PyTorch {torch.__version__}")
    
    print("2. Testing enhanced CNN model...")
    from models.enhanced_cnn_model import EnhancedAntiSpoofingCNN
    
    device = torch.device('cpu')  # Use CPU for testing
    model = EnhancedAntiSpoofingCNN(num_classes=2, use_uncertainty=True)
    
    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224)
    logits, features, uncertainty = model(test_input)
    
    print(f"   ✅ Model output shapes:")
    print(f"      Logits: {logits.shape}")
    print(f"      Features: {features.shape}")
    print(f"      Uncertainty: {uncertainty.shape}")
    
    print("3. Testing augmentation...")
    from models.minimal_antispoofing_augmentation import PrintAttackAugmentation
    
    aug = PrintAttackAugmentation()
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    augmented = aug.apply_paper_texture_effect(test_image)
    print(f"   ✅ Augmentation working: {augmented.shape}")
    
    print("4. Testing training components...")
    from models.training import UncertaintyLoss, EnhancedEarlyStopping
    
    loss_fn = UncertaintyLoss()
    early_stop = EnhancedEarlyStopping(patience=5)
    print("   ✅ Training components imported")
    
    print("\n🎉 PHASE 3 BASIC VALIDATION PASSED!")
    print("✅ Enhanced CNN architecture working")
    print("✅ Anti-spoofing augmentation working") 
    print("✅ Training pipeline components working")
    
except Exception as e:
    print(f"❌ Validation failed: {str(e)}")
    import traceback
    traceback.print_exc()

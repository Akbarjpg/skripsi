#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test if the train_epoch method signature is correct
import inspect
from src.models.training import EnhancedModelTrainer

print("Testing EnhancedModelTrainer.train_epoch method signature...")

# Get the method signature
train_epoch_method = getattr(EnhancedModelTrainer, 'train_epoch')
signature = inspect.signature(train_epoch_method)

print(f"Method signature: {signature}")
print(f"Parameters: {list(signature.parameters.keys())}")

# Test creating a trainer
try:
    from src.models.enhanced_cnn_model import EnhancedAntiSpoofingCNN
    dummy_model = EnhancedAntiSpoofingCNN()
    
    trainer = EnhancedModelTrainer(
        model=dummy_model,  # Use actual model instead of None
        train_loader=None,
        val_loader=None,
        device='cpu',
        model_save_path='test',
        log_dir='test_logs',
        use_uncertainty=True
    )
    print("✅ EnhancedModelTrainer created successfully")
    print(f"Trainer has train_epoch method: {hasattr(trainer, 'train_epoch')}")
    
    # Check method on instance
    if hasattr(trainer, 'train_epoch'):
        instance_signature = inspect.signature(trainer.train_epoch)
        print(f"Instance method signature: {instance_signature}")
        print(f"Instance parameters: {list(instance_signature.parameters.keys())}")
        
except Exception as e:
    print(f"❌ Error creating trainer: {e}")
    import traceback
    traceback.print_exc()

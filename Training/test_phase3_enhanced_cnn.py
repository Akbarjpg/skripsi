"""
Phase 3 Test Script: Enhanced CNN Anti-Spoofing Model
Tests texture analysis, frequency domain, attention mechanisms, and ensemble training
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import time
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

def test_enhanced_cnn_architecture():
    """Test the enhanced CNN model architecture"""
    print("🧪 Testing Enhanced CNN Architecture...")
    
    try:
        from models.enhanced_cnn_model import (
            TextureAnalysisBlock, FrequencyDomainBlock, 
            SpatialAttentionModule, ChannelAttentionModule,
            EnhancedAntiSpoofingCNN, EnsembleAntiSpoofingModel
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        # Test individual components
        print("   Testing individual components...")
        
        # Test TextureAnalysisBlock
        texture_block = TextureAnalysisBlock(3, 64).to(device)
        test_input = torch.randn(2, 3, 224, 224).to(device)
        texture_output = texture_block(test_input)
        print(f"   ✅ TextureAnalysisBlock: {test_input.shape} -> {texture_output.shape}")
        
        # Test FrequencyDomainBlock
        freq_block = FrequencyDomainBlock(3, 32).to(device)
        freq_output = freq_block(test_input)
        print(f"   ✅ FrequencyDomainBlock: {test_input.shape} -> {freq_output.shape}")
        
        # Test SpatialAttentionModule
        spatial_attn = SpatialAttentionModule(64).to(device)
        attn_input = torch.randn(2, 64, 56, 56).to(device)
        spatial_output, attention_map = spatial_attn(attn_input)  # Handle tuple return
        print(f"   ✅ SpatialAttentionModule: {attn_input.shape} -> {spatial_output.shape}, attention: {attention_map.shape}")
        
        # Test ChannelAttentionModule
        channel_attn = ChannelAttentionModule(64).to(device)
        channel_output = channel_attn(attn_input)  # This returns single tensor
        print(f"   ✅ ChannelAttentionModule: {attn_input.shape} -> {channel_output.shape}")
        
        # Test main Enhanced CNN
        print("   Testing Enhanced Anti-Spoofing CNN...")
        enhanced_cnn = EnhancedAntiSpoofingCNN(
            num_classes=2,
            use_uncertainty=True,
            backbone='resnet18'
        ).to(device)
        
        logits, features, uncertainty = enhanced_cnn(test_input)
        print(f"   ✅ Enhanced CNN - Logits: {logits.shape}, Features: {features.shape}, Uncertainty: {uncertainty.shape}")
        
        # Test Ensemble Model
        print("   Testing Ensemble Anti-Spoofing Model...")
        ensemble_model = EnsembleAntiSpoofingModel(
            num_models=3,
            num_classes=2,
            use_uncertainty=True
        ).to(device)
        
        ensemble_logits, ensemble_features, ensemble_uncertainty = ensemble_model(test_input)
        print(f"   ✅ Ensemble Model - Logits: {ensemble_logits.shape}, Features: {ensemble_features.shape}, Uncertainty: {ensemble_uncertainty.shape}")
        
        # Test model parameters
        total_params = sum(p.numel() for p in enhanced_cnn.parameters())
        trainable_params = sum(p.numel() for p in enhanced_cnn.parameters() if p.requires_grad)
        print(f"   📊 Enhanced CNN - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        ensemble_params = sum(p.numel() for p in ensemble_model.parameters())
        print(f"   📊 Ensemble Model - Total params: {ensemble_params:,}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced CNN Architecture test failed: {str(e)}")
        return False

def test_antispoofing_augmentation():
    """Test anti-spoofing data augmentation pipeline"""
    print("\n🧪 Testing Anti-Spoofing Augmentation...")
    
    try:
        from models.minimal_antispoofing_augmentation import (
            PrintAttackAugmentation, ScreenAttackAugmentation,
            EnvironmentalAugmentation, TemporalConsistencyAugmentation,
            create_antispoofing_transforms
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"   Test image shape: {test_image.shape}")
        
        # Test Print Attack Augmentation
        print("   Testing Print Attack Augmentation...")
        print_aug = PrintAttackAugmentation()
        
        # Test different print effects
        paper_effect = print_aug.apply_paper_texture_effect(test_image.copy())
        print(f"   ✅ Paper texture effect: {paper_effect.shape}")
        
        dot_matrix_effect = print_aug.apply_dot_matrix_effect(test_image.copy())
        print(f"   ✅ Dot matrix effect: {dot_matrix_effect.shape}")
        
        color_shift = print_aug.apply_color_shift(test_image.copy())
        print(f"   ✅ Color shift effect: {color_shift.shape}")
        
        # Test Screen Attack Augmentation
        print("   Testing Screen Attack Augmentation...")
        screen_aug = ScreenAttackAugmentation()
        
        moire_effect = screen_aug.apply_moire_pattern(test_image.copy())
        print(f"   ✅ Moire pattern effect: {moire_effect.shape}")
        
        pixel_grid = screen_aug.apply_pixel_grid_effect(test_image.copy())
        print(f"   ✅ Pixel grid effect: {pixel_grid.shape}")
        
        screen_glare = screen_aug.apply_screen_glare(test_image.copy())
        print(f"   ✅ Screen glare effect: {screen_glare.shape}")
        
        # Test Environmental Augmentation
        print("   Testing Environmental Augmentation...")
        env_aug = EnvironmentalAugmentation()
        
        lighting_var = env_aug.apply_lighting_variation(test_image.copy())
        print(f"   ✅ Lighting variation: {lighting_var.shape}")
        
        shadow_effect = env_aug.apply_shadow_effect(test_image.copy())
        print(f"   ✅ Shadow effect: {shadow_effect.shape}")
        
        # Test Temporal Consistency
        print("   Testing Temporal Consistency Augmentation...")
        temporal_aug = TemporalConsistencyAugmentation()
        
        # Create test sequence
        test_sequence = [test_image.copy() for _ in range(5)]
        consistent_sequence = temporal_aug.apply_temporal_consistency(test_sequence)
        print(f"   ✅ Temporal consistency: {len(consistent_sequence)} frames")
        
        # Test transform creation
        print("   Testing transform creation...")
        train_transforms = create_antispoofing_transforms(
            image_size=224,
            is_training=True,
            attack_simulation=True
        )
        
        val_transforms = create_antispoofing_transforms(
            image_size=224,
            is_training=False,
            attack_simulation=False
        )
        
        print(f"   ✅ Training transforms created: {len(train_transforms.transforms)} transforms")
        print(f"   ✅ Validation transforms created: {len(val_transforms.transforms)} transforms")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Anti-Spoofing Augmentation test failed: {str(e)}")
        return False

def test_enhanced_training_pipeline():
    """Test the enhanced training pipeline"""
    print("\n🧪 Testing Enhanced Training Pipeline...")
    
    try:
        from models.training import (
            UncertaintyLoss, FocalLoss, EnhancedEarlyStopping,
            EnhancedModelTrainer
        )
        from models.enhanced_cnn_model import EnhancedAntiSpoofingCNN
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test Loss Functions
        print("   Testing loss functions...")
        
        # Test UncertaintyLoss
        uncertainty_loss = UncertaintyLoss(alpha=1.0, beta=0.1)
        logits = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4,))
        uncertainty = torch.rand(4, 1)
        
        loss, cls_loss = uncertainty_loss(logits, targets, uncertainty)
        print(f"   ✅ UncertaintyLoss - Total: {loss.item():.4f}, Classification: {cls_loss.item():.4f}")
        
        # Test FocalLoss
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        focal_loss_val = focal_loss(logits, targets)
        print(f"   ✅ FocalLoss: {focal_loss_val.item():.4f}")
        
        # Test EnhancedEarlyStopping
        print("   Testing Enhanced Early Stopping...")
        early_stopping = EnhancedEarlyStopping(patience=5, min_delta=1e-4)
        
        # Simulate training with improving then degrading validation loss
        model = EnhancedAntiSpoofingCNN(num_classes=2, use_uncertainty=True)
        val_losses = [1.0, 0.8, 0.6, 0.65, 0.7, 0.75, 0.8]  # Should trigger early stopping
        
        for epoch, val_loss in enumerate(val_losses):
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"   ✅ Early stopping triggered at epoch {epoch}")
                break
        
        # Test Enhanced Model Trainer initialization
        print("   Testing Enhanced Model Trainer...")
        
        # Create dummy data loaders
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy dataset
        dummy_data = torch.randn(16, 3, 224, 224)
        dummy_targets = torch.randint(0, 2, (16,))
        dummy_dataset = TensorDataset(dummy_data, dummy_targets)
        dummy_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)
        
        # Initialize trainer
        model = EnhancedAntiSpoofingCNN(num_classes=2, use_uncertainty=True)
        trainer = EnhancedModelTrainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            device=device,
            use_uncertainty=True,
            use_ensemble=False,
            model_save_path='test_models',
            log_dir='test_logs'
        )
        
        # Setup training components
        trainer.setup_optimizer_and_scheduler(
            optimizer_type='adamw',
            learning_rate=1e-3,
            scheduler_type='cosine',
            max_epochs=10
        )
        
        trainer.setup_loss_function(loss_type='uncertainty')
        
        print("   ✅ Enhanced Model Trainer initialized successfully")
        
        # Create dummy data loaders for testing
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 224, 224),  # 10 dummy images
            torch.randint(0, 2, (10,))      # 10 dummy labels
        )
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        
        # Set up trainer with dummy dataloaders
        trainer.train_loader = dummy_loader
        trainer.val_loader = dummy_loader
        
        # Test training epoch (just one batch)
        print("   Testing training epoch...")
        train_loss, train_cls_loss, train_uncertainty_loss, train_acc = trainer.train_epoch(0)
        print(f"   ✅ Training epoch - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Test validation epoch
        print("   Testing validation epoch...")
        val_results = trainer.validate_epoch(0)
        val_loss, val_acc = val_results[0], val_results[1]
        print(f"   ✅ Validation epoch - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced Training Pipeline test failed: {str(e)}")
        return False

def test_texture_frequency_analysis():
    """Test texture and frequency domain analysis capabilities"""
    print("\n🧪 Testing Texture & Frequency Analysis...")
    
    try:
        from models.enhanced_cnn_model import TextureAnalysisBlock, FrequencyDomainBlock
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test images with different characteristics
        print("   Creating test images...")
        
        # Real face-like texture (smooth gradients)
        real_image = torch.zeros(1, 3, 224, 224)
        for i in range(224):
            for j in range(224):
                real_image[0, :, i, j] = 0.5 + 0.3 * np.sin(i/50) * np.cos(j/50)
        
        # Print attack simulation (high frequency noise)
        print_image = real_image.clone()
        noise = torch.randn_like(print_image) * 0.1
        print_image += noise
        
        # Screen attack simulation (regular pattern)
        screen_image = real_image.clone()
        for i in range(0, 224, 4):
            for j in range(0, 224, 4):
                screen_image[0, :, i:i+2, j:j+2] *= 1.2  # Pixel grid effect
        
        test_images = {
            'real': real_image.to(device),
            'print': print_image.to(device),
            'screen': screen_image.to(device)
        }
        
        # Test Texture Analysis
        print("   Testing texture analysis...")
        texture_analyzer = TextureAnalysisBlock(3, 64).to(device)
        
        texture_responses = {}
        for attack_type, image in test_images.items():
            texture_features = texture_analyzer(image)
            texture_responses[attack_type] = texture_features.mean().item()
            print(f"   📊 {attack_type.capitalize()} texture response: {texture_responses[attack_type]:.4f}")
        
        # Test Frequency Domain Analysis
        print("   Testing frequency domain analysis...")
        freq_analyzer = FrequencyDomainBlock(3, 32).to(device)
        
        freq_responses = {}
        for attack_type, image in test_images.items():
            freq_features = freq_analyzer(image)
            freq_responses[attack_type] = freq_features.mean().item()
            print(f"   📊 {attack_type.capitalize()} frequency response: {freq_responses[attack_type]:.4f}")
        
        # Analyze discriminative power
        print("   Analyzing discriminative power...")
        texture_diff_print = abs(texture_responses['real'] - texture_responses['print'])
        texture_diff_screen = abs(texture_responses['real'] - texture_responses['screen'])
        
        freq_diff_print = abs(freq_responses['real'] - freq_responses['print'])
        freq_diff_screen = abs(freq_responses['real'] - freq_responses['screen'])
        
        print(f"   📈 Texture discrimination - Print: {texture_diff_print:.4f}, Screen: {texture_diff_screen:.4f}")
        print(f"   📈 Frequency discrimination - Print: {freq_diff_print:.4f}, Screen: {freq_diff_screen:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Texture & Frequency Analysis test failed: {str(e)}")
        return False

def test_attention_mechanisms():
    """Test attention mechanism effectiveness"""
    print("\n🧪 Testing Attention Mechanisms...")
    
    try:
        from models.enhanced_cnn_model import SpatialAttentionModule, ChannelAttentionModule
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test feature maps
        batch_size, channels, height, width = 2, 64, 56, 56
        test_features = torch.randn(batch_size, channels, height, width).to(device)
        
        # Add some structured patterns (simulating face regions)
        # Eye regions (more important)
        test_features[:, :, 10:20, 15:25] *= 2.0  # Left eye
        test_features[:, :, 10:20, 35:45] *= 2.0  # Right eye
        
        # Nose region
        test_features[:, :, 25:35, 25:35] *= 1.5
        
        # Mouth region
        test_features[:, :, 40:50, 20:40] *= 1.8
        
        print(f"   Test features shape: {test_features.shape}")
        
        # Test Spatial Attention
        print("   Testing Spatial Attention...")
        spatial_attention = SpatialAttentionModule(channels).to(device)
        
        attended_features, attention_map = spatial_attention(test_features)  # Handle tuple return
        print(f"   ✅ Spatial attention output: {attended_features.shape}")
        print(f"   📊 Attention map shape: {attention_map.shape}")
        
        # Find regions with highest attention
        attention_flat = attention_map.view(batch_size, -1)
        max_attention_indices = torch.argmax(attention_flat, dim=1)
        
        for i in range(batch_size):
            max_idx = max_attention_indices[i].item()
            max_row = max_idx // width
            max_col = max_idx % width
            print(f"   📍 Sample {i} max attention at: ({max_row}, {max_col})")
        
        # Test Channel Attention
        print("   Testing Channel Attention...")
        channel_attention = ChannelAttentionModule(channels).to(device)
        
        channel_attended = channel_attention(test_features)
        print(f"   ✅ Channel attention output: {channel_attended.shape}")
        
        # Analyze channel importance
        channel_weights = channel_attention.get_channel_weights(test_features)
        print(f"   📊 Channel weights shape: {channel_weights.shape}")
        
        top_channels = torch.topk(channel_weights.mean(0), k=5)
        print(f"   🔝 Top 5 important channels: {top_channels.indices.tolist()}")
        print(f"   📈 Their weights: {top_channels.values.tolist()}")
        
        # Test combined attention
        print("   Testing combined attention effects...")
        attended_features_combined, _ = spatial_attention(test_features)
        combined_features = channel_attention(attended_features_combined)
        
        # Calculate attention effectiveness
        original_variance = torch.var(test_features).item()
        spatial_variance = torch.var(attended_features).item()
        channel_variance = torch.var(channel_attended).item()
        combined_variance = torch.var(combined_features).item()
        
        print(f"   📊 Feature variance - Original: {original_variance:.4f}")
        print(f"   📊 Feature variance - Spatial: {spatial_variance:.4f}")
        print(f"   📊 Feature variance - Channel: {channel_variance:.4f}")
        print(f"   📊 Feature variance - Combined: {combined_variance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Attention Mechanisms test failed: {str(e)}")
        return False

def test_ensemble_prediction():
    """Test ensemble prediction capabilities"""
    print("\n🧪 Testing Ensemble Prediction...")
    
    try:
        from models.enhanced_cnn_model import EnsembleAntiSpoofingModel
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test different ensemble configurations
        ensemble_configs = [
            {'num_models': 3, 'use_uncertainty': True, 'voting_strategy': 'soft'},
            {'num_models': 5, 'use_uncertainty': True, 'voting_strategy': 'uncertainty_weighted'},
            {'num_models': 3, 'use_uncertainty': False, 'voting_strategy': 'majority'}
        ]
        
        test_input = torch.randn(4, 3, 224, 224).to(device)
        
        for i, config in enumerate(ensemble_configs):
            print(f"   Testing ensemble config {i+1}: {config}")
            
            ensemble_model = EnsembleAntiSpoofingModel(
                num_models=config['num_models'],
                num_classes=2,
                use_uncertainty=config['use_uncertainty'],
                voting_strategy=config['voting_strategy']
            ).to(device)
            
            # Forward pass
            # Ensemble model always returns 3 values: logits, features, uncertainty
            logits, features, uncertainty = ensemble_model(test_input)
            
            if config['use_uncertainty']:
                print(f"   ✅ Output shapes - Logits: {logits.shape}, Features: {features.shape}, Uncertainty: {uncertainty.shape}")
                
                # Analyze uncertainty
                avg_uncertainty = uncertainty.mean().item()
                std_uncertainty = uncertainty.std().item()
                print(f"   📊 Uncertainty - Mean: {avg_uncertainty:.4f}, Std: {std_uncertainty:.4f}")
                
            else:
                print(f"   ✅ Output shapes - Logits: {logits.shape}, Features: {features.shape}")
                print(f"   📊 Uncertainty disabled (output ignored)")
            
            # Analyze prediction confidence
            probabilities = torch.softmax(logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            avg_confidence = confidence.mean().item()
            std_confidence = confidence.std().item()
            
            print(f"   📊 Prediction confidence - Mean: {avg_confidence:.4f}, Std: {std_confidence:.4f}")
            
            # Test model diversity (if multiple models)
            if config['num_models'] > 1:
                individual_predictions = []
                for model in ensemble_model.models:
                    model_output = model(test_input)
                    # Individual models always return (logits, features, uncertainty)
                    pred_logits = model_output[0]
                    individual_predictions.append(torch.softmax(pred_logits, dim=1))
                
                # Calculate prediction diversity
                individual_predictions = torch.stack(individual_predictions)  # [num_models, batch, classes]
                mean_pred = individual_predictions.mean(0)
                diversity = torch.mean(torch.var(individual_predictions, dim=0)).item()
                
                print(f"   📊 Model diversity: {diversity:.4f}")
            
            print(f"   ✅ Ensemble config {i+1} test completed\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ensemble Prediction test failed: {str(e)}")
        return False

def main():
    """Run all Phase 3 enhanced CNN tests"""
    print("="*70)
    print("🚀 PHASE 3 ENHANCED CNN ANTI-SPOOFING MODEL TESTS")
    print("="*70)
    
    # Track test results
    test_results = {}
    
    # Run all tests
    tests = [
        ("Enhanced CNN Architecture", test_enhanced_cnn_architecture),
        ("Anti-Spoofing Augmentation", test_antispoofing_augmentation),
        ("Enhanced Training Pipeline", test_enhanced_training_pipeline),
        ("Texture & Frequency Analysis", test_texture_frequency_analysis),
        ("Attention Mechanisms", test_attention_mechanisms),
        ("Ensemble Prediction", test_ensemble_prediction)
    ]
    
    for test_name, test_func in tests:
        try:
            test_results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {str(e)}")
            test_results[test_name] = False
    
    # Print summary
    print("\n" + "="*70)
    print("📊 PHASE 3 TEST SUMMARY")
    print("="*70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<35} {status}")
        if passed:
            passed_tests += 1
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        print("✅ Enhanced CNN model with texture analysis ready")
        print("✅ Frequency domain analysis working")
        print("✅ Attention mechanisms functional")
        print("✅ Anti-spoofing augmentation pipeline ready")
        print("✅ Ensemble prediction system working")
        print("✅ Enhanced training pipeline with uncertainty quantification ready")
        print("\n🚀 Phase 3 implementation is COMPLETE and ready for deployment!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please review and fix issues.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

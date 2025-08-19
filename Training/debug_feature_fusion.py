"""
Debug script to trace feature dimensions through the Enhanced CNN
"""
import sys
import torch
sys.path.append('src')

from models.enhanced_cnn_model import EnhancedAntiSpoofingCNN

print("🔍 Debugging Enhanced CNN Feature Flow...")

# Create model with same parameters as test
model = EnhancedAntiSpoofingCNN(
    num_classes=2,
    use_uncertainty=True,
    backbone='resnet18'
)

# Create test input
x = torch.randn(1, 3, 224, 224)

print(f"Input shape: {x.shape}")

# Manually trace through the model
with torch.no_grad():
    # Input normalization
    x = model.input_norm(x)
    print(f"After input_norm: {x.shape}")
    
    # Stem processing
    x = model.stem(x)
    print(f"After stem: {x.shape}")
    
    # Layer 1
    x = model.layer1(x)
    print(f"After layer1: {x.shape}")
    
    # Texture and frequency features at layer 1
    if model.use_texture_analysis:
        texture1 = model.texture_branch[0](x)
        print(f"Texture branch 0 output: {texture1.shape}")
    
    if model.use_frequency_analysis:
        freq1 = model.frequency_branch[0](x)
        print(f"Frequency branch 0 output: {freq1.shape}")
    
    # Apply attention
    if model.use_attention:
        x = model.channel_attention1(x)
        x, att_map = model.spatial_attention1(x)
        print(f"After attention layer 1: {x.shape}")
    
    # Layer 2
    x = model.layer2(x)
    print(f"After layer2: {x.shape}")
    
    # Layer 3
    x = model.layer3(x)
    print(f"After layer3: {x.shape}")
    
    # Simulate the actual forward pass feature collection
    print(f"Main feature channels: {x.shape[1]}")
    features = []
    
    # Layer 1 processing 
    temp_x = model.layer1(model.stem(model.input_norm(torch.randn(1, 3, 224, 224))))
    if model.use_texture_analysis:
        texture_feat = model.texture_branch[0](temp_x)
        features.append(texture_feat)
        print(f"Layer 1 texture feature: {texture_feat.shape}")
    if model.use_frequency_analysis:
        freq_feat = model.frequency_branch[0](temp_x)
        features.append(freq_feat)
        print(f"Layer 1 frequency feature: {freq_feat.shape}")
    
    # Layer 2 processing
    temp_x = model.layer2(temp_x)
    if model.use_texture_analysis:
        texture_feat = model.texture_branch[1](temp_x)
        features.append(texture_feat)
        print(f"Layer 2 texture feature: {texture_feat.shape}")
    if model.use_frequency_analysis:
        freq_feat = model.frequency_branch[1](temp_x)
        features.append(freq_feat)
        print(f"Layer 2 frequency feature: {freq_feat.shape}")
    
    # Layer 3 processing
    temp_x = model.layer3(temp_x)
    if model.use_texture_analysis:
        texture_feat = model.texture_branch[2](temp_x)
        features.append(texture_feat)
        print(f"Layer 3 texture feature: {texture_feat.shape}")
    if model.use_frequency_analysis:
        freq_feat = model.frequency_branch[2](temp_x)
        features.append(freq_feat)
        print(f"Layer 3 frequency feature: {freq_feat.shape}")
    
    print(f"Total features collected: {len(features)}")
    
    # Apply the same logic as forward pass
    if features:
        if model.use_texture_analysis and model.use_frequency_analysis:
            # Both enabled: features has 6 elements, take last 2
            last_features = features[-2:]
            print(f"Using last 2 features (both texture and frequency enabled)")
        elif model.use_texture_analysis or model.use_frequency_analysis:
            # Only one enabled: features has 3 elements, take last 1
            last_features = features[-1:]
            print(f"Using last 1 feature (only one branch enabled)")
        else:
            last_features = []
        
        print(f"Last features to be used: {len(last_features)}")
        fusion_channels = x.shape[1]  # Main features: 512
        for i, feat in enumerate(last_features):
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (x.size(2), x.size(3)))
            print(f"Pooled last feature {i}: {pooled.shape}")
            fusion_channels += pooled.shape[1]
    
    print(f"Total expected fusion channels: {fusion_channels}")
    print(f"Model fusion_channels setting: {model.feature_fusion[2].in_features}")
    
    if fusion_channels != model.feature_fusion[2].in_features:
        print("\n🎯 Issue found! The fusion channels calculation is incorrect.")
    else:
        print("\n✅ Fusion channels calculation is correct!")

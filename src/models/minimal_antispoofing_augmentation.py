"""
Minimal Anti-Spoofing Augmentation (without albumentations dependency)
Simplified version for testing
"""

import numpy as np
import cv2
import torch
from torchvision import transforms
import random
from typing import List, Tuple, Optional

class MinimalPrintAttackAugmentation:
    """Minimal print attack augmentation without external dependencies"""
    
    def __init__(self):
        self.noise_levels = [0.01, 0.02, 0.03, 0.05]
        
    def apply_paper_texture_effect(self, image):
        """Simple paper texture simulation"""
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
            c = 1
            
        # Add subtle noise to simulate paper texture
        noise = np.random.normal(0, random.choice(self.noise_levels), image.shape)
        noisy_image = np.clip(image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def apply_dot_matrix_effect(self, image):
        """Simple dot matrix pattern simulation"""
        result = image.copy()
        
        # Add dot pattern every few pixels
        step = random.randint(3, 6)
        for i in range(0, image.shape[0], step):
            for j in range(0, image.shape[1], step):
                if (i + j) % 2 == 0:
                    result[i:i+1, j:j+1] = np.clip(result[i:i+1, j:j+1] * 0.9, 0, 255)
        
        return result
    
    def apply_color_shift(self, image):
        """Simple color shift simulation"""
        if len(image.shape) == 3:
            # Random color shift
            shift = np.random.randint(-10, 10, size=image.shape[2])
            shifted = image.astype(np.float32) + shift.reshape(1, 1, -1)
            return np.clip(shifted, 0, 255).astype(np.uint8)
        return image

class MinimalScreenAttackAugmentation:
    """Minimal screen attack augmentation"""
    
    def apply_moire_pattern(self, image):
        """Simple moire pattern simulation"""
        h, w = image.shape[:2]
        
        # Create moire pattern
        y, x = np.ogrid[:h, :w]
        pattern = (np.sin(x * 0.1) * np.sin(y * 0.1) * 20).astype(np.uint8)
        
        if len(image.shape) == 3:
            pattern = np.expand_dims(pattern, axis=2)
            
        result = np.clip(image.astype(np.float32) + pattern, 0, 255).astype(np.uint8)
        return result
    
    def apply_pixel_grid_effect(self, image):
        """Simple pixel grid simulation"""
        result = image.copy()
        
        # Add grid lines
        grid_size = random.randint(4, 8)
        for i in range(0, image.shape[0], grid_size):
            result[i:i+1, :] = np.clip(result[i:i+1, :] * 0.95, 0, 255)
        
        for j in range(0, image.shape[1], grid_size):
            result[:, j:j+1] = np.clip(result[:, j:j+1] * 0.95, 0, 255)
            
        return result
    
    def apply_screen_glare(self, image):
        """Simple screen glare simulation"""
        h, w = image.shape[:2]
        
        # Create glare effect
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < (min(h, w) // 4) ** 2
        
        glare = np.zeros_like(image)
        if len(image.shape) == 3:
            glare[mask] = [20, 20, 20]
        else:
            glare[mask] = 20
            
        result = np.clip(image.astype(np.float32) + glare, 0, 255).astype(np.uint8)
        return result

class MinimalEnvironmentalAugmentation:
    """Minimal environmental augmentation"""
    
    def apply_lighting_variation(self, image):
        """Simple lighting variation"""
        brightness_factor = random.uniform(0.7, 1.3)
        result = np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        return result
    
    def apply_shadow_effect(self, image):
        """Simple shadow effect"""
        h, w = image.shape[:2]
        
        # Create shadow mask
        shadow_mask = np.ones((h, w), dtype=np.float32)
        shadow_start = random.randint(0, w // 2)
        shadow_end = random.randint(w // 2, w)
        
        shadow_mask[:, shadow_start:shadow_end] *= 0.7
        
        if len(image.shape) == 3:
            shadow_mask = np.expand_dims(shadow_mask, axis=2)
            
        result = np.clip(image.astype(np.float32) * shadow_mask, 0, 255).astype(np.uint8)
        return result

class MinimalTemporalConsistencyAugmentation:
    """Minimal temporal consistency augmentation"""
    
    def apply_temporal_consistency(self, sequence):
        """Apply consistent augmentation across sequence"""
        if not sequence:
            return sequence
            
        # Apply same random augmentation to all frames
        aug_type = random.choice(['brightness', 'noise', 'none'])
        
        if aug_type == 'brightness':
            factor = random.uniform(0.8, 1.2)
            return [np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8) for frame in sequence]
        elif aug_type == 'noise':
            noise_level = random.choice([0.01, 0.02])
            noise = np.random.normal(0, noise_level, sequence[0].shape)
            return [np.clip(frame.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8) for frame in sequence]
        
        return sequence

def create_antispoofing_transforms(image_size=224, is_training=True, attack_simulation=True):
    """Create anti-spoofing transforms without albumentations"""
    
    transform_list = []
    
    if is_training:
        # Basic augmentations
        transform_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Common transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

# Compatibility exports
PrintAttackAugmentation = MinimalPrintAttackAugmentation
ScreenAttackAugmentation = MinimalScreenAttackAugmentation
EnvironmentalAugmentation = MinimalEnvironmentalAugmentation
TemporalConsistencyAugmentation = MinimalTemporalConsistencyAugmentation

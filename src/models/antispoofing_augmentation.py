"""
Advanced Data Augmentation for Anti-Spoofing Training
Simulates various spoofing attacks and environmental conditions
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import random
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional, List, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PrintAttackAugmentation:
    """
    Simulate print attack artifacts
    """
    
    def __init__(self, print_quality_range=(0.3, 0.8), 
                 paper_texture_strength=0.1,
                 jpeg_quality_range=(30, 80)):
        self.print_quality_range = print_quality_range
        self.paper_texture_strength = paper_texture_strength
        self.jpeg_quality_range = jpeg_quality_range
    
    def add_print_artifacts(self, image):
        """Add print-specific artifacts"""
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        # Reduce print quality
        quality_factor = random.uniform(*self.print_quality_range)
        
        # Convert to lower quality JPEG simulation
        jpeg_quality = random.randint(*self.jpeg_quality_range)
        
        # Add paper texture
        if random.random() < 0.7:  # 70% chance
            image = self.add_paper_texture(image)
        
        # Add printing artifacts
        if random.random() < 0.5:  # 50% chance
            image = self.add_dot_matrix_pattern(image)
        
        # Color degradation
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(quality_factor)
        
        # Slight blur from printing process
        if random.random() < 0.4:  # 40% chance
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        return image
    
    def add_paper_texture(self, image):
        """Add paper texture noise"""
        # Create paper texture pattern
        width, height = image.size
        texture = np.random.normal(0, self.paper_texture_strength * 255, (height, width, 3))
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Apply texture
        textured = img_array.astype(np.float32) + texture
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        
        return Image.fromarray(textured)
    
    def add_dot_matrix_pattern(self, image):
        """Add dot matrix printing pattern"""
        width, height = image.size
        
        # Create dot pattern
        dot_size = random.randint(2, 4)
        pattern = np.ones((height, width, 3))
        
        for y in range(0, height, dot_size * 2):
            for x in range(0, width, dot_size * 2):
                if random.random() < 0.3:  # Random dots
                    pattern[y:y+dot_size, x:x+dot_size] *= 0.9
        
        # Apply pattern
        img_array = np.array(image).astype(np.float32)
        patterned = img_array * pattern
        
        return Image.fromarray(np.clip(patterned, 0, 255).astype(np.uint8))
    
    def __call__(self, image):
        return self.add_print_artifacts(image)

class ScreenAttackAugmentation:
    """
    Simulate screen display attacks
    """
    
    def __init__(self, moire_strength_range=(0.1, 0.3),
                 screen_reflection_prob=0.5,
                 pixel_grid_prob=0.4):
        self.moire_strength_range = moire_strength_range
        self.screen_reflection_prob = screen_reflection_prob
        self.pixel_grid_prob = pixel_grid_prob
    
    def add_screen_artifacts(self, image):
        """Add screen-specific artifacts"""
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        # Add moire patterns
        if random.random() < 0.6:  # 60% chance
            image = self.add_moire_pattern(image)
        
        # Add screen reflection
        if random.random() < self.screen_reflection_prob:
            image = self.add_screen_reflection(image)
        
        # Add pixel grid pattern
        if random.random() < self.pixel_grid_prob:
            image = self.add_pixel_grid(image)
        
        # Screen brightness variation
        if random.random() < 0.3:  # 30% chance
            brightness_enhancer = ImageEnhance.Brightness(image)
            brightness_factor = random.uniform(0.8, 1.2)
            image = brightness_enhancer.enhance(brightness_factor)
        
        return image
    
    def add_moire_pattern(self, image):
        """Add moire interference pattern"""
        width, height = image.size
        
        # Create moire pattern
        strength = random.uniform(*self.moire_strength_range)
        frequency = random.uniform(10, 30)
        
        x = np.linspace(0, 2 * np.pi * frequency, width)
        y = np.linspace(0, 2 * np.pi * frequency, height)
        X, Y = np.meshgrid(x, y)
        
        # Interference pattern
        pattern1 = np.sin(X) * np.sin(Y)
        pattern2 = np.sin(X * 1.1) * np.sin(Y * 1.1)
        moire = (pattern1 + pattern2) * strength
        
        # Apply to image
        img_array = np.array(image).astype(np.float32)
        for c in range(3):  # RGB channels
            img_array[:, :, c] += moire * 255
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def add_screen_reflection(self, image):
        """Add screen reflection artifacts"""
        width, height = image.size
        
        # Create reflection pattern
        reflection_strength = random.uniform(0.1, 0.4)
        
        # Gradient reflection
        gradient = np.linspace(0, reflection_strength, width)
        reflection = np.tile(gradient, (height, 1))
        
        # Apply reflection
        img_array = np.array(image).astype(np.float32)
        for c in range(3):
            img_array[:, :, c] += reflection * 255
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def add_pixel_grid(self, image):
        """Add screen pixel grid pattern"""
        width, height = image.size
        
        # Create pixel grid
        grid_size = random.randint(2, 4)
        grid_strength = random.uniform(0.05, 0.15)
        
        grid = np.ones((height, width))
        
        # Vertical lines
        for x in range(0, width, grid_size):
            grid[:, x] *= (1 - grid_strength)
        
        # Horizontal lines
        for y in range(0, height, grid_size):
            grid[y, :] *= (1 - grid_strength)
        
        # Apply grid
        img_array = np.array(image).astype(np.float32)
        for c in range(3):
            img_array[:, :, c] *= grid
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def __call__(self, image):
        return self.add_screen_artifacts(image)

class EnvironmentalAugmentation:
    """
    Simulate various environmental conditions
    """
    
    def __init__(self, lighting_variation_range=(0.5, 1.5),
                 shadow_prob=0.3,
                 glare_prob=0.2):
        self.lighting_variation_range = lighting_variation_range
        self.shadow_prob = shadow_prob
        self.glare_prob = glare_prob
    
    def apply_lighting_variation(self, image):
        """Apply lighting variations"""
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        # Overall brightness variation
        brightness_factor = random.uniform(*self.lighting_variation_range)
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(brightness_factor)
        
        # Add shadows
        if random.random() < self.shadow_prob:
            image = self.add_shadow(image)
        
        # Add glare/reflection
        if random.random() < self.glare_prob:
            image = self.add_glare(image)
        
        return image
    
    def add_shadow(self, image):
        """Add shadow effects"""
        width, height = image.size
        
        # Create shadow pattern
        shadow_strength = random.uniform(0.2, 0.6)
        shadow_size = random.uniform(0.2, 0.8)
        
        # Random shadow position
        center_x = random.uniform(0.2, 0.8) * width
        center_y = random.uniform(0.2, 0.8) * height
        
        # Create circular shadow
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = min(width, height) * shadow_size
        
        shadow_mask = 1 - np.clip(distance / max_distance, 0, 1) * shadow_strength
        
        # Apply shadow
        img_array = np.array(image).astype(np.float32)
        for c in range(3):
            img_array[:, :, c] *= shadow_mask
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def add_glare(self, image):
        """Add glare effects"""
        width, height = image.size
        
        # Create glare pattern
        glare_strength = random.uniform(0.3, 0.7)
        glare_size = random.uniform(0.1, 0.4)
        
        # Random glare position
        center_x = random.uniform(0.3, 0.7) * width
        center_y = random.uniform(0.3, 0.7) * height
        
        # Create circular glare
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = min(width, height) * glare_size
        
        glare_mask = np.exp(-distance / max_distance) * glare_strength
        
        # Apply glare
        img_array = np.array(image).astype(np.float32)
        for c in range(3):
            img_array[:, :, c] += glare_mask * 255
        
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    def __call__(self, image):
        return self.apply_lighting_variation(image)

class AntiSpoofingAugmentation:
    """
    Comprehensive augmentation pipeline for anti-spoofing
    """
    
    def __init__(self, image_size=224, is_training=True, attack_simulation_prob=0.5):
        self.image_size = image_size
        self.is_training = is_training
        self.attack_simulation_prob = attack_simulation_prob
        
        # Initialize attack simulations
        self.print_attack = PrintAttackAugmentation()
        self.screen_attack = ScreenAttackAugmentation()
        self.environmental = EnvironmentalAugmentation()
        
        # Standard augmentations
        if is_training:
            self.geometric_transforms = A.Compose([
                A.RandomRotate90(p=0.1),
                A.Rotate(limit=15, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.3
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.2),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
            ])
            
            self.color_transforms = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.4
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            ])
            
            self.noise_transforms = A.Compose([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ])
        else:
            # No augmentation for validation/test
            self.geometric_transforms = A.Compose([])
            self.color_transforms = A.Compose([])
            self.noise_transforms = A.Compose([])
        
        # Final transforms
        self.final_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def apply_attack_simulation(self, image, attack_type='random'):
        """Apply spoofing attack simulation"""
        if not self.is_training:
            return image
        
        if attack_type == 'random':
            attack_type = random.choice(['print', 'screen', 'environmental', 'none'])
        
        if attack_type == 'print':
            image = self.print_attack(image)
        elif attack_type == 'screen':
            image = self.screen_attack(image)
        elif attack_type == 'environmental':
            image = self.environmental(image)
        # 'none' means no attack simulation
        
        return image
    
    def __call__(self, image, label=None):
        """Apply complete augmentation pipeline"""
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        # Apply attack simulation with probability
        if self.is_training and random.random() < self.attack_simulation_prob:
            image = self.apply_attack_simulation(image)
        
        # Convert to numpy for albumentations
        image = np.array(image)
        
        # Apply geometric transforms
        if self.geometric_transforms:
            transformed = self.geometric_transforms(image=image)
            image = transformed['image']
        
        # Apply color transforms
        if self.color_transforms:
            transformed = self.color_transforms(image=image)
            image = transformed['image']
        
        # Apply noise transforms
        if self.noise_transforms:
            transformed = self.noise_transforms(image=image)
            image = transformed['image']
        
        # Final transforms (resize, normalize, to tensor)
        transformed = self.final_transform(image=image)
        image = transformed['image']
        
        if label is not None:
            return image, label
        else:
            return image

class TemporalConsistencyAugmentation:
    """
    Augmentation for temporal consistency in video sequences
    """
    
    def __init__(self, sequence_length=5, consistency_prob=0.8):
        self.sequence_length = sequence_length
        self.consistency_prob = consistency_prob
        
    def generate_consistent_sequence(self, base_image, sequence_length=None):
        """Generate temporally consistent sequence"""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Base transformation parameters
        base_brightness = random.uniform(0.8, 1.2)
        base_contrast = random.uniform(0.8, 1.2)
        base_rotation = random.uniform(-5, 5)
        
        sequence = []
        
        for i in range(sequence_length):
            # Small variations around base parameters
            if random.random() < self.consistency_prob:
                # Consistent transformation
                brightness = base_brightness + random.uniform(-0.1, 0.1)
                contrast = base_contrast + random.uniform(-0.1, 0.1)
                rotation = base_rotation + random.uniform(-2, 2)
            else:
                # Inconsistent transformation (simulate spoofing)
                brightness = random.uniform(0.5, 1.5)
                contrast = random.uniform(0.5, 1.5)
                rotation = random.uniform(-15, 15)
            
            # Apply transformations
            transformed = TF.adjust_brightness(base_image, brightness)
            transformed = TF.adjust_contrast(transformed, contrast)
            transformed = TF.rotate(transformed, rotation)
            
            sequence.append(transformed)
        
        return sequence

def create_antispoofing_transforms(image_size=224, is_training=True, 
                                 attack_simulation_prob=0.5):
    """
    Factory function to create anti-spoofing transforms
    """
    return AntiSpoofingAugmentation(
        image_size=image_size,
        is_training=is_training,
        attack_simulation_prob=attack_simulation_prob
    )

if __name__ == "__main__":
    # Test augmentations
    print("🎨 Testing Anti-Spoofing Augmentations")
    print("=" * 50)
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Test print attack simulation
    print("\\n📄 Testing Print Attack Simulation:")
    print_aug = PrintAttackAugmentation()
    print_result = print_aug(dummy_image)
    print(f"   Input size: {dummy_image.size}")
    print(f"   Output size: {print_result.size}")
    
    # Test screen attack simulation
    print("\\n💻 Testing Screen Attack Simulation:")
    screen_aug = ScreenAttackAugmentation()
    screen_result = screen_aug(dummy_image)
    print(f"   Input size: {dummy_image.size}")
    print(f"   Output size: {screen_result.size}")
    
    # Test environmental augmentation
    print("\\n🌤️  Testing Environmental Augmentation:")
    env_aug = EnvironmentalAugmentation()
    env_result = env_aug(dummy_image)
    print(f"   Input size: {dummy_image.size}")
    print(f"   Output size: {env_result.size}")
    
    # Test complete pipeline
    print("\\n🔧 Testing Complete Augmentation Pipeline:")
    
    # Training transforms
    train_transforms = create_antispoofing_transforms(
        image_size=224,
        is_training=True,
        attack_simulation_prob=0.5
    )
    
    train_result = train_transforms(dummy_image)
    print(f"   Training output shape: {train_result.shape}")
    print(f"   Training output type: {type(train_result)}")
    
    # Validation transforms
    val_transforms = create_antispoofing_transforms(
        image_size=224,
        is_training=False
    )
    
    val_result = val_transforms(dummy_image)
    print(f"   Validation output shape: {val_result.shape}")
    print(f"   Validation output type: {type(val_result)}")
    
    # Test temporal consistency
    print("\\n⏱️  Testing Temporal Consistency:")
    temporal_aug = TemporalConsistencyAugmentation(sequence_length=3)
    sequence = temporal_aug.generate_consistent_sequence(dummy_image)
    print(f"   Sequence length: {len(sequence)}")
    print(f"   Frame size: {sequence[0].size}")
    
    print("\\n✅ All Augmentation Tests Completed Successfully!")

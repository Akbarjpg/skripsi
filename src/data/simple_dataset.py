"""
Fallback dataset analysis tanpa albumentations
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

def analyze_dataset_simple(data_dir):
    """
    Analyze dataset tanpa dependency external
    """
    if not os.path.exists(data_dir):
        return {"error": "Dataset directory not found"}
    
    real_count = 0
    fake_count = 0
    total_files = 0
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_files += 1
            if 'real' in filename.lower():
                real_count += 1
            elif 'fake' in filename.lower():
                fake_count += 1
    
    return {
        "total_files": total_files,
        "real_images": real_count,
        "fake_images": fake_count,
        "balance_ratio": real_count / max(fake_count, 1)
    }

def create_simple_dataset(data_dir, image_size=(224, 224)):
    """
    Simple dataset creation tanpa augmentation complex
    """
    
    class SimpleDataset(Dataset):
        def __init__(self, data_dir, image_size):
            self.data_dir = data_dir
            self.image_size = image_size
            self.samples = []
            
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filepath = os.path.join(data_dir, filename)
                        label = 1 if 'real' in filename.lower() else 0
                        self.samples.append((filepath, label))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            filepath, label = self.samples[idx]
            
            # Load image
            image = cv2.imread(filepath)
            if image is None:
                # Return dummy data if image cannot be loaded
                image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            else:
                image = cv2.resize(image, self.image_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            
            return image, label
    
    return SimpleDataset(data_dir, image_size)

if __name__ == "__main__":
    # Test simple dataset
    if os.path.exists("test_img/color"):
        analysis = analyze_dataset_simple("test_img/color")
        print(f"Dataset analysis: {analysis}")
        
        dataset = create_simple_dataset("test_img/color")
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample_image, sample_label = dataset[0]
            print(f"Sample shape: {sample_image.shape}, label: {sample_label}")
    else:
        print("Dataset directory not found")

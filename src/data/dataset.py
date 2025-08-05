"""
Dataset loader dan preprocessing untuk Face Anti-Spoofing
Mendukung format gambar dengan label real/fake dari nama file
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

class FaceAntiSpoofingDataset(Dataset):
    """
    Dataset untuk Face Anti-Spoofing dengan preprocessing yang robust
    """
    
    def __init__(self, image_paths, labels, transform=None, target_size=(224, 224)):
        """
        Args:
            image_paths (list): List path ke gambar
            labels (list): List label (0=fake, 1=real)
            transform: Augmentasi data
            target_size: Ukuran target gambar
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load dan preprocess gambar
        """
        try:
            # Load gambar
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize ke target size
            image = cv2.resize(image, self.target_size)
            
            # Apply transforms jika ada
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Default normalization
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            logging.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return dummy data jika error
            dummy_image = torch.zeros((3, self.target_size[0], self.target_size[1]))
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_image, dummy_label

def parse_dataset(data_dir):
    """
    Parse dataset dari direktori dengan format: video_id_frame_number_label.jpg
    
    Args:
        data_dir: Path ke direktori color images
        
    Returns:
        image_paths, labels, metadata
    """
    image_paths = []
    labels = []
    metadata = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(data_dir, filename)
            
            # Parse label dari nama file
            if '_real.' in filename:
                label = 1  # Real
                label_str = 'real'
            elif '_fake.' in filename:
                label = 0  # Fake
                label_str = 'fake'
            else:
                continue  # Skip file yang tidak sesuai format
            
            # Extract metadata
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) >= 3:
                video_id = parts[0]
                frame_number = parts[1] if parts[1].isdigit() else parts[1] + '_' + parts[2]
                
                metadata.append({
                    'filename': filename,
                    'video_id': video_id,
                    'frame_number': frame_number,
                    'label_str': label_str,
                    'label': label
                })
                
                image_paths.append(filepath)
                labels.append(label)
    
    return image_paths, labels, metadata

def get_train_transforms(target_size=(224, 224)):
    """
    Augmentasi untuk training data
    """
    return A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(target_size=(224, 224)):
    """
    Preprocessing untuk validation/test data
    """
    return A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def create_dataloaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, 
                      target_size=(224, 224), num_workers=4, random_state=42):
    """
    Buat DataLoader untuk training, validation, dan testing
    
    Args:
        data_dir: Path ke direktori dataset
        batch_size: Ukuran batch
        val_split: Proporsi data untuk validation
        test_split: Proporsi data untuk testing
        target_size: Ukuran target gambar
        num_workers: Jumlah worker untuk DataLoader
        random_state: Random seed
    
    Returns:
        train_loader, val_loader, test_loader, class_counts
    """
    
    # Parse dataset
    image_paths, labels, metadata = parse_dataset(data_dir)
    
    # Konversi ke DataFrame untuk analisis
    df = pd.DataFrame(metadata)
    
    print(f"Total gambar: {len(image_paths)}")
    print(f"Distribusi label:")
    print(df['label_str'].value_counts())
    
    # Split dataset
    # Pertama split untuk train dan temp (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, 
        test_size=(val_split + test_split), 
        stratify=labels,
        random_state=random_state
    )
    
    # Split temp menjadi val dan test
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=random_state
    )
    
    print(f"\\nSplit dataset:")
    print(f"Train: {len(train_paths)} gambar")
    print(f"Validation: {len(val_paths)} gambar")
    print(f"Test: {len(test_paths)} gambar")
    
    # Hitung distribusi kelas untuk weights
    unique, counts = np.unique(train_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Buat dataset
    train_dataset = FaceAntiSpoofingDataset(
        train_paths, train_labels, 
        transform=get_train_transforms(target_size),
        target_size=target_size
    )
    
    val_dataset = FaceAntiSpoofingDataset(
        val_paths, val_labels,
        transform=get_val_transforms(target_size),
        target_size=target_size
    )
    
    test_dataset = FaceAntiSpoofingDataset(
        test_paths, test_labels,
        transform=get_val_transforms(target_size),
        target_size=target_size
    )
    
    # Buat DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_counts

def analyze_dataset(data_dir):
    """
    Analisis mendalam dataset
    """
    image_paths, labels, metadata = parse_dataset(data_dir)
    df = pd.DataFrame(metadata)
    
    print("=== ANALISIS DATASET ===")
    print(f"Total gambar: {len(image_paths)}")
    print(f"\\nDistribusi label:")
    print(df['label_str'].value_counts())
    print(f"\\nPersentase:")
    print(df['label_str'].value_counts(normalize=True) * 100)
    
    print(f"\\nJumlah video ID unik: {df['video_id'].nunique()}")
    print(f"Video ID: {sorted(df['video_id'].unique())}")
    
    # Analisis per video
    video_stats = df.groupby(['video_id', 'label_str']).size().unstack(fill_value=0)
    print(f"\\nDistribusi per video:")
    print(video_stats)
    
    return df

if __name__ == "__main__":
    # Test dataset loader
    data_dir = "test_img/color"
    
    # Analisis dataset
    df = analyze_dataset(data_dir)
    
    # Buat dataloaders
    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        data_dir, batch_size=16, target_size=(224, 224)
    )
    
    print(f"\\nClass counts: {class_counts}")
    
    # Test loading batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break

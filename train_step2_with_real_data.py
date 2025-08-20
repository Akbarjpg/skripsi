#!/usr/bin/env python3
"""
Step 2 Training Script with Real Dataset
=======================================

This script trains the Enhanced CNN model using the actual test_img dataset:
- Color images: test_img/color/ with *_real.jpg and *_fake.jpg
- Depth images: test_img/depth/ with *_real.jpg and *_fake.jpg
- Automatic train/val/test splitting
- Complete training loop with validation
"""

import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class TestImgDataset(Dataset):
    """
    Dataset class specifically for your test_img folder structure
    """
    
    def __init__(self, data_dir="test_img", split="train", use_depth=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_depth = use_depth
        
        # Choose between color and depth images
        image_dir = self.data_dir / ('depth' if use_depth else 'color')
        
        if not image_dir.exists():
            print(f"Warning: {image_dir} does not exist")
            self.samples = []
            return
        
        # Load all images and their labels
        all_samples = []
        
        # Load real images (label = 1)
        real_files = list(image_dir.glob('*_real.jpg'))
        for img_path in real_files:
            all_samples.append((str(img_path), 1))
        
        # Load fake images (label = 0)
        fake_files = list(image_dir.glob('*_fake.jpg'))
        for img_path in fake_files:
            all_samples.append((str(img_path), 0))
        
        # Shuffle for random splitting
        random.seed(42)
        random.shuffle(all_samples)
        
        # Split ratios: 70% train, 20% val, 10% test
        total_samples = len(all_samples)
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.9)
        
        # Select samples based on split
        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'val':
            self.samples = all_samples[train_end:val_end]
        elif split == 'test':
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Count real and fake samples
        real_count = sum(1 for _, label in self.samples if label == 1)
        fake_count = len(self.samples) - real_count
        
        print(f"✓ {split.upper()} SET: {len(self.samples)} samples")
        print(f"  Real: {real_count}, Fake: {fake_count}")
        print(f"  Using {'depth' if use_depth else 'color'} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, torch.tensor(label, dtype=torch.long)


def train_step2_model():
    """Train the Step 2 Enhanced CNN Model with real data"""
    
    print("=== Step 2 CNN Training with Real Dataset ===")
    set_random_seeds(42)
    
    # Check dataset availability
    if not Path("test_img").exists():
        print("❌ test_img folder not found!")
        return False
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Import the enhanced model
        from models.antispoofing_cnn_model import EnhancedAntiSpoofingCNN
        print("✓ Imported EnhancedAntiSpoofingCNN")
    except ImportError as e:
        print(f"❌ Failed to import model: {e}")
        return False
    
    # Create datasets
    print("\n--- Loading Datasets ---")
    train_dataset = TestImgDataset("test_img", "train", use_depth=False)
    val_dataset = TestImgDataset("test_img", "val", use_depth=False)
    test_dataset = TestImgDataset("test_img", "test", use_depth=False)
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return False
    
    # Create data loaders
    batch_size = 16  # Smaller batch size for memory efficiency
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n--- Data Loaders Created ---")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n--- Creating Model ---")
    model = EnhancedAntiSpoofingCNN()
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"✓ Model and training setup complete")
    
    # Training loop
    num_epochs = 20  # Reduced for quick testing
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"\n--- Starting Training ({num_epochs} epochs) ---")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs['logits'], target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs['logits'].data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_acc = 100. * correct_train / total_train
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs['logits'], target)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs['logits'].data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_acc = 100. * correct_val / total_val
        val_accuracies.append(val_acc)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }, 'best_antispoofing_model.pth')
            print(f"  ★ New best model saved! Val Acc: {val_acc:.2f}%")
        
        print()
    
    # Final evaluation on test set
    print("--- Final Test Evaluation ---")
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs['logits'].data, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_predictions) * 100
    test_precision = precision_score(test_labels, test_predictions, average='weighted') * 100
    test_recall = recall_score(test_labels, test_predictions, average='weighted') * 100
    test_f1 = f1_score(test_labels, test_predictions, average='weighted') * 100
    
    print(f"✓ TEST RESULTS:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Precision: {test_precision:.2f}%")
    print(f"  Recall: {test_recall:.2f}%")
    print(f"  F1-Score: {test_f1:.2f}%")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save training results
    results = {
        'final_test_accuracy': test_acc,
        'best_validation_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'total_epochs': num_epochs,
        'dataset_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset)
        }
    }
    
    import json
    with open('step2_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Results saved to: step2_training_results.json")
    print(f"💾 Best model saved to: best_antispoofing_model.pth")
    
    return True


if __name__ == "__main__":
    print("Starting Step 2 Enhanced CNN Training...")
    
    success = train_step2_model()
    
    if success:
        print("\n✅ Step 2 training completed successfully!")
        print("🚀 Your enhanced CNN model is now trained and ready!")
    else:
        print("\n❌ Step 2 training failed")
        print("🔧 Please check the error messages above")

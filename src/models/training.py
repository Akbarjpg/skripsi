"""
Training pipeline untuk Face Anti-Spoofing CNN Model
Dengan monitoring, checkpointing, dan evaluation yang komprehensif
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm import tqdm
import logging

# Optional imports for visualization and tensorboard
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

class FocalLoss(nn.Module):
    """
    Focal Loss untuk mengatasi class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """
    Early stopping untuk mencegah overfitting
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ModelTrainer:
    """
    Comprehensive trainer untuk Face Anti-Spoofing model
    """
    
    def __init__(self, model, train_loader, val_loader, device, 
                 model_save_path='models', log_dir='logs'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_save_path = model_save_path
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup tensorboard (optional)
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            print("📊 TensorBoard logging disabled: tensorboard not available")
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def calculate_class_weights(self, class_counts):
        """
        Calculate class weights untuk mengatasi imbalanced dataset
        """
        total_samples = sum(class_counts.values())
        weights = {}
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (len(class_counts) * count)
        
        weight_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())], 
                                   dtype=torch.float32).to(self.device)
        return weight_tensor
    
    def train_epoch(self, optimizer, criterion, epoch):
        """
        Train model untuk satu epoch
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
            
            # Log to tensorboard (every 100 batches)
            if batch_idx % 100 == 0 and self.writer:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/BatchAcc', current_acc, step)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def validate_epoch(self, criterion, epoch):
        """
        Validate model
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                # Store predictions for detailed metrics
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                current_loss = running_loss / total_samples
                current_acc = running_corrects.double() / total_samples
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # AUC Score
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        except:
            auc = 0.5
        
        return epoch_loss, epoch_acc.item(), precision, recall, f1, auc
    
    def train(self, num_epochs=50, learning_rate=0.001, weight_decay=1e-4,
              class_counts=None, use_focal_loss=False, patience=10):
        """
        Main training loop
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=1, gamma=2)
        else:
            if class_counts is not None:
                class_weights = self.calculate_class_weights(class_counts)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate_epoch(criterion, epoch)
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to tensorboard (optional)
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                self.writer.add_scalar('Val/Precision', val_precision, epoch)
                self.writer.add_scalar('Val/Recall', val_recall, epoch)
                self.writer.add_scalar('Val/F1', val_f1, epoch)
                self.writer.add_scalar('Val/AUC', val_auc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - '
                f'Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f} - LR: {current_lr:.6f}'
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(epoch, val_acc, 'best_model.pth')
                self.logger.info(f'New best model saved with Val Acc: {val_acc:.4f}')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch, val_acc, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                self.logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time/3600:.2f} hours')
        self.logger.info(f'Best validation accuracy: {best_val_acc:.4f}')
        
        # Save final model
        self.save_model(epoch, val_acc, 'final_model.pth')
        
        # Plot training history
        self.plot_training_history()
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        return self.history
    
    def save_model(self, epoch, val_acc, filename):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        save_path = os.path.join(self.model_save_path, filename)
        torch.save(checkpoint, save_path)
        
    def load_model(self, filename):
        """
        Load model checkpoint
        """
        load_path = os.path.join(self.model_save_path, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def plot_training_history(self):
        """
        Plot training history (requires matplotlib)
        """
        if not self.history['train_loss']:
            return
            
        if not MATPLOTLIB_AVAILABLE:
            print("📊 Plotting skipped: matplotlib not available")
            print(f"Training completed - Final Val Acc: {self.history['val_acc'][-1]:.4f}")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['lr'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Validation metrics comparison
        if len(self.history['val_acc']) > 0:
            best_epoch = np.argmax(self.history['val_acc'])
            axes[1, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
            axes[1, 1].plot(self.history['val_acc'], label='Val Accuracy', color='orange')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_model(model, test_loader, device, class_names=['Fake', 'Real']):
    """
    Comprehensive model evaluation
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # AUC
    auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    
    print(f"=== MODEL EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"\\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    # Plot confusion matrix (optional)
    if MATPLOTLIB_AVAILABLE and SEABORN_AVAILABLE:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    else:
        print("📊 Confusion matrix plotting skipped: matplotlib/seaborn not available")
        print(f"Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

if __name__ == "__main__":
    # Example usage akan dijalankan dari script terpisah
    print("Training module loaded successfully!")
    print("Use this module by importing: from src.models.training import ModelTrainer")

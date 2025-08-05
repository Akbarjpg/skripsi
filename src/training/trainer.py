"""
Training module for Face Anti-Spoofing model
Clean, organized training pipeline
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
from typing import Dict, Any, Tuple, Optional
import time

from ..utils.logger import get_training_logger
from ..utils.config import SystemConfig
from ..models.simple_model import SimpleLivenessModel


class ModelTrainer:
    """Handles model training with comprehensive logging and monitoring"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = get_training_logger()
        self.device = self._get_device()
        
        # Training metrics
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'best_val_accuracy': 0.0,
            'epochs_completed': 0
        }
        
        self.logger.info(f"ModelTrainer initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"Using specified device: {device}")
        
        return device
    
    def train(self) -> bool:
        """Main training loop"""
        self.logger.info("🎯 Starting model training...")
        
        try:
            # Initialize model
            model = self._create_model()
            
            # Setup data loaders
            train_loader, val_loader = self._setup_data_loaders()
            
            # Setup optimizer and criterion
            optimizer = self._setup_optimizer(model)
            criterion = self._setup_criterion()
            
            # Training loop
            best_model_state = None
            epochs_without_improvement = 0
            
            for epoch in range(self.config.training.num_epochs):
                self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
                
                # Train phase
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, optimizer, criterion
                )
                
                # Validation phase
                val_loss, val_acc = self._validate_epoch(
                    model, val_loader, criterion
                )
                
                # Update metrics
                self.metrics['train_losses'].append(train_loss)
                self.metrics['val_losses'].append(val_loss)
                self.metrics['train_accuracies'].append(train_acc)
                self.metrics['val_accuracies'].append(val_acc)
                self.metrics['epochs_completed'] = epoch + 1
                
                # Check for improvement
                if val_acc > self.metrics['best_val_accuracy']:
                    self.metrics['best_val_accuracy'] = val_acc
                    best_model_state = model.state_dict().copy()
                    epochs_without_improvement = 0
                    self.logger.info(f"✅ New best validation accuracy: {val_acc:.4f}")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    self.logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
                
                # Log progress
                self.logger.info(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            
            # Save best model
            if best_model_state:
                self._save_model(best_model_state)
                self._save_metrics()
                
                self.logger.info("✅ Training completed successfully")
                return True
            else:
                self.logger.error("❌ No model improvement during training")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False
    
    def _create_model(self) -> nn.Module:
        """Create and initialize model"""
        self.logger.info(f"Creating model: {self.config.model.architecture}")
        
        if self.config.model.architecture == "simple_cnn":
            model = SimpleLivenessModel(
                input_size=self.config.model.input_size,
                num_classes=self.config.model.num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        else:
            # Could add more architectures here
            raise ValueError(f"Unknown architecture: {self.config.model.architecture}")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        return model
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders"""
        self.logger.info("Setting up data loaders...")
        
        # For now, create dummy data loaders
        # In production, replace with actual dataset loading
        class DummyDataset(Dataset):
            def __init__(self, size: int = 1000):
                self.size = size
                self.input_size = self.config.model.input_size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # Random dummy data
                x = torch.randn(3, *self.input_size)
                y = torch.randint(0, 2, (1,)).long().squeeze()
                return x, y
        
        # Create datasets
        train_dataset = DummyDataset(800)
        val_dataset = DummyDataset(200)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers
        )
        
        self.logger.info(f"Data loaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config.training.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        self.logger.info(f"Optimizer: {self.config.training.optimizer}, LR: {self.config.training.learning_rate}")
        
        return optimizer
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion"""
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _save_model(self, model_state: Dict[str, Any]) -> None:
        """Save model checkpoint"""
        model_dir = Path(self.config.model.save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model_state,
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'training_completed': True
        }
        
        torch.save(checkpoint, self.config.model.save_path)
        self.logger.info(f"Model saved to: {self.config.model.save_path}")
    
    def _save_metrics(self) -> None:
        """Save training metrics"""
        metrics_path = Path(self.config.model.save_path).parent / "training_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {metrics_path}")


def load_trained_model(model_path: str, device: torch.device = None) -> Optional[nn.Module]:
    """Load a trained model from checkpoint"""
    logger = get_training_logger()
    
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Recreate model (you might need to adjust this based on saved config)
        model = SimpleLivenessModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

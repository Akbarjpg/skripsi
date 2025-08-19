"""
Enhanced Training pipeline untuk Face Anti-Spoofing CNN Model
With advanced anti-spoofing features, uncertainty quantification, and ensemble training
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    roc_auc_score, classification_report
)
from tqdm import tqdm
import logging
import json
import pickle
from collections import defaultdict

# Import enhanced models and augmentations
from .enhanced_cnn_model import EnhancedAntiSpoofingCNN, EnsembleAntiSpoofingModel
from .minimal_antispoofing_augmentation import create_antispoofing_transforms

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

class UncertaintyLoss(nn.Module):
    """
    Loss function that incorporates uncertainty quantification
    """
    
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for uncertainty regularization
        self.classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, uncertainty=None):
        # Standard classification loss
        cls_loss = self.classification_loss(logits, targets)
        
        if uncertainty is not None:
            # Uncertainty regularization: prevent overconfident predictions
            uncertainty_reg = torch.mean(uncertainty)
            
            # Weighted uncertainty loss based on prediction confidence
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidence_weight = 1.0 - max_probs  # Higher weight for low confidence predictions
            
            weighted_uncertainty = torch.mean(uncertainty.squeeze() * confidence_weight)
            
            total_loss = self.alpha * cls_loss + self.beta * (uncertainty_reg + weighted_uncertainty)
        else:
            total_loss = cls_loss
        
        return total_loss, cls_loss

class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss untuk mengatasi class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', weight=self.class_weights)(inputs, targets)
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
    Enhanced Early stopping dengan model checkpointing
    """
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, 
                 save_best_model=True, model_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.save_best_model = save_best_model
        self.model_path = model_path
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, epoch)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            print(f"Early stopping at epoch {epoch}, best epoch was {self.best_epoch}")
            return True
        return False
    
    def save_checkpoint(self, model, epoch):
        self.best_weights = model.state_dict().copy()
        self.best_epoch = epoch
        
        if self.save_best_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.best_weights,
                'best_loss': self.best_loss
            }, self.model_path)

class EnhancedEarlyStopping:
    """
    Enhanced Early stopping with additional features for uncertainty-aware training
    """
    def __init__(self, patience=15, min_delta=1e-4, restore_best_weights=True, 
                 mode='min', factor=0.1, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.factor = factor
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        self.early_stop = False
        
        # Mode-specific initialization
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf  # Changed from np.Inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf  # Changed from -np.Inf
    
    def __call__(self, val_metric, model, epoch=None):
        """
        Check if early stopping criteria is met
        
        Args:
            val_metric: Validation metric to monitor
            model: Model to save best weights
            epoch: Current epoch number
        """
        if epoch is None:
            epoch = self.counter
        
        # Check if current score is better than best
        if self.is_better(val_metric, self.best_score):
            self.best_score = val_metric
            self.counter = 0
            self.best_epoch = epoch
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            
            if self.verbose:
                print(f"Validation metric improved to {val_metric:.6f}")
        
        else:
            self.counter += 1
            
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            # Check if patience exceeded
            if self.counter >= self.patience:
                self.early_stop = True
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"Restored best weights from epoch {self.best_epoch}")
    
    def is_better(self, current, best):
        """Check if current score is better than best"""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)

class EnhancedModelTrainer:
    """
    Enhanced trainer for anti-spoofing models with advanced features
    """
    
    def __init__(self, model, train_loader, val_loader, device, 
                 model_save_path='models', log_dir='logs', use_tensorboard=True,
                 use_uncertainty=True, use_ensemble=False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_uncertainty = use_uncertainty
        self.use_ensemble = use_ensemble
        
        # Setup directories
        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_path = model_save_path
        
        # Setup logging
        self.setup_logging(log_dir)
        
        # Setup tensorboard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'uncertainty_loss': [],
            'learning_rate': []
        }
        
        # Performance metrics
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.training_start_time = None
        
        print(f"✅ Enhanced Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Use uncertainty: {use_uncertainty}")
        print(f"   Use ensemble: {use_ensemble}")
        print(f"   Tensorboard: {self.use_tensorboard}")
    
    def setup_logging(self, log_dir):
        """Setup logging configuration"""
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_optimizer_and_scheduler(self, optimizer_type='adamw', learning_rate=1e-3,
                                    weight_decay=1e-4, scheduler_type='cosine',
                                    max_epochs=100):
        """Setup optimizer and learning rate scheduler"""
        
        # Optimizer selection
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Scheduler selection
        if scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_type.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type.lower() == 'onecycle':
            steps_per_epoch = len(self.train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                epochs=max_epochs,
                steps_per_epoch=steps_per_epoch
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {optimizer_type}, LR: {learning_rate}")
        self.logger.info(f"Scheduler: {scheduler_type}")
    
    def setup_loss_function(self, loss_type='uncertainty', class_weights=None, 
                          focal_alpha=1.0, focal_gamma=2.0):
        """Setup loss function"""
        
        if loss_type == 'uncertainty' and self.use_uncertainty:
            self.criterion = UncertaintyLoss(alpha=1.0, beta=0.1)
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=class_weights
            )
        elif loss_type == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Loss function: {loss_type}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_uncertainty_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_uncertainty:
                logits, features, uncertainty = self.model(data)
                loss, cls_loss = self.criterion(logits, targets, uncertainty)
                uncertainty_loss = torch.mean(uncertainty).item()
            else:
                if self.use_ensemble:
                    logits, features, uncertainty = self.model(data)
                    loss, cls_loss = self.criterion(logits, targets, uncertainty)
                    uncertainty_loss = torch.mean(uncertainty).item()
                else:
                    logits, features = self.model(data)
                    loss = self.criterion(logits, targets)
                    cls_loss = loss
                    uncertainty_loss = 0.0
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss
            total_uncertainty_loss += uncertainty_loss
            
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Update learning rate for OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_uncertainty_loss = total_uncertainty_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_cls_loss, avg_uncertainty_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for data, targets in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.use_uncertainty or self.use_ensemble:
                    logits, features, uncertainty = self.model(data)
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                else:
                    logits, features = self.model(data)
                
                # Calculate loss
                if self.use_uncertainty:
                    loss, _ = self.criterion(logits, targets, uncertainty)
                else:
                    loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Store for metrics calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, np.array(all_probabilities)[:, 1])
        except:
            auc = 0.0
        
        # Calculate uncertainty statistics if available
        uncertainty_stats = {}
        if all_uncertainties:
            uncertainty_stats = {
                'mean_uncertainty': np.mean(all_uncertainties),
                'std_uncertainty': np.std(all_uncertainties)
            }
        
        return avg_loss, accuracy, precision, recall, f1, auc, uncertainty_stats
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_save_path, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.model_save_path, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 New best model saved: {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, epochs=100, early_stopping_patience=15, save_every=5,
              validate_every=1, resume_from=None):
        """Main training loop"""
        
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            start_epoch, last_metrics = self.load_checkpoint(resume_from)
            start_epoch += 1
            self.logger.info(f"🔄 Resuming training from epoch {start_epoch}")
        
        # Setup early stopping
        early_stopping = EnhancedEarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            restore_best_weights=True
        )
        
        self.training_start_time = time.time()
        
        self.logger.info("🚀 Starting enhanced training...")
        self.logger.info(f"   Total epochs: {epochs}")
        self.logger.info(f"   Early stopping patience: {early_stopping_patience}")
        self.logger.info(f"   Validation every: {validate_every} epochs")
        
        try:
            for epoch in range(start_epoch, epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_loss, train_cls_loss, train_uncertainty_loss, train_acc = self.train_epoch(epoch)
                
                # Validation phase
                if epoch % validate_every == 0:
                    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, uncertainty_stats = self.validate_epoch(epoch)
                    
                    # Update learning rate scheduler
                    if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                    
                    # Update history
                    self.history['train_loss'].append(train_loss)
                    self.history['train_acc'].append(train_acc)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)
                    self.history['val_auc'].append(val_auc)
                    self.history['uncertainty_loss'].append(train_uncertainty_loss)
                    self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                    
                    # Check for best model
                    is_best = val_auc > self.best_val_auc
                    if is_best:
                        self.best_val_acc = val_acc
                        self.best_val_auc = val_auc
                    
                    # Log metrics
                    epoch_time = time.time() - epoch_start_time
                    self.log_epoch_metrics(
                        epoch, train_loss, train_acc, val_loss, val_acc, 
                        val_precision, val_recall, val_f1, val_auc,
                        uncertainty_stats, epoch_time, is_best
                    )
                    
                    # Tensorboard logging
                    if self.use_tensorboard:
                        self.log_to_tensorboard(
                            epoch, train_loss, train_acc, val_loss, val_acc,
                            val_auc, train_uncertainty_loss, uncertainty_stats
                        )
                    
                    # Save checkpoint
                    if epoch % save_every == 0 or is_best:
                        metrics = {
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'val_auc': val_auc,
                            'val_f1': val_f1
                        }
                        self.save_checkpoint(epoch, metrics, is_best)
                    
                    # Early stopping check
                    early_stopping(val_loss, self.model)
                    if early_stopping.early_stop:
                        self.logger.info(f"⏹️ Early stopping triggered at epoch {epoch}")
                        break
                
                else:
                    # No validation this epoch
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {epoch+1:3d}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Time: {epoch_time:.1f}s"
                    )
        
        except KeyboardInterrupt:
            self.logger.info("⏸️ Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"❌ Training error: {str(e)}")
            raise e
        
        finally:
            # Training complete
            total_time = time.time() - self.training_start_time
            self.logger.info("🎯 Training completed!")
            self.logger.info(f"   Total training time: {total_time/3600:.2f} hours")
            self.logger.info(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
            self.logger.info(f"   Best validation AUC: {self.best_val_auc:.4f}")
            
            # Save final model
            final_metrics = {
                'best_val_acc': self.best_val_acc,
                'best_val_auc': self.best_val_auc,
                'total_epochs': len(self.history['train_loss']),
                'total_time': total_time
            }
            self.save_checkpoint(epoch, final_metrics, is_best=False)
            
            # Close tensorboard writer
            if self.use_tensorboard:
                self.writer.close()
        
        return self.history
    
    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc,
                         val_precision, val_recall, val_f1, val_auc, uncertainty_stats,
                         epoch_time, is_best):
        """Log detailed epoch metrics"""
        
        # Basic metrics
        log_msg = (
            f"Epoch {epoch+1:3d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Val AUC: {val_auc:.4f} | F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        if is_best:
            log_msg += " 🌟 NEW BEST!"
        
        self.logger.info(log_msg)
        
        # Detailed validation metrics
        self.logger.info(
            f"         Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Uncertainty statistics
        if uncertainty_stats:
            self.logger.info(
                f"         Uncertainty - Mean: {uncertainty_stats['mean_uncertainty']:.4f} | "
                f"Std: {uncertainty_stats['std_uncertainty']:.4f}"
            )
    
    def log_to_tensorboard(self, epoch, train_loss, train_acc, val_loss, val_acc,
                          val_auc, uncertainty_loss, uncertainty_stats):
        """Log metrics to tensorboard"""
        if not self.use_tensorboard:
            return
        
        # Loss metrics
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Loss/Uncertainty', uncertainty_loss, epoch)
        
        # Accuracy metrics
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.writer.add_scalar('AUC/Validation', val_auc, epoch)
        
        # Learning rate
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Uncertainty statistics
        if uncertainty_stats:
            self.writer.add_scalar('Uncertainty/Mean', uncertainty_stats['mean_uncertainty'], epoch)
            self.writer.add_scalar('Uncertainty/Std', uncertainty_stats['std_uncertainty'], epoch)
    
    def evaluate_model(self, test_loader, save_predictions=True):
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        all_features = []
        
        total_correct = 0
        total_samples = 0
        
        self.logger.info("🔍 Starting comprehensive model evaluation...")
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc='Evaluating'):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.use_uncertainty or self.use_ensemble:
                    logits, features, uncertainty = self.model(data)
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                else:
                    logits, features = self.model(data)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits.data, 1)
                
                # Statistics
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_features.extend(features.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = 100. * total_correct / total_samples
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, np.array(all_probabilities)[:, 1])
        except:
            auc = 0.0
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Calculate class-specific metrics
        class_report = classification_report(
            all_targets, all_predictions,
            target_names=['Real', 'Fake'],
            output_dict=True
        )
        
        # Uncertainty analysis
        uncertainty_analysis = {}
        if all_uncertainties:
            uncertainty_array = np.array(all_uncertainties)
            uncertainty_analysis = {
                'mean_uncertainty': np.mean(uncertainty_array),
                'std_uncertainty': np.std(uncertainty_array),
                'uncertainty_by_class': {
                    'real': np.mean(uncertainty_array[np.array(all_targets) == 0]),
                    'fake': np.mean(uncertainty_array[np.array(all_targets) == 1])
                }
            }
        
        # Compile results
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'class_report': class_report,
            'uncertainty_analysis': uncertainty_analysis,
            'total_samples': total_samples
        }
        
        # Save predictions if requested
        if save_predictions:
            predictions_data = {
                'predictions': all_predictions,
                'targets': all_targets,
                'probabilities': all_probabilities,
                'uncertainties': all_uncertainties,
                'features': all_features
            }
            
            predictions_path = os.path.join(self.model_save_path, 'test_predictions.pkl')
            with open(predictions_path, 'wb') as f:
                pickle.dump(predictions_data, f)
            
            self.logger.info(f"💾 Predictions saved: {predictions_path}")
        
        # Log results
        self.logger.info("📊 Evaluation Results:")
        self.logger.info(f"   Accuracy: {accuracy:.2f}%")
        self.logger.info(f"   Precision: {precision:.4f}")
        self.logger.info(f"   Recall: {recall:.4f}")
        self.logger.info(f"   F1-Score: {f1:.4f}")
        self.logger.info(f"   AUC: {auc:.4f}")
        
        if uncertainty_analysis:
            self.logger.info(f"   Mean Uncertainty: {uncertainty_analysis['mean_uncertainty']:.4f}")
            self.logger.info(f"   Uncertainty (Real): {uncertainty_analysis['uncertainty_by_class']['real']:.4f}")
            self.logger.info(f"   Uncertainty (Fake): {uncertainty_analysis['uncertainty_by_class']['fake']:.4f}")
        
        return evaluation_results
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
        
    def setup_logging(self, save_dir=None, verbose=True):
        """Setup logging configuration"""
        log_dir = save_dir or self.log_dir
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler() if verbose else logging.NullHandler()
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

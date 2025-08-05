"""
Main script untuk training Face Anti-Spoofing CNN Model
Menggabungkan semua komponen untuk training end-to-end
"""

import os
import sys
import torch
import argparse
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.dataset import create_dataloaders, analyze_dataset
from src.models.cnn_model import create_model, count_parameters
from src.models.training import ModelTrainer, evaluate_model

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Face Anti-Spoofing CNN Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='test_img/color',
                      help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                      help='Test split ratio')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                      help='Target image size (height width)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='custom',
                      choices=['custom', 'pretrained', 'multiscale'],
                      help='Type of model to train')
    parser.add_argument('--backbone', type=str, default='resnet18',
                      help='Backbone for pretrained model')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--use_focal_loss', action='store_true',
                      help='Use focal loss for class imbalance')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--model_save_path', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for logs and tensorboard')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training')
    parser.add_argument('--evaluate_only', action='store_true',
                      help='Only evaluate model without training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting Face Anti-Spoofing Model Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load and analyze dataset
    logger.info("Loading and analyzing dataset...")
    df = analyze_dataset(args.data_dir)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, class_counts = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        target_size=tuple(args.target_size),
        num_workers=args.num_workers,
        random_state=args.seed
    )
    
    logger.info(f"Class distribution: {class_counts}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model_kwargs = {
        'num_classes': 2,
        'dropout_rate': args.dropout_rate
    }
    
    if args.model_type == 'pretrained':
        model_kwargs['backbone'] = args.backbone
        model_kwargs['pretrained'] = True
    
    model = create_model(args.model_type, **model_kwargs)
    
    # Print model info
    logger.info("Model Information:")
    count_parameters(model)
    
    # Move model to device
    model.to(device)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_save_path=args.model_save_path,
        log_dir=args.log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        epoch, val_acc = trainer.load_model(args.resume)
        logger.info(f"Resumed from epoch {epoch} with validation accuracy {val_acc:.4f}")
    
    # Evaluate only mode
    if args.evaluate_only:
        logger.info("Evaluation mode - testing model...")
        
        if args.resume is None:
            logger.error("--resume must be specified for evaluation mode")
            return
        
        # Evaluate on test set
        results = evaluate_model(model, test_loader, device)
        
        logger.info("Evaluation completed successfully!")
        return
    
    # Training mode
    logger.info("Starting training...")
    
    try:
        # Train model
        history = trainer.train(
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            class_counts=class_counts,
            use_focal_loss=args.use_focal_loss,
            patience=args.patience
        )
        
        logger.info("Training completed successfully!")
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        
        # Load best model for evaluation
        best_model_path = os.path.join(args.model_save_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            trainer.load_model('best_model.pth')
            logger.info("Loaded best model for final evaluation")
        
        results = evaluate_model(model, test_loader, device)
        
        # Save final results
        import json
        results_to_save = {
            'training_args': vars(args),
            'class_counts': class_counts,
            'final_test_accuracy': float(results['accuracy']),
            'final_test_auc': float(results['auc']),
            'per_class_precision': [float(p) for p in results['precision']],
            'per_class_recall': [float(r) for r in results['recall']],
            'per_class_f1': [float(f) for f in results['f1']],
            'training_history': history
        }
        
        results_file = os.path.join(args.log_dir, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Final results saved to {results_file}")
        logger.info(f"Final Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Final Test AUC: {results['auc']:.4f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save current state
        if trainer.current_challenge:
            trainer.save_model(
                epoch=-1, 
                val_acc=0.0, 
                filename='interrupted_model.pth'
            )
            logger.info("Saved interrupted model checkpoint")
            
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

def quick_test():
    """
    Quick test dengan parameter default untuk development
    """
    import subprocess
    import sys
    
    # Test dengan parameter minimal
    cmd = [
        sys.executable, 
        'train_model.py',
        '--data_dir', 'test_img/color',
        '--batch_size', '16',
        '--epochs', '5',
        '--model_type', 'custom',
        '--learning_rate', '0.001',
        '--target_size', '224', '224'
    ]
    
    print("Running quick test...")
    print("Command:", ' '.join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")

if __name__ == "__main__":
    # Check if running in quick test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--quick_test':
        quick_test()
    else:
        main()

"""
Enhanced Configuration management untuk Face Anti-Spoofing system
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .logger import get_logger

@dataclass
class ModelConfig:
    """Konfigurasi model"""
    architecture: str = "custom_cnn"  # custom_cnn, resnet18, efficientnet_b0
    num_classes: int = 2
    input_size: tuple = (224, 224)
    dropout_rate: float = 0.3
    pretrained: bool = True
    freeze_backbone: bool = False
    save_path: str = "models/face_antispoofing_model.pth"

@dataclass
class TrainingConfig:
    """Konfigurasi training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler: str = "step"
    warmup_epochs: int = 0
    num_workers: int = 4
    early_stopping_patience: int = 10
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
@dataclass
class DataConfig:
    """Konfigurasi data"""
    data_dir: str = "test_img/color"
    dataset_path: str = "test_img"  # Added for compatibility
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    augment_data: bool = True
    augmentation: bool = True  # Added for compatibility
    normalize: bool = True
    resize_to: list = None  # Added for compatibility
    
    def __post_init__(self):
        if self.resize_to is None:
            self.resize_to = [224, 224]
    
@dataclass
class DetectionConfig:
    """Konfigurasi deteksi"""
    landmark_method: str = "mediapipe"  # mediapipe or dlib
    confidence_threshold: float = 0.7
    min_detection_confidence: float = 0.5  # Added for compatibility
    min_tracking_confidence: float = 0.5   # Added for compatibility
    blink_ear_threshold: float = 0.25
    blink_consecutive_frames: int = 2
    head_movement_threshold: float = 10.0
    head_pose_threshold: float = 15.0       # Added for compatibility
    
@dataclass
class ChallengeConfig:
    """Konfigurasi challenge system"""
    enable_challenges: bool = True
    enabled: bool = True  # Added for compatibility with JSON config
    challenge_types: list = None  # Added for compatibility
    timeout_seconds: float = 10.0  # Added for compatibility
    min_challenges: int = 2  # Added for compatibility
    success_threshold: float = 0.8  # Added for compatibility
    blink_timeout: float = 10.0
    head_movement_timeout: float = 15.0
    sequence_timeout: float = 20.0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.challenge_types is None:
            self.challenge_types = ["blink", "head_movement", "smile"]
    
@dataclass
class WebConfig:
    """Konfigurasi web application"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "your-secret-key-here"
    database_url: str = "sqlite:///attendance.db"
    
@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    detection: DetectionConfig
    challenge: ChallengeConfig
    web: WebConfig
    
    device: str = "auto"  # auto, cpu, cuda
    log_level: str = "INFO"
    save_predictions: bool = True
    model_save_path: str = "models/best_model.pth"
    logs_dir: str = "logs"

def get_default_config() -> SystemConfig:
    """Get default system configuration"""
    return SystemConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        detection=DetectionConfig(),
        challenge=ChallengeConfig(),
        web=WebConfig()
    )

def save_config(config: SystemConfig, config_path: str):
    """Save configuration to file"""
    config_dict = asdict(config)
    
    # Ensure config directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


class ConfigManager:
    """Enhanced configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.config_path = config_path or "config/default.json"
        self.project_root = Path(__file__).parent.parent.parent
    
    def load_config(self) -> SystemConfig:
        """Load configuration with fallback to defaults"""
        config_file = self.project_root / self.config_path
        
        if config_file.exists():
            try:
                return load_config(str(config_file))
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_file}: {e}")
                self.logger.info("Using default configuration")
        
        return get_default_config()
    
    def save_config(self, config: SystemConfig) -> bool:
        """Save configuration to file"""
        try:
            config_file = self.project_root / self.config_path
            save_config(config, str(config_file))
            self.logger.info(f"Configuration saved to {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate current configuration"""
        try:
            config = self.load_config()
            
            # Basic validation checks
            if not config.model.architecture:
                self.logger.error("Model architecture not specified")
                return False
            
            if config.training.num_epochs <= 0:
                self.logger.error("Invalid training epochs")
                return False
            
            if config.web.port <= 0 or config.web.port > 65535:
                self.logger.error("Invalid web port")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

def load_config(config_path: str) -> SystemConfig:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config objects from dict
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    detection_config = DetectionConfig(**config_dict.get('detection', {}))
    challenge_config = ChallengeConfig(**config_dict.get('challenge', {}))
    web_config = WebConfig(**config_dict.get('web', {}))
    
    # Remove nested configs from main dict
    main_config = {k: v for k, v in config_dict.items() 
                   if k not in ['model', 'training', 'data', 'detection', 'challenge', 'web']}
    
    return SystemConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        detection=detection_config,
        challenge=challenge_config,
        web=web_config,
        **main_config
    )

def create_quick_test_config() -> SystemConfig:
    """Create configuration for quick testing"""
    config = get_default_config()
    
    # Reduce parameters for quick testing
    config.training.num_epochs = 5
    config.training.batch_size = 16
    config.training.early_stopping_patience = 3
    
    # Use smaller model
    config.model.architecture = "custom_cnn"
    config.model.input_size = (128, 128)
    
    # Enable debug mode
    config.web.debug = True
    
    return config

def create_production_config() -> SystemConfig:
    """Create configuration for production"""
    config = get_default_config()
    
    # Production parameters
    config.training.num_epochs = 100
    config.training.batch_size = 32
    config.training.learning_rate = 0.0001
    
    # Use robust model
    config.model.architecture = "efficientnet_b0"
    config.model.pretrained = True
    
    # Production web settings
    config.web.debug = False
    config.web.host = "0.0.0.0"
    
    return config

# Predefined configurations
CONFIGS = {
    'default': get_default_config,
    'quick_test': create_quick_test_config,
    'production': create_production_config
}

def get_config(config_name: str = 'default') -> SystemConfig:
    """Get predefined configuration by name"""
    if config_name in CONFIGS:
        return CONFIGS[config_name]()
    else:
        return get_default_config()

if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Create and save default config
    config = get_default_config()
    save_config(config, "config_default.json")
    print("✓ Default config saved")
    
    # Create and save quick test config
    quick_config = create_quick_test_config()
    save_config(quick_config, "config_quick_test.json")
    print("✓ Quick test config saved")
    
    # Test loading
    loaded_config = load_config("config_default.json")
    print("✓ Config loaded successfully")
    
    print(f"Model architecture: {loaded_config.model.architecture}")
    print(f"Training epochs: {loaded_config.training.num_epochs}")
    print(f"Web port: {loaded_config.web.port}")

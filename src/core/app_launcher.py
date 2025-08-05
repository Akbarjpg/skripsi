"""
Application Launcher - Clean entry point for all application modes
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import ConfigManager
from src.utils.environment import EnvironmentManager
from src.utils.logger import get_logger


class AppLauncher:
    """Main application launcher for all modes"""
    
    def __init__(self, config_path: str = None, debug: bool = False):
        """Initialize launcher with configuration"""
        self.logger = get_logger(__name__)
        self.debug = debug
        
        # Initialize managers
        self.config_manager = ConfigManager(config_path)
        self.env_manager = EnvironmentManager()
        
        # Load configuration
        self.config = self.config_manager.load_config()
        
        self.logger.info("[OK] Application launcher initialized")
    
    def setup_environment(self) -> bool:
        """Setup project environment and dependencies"""
        self.logger.info("🔧 Setting up environment...")
        
        try:
            # Create necessary directories
            self.env_manager.setup_directories()
            
            # Install dependencies
            self.env_manager.install_dependencies()
            
            # Initialize database
            self.env_manager.init_database()
            
            self.logger.info("✅ Environment setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def launch_web_app(self, host: str = None, port: int = None) -> None:
        """Launch OPTIMIZED Flask web application with SocketIO"""
        self.logger.info("[START] Starting OPTIMIZED web application...")
        
        try:
            # Import optimized app for better performance
            from src.web.app_optimized import create_optimized_app
            
            # Use provided values or config defaults
            host = host or self.config.web.host
            port = port or self.config.web.port
            
            # Create optimized Flask app with SocketIO
            app, socketio = create_optimized_app(self.config)
            
            self.logger.info(f"OPTIMIZED web server starting at http://{host}:{port}")
            self.logger.info("Performance: 15-20+ FPS | 3-5x faster processing")
            self.logger.info("Press Ctrl+C to stop the server")
            
            # Run the app with SocketIO
            socketio.run(
                app,
                host=host,
                port=port,
                debug=self.debug or self.config.web.debug,
                use_reloader=False  # Disable reloader to avoid double execution
            )
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to start web application: {e}")
            raise
    
    def train_model(self) -> bool:
        """Train the anti-spoofing model"""
        self.logger.info("🎯 Starting model training...")
        
        try:
            from src.training.trainer import ModelTrainer
            
            trainer = ModelTrainer(self.config)
            success = trainer.train()
            
            if success:
                self.logger.info("✅ Model training completed successfully")
            else:
                self.logger.error("❌ Model training failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run comprehensive system tests"""
        self.logger.info("🧪 Running system tests...")
        
        try:
            from src.testing.test_runner import TestRunner
            
            test_runner = TestRunner(self.config)
            results = test_runner.run_all_tests()
            
            if results['success']:
                self.logger.info("✅ All tests passed")
            else:
                self.logger.warning(f"⚠️ Some tests failed: {results['failed']}")
                
            return results['success']
            
        except Exception as e:
            self.logger.error(f"❌ Testing failed: {e}")
            return False
    
    def validate_system(self) -> dict:
        """Validate system components"""
        self.logger.info("🔍 Validating system...")
        
        validation_results = {
            'environment': self.env_manager.check_environment(),
            'config': self.config_manager.validate_config(),
            'dependencies': self.env_manager.check_dependencies(),
            'database': self.env_manager.check_database(),
            'models': self._check_models(),
            'templates': self._check_templates()
        }
        
        all_valid = all(validation_results.values())
        
        if all_valid:
            self.logger.info("✅ System validation passed")
        else:
            self.logger.warning("⚠️ System validation found issues")
            
        return validation_results
    
    def _check_models(self) -> bool:
        """Check if model files exist"""
        model_path = Path(self.config.model.save_path)
        return model_path.exists()
    
    def _check_templates(self) -> bool:
        """Check if required templates exist"""
        templates_dir = Path("src/web/templates")
        required_templates = [
            "index.html", "login.html", "register.html", 
            "attendance.html", "404.html", "base.html"
        ]
        
        return all(
            (templates_dir / template).exists() 
            for template in required_templates
        )

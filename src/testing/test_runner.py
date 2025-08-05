"""
Comprehensive testing module for Face Anti-Spoofing system
"""

import os
import sys
import time
import unittest
from pathlib import Path
from typing import Dict, Any, List
import sqlite3

from ..utils.logger import get_logger
from ..utils.config import SystemConfig
from ..utils.environment import EnvironmentManager


class TestRunner:
    """Comprehensive test runner for the entire system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.env_manager = EnvironmentManager()
        
        self.test_results = {
            'environment': False,
            'dependencies': False,
            'database': False,
            'models': False,
            'web_app': False,
            'integration': False
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        self.logger.info("🧪 Starting comprehensive system tests...")
        
        start_time = time.time()
        
        # Run each test category
        self.test_results['environment'] = self._test_environment()
        self.test_results['dependencies'] = self._test_dependencies()
        self.test_results['database'] = self._test_database()
        self.test_results['models'] = self._test_models()
        self.test_results['web_app'] = self._test_web_app()
        self.test_results['integration'] = self._test_integration()
        
        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"🏁 Testing completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"📊 Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.info("✅ All tests passed!")
        else:
            failed_tests = [test for test, result in self.test_results.items() if not result]
            self.logger.warning(f"❌ Failed tests: {failed_tests}")
        
        return {
            'success': passed_tests == total_tests,
            'passed': passed_tests,
            'total': total_tests,
            'failed': [test for test, result in self.test_results.items() if not result],
            'elapsed_time': elapsed_time,
            'details': self.test_results
        }
    
    def _test_environment(self) -> bool:
        """Test environment setup"""
        self.logger.info("Testing environment...")
        
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                self.logger.error("Python 3.8+ required")
                return False
            
            # Check essential directories
            essential_dirs = [
                "src", "src/web", "src/models", "src/utils",
                "src/web/templates"
            ]
            
            for directory in essential_dirs:
                if not Path(directory).exists():
                    self.logger.error(f"Missing directory: {directory}")
                    return False
            
            # Check essential files
            essential_files = [
                "main.py", "requirements.txt",
                "src/web/app_clean.py",
                "src/utils/config.py"
            ]
            
            for file_path in essential_files:
                if not Path(file_path).exists():
                    self.logger.error(f"Missing file: {file_path}")
                    return False
            
            self.logger.info("✅ Environment test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment test failed: {e}")
            return False
    
    def _test_dependencies(self) -> bool:
        """Test if all dependencies are available"""
        self.logger.info("Testing dependencies...")
        
        try:
            # Test core dependencies
            import torch
            import numpy as np
            from flask import Flask
            import cv2
            
            # Test versions
            self.logger.info(f"PyTorch: {torch.__version__}")
            self.logger.info(f"NumPy: {np.__version__}")
            self.logger.info(f"OpenCV: {cv2.__version__}")
            
            # Test CUDA if available
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                self.logger.info("CUDA not available, using CPU")
            
            self.logger.info("✅ Dependencies test passed")
            return True
            
        except ImportError as e:
            self.logger.error(f"Missing dependency: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Dependencies test failed: {e}")
            return False
    
    def _test_database(self) -> bool:
        """Test database functionality"""
        self.logger.info("Testing database...")
        
        try:
            # Test database connection
            test_db_path = "test_attendance.db"
            
            with sqlite3.connect(test_db_path) as conn:
                cursor = conn.cursor()
                
                # Create test table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS test_users (
                        id INTEGER PRIMARY KEY,
                        username TEXT NOT NULL
                    )
                ''')
                
                # Insert test data
                cursor.execute("INSERT INTO test_users (username) VALUES (?)", ("test_user",))
                
                # Read test data
                cursor.execute("SELECT username FROM test_users WHERE username = ?", ("test_user",))
                result = cursor.fetchone()
                
                if not result or result[0] != "test_user":
                    self.logger.error("Database read/write test failed")
                    return False
                
                # Clean up
                cursor.execute("DROP TABLE test_users")
                conn.commit()
            
            # Remove test database
            if Path(test_db_path).exists():
                Path(test_db_path).unlink()
            
            self.logger.info("✅ Database test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database test failed: {e}")
            return False
    
    def _test_models(self) -> bool:
        """Test model functionality"""
        self.logger.info("Testing models...")
        
        try:
            # Test model creation
            from ..models.simple_model import SimpleLivenessModel
            import torch
            
            model = SimpleLivenessModel()
            
            # Test model forward pass
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(test_input)
            
            if output.shape[0] != 1 or output.shape[1] != 2:
                self.logger.error(f"Unexpected model output shape: {output.shape}")
                return False
            
            self.logger.info("✅ Models test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Models test failed: {e}")
            return False
    
    def _test_web_app(self) -> bool:
        """Test web application"""
        self.logger.info("Testing web application...")
        
        try:
            # Test Flask app creation
            from ..web.app_clean import create_app
            
            app = create_app()
            
            # Test app configuration
            if not app.config.get('SECRET_KEY'):
                self.logger.error("Flask app missing secret key")
                return False
            
            # Test routes exist
            with app.test_client() as client:
                # Test home page
                response = client.get('/')
                if response.status_code not in [200, 302]:  # Allow redirects
                    self.logger.error(f"Home page test failed: {response.status_code}")
                    return False
                
                # Test login page
                response = client.get('/login')
                if response.status_code != 200:
                    self.logger.error(f"Login page test failed: {response.status_code}")
                    return False
            
            self.logger.info("✅ Web app test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Web app test failed: {e}")
            return False
    
    def _test_integration(self) -> bool:
        """Test system integration"""
        self.logger.info("Testing system integration...")
        
        try:
            # Test configuration loading
            from ..utils.config import ConfigManager
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            if not config:
                self.logger.error("Configuration loading failed")
                return False
            
            # Test environment manager
            env_valid = self.env_manager.check_environment()
            if not env_valid:
                self.logger.error("Environment validation failed")
                return False
            
            self.logger.info("✅ Integration test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Integration test failed: {e}")
            return False


class UnitTestSuite(unittest.TestCase):
    """Unit tests for individual components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = self._get_test_config()
    
    def _get_test_config(self):
        """Get test configuration"""
        from ..utils.config import get_default_config
        return get_default_config()
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsNotNone(self.config)
        self.assertIsNotNone(self.config.model)
        self.assertIsNotNone(self.config.web)
    
    def test_model_creation(self):
        """Test model creation"""
        from ..models.simple_model import SimpleLivenessModel
        import torch
        
        model = SimpleLivenessModel()
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test forward pass
        test_input = torch.randn(1, 3, 224, 224)
        output = model(test_input)
        self.assertEqual(output.shape, (1, 2))
    
    def test_logger(self):
        """Test logging functionality"""
        from ..utils.logger import get_logger
        
        logger = get_logger("test")
        self.assertIsNotNone(logger)
        
        # Test logging doesn't crash
        logger.info("Test log message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")


def run_unit_tests():
    """Run unit tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(UnitTestSuite)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

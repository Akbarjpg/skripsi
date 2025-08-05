"""
Environment management for Face Anti-Spoofing system
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import importlib.util

from .logger import get_logger


class EnvironmentManager:
    """Manages system environment, dependencies, and setup"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.project_root = Path(__file__).parent.parent.parent
        
    def setup_directories(self) -> List[str]:
        """Create necessary project directories"""
        directories = [
            "logs", "models", "data/processed", "data/splits",
            "checkpoints", "outputs", "config"
        ]
        
        created_dirs = []
        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
                self.logger.info(f"Created directory: {dir_path}")
        
        return created_dirs
    
    def check_environment(self) -> bool:
        """Check if environment is properly set up"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                self.logger.error(f"Python 3.8+ required, got {python_version}")
                return False
            
            # Check essential directories exist
            essential_dirs = ["src", "src/web", "src/models", "src/utils"]
            for directory in essential_dirs:
                dir_path = self.project_root / directory
                if not dir_path.exists():
                    self.logger.error(f"Missing directory: {dir_path}")
                    return False
            
            self.logger.info("✅ Environment check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment check failed: {e}")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are installed"""
        required_packages = {
            'torch': 'torch',
            'cv2': 'opencv-python',
            'flask': 'flask',
            'numpy': 'numpy',
            'PIL': 'Pillow',
            'mediapipe': 'mediapipe',
        }
        
        results = {}
        for module_name, package_name in required_packages.items():
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    results[package_name] = True
                    self.logger.debug(f"✅ {package_name} is installed")
                else:
                    results[package_name] = False
                    self.logger.warning(f"❌ {package_name} is missing")
            except ImportError:
                results[package_name] = False
                self.logger.warning(f"❌ {package_name} is missing")
        
        return results
    
    def install_dependencies(self) -> bool:
        """Install project dependencies from requirements.txt"""
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            self.logger.error("requirements.txt not found")
            return False
        
        try:
            self.logger.info("Installing dependencies...")
            
            # Use pip to install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True, check=True)
            
            self.logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            return False
    
    def init_database(self) -> bool:
        """Initialize SQLite database with required tables"""
        db_path = self.project_root / "attendance.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        full_name TEXT NOT NULL,
                        email TEXT,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        face_encoding BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Attendance records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        check_type TEXT DEFAULT 'in',
                        confidence_score REAL,
                        liveness_score REAL,
                        challenge_passed BOOLEAN DEFAULT 0,
                        image_path TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # Verification sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS verification_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        user_id INTEGER,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        status TEXT DEFAULT 'pending',
                        challenges_completed INTEGER DEFAULT 0,
                        final_score REAL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                conn.commit()
                self.logger.info("✅ Database initialized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False
    
    def check_database(self) -> bool:
        """Check if database is properly set up"""
        db_path = self.project_root / "attendance.db"
        
        if not db_path.exists():
            return False
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check if required tables exist
                required_tables = ['users', 'attendance_records', 'verification_sessions']
                
                for table in required_tables:
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)
                    )
                    if not cursor.fetchone():
                        self.logger.error(f"Missing table: {table}")
                        return False
                
                self.logger.info("✅ Database check passed")
                return True
                
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'project_root': str(self.project_root),
            'environment_valid': self.check_environment(),
            'database_valid': self.check_database(),
            'dependencies': self.check_dependencies()
        }
        
        return info
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files and caches"""
        temp_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/logs/*.log"
        ]
        
        cleaned_files = []
        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                        cleaned_files.append(str(file_path))
                except Exception as e:
                    self.logger.warning(f"Could not clean {file_path}: {e}")
        
        if cleaned_files:
            self.logger.info(f"Cleaned {len(cleaned_files)} temporary files")
        
        return cleaned_files

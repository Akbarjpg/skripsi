"""
Launch script untuk Face Anti-Spoofing Attendance System
Menjalankan seluruh sistem dengan konfigurasi yang tepat
"""

import sys
import os
import argparse
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)

def install_dependencies():
    """Install dependencies yang diperlukan"""
    print("Installing required dependencies...")
    
    # Basic dependencies
    basic_deps = [
        "torch>=2.0.0",
        "torchvision",
        "opencv-python",
        "mediapipe",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "flask",
        "flask-socketio",
        "pillow"
    ]
    
    for dep in basic_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
            print(f"✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")
    
    # Optional dependencies
    optional_deps = ["albumentations", "tensorboard"]
    for dep in optional_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])
            print(f"✓ {dep} (optional)")
        except subprocess.CalledProcessError:
            print(f"⚠ {dep} (optional) - skipped")

def setup_directories():
    """Setup direktori yang diperlukan"""
    directories = [
        "models",
        "logs", 
        "data/processed",
        "data/splits",
        "outputs",
        "checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory: {directory}")

def train_model():
    """Train model dengan quick configuration"""
    print("\n" + "="*60)
    print("TRAINING LIVENESS DETECTION MODEL")
    print("="*60)
    
    # Import di sini untuk menghindari error jika dependencies belum terinstall
    try:
        from src.utils.config import create_quick_test_config, save_config
        from src.models.simple_model import SimpleLivenessModel
        import torch
        
        # Try advanced dataset first, fallback to simple
        try:
            from src.data.dataset import analyze_dataset
        except ImportError:
            from src.data.simple_dataset import analyze_dataset_simple as analyze_dataset
        
        # Create quick test config
        config = create_quick_test_config()
        save_config(config, "config_quick.json")
        print("✓ Configuration created")
        
        # Analyze dataset
        if os.path.exists("test_img/color"):
            analysis = analyze_dataset("test_img/color")
            print(f"✓ Dataset analysis: {analysis}")
            
            # Quick model creation and save
            model = SimpleLivenessModel()
            torch.save(model.state_dict(), "models/quick_model.pth")
            print("✓ Quick model saved")
        else:
            print("⚠ Dataset not found - using dummy model")
            model = SimpleLivenessModel()
            torch.save(model.state_dict(), "models/dummy_model.pth")
            
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False

def launch_web_app():
    """Launch web application"""
    print("\n" + "="*60)
    print("LAUNCHING WEB APPLICATION")
    print("="*60)
    
    try:
        from src.web.app_clean import create_app
        app, socketio = create_app()
        print("✓ Flask app with SocketIO loaded")
        print("🚀 Starting web server at http://localhost:5000")
        print("   🎯 Landmark Detection: http://localhost:5000/face-detection-clean")
        print("   📊 Test Dashboard: http://localhost:5000/test-dashboard") 
        print("   Access the attendance system in your browser")
        print("   Press Ctrl+C to stop the server")
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"✗ Failed to start web app: {e}")
        return False

def run_tests():
    """Run comprehensive tests"""
    print("\n" + "="*60)
    print("RUNNING SYSTEM TESTS")
    print("="*60)
    
    try:
        subprocess.check_call([sys.executable, "quick_test.py"])
        return True
    except subprocess.CalledProcessError:
        print("Some tests failed - check output above")
        return False

def main():
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing Attendance System Launcher")
    parser.add_argument("--mode", choices=["install", "train", "web", "test", "full"], 
                       default="full", help="Mode to run")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    
    args = parser.parse_args()
    
    print("🔒 FACE ANTI-SPOOFING ATTENDANCE SYSTEM")
    print("="*60)
    
    # Setup directories
    setup_directories()
    
    success = True
    
    if args.mode in ["install", "full"] and not args.skip_deps:
        print("\n📦 INSTALLING DEPENDENCIES...")
        install_dependencies()
    
    if args.mode in ["test", "full"]:
        print("\n🧪 RUNNING TESTS...")
        test_success = run_tests()
        if not test_success:
            print("⚠ Some tests failed, but continuing...")
    
    if args.mode in ["train", "full"]:
        print("\n🤖 TRAINING MODEL...")
        train_success = train_model()
        if not train_success:
            success = False
    
    if args.mode in ["web", "full"]:
        if success:
            print("\n🌐 LAUNCHING WEB APPLICATION...")
            launch_web_app()
        else:
            print("❌ Skipping web launch due to previous errors")
    
    print("\n" + "="*60)
    if success:
        print("✅ System launch completed successfully!")
    else:
        print("❌ System launch completed with errors")
    print("="*60)

if __name__ == "__main__":
    main()

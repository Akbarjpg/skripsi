#!/usr/bin/env python3
"""
Setup Script for Step 3 Enhanced Challenge System
Run this on a new device to install all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version compatible")
    return True

def setup_virtual_environment():
    """Set up virtual environment"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    print("📁 Creating virtual environment...")
    return run_command(f"{sys.executable} -m venv .venv", 
                      "Creating virtual environment")

def get_activation_command():
    """Get the correct activation command for the platform"""
    if sys.platform == "win32":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"

def install_requirements():
    """Install requirements using pip"""
    
    # Upgrade pip first
    pip_upgrade = f"{sys.executable} -m pip install --upgrade pip"
    if not run_command(pip_upgrade, "Upgrading pip"):
        return False
    
    # Install requirements
    requirements_file = "requirements_step3_minimal.txt"
    if not Path(requirements_file).exists():
        print(f"❌ {requirements_file} not found!")
        return False
    
    install_cmd = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(install_cmd, f"Installing packages from {requirements_file}")

def test_imports():
    """Test if all critical imports work"""
    print(f"\n{'='*60}")
    print("🧪 TESTING IMPORTS")
    print(f"{'='*60}")
    
    critical_modules = {
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe', 
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pygame': 'Pygame (Audio)',
        'pyttsx3': 'Text-to-Speech',
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO'
    }
    
    failed_imports = []
    
    for module, name in critical_modules.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    
    print("\n🎉 All critical imports successful!")
    return True

def test_camera():
    """Test camera access"""
    print(f"\n{'='*60}")
    print("📷 TESTING CAMERA ACCESS")
    print(f"{'='*60}")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"✅ Camera working - Frame size: {frame.shape}")
            return True
        else:
            print("❌ Cannot read frames from camera")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_audio():
    """Test audio system (voice disabled)"""
    print(f"\n{'='*60}")
    print("🔊 TESTING AUDIO SYSTEM (Voice Disabled)") 
    print(f"{'='*60}")
    
    try:
        import pygame
        pygame.mixer.init()
        print("✅ Pygame audio initialized")
        
        # Remove pyttsx3 test
        print("ℹ️  Text-to-speech disabled by user preference")
        
        return True
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 STEP 3 ENHANCED CHALLENGE SYSTEM SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup virtual environment (recommended)
    print(f"\n🤔 Do you want to use a virtual environment? (recommended)")
    use_venv = input("Enter 'y' for yes, 'n' for no: ").strip().lower()
    
    if use_venv == 'y':
        if not setup_virtual_environment():
            return False
        
        activation_cmd = get_activation_command()
        print(f"\n📝 TO ACTIVATE VIRTUAL ENVIRONMENT:")
        print(f"   {activation_cmd}")
        print(f"\n⚠️  Please activate the virtual environment and run this script again!")
        print(f"   Or install packages globally by running with 'n'")
        return True
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Test everything
    if not test_imports():
        return False
    
    if not test_camera():
        print("⚠️  Camera not working - some features may not work")
    
    if not test_audio():
        print("⚠️  Audio not working - Step 3 audio features disabled")
    
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETE!")
    print(f"{'='*60}")
    print("✅ Step 3 Enhanced Challenge System is ready!")
    print("\n📋 NEXT STEPS:")
    print("1. Test the system: python test_step3_enhanced_challenges.py")
    print("2. Run web interface: python -m src.web.app")
    print("3. Or run minimal test: python minimal_antispoofing.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed! Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup successful!")
        sys.exit(0)

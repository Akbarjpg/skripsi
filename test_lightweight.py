#!/usr/bin/env python3
"""
Quick test untuk memastikan semua import bekerja tanpa matplotlib/seaborn
"""

print("🔍 Testing lightweight imports...")

try:
    print("1. Testing utils import...")
    from src.utils import setup_directories
    print("✅ Utils imported successfully")
    
    print("2. Testing core app launcher...")
    from src.core.app_launcher import AppLauncher
    print("✅ AppLauncher imported successfully")
    
    print("3. Testing landmark detection...")
    from src.detection.landmark_detection import FacialLandmarkDetector
    print("✅ FacialLandmarkDetector imported successfully")
    
    print("4. Testing model training...")
    from src.models.training import CNNTrainer
    print("✅ CNNTrainer imported successfully")
    
    print("5. Testing web app...")
    from src.web.app_clean import create_app
    print("✅ Flask app factory imported successfully")
    
    print("\n🎉 All core components imported successfully!")
    print("✅ System is ready for lightweight deployment")
    print("📝 No matplotlib/seaborn/tensorboard required for basic operation")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some dependencies may still be missing")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🚀 To start the system:")
print("python main.py --mode web")

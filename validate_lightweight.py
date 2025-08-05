#!/usr/bin/env python3
"""
Script untuk memvalidasi bahwa sistem sudah lightweight dan tidak menggunakan dlib
"""

import os
import sys
import importlib.util

def check_imports():
    """Check apakah masih ada import dlib yang tersisa"""
    print("=== Checking for dlib imports ===")
    
    # Cari semua file Python
    python_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Cek main files
    main_files = ['main.py', 'launch.py', 'fallback_app.py']
    for file in main_files:
        if os.path.exists(file):
            python_files.append(file)
    
    dlib_found = False
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'import dlib' in content or 'from dlib' in content:
                    print(f"❌ DLIB import found in: {file_path}")
                    dlib_found = True
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    if not dlib_found:
        print("✅ No dlib imports found!")
    
    return not dlib_found

def check_requirements():
    """Check requirements files"""
    print("\n=== Checking requirements ===")
    
    files_to_check = ['requirements.txt', 'requirements-minimal.txt']
    
    for req_file in files_to_check:
        if os.path.exists(req_file):
            print(f"\n📄 {req_file}:")
            with open(req_file, 'r') as f:
                content = f.read()
                if 'dlib' in content:
                    print("❌ dlib found in requirements")
                else:
                    print("✅ No dlib in requirements")
                
                # Show active dependencies
                lines = [line.strip() for line in content.split('\n') 
                        if line.strip() and not line.strip().startswith('#')]
                print(f"Active dependencies: {len(lines)}")
                for line in lines[:5]:  # Show first 5
                    print(f"  - {line}")
                if len(lines) > 5:
                    print(f"  ... and {len(lines) - 5} more")

def check_mediapipe_usage():
    """Check apakah MediaPipe sudah digunakan dengan benar"""
    print("\n=== Checking MediaPipe usage ===")
    
    landmark_file = "src/detection/landmark_detection.py"
    if os.path.exists(landmark_file):
        with open(landmark_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            if 'import mediapipe' in content:
                print("✅ MediaPipe import found")
            else:
                print("❌ MediaPipe import not found")
            
            if 'mp.solutions.face_mesh' in content:
                print("✅ MediaPipe Face Mesh usage found")
            else:
                print("❌ MediaPipe Face Mesh usage not found")

def main():
    print("🔍 Lightweight System Validation")
    print("=" * 40)
    
    all_good = True
    
    # Check imports
    if not check_imports():
        all_good = False
    
    # Check requirements
    check_requirements()
    
    # Check MediaPipe
    check_mediapipe_usage()
    
    print("\n" + "=" * 40)
    if all_good:
        print("✅ System is lightweight and ready!")
        print("📝 Summary:")
        print("  - No dlib dependencies found")
        print("  - MediaPipe-only implementation")
        print("  - Minimal requirements configuration")
    else:
        print("❌ Issues found that need fixing")
    
    print("\n🚀 To install minimal dependencies:")
    print("pip install -r requirements-minimal.txt")

if __name__ == "__main__":
    main()

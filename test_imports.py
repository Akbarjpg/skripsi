"""Minimal test to debug imports step by step"""

print("1. Testing basic imports...")
try:
    import os
    import sys
    import time
    import json
    import csv
    import sqlite3
    print("✅ Basic modules imported")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    exit(1)

print("2. Testing numpy and cv2...")
try:
    import numpy as np
    import cv2
    print("✅ NumPy and OpenCV imported")
except Exception as e:
    print(f"❌ NumPy/OpenCV import failed: {e}")
    exit(1)

print("3. Testing psutil and platform...")
try:
    import psutil
    import platform
    print("✅ psutil and platform imported")
except Exception as e:
    print(f"❌ psutil/platform import failed: {e}")
    exit(1)

print("4. Testing project path setup...")
try:
    sys.path.append(os.path.dirname(__file__))
    print(f"✅ Added {os.path.dirname(__file__)} to path")
except Exception as e:
    print(f"❌ Path setup failed: {e}")
    exit(1)

print("5. Testing metrics collector import...")
try:
    from src.testing.metrics_collector import MetricsCollector
    print("✅ MetricsCollector imported")
except Exception as e:
    print(f"❌ MetricsCollector import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("6. Testing report generator import...")
try:
    from src.testing.report_generator import ReportGenerator
    print("✅ ReportGenerator imported")
except Exception as e:
    print(f"❌ ReportGenerator import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("7. Testing data exporter import...")
try:
    from src.testing.data_exporter import DataExporter
    print("✅ DataExporter imported")
except Exception as e:
    print(f"❌ DataExporter import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("8. All imports successful! 🎉")

"""Minimal Step 7 Test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("1. Testing SimpleAttendanceLogger import...")
try:
    from src.logging.simple_attendance_logger import SimpleAttendanceLogger
    print("✅ SimpleAttendanceLogger - OK")
except Exception as e:
    print(f"❌ SimpleAttendanceLogger - {e}")
    exit(1)

print("\n2. Testing BasicReportGenerator import...")
try:
    from src.reports.basic_report_generator_simple import BasicReportGenerator
    print("✅ BasicReportGenerator - OK")
except Exception as e:
    print(f"❌ BasicReportGenerator - {e}")
    exit(1)

print("\n3. Testing admin blueprint import...")
try:
    from src.web.basic_admin_view import admin_bp
    print("✅ Admin Blueprint - OK")
except Exception as e:
    print(f"❌ Admin Blueprint - {e}")
    exit(1)

print("\n4. Testing simple logging...")
try:
    logger = SimpleAttendanceLogger()
    logger.log_attendance_attempt(
        user_id="test_001",
        user_name="Test User", 
        status="success",
        confidence_score=0.95,
        antispoofing_passed=True,
        recognition_time=0.1,
        notes="Quick test"
    )
    print("✅ Logging test - OK")
except Exception as e:
    print(f"❌ Logging test - {e}")
    exit(1)

print("\n🎉 All Step 7 components working perfectly!")
print("Ready for thesis demonstration!")

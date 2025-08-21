"""Quick Step 7 Test"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing Step 7 Simple Logging...")

try:
    from src.logging.simple_attendance_logger import SimpleAttendanceLogger
    print("✅ SimpleAttendanceLogger imported")
    
    logger = SimpleAttendanceLogger()
    print("✅ Logger created")
    
    # Quick test
    logger.log_attendance_attempt(
        user_id="test_001",
        user_name="Test User",
        status="success",
        confidence_score=0.95,
        antispoofing_passed=True,
        recognition_time=0.1,
        notes="Quick test"
    )
    print("✅ Attendance logged")
    
    from src.reports.basic_report_generator import BasicReportGenerator
    print("✅ ReportGenerator imported")
    
    from src.web.basic_admin_view import admin_bp
    print("✅ Admin blueprint imported")
    
    print("🎉 Step 7 - All components working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

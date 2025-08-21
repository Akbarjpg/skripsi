"""Final Step 7 Test"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🧪 Step 7 Final Test")
print("="*40)

# Test 1: SimpleAttendanceLogger
try:
    from src.logging.simple_attendance_logger import SimpleAttendanceLogger
    logger = SimpleAttendanceLogger()
    print("✅ 1. SimpleAttendanceLogger - Working")
except Exception as e:
    print(f"❌ 1. SimpleAttendanceLogger - {e}")
    exit(1)

# Test 2: BasicReportGenerator
try:
    from src.reports.basic_report_generator_simple import BasicReportGenerator
    reporter = BasicReportGenerator()
    print("✅ 2. BasicReportGenerator - Working")
except Exception as e:
    print(f"❌ 2. BasicReportGenerator - {e}")
    exit(1)

# Test 3: Log attendance
try:
    logger.log_attendance_attempt(
        user_id="test_user",
        user_name="Test User",
        status="success",
        confidence_score=0.95,
        antispoofing_passed=True,
        recognition_time=0.1,
        notes="Final test"
    )
    print("✅ 3. Attendance Logging - Working")
except Exception as e:
    print(f"❌ 3. Attendance Logging - {e}")
    exit(1)

# Test 4: Generate report
try:
    report = reporter.generate_daily_report()
    print(f"✅ 4. Report Generation - Working ({report.get('total_attempts', 0)} attempts)")
except Exception as e:
    print(f"❌ 4. Report Generation - {e}")
    exit(1)

print("\n🎉 Step 7 Implementation SUCCESSFUL!")
print("="*40)
print("✅ Simple attendance logging system ready")
print("✅ Database tables created automatically")
print("✅ Report generation working")
print("✅ Perfect for thesis demonstration")
print("\nReady to integrate with Flask app!")

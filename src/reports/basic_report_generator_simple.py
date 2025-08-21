"""
Simple Basic Report Generator - Minimal Version for Step 7
"""

from pathlib import Path
from io import StringIO
import logging
import csv
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..logging.simple_attendance_logger import SimpleAttendanceLogger

# Create simple logger
logger = logging.getLogger(__name__)


class BasicReportGenerator:
    """
    Basic report generator for attendance system
    Creates simple reports suitable for thesis documentation
    """
    
    def __init__(self, logger_instance: SimpleAttendanceLogger = None, output_dir: str = "reports"):
        """
        Initialize report generator
        
        Args:
            logger_instance: SimpleAttendanceLogger instance
            output_dir: Directory to save reports
        """
        self.logger = logger_instance or SimpleAttendanceLogger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_logger = logger
    
    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """
        Generate daily attendance report
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with daily attendance statistics
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            summary = self.logger.get_daily_attendance_summary(date)
            return {
                "date": date,
                "total_attempts": summary.get("total_attempts", 0),
                "successful_attendance": summary.get("successful_count", 0),
                "failed_attempts": summary.get("failed_count", 0),
                "duplicate_attempts": summary.get("duplicate_count", 0),
                "success_rate": summary.get("success_rate", 0.0),
                "unique_users": summary.get("unique_users", 0)
            }
        except Exception as e:
            self.system_logger.error(f"Daily report generation failed: {e}")
            return {}
    
    def export_to_csv(self, filepath: str, days: int = 30) -> bool:
        """
        Export attendance data to CSV file
        
        Args:
            filepath: Path to save CSV file
            days: Number of days to export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.logger.export_attendance_data(days)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                if not data:
                    csvfile.write("No data available\n")
                    return True
                
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return True
            
        except Exception as e:
            self.system_logger.error(f"CSV export failed: {e}")
            return False
    
    def export_to_json(self, filepath: str, days: int = 30) -> bool:
        """
        Export attendance data to JSON file
        
        Args:
            filepath: Path to save JSON file
            days: Number of days to export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.logger.export_attendance_data(days)
            
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.system_logger.error(f"JSON export failed: {e}")
            return False
    
    def generate_thesis_data_export(self) -> Dict[str, Any]:
        """
        Generate comprehensive data export for thesis documentation
        
        Returns:
            Dictionary with all relevant data for thesis
        """
        try:
            return {
                "export_date": datetime.now().isoformat(),
                "daily_summary": self.generate_daily_report(),
                "weekly_data": self.logger.export_attendance_data(7),
                "monthly_data": self.logger.export_attendance_data(30),
                "system_stats": {
                    "total_users": len(self.logger.get_unique_users()),
                    "total_attempts": self.logger.get_total_attempts(),
                    "overall_success_rate": self.logger.get_overall_success_rate()
                }
            }
        except Exception as e:
            self.system_logger.error(f"Thesis data export failed: {e}")
            return {}

"""
Basic Report Generator for Step 7 - Thesis Version
Simple reporting functionality for attendance data
"""

import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
# import matplotlib.pyplot as plt  # Commented out for now
# import pandas as pd  # Commented out for now
# import seaborn as sns  # Commented out for now
from io import StringIO
import logging
import csv
import json

from ..logging.simple_attendance_logger import SimpleAttendanceLogger

# Create simple logger
logger = logging.getLogger(__name__)


class BasicReportGenerator:
    """
    Basic report generator for attendance system
    Creates simple reports suitable for thesis documentation
    """
    
    def __init__(self, logger: SimpleAttendanceLogger = None, output_dir: str = "reports"):
        """
        Initialize report generator
        
        Args:
            logger: Attendance logger instance
            output_dir: Directory to save reports
        """
        self.logger = logger or SimpleAttendanceLogger()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_logger = logger
    
    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """
        Generate daily attendance report
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with report data
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # Get daily summary
            summary = self.logger.get_daily_attendance_summary(date)
            
            # Get detailed attendance records
            attendance_records = []
            with self.logger._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, user_name, timestamp, status, 
                           confidence_score, antispoofing_passed, recognition_time
                    FROM attendance_log 
                    WHERE DATE(timestamp) = ?
                    ORDER BY timestamp
                """, (date,))
                
                for row in cursor.fetchall():
                    attendance_records.append({
                        'user_id': row[0],
                        'user_name': row[1],
                        'timestamp': row[2],
                        'status': row[3],
                        'confidence_score': row[4],
                        'antispoofing_passed': row[5],
                        'recognition_time': row[6]
                    })
            
            report = {
                'report_type': 'daily_report',
                'date': date,
                'generated_at': datetime.now().isoformat(),
                'summary': summary,
                'attendance_records': attendance_records,
                'total_unique_users': len(set(r['user_id'] for r in attendance_records if r['status'] == 'success'))
            }
            
            return report
            
        except Exception as e:
            self.system_logger.error(f"Failed to generate daily report: {e}")
            return {}
    
    def generate_weekly_report(self, start_date: str = None) -> Dict[str, Any]:
        """
        Generate weekly attendance report
        
        Args:
            start_date: Start date in YYYY-MM-DD format (default: week ago)
            
        Returns:
            Dictionary with report data
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=6)).strftime('%Y-%m-%d')
            
            # Get statistics for the week
            stats = self.logger.get_attendance_statistics(days=7)
            
            # Get daily breakdown for the week
            daily_reports = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            
            for i in range(7):
                date_str = current_date.strftime('%Y-%m-%d')
                daily_summary = self.logger.get_daily_attendance_summary(date_str)
                daily_reports.append(daily_summary)
                current_date += timedelta(days=1)
            
            report = {
                'report_type': 'weekly_report',
                'period': f"{start_date} to {end_date}",
                'generated_at': datetime.now().isoformat(),
                'overall_statistics': stats,
                'daily_breakdown': daily_reports,
                'week_summary': {
                    'total_working_days': len([d for d in daily_reports if d['total_attempts'] > 0]),
                    'avg_daily_attendance': sum(d['successful_attendance'] for d in daily_reports) / 7,
                    'best_day': max(daily_reports, key=lambda x: x['successful_attendance']) if daily_reports else None,
                    'worst_day': min(daily_reports, key=lambda x: x['successful_attendance']) if daily_reports else None
                }
            }
            
            return report
            
        except Exception as e:
            self.system_logger.error(f"Failed to generate weekly report: {e}")
            return {}
    
    def export_to_csv(self, data: List[Dict], filename: str = None) -> str:
        """
        Export data to CSV file
        
        Args:
            data: List of dictionaries to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"attendance_export_{timestamp}.csv"
            
            filepath = self.output_dir / filename
            
            if not data:
                self.system_logger.warning("No data to export")
                return str(filepath)
            
            # Get all unique keys from all dictionaries
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(list(fieldnames))
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            self.system_logger.info(f"Data exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.system_logger.error(f"Failed to export to CSV: {e}")
            return ""
    
    def export_to_json(self, data: Any, filename: str = None) -> str:
        """
        Export data to JSON file
        
        Args:
            data: Data to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"attendance_report_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False, default=str)
            
            self.system_logger.info(f"Report exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.system_logger.error(f"Failed to export to JSON: {e}")
            return ""
    
    def create_attendance_chart(self, daily_data: List[Dict], filename: str = None) -> str:
        """
        Create attendance chart for thesis documentation
        
        Args:
            daily_data: List of daily attendance data
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to chart image
        """
        try:
            if not daily_data:
                self.system_logger.warning("No data to create chart")
                return ""
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"attendance_chart_{timestamp}.png"
            
            filepath = self.output_dir / filename
            
            # Prepare data
            dates = [d['date'] for d in daily_data]
            successful = [d['successful_attendance'] for d in daily_data]
            total = [d['total_attempts'] for d in daily_data]
            
            # Create chart
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(dates, successful, marker='o', label='Successful Attendance', color='green')
            plt.plot(dates, total, marker='s', label='Total Attempts', color='blue')
            plt.title('Daily Attendance Trends')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            success_rates = [d['success_rate'] for d in daily_data if 'success_rate' in d]
            if success_rates:
                plt.bar(dates[:len(success_rates)], success_rates, color='lightblue')
                plt.title('Daily Success Rate (%)')
                plt.xlabel('Date')
                plt.ylabel('Success Rate (%)')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.system_logger.info(f"Chart saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.system_logger.error(f"Failed to create chart: {e}")
            return ""
    
    def generate_thesis_data_export(self, days: int = 30) -> Dict[str, str]:
        """
        Generate comprehensive data export for thesis documentation
        
        Args:
            days: Number of days to export
            
        Returns:
            Dictionary with paths to exported files
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Get all attendance data
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            attendance_data = self.logger.export_attendance_data(start_date=start_date)
            
            # Get statistics
            statistics = self.logger.get_attendance_statistics(days=days)
            
            # Generate weekly reports
            weekly_reports = []
            for week in range(0, days, 7):
                week_start = (datetime.now() - timedelta(days=days-week)).strftime('%Y-%m-%d')
                weekly_report = self.generate_weekly_report(week_start)
                weekly_reports.append(weekly_report)
            
            # Export files
            exports = {}
            
            # CSV export
            if attendance_data:
                csv_path = self.export_to_csv(
                    attendance_data, 
                    f"thesis_attendance_data_{timestamp}.csv"
                )
                exports['attendance_csv'] = csv_path
            
            # Statistics JSON
            stats_path = self.export_to_json(
                statistics, 
                f"thesis_statistics_{timestamp}.json"
            )
            exports['statistics_json'] = stats_path
            
            # Weekly reports JSON
            weekly_path = self.export_to_json(
                weekly_reports, 
                f"thesis_weekly_reports_{timestamp}.json"
            )
            exports['weekly_reports_json'] = weekly_path
            
            # Create charts
            if statistics.get('daily_breakdown'):
                chart_path = self.create_attendance_chart(
                    statistics['daily_breakdown'], 
                    f"thesis_attendance_chart_{timestamp}.png"
                )
                exports['attendance_chart'] = chart_path
            
            # Generate summary report
            summary_report = {
                'export_info': {
                    'generated_at': datetime.now().isoformat(),
                    'period_days': days,
                    'start_date': start_date,
                    'end_date': datetime.now().strftime('%Y-%m-%d')
                },
                'data_summary': {
                    'total_records': len(attendance_data),
                    'unique_users': len(set(r['user_id'] for r in attendance_data)),
                    'successful_attempts': len([r for r in attendance_data if r['status'] == 'success']),
                    'total_attempts': len(attendance_data)
                },
                'statistics': statistics,
                'export_files': exports
            }
            
            summary_path = self.export_to_json(
                summary_report, 
                f"thesis_export_summary_{timestamp}.json"
            )
            exports['export_summary'] = summary_path
            
            self.system_logger.info(f"Thesis data export completed: {len(exports)} files generated")
            return exports
            
        except Exception as e:
            self.system_logger.error(f"Failed to generate thesis export: {e}")
            return {}
    
    def create_simple_html_report(self, report_data: Dict) -> str:
        """
        Create simple HTML report for viewing
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            Path to HTML file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"attendance_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Attendance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                    .summary {{ margin: 20px 0; }}
                    .stats {{ display: flex; gap: 20px; }}
                    .stat-box {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; min-width: 150px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .success {{ color: green; }}
                    .failed {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Attendance System Report</h1>
                    <p>Generated: {report_data.get('generated_at', datetime.now().isoformat())}</p>
                    <p>Report Type: {report_data.get('report_type', 'General Report')}</p>
                </div>
                
                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <div class="stats">
                        <div class="stat-box">
                            <h3>Total Attempts</h3>
                            <p>{report_data.get('summary', {}).get('total_attempts', 0)}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Successful</h3>
                            <p class="success">{report_data.get('summary', {}).get('successful_attendance', 0)}</p>
                        </div>
                        <div class="stat-box">
                            <h3>Success Rate</h3>
                            <p>{report_data.get('summary', {}).get('success_rate', 0)}%</p>
                        </div>
                        <div class="stat-box">
                            <h3>Unique Users</h3>
                            <p>{report_data.get('summary', {}).get('unique_users', 0)}</p>
                        </div>
                    </div>
                </div>
            """
            
            # Add attendance records table if available
            if 'attendance_records' in report_data and report_data['attendance_records']:
                html_content += """
                <div class="records">
                    <h2>Attendance Records</h2>
                    <table>
                        <tr>
                            <th>User ID</th>
                            <th>Name</th>
                            <th>Time</th>
                            <th>Status</th>
                            <th>Confidence</th>
                            <th>Anti-spoofing</th>
                        </tr>
                """
                
                for record in report_data['attendance_records']:
                    status_class = 'success' if record['status'] == 'success' else 'failed'
                    antispoofing = '✓' if record['antispoofing_passed'] else '✗'
                    html_content += f"""
                        <tr>
                            <td>{record['user_id']}</td>
                            <td>{record['user_name'] or 'N/A'}</td>
                            <td>{record['timestamp']}</td>
                            <td class="{status_class}">{record['status']}</td>
                            <td>{record['confidence_score']:.3f if record['confidence_score'] else 'N/A'}</td>
                            <td>{antispoofing}</td>
                        </tr>
                    """
                
                html_content += "</table></div>"
            
            html_content += """
            </body>
            </html>
            """
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.system_logger.info(f"HTML report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.system_logger.error(f"Failed to create HTML report: {e}")
            return ""


# Helper function to fix database connection method
def _monkey_patch_logger():
    """Add connection method to SimpleAttendanceLogger if not present"""
    def _get_connection(self):
        import sqlite3
        return sqlite3.connect(self.db_path)
    
    if not hasattr(SimpleAttendanceLogger, '_get_connection'):
        SimpleAttendanceLogger._get_connection = _get_connection

# Apply the patch
_monkey_patch_logger()

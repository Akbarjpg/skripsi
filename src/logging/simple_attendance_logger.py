"""
Simple Attendance Logger for Step 7 - Thesis Version
Basic logging functionality without complex security features
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Create simple logger without dependencies
logger = logging.getLogger(__name__)


class SimpleAttendanceLogger:
    """
    Simple attendance logging system for thesis demonstration
    Focuses on core attendance functionality without complex security
    """
    
    def __init__(self, db_path: str = "attendance.db"):
        """
        Initialize simple attendance logger
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logger
        
        # Ensure database exists
        self._init_simple_logging_tables()
    
    def _init_simple_logging_tables(self):
        """Initialize simple logging tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Simple attendance log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        user_name TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL,
                        confidence_score REAL,
                        antispoofing_passed BOOLEAN,
                        recognition_time REAL,
                        notes TEXT
                    )
                """)
                
                # Simple system events log
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT NOT NULL,
                        event_description TEXT,
                        user_id TEXT,
                        details TEXT
                    )
                """)
                
                # Daily attendance summary
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL UNIQUE,
                        total_attempts INTEGER DEFAULT 0,
                        successful_attendance INTEGER DEFAULT 0,
                        failed_antispoofing INTEGER DEFAULT 0,
                        failed_recognition INTEGER DEFAULT 0,
                        unique_users INTEGER DEFAULT 0,
                        avg_processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_log_date ON attendance_log(DATE(timestamp))")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_log_user ON attendance_log(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)")
                
                conn.commit()
                self.logger.info("Simple logging tables initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize logging tables: {e}")
            raise
    
    def log_attendance_attempt(self, user_id: str, user_name: str = None, 
                             status: str = "success", confidence_score: float = None,
                             antispoofing_passed: bool = None, recognition_time: float = None,
                             notes: str = None) -> bool:
        """
        Log an attendance attempt
        
        Args:
            user_id: User identifier
            user_name: User's name
            status: Status of attempt (success, failed, unknown)
            confidence_score: Recognition confidence score
            antispoofing_passed: Whether anti-spoofing passed
            recognition_time: Time taken for recognition
            notes: Additional notes
            
        Returns:
            True if logged successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO attendance_log 
                    (user_id, user_name, status, confidence_score, 
                     antispoofing_passed, recognition_time, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, user_name, status, confidence_score, 
                      antispoofing_passed, recognition_time, notes))
                
                conn.commit()
                
                # Log system event
                self.log_system_event(
                    "attendance_attempt", 
                    f"Attendance attempt by {user_name or user_id}: {status}",
                    user_id=user_id
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log attendance attempt: {e}")
            return False
    
    def log_system_event(self, event_type: str, description: str, 
                        user_id: str = None, details: Dict = None) -> bool:
        """
        Log a system event
        
        Args:
            event_type: Type of event (attendance_attempt, antispoofing_fail, etc.)
            description: Human readable description
            user_id: User ID if applicable
            details: Additional details as dictionary
            
        Returns:
            True if logged successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                details_str = json.dumps(details) if details else None
                
                cursor.execute("""
                    INSERT INTO system_events 
                    (event_type, event_description, user_id, details)
                    VALUES (?, ?, ?, ?)
                """, (event_type, description, user_id, details_str))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
            return False
    
    def get_daily_attendance_summary(self, date: str = None) -> Dict[str, Any]:
        """
        Get attendance summary for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with attendance summary
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get attendance stats for the date
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_attempts,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_attendance,
                        COUNT(CASE WHEN antispoofing_passed = 0 THEN 1 END) as failed_antispoofing,
                        COUNT(CASE WHEN status = 'failed' AND antispoofing_passed = 1 THEN 1 END) as failed_recognition,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(recognition_time) as avg_processing_time,
                        AVG(confidence_score) as avg_confidence
                    FROM attendance_log 
                    WHERE DATE(timestamp) = ?
                """, (date,))
                
                result = cursor.fetchone()
                
                if result:
                    summary = {
                        'date': date,
                        'total_attempts': result[0] or 0,
                        'successful_attendance': result[1] or 0,
                        'failed_antispoofing': result[2] or 0,
                        'failed_recognition': result[3] or 0,
                        'unique_users': result[4] or 0,
                        'avg_processing_time': round(result[5] or 0, 3),
                        'avg_confidence': round(result[6] or 0, 3),
                        'success_rate': round((result[1] or 0) / max(result[0] or 1, 1) * 100, 2)
                    }
                    
                    # Store in daily summary table
                    self._update_daily_summary(date, summary)
                    
                    return summary
                
                return {'date': date, 'total_attempts': 0}
                
        except Exception as e:
            self.logger.error(f"Failed to get daily summary: {e}")
            return {}
    
    def _update_daily_summary(self, date: str, summary: Dict):
        """Update daily summary table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_summary 
                    (date, total_attempts, successful_attendance, failed_antispoofing, 
                     failed_recognition, unique_users, avg_processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (date, summary['total_attempts'], summary['successful_attendance'],
                      summary['failed_antispoofing'], summary['failed_recognition'],
                      summary['unique_users'], summary['avg_processing_time']))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update daily summary: {e}")
    
    def get_user_attendance_history(self, user_id: str, days: int = 7) -> List[Dict]:
        """
        Get attendance history for a specific user
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of attendance records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=days)
                
                cursor.execute("""
                    SELECT * FROM attendance_log 
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (user_id, start_date))
                
                columns = [description[0] for description in cursor.description]
                records = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to get user attendance history: {e}")
            return []
    
    def get_recent_system_events(self, event_type: str = None, limit: int = 50) -> List[Dict]:
        """
        Get recent system events
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of system events
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if event_type:
                    cursor.execute("""
                        SELECT * FROM system_events 
                        WHERE event_type = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (event_type, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM system_events 
                        ORDER BY timestamp DESC LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                events = []
                
                for row in cursor.fetchall():
                    event = dict(zip(columns, row))
                    if event['details']:
                        try:
                            event['details'] = json.loads(event['details'])
                        except:
                            pass
                    events.append(event)
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to get system events: {e}")
            return []
    
    def get_attendance_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get overall attendance statistics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=days)
                
                # Overall stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_attempts,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(confidence_score) as avg_confidence,
                        AVG(recognition_time) as avg_processing_time
                    FROM attendance_log 
                    WHERE timestamp >= ?
                """, (start_date,))
                
                result = cursor.fetchone()
                
                # Daily breakdown
                cursor.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as attempts,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful
                    FROM attendance_log 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """, (start_date,))
                
                daily_stats = []
                for row in cursor.fetchall():
                    daily_stats.append({
                        'date': row[0],
                        'attempts': row[1],
                        'successful': row[2],
                        'success_rate': round(row[2] / max(row[1], 1) * 100, 2)
                    })
                
                # User performance
                cursor.execute("""
                    SELECT 
                        user_id,
                        user_name,
                        COUNT(*) as total_attempts,
                        COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                        AVG(confidence_score) as avg_confidence
                    FROM attendance_log 
                    WHERE timestamp >= ?
                    GROUP BY user_id, user_name
                    ORDER BY successful DESC
                """, (start_date,))
                
                user_stats = []
                for row in cursor.fetchall():
                    user_stats.append({
                        'user_id': row[0],
                        'user_name': row[1],
                        'total_attempts': row[2],
                        'successful': row[3],
                        'success_rate': round(row[3] / max(row[2], 1) * 100, 2),
                        'avg_confidence': round(row[4] or 0, 3)
                    })
                
                return {
                    'period_days': days,
                    'total_attempts': result[0] or 0,
                    'successful_attempts': result[1] or 0,
                    'unique_users': result[2] or 0,
                    'overall_success_rate': round((result[1] or 0) / max(result[0] or 1, 1) * 100, 2),
                    'avg_confidence': round(result[3] or 0, 3),
                    'avg_processing_time': round(result[4] or 0, 3),
                    'daily_breakdown': daily_stats,
                    'user_performance': user_stats
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get attendance statistics: {e}")
            return {}
    
    def export_attendance_data(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Export attendance data for thesis analysis
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of attendance records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM attendance_log"
                params = []
                
                if start_date or end_date:
                    query += " WHERE"
                    conditions = []
                    
                    if start_date:
                        conditions.append(" DATE(timestamp) >= ?")
                        params.append(start_date)
                    
                    if end_date:
                        conditions.append(" DATE(timestamp) <= ?")
                        params.append(end_date)
                    
                    query += " AND".join(conditions)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                
                columns = [description[0] for description in cursor.description]
                records = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to export attendance data: {e}")
            return []
    
    def cleanup_old_logs(self, days_to_keep: int = 90) -> bool:
        """
        Clean up old log entries to maintain database size
        
        Args:
            days_to_keep: Number of days of logs to keep
            
        Returns:
            True if cleanup successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean old system events
                cursor.execute("""
                    DELETE FROM system_events 
                    WHERE timestamp < ? AND event_type != 'attendance_attempt'
                """, (cutoff_date,))
                
                events_deleted = cursor.rowcount
                
                # Clean old daily summaries (keep them longer)
                old_cutoff = datetime.now() - timedelta(days=days_to_keep * 2)
                cursor.execute("""
                    DELETE FROM daily_summary 
                    WHERE created_at < ?
                """, (old_cutoff,))
                
                summaries_deleted = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Cleaned up {events_deleted} old events and {summaries_deleted} old summaries")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return False

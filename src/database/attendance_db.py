"""
Attendance Database Management for Step 4
Handles user registration, face embeddings, and attendance records
"""

import sqlite3
import json
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import pickle
import base64

from ..utils.logger import get_logger


class AttendanceDatabase:
    """
    Database manager for attendance system with face recognition
    """
    
    def __init__(self, db_path: str = "attendance.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = get_logger(__name__)
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        email TEXT,
                        role TEXT DEFAULT 'employee',
                        department TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Face embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        embedding_vector TEXT NOT NULL,
                        embedding_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        quality_score REAL,
                        num_images INTEGER DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
                    )
                """)
                
                # Attendance records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        confidence_score REAL NOT NULL,
                        antispoofing_score REAL,
                        recognition_time REAL,
                        device_info TEXT,
                        session_id TEXT,
                        attendance_type TEXT DEFAULT 'check_in',
                        location TEXT,
                        notes TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                """)
                
                # Antispoofing attempts table (for security monitoring)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS antispoofing_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        result TEXT NOT NULL,
                        confidence_score REAL,
                        challenge_results TEXT,
                        device_info TEXT,
                        ip_address TEXT,
                        failure_reason TEXT
                    )
                """)
                
                # System logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        level TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON face_embeddings(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_user_id ON attendance_records(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance_records(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_antispoofing_session ON antispoofing_attempts(session_id)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _serialize_embedding(self, embedding: np.ndarray) -> str:
        """
        Serialize numpy array to string for database storage
        
        Args:
            embedding: Face embedding as numpy array
            
        Returns:
            Serialized embedding as base64 string
        """
        return base64.b64encode(pickle.dumps(embedding)).decode('utf-8')
    
    def _deserialize_embedding(self, embedding_str: str) -> np.ndarray:
        """
        Deserialize string to numpy array
        
        Args:
            embedding_str: Serialized embedding string
            
        Returns:
            Face embedding as numpy array
        """
        return pickle.loads(base64.b64decode(embedding_str.encode('utf-8')))
    
    def register_user(self, user_id: str, name: str, email: str = None, 
                     role: str = "employee", department: str = None) -> bool:
        """
        Register a new user
        
        Args:
            user_id: Unique user identifier
            name: User's full name
            email: User's email address
            role: User's role (employee, admin, etc.)
            department: User's department
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO users (user_id, name, email, role, department)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, name, email, role, department))
                
                conn.commit()
                self.logger.info(f"User {user_id} registered successfully")
                return True
                
        except sqlite3.IntegrityError:
            self.logger.warning(f"User {user_id} already exists")
            return False
        except Exception as e:
            self.logger.error(f"Failed to register user {user_id}: {e}")
            return False
    
    def store_face_embedding(self, user_id: str, embedding: np.ndarray,
                           quality_score: float = None, num_images: int = 1,
                           metadata: Dict = None) -> bool:
        """
        Store face embedding for a user
        
        Args:
            user_id: User identifier
            embedding: Face embedding vector
            quality_score: Quality score of the embedding
            num_images: Number of images used to create embedding
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute("SELECT id FROM users WHERE user_id = ?", (user_id,))
                if not cursor.fetchone():
                    self.logger.error(f"User {user_id} not found")
                    return False
                
                # Serialize embedding and metadata
                embedding_str = self._serialize_embedding(embedding)
                metadata_str = json.dumps(metadata) if metadata else None
                
                # Delete existing embedding for this user (one embedding per user)
                cursor.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
                
                # Insert new embedding
                cursor.execute("""
                    INSERT INTO face_embeddings 
                    (user_id, embedding_vector, embedding_metadata, quality_score, num_images)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, embedding_str, metadata_str, quality_score, num_images))
                
                conn.commit()
                self.logger.info(f"Face embedding stored for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store embedding for {user_id}: {e}")
            return False
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get face embedding for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            Face embedding or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT embedding_vector FROM face_embeddings 
                    WHERE user_id = ?
                """, (user_id,))
                
                result = cursor.fetchone()
                if result:
                    return self._deserialize_embedding(result[0])
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get embedding for {user_id}: {e}")
            return None
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all face embeddings from database
        
        Returns:
            Dictionary mapping user_id to face embedding
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT fe.user_id, fe.embedding_vector 
                    FROM face_embeddings fe
                    JOIN users u ON fe.user_id = u.user_id
                    WHERE u.is_active = 1
                """)
                
                embeddings = {}
                for user_id, embedding_str in cursor.fetchall():
                    embeddings[user_id] = self._deserialize_embedding(embedding_str)
                
                self.logger.info(f"Loaded {len(embeddings)} embeddings from database")
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Failed to get all embeddings: {e}")
            return {}
    
    def record_attendance(self, user_id: str, confidence_score: float,
                         antispoofing_score: float = None, recognition_time: float = None,
                         device_info: str = None, session_id: str = None,
                         attendance_type: str = "check_in", location: str = None,
                         notes: str = None) -> bool:
        """
        Record attendance for a user
        
        Args:
            user_id: User identifier
            confidence_score: Face recognition confidence
            antispoofing_score: Anti-spoofing confidence
            recognition_time: Time taken for recognition
            device_info: Information about the device used
            session_id: Session identifier
            attendance_type: Type of attendance (check_in, check_out)
            location: Location where attendance was recorded
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO attendance_records 
                    (user_id, confidence_score, antispoofing_score, recognition_time,
                     device_info, session_id, attendance_type, location, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, confidence_score, antispoofing_score, recognition_time,
                      device_info, session_id, attendance_type, location, notes))
                
                conn.commit()
                self.logger.info(f"Attendance recorded for {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to record attendance for {user_id}: {e}")
            return False
    
    def get_user_attendance(self, user_id: str, days: int = 30) -> List[Dict]:
        """
        Get attendance records for a user
        
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
                    SELECT ar.*, u.name 
                    FROM attendance_records ar
                    JOIN users u ON ar.user_id = u.user_id
                    WHERE ar.user_id = ? AND ar.timestamp >= ?
                    ORDER BY ar.timestamp DESC
                """, (user_id, start_date))
                
                columns = [description[0] for description in cursor.description]
                records = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to get attendance for {user_id}: {e}")
            return []
    
    def get_daily_attendance(self, date: str = None) -> List[Dict]:
        """
        Get all attendance records for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            List of attendance records for the date
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT ar.*, u.name, u.department
                    FROM attendance_records ar
                    JOIN users u ON ar.user_id = u.user_id
                    WHERE DATE(ar.timestamp) = ?
                    ORDER BY ar.timestamp DESC
                """, (date,))
                
                columns = [description[0] for description in cursor.description]
                records = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to get daily attendance for {date}: {e}")
            return []
    
    def log_antispoofing_attempt(self, session_id: str, result: str,
                               confidence_score: float = None,
                               challenge_results: Dict = None,
                               device_info: str = None, ip_address: str = None,
                               failure_reason: str = None) -> bool:
        """
        Log an anti-spoofing attempt for security monitoring
        
        Args:
            session_id: Session identifier
            result: Result of anti-spoofing check (passed/failed)
            confidence_score: Confidence score
            challenge_results: Results of individual challenges
            device_info: Device information
            ip_address: IP address of the request
            failure_reason: Reason for failure if applicable
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                challenge_results_str = json.dumps(challenge_results) if challenge_results else None
                
                cursor.execute("""
                    INSERT INTO antispoofing_attempts 
                    (session_id, result, confidence_score, challenge_results,
                     device_info, ip_address, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, result, confidence_score, challenge_results_str,
                      device_info, ip_address, failure_reason))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to log antispoofing attempt: {e}")
            return False
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get complete user information
        
        Args:
            user_id: User identifier
            
        Returns:
            User information dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT u.*, 
                           fe.quality_score,
                           fe.num_images,
                           fe.created_at as embedding_created
                    FROM users u
                    LEFT JOIN face_embeddings fe ON u.user_id = fe.user_id
                    WHERE u.user_id = ?
                """, (user_id,))
                
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    user_info = dict(zip(columns, result))
                    
                    # Get recent attendance
                    recent_attendance = self.get_user_attendance(user_id, days=7)
                    user_info['recent_attendance'] = recent_attendance
                    
                    return user_info
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get user info for {user_id}: {e}")
            return None
    
    def get_all_users(self) -> List[Dict]:
        """
        Get all registered users
        
        Returns:
            List of user information dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT u.*, 
                           fe.quality_score,
                           fe.num_images,
                           COUNT(ar.id) as total_attendance
                    FROM users u
                    LEFT JOIN face_embeddings fe ON u.user_id = fe.user_id
                    LEFT JOIN attendance_records ar ON u.user_id = ar.user_id
                    GROUP BY u.user_id
                    ORDER BY u.name
                """)
                
                columns = [description[0] for description in cursor.description]
                users = []
                
                for row in cursor.fetchall():
                    user = dict(zip(columns, row))
                    users.append(user)
                
                return users
                
        except Exception as e:
            self.logger.error(f"Failed to get all users: {e}")
            return []
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user and all associated data
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete user (cascade will handle embeddings)
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"User {user_id} deleted successfully")
                    return True
                else:
                    self.logger.warning(f"User {user_id} not found")
                    return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count users
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
                stats['active_users'] = cursor.fetchone()[0]
                
                # Count embeddings
                cursor.execute("SELECT COUNT(*) FROM face_embeddings")
                stats['total_embeddings'] = cursor.fetchone()[0]
                
                # Count attendance records
                cursor.execute("SELECT COUNT(*) FROM attendance_records")
                stats['total_attendance_records'] = cursor.fetchone()[0]
                
                # Today's attendance
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute("""
                    SELECT COUNT(*) FROM attendance_records 
                    WHERE DATE(timestamp) = ?
                """, (today,))
                stats['todays_attendance'] = cursor.fetchone()[0]
                
                # Antispoofing attempts
                cursor.execute("SELECT COUNT(*) FROM antispoofing_attempts")
                stats['total_antispoofing_attempts'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False

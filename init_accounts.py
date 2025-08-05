#!/usr/bin/env python3
"""
Database initialization script to create default user accounts
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash

def init_database_with_accounts():
    """Initialize database and create default accounts"""
    
    db_path = 'attendance.db'
    
    print("🗄️ Initializing database with default accounts...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create users table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create attendance records table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                check_type TEXT DEFAULT 'in',
                confidence_score REAL,
                liveness_score REAL,
                security_level TEXT,
                methods_passed INTEGER,
                processing_time REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default accounts
        accounts = [
            ('admin', 'admin', 'Administrator', 'admin@system.com', 'admin'),
            ('user', 'user123', 'Regular User', 'user@system.com', 'user'),
            ('demo', 'demo', 'Demo User', 'demo@system.com', 'user'),
            ('guest', '123456', 'Guest User', 'guest@system.com', 'user')
        ]
        
        for username, password, full_name, email, role in accounts:
            try:
                # Check if account exists
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
                if cursor.fetchone()[0] == 0:
                    # Create account
                    password_hash = generate_password_hash(password)
                    cursor.execute('''
                        INSERT INTO users (username, full_name, email, password_hash, role)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (username, full_name, email, password_hash, role))
                    print(f"✅ Created account: {username} / {password} ({full_name})")
                else:
                    print(f"ℹ️ Account already exists: {username}")
            except Exception as e:
                print(f"❌ Error creating account {username}: {e}")
        
        conn.commit()
    
    print("\n🎉 Database initialization complete!")
    print("\n📋 Available Accounts:")
    print("┌──────────┬──────────┬─────────────────┬──────────────────┐")
    print("│ Username │ Password │ Full Name       │ Role             │")
    print("├──────────┼──────────┼─────────────────┼──────────────────┤")
    print("│ admin    │ admin    │ Administrator   │ admin            │")
    print("│ user     │ user123  │ Regular User    │ user             │")
    print("│ demo     │ demo     │ Demo User       │ user             │")
    print("│ guest    │ 123456   │ Guest User      │ user             │")
    print("└──────────┴──────────┴─────────────────┴──────────────────┘")
    print("\n🚀 You can now login with any of these accounts!")

if __name__ == "__main__":
    init_database_with_accounts()

#!/usr/bin/env python3
"""
Fix database schema for local image saving
"""

import sqlite3
from pathlib import Path

def fix_database_schema():
    """Fix the face_data table to allow NULL face_encoding"""
    print("🔧 Fixing Database Schema for Local Image Saving\n")
    
    db_path = Path("attendance.db")
    
    if not db_path.exists():
        print("⚠️ Database doesn't exist - will be created properly on first use")
        return True
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            # Check current schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='face_data'")
            result = cursor.fetchone()
            
            if result:
                current_schema = result[0]
                print(f"📋 Current schema:")
                print(f"   {current_schema}")
                
                # Check if face_encoding has NOT NULL constraint
                if "face_encoding TEXT NOT NULL" in current_schema:
                    print(f"\n🔧 Found NOT NULL constraint on face_encoding - fixing...")
                    
                    # Create backup table with correct schema
                    cursor.execute("""
                        CREATE TABLE face_data_backup (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            face_position TEXT NOT NULL,
                            face_encoding TEXT,
                            image_path TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Copy existing data
                    cursor.execute("""
                        INSERT INTO face_data_backup (id, user_id, face_position, face_encoding, created_at)
                        SELECT id, user_id, face_position, face_encoding, created_at FROM face_data
                    """)
                    
                    # Drop old table
                    cursor.execute("DROP TABLE face_data")
                    
                    # Rename backup to original
                    cursor.execute("ALTER TABLE face_data_backup RENAME TO face_data")
                    
                    print(f"✅ Schema fixed - face_encoding is now nullable")
                    
                elif "face_encoding TEXT," in current_schema or "face_encoding TEXT" in current_schema:
                    print(f"✅ Schema already allows NULL face_encoding")
                    
                    # Check if image_path column exists
                    cursor.execute("PRAGMA table_info(face_data)")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    if 'image_path' not in columns:
                        print(f"📝 Adding image_path column...")
                        cursor.execute("ALTER TABLE face_data ADD COLUMN image_path TEXT")
                        print(f"✅ image_path column added")
                    else:
                        print(f"✅ image_path column already exists")
                
                else:
                    print(f"⚠️ Unexpected schema format")
                    return False
                    
            else:
                print(f"📝 face_data table doesn't exist - will be created with correct schema")
                
                # Create table with correct schema
                cursor.execute("""
                    CREATE TABLE face_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        face_position TEXT NOT NULL,
                        face_encoding TEXT,
                        image_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print(f"✅ Created face_data table with correct schema")
            
            # Verify final schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='face_data'")
            result = cursor.fetchone()
            if result:
                print(f"\n📋 Final schema:")
                print(f"   {result[0]}")
            
            conn.commit()
            
        print(f"\n🎉 Database schema fix completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_insert_with_image_path():
    """Test inserting a record with only image_path (no face_encoding)"""
    print(f"\n🧪 Testing Insert with Image Path Only...")
    
    try:
        with sqlite3.connect("attendance.db") as conn:
            cursor = conn.cursor()
            
            # Test insert
            test_user_id = 998
            test_position = 'test'
            test_image_path = 'static/faces/test_image.jpg'
            
            cursor.execute("""
                INSERT INTO face_data (user_id, face_position, image_path)
                VALUES (?, ?, ?)
            """, (test_user_id, test_position, test_image_path))
            
            # Verify insert
            cursor.execute("""
                SELECT id, user_id, face_position, face_encoding, image_path
                FROM face_data 
                WHERE user_id = ?
            """, (test_user_id,))
            
            result = cursor.fetchone()
            if result:
                print(f"✅ Test record inserted successfully:")
                print(f"   ID: {result[0]}")
                print(f"   User ID: {result[1]}")
                print(f"   Position: {result[2]}")
                print(f"   Face Encoding: {result[3]} (should be None)")
                print(f"   Image Path: {result[4]}")
                
                # Clean up test record
                cursor.execute("DELETE FROM face_data WHERE user_id = ?", (test_user_id,))
                conn.commit()
                print(f"🗑️ Test record cleaned up")
                
                return True
            else:
                print(f"❌ Test record not found")
                return False
                
    except Exception as e:
        print(f"❌ Insert test failed: {e}")
        return False

def main():
    print("🔧 Database Schema Fix for Local Image Saving")
    print("=" * 50)
    
    # Fix schema
    schema_ok = fix_database_schema()
    
    if schema_ok:
        # Test insert
        insert_ok = test_insert_with_image_path()
        
        if insert_ok:
            print(f"\n🎉 Database is ready for local image saving!")
            print(f"\n✨ What changed:")
            print(f"   • face_encoding column is now nullable")
            print(f"   • image_path column is available")
            print(f"   • Can store records with only image paths")
            
            print(f"\n🔄 Next step:")
            print(f"   Run: python test_direct_image_saving.py")
        else:
            print(f"\n❌ Insert test failed")
    else:
        print(f"\n❌ Schema fix failed")

if __name__ == "__main__":
    main()

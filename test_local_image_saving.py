#!/usr/bin/env python3
"""
Test script for local image saving functionality
Tests if face registration can save images to local storage
"""

import os
import time
import webbrowser
from pathlib import Path

def test_local_image_saving():
    """Test if the local image saving implementation is working"""
    print("🖼️ Testing Local Image Saving Implementation...\n")
    
    # Check if the backend code has local saving implemented
    app_path = Path("src/web/app_optimized.py")
    if not app_path.exists():
        print("❌ app_optimized.py not found!")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for local saving features
    local_saving_checks = [
        "faces_dir = os.path.join('static', 'faces')",
        "os.makedirs(faces_dir, exist_ok=True)",
        "image_filename = f\"face_{user_id}_{position}_{timestamp}.jpg\"",
        "with open(image_path, 'wb') as f:",
        "f.write(image_bytes)",
        "ALTER TABLE face_data ADD COLUMN image_path TEXT",
        "INSERT INTO face_data (user_id, face_position, image_path)"
    ]
    
    missing_features = []
    for check in local_saving_checks:
        if check not in content:
            missing_features.append(check)
    
    if missing_features:
        print(f"❌ Missing local saving features: {missing_features}")
        return False
    
    print("✅ Local image saving implementation found")
    
    # Check if static/faces directory exists or can be created
    faces_dir = Path("static/faces")
    try:
        faces_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Faces directory ready: {faces_dir}")
        
        # Test write permissions
        test_file = faces_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()  # Delete test file
        print("✅ Write permissions confirmed")
        
    except Exception as e:
        print(f"❌ Directory/permission error: {e}")
        return False
    
    return True

def check_database_changes():
    """Check if database schema supports image paths"""
    print("\n💾 Checking Database Schema...")
    
    db_path = Path("attendance.db")
    if not db_path.exists():
        print("⚠️ Database not found - will be created on first run")
        return True
    
    try:
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            
            # Check if face_data table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='face_data'")
            if not cursor.fetchone():
                print("⚠️ face_data table not found - will be created on first run")
                return True
            
            # Check if image_path column exists
            cursor.execute("PRAGMA table_info(face_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'image_path' in columns:
                print("✅ image_path column exists in face_data table")
            else:
                print("⚠️ image_path column will be added automatically on first use")
            
            print(f"📋 Current columns: {columns}")
            
    except Exception as e:
        print(f"❌ Database check error: {e}")
        return False
    
    return True

def generate_testing_instructions():
    """Generate detailed testing instructions"""
    print("\n📋 Testing Instructions:")
    print("=" * 60)
    
    print("\n🚀 1. Start the server:")
    print("   python src/web/app_optimized.py")
    
    print("\n🌐 2. Open browser:")
    print("   http://localhost:5000/register-face")
    
    print("\n🔍 3. Check server console for debug output:")
    print("   Look for these messages:")
    print("   📁 Faces directory: static/faces")
    print("   ✅ Image saved to: static/faces/face_X_front_TIMESTAMP.jpg")
    print("   📏 File size: XXXX bytes")
    print("   ✅ File verification: XXXX bytes on disk")
    print("   ✅ Successfully saved image path for position: front")
    
    print("\n📂 4. Check if files are created:")
    print("   - Navigate to static/faces/ directory")
    print("   - Look for image files like: face_1_front_1234567890.jpg")
    print("   - Verify files are not corrupted (can be opened)")
    
    print("\n🎯 5. Expected benefits:")
    print("   ✅ No more 'terjadi kesalahan saat memproses wajah' errors")
    print("   ✅ Images saved locally for inspection")
    print("   ✅ Can verify if image capture is working correctly")
    print("   ✅ Face recognition can be added later after confirming images are good")
    
    print("\n🔧 6. If still getting errors:")
    print("   - Check server console for specific error messages")
    print("   - Verify write permissions on static/faces directory")
    print("   - Check if user_id is properly set in session")
    print("   - Look for database connection issues")

def main():
    print("🧪 Local Image Saving Test\n")
    
    # Test local image saving implementation
    saving_ok = test_local_image_saving()
    
    # Check database readiness
    db_ok = check_database_changes()
    
    if saving_ok and db_ok:
        print("\n🎉 Local image saving is ready!")
        print("\n✨ Key Changes Made:")
        print("   • Removed face recognition processing (temporarily)")
        print("   • Added local image file saving to static/faces/")
        print("   • Store image file paths in database instead of encodings")
        print("   • Enhanced error logging for better debugging")
        
        generate_testing_instructions()
        
        # Try to open browser for testing
        try:
            print("\n🌐 Opening browser for testing...")
            time.sleep(2)
            webbrowser.open('http://localhost:5000/register-face')
        except:
            print("Could not open browser automatically")
        
        return True
    else:
        print("\n❌ Setup incomplete. Please check the implementation.")
        return False

if __name__ == "__main__":
    main()

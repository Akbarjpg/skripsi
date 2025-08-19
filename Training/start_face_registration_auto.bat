@echo off
echo ===================================
echo   Starting Automatic Face Registration
echo ===================================
echo.
echo 1. Starting Flask server...
cd /d "d:\Codingan\skripsi\dari nol"
python src/web/app_optimized.py
echo.
echo 2. Server should now be running at:
echo    http://localhost:5000
echo.
echo 3. To test face registration:
echo    http://localhost:5000/register_face
echo.
echo Features to test:
echo - Automatic face detection
echo - 3-2-1 countdown
echo - Auto capture
echo - Position progression
echo.
pause

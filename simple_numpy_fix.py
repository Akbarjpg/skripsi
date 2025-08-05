"""
Simple solution for numpy conflicts
"""

def create_virtual_environment_guide():
    """Create step-by-step guide for virtual environment setup"""
    guide = """
NUMPY CONFLICT RESOLUTION - VIRTUAL ENVIRONMENT SOLUTION
========================================================

The system has numpy version conflicts between:
- mediapipe requires numpy<2
- tensorflow requires numpy<2.2.0,>=1.26.0  
- Current numpy is 2.3.1

RECOMMENDED SOLUTION - Use Virtual Environment:

1. CREATE VIRTUAL ENVIRONMENT:
   python -m venv anti_spoof_env

2. ACTIVATE VIRTUAL ENVIRONMENT:
   anti_spoof_env\\Scripts\\activate

3. INSTALL COMPATIBLE PACKAGES:
   pip install numpy==1.26.4
   pip install opencv-python==4.8.0
   pip install mediapipe==0.10.0
   pip install flask==2.3.0
   pip install flask-socketio==5.3.0
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

4. TEST THE SYSTEM:
   python test_fix.py

5. RUN THE SYSTEM:
   python launch.py --mode web

ALTERNATIVE QUICK FIX (if virtual env is not preferred):
======================================================

1. DOWNGRADE NUMPY:
   pip install numpy==1.26.4 --force-reinstall

2. RESTART PYTHON INTERPRETER

3. TEST: python -c "import numpy; print(numpy.__version__)"

4. If working, run: python launch.py --mode web

TROUBLESHOOTING:
===============
- If pip conflicts persist: pip install --force-reinstall --no-deps numpy==1.26.4
- If mediapipe fails: pip install mediapipe==0.10.0 --force-reinstall
- Always restart terminal after numpy changes
"""
    
    with open("NUMPY_FIX_GUIDE.txt", "w", encoding="utf-8") as f:
        f.write(guide)
    
    print("Created NUMPY_FIX_GUIDE.txt")
    print(guide)

if __name__ == "__main__":
    create_virtual_environment_guide()

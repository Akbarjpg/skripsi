# Step 3 Enhanced Challenge System - Installation Guide

## Quick Installation on New Device

### Method 1: Automatic Setup (Recommended)
```bash
# 1. Download/copy these files to the new device:
# - setup_step3.py
# - requirements_step3_minimal.txt 
# - All src/ folder contents

# 2. Run the setup script:
python setup_step3.py

# 3. Follow the prompts to install everything automatically
```

### Method 2: Manual Installation

#### Prerequisites
- Python 3.8 or higher
- Webcam/camera access
- Audio output (speakers/headphones)

#### Step-by-Step Installation

1. **Create virtual environment (recommended):**
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
```

2. **Install essential packages:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install from requirements file
pip install -r requirements_step3_minimal.txt
```

3. **Or install manually:**
```bash
# Core AI/Computer Vision
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python>=4.8.0
pip install mediapipe>=0.10.0
pip install numpy>=1.24.0

# Audio (essential for Step 3)
pip install pygame>=2.5.0
pip install pyttsx3>=2.90

# Web framework
pip install flask>=2.3.0
pip install flask-socketio>=5.3.0

# Image processing
pip install Pillow>=10.0.0
pip install scikit-learn>=1.3.0
```

### Files to Copy to New Device

**Essential Files:**
```
📁 src/
├── 📁 challenge/
│   ├── __init__.py
│   ├── challenge_response.py
│   ├── distance_challenge.py
│   └── audio_feedback.py
├── 📁 detection/
│   ├── __init__.py
│   └── landmark_detection.py
├── 📁 web/
│   ├── __init__.py
│   ├── app.py
│   └── 📁 templates/
│       └── attendance_sequential.html
└── 📁 utils/
    ├── __init__.py
    ├── logger.py
    └── config.py

📄 requirements_step3_minimal.txt
📄 setup_step3.py
📄 test_step3_enhanced_challenges.py
📄 attendance.db (if exists)
```

### Verification

**Test imports:**
```python
python -c "
import cv2, mediapipe, torch, numpy
import pygame, pyttsx3, flask
print('✅ All packages imported successfully!')
"
```

**Test camera:**
```python
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
print('✅ Camera working!' if ret else '❌ Camera not working')
"
```

**Test audio:**
```python
python -c "
import pygame; pygame.mixer.init()
import pyttsx3; engine = pyttsx3.init(); engine.stop()
print('✅ Audio system working!')
"
```

### Running the System

**Test Step 3 features:**
```bash
python test_step3_enhanced_challenges.py
```

**Run web interface:**
```bash
python -m src.web.app
# Then open: http://localhost:5000
```

### Troubleshooting

**Common Issues:**

1. **NumPy version conflicts:**
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

2. **MediaPipe issues:**
```bash
pip uninstall mediapipe
pip install mediapipe==0.10.0
```

3. **Audio not working:**
```bash
# Try alternative pygame installation
pip uninstall pygame
pip install pygame==2.5.0

# For pyttsx3 issues on Linux:
sudo apt-get install espeak espeak-data libespeak-dev
```

4. **Camera access issues:**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other apps are using the camera

5. **PyTorch issues:**
```bash
# For CPU-only installation
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- Webcam
- Audio output

**Recommended:**
- Python 3.9+
- 8GB RAM
- Good quality webcam
- Speakers/headphones for audio feedback

### Features Available After Installation

✅ **Step 3 Enhanced Challenge System:**
- Distance challenges (move closer/farther)
- Audio feedback with success/failure sounds
- Text-to-speech voice instructions
- Enhanced retry logic
- Security measures and time limits
- Real-time feedback
- Web interface with interactive controls

### Next Steps

After successful installation:
1. Test the challenge system
2. Customize audio settings if needed
3. Proceed to Step 4 implementation
4. Or integrate with existing systems

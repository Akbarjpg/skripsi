import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'web'))

from app_optimized import EnhancedFrameProcessor

proc = EnhancedFrameProcessor()

# Test poor lighting (very dark frame)
poor_frame = np.full((480, 640, 3), 36, dtype=np.uint8)
result = proc.assess_frame_quality(poor_frame)

brightness = np.mean(cv2.cvtColor(poor_frame, cv2.COLOR_BGR2GRAY))
lighting_score = result['lighting_score']

print(f"Poor lighting test:")
print(f"  Brightness: {brightness:.1f}")
print(f"  Lighting Score: {lighting_score:.3f}")
print(f"  Expected: < 0.5")
print(f"  Result: {'PASS' if lighting_score < 0.5 else 'FAIL'}")

if lighting_score < 0.5:
    print("✅ Poor lighting detection is working!")
else:
    print("❌ Poor lighting detection needs more work")

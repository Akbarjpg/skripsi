#!/usr/bin/env python3
"""
Quick Background Detection Test
"""

import os
import sys
import numpy as np
import cv2

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'web'))

def test_background_detection():
    """Test background detection with debug info"""
    print("🔍 Testing Background Detection...")
    
    try:
        from app_optimized import EnhancedFrameProcessor
        processor = EnhancedFrameProcessor()
        
        # Create screen background with obvious patterns
        screen_bg = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create rectangular pattern
        for i in range(0, 480, 20):
            screen_bg[i:i+2, :] = [120, 120, 120]
        for j in range(0, 640, 20):
            screen_bg[:, j:j+2] = [120, 120, 120]
        screen_bg[screen_bg == 0] = [80, 80, 80]
        
        # Test background analysis
        result = processor.detect_background_context(screen_bg)
        
        print(f"Screen Likelihood: {result['screen_likelihood']:.3f}")
        print(f"Photo Likelihood: {result['photo_likelihood']:.3f}")
        print(f"Natural Likelihood: {result['natural_likelihood']:.3f}")
        print(f"Edge Density: {result.get('edge_density', 'N/A'):.3f}")
        
        # Debug edge analysis
        gray = cv2.cvtColor(screen_bg, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_count / total_pixels
        
        print(f"\nDEBUG INFO:")
        print(f"Edge count: {edge_count}")
        print(f"Total pixels: {total_pixels}")
        print(f"Edge density: {edge_density:.4f}")
        
        # Test line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        if lines is not None:
            print(f"Lines detected: {len(lines)}")
            h_lines = v_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                if abs(angle) < 15 or abs(angle) > 165:
                    h_lines += 1
                elif 75 < abs(angle) < 105:
                    v_lines += 1
            print(f"Horizontal lines: {h_lines}")
            print(f"Vertical lines: {v_lines}")
            screen_score = min(1.0, (h_lines + v_lines) / 20.0)
            print(f"Calculated screen score: {screen_score:.3f}")
        else:
            print("No lines detected")
        
        print(f"\n✅ Background detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_background_detection()

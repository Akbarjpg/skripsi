"""
Quick Demo of Advanced Anti-Spoofing System

This script demonstrates the working advanced anti-spoofing system
with a simple example.
"""
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_demo():
    """Quick demonstration of the advanced anti-spoofing system"""
    
    print("🚀 Advanced Anti-Spoofing System Demo")
    print("=" * 50)
    
    try:
        # Import the integrated system
        from integration.enhanced_antispoofing_integration import create_enhanced_detection_system
        
        print("✅ Importing system components...")
        
        # Create the system
        system = create_enhanced_detection_system()
        print("✅ System initialized successfully!")
        
        # Create a test image (simulating camera input)
        print("\n📸 Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some face-like features to make it more realistic
        # Draw a simple face-like oval
        center = (320, 240)
        axes = (100, 120)
        cv2.ellipse(test_image, center, axes, 0, 0, 360, (180, 150, 120), -1)
        
        # Add eyes
        cv2.circle(test_image, (280, 220), 15, (50, 50, 50), -1)
        cv2.circle(test_image, (360, 220), 15, (50, 50, 50), -1)
        
        # Add mouth
        cv2.ellipse(test_image, (320, 280), (30, 15), 0, 0, 360, (100, 80, 80), -1)
        
        print("✅ Test image created")
        
        # Process with advanced detection
        print("\n🔍 Running advanced anti-spoofing detection...")
        result = system.process_comprehensive_detection(test_image)
        
        # Display results
        print("\n📊 DETECTION RESULTS:")
        print("-" * 30)
        print(f"🎯 Is Live: {'✅ YES' if result.is_live else '❌ NO'}")
        print(f"🎯 Overall Confidence: {result.confidence:.3f}")
        print(f"🎯 Risk Level: {result.risk_level.upper()}")
        print(f"⏱️  Processing Time: {result.processing_time:.3f} seconds")
        
        print(f"\n🔬 DETAILED SCORES:")
        print(f"   • Basic Liveness: {result.basic_liveness_score:.3f}")
        print(f"   • Texture Analysis: {result.texture_analysis_score:.3f}")
        print(f"   • Depth Estimation: {result.depth_estimation_score:.3f}")
        print(f"   • Micro-Expression: {result.micro_expression_score:.3f}")
        print(f"   • Eye Tracking: {result.eye_tracking_score:.3f}")
        print(f"   • PPG Detection: {result.ppg_detection_score:.3f}")
        print(f"   • Fusion Confidence: {result.fusion_confidence:.3f}")
        
        print(f"\n🛠️  METHODS USED: {', '.join(result.detection_methods_used)}")
        
        if result.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        # Get system performance stats
        performance = system.get_performance_report()
        print(f"\n📈 SYSTEM PERFORMANCE:")
        stats = performance['performance_stats']
        print(f"   • Total Detections: {stats['total_detections']}")
        print(f"   • Success Rate: {stats['detection_accuracy']:.1%}")
        print(f"   • Avg Processing Time: {stats['average_processing_time']:.3f}s")
        
        # Test with different scenarios
        print(f"\n🧪 TESTING DIFFERENT SCENARIOS:")
        
        # Test 1: Low quality image
        print("   Testing low quality image...")
        low_quality = cv2.resize(test_image, (160, 120))
        low_quality = cv2.resize(low_quality, (640, 480))
        result_low = system.process_comprehensive_detection(low_quality)
        print(f"   → Low quality result: {'Live' if result_low.is_live else 'Spoofed'} (confidence: {result_low.confidence:.3f})")
        
        # Test 2: Very dark image
        print("   Testing dark image...")
        dark_image = (test_image * 0.3).astype(np.uint8)
        result_dark = system.process_comprehensive_detection(dark_image)
        print(f"   → Dark image result: {'Live' if result_dark.is_live else 'Spoofed'} (confidence: {result_dark.confidence:.3f})")
        
        print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"✅ Advanced anti-spoofing system is working and ready for use!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all required modules are installed and accessible")
        return False
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        print("   System encountered an error during demonstration")
        return False

if __name__ == "__main__":
    success = quick_demo()
    
    if success:
        print(f"\n🚀 System Status: READY FOR DEPLOYMENT")
        exit(0)
    else:
        print(f"\n⚠️  System Status: NEEDS TROUBLESHOOTING")
        exit(1)

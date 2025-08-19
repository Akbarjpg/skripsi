#!/usr/bin/env python3
"""
Start optimized server dengan fixes applied
"""

import sys
import os

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def start_optimized_server():
    print("🚀 Starting Optimized Anti-Spoofing Server")
    print("=" * 50)
    
    try:
        # Import launcher
        from src.core.app_launcher import launch_web_app
        print("✅ Launcher imported successfully")
        
        # Start server
        print("🌐 Starting web application...")
        launch_web_app()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server start failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_optimized_server()

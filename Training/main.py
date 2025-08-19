#!/usr/bin/env python3
"""
Face Anti-Spoofing Attendance System
Main Application Entry Point

Usage:
    python main.py --mode [web|train|test]
    python main.py --help

Author: Face Anti-Spoofing Team
Version: 1.0.0
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.app_launcher import AppLauncher
from src.utils.logger import setup_logging


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Face Anti-Spoofing Attendance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode web              # Launch web interface
    python main.py --mode train            # Train the model
    python main.py --mode test             # Run system tests
    python main.py --mode setup            # Setup and install dependencies
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['web', 'train', 'test', 'setup'],
        default='web',
        help='Application mode (default: web)'
    )
    
    parser.add_argument(
        '--config',
        default='config/default.json',
        help='Configuration file path (default: config/default.json)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web server port (default: 5000)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Web server host (default: localhost)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("FACE ANTI-SPOOFING ATTENDANCE SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug: {args.debug}")
    
    try:
        # Initialize application launcher
        launcher = AppLauncher(
            config_path=args.config,
            debug=args.debug
        )
        
        # Execute based on mode
        if args.mode == 'web':
            launcher.launch_web_app(host=args.host, port=args.port)
        elif args.mode == 'train':
            launcher.train_model()
        elif args.mode == 'test':
            launcher.run_tests()
        elif args.mode == 'setup':
            launcher.setup_environment()
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[ERROR] Application failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

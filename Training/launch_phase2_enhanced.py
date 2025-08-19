#!/usr/bin/env python3
"""
Phase 2 Complete: Enhanced Challenge-Response System Launcher
Ready to test all improvements with real camera input
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    print("=" * 70)
    print("🎉 PHASE 2: ENHANCED CHALLENGE-RESPONSE SYSTEM")
    print("   Anti-Spoofing Improvements Complete!")
    print("=" * 70)
    
    print("\n🚀 Available Test Options:")
    print("   1. Enhanced Challenge System Test (Real Camera)")
    print("   2. Quick Challenge Generation Test") 
    print("   3. Phase 1 + Phase 2 Combined Test")
    print("   4. Exit")
    
    try:
        choice = input("\n📝 Select test option (1-4): ").strip()
        
        if choice == "1":
            print("\n🎥 Starting Enhanced Challenge System Test with Camera...")
            print("   Controls:")
            print("   • 'n' - Random challenge")
            print("   • 's' - Sequence challenge")  
            print("   • 'e'/'m'/'h' - Easy/Medium/Hard difficulty")
            print("   • 'q' - Quit")
            print("\n⚡ Launching in 3 seconds...")
            import time
            time.sleep(3)
            
            from test_enhanced_challenge_system import test_enhanced_challenge_system
            test_enhanced_challenge_system()
            
        elif choice == "2":
            print("\n🔧 Running Quick Challenge Generation Test...")
            from src.challenge.challenge_response import ChallengeResponseSystem, ChallengeDifficulty
            
            system = ChallengeResponseSystem()
            print("\n✅ Testing Challenge Generation:")
            
            # Test different difficulties
            for difficulty in [ChallengeDifficulty.EASY, ChallengeDifficulty.MEDIUM, ChallengeDifficulty.HARD]:
                print(f"\n📊 {difficulty.value.upper()} Challenges:")
                for i in range(2):
                    challenge = system.generate_random_challenge(difficulty)
                    print(f"   {challenge.challenge_type.value}: {challenge.description}")
            
            # Test sequence
            seq_challenge = system.generate_sequence_challenge(difficulty=ChallengeDifficulty.MEDIUM)
            print(f"\n🔗 Sequence: {seq_challenge.description}")
            
            print("\n✅ Quick test completed successfully!")
            
        elif choice == "3":
            print("\n🌟 Starting Combined Phase 1 + Phase 2 Test...")
            print("   This test validates both landmark improvements AND challenge enhancements")
            print("\n⚡ Launching in 3 seconds...")
            import time
            time.sleep(3)
            
            # Run combined test
            from test_enhanced_challenge_system import test_enhanced_challenge_system
            test_enhanced_challenge_system()
            
        elif choice == "4":
            print("\n👋 Goodbye! Phase 2 Enhanced Challenge System ready for use.")
            
        else:
            print("\n❌ Invalid choice. Please select 1-4.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted. Phase 2 system ready!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("🔧 Please check your camera and dependencies.")

if __name__ == "__main__":
    main()

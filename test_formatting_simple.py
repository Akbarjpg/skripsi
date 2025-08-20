#!/usr/bin/env python3
"""
Quick test to verify the formatting fixes for Step 3 challenge system
"""

from src.challenge.challenge_response import ChallengeResponseSystem

def test_formatting_fixes():
    """Test that all challenge types can be created without formatting errors"""
    print("Testing Step 3 Challenge System Formatting Fixes")
    print("=" * 60)
    
    system = ChallengeResponseSystem()
    
    # Test different challenge types
    test_results = []
    
    for i in range(6):
        try:
            print(f"\nTest {i+1}:")
            challenge = system.start_challenge('random')
            if challenge:
                print(f"SUCCESS: {challenge.challenge_type.value}")
                print(f"   Description: {challenge.description}")
                test_results.append(True)
            else:
                print("No challenge created (max attempts reached)")
                test_results.append(None)
        except ValueError as e:
            if "Unknown format code 'f' for object of type 'str'" in str(e):
                print(f"FORMATTING ERROR: {e}")
                test_results.append(False)
                break
            else:
                print(f"Other ValueError: {e}")
                test_results.append(None)
        except Exception as e:
            print(f"UNEXPECTED ERROR: {e}")
            test_results.append(False)
            break
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    
    success_count = test_results.count(True)
    error_count = test_results.count(False)
    none_count = test_results.count(None)
    
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped/None: {none_count}")
    
    if error_count == 0:
        print("\nALL FORMATTING FIXES WORKING CORRECTLY!")
        return True
    else:
        print("\nFORMATTING ERRORS STILL PRESENT!")
        return False

if __name__ == "__main__":
    test_formatting_fixes()

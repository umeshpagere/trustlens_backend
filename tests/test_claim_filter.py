import sys
import os

# Add the project directory to sys.path for imports
sys.path.append("/Users/umeshpagere/Downloads/trustlens-2-main/backend")

from app.services.evidence_pipeline.claim_strength_filter import filter_claims

def test_filter_claims():
    test_cases = [
        # Metadata
        ("The video was posted on 08.03.2026", False),
        ("Watch this video", False),
        ("This post is viral", False),
        ("Caption reads: Hello world", False),
        
        # Questions
        ("What if Iranians don't rise up?", False),
        ("Is it raining in London?", False),
        
        # Short / Vague
        ("This is shocking", False),
        ("Great results", False),
        
        # Strong Factual
        ("Amitabh Bachchan purchased land in Ayodhya.", True),
        ("The US launched airstrikes against Iran.", True),
        ("WHO declared a global health emergency on January 30, 2020.", True),
        
        # Medium Factual
        ("Israeli President Isaac Herzog criticized Iran.", True),
        ("Air India canceled flights due to conflict.", True)
    ]
    
    inputs = [case[0] for case in test_cases]
    expected_outputs = [case[0] for case in test_cases if case[1]]
    
    print("\n--- Starting Claim Filter Test ---")
    filtered = filter_claims(inputs)
    
    print(f"Total Inputs: {len(inputs)}")
    print(f"Filtered Outputs: {len(filtered)}")
    
    success = True
    for expected in expected_outputs:
        if expected not in filtered:
            print(f"FAIL: Expected claim not found: {expected}")
            success = False
            
    for claim in filtered:
        if claim not in expected_outputs:
            print(f"FAIL: Unexpected claim allowed: {claim}")
            success = False
            
    if success:
        print("✅ SUCCESS: All test cases passed.")
    else:
        print("❌ FAILURE: Test cases did not match expected behavior.")

if __name__ == "__main__":
    test_filter_claims()

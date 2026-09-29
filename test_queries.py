import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "groundwater-chatbot"))
from app import process_query

test_cases = [
    ("What is the groundwater extraction in Hyderabad?", "extraction", "Hyderabad", True),
    ("What is the groundwater recharge in Karimnagar?", "recharge", "Karimnagar", True),
    ("What is the groundwater availability in Adilabad?", "availability", "Adilabad", True),
    ("What is the extraction stage in Jagtial?", "stage", "Jagtial", True),
    ("Which district has the highest groundwater extraction?", "comparison", "Nalgonda", True),
    ("What is the future groundwater availability in Hyderabad?", "future_availability", "Hyderabad", True),
    ("What is the groundwater extraction?", "extraction", None, False),
]

all_passed = True
print("=" * 70)
print("RUNNING 7 FINAL REQUIRED ACCEPTANCE TESTS")
print("=" * 70)

for query, expected_intent, expected_loc, expected_success in test_cases:
    res = process_query(query)
    intent_match = res.get("intent") == expected_intent
    loc_match = res.get("location") == expected_loc
    succ_match = res.get("success") == expected_success
    passed = intent_match and loc_match and succ_match
    
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] Query: {query}")
    print(f"  Predicted Intent: {res.get('intent')} | Expected: {expected_intent}")
    print(f"  Detected District: {res.get('location')} | Expected: {expected_loc}")
    if res.get("success"):
        print(f"  Retrieved Answer: {res.get('answer')}")
    else:
        print(f"  Friendly Fallback: {res.get('error')}")
    print("-" * 70)
    if not passed:
        all_passed = False

if all_passed:
    print("RESULT: ALL 7 REQUIRED VERIFICATION TESTS PASSED SUCCESSFULLY!")
else:
    print("RESULT: VERIFICATION FAILED!")
    sys.exit(1)

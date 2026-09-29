"""
verify_all.py
Runs automated verification tests against the Groundwater Chatbot retrieval engine
covering all SIH 1697 requirements and requested edge cases.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "groundwater-chatbot"))
from app import process_query

test_suite = [
    {
        "query": "What is the groundwater recharge in Hyderabad?",
        "expected_intent": "recharge",
        "expected_loc": "Hyderabad",
        "expected_metric": "Total annual groundwater recharge",
        "expected_success": True
    },
    {
        "query": "What is the industrial groundwater extraction in Hyderabad?",
        "expected_intent": "extraction",
        "expected_loc": "Hyderabad",
        "expected_metric": "Industrial - Annual extraction",
        "expected_success": True
    },
    {
        "query": "What is the domestic groundwater extraction in Hyderabad?",
        "expected_intent": "extraction",
        "expected_loc": "Hyderabad",
        "expected_metric": "Domestic - Annual Extraction",
        "expected_success": True
    },
    {
        "query": "What is the irrigation groundwater extraction in Nalgonda?",
        "expected_intent": "extraction",
        "expected_loc": "Nalgonda",
        "expected_metric": "Irrigation - Annual extraction",
        "expected_success": True
    },
    {
        "query": "Which district has the highest groundwater recharge?",
        "expected_intent": "comparison",
        "expected_loc": "Nalgonda",
        "expected_metric": "Total annual groundwater recharge",
        "expected_success": True
    },
    {
        "query": "Which district has the lowest groundwater extraction?",
        "expected_intent": "comparison",
        "expected_loc": "Hyderabad",
        "expected_metric": "Total Annual Extraction",
        "expected_success": True
    },
    {
        "query": "Which district has the highest extraction stage?",
        "expected_intent": "comparison",
        "expected_loc": "Hyderabad",
        "expected_metric": "Stage of GW extraction (%)",
        "expected_success": True
    },
    {
        "query": "Which district has the highest future groundwater availability?",
        "expected_intent": "comparison",
        "expected_loc": "Suryapet",
        "expected_metric": "Net GW availability for future",
        "expected_success": True
    },
    {
        "query": "What is the groundwater extraction stage in Hyderabad?",
        "expected_intent": "stage",
        "expected_loc": "Hyderabad",
        "expected_metric": "Stage of GW extraction (%)",
        "expected_success": True
    },
    {
        "query": "What is the future groundwater availability in Karimnagar?",
        "expected_intent": "future_availability",
        "expected_loc": "Karimnagar",
        "expected_metric": "Net GW availability for future",
        "expected_success": True
    },
    {
        "query": "Compare groundwater recharge in Adilabad and Jagtial.",
        "expected_intent": "comparison",
        "expected_loc": "Adilabad, Jagtial",
        "expected_metric": "Total annual groundwater recharge",
        "expected_success": True
    },
    {
        "query": "What is groundwater extraction in Chennai?",
        "expected_intent": "extraction",
        "expected_loc": "Chennai",
        "expected_metric": None,
        "expected_success": False,
        "expected_msg_contains": "That information is not available in the current government dataset."
    },
    {
        "query": "What is the groundwater extraction in Hyderabad?",
        "expected_intent": "extraction",
        "expected_loc": "Hyderabad",
        "expected_metric": "Total Annual Extraction",
        "expected_success": True
    },
    {
        "query": "What is the groundwater availability in Adilabad?",
        "expected_intent": "availability",
        "expected_loc": "Adilabad",
        "expected_metric": "Annual Extractable Groundwater Resource",
        "expected_success": True
    },
    {
        "query": "What is the extraction stage in Jagtial?",
        "expected_intent": "stage",
        "expected_loc": "Jagtial",
        "expected_metric": "Stage of GW extraction (%)",
        "expected_success": True
    },
    {
        "query": "What is the groundwater extraction?",
        "expected_intent": "extraction",
        "expected_loc": None,
        "expected_metric": None,
        "expected_success": False,
        "expected_msg_contains": "Please mention a Telangana district such as Hyderabad, Karimnagar or Adilabad."
    },
    {
        "query": "",
        "expected_intent": None,
        "expected_loc": None,
        "expected_metric": None,
        "expected_success": False,
        "expected_msg_contains": "Please enter a valid groundwater query."
    }
]

print("=" * 80)
print("EXECUTING SIH 1697 COMPREHENSIVE VERIFICATION SUITE")
print("=" * 80)

passed_count = 0
failed_count = 0

for idx, case in enumerate(test_suite, 1):
    q = case["query"]
    res = process_query(q)
    
    succ_ok = (res.get("success") == case["expected_success"])
    intent_ok = (case["expected_intent"] is None or res.get("intent") == case["expected_intent"])
    loc_ok = (res.get("location") == case["expected_loc"])
    metric_ok = (case["expected_metric"] is None or res.get("retrieved_metric") == case["expected_metric"])
    
    msg_ok = True
    if "expected_msg_contains" in case:
        msg_ok = case["expected_msg_contains"] in res.get("error", "")

    all_ok = succ_ok and intent_ok and loc_ok and metric_ok and msg_ok
    status = "PASS" if all_ok else "FAIL"

    if all_ok:
        passed_count += 1
    else:
        failed_count += 1

    print(f"[{status}] Test #{idx}: \"{q}\"")
    if res.get("intent"):
        print(f"  • Predicted Intent:   {res.get('intent')} (Expected: {case['expected_intent']})")
    print(f"  • Detected District:  {res.get('location')} (Expected: {case['expected_loc']})")
    if res.get("retrieved_metric"):
        print(f"  • Retrieved Metric:   {res.get('retrieved_metric')}")
    if res.get("retrieved_value"):
        print(f"  • Retrieved Value:    {res.get('retrieved_value')}")
    if res.get("answer"):
        print(f"  • Formatted Answer:   {res.get('answer')}")
    if res.get("error"):
        print(f"  • Fallback / Error:   {res.get('error')}")
    print("-" * 80)

print(f"\nFINAL SUMMARY: {passed_count} PASSED, {failed_count} FAILED out of {len(test_suite)} tests.")
if failed_count == 0:
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
else:
    sys.exit(1)

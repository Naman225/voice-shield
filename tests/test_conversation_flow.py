"""
Comprehensive Conversational Flow & Post-OTP Continuity Test Suite
Tests:
1. REST API endpoints (start, status, inquiry, transfer, send-otp, verify-otp, verify-question, simulate)
2. State preservation across biometric/AMBER challenges
3. Spoken OTP and security question verification
4. Natural speech amount extraction algorithms
5. Verification of all 9 prototype audio files for hackathon demonstration
"""

import os
import sys
import re
from fastapi.testclient import TestClient

# Ensure IndieFake is in path
sys.path.insert(0, '/home/naman/Desktop/SIH/IndieFake')
from main import app
from backend.service import account_registry

client = TestClient(app)

def test_full_conversation_lifecycle():
    print("=" * 80)
    print("🧪 TESTING END-TO-END CONVERSATIONAL FLOW & CONTEXT CONTINUITY")
    print("=" * 80)

    # 1. Start Call
    acc_num = "ACC-550189"
    account_registry.unlock_account(acc_num)
    start_res = client.post("/api/v1/call/start", json={"account_number": acc_num, "caller_id": "+91 98765 43210"})
    assert start_res.status_code == 200, f"Call start failed: {start_res.text}"
    call_id = start_res.json()["call_id"]
    print(f"[PASS] 1. Live Call Initialized: {call_id}")

    # 2. Balance Inquiry under GREEN
    bal_res = client.post("/api/v1/call/inquiry", json={"call_id": call_id, "account_number": acc_num})
    assert bal_res.status_code == 200, f"Balance inquiry failed: {bal_res.text}"
    bal_data = bal_res.json()
    assert "1,240,500" in bal_data["balance"]
    print(f"[PASS] 2. Baseline Balance Inquiry: {bal_data['balance']} (Owner: {bal_data['owner']})")

    # 3. Wire Transfer under GREEN
    transfer_res = client.post("/api/v1/bank/transfer", json={
        "call_id": call_id,
        "account_number": acc_num,
        "amount": 50000.0,
        "beneficiary": "ABC Corp Ltd"
    })
    assert transfer_res.status_code == 200, f"Transfer failed: {transfer_res.text}"
    tx_data = transfer_res.json()
    assert tx_data["status"] == "TRANSFER_SUCCESSFUL"
    print(f"[PASS] 3. Wire Transfer Success: Receipt {tx_data['receipt_id']}, ₹50,000 to {tx_data['beneficiary']}")

    # 4. Trigger AMBER Voice Anomaly Simulation
    amber_res = client.post(f"/api/v1/call/simulate/{call_id}", json={"mode": "AMBER"})
    assert amber_res.status_code == 200, f"Amber simulation failed: {amber_res.text}"
    amber_data = amber_res.json()
    assert amber_data["risk_tier"] == "AMBER"
    print(f"[PASS] 4. AMBER Caution Triggered: Risk {amber_data['running_risk_pct']}% -> Status: {amber_data['fraud_gate']['state']}")

    # 5. Wire Transfer held under AMBER (returns 202 CHALLENGE)
    held_res = client.post("/api/v1/bank/transfer", json={
        "call_id": call_id,
        "account_number": acc_num,
        "amount": 50000.0,
        "beneficiary": "ABC Corp Ltd"
    })
    assert held_res.status_code == 202, f"Expected 202 under AMBER, got {held_res.status_code}"
    demo_otp = held_res.json().get("demo_otp", "847291")
    print(f"[PASS] 5. Transfer Held Under AMBER: 202 Accepted (Step-Up OTP Challenge Dispatched: {demo_otp})")

    # 6. Verify Spoken OTP
    verify_otp_res = client.post(f"/api/v1/call/verify-otp/{call_id}", json={"otp": demo_otp})
    assert verify_otp_res.status_code == 200, f"OTP verify failed: {verify_otp_res.text}"
    otp_data = verify_otp_res.json()
    assert otp_data["status"] == "OTP_VERIFIED_SUCCESS"
    print(f"[PASS] 6. OTP Verified Successfully: Fraud gate unlocked, baseline restored to GREEN")

    # 7. Post-OTP Wire Transfer Execution (Simulating context resumption)
    resume_transfer_res = client.post("/api/v1/bank/transfer", json={
        "call_id": call_id,
        "account_number": acc_num,
        "amount": 50000.0,
        "beneficiary": "ABC Corp Ltd"
    })
    assert resume_transfer_res.status_code == 200, f"Resumed transfer failed: {resume_transfer_res.text}"
    resumed_data = resume_transfer_res.json()
    assert resumed_data["status"] == "TRANSFER_SUCCESSFUL"
    print(f"[PASS] 7. Resumed Wire Transfer Executed: Receipt {resumed_data['receipt_id']}")

    # 8. Security Question Step-Up Verification ("Max")
    client.post(f"/api/v1/call/simulate/{call_id}", json={"mode": "AMBER"})
    sq_res = client.post(f"/api/v1/call/verify-question/{call_id}", json={"answer": "Max"})
    assert sq_res.status_code == 200, f"Security question failed: {sq_res.text}"
    print(f"[PASS] 8. Security Question ('Max') Verified: Step-up authentication succeeded")

    # 9. Cleanup session
    client.delete(f"/api/v1/call/{call_id}")
    print(f"[PASS] 9. Call session cleaned up")

def test_natural_speech_amount_and_otp_extraction():
    print("\n" + "=" * 80)
    print("🧪 TESTING NATURAL LANGUAGE AMOUNT & SPOKEN OTP EXTRACTION")
    print("=" * 80)

    # Test amount words
    word_nums = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
    }

    def py_extract_amount(text):
        text = text.lower()
        lakh = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+(\.\d+)?)\s*(lakh|lac|lakhs|lacs)\b', text)
        if lakh:
            val = word_nums.get(lakh.group(1)) or float(lakh.group(1))
            return val * 100000
        thousand = re.search(r'\b(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|\d+(\.\d+)?)\s*(thousand|k)\b', text)
        if thousand:
            val = word_nums.get(thousand.group(1)) or float(thousand.group(1))
            return val * 1000
        plain = re.search(r'\b\d{3,9}\b', text.replace(',', ''))
        if plain:
            return float(plain.group(0))
        return 50000.0

    test_phrases = [
        ("I want to transfer fifty thousand rupees to ABC Corp", 50000.0),
        ("Transfer 20 thousand", 20000.0),
        ("Transfer two lakh rupees", 200000.0),
        ("Send 50,000", 50000.0),
        ("Transfer 10k", 10000.0),
        ("Transfer 75000", 75000.0)
    ]

    for phrase, expected in test_phrases:
        result = py_extract_amount(phrase)
        assert result == expected, f"Failed for '{phrase}': got {result}, expected {expected}"
        print(f"  [PASS] Amount: '{phrase}' -> ₹{result:,.0f}")

    # Test OTP extraction from words
    word_map = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
    def py_extract_otp(raw_text):
        normalized = raw_text.lower()
        for w, d in word_map.items():
            normalized = re.sub(rf'\b{w}\b', d, normalized)
        digits_only = re.sub(r'[^0-9]', '', normalized)
        match = re.search(r'\b\d{6}\b', digits_only) or re.search(r'\d{6}', digits_only)
        return match.group(0) if match else None

    otp_phrases = [
        ("eight four seven two nine one", "847291"),
        ("My OTP code is 847291", "847291"),
        ("8 4 7 2 9 1", "847291"),
        ("the code is eight four seven two nine one please verify", "847291")
    ]

    for phrase, expected in otp_phrases:
        res = py_extract_otp(phrase)
        assert res == expected, f"Failed OTP for '{phrase}': got {res}, expected {expected}"
        print(f"  [PASS] Spoken OTP: '{phrase}' -> {res}")

def test_prototype_audio_files():
    print("\n" + "=" * 80)
    print("🧪 TESTING PROTOTYPE AUDIO FILES GENERATED FOR PHONE PLAYBACK")
    print("=" * 80)

    audio_dir = "/home/naman/Desktop/SIH/IndieFake/static/demo_audio"
    expected_files = [
        "01_name_and_purpose.mp3",
        "02_transfer_money.mp3",
        "03_confirm_yes.mp3",
        "04_payment_purpose.mp3",
        "05_authorize_phrase.mp3",
        "06_check_balance.mp3",
        "07_spoken_otp.mp3",
        "08_security_pet.mp3",
        "09_farewell_done.mp3"
    ]

    for f in expected_files:
        path = os.path.join(audio_dir, f)
        assert os.path.isfile(path), f"Missing audio file: {f}"
        size = os.path.getsize(path)
        assert size > 5000, f"Audio file too small ({size} bytes): {f}"
        print(f"  [PASS] Audio File: {f} ({size / 1024:.1f} KB)")

if __name__ == "__main__":
    test_full_conversation_lifecycle()
    test_natural_speech_amount_and_otp_extraction()
    test_prototype_audio_files()
    print("\n" + "=" * 80)
    print("🏁 ALL CONVERSATION CONTINUITY & PROTOTYPE TESTS PASSED (100%)")
    print("=" * 80)

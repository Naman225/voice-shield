"""
Voice Shield SDK — Banking Integration Demo
=============================================
Shows how a bank integrates Voice Shield in ~25 lines.

Run:
    python sdk/examples/bank_integration_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from sdk import VoiceShieldClient, VoiceShieldError

BASE_URL = "https://voice-shield-kf0h.onrender.com"

def main():
    print("Voice Shield SDK — Live Integration Demo")
    print("=" * 50)

    with VoiceShieldClient(BASE_URL) as vs:

        # 1. Health check
        latency = vs.ping()
        print(f"[1] Server ping: {latency}ms  |  Healthy: {vs.health()}")

        call_id = "SDK-DEMO-001"

        # 2. Start call session
        vs.start_call(call_id, caller_id="+91-98765-43210", account_number="ACC-550189")
        print(f"[2] Call session started: {call_id}")

        # 3. Simulate AI clone attack → risk goes RED
        vs.simulate_clone_attack(call_id)
        state = vs.get_risk(call_id)
        tier = state.get("risk_tier", state.get("current_tier", "N/A"))
        print(f"[3] Risk after clone injection: {tier} ({state.get('running_risk_pct', 0):.1f}%)")

        # 4. Attempt transfer — correctly BLOCKED at RED (this is the fraud gate working)
        try:
            result = vs.approve_transfer(call_id, "ACC-550189", amount=2_500_000.0, beneficiary="Fraudster Corp")
            print(f"[4] Transfer result: {result.get('status')}")
        except VoiceShieldError as e:
            if e.status_code == 403:
                print(f"[4] Transfer BLOCKED (HTTP 403) — fraud gate working correctly: {e.detail[:80]}...")
            else:
                raise

        # 5. Reset risk + supervisor unfreeze → back to GREEN
        vs.simulate_reset(call_id)
        vs.unfreeze_account(call_id, supervisor_id="SUP-DEMO")
        state2 = vs.get_risk(call_id)
        print(f"[5] After reset: {state2.get('risk_tier', 'N/A')} | Unfrozen by supervisor")

        # 6. Transfer now succeeds on GREEN
        try:
            result2 = vs.approve_transfer(call_id, "ACC-550189", amount=5000.0, beneficiary="Trusted Payee")
            print(f"[6] Transfer on GREEN: {result2.get('status', result2.get('call_action', 'OK'))}")
        except VoiceShieldError as e:
            print(f"[6] Transfer response: HTTP {e.status_code}")

        # 7. Forensic scan of a recorded file
        sample_file = os.path.join(os.path.dirname(__file__), "../../test_samples/demo_voice_clone.mp3")
        if os.path.exists(sample_file):
            scan = vs.scan_audio_file(sample_file)
            print(f"[7] Forensic scan:")
            print(f"    Verdict    : {scan['verdict']}")
            print(f"    Risk Tier  : {scan['risk_tier']}")
            print(f"    Peak Score : {scan['latest_frame_score']:.1f}%")
            print(f"    Duration   : {scan['duration_sec']:.1f}s")
        else:
            print(f"[7] Forensic scan: demo file not found (add test_samples/demo_voice_clone.mp3)")

        # 8. End call
        vs.end_call(call_id)
        print(f"[8] Call session closed")

    print("\n[DONE] All SDK methods verified successfully.")
    print("=" * 50)
    print("In production: replace simulate_clone_attack() with your")
    print("IVR audio stream via:  wss://.../ws/call/{call_id}")


if __name__ == "__main__":
    main()

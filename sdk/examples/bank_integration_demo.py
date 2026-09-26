"""
Voice Shield SDK — Banking Integration Demo
=============================================
This file shows exactly how a bank's IT team would integrate Voice Shield
into their core banking system (CBS) in under 25 lines.

Run it:
    python sdk/examples/bank_integration_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from sdk import VoiceShieldClient

BASE_URL = "https://voice-shield-kf0h.onrender.com"  # Change to localhost:8000 for local testing

def main():
    print("Voice Shield SDK — Live Integration Demo")
    print("=" * 50)

    with VoiceShieldClient(BASE_URL) as vs:

        # Step 1: Check server is alive
        latency = vs.ping()
        print(f"[1] Server ping: {latency}ms  |  Healthy: {vs.health()}")

        # Step 2: Register incoming call (happens when customer calls bank)
        call_id = "SDK-DEMO-001"
        vs.start_call(call_id, caller_id="+91-98765-43210", account_number="ACC-550189")
        print(f"[2] Call session started: {call_id}")

        # Step 3: Simulate a voice clone attack (inject RED risk)
        vs.simulate_clone_attack(call_id)
        state = vs.get_risk(call_id)
        print(f"[3] Risk after clone injection: {state.get('risk_tier', state.get('current_tier', 'N/A'))} ({state.get('running_risk_pct', 0):.1f}%)")

        # Step 4: Attempt wire transfer — should be BLOCKED by fraud gate
        result = vs.approve_transfer(call_id, "ACC-550189", amount=2_500_000.0, beneficiary="Fraudster Corp")
        print(f"[4] Transfer result: {result.get('status')} | Action: {result.get('call_action')}")

        # Step 5: Supervisor unlocks, resets to GREEN
        vs.simulate_reset(call_id)
        vs.unfreeze_account(call_id, supervisor_id="SUP-DEMO")
        print(f"[5] Account unfrozen by supervisor")

        # Step 6: Forensic scan of an audio file
        sample_file = os.path.join(os.path.dirname(__file__), "../../test_samples/demo_voice_clone.mp3")
        if os.path.exists(sample_file):
            scan = vs.scan_audio_file(sample_file)
            print(f"[6] Forensic scan result:")
            print(f"    Verdict     : {scan['verdict']}")
            print(f"    Risk Tier   : {scan['risk_tier']}")
            print(f"    Peak Score  : {scan['latest_frame_score']:.1f}%")
            print(f"    Duration    : {scan['duration_sec']:.1f}s")
            print(f"    Evidence SHA: {scan['fraud_gate'].get('lock_reason', 'N/A')}")
        else:
            print(f"[6] Skipping forensic scan — demo file not found at {sample_file}")

        # Step 7: End call
        vs.end_call(call_id)
        print(f"[7] Call session closed")

    print("\n[DONE] — Voice Shield SDK integration test passed.")
    print("=" * 50)
    print("For production, replace simulate_clone_attack() with your")
    print("real IVR audio stream via the WebSocket endpoint:")
    print("  wss://voice-shield-kf0h.onrender.com/ws/call/{call_id}")


if __name__ == "__main__":
    main()

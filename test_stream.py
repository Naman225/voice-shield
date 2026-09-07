"""
Real-Time Audio Stream Test Harness
Simulates live telephone audio packets streaming into the FastAPI Backend.
"""

import time
import requests
import numpy as np

BASE_URL = "http://localhost:8000"


def run_stream_simulation():
    print("=" * 70)
    print("  SIH 2026: Real-Time Voice Cloning Detection Backend Test Harness  ")
    print("=" * 70)

    # 1. Health check
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"[Health Check] Server Response: {r.json()['status']}")
    except Exception as e:
        print(f"[Error] Backend server is not running on {BASE_URL}. Run 'python main.py' first.")
        return

    # 2. Start Call Session
    call_id = f"CALL-DEMO-{int(time.time())}"
    start_payload = {
        "call_id": call_id,
        "caller_id": "CEO Executive Line (+91 98765 43210)",
        "account_number": "ACC-550189",
    }
    r = requests.post(f"{BASE_URL}/api/v1/call/start", json=start_payload)
    print(f"\n[1] Started Call Session: {call_id}")
    print(f"    Target Account: {start_payload['account_number']}")

    # 3. Simulate Phase 1: Authentic Indian Speech (10 seconds, 40 chunks of 250ms)
    print("\n[2] Phase 1: Streaming Genuine Speech (Authentic Human Voice)...")
    sample_rate = 16000
    chunk_duration_sec = 0.25
    samples_per_chunk = int(sample_rate * chunk_duration_sec)

    for i in range(1, 16):
        # Generate 250ms of synthetic audio PCM samples
        t = np.linspace(0, chunk_duration_sec, samples_per_chunk, endpoint=False)
        pcm_samples = (np.sin(2 * np.pi * 440 * t) * 0.3).tolist()

        chunk_payload = {
            "call_id": call_id,
            "samples": pcm_samples,
            "sample_rate": sample_rate,
            "is_simulated_clone": False,
        }
        res = requests.post(f"{BASE_URL}/api/v1/call/chunk", json=chunk_payload).json()
        print(
            f"  [Chunk {i:02d}] Duration: {res['duration_sec']}s | Frame Score: {res['latest_frame_score']}% | "
            f"Running Risk (EMA): {res['running_risk_pct']}% | Tier: [{res['risk_tier']}] | Gate: {res['fraud_gate']['state']}"
        )
        time.sleep(0.15)  # Simulate real-time stream pacing

    # 4. Simulate Phase 2: Cloned Synthetic Impersonation Attack Initiated!
    print("\n[3] Phase 2: ATTACK INITIATED! Streaming Cloned Voice Audio Packet (Deepfake)...")
    for i in range(16, 35):
        t = np.linspace(0, chunk_duration_sec, samples_per_chunk, endpoint=False)
        pcm_samples = (np.sin(2 * np.pi * 880 * t) * 0.8).tolist()

        chunk_payload = {
            "call_id": call_id,
            "samples": pcm_samples,
            "sample_rate": sample_rate,
            "is_simulated_clone": True,
        }
        res = requests.post(f"{BASE_URL}/api/v1/call/chunk", json=chunk_payload).json()
        print(
            f"  [Chunk {i:02d}] Duration: {res['duration_sec']}s | Frame Score: {res['latest_frame_score']}% | "
            f"Running Risk (EMA): {res['running_risk_pct']}% | Tier: \033[91m[{res['risk_tier']}]\033[0m | Gate: \033[91m{res['fraud_gate']['state']}\033[0m"
        )
        if res["fraud_gate"]["is_frozen"]:
            print(f"\n  ===> \033[91mACTIVE FRAUD FREEZE ACTIVATED!\033[0m")
            print(f"       Reason: {res['fraud_gate']['lock_reason']}")
            print(f"       Action: {res['fraud_gate']['action_prompt']}")
            break
        time.sleep(0.15)

    # 5. Fetch final status summary
    print("\n[4] Fetching Final Session Summary & Risk History...")
    status = requests.get(f"{BASE_URL}/api/v1/call/status/{call_id}").json()
    print(f"    Call Duration: {status['duration_sec']}s")
    print(f"    Total Chunks Processed: {status['chunks_ingested']}")
    print(f"    Final Running Risk Score: {status['running_risk_pct']}%")

    print("\n" + "=" * 70)
    print("  Test Simulation Complete. Backend is operational and responsive!  ")
    print("=" * 70)


if __name__ == "__main__":
    run_stream_simulation()

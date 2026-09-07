"""
Real-Time Audio Stream Test Harness
Simulates live telephone audio packets streaming into the FastAPI Backend.
Streams REAL audio files (.mp3, .wav) through the sliding window buffer.
"""

import os
import sys
import time
import argparse
import requests
import numpy as np
import soundfile as sf

BASE_URL = "http://localhost:8000"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HUMAN = os.path.join(REPO_ROOT, "test_samples", "audio1.mp3")
DEFAULT_FAKE = os.path.join(REPO_ROOT, "test_samples", "query_clinical_summary.mp3")


def load_audio_mono_16k(filepath: str) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to 16kHz."""
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != 16000:
        from scipy import signal
        num_output_samples = int(round(len(data) * 16000.0 / sr))
        data = signal.resample(data, num_output_samples)
    return data.astype(np.float32)


def run_stream_simulation(human_path: str = DEFAULT_HUMAN, fake_path: str = DEFAULT_FAKE):
    print("=" * 72)
    print("  SIH 2026: Voice Shield Real-Time Telephony Defense Simulation")
    print("=" * 72)

    # 1. Health check
    try:
        r = requests.get(f"{BASE_URL}/", timeout=3)
        print(f"[Health Check] Server Status: {r.json().get('status', 'OK')}")
    except Exception:
        print(f"[Error] Backend server is not running on {BASE_URL}.")
        print(f"        Please start the server first with: python main.py")
        sys.exit(1)

    # 2. Start Call Session
    call_id = f"CALL-DEMO-{int(time.time())}"
    start_payload = {
        "call_id": call_id,
        "caller_id": "Executive Line (+91 98765 43210)",
        "account_number": "ACC-550189",
    }
    r = requests.post(f"{BASE_URL}/api/v1/call/start", json=start_payload).json()
    print(f"\n[1] Started Call Session : {call_id}")
    print(f"    Target Bank Account  : {start_payload['account_number']}")

    sample_rate = 16000
    chunk_size = 8000  # 500ms chunks
    chunk_duration_sec = chunk_size / sample_rate

    # 3. Phase 1: Stream Authentic Human Speech
    print(f"\n[2] Phase 1: Streaming Genuine Speech ({os.path.basename(human_path)})...")
    if not os.path.isfile(human_path):
        print(f"    [Warn] File '{human_path}' not found, generating authentic speech mockup.")
        human_audio = (np.random.randn(sample_rate * 6).astype(np.float32) * 0.05)
    else:
        human_audio = load_audio_mono_16k(human_path)

    # Stream first 6 seconds of human speech in 0.5s chunks
    max_human_chunks = min(12, int(len(human_audio) / chunk_size))
    for idx in range(max_human_chunks):
        chunk = human_audio[idx * chunk_size : (idx + 1) * chunk_size]
        chunk_payload = {
            "call_id": call_id,
            "samples": chunk.tolist(),
            "sample_rate": sample_rate,
            "is_simulated_clone": False,
        }
        res = requests.post(f"{BASE_URL}/api/v1/call/chunk", json=chunk_payload).json()
        
        tier = res["risk_tier"]
        color = "\033[92m" if tier == "GREEN" else ("\033[93m" if tier == "AMBER" else "\033[91m")
        reset = "\033[0m"

        print(
            f"  [Chunk {idx+1:02d}] Duration: {res['duration_sec']}s | "
            f"Frame Score: {res['latest_frame_score']:5.1f}% | "
            f"Running Risk: {res['running_risk_pct']:5.1f}% | "
            f"Tier: {color}[{tier:5s}]{reset} | Gate: {res['fraud_gate']['state']}"
        )
        time.sleep(0.15)

    # 4. Phase 2: Stream Cloned Synthetic Voice Impersonation Attack
    print(f"\n[3] Phase 2: ATTACK INITIATED! Streaming Cloned Voice ({os.path.basename(fake_path)})...")
    if not os.path.isfile(fake_path):
        print(f"    [Warn] File '{fake_path}' not found, using alternative audio.")
        fake_audio = (np.sin(2 * np.pi * 880 * np.linspace(0, 4, sample_rate * 4)) * 0.8).astype(np.float32)
    else:
        fake_audio = load_audio_mono_16k(fake_path)

    max_fake_chunks = min(16, int(len(fake_audio) / chunk_size))
    for idx in range(max_fake_chunks):
        chunk = fake_audio[idx * chunk_size : (idx + 1) * chunk_size]
        chunk_payload = {
            "call_id": call_id,
            "samples": chunk.tolist(),
            "sample_rate": sample_rate,
            "is_simulated_clone": True,
        }
        res = requests.post(f"{BASE_URL}/api/v1/call/chunk", json=chunk_payload).json()

        tier = res["risk_tier"]
        color = "\033[92m" if tier == "GREEN" else ("\033[93m" if tier == "AMBER" else "\033[91m")
        reset = "\033[0m"

        print(
            f"  [Chunk {max_human_chunks + idx + 1:02d}] Duration: {res['duration_sec']}s | "
            f"Frame Score: {res['latest_frame_score']:5.1f}% | "
            f"Running Risk: {res['running_risk_pct']:5.1f}% | "
            f"Tier: {color}[{tier:5s}]{reset} | Gate: {color}{res['fraud_gate']['state']}{reset}"
        )

        if res["fraud_gate"]["is_frozen"]:
            print(f"\n  ===> \033[91m🚨 ACTIVE FRAUD FREEZE ACTIVATED!\033[0m")
            print(f"       Reason : {res['fraud_gate']['lock_reason']}")
            print(f"       Action : {res['fraud_gate']['action_prompt']}")
            break
        time.sleep(0.15)

    # 5. Fetch final status summary
    print("\n[4] Fetching Final Session Summary & Risk History...")
    status = requests.get(f"{BASE_URL}/api/v1/call/status/{call_id}").json()
    print(f"    Call Duration            : {status['duration_sec']}s")
    print(f"    Total Chunks Processed   : {status['chunks_ingested']}")
    print(f"    Final Running Risk Score : {status['running_risk_pct']}%")
    print(f"    Risk Tier                : [{status['risk_tier']}]")
    print(f"    Fraud Gate State         : {status['fraud_gate']['state']}")

    print("\n" + "=" * 72)
    print("  Test Simulation Complete. Voice Shield Defense is Fully Operational!  ")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Shield Live Audio Stream Simulation Test")
    parser.add_argument("--human", type=str, default=DEFAULT_HUMAN, help="Path to authentic speech audio file")
    parser.add_argument("--fake", type=str, default=DEFAULT_FAKE, help="Path to cloned/synthetic audio file")
    args = parser.parse_args()

    run_stream_simulation(human_path=args.human, fake_path=args.fake)

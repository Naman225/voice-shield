"""
Real-Time Audio Stream Test Harness
Simulates live telephone audio packets streaming into the FastAPI Backend.
Streams REAL audio files (.mp3, .wav) through the WebSocket sliding window buffer.

Usage:
    # Requires the server to be running first: python main.py
    python test_stream.py
    python test_stream.py --human test_samples/audio1.mp3 --fake test_samples/query_clinical_summary.mp3
"""

import os
import sys
import time
import json
import asyncio
import argparse
import requests
import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    print("[Error] 'websockets' package not found. Install: pip install websockets")
    sys.exit(1)

BASE_URL  = "http://localhost:8000"
WS_URL    = "ws://localhost:8000"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HUMAN = os.path.join(REPO_ROOT, "test_samples", "audio1.mp3")
DEFAULT_FAKE  = os.path.join(REPO_ROOT, "test_samples", "query_clinical_summary.mp3")


def load_audio_mono_16k(filepath: str) -> np.ndarray:
    """Load an audio file, convert to mono, and resample to 16 kHz."""
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    if sr != 16000:
        from scipy import signal
        num_out = int(round(len(data) * 16000.0 / sr))
        data = signal.resample(data, num_out)
    return data.astype(np.float32)


def health_check() -> bool:
    """Verify the backend server is reachable."""
    try:
        r = requests.get(f"{BASE_URL}/", timeout=3)
        status = r.json().get("status", "OK")
        print(f"[Health Check] Server Status: {status}")
        return True
    except Exception:
        print(f"[Error] Backend server is not running on {BASE_URL}.")
        print(f"        Start it with: python main.py")
        return False


async def stream_audio_over_ws(
    call_id: str,
    audio: np.ndarray,
    label: str,
    account: str = "ACC-550189",
    caller_id: str = "+91 98765 43210",
    chunk_size: int = 8000,
    delay: float = 0.15,
):
    """Stream audio chunks over WebSocket and print real-time risk updates."""
    ws_endpoint = f"{WS_URL}/ws/call/{call_id}?account={account}&caller_id={caller_id}"
    total_chunks = min(16, int(len(audio) / chunk_size))
    last_res = {}

    async with websockets.connect(ws_endpoint) as ws:
        for idx in range(total_chunks):
            chunk = audio[idx * chunk_size : (idx + 1) * chunk_size]

            payload = json.dumps({
                "samples": chunk.tolist(),
                "sample_rate": 16000,
                "account_number": account,
                "caller_id": caller_id,
            })
            await ws.send(payload)

            raw = await ws.recv()
            res = json.loads(raw)
            last_res = res

            tier  = res.get("risk_tier", "GREEN")
            risk  = res.get("running_risk_pct", 0.0)
            score = res.get("latest_frame_score", 0.0)
            gate  = res.get("fraud_gate", {})

            color = "\033[92m" if tier == "GREEN" else ("\033[93m" if tier == "AMBER" else "\033[91m")
            reset = "\033[0m"

            print(
                f"  [{label} {idx+1:02d}] "
                f"Duration: {res.get('duration_sec', 0.0):.1f}s | "
                f"Frame: {score:5.1f}% | "
                f"Risk: {risk:5.1f}% | "
                f"Tier: {color}[{tier:5s}]{reset} | "
                f"Gate: {gate.get('state', '?')}"
            )

            if gate.get("is_frozen"):
                print(f"\n  ===> {color}FRAUD GATE ACTIVATED - ACCOUNT FROZEN!{reset}")
                print(f"       Reason : {gate.get('lock_reason', '')}")
                print(f"       Action : {gate.get('action_prompt', '')}")
                break

            await asyncio.sleep(delay)

    return last_res


async def run_async_simulation(human_path: str, fake_path: str):
    print("=" * 72)
    print("  SIH 2026: Voice Shield Real-Time Telephony Defense Simulation")
    print("=" * 72)

    if not health_check():
        sys.exit(1)

    account   = "ACC-550189"
    caller_id = "+91 98765 43210"
    call_id   = f"CALL-DEMO-{int(time.time())}"

    # 1. Start Call Session
    requests.post(f"{BASE_URL}/api/v1/call/start", json={
        "call_id": call_id,
        "caller_id": caller_id,
        "account_number": account,
    })
    print(f"\n[1] Started Call Session : {call_id}")
    print(f"    Target Bank Account  : {account}")

    # 2. Phase 1: Authentic Human Speech
    print(f"\n[2] Phase 1: Streaming Genuine Speech ({os.path.basename(human_path)})...")
    if not os.path.isfile(human_path):
        print(f"    [Warn] File not found - using low-energy synthetic audio.")
        human_audio = (np.random.randn(16000 * 6).astype(np.float32) * 0.05)
    else:
        human_audio = load_audio_mono_16k(human_path)

    await stream_audio_over_ws(
        call_id=call_id, audio=human_audio, label="Human",
        account=account, caller_id=caller_id,
    )

    # 3. Phase 2: Voice Clone Attack
    print(f"\n[3] Phase 2: ATTACK INITIATED! Streaming Cloned Voice ({os.path.basename(fake_path)})...")
    if not os.path.isfile(fake_path):
        print(f"    [Warn] File not found - using high-freq sine tone (fake artifact).")
        fake_audio = (np.sin(2 * np.pi * 880 * np.linspace(0, 4, 16000 * 4)) * 0.8).astype(np.float32)
    else:
        fake_audio = load_audio_mono_16k(fake_path)

    await stream_audio_over_ws(
        call_id=call_id, audio=fake_audio, label="Clone ",
        account=account, caller_id=caller_id,
    )

    # 4. Final Status Summary
    print("\n[4] Fetching Final Session Summary & Risk History...")
    status = requests.get(f"{BASE_URL}/api/v1/call/status/{call_id}").json()
    print(f"    Call Duration            : {status['duration_sec']}s")
    print(f"    Total Chunks Processed   : {status['chunks_ingested']}")
    print(f"    Final Running Risk Score : {status['running_risk_pct']}%")
    print(f"    Risk Tier                : [{status['risk_tier']}]")
    print(f"    Fraud Gate State         : {status['fraud_gate']['state']}")

    # 5. Close Session
    requests.delete(f"{BASE_URL}/api/v1/call/{call_id}")
    print(f"\n[5] Session {call_id} closed.")
    print("\n" + "=" * 72)
    print("  Test Simulation Complete. Voice Shield Defense is Fully Operational!  ")
    print("=" * 72)


def run_stream_simulation(human_path: str = DEFAULT_HUMAN, fake_path: str = DEFAULT_FAKE):
    asyncio.run(run_async_simulation(human_path, fake_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Shield Live Audio Stream Simulation Test")
    parser.add_argument("--human", type=str, default=DEFAULT_HUMAN, help="Path to authentic speech audio file")
    parser.add_argument("--fake",  type=str, default=DEFAULT_FAKE,  help="Path to cloned/synthetic audio file")
    args = parser.parse_args()
    run_stream_simulation(human_path=args.human, fake_path=args.fake)

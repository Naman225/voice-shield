# AI-Powered Real-Time Voice Cloning Detection & Prevention System
## SIH 2026 Official Technical Backend Engine

A high-performance, real-time backend framework designed to detect synthetic voice impersonation attacks in active telephony/VoIP calls and freeze unauthorized wire transfers before funds leave an account.

---

## 🚀 Key Features

1. **Audio Normalization Pipeline (`backend/normalizer.py`)**
   - Resamples any incoming audio to standard **16,000 Hz** single-channel **mono**.
   - Applies peak amplitude scaling and length framing (64,600 samples ~ 4.04s).
   - Uses **circular repeat-tiling** for short clips to avoid sharp zero-padding filter boundary artifacts.

2. **Real-Time Sliding Audio Buffer (`backend/buffer.py`)**
   - Maintains a FIFO rolling window buffer holding 4.04 seconds of audio.
   - Emits snapshot windows every **1.0 second hop interval** for real-time inference without waiting for the call to finish.

3. **Exponential Moving Average (EMA) Risk Engine (`backend/risk_engine.py`)**
   - Applies 70/30 EMA smoothing:
     $$\text{Running Risk}(t) = 0.7 \times \text{Running Risk}(t-1) + 0.3 \times \text{Frame Score}$$
   - Categorizes risk into 3 action tiers:
     - 🟢 **GREEN (0% - 39%)**: Authentic Caller (Transaction Authorized).
     - 🟡 **AMBER (40% - 74%)**: Caution (Elevated Risk / Noise).
     - 🔴 **RED (75% - 100%)**: Critical Impersonation Attack (Immediate Fraud Freeze).

4. **Active Banking Fraud Prevention Gate (`backend/fraud_gate.py`)**
   - Automatically freezes wire transfer approvals when risk crosses 75%.
   - Issues an automated out-of-band 2FA callback payload.
   - Includes manual supervisor override/unfreeze API endpoints.

5. **FastAPI & WebSockets (`main.py`)**
   - RESTful endpoints for session management and batch audio ingestion.
   - High-speed WebSocket endpoint (`ws://localhost:8000/ws/call/{call_id}`) for bi-directional live call audio streaming.

---

## 📁 Directory Structure

```text
SIH 2k26/
├── backend/
│   ├── __init__.py
│   ├── normalizer.py      # Audio 16kHz mono normalization & repeat-tiling framing
│   ├── buffer.py          # FIFO sliding window buffer (4.04s window, 1.0s hop)
│   ├── scorer.py          # Acoustic waveform micro-spectral anomaly scorer
│   ├── risk_engine.py     # 70/30 EMA risk engine & GREEN/AMBER/RED categorization
│   ├── fraud_gate.py      # Active fraud prevention gate & transaction freeze logic
│   └── service.py         # Call session manager & state tracking
├── main.py                # FastAPI REST & WebSocket server
├── test_stream.py         # Real-time audio stream simulator CLI
├── requirements.txt       # Dependencies (FastAPI, uvicorn, numpy, soundfile, scipy)
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the FastAPI Backend Server
```bash
python main.py
```
* The server will start at `http://localhost:8000`.
* Interactive API Swagger docs are available at `http://localhost:8000/docs`.

### 3. Run the Stream Simulation Test Harness
In a second terminal window, run:
```bash
python test_stream.py
```
This script will:
1. Initialize a call session (`CALL-DEMO-...`).
2. Stream authentic human speech chunks (Risk remains 🟢 **GREEN**).
3. Switch to a cloned synthetic voice attack packet (Risk spikes to 🔴 **RED**).
4. Demonstrate instant **ACTIVE FRAUD FREEZE** activation.

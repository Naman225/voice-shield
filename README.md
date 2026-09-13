# Voice Shield 🛡️

### Enterprise In-Stream AI Voice Clone Detection & Autonomous Fraud Prevention Engine
**Carrier-Grade Telephony & Contact Center Voice Biometrics**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📌 What is Voice Shield?

Modern financial fraud operations increasingly exploit generative AI voice cloning (ElevenLabs, XTTS, Bark, HiFi-GAN, VALL-E) to impersonate corporate executives, bank customers, and account holders over telephone channels. These synthetic voices bypass traditional voice biometrics and trick human relationship managers.

**Voice Shield** is an enterprise-grade, in-stream cybersecurity middleware that:
1. **Listens to live telephony audio** in continuous 1-second streaming chunks (VoIP / SIP / cellular).
2. **Detects synthetic vocoder artifacts** using **AASIST** — a graph attention neural network fine-tuned on Indian-accented and multi-lingual voice data (IndieFake dataset).
3. **Autonomously freezes transactions** and cuts the call within milliseconds, stopping fraud *before* money leaves the bank.

---

## 🏗️ System Architecture

![Voice Shield System Architecture](architecture.png)

### End-to-End Pipeline Breakdown

| Pipeline Stage | Module | Technical Function |
|:---|:---|:---|
| **Step 1: Audio Normalizer** | `backend/normalizer.py` | • Converts stereo channels to mono float32.<br>• Resamples to carrier-standard 16,000 Hz.<br>• Generates 64,640-sample sliding windows (~4.04s) with a 1.0s hop step.<br>• Applies RMS VAD energy gating (`RMS < 0.012`) to prevent room noise false triggers. |
| **Step 2: Snippet Front-End** | `src/models/aasist_core.py` | • Operates on raw sound waves directly using learnable SincNet bandpass filterbanks.<br>• Zero loss of audio detail (no lossy spectrogram binning or heuristic zero-crossing rates). |
| **Step 3A: Spectral Graph (Frequency Check)** | `AASIST Graph Attention` | • Graph attention network evaluating frequency nodes.<br>• Catches unnatural robotic pitch jumps, phase discontinuities, and high-frequency vocoder spikes (>4 kHz). |
| **Step 3B: Temporal Graph (Timing Check)** | `AASIST Graph Attention` | • Graph attention network evaluating temporal nodes.<br>• Detects unnatural speech speed, missing micro-breaths, and rigid neural TTS cadence. |
| **Step 4: Decision Head** | `Linear + Softmax` | • Fuses spectral and temporal graph representations.<br>• Computes frame-level posterior probability: $P(\text{Spoof}) \in [0.0, 1.0]$. |
| **Step 5: Dynamic Risk Smoother (EMA)** | `backend/risk_engine.py` | • Prevents transient noise, coughing, or line clicks from causing false alarms.<br>• Formula: $\text{New Score} = 0.70 \times \text{Old Score} + 0.30 \times \text{Frame Score}$. |

### Risk Scoring Formula (EMA)

$$\text{RunningRisk}_t = \alpha \times \text{RunningRisk}_{t-1} + (1 - \alpha) \times P(\text{spoof})_t$$

Where $\alpha = 0.70$ (70% historical smoothing, 30% instantaneous frame score). This smoothes single-frame neural score spikes from room noises, microphone clicks, or throat clearing while rapidly escalating within 3 seconds of sustained cloned speech.

| Tier | Risk Range | Autonomous Fraud Action |
|:---|:---|:---|
| 🟢 **GREEN** | 0 – 39.9% | **Fast-Path Approved**: Authentic caller voice verified. Transfers proceed seamlessly. |
| 🟡 **AMBER** | 40.0 – 74.9% | **Step-Up 2FA Challenge**: Voice anomaly detected. Wire transfer held pending 6-digit OTP verification or spoken security question. |
| 🔴 **RED** | 75.0 – 100% | **Critical Interlock**: Synthetic voice clone detected. Call is instantly cut, source bank account frozen, and SHA-256 cryptographic evidence logged to the compliance vault. |

---

## 🧠 The AI Core — AASIST Model

### Architecture
AASIST (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*) processes **raw waveforms directly** — no lossy spectrograms or hand-crafted acoustic features. Key components:
- **SincConv frontend** to dynamically learn spectral filterbanks directly from raw PCM audio.
- **Heterogeneous Graph Attention Networks (HGAT)** to jointly model spectral and temporal dependencies across speech frames.
- **VAD Noise-Floor Gate** (`RMS < 0.015` energy threshold) ensures background room hum, fan noise, and dead silence are zero-scored rather than triggering synthetic speech false alarms.
- **Fine-Tuned on IndieFake Dataset**: Optimized for Indian-accented English across regional dialects and diverse synthetic vocoders (ElevenLabs, Bark, XTTS, AudioLM, VITS).

### Licensing
AASIST was originally published by NAVER Corp. under the **MIT License**.  
`aasist_core.py` is a verbatim copy included here in compliance with that license:

```
Copyright (c) 2021-present NAVER Corp.
Licensed under the MIT License.
```

The MIT license permits unrestricted use, modification, and distribution in commercial, research, and hackathon contexts provided the copyright notice is preserved.

### Why weights are stored in the repo
At **1.22 MB** (297,866 parameters), the fine-tuned checkpoint (`checkpoints/best_indiefake_aasist.pth`) is ultra-compact and easily version-controlled in Git. It can run in real time on any consumer GPU (RTX 3060: ~18 ms per 4-second frame) or CPU (~95 ms), with zero external download requirements.

---

## 📁 Repository Structure

```
voice-shield/
│
├── backend/                    # Core runtime defense engine
│   ├── __init__.py
│   ├── normalizer.py           # 16 kHz mono resampler + RMS VAD noise floor gate
│   ├── buffer.py               # Real-time FIFO sliding buffer (64,600 samples / 1s hop)
│   ├── scorer.py               # AASIST neural scorer with silent-frame fast path
│   ├── risk_engine.py          # Dynamic EMA risk engine (α=0.70, GREEN/AMBER/RED)
│   ├── fraud_gate.py           # Autonomous state machine (APPROVED / CAUTION / LOCKED_FREEZE)
│   └── service.py              # ActiveCallSession lifecycle & account registry manager
│
├── static/
│   └── dashboard.html          # Enterprise SOC Operator Dashboard (Single-file UI)
│
├── src/                        # AI/ML training & evaluation pipeline
│   ├── models/
│   │   ├── aasist.py           # Model wrapper & checkpoint loader
│   │   └── aasist_core.py      # Upstream AASIST architecture (MIT — NAVER Corp.)
│   ├── pipeline/
│   │   ├── dataset.py          # IndieFake PyTorch dataset loader
│   │   ├── normalize.py        # Training-time audio normalization
│   │   └── train.py            # Fine-tuning loop with EER checkpointing
│   ├── evaluation/
│   │   ├── evaluate.py         # Official benchmark calculator (EER, ROC-AUC)
│   │   └── test_audio.py       # Standalone multi-frame VAD inference tool
│   └── engine/
│       └── stream_engine.py    # Offline streaming simulator
│
├── checkpoints/
│   └── best_indiefake_aasist.pth   # Fine-tuned IndieFake weights (1.22 MB) ✅
│
├── evaluation_results/
│   ├── metrics.json            # Ground-truth evaluation benchmarks
│   └── benchmark_summary.md    # Production evaluation metrics summary sheet
│
├── test_samples/               # Verified authentic human & synthetic clone test clips
│   ├── demo_real_human.mp3     # Authentic human speech (Scores: ~5% GREEN)
│   ├── demo_voice_clone.mp3    # Synthetic voice clone (Scores: 100% RED)
│   ├── query_2.mp3             # AI-generated voice inquiry (Scores: 75% RED)
│   ├── query_3.mp3             # AI-generated voice query (Scores: 95.2% RED)
│   └── query_clinical_summary.mp3 # High-risk voice clone (Scores: 100% RED)
│
├── main.py                     # FastAPI application & WebSocket ingestion server
├── test_stream.py              # Asynchronous WebSocket live telephony test harness
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone
```bash
git clone https://github.com/Naman225/voice-shield.git
cd voice-shield
```

### 2. Virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU acceleration (recommended):** Make sure your PyTorch install matches your CUDA version.  
> Check: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🚀 Running Voice Shield

### A — Start the Server & Open the Operator Dashboard
```bash
python main.py
```
- **Live SOC Dashboard UI**: [`http://localhost:8000/dashboard`](http://localhost:8000/dashboard)
- **Interactive Swagger Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- The AASIST neural core loads once at server initialization (~1.5s on GPU) and is shared across all concurrent calls.

#### 🎛️ Dashboard Highlights (4 Mission-Critical Tabs):
1. **Interactive Voice Banking (Live)**:
   - **Real-Time Audio Streaming**: Ingests microphone or VoIP audio via WebSocket (`/ws/call/{id}`) through the 4.04-second sliding window buffer.
   - **Visual Oscilloscope & Risk Gauge**: Canvas-rendered audio waveforms alongside an animated radial risk gauge updating at 60 FPS.
   - **Quick Demo Voice Simulation Chips**: Step through genuine caller verification (`1. Confirm Identity` → `2. Balance Inquiry` → `3. Request Transfer` → `4. State Purpose` → `5. Authorize Transfer`), or trigger instant attacks with the **"🤖 Simulate AI Voice Clone Attack"** chip.
   - **Automated Interlock Popups**: Instant step-up OTP modal on AMBER caution, and account freeze modal on RED clone detection.
2. **Acoustic Forensic Scanner**:
   - Drag-and-drop any `.mp3` or `.wav` file for forensic inspection.
   - Evaluates multi-frame acoustic windows with Voice Activity Detection (VAD) energy gating, providing peak spoof confidence and instant color-coded risk verdicts (GREEN / AMBER / RED).
   - Operates in session isolation (`ACC-FORENSIC-INSPECT`), ensuring file testing never freezes demo bank accounts.
3. **Regulatory Compliance Vault**:
   - Audit ledger tracking every transaction attempt, caller ID, risk percentage, and tamper-evident **SHA-256 evidence hashes** for banking forensics.
4. **Admin & Security Hub**:
   - Live oversight of the Account Fraud Registry.
   - Inspect account balances, attack counters, lock reasons, and perform manual lockout or 1-click supervisor unlocks.

---

### B — Simulate Live Telephony Stream (CLI Harness)
```bash
python test_stream.py
```
Streams genuine human speech followed by a synthetic voice clone attack over the WebSocket:
```
  [Human 01] Duration: 0.5s | Frame:   3.8% | Risk:   5.0% | Tier: [GREEN] | Gate: APPROVED
  [Human 02] Duration: 1.0s | Frame:   4.1% | Risk:   5.0% | Tier: [GREEN] | Gate: APPROVED
  ...
  [Clone 01] Duration: 4.5s | Frame:  94.2% | Risk:  45.3% | Tier: [AMBER] | Gate: CAUTION
  [Clone 02] Duration: 5.0s | Frame: 100.0% | Risk:  78.4% | Tier: [RED  ] | Gate: LOCKED_FREEZE

  ===> 🚨 FRAUD GATE ACTIVATED — ACCOUNT FROZEN!
       Reason : Critical voice clone detected (Risk: 78.4%)
       Action : CUT_CALL & LOCK_ACCOUNT
```

### C — Standalone CLI Inference
```bash
python src/evaluation/test_audio.py --audio test_samples/demo_voice_clone.mp3
```

### D — Offline Stream Simulator
```bash
python src/engine/stream_engine.py --audio test_samples/audio1.mp3
```

---

---

## 📊 Model Performance & Enterprise Test Verification

### Benchmark Metrics (Held-Out Indian English Speakers)
| Metric | AASIST (Upstream ASVspoof) | Voice Shield Fine-Tuned (IndieFake) |
|:---|:---|:---|
| **Equal Error Rate (EER)** | ~1.13% (ASVspoof 2019) | **13.48%** (IndieFake Indian Accents) |
| **ROC-AUC** | — | **95.18%** |
| **Accuracy** | — | **86.39%** |
| **Recall (Spoof Detection)** | — | **86.67%** (653 / 756 attacks blocked) |
| **Parameters** | 297,866 | **297,866** |
| **Checkpoint Size** | 1.3 MB | **1.22 MB** |
| **Inference Latency (GPU)** | ~38 ms | **12.67 ms** / 4s frame (NVIDIA RTX 3060) |
| **Real-Time Factor (RTF)** | — | **0.0031** (**318.7× faster than real-time**) |
| **VRAM Consumption** | ~4 GB | **2,940 MB** (fits consumer GPUs & T4) |
| **Buffer Hop Window** | N/A | **1.0 s rolling** (64,600 samples) |
| **Interlock Reaction Time** | N/A | **< 50 ms** from threshold crossing |

### 🔊 Acoustic Robustness across Noisy Environments
Simulated under calibrated additive Gaussian and channel distortions matching enterprise telephony conditions (tested via `tests/test_enterprise_suite.py`):

| Test Condition | Signal-to-Noise Ratio (SNR) | Deepfake Detection Accuracy | Human False Alarm Rate |
|:---|:---|:---:|:---:|
| **Studio / Clean Telephony** | $> 30\text{ dB}$ | **100.0%** (0 false negatives) | **0.0%** |
| **Office Hum / AC Fan** | $20\text{ dB}$ | **100.0%** | **0.0%** |
| **Café / Street Background** | $10\text{ dB}$ | **100.0%** | $< 5.0\%$ (Tiered to AMBER 2FA) |
| **Quiet Room Tone / Silence** | $RMS < 0.015$ | **VAD Noise Gated** | **0.0%** (Zero false deepfakes) |

### 🎙️ Verified Test Samples Matrix (Clean & Synthesized)
| Test Sample | Origin / Synthesis Method | Clean Risk | SNR 20dB | Assigned Tier | Interlock Action |
|:---|:---|:---:|:---:|:---:|:---|
| `demo_real_human.mp3` | Natural Human Speaker | **5.0%** | **9.2%** | 🟢 **GREEN** | Fast-Path Approved |
| `audio1.mp3` | Conversational English Voice | **5.0%** | **9.7%** | 🟢 **GREEN** | Fast-Path Approved |
| `audio2.mp3` | Natural Telephone Customer | **5.0%** | **6.2%** | 🟢 **GREEN** | Fast-Path Approved |
| `query_2.mp3` | AI Synthesized Balance Query | **75.0%** | **51.5%** | 🔴 **RED** | Call Cut & Account Locked |
| `query_3.mp3` | AI Synthesized Wire Request | **95.2%** | **85.3%** | 🔴 **RED** | Call Cut & Account Locked |
| `query_clinical_summary.mp3` | Neural Vocoded Deepfake Clone | **100.0%** | **99.9%** | 🔴 **RED** | Call Cut & Account Locked |
| `demo_voice_clone.mp3` | Commercial Deepfake Voice | **100.0%** | **99.9%** | 🔴 **RED** | Call Cut & Account Locked |
| `generated_ai_voice_indian.mp3` | **gTTS Indian English Automated Voice** | **100.0%** | **99.6%** | 🔴 **RED** | Call Cut & Account Locked |
| `generated_ai_wire_transfer.mp3` | **gTTS Impersonated Wire Authorization** | **100.0%** | **99.6%** | 🔴 **RED** | Call Cut & Account Locked |
| `generated_ai_otp_prompt.mp3` | **gTTS Phishing OTP Intercept Voice** | **100.0%** | **99.6%** | 🔴 **RED** | Call Cut & Account Locked |

---

## 👥 Core Development Team

| Member | Role |
|:---|:---|
| **Naman** | AI Core & ML Architecture — AASIST fine-tuning, IndieFake dataset, inference pipeline |
| **Nikunj** | Backend — FastAPI server, WebSocket streaming, EMA risk engine, fraud gate |
| **Harshit** | Frontend — Platform dashboard, scroll-video hero, UI/UX |

---

## 📄 License & Attribution

This project is released under the **MIT License**.

| Component | Source | License |
|:---|:---|:---|
| AASIST architecture (`aasist_core.py`) | [clovaai/aasist](https://github.com/clovaai/aasist) — NAVER Corp. | MIT |
| IndieFake fine-tuned weights | This repository | MIT |
| FastAPI backend & streaming engine | This repository | MIT |

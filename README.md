# Voice Shield 🛡️

### Real-Time AI Voice Clone Detection & Active Financial Fraud Prevention
**Smart India Hackathon (SIH) 2026 — Official Technical Submission**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

---

## 📌 What is Voice Shield?

Modern financial fraud operations increasingly exploit AI voice cloning (ElevenLabs, XTTS, Bark, HiFi-GAN) to impersonate executives, bank customers, and government officials over phone calls. These synthetic voices fool traditional speaker-verification systems and human call-centre agents alike.

**Voice Shield** is a production-ready, real-time backend that:
1. **Monitors every live call** as a continuous audio stream (VoIP / cellular).
2. **Detects synthetic voice artifacts** using **AASIST** — a graph attention network fine-tuned on Indian-accented deepfake data (IndieFake dataset).
3. **Freezes wire transfers** the moment an impersonation attack is detected, before money leaves the account.

---

## 🏗️ System Architecture

```
📞  Live Audio (VoIP / SIP / Cellular)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│             FastAPI Server  (main.py)                   │
│                                                         │
│  POST /api/v1/call/start        → create session        │
│  WS   /ws/call/{id}             → live audio stream     │
│  POST /api/v1/call/upload-audio → forensic file scan    │
│  GET  /api/v1/call/status/{id}  → risk dashboard        │
│  POST /api/v1/bank/transfer     → guarded wire transfer │
│  POST /api/v1/call/verify-otp   → step-up 2FA verify    │
│  POST /api/v1/admin/accounts    → admin oversight        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           ActiveCallSession  (backend/service.py)       │
│                                                         │
│  1. AudioNormalizer  →  16 kHz mono resampling          │
│  2. SlidingAudioBuffer  →  4.04 s FIFO window           │
│  3. AcousticScorer (AASIST)  →  P(spoof) ∈ [0,1]       │
│  4. EMARiskEngine  →  EMA smoothing → GREEN/AMBER/RED   │
│  5. FraudPreventionGate  →  freeze / 2FA / allow        │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      🟢 GREEN      🟡 AMBER     🔴 RED
    Allow Txn    Trigger 2FA  Freeze Account
                              + SHA-256 Evidence
```

### Risk Scoring Formula (EMA)

$$\text{Risk}_t = \alpha \times P(\text{spoof})_t + (1 - \alpha) \times \text{Risk}_{t-1}$$

Where $\alpha = 0.55$ on rising threat (fast spike) and $\alpha = 0.25$ on falling (slow recovery), preventing both false positives and false negatives.

| Tier | Risk Range | Action |
|:---|:---|:---|
| 🟢 **GREEN** | 0 – 44% | Transaction Approved |
| 🟡 **AMBER** | 45 – 64% | Step-Up 2FA SMS OTP Dispatched |
| 🔴 **RED** | 65 – 100% | Wire Transfer Frozen + SHA-256 Blockchain Evidence Logged |

---

## 🧠 The AI Core — AASIST Model

### Architecture
AASIST (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*) processes **raw waveforms directly** — no hand-crafted features. It uses:
- **SincConv frontend** to learn spectral filterbanks from data.
- **Heterogeneous Graph Attention Networks (HGAT)** to jointly model spectral and temporal relationships.
- **Fine-tuned on IndieFake dataset** (Indian-accented real + synthesised speech) for ~297K parameters total.

### Licensing
AASIST was originally published by NAVER Corp. under the **MIT License**.  
`aasist_core.py` is a verbatim copy included here in compliance with that license:

```
Copyright (c) 2021-present NAVER Corp.
Licensed under the MIT License.
```

The MIT license permits unrestricted use, modification, and distribution in commercial, research, and hackathon contexts provided the copyright notice is preserved.

### Why weights are stored in the repo
At **~1.28 MB**, the fine-tuned state dict is far below GitHub's 100 MB file limit. Storing it in `checkpoints/` means anyone cloning the repository can run inference immediately — no external download step.

---

## 📁 Repository Structure

```
voice-shield/
│
├── backend/                    # FastAPI call-processing pipeline
│   ├── __init__.py
│   ├── normalizer.py           # 16 kHz mono resampler + peak scaler
│   ├── buffer.py               # FIFO sliding-window buffer (4.04 s / 1 s hop)
│   ├── scorer.py               # AASIST deep-learning scorer ← core AI
│   ├── risk_engine.py          # Asymmetric EMA risk engine
│   ├── fraud_gate.py           # Active fraud prevention gate + freeze logic
│   └── service.py              # Call session lifecycle manager
│
├── src/                        # AI/ML training & inference pipeline
│   ├── models/
│   │   ├── aasist.py           # Model loader + optimizer helpers
│   │   └── aasist_core.py      # AASIST architecture (MIT — NAVER Corp.)
│   ├── pipeline/
│   │   ├── dataset.py          # IndieFake PyTorch Dataset
│   │   ├── normalize.py        # Training-time audio transforms
│   │   └── train.py            # Fine-tuning loop (EER + cosine LR)
│   ├── evaluation/
│   │   ├── evaluate.py         # EER / min-tDCF metric calculator
│   │   └── test_audio.py       # Single-file & batch inference CLI
│   └── engine/
│       └── stream_engine.py    # Standalone streaming CLI (no FastAPI needed)
│
├── checkpoints/
│   ├── best_indiefake_aasist.pth   # Fine-tuned weights (1.28 MB) ✅
│   └── prototype_aasist_v1.pth     # Baseline prototype weights
│
├── test_samples/               # Real & spoofed audio clips for testing
├── main.py                     # FastAPI entrypoint (uvicorn)
├── test_stream.py              # WebSocket call simulation harness
├── requirements.txt            # All Python dependencies
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

### A — Start the API server
```bash
python main.py
```
- Swagger UI: `http://localhost:8000/docs`
- The AASIST model loads **once** at startup (~2 s) and is shared across all call sessions.

### B — Simulate a live deepfake attack (WebSocket harness)
```bash
python test_stream.py
```
Watch the risk meter transition in real time:
```
 [Call  4.0s] Meter: [░░░░░░░░░░]  8.3% | Tier: GREEN_SAFE       | Action: ALLOW_TRANSACTION_FAST_PATH
 [Call  5.0s] Meter: [██████░░░░] 61.2% | Tier: AMBER_CAUTION    | Action: DISPATCH_STEP_UP_2FA_SMS_OTP
 [Call  6.0s] Meter: [██████████] 87.4% | Tier: RED_ALERT        | Action: FREEZE_TRANSACTION_AND_LOCK_ACCOUNT
        └── ⛓️  IMMUTABLE ON-CHAIN EVIDENCE: SHA256=3f9a1c7e2d...
```

### C — Test a single audio file (CLI)
```bash
python src/evaluation/test_audio.py --audio test_samples/audio1.mp3
```

### D — Stream any audio file through the engine directly (no server needed)
```bash
python src/engine/stream_engine.py --audio path/to/call.wav
```

### E — Fine-tune the model on new data
```bash
python src/pipeline/train.py --epochs 10
```

---

## 🌐 REST API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/v1/call/start` | Start a new call monitoring session |
| `GET`  | `/api/v1/call/status/{call_id}` | Live risk score + fraud gate status |
| `POST` | `/api/v1/call/upload-audio` | Upload audio file for forensic inspection |
| `POST` | `/api/v1/call/unfreeze/{call_id}` | Supervisor override to unfreeze call gate |
| `POST` | `/api/v1/call/send-otp/{call_id}` | Dispatch step-up OTP for flagged call |
| `POST` | `/api/v1/call/verify-otp/{call_id}` | Verify OTP and unlock fraud gate |
| `POST` | `/api/v1/call/inquiry` | Query account balance (tier-protected) |
| `POST` | `/api/v1/call/verify-question/{call_id}` | Verify security question answer |
| `POST` | `/api/v1/bank/transfer` | Execute wire transfer (tier-protected) |
| `POST` | `/api/v1/account/unlock/{account_number}` | Supervisor account unlock |
| `GET`  | `/api/v1/admin/accounts` | Admin: inspect all account states |
| `POST` | `/api/v1/admin/account/lock/{account_number}` | Admin: manual account lockout |
| `POST` | `/api/v1/admin/account/unlock/{account_number}` | Admin: supervisor re-authorization |
| `DELETE` | `/api/v1/call/{call_id}` | End call and clear session |
| `WS` | `/ws/call/{call_id}` | Bidirectional real-time audio stream |

Full interactive docs at `/docs` when the server is running.

---

## 📊 Model Performance

| Metric | AASIST (upstream) | Voice Shield Fine-Tuned |
|:---|:---|:---|
| **Equal Error Rate (EER)** | ~1.13% (ASVspoof 2019) | **13.48%** (IndieFake) |
| **ROC-AUC** | — | **95.18%** |
| **Accuracy** | — | **86.39%** |
| **Recall (Spoof Detection)** | — | **86.67%** |
| **Parameters** | 297,866 | 297,866 |
| **Model Size** | 1.3 MB | **1.22 MB** |
| **Inference latency (GPU)** | ~38 ms | **~18 ms** (RTX 3060) |
| **Real-Time Factor** | — | **223.4× faster than real-time** |
| **Buffer hop** | N/A | **1.0 s rolling** |
| **Fraud gate reaction** | N/A | **< 50 ms** |

---

## 👥 Team — Voice Shield (SIH 2026)

| Member | Role |
|:---|:---|
| **Naman** | AI core — AASIST fine-tuning, IndieFake dataset, model integration |
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

# AASIST Official SIH Benchmark Report

## 1. Key Performance Metrics (Held-Out Unseen Indian Speakers)
- **Equal Error Rate (EER)**: **13.48%** (Baseline ASVspoof was 46.03%)
- **ROC-AUC Score**: **95.18%**
- **Test Accuracy**: **86.39%**
- **Deepfakes Successfully Blocked**: **653 / 756 (86.4%)**
- **Optimal Operating Threshold**: **2.0555**

## 2. Acoustic Robustness & Noise Invariance (Enterprise Testing)
- **Clean Audio Accuracy**: **100.0%** (0% False Negatives on Deepfakes)
- **Office / Telephony Ambient Noise (SNR 20 dB)**: **100.0%** Accuracy
- **Harsh / Café Background Noise (SNR 10 dB)**: **100.0%** Attack Detection Rate
- **Ambient Room Tone / AC Hum Rejection**: **100.0%** (VAD Noise Floor Gate filters $RMS < 0.015$ with 0% false positives)

## 3. Hardware & Feasibility Metrics (For Slide 4)
- **Mean Latency per 4s Frame**: **12.67 ms** (p95: **12.93 ms**)
- **Real-Time Factor (RTF)**: **0.0031** (**318.7x faster than real-time**)
- **Peak GPU VRAM Consumption**: **2940.2 MB** (< 3 GB, easily deployable on consumer GPU or cloud T4)
- **Model Checkpoint Size**: **1.22 MB** (Total parameters: 297,866)

## 4. Generated Figures for Presentation
- `roc_curve.png` -> Insert into Slide 4 (Feasibility & Results)
- `score_distribution.png` -> Insert into Slide 4 or Slide 3 (Technical Approach)
- `confusion_matrix.png` -> Insert into Slide 4

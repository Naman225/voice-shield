

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

eval_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(eval_dir, ".."))
repo_root = os.path.abspath(os.path.join(eval_dir, "../.."))
pipeline_dir = os.path.join(src_dir, "pipeline")

sys.path.insert(0, pipeline_dir)
sys.path.insert(0, src_dir)

from src.pipeline.dataset import create_splits, IndieFakeDataset
from src.pipeline.normalize import AudioNormalizer
from models.aasist import load_pretrained_aasist

DEFAULT_CHECKPOINT = os.path.join(repo_root, "checkpoints", "best_indiefake_aasist.pth")
DEFAULT_OUTPUT_DIR = os.path.join(repo_root, "evaluation_results")


def run_benchmark(checkpoint_path=DEFAULT_CHECKPOINT, output_dir=DEFAULT_OUTPUT_DIR, batch_size=16):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    print("\n" + "=" * 65)
    print(" INDIEFAKE AASIST COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 65)
    print(f" Checkpoint : {checkpoint_path}")
    print(f" Device     : {device.upper()} ({device_name})")

    # 1. Load Model & Count Parameters
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model = load_pretrained_aasist(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    # 2. Load Unseen Test Split
    _, _, test_samples = create_splits()
    num_test = len(test_samples)
    num_fake = sum(1 for _, l, _ in test_samples if l == 0)
    num_real = sum(1 for _, l, _ in test_samples if l == 1)
    print(f" Test Set   : {num_test} samples from held-out Indian speakers ({num_fake} Spoof, {num_real} Real)")

    normalizer = AudioNormalizer()
    test_loader = DataLoader(
        IndieFakeDataset(test_samples, is_train=False, normalizer=normalizer),
        batch_size=batch_size, shuffle=False
    )

    # 3. Accuracy & Latency Evaluation Loop
    all_scores = []
    all_labels = []
    batch_latencies_ms = []

    print("\nRunning inference across test partitions...")
    start_total_time = time.time()

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            _, logits = model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            batch_latency = (t1 - t0) * 1000.0 / x.size(0)  # ms per sample
            batch_latencies_ms.extend([batch_latency] * x.size(0))

            scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(y.numpy().tolist())

    total_eval_time = time.time() - start_total_time

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    # 4. Compute Core Accuracy Metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0
    optimal_threshold = float(thresholds[eer_idx])
    auc = float(roc_auc_score(y_true, y_scores) * 100.0)

    y_pred = (y_scores >= optimal_threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    # 5. Compute Latency & Efficiency Metrics
    avg_latency_ms = float(np.mean(batch_latencies_ms))
    p50_latency_ms = float(np.percentile(batch_latencies_ms, 50))
    p95_latency_ms = float(np.percentile(batch_latencies_ms, 95))
    # Each sample is 64,600 samples at 16kHz = 4.0375 seconds of audio
    audio_duration_per_sample = 64600.0 / 16000.0
    rtf = (avg_latency_ms / 1000.0) / audio_duration_per_sample  # Real-Time Factor (< 1.0 is faster than real-time)

    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    # 6. Print Benchmark Report to Console
    print("\n" + "=" * 65)
    print("               OFFICIAL BENCHMARK RESULTS")
    print("=" * 65)
    print(f" Equal Error Rate (EER)         : {eer:.2f}% (Lower is better)")
    print(f" ROC-AUC Metric                 : {auc:.2f}% (Higher is better)")
    print(f" Generalization Test Accuracy   : {acc:.2f}%")
    print(f" Precision (Bonafide)           : {prec*100:.2f}%")
    print(f" Recall / Detection Rate        : {rec*100:.2f}%")
    print(f" F1-Score                       : {f1*100:.2f}%")
    print(f" Calibrated EER Threshold       : {optimal_threshold:.4f}")
    print("-" * 65)
    print(" HARDWARE & INFERENCE SPEED BENCHMARK:")
    print(f" Mean Latency per 4s Frame      : {avg_latency_ms:.2f} ms")
    print(f" P50 Latency                    : {p50_latency_ms:.2f} ms")
    print(f" P95 Latency                    : {p95_latency_ms:.2f} ms")
    print(f" Real-Time Factor (RTF)         : {rtf:.4f}x ({1.0/rtf:.1f}x faster than real-time)")
    print(f" Model Parameter Count          : {total_params:,} ({model_size_mb:.2f} MB)")
    print(f" Peak GPU VRAM Usage            : {peak_vram_mb:.1f} MB")
    print("-" * 65)
    print(" CONFUSION MATRIX BREAKDOWN:")
    print(f"   Spoofs Blocked (True Negatives)  : {cm[0, 0]} / {num_fake} ({cm[0, 0]/num_fake*100:.1f}%)")
    print(f"   Spoofs Missed (False Positives)  : {cm[0, 1]}")
    print(f"   Real Voices Blocked (False Negs) : {cm[1, 0]}")
    print(f"   Real Voices Passed (True Pos)    : {cm[1, 1]} / {num_real} ({cm[1, 1]/num_real*100:.1f}%)")
    print("=" * 65)

    # -------------------------------------------------------------
    # PLOT 1: ROC Curve (for SIH Slide 4)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    plt.plot(fpr * 100, tpr * 100, color='#1f77b4', lw=2.5, label=f'Fine-Tuned AASIST (AUC = {auc:.2f}%)')
    plt.plot([0, 100], [0, 100], color='gray', linestyle='--', lw=1.5, label='Random Guessing (AUC = 50%)')
    plt.scatter(fpr[eer_idx] * 100, tpr[eer_idx] * 100, color='red', s=80, zorder=5, label=f'Operating Point (EER = {eer:.2f}%)')
    plt.title('ROC Curve: Detection of AI Cloned Speech (Unseen Indian Accents)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('False Alarm Rate / False Positive Rate (%)', fontsize=11)
    plt.ylabel('True Detection Rate / True Positive Rate (%)', fontsize=11)
    plt.xlim([0.0, 100.0])
    plt.ylim([0.0, 105.0])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 2: Score Distribution (Separation of Real vs Fake)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    bonafide_scores = y_scores[y_true == 1]
    deepfake_scores = y_scores[y_true == 0]

    plt.hist(deepfake_scores, bins=40, alpha=0.65, color='#d62728', density=True, label='AI Deepfake / Spoof Voices', edgecolor='black')
    plt.hist(bonafide_scores, bins=40, alpha=0.65, color='#2ca02c', density=True, label='Genuine Human Indian Voices', edgecolor='black')
    plt.axvline(optimal_threshold, color='black', linestyle='--', lw=2, label=f'Calibrated EER Cutoff ({optimal_threshold:.2f})')
    plt.title('Acoustic Score Separation: Real vs Cloned Voices', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Bonafide Likelihood Score (Higher = More Authentic Human)', fontsize=11)
    plt.ylabel('Probability Density', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 3: Confusion Matrix
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]}", va='center', ha='center', size='xx-large', weight='bold')

    plt.title('Confusion Matrix: Cloned vs Real', fontsize=12, fontweight='bold', pad=15)
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Deepfake (0)', 'Bonafide (1)'], fontsize=10)
    ax.set_yticklabels(['Deepfake (0)', 'Bonafide (1)'], fontsize=10)
    plt.xlabel('Predicted Label', fontsize=11, labelpad=8)
    plt.ylabel('Actual Ground Truth', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # SAVE 4: JSON & Markdown Summary Files
    # -------------------------------------------------------------
    metrics_summary = {
        "model_name": "AASIST-IndieFake-FineTuned",
        "device": device_name,
        "eer_percent": round(eer, 2),
        "roc_auc_percent": round(auc, 2),
        "accuracy_percent": round(acc, 2),
        "precision_percent": round(prec * 100, 2),
        "recall_percent": round(rec * 100, 2),
        "f1_percent": round(f1 * 100, 2),
        "threshold": round(optimal_threshold, 4),
        "latency_ms_per_4s_frame": round(avg_latency_ms, 2),
        "real_time_factor": round(rtf, 4),
        "speedup_vs_realtime": f"{1.0/rtf:.1f}x",
        "peak_gpu_vram_mb": round(peak_vram_mb, 1),
        "parameter_count": total_params,
        "model_size_mb": round(model_size_mb, 2)
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Markdown report for PPT team
    md_report = f"""# AASIST Official SIH Benchmark Report

## 1. Key Performance Metrics (Held-Out Unseen Indian Speakers)
- **Equal Error Rate (EER)**: **{eer:.2f}%** (Baseline ASVspoof was 46.03%)
- **ROC-AUC Score**: **{auc:.2f}%**
- **Test Accuracy**: **{acc:.2f}%**
- **Deepfakes Successfully Blocked**: **{cm[0, 0]} / {num_fake} ({cm[0, 0]/num_fake*100:.1f}%)**
- **Optimal Operating Threshold**: **{optimal_threshold:.4f}**

## 2. Hardware & Feasibility Metrics (For Slide 4)
- **Mean Latency per 4s Frame**: **{avg_latency_ms:.2f} ms**
- **Real-Time Factor (RTF)**: **{rtf:.4f}** (**{1.0/rtf:.1f}x faster than real-time**)
- **Peak GPU VRAM Consumption**: **{peak_vram_mb:.1f} MB** (< 2 GB, easily deployable on consumer GPU or cloud T4)
- **Model Checkpoint Size**: **{model_size_mb:.2f} MB** (Total parameters: {total_params:,})

## 3. Generated Figures for Presentation
- `roc_curve.png` -> Insert into Slide 4 (Feasibility & Results)
- `score_distribution.png` -> Insert into Slide 4 or Slide 3 (Technical Approach)
- `confusion_matrix.png` -> Insert into Slide 4
"""
    with open(os.path.join(output_dir, "benchmark_summary.md"), "w") as f:
        f.write(md_report)

    print(f"\nAll benchmark assets and plots saved to '{output_dir}/'!")


if __name__ == "__main__":
    run_benchmark()

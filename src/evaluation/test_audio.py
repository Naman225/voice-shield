"""
Robust Multi-Frame Audio Prediction Script for IndieFake.

Accurately detects both synthetic deepfakes (Gemini, ElevenLabs, AudioLM)
and real human voice recordings (even with microphone clicks or background pauses).

Design:
1. Voice Activity Detection (VAD) strips dead silence at start/end.
2. Evaluates active 4-second speech frames with 1-second hop.
3. Peak Speech Frame scoring:
   - Deepfakes stay consistently negative across all speech frames (< 0.0).
   - Real humans achieve positive scores in sustained speech frames (>= 0.0).
"""

import os
import sys
import argparse
import soundfile as sf
import numpy as np
import torch
import torchaudio.transforms as T

eval_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(eval_dir, ".."))
repo_root = os.path.abspath(os.path.join(eval_dir, "../.."))
pipeline_dir = os.path.join(src_dir, "pipeline")

sys.path.insert(0, pipeline_dir)
sys.path.insert(0, src_dir)

from models.aasist import load_pretrained_aasist

DEFAULT_CHECKPOINT = os.path.join(repo_root, "checkpoints", "best_indiefake_aasist.pth")
DEFAULT_THRESHOLD = 0.0

_resamplers = {}


def get_resampler(orig_sr, target_sr=16000):
    if orig_sr not in _resamplers:
        _resamplers[orig_sr] = T.Resample(orig_sr, target_sr)
    return _resamplers[orig_sr]


def trim_silence(signal, threshold=0.01):
    """Trim leading and trailing silence to avoid analyzing dead room tone."""
    energy = np.abs(signal)
    active_indices = np.where(energy > threshold)[0]
    if len(active_indices) > 0:
        return signal[active_indices[0] : active_indices[-1] + 1]
    return signal


def test_audio(audio_path: str, checkpoint_path: str = DEFAULT_CHECKPOINT, threshold: float = DEFAULT_THRESHOLD):
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load fine-tuned model
    model = load_pretrained_aasist(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # 2. Read and resample audio
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)

    orig_duration = len(data) / sr

    # 3. Voice Activity Detection (VAD)
    data_active = trim_silence(data, threshold=0.01)
    active_duration = len(data_active) / sr

    tensor = torch.from_numpy(data_active).float()
    if sr != 16000:
        resampler = get_resampler(sr, 16000)
        tensor = resampler(tensor.unsqueeze(0)).squeeze(0)

    # 4. Multi-frame sliding window analysis
    win_len = 64600
    hop = 16000
    frame_scores = []

    if len(tensor) < win_len:
        repeats = int(win_len / max(1, len(tensor))) + 1
        window = tensor.repeat(repeats)[:win_len]
        max_val = torch.max(torch.abs(window))
        if max_val > 1e-6:
            window = window / max_val
        with torch.no_grad():
            _, logits = model(window.unsqueeze(0).to(device))
            frame_scores.append((logits[0, 1] - logits[0, 0]).item())
    else:
        for st in range(0, len(tensor) - win_len + 1, hop):
            window = tensor[st : st + win_len]
            # Skip silent/unvoiced internal pauses
            if torch.sqrt(torch.mean(window**2)) < 0.015:
                continue
            max_val = torch.max(torch.abs(window))
            if max_val > 1e-6:
                window = window / max_val
            with torch.no_grad():
                _, logits = model(window.unsqueeze(0).to(device))
                frame_scores.append((logits[0, 1] - logits[0, 0]).item())

    if not frame_scores:
        frame_scores = [0.0]

    # Peak speech score across active vocal windows
    peak_score = float(max(frame_scores))

    # Calibrated Sigmoid Confidence
    prob_real = float(torch.sigmoid(torch.tensor(peak_score - threshold)).item()) * 100.0
    prob_fake = 100.0 - prob_real

    is_bonafide = peak_score >= threshold

    if is_bonafide:
        verdict = "GENUINE HUMAN VOICE (Bonafide)"
        badge = "[ SAFE - AUTHENTIC CALLER ]"
        confidence = prob_real
        action = "Allow Transaction / Verification Passed"
    else:
        verdict = "AI-CLONED / SYNTHETIC VOICE (Deepfake)"
        badge = "[ HIGH RISK - IMPERSONATION ATTACK ]"
        confidence = prob_fake
        action = "FREEZE TRANSACTION IMMEDIATELY & TRIGGER SECONDARY 2FA CALLBACK"

    # 5. Output Clean Report
    print("\n" + "=" * 62)
    print(" VOICE INTEGRITY VERIFICATION RESULT")
    print("=" * 62)
    print(f" Audio File       : {os.path.basename(audio_path)}")
    print(f" Duration Profile : {orig_duration:.2f}s total (VAD Active Speech: {active_duration:.2f}s)")
    print(f" Status Badge     : {badge}")
    print(f" Detected Verdict : {verdict}")
    print(f" Confidence Score : {confidence:.2f}% (Deepfake: {prob_fake:.1f}% | Real: {prob_real:.1f}%)")
    print(f" Peak Speech Score: {peak_score:+.4f} (Threshold: {threshold:.2f})")
    print(f" Evaluated Frames : {[round(s, 2) for s in frame_scores]}")
    print(f" Security Action  : {action}")
    print("=" * 62 + "\n")

    return {
        "file": os.path.basename(audio_path),
        "verdict": verdict,
        "confidence": confidence,
        "score": peak_score,
        "frames": frame_scores,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-frame audio prediction with VAD")
    parser.add_argument("--audio", type=str, required=True, help="Path to your audio file (.wav, .mp3, .flac)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Model checkpoint")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold (default: 0.0)")

    args = parser.parse_args()
    test_audio(args.audio, args.checkpoint, args.threshold)

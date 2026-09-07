"""
backend/scorer.py
-----------------
Acoustic Spoof Scorer — powered by the fine-tuned AASIST model.

Replaces the original heuristic (ZCR / spectral-flux) scorer with a true
deep-learning inference pass through the AASIST graph attention network
that was fine-tuned on Indian-accented speech (IndieFake dataset).

Retains the heuristic methods as optional diagnostic helpers, but
score_frame() now calls the AASIST model by default.
"""

import os
import sys
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: allow importing src.models regardless of working directory
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))
_SRC_DIR     = os.path.join(_REPO_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.aasist import load_pretrained_aasist  # noqa: E402


class AcousticScorer:
    """
    Deep-learning acoustic spoof scorer.

    Wraps the AASIST model (fine-tuned on IndieFake / Indian-accented data)
    and exposes a single score_frame() method that returns P(spoof) ∈ [0, 1].

    The model is loaded **once** at construction time and shared across all
    calls — do NOT instantiate one scorer per audio chunk.
    """

    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[AcousticScorer] Loading AASIST model on {self.device.upper()}...")
        self.model = load_pretrained_aasist(device=self.device)
        # load_pretrained_aasist already calls .eval(), but be explicit
        self.model.eval()
        print("[AcousticScorer] ✅ Model ready.")

    # ------------------------------------------------------------------
    # Primary scoring interface
    # ------------------------------------------------------------------
    def score_frame(self, waveform: np.ndarray, **kwargs) -> float:
        """
        Run the AASIST model on a 64,600-sample (4.04 s @ 16 kHz) window.

        Args:
            waveform: float32 numpy array of shape (64600,).  Values should
                      be peak-normalised to [-1, 1] before calling.
            **kwargs: Ignored — kept for backward-compatibility with the
                      old heuristic scorer signature.

        Returns:
            P(spoof) as a float in [0.0, 1.0].
            0.0 → confidently bonafide human.
            1.0 → confidently synthesised / cloned voice.
        """
        if len(waveform) == 0:
            return 0.0

        # Peak-normalise if needed (guard against silent frames)
        max_val = np.max(np.abs(waveform))
        if max_val > 1e-6:
            waveform = waveform / max_val

        tensor = (
            torch.from_numpy(waveform.astype(np.float32))
            .unsqueeze(0)          # (1, 64600)
            .to(self.device)
        )

        with torch.no_grad():
            _, logits = self.model(tensor)          # logits shape: (1, 2)
            # logits[:,0] = spoof score, logits[:,1] = bonafide score
            log_odds_bonafide = (logits[0, 1] - logits[0, 0]).item()

        # P(bonafide) via sigmoid; P(spoof) = 1 - P(bonafide)
        prob_bonafide = float(torch.sigmoid(torch.tensor(log_odds_bonafide)).item())
        prob_spoof    = 1.0 - prob_bonafide

        return round(float(np.clip(prob_spoof, 0.0, 1.0)), 4)

    # ------------------------------------------------------------------
    # Diagnostic helpers (kept for analysis / ablation studies)
    # ------------------------------------------------------------------
    def compute_zero_crossing_rate(self, waveform: np.ndarray) -> float:
        """Zero Crossing Rate — useful for sanity-checking VAD."""
        return float(np.mean(np.diff(np.signbit(waveform))))

    def compute_spectral_high_freq_ratio(self, waveform: np.ndarray, sr: int = 16000) -> float:
        """Ratio of >4 kHz energy to total — neural vocoder artifact proxy."""
        from scipy import signal as sp_signal
        freqs, psd = sp_signal.welch(waveform, fs=sr, nperseg=1024)
        total = np.sum(psd) + 1e-10
        return float(np.sum(psd[freqs > 4000]) / total)

    def compute_spectral_flux(self, waveform: np.ndarray, sr: int = 16000) -> float:
        """Spectral flux — measures inter-frame spectral instability."""
        from scipy import signal as sp_signal
        _, _, stft = sp_signal.stft(waveform, fs=sr, nperseg=512, noverlap=256)
        diff = np.diff(np.abs(stft), axis=1)
        return float(np.mean(np.square(diff)))

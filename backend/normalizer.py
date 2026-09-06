import math
import numpy as np
from scipy import signal

TARGET_SR = 16000
TARGET_SAMPLES = 64600  # ~4.0375s at 16kHz


class AudioNormalizer:
    """
    Standardizes audio streams into fixed-length 16kHz mono float32 arrays.
    SincNet raw waveform model input format.
    """

    def __init__(self, target_sr: int = TARGET_SR, target_samples: int = TARGET_SAMPLES):
        self.target_sr = target_sr
        self.target_samples = target_samples

    def to_mono(self, waveform: np.ndarray) -> np.ndarray:
        """Converts multi-channel audio to single-channel mono."""
        if waveform.ndim > 1:
            # If shape is (channels, samples) or (samples, channels)
            if waveform.shape[0] < waveform.shape[1]:
                waveform = np.mean(waveform, axis=0)
            else:
                waveform = np.mean(waveform, axis=1)
        return waveform.astype(np.float32)

    def resample(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples audio array to target 16,000 Hz."""
        if orig_sr == self.target_sr or orig_sr <= 0:
            return waveform.astype(np.float32)

        num_output_samples = int(round(len(waveform) * float(self.target_sr) / orig_sr))
        resampled = signal.resample(waveform, num_output_samples)
        return resampled.astype(np.float32)

    def normalize_amplitude(self, waveform: np.ndarray) -> np.ndarray:
        """Scales signal amplitude to [-1.0, 1.0]."""
        max_val = np.max(np.abs(waveform))
        if max_val > 1e-6:
            waveform = waveform / max_val
        return waveform.astype(np.float32)

    def fix_length(self, waveform: np.ndarray, is_train: bool = False) -> np.ndarray:
        """
        Fixes waveform length to exactly 64,600 samples.
        Short clips use circular repeat-tiling to avoid sharp zero-padding artifacts.
        Long clips use center cropping (or random offset during training).
        """
        curr_len = len(waveform)
        if curr_len == self.target_samples:
            return waveform

        if curr_len < self.target_samples:
            # Repeat-tiling to preserve harmonic periodicity
            repeats = math.ceil(self.target_samples / max(1, curr_len))
            tiled = np.tile(waveform, repeats)
            return tiled[: self.target_samples].astype(np.float32)
        else:
            # Crop
            if is_train:
                start = np.random.randint(0, curr_len - self.target_samples + 1)
            else:
                start = (curr_len - self.target_samples) // 2
            return waveform[start : start + self.target_samples].astype(np.float32)

    def process(self, raw_samples: np.ndarray, orig_sr: int = TARGET_SR) -> np.ndarray:
        """Executes full normalization pipeline."""
        mono = self.to_mono(raw_samples)
        resampled = self.resample(mono, orig_sr)
        normalized = self.normalize_amplitude(resampled)
        framed = self.fix_length(normalized)
        return framed

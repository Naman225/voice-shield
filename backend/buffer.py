import collections
import numpy as np

WINDOW_SIZE = 64600  # ~4.04s at 16kHz
HOP_SIZE = 16000     # 1.0s hop step at 16kHz


class SlidingAudioBuffer:
    """
    Real-time First-In First-Out (FIFO) sliding audio buffer.
    Accumulates streaming PCM audio packets and emits 64,600-sample snapshots every 1.0s.
    """

    def __init__(self, window_size: int = WINDOW_SIZE, hop_size: int = HOP_SIZE):
        self.window_size = window_size
        self.hop_size = hop_size
        self._buffer = collections.deque(maxlen=window_size * 2)
        self.total_samples_ingested = 0
        self.samples_since_last_hop = 0

    def push_samples(self, samples: np.ndarray):
        """Append incoming PCM float32 samples to the FIFO queue."""
        if len(samples) == 0:
            return
        flat_samples = samples.flatten().tolist()
        self._buffer.extend(flat_samples)
        num_added = len(flat_samples)
        self.total_samples_ingested += num_added
        self.samples_since_last_hop += num_added

    def is_window_ready(self) -> bool:
        """Returns True if the buffer contains at least 1 full window of audio."""
        return len(self._buffer) >= self.window_size

    def should_hop(self) -> bool:
        """Returns True if enough new samples arrived since the last snapshot inference."""
        return len(self._buffer) >= self.window_size and self.samples_since_last_hop >= self.hop_size

    def get_latest_window(self) -> np.ndarray:
        """
        Extracts the latest 64,600-sample audio array from the buffer.
        Resets the hop sample counter.
        """
        if len(self._buffer) < self.window_size:
            # Pad with repeat-tiling if less than full window
            arr = np.array(self._buffer, dtype=np.float32)
            if len(arr) == 0:
                return np.zeros(self.window_size, dtype=np.float32)
            repeats = int(np.ceil(self.window_size / len(arr)))
            return np.tile(arr, repeats)[: self.window_size]

        buf_arr = np.array(self._buffer, dtype=np.float32)
        latest_window = buf_arr[-self.window_size :]
        self.samples_since_last_hop = 0
        return latest_window

    def reset(self):
        """Clear buffer state."""
        self._buffer.clear()
        self.total_samples_ingested = 0
        self.samples_since_last_hop = 0

import math 
import torch
import torchaudio
import torchaudio.transforms as T 
import soundfile as sf 

TARGET_SR = 16000
TARGET_SAMPLES = 64600 ## 16khz


class AudioNormalizer:
    def __init__(self, target_sr = TARGET_SR, target_samples = TARGET_SAMPLES):
        self.target_sr = target_sr
        self.target_samples = target_samples

        self.resamplers = {} ## cache data

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
        return self._resamplers[orig_sr]

    def load_process(self, file_path : str, is_train: bool = True):
        try:
            data, orig_sr = sf.read(file_path, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 2:
                waveform = waveform.t()
        except Exception:
            waveform, orig_sr = torchaudio.load(file_path)


        ## To mono 
        if waveform.ndim == 2:
            waveform = torch.mean(waveform, dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        ## Resample to 16khz

        if orig_sr != self.target_sr:
            resampler = self._get_resampler(orig_sr)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        ## Normalize volume
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-6:
            waveform = waveform / max_val

        ## Fix len to 64,600 samples

        curr_len = waveform.shape[0]
        if curr_len > self.target_samples:
            start = torch.randint(0, curr_len - self.target_samples + 1, (1,)).item() if is_train else (curr_len - self.target_samples) // 2
            waveform = waveform[start : start + self.target_samples]
        elif curr_len < self.target_samples:
            # If shorter: repeat-tile sound to preserve pitch without zero-padding artifacts
            repeats = math.ceil(self.target_samples / max(1, curr_len))
            waveform = waveform.repeat(repeats)[: self.target_samples]
        return waveform 
    
    

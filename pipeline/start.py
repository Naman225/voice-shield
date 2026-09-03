import os, glob
import torch 
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T 

from torch.utils.data  import DataLoader, Dataset

TARGET_SR = 16000
TARGET_LEN = 64600
AUDIOS_ROOT = "./data"

def _to_fixed_length(waveform: torch.Tensor , sr: int) -> torch.Tensor:
    if waveform.ndim > 1:
        waveform = waveform.mean(dim = 0)
    if sr != TARGET_SR:
        waveform = T.Resample(sr, TARGET_SR)(waveform.unsqueeze(0)).squeeze(0)
    if waveform.shape[0] > TARGET_LEN: 
        waveform = waveform[:TARGET_LEN]
    elif waveform.shape[0] < TARGET_LEN:
        waveform = F.pad(waveform, (0, TARGET_LEN - waveform.shape[0]))
    return waveform

class IndieFakeDataset(Dataset):
    def __init__(self, root=AUDIOS_ROOT, extensions=(".wav", ".flac", ".mp3")):
        self.samples = []  # list of (filepath, label)
        speaker_folders = sorted(glob.glob(os.path.join(root, "Speaker-*")))
        if not speaker_folders:
            raise FileNotFoundError(
                f"No Speaker-* folders found under {root} -- check the Drive "
                f"shortcut was added and the path above matches your mount."
            )
        for speaker_dir in speaker_folders:
            for label_name, label in [("Bonafides", 1), ("Deepfakes", 0)]:
                class_dir = os.path.join(speaker_dir, label_name)
                if not os.path.isdir(class_dir):
                    continue
                for ext in extensions:
                    for filepath in glob.glob(os.path.join(class_dir, f"*{ext}")):
                        self.samples.append((filepath, label))
 
        n_real = sum(1 for _, l in self.samples if l == 1)
        n_fake = sum(1 for _, l in self.samples if l == 0)
        print(f"IndieFake index: {len(speaker_folders)} speakers, "
              f"{n_real} bonafide, {n_fake} spoof, {len(self.samples)} total")
 
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        waveform = _to_fixed_length(waveform, sr)
        return waveform, torch.tensor(label, dtype=torch.long)
 
 
if __name__ == "__main__":
    dataset = IndieFakeDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
 
    try:
        for wf_batch, label_batch in loader:
            print("SUCCESS: IndieFake dataset pipeline is working!")
            print("---------------------------------------------")
            print(f"Waveform batch shape: {wf_batch.shape}")
            print("  Meaning: [batch_size, 64600 raw audio samples]")
            print(f"Label batch shape:    {label_batch.shape}")
            print(f"Label values in this batch: {label_batch.tolist()}  (0=spoof, 1=bonafide)")
            break
    except Exception as error:
        print(f"ERROR: pipeline crashed. Details: {error}")

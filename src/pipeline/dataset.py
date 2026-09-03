import os
import glob
import random 
import torch

from torch.utils.data import DataLoader, Dataset
from normalize import AudioNormalizer

class IndieFakeDataset(Dataset):
    def __init__(self, samples, is_train = False, normalizer = None):
        self.samples = samples  
        self.is_train = is_train
        self.normalizer = normalizer or AudioNormalizer()   

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, speaker_id = self.samples[idx]
        waveform = self.normalizer.load_process(file_path, is_train = self.is_train) # Fixed to use self.is_train
        return waveform, torch.tensor(label, dtype = torch.long), speaker_id

def create_splits(data_root = "IndieFake/data", seed = 42):
    speaker_dirs = sorted(glob.glob(os.path.join(data_root, "Speaker-*")))
    rng = random.Random(seed)
    rng.shuffle(speaker_dirs)

    train_spks = set(speaker_dirs[:35])
    val_spks = set(speaker_dirs[35:40])
    test_spks = set(speaker_dirs[40:])

    def get_samples(spk_set):
        samples = []
        for s in spk_set:
            spk_id = os.path.basename(s)
            # Bonafides = Real (1), Deepfakes = Spoof (0)
            for f in glob.glob(os.path.join(s, "Bonafides", "*.wav")):
                samples.append((f, 1, spk_id))
            for f in glob.glob(os.path.join(s, "Deepfakes", "*.wav")):
                samples.append((f, 0, spk_id))
        return samples
    
    return get_samples(train_spks), get_samples(val_spks), get_samples(test_spks)


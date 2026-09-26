"""
scripts/scale_production_dataset.py
------------------------------------
Builds a production-grade balanced dataset for banking voice fraud middleware:
1. Real Speech:
   - InDeepFake Real (iPhone + YouTube interviews)
   - Google FLEURS (1,000 native Hindi + 1,000 native Tamil speakers)
2. Synthetic Voice Clones:
   - IndicSynth (1,925 Hindi FreeVC + 1,651 Tamil FreeVC)
   - InDeepFake F5-TTS (586 English Flow-Matching Clones)

Target: ~6,400 clean 16kHz mono WAV files with ~1:2 ratio for robust production deployment.
"""

import os
import io
import soundfile as sf
import torchaudio
import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_OUT_ROOT = os.path.join(REPO_ROOT, "data")
TARGET_SR     = 16000

def save_wav(waveform_tensor, orig_sr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if orig_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
        waveform_tensor = resampler(waveform_tensor.unsqueeze(0)).squeeze(0)
    max_val = torch.max(torch.abs(waveform_tensor))
    if max_val > 1e-6:
        waveform_tensor = waveform_tensor / max_val
    torchaudio.save(out_path, waveform_tensor.unsqueeze(0) if waveform_tensor.ndim == 1 else waveform_tensor, TARGET_SR)

def main():
    print("=" * 60)
    print(" Scaling to Production-Grade Middleware Dataset (~6,400 samples)")
    print("=" * 60)

    # 1. Real Speech from Google FLEURS (Hindi & Tamil)
    print("\n[Step 1/2] Adding Real Native Indian Speech from Google FLEURS...")
    
    # Hindi Real
    print("Loading Google FLEURS Hindi...")
    ds_hi = load_dataset("google/fleurs", "hi_in", split="train")
    print(f"Extracting {min(1000, len(ds_hi))} real Hindi recordings...")
    for i in tqdm(range(min(1000, len(ds_hi))), desc="Saving Real Hindi"):
        item = ds_hi[i]["audio"]
        wav = torch.from_numpy(item["array"]).float()
        out_f = os.path.join(DATA_OUT_ROOT, "hi", "bonafide", f"fleurs_hi_{i:04d}.wav")
        save_wav(wav, item["sampling_rate"], out_f)

    # Tamil Real
    print("Loading Google FLEURS Tamil...")
    ds_ta = load_dataset("google/fleurs", "ta_in", split="train")
    print(f"Extracting {min(1000, len(ds_ta))} real Tamil recordings...")
    for i in tqdm(range(min(1000, len(ds_ta))), desc="Saving Real Tamil"):
        item = ds_ta[i]["audio"]
        wav = torch.from_numpy(item["array"]).float()
        out_f = os.path.join(DATA_OUT_ROOT, "ta", "bonafide", f"fleurs_ta_{i:04d}.wav")
        save_wav(wav, item["sampling_rate"], out_f)

    # 2. Extract remaining IndicSynth Spoofs (all 1,925 Hindi + 1,651 Tamil)
    print("\n[Step 2/2] Extracting all cached IndicSynth AI Voice Clones...")
    hi_p = hf_hub_download(repo_id="vdivyasharma/IndicSynth", filename="Hindi/train-00000-of-00107.parquet", repo_type="dataset")
    df_hi = pd.read_parquet(hi_p)
    print(f"Saving {len(df_hi)} total Hindi voice clones...")
    for i in tqdm(range(len(df_hi)), desc="Hindi Spoofs"):
        out_f = os.path.join(DATA_OUT_ROOT, "hi", "spoof", f"indicsynth_hi_{i:04d}.wav")
        if not os.path.isfile(out_f):
            audio_bytes = df_hi.iloc[i]["audio"]["bytes"]
            data, sr = sf.read(io.BytesIO(audio_bytes))
            save_wav(torch.from_numpy(data).float(), sr, out_f)

    ta_p = hf_hub_download(repo_id="vdivyasharma/IndicSynth", filename="Tamil/train-00000-of-00171.parquet", repo_type="dataset")
    df_ta = pd.read_parquet(ta_p)
    print(f"Saving {len(df_ta)} total Tamil voice clones...")
    for i in tqdm(range(len(df_ta)), desc="Tamil Spoofs"):
        out_f = os.path.join(DATA_OUT_ROOT, "ta", "spoof", f"indicsynth_ta_{i:04d}.wav")
        if not os.path.isfile(out_f):
            audio_bytes = df_ta.iloc[i]["audio"]["bytes"]
            data, sr = sf.read(io.BytesIO(audio_bytes))
            save_wav(torch.from_numpy(data).float(), sr, out_f)

    print("\n" + "=" * 60)
    print(" PRODUCTION DATASET READY!")
    print("=" * 60)
    import glob
    for lang in ["hi", "ta", "en"]:
        b = len(glob.glob(os.path.join(DATA_OUT_ROOT, lang, "bonafide", "*.wav")))
        s = len(glob.glob(os.path.join(DATA_OUT_ROOT, lang, "spoof", "*.wav")))
        print(f" Language [{lang.upper()}]: {b:4d} Real (Bonafide)  |  {s:4d} AI Clones (Spoof)  |  Total: {b+s:4d}")
    print("=" * 60)

if __name__ == "__main__":
    main()

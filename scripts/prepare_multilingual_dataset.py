"""
scripts/prepare_multilingual_dataset.py
-----------------------------------------
Prepares the complete training dataset for Hindi, Tamil, and English:

1. Real (Bonafide) speech:
   - Extracted from /home/naman/Downloads/InDeepFake/Real
   - Language identified via Whisper (Hindi 'hi', Tamil 'ta', English 'en')
   - Converted to 16kHz mono WAV in data/<lang>/bonafide/

2. Fake (Spoof) synthetic voice clones:
   - Extracted from IndicSynth (Hindi & Tamil cached parquet files: freevc24 / xtts)
   - Converted to 16kHz mono WAV in data/<lang>/spoof/
   - Plus English voice clones from InDeepFake F5-TTS

Result: Clean, balanced dataset in /home/naman/Desktop/SIH/IndieFake/data/ ready for train.py
"""

import os
import io
import sys
import glob
import subprocess
import pandas as pd
import soundfile as sf
import torchaudio
import torch
import whisper
from tqdm import tqdm
from huggingface_hub import hf_hub_download

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_OUT_ROOT = os.path.join(REPO_ROOT, "data")
INDEEP_ROOT   = "/home/naman/Downloads/InDeepFake"

TARGET_SR = 16000

def resample_and_save(audio_data, orig_sr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if isinstance(audio_data, bytes):
        audio_data, orig_sr = sf.read(io.BytesIO(audio_data))
    
    waveform = torch.from_numpy(audio_data).float()
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=-1) if waveform.shape[-1] < waveform.shape[0] else waveform.mean(dim=0)
    elif waveform.ndim == 1:
        pass
    
    if orig_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
    
    # Normalize peak amplitude
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1e-6:
        waveform = waveform / max_val
        
    torchaudio.save(out_path, waveform.unsqueeze(0), TARGET_SR)

def main():
    print("=" * 60)
    print(" Voice Shield — Multilingual Dataset Preparation")
    print(" Languages: Hindi (hi), Tamil (ta), English (en)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper base model on {device}...")
    whisper_model = whisper.load_model("base", device=device)

    # ─────────────────────────────────────────────────────────────
    # Step 1: Real (Bonafide) Speech from InDeepFake
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 1/3] Processing Real Indian Speech from InDeepFake/Real...")
    real_videos = glob.glob(os.path.join(INDEEP_ROOT, "Real", "**", "*.mp4"), recursive=True)
    print(f"Found {len(real_videos)} real speaker videos.")

    real_counts = {"hi": 0, "ta": 0, "en": 0}

    for v in tqdm(real_videos, desc="Classifying & Extracting Real Audio"):
        try:
            audio = whisper.load_audio(v)
            audio_pad = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio_pad, n_mels=80).to(device)
            _, probs = whisper_model.detect_language(mel)
            top_lang = max(probs, key=probs.get)

            if top_lang in ["hi", "ta", "en"]:
                base = os.path.splitext(os.path.basename(v))[0]
                out_wav = os.path.join(DATA_OUT_ROOT, top_lang, "bonafide", f"{base}.wav")
                resample_and_save(audio, 16000, out_wav)
                real_counts[top_lang] += 1
        except Exception:
            continue

    print(f"Real speech extracted: Hindi={real_counts['hi']}, Tamil={real_counts['ta']}, English={real_counts['en']}")

    # ─────────────────────────────────────────────────────────────
    # Step 2: IndicSynth Spoof (Voice Clones) for Hindi & Tamil
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 2/3] Extracting SOTA Voice Clones from IndicSynth (Hugging Face)...")

    # Hindi
    print("Loading IndicSynth Hindi parquet...")
    hi_p = hf_hub_download(repo_id="vdivyasharma/IndicSynth", filename="Hindi/train-00000-of-00107.parquet", repo_type="dataset")
    df_hi = pd.read_parquet(hi_p)
    max_spoof = 1000

    print(f"Extracting {min(max_spoof, len(df_hi))} Hindi synthetic voice clones...")
    for i in tqdm(range(min(max_spoof, len(df_hi))), desc="Saving Hindi Spoofs"):
        row = df_hi.iloc[i]
        out_wav = os.path.join(DATA_OUT_ROOT, "hi", "spoof", f"indicsynth_hi_{i:04d}.wav")
        resample_and_save(row["audio"]["bytes"], 24000, out_wav)

    # Tamil
    print("Loading IndicSynth Tamil parquet...")
    ta_p = hf_hub_download(repo_id="vdivyasharma/IndicSynth", filename="Tamil/train-00000-of-00171.parquet", repo_type="dataset")
    df_ta = pd.read_parquet(ta_p)

    print(f"Extracting {min(max_spoof, len(df_ta))} Tamil synthetic voice clones...")
    for i in tqdm(range(min(max_spoof, len(df_ta))), desc="Saving Tamil Spoofs"):
        row = df_ta.iloc[i]
        out_wav = os.path.join(DATA_OUT_ROOT, "ta", "spoof", f"indicsynth_ta_{i:04d}.wav")
        resample_and_save(row["audio"]["bytes"], 24000, out_wav)

    # ─────────────────────────────────────────────────────────────
    # Step 3: English Spoofs from InDeepFake F5-TTS
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 3/3] Extracting English F5-TTS Voice Clones from InDeepFake/Fake...")
    f5_videos = glob.glob(os.path.join(INDEEP_ROOT, "Fake", "F5TTS+Wav2LIP", "**", "*.mp4"), recursive=True)[:max_spoof]
    print(f"Extracting {len(f5_videos)} English voice clones...")

    for i, fv in enumerate(tqdm(f5_videos, desc="Saving English Spoofs")):
        try:
            base = os.path.splitext(os.path.basename(fv))[0]
            out_wav = os.path.join(DATA_OUT_ROOT, "en", "spoof", f"f5tts_en_{base}.wav")
            audio = whisper.load_audio(fv)
            resample_and_save(audio, 16000, out_wav)
        except Exception:
            continue

    print("\n" + "=" * 60)
    print(" DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    for lang in ["hi", "ta", "en"]:
        b_cnt = len(glob.glob(os.path.join(DATA_OUT_ROOT, lang, "bonafide", "*.wav")))
        s_cnt = len(glob.glob(os.path.join(DATA_OUT_ROOT, lang, "spoof", "*.wav")))
        print(f" Language [{lang.upper()}]: {b_cnt} Real (Bonafide)  |  {s_cnt} AI Voice Clones (Spoof)")
    print("=" * 60)
    print(f"All audio files saved to: {DATA_OUT_ROOT}")

if __name__ == "__main__":
    main()

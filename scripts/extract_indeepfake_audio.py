"""
scripts/extract_indeepfake_audio.py
------------------------------------
Extracts audio from /home/naman/Downloads/InDeepFake:
1. Scans all Real videos (407 files)
2. Uses Whisper to detect spoken language (Hindi 'hi', Tamil 'ta', English 'en', etc.)
3. Extracts 16kHz mono audio as Bonafide (Real) samples
4. Finds corresponding Fake videos in F5TTS+Wav2LIP, E2TTS+Wav2LIP, and Wav2LIP+SV2TTS
5. Extracts 16kHz mono audio as Spoof (Fake) samples
6. Populates data/<lang>/bonafide and data/<lang>/spoof ready for train.py!
"""

import os
import sys
import glob
import subprocess
import whisper
import torch
from collections import defaultdict
from tqdm import tqdm

DATA_OUT_ROOT = "/home/naman/Desktop/SIH/IndieFake/data"
INDEEPFAKE_DIR = "/home/naman/Downloads/InDeepFake"

TARGET_LANGS = {"hi", "ta", "en", "te", "bn"}

def extract_wav(video_path, out_wav_path):
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_wav_path, "-loglevel", "quiet"
    ]
    subprocess.run(cmd, check=True)

def main():
    print("=" * 60)
    print(" InDeepFake Multilingual Audio Extractor & Language Classifier")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper base model on {device}...")
    whisper_model = whisper.load_model("base", device=device)

    real_videos = glob.glob(os.path.join(INDEEPFAKE_DIR, "Real", "**", "*.mp4"), recursive=True)
    print(f"Found {len(real_videos)} Real videos.")

    video_lang = {}
    lang_counts = defaultdict(int)

    print("\n[Step 1/3] Detecting language for all Real videos...")
    for v in tqdm(real_videos):
        try:
            audio = whisper.load_audio(v)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=80).to(device)
            _, probs = whisper_model.detect_language(mel)
            top_lang = max(probs, key=probs.get)
            
            basename = os.path.basename(v)
            clean_id = os.path.splitext(basename)[0]
            video_lang[clean_id] = top_lang
            lang_counts[top_lang] += 1
            
            if top_lang in TARGET_LANGS:
                out_path = os.path.join(DATA_OUT_ROOT, top_lang, "bonafide", f"{clean_id}.wav")
                extract_wav(v, out_path)
        except Exception:
            continue

    print("\nReal Video Language Breakdown:")
    for l, cnt in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {l}: {cnt} speakers")

    # 2. Extract Fake audio from TTS-based models
    tts_fake_dirs = [
        os.path.join(INDEEPFAKE_DIR, "Fake", "F5TTS+Wav2LIP"),
        os.path.join(INDEEPFAKE_DIR, "Fake", "E2TTS+Wav2LIP"),
        os.path.join(INDEEPFAKE_DIR, "Fake", "Wav2LIP+SV2TTS"),
    ]

    print("\n[Step 2/3] Extracting synthetic voice deepfakes...")
    extracted_spoof = defaultdict(int)

    for fake_dir in tts_fake_dirs:
        model_name = os.path.basename(fake_dir)
        fake_vids = glob.glob(os.path.join(fake_dir, "**", "*.mp4"), recursive=True)
        print(f"Processing {len(fake_vids)} files from {model_name}...")

        for fv in tqdm(fake_vids):
            fname = os.path.basename(fv)
            clean_name = os.path.splitext(fname)[0]
            
            matched_lang = None
            for real_id, l in video_lang.items():
                if real_id in clean_name:
                    matched_lang = l
                    break
            
            if not matched_lang or matched_lang not in TARGET_LANGS:
                matched_lang = "en"

            out_wav = os.path.join(DATA_OUT_ROOT, matched_lang, "spoof", f"{model_name}_{clean_name}.wav")
            try:
                extract_wav(fv, out_wav)
                extracted_spoof[matched_lang] += 1
            except Exception:
                continue

    print("\n[Step 3/3] Extraction Summary:")
    print("=" * 60)
    for lang in TARGET_LANGS:
        b_count = len(glob.glob(os.path.join(DATA_OUT_ROOT, lang, "bonafide", "*.wav")))
        s_count = len(glob.glob(os.path.join(DATA_OUT_ROOT, lang, "spoof", "*.wav")))
        print(f"Language [{lang}]: {b_count} Bonafide (Real) | {s_count} Spoof (Fake)")
    print("=" * 60)
    print(f"Audio ready in {DATA_OUT_ROOT} for training!")

if __name__ == "__main__":
    main()

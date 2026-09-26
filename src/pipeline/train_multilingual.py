"""
src/pipeline/train_multilingual.py
-----------------------------------
Fine-tunes the AASIST model on IndicFake (Hindi + Tamil) combined with the
existing IndieFake English data.

How to run:
    python src/pipeline/train_multilingual.py \
        --indicfake_root /path/to/IndicFake \
        --epochs 15 \
        --languages hi,ta

Dataset folder structure expected for IndicFake (try both naming conventions):
    /path/to/IndicFake/
        hi/
            bonafide/  OR  real/      <- real human speech
            spoof/     OR  fake/      <- AI generated speech
        ta/
            bonafide/
            spoof/

Mixed ratio: 30% English IndieFake + 70% IndicFake (prevents catastrophic forgetting)
Output: checkpoints/best_multilingual_aasist.pth
"""

import os, sys, glob, shutil, random, argparse, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT   = os.path.abspath(os.path.join(SRC_DIR, ".."))
CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")
for p in [SCRIPT_DIR, SRC_DIR]:
    if p not in sys.path: sys.path.insert(0, p)

from normalize import AudioNormalizer
from models.aasist import load_pretrained_aasist, get_optimizer_groups

os.makedirs(CHECKPOINTS, exist_ok=True)

LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil", "bn": "Bengali"}
TARGET_CKPT   = os.path.join(CHECKPOINTS, "best_multilingual_aasist.pth")
ENGLISH_CKPT  = os.path.join(CHECKPOINTS, "best_indiefake_aasist.pth")
BACKUP_CKPT   = os.path.join(CHECKPOINTS, "prototype_aasist_v1.pth")


# ──────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _load_english_samples(data_root):
    samples = []
    # 1. Check Speaker-* format (IndieFake original format)
    for spk_dir in glob.glob(os.path.join(data_root, "Speaker-*")):
        spk_id = os.path.basename(spk_dir)
        for f in glob.glob(os.path.join(spk_dir, "Bonafides", "*.wav")):
            samples.append((f, 1, "en", spk_id))
        for f in glob.glob(os.path.join(spk_dir, "Deepfakes", "*.wav")):
            samples.append((f, 0, "en", spk_id))
    # 2. Check en/bonafide and en/spoof format
    en_dir = os.path.join(data_root, "en")
    if os.path.isdir(en_dir):
        for f in glob.glob(os.path.join(en_dir, "bonafide", "*.wav")):
            samples.append((f, 1, "en", "en_real"))
        for f in glob.glob(os.path.join(en_dir, "spoof", "*.wav")):
            samples.append((f, 0, "en", "en_fake"))
    return samples


def _load_indicfake_lang(lang_dir, lang):
    """Try multiple folder name conventions for real/spoof."""
    real_names  = ["bonafide", "real", "genuine", "authentic"]
    spoof_names = ["spoof", "fake", "synthetic", "deepfake"]
    samples = []
    for d in real_names:
        for ext in ["*.wav", "*.flac", "*.mp3"]:
            for f in glob.glob(os.path.join(lang_dir, d, ext)):
                samples.append((f, 1, lang, lang))
    for d in spoof_names:
        for ext in ["*.wav", "*.flac", "*.mp3"]:
            for f in glob.glob(os.path.join(lang_dir, d, ext)):
                samples.append((f, 0, lang, lang))
    return samples


def _load_indicfake_samples(root, languages):
    samples = []
    for lang in languages:
        if lang == "en": continue
        lang_dir = os.path.join(root, lang)
        if not os.path.isdir(lang_dir):
            print(f"  [WARN] Language folder not found: {lang_dir}")
            continue
        lang_samples = _load_indicfake_lang(lang_dir, lang)
        real_c  = sum(1 for s in lang_samples if s[1] == 1)
        spoof_c = sum(1 for s in lang_samples if s[1] == 0)
        print(f"  [IndicFake/{LANG_NAMES.get(lang,lang)}] {real_c} real + {spoof_c} spoof")
        samples.extend(lang_samples)
    return samples


class MultilingualDataset(Dataset):
    def __init__(self, samples, is_train=True, normalizer=None):
        self.samples    = samples
        self.is_train   = is_train
        self.normalizer = normalizer or AudioNormalizer()

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label, lang, _ = self.samples[idx]
        wav = self.normalizer.load_process(path, is_train=self.is_train)
        return wav, torch.tensor(label, dtype=torch.long), lang


def build_mixed_dataset(eng_root, indicfake_root, languages,
                        english_ratio=0.30, max_per_lang=5000, seed=42):
    rng = random.Random(seed)
    en_samps = _load_english_samples(eng_root)
    print(f"  [IndieFake/English] {len(en_samps)} samples")
    ml_samps = _load_indicfake_samples(indicfake_root, languages)

    # Cap per-language
    by_lang = {}
    for s in ml_samps:
        by_lang.setdefault(s[2], []).append(s)
    capped = []
    for lang, ls in by_lang.items():
        rng.shuffle(ls); capped.extend(ls[:max_per_lang])

    n_new   = len(capped)
    en_tgt  = int((n_new / (1 - english_ratio)) * english_ratio)
    rng.shuffle(en_samps)
    kept_en = en_samps[:min(en_tgt, len(en_samps))]
    all_s   = kept_en + capped
    rng.shuffle(all_s)
    print(f"  [Mixed] EN: {len(kept_en)} | New languages: {n_new} | Total: {len(all_s)}")

    n = len(all_s)
    n_tr, n_vl = int(n*0.80), int(n*0.10)
    return all_s[:n_tr], all_s[n_tr:n_tr+n_vl], all_s[n_tr+n_vl:]


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx]+fnr[idx])/2*100), float(thr[idx])


def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            _, logits = model(x.to(device))
            scores.extend((logits[:,1]-logits[:,0]).cpu().numpy())
            labels.extend(y.numpy())
    if device == "cuda": torch.cuda.empty_cache()
    eer, thr = compute_eer(np.array(labels), np.array(scores))
    try:   auc = roc_auc_score(labels, scores)*100
    except: auc = 0.0
    acc = (np.array(scores)>=thr).astype(int)
    acc = (acc == np.array(labels)).mean()*100
    return eer, auc, acc, thr


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs  = [l.strip().lower() for l in args.languages.split(",")]
    if "en" not in langs: langs = ["en"] + langs

    print("\n" + "="*60)
    print(f"  Multilingual AASIST Fine-Tuning")
    print(f"  Languages : {[LANG_NAMES.get(l,l) for l in langs]}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print("="*60)

    print("\n[1] Building mixed multilingual dataset...")
    train_s, val_s, test_s = build_mixed_dataset(
        eng_root       = os.path.join(REPO_ROOT, "data"),
        indicfake_root = args.indicfake_root,
        languages      = langs,
        english_ratio  = 0.30,
        max_per_lang   = args.max_per_lang,
    )
    print(f"    Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    normalizer   = AudioNormalizer()
    nw = 2 if device == "cuda" else 0
    train_loader = DataLoader(MultilingualDataset(train_s, True,  normalizer), args.batch_size, shuffle=True,  num_workers=nw, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(MultilingualDataset(val_s,   False, normalizer), args.batch_size, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(MultilingualDataset(test_s,  False, normalizer), args.batch_size, shuffle=False, num_workers=nw)

    n_real = sum(1 for _,l,_,_ in train_s if l==1)
    n_fake = sum(1 for _,l,_,_ in train_s if l==0)
    tot    = len(train_s)
    wts    = torch.tensor([tot/(2*max(1,n_fake)), tot/(2*max(1,n_real))], device=device)
    crit   = nn.CrossEntropyLoss(weight=wts)

    print("\n[2] Loading warm-start checkpoint...")
    src = TARGET_CKPT if os.path.isfile(TARGET_CKPT) else ENGLISH_CKPT
    if os.path.isfile(ENGLISH_CKPT) and not os.path.isfile(BACKUP_CKPT):
        shutil.copyfile(ENGLISH_CKPT, BACKUP_CKPT)
        print(f"    Backed up English checkpoint -> {BACKUP_CKPT}")
    model = load_pretrained_aasist(checkpoint_path=src if os.path.isfile(src) else None, device=device)
    model.train()

    opt   = torch.optim.Adam(get_optimizer_groups(model, args.frontend_lr, args.backend_lr))
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
    best_eer = 100.0; history = []

    print(f"\n[3] Fine-tuning for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs+1):
        model.train(); t0 = time.time(); total_loss = 0.0
        for step, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            if (step+1) % 30 == 0 or (step+1) == len(train_loader):
                print(f"  Ep [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {total_loss/(step+1):.4f}", end="\r")
        sched.step()

        val_eer, val_auc, val_acc, _ = evaluate(model, val_loader, device)
        print(f"\n  Epoch {epoch:02d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Val EER: {val_eer:.2f}% | AUC: {val_auc:.2f}% | Acc: {val_acc:.2f}% | {time.time()-t0:.0f}s")
        history.append({"epoch": epoch, "val_eer": val_eer, "val_auc": val_auc})

        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), TARGET_CKPT)
            print(f"  --> Saved best (EER: {val_eer:.2f}%) to {TARGET_CKPT}")

    print("\n[4] Final test evaluation...")
    if os.path.isfile(TARGET_CKPT):
        model.load_state_dict(torch.load(TARGET_CKPT, map_location=device, weights_only=True))
    test_eer, test_auc, test_acc, thresh = evaluate(model, test_loader, device)

    print("\n" + "="*60)
    print(f"  DONE | Languages: {[LANG_NAMES.get(l,l) for l in langs]}")
    print(f"  Test EER:  {test_eer:.2f}%  |  AUC: {test_auc:.2f}%  |  Acc: {test_acc:.2f}%")
    print(f"  Checkpoint: {TARGET_CKPT}")
    print("="*60)

    results = {
        "model_name": "AASIST-Multilingual-IndieFake+IndicFake",
        "languages": langs, "test_eer_percent": round(test_eer,2),
        "test_auc_percent": round(test_auc,2), "test_accuracy": round(test_acc,2),
        "threshold": round(thresh,4), "epochs": args.epochs, "history": history,
    }
    out = os.path.join(REPO_ROOT, "evaluation_results", "multilingual_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"  Metrics: {out}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--indicfake_root", required=True, help="Root dir of IndicFake dataset")
    p.add_argument("--languages",  default="hi,ta", help="Comma-separated codes (hi,ta). en always included.")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch_size", type=int,   default=6)
    p.add_argument("--backend_lr", type=float, default=1e-4)
    p.add_argument("--frontend_lr",type=float, default=1e-6)
    p.add_argument("--max_per_lang",type=int,  default=5000)
    train(p.parse_args())

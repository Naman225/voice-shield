import os
import sys

# Ensure both current directory and parent 'src' are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import shutil
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import create_splits, IndieFakeDataset
from normalize import AudioNormalizer
from models.aasist import load_pretrained_aasist, get_optimizer_groups


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100.0
    return eer, thresholds[idx]

def train(epochs=12, backend_lr=3e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f" AASIST Overnight Training on IndieFake (Option B)")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f" Total Epochs: {epochs} | Backend LR: {backend_lr}")
    print(f"==================================================")

    # 1. Prepare 40/5/5 speaker-disjoint splits
    train_samples, val_samples, test_samples = create_splits()
    print(f"Dataset Partitions: {len(train_samples)} Train, {len(val_samples)} Val, {len(test_samples)} Test.")

    normalizer = AudioNormalizer()
    train_loader = DataLoader(IndieFakeDataset(train_samples, is_train=True, normalizer=normalizer),
                              batch_size=6, shuffle=True, pin_memory=True)
    val_loader = DataLoader(IndieFakeDataset(val_samples, is_train=False, normalizer=normalizer),
                            batch_size=6, shuffle=False)
    test_loader = DataLoader(IndieFakeDataset(test_samples, is_train=False, normalizer=normalizer),
                             batch_size=12, shuffle=False)

    # 2. Weighted loss for class imbalance
    n_fake = sum(1 for _, l, _ in train_samples if l == 0)
    n_real = sum(1 for _, l, _ in train_samples if l == 1)
    weights = torch.tensor([len(train_samples)/(2*max(1, n_fake)), len(train_samples)/(2*max(1, n_real))], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 3. Model setup & Warm Start from current prototype checkpoint
    checkpoint_dir = os.path.abspath(os.path.join(current_dir, "../../checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    existing_ckpt = os.path.join(checkpoint_dir, "best_indiefake_aasist.pth")
    backup_prototype = os.path.join(checkpoint_dir, "prototype_aasist_v1.pth")

    # Preserve prototype backup for teammates
    if os.path.isfile(existing_ckpt) and not os.path.isfile(backup_prototype):
        shutil.copyfile(existing_ckpt, backup_prototype)
        print(f"[Backup] Preserved prototype checkpoint for teammates at: {backup_prototype}")

    if os.path.isfile(existing_ckpt):
        print(f"[Warm-Start] Resuming training from existing checkpoint: {existing_ckpt}")
        model = load_pretrained_aasist(checkpoint_path=existing_ckpt, device=device)
    else:
        print("[Init] Starting fine-tuning from official ASVspoof pretrained weights.")
        model = load_pretrained_aasist(device=device)

    # 4. Optimizer with 3e-5 backend LR and gradient clipping
    optimizer = torch.optim.Adam(get_optimizer_groups(model, frontend_lr=1e-6, backend_lr=backend_lr))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_val_eer = 60.44  # Starting from previous best validation EER

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping for stable overnight training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(train_loader)}] Loss: {total_loss / (step + 1):.4f}", end="\r")

        scheduler.step()

        # Validation check
        model.eval()
        scores, labels = [], []
        if device == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                _, logits = model(x)
                score = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                scores.extend(score)
                labels.extend(y.numpy())

        if device == "cuda":
            torch.cuda.empty_cache()

        val_eer, _ = compute_eer(np.array(labels), np.array(scores))
        print(f"\nEpoch {epoch}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val EER: {val_eer:.2f}%")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), existing_ckpt)
            print(f"  ==> Saved new best model (Val EER: {val_eer:.2f}%) to: {existing_ckpt}")

    # 6. Final Evaluation on Unseen Test Speakers
    print("\n==================================================")
    print(" Running Final Evaluation on Unseen Test Speakers")
    print("==================================================")
    if os.path.isfile(existing_ckpt):
        model.load_state_dict(torch.load(existing_ckpt, map_location=device))

    model.eval()
    t_scores, t_labels = [], []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            _, logits = model(x)
            score = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            t_scores.extend(score)
            t_labels.extend(y.numpy())

    final_test_eer, opt_thresh = compute_eer(np.array(t_labels), np.array(t_scores))
    preds = (np.array(t_scores) >= opt_thresh).astype(int)
    final_acc = (preds == np.array(t_labels)).mean() * 100.0

    print(f" Final Test EER on Unseen Indian Speakers: {final_test_eer:.2f}%")
    print(f" Final Test Accuracy:                      {final_acc:.2f}%")
    print(f" Optimal Decision Threshold:               {opt_thresh:.4f}")
    print("==================================================")

if __name__ == "__main__":
    train()

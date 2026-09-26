"""
train.py — Voice Shield Multilingual Training Entry Point
==========================================================
Edit train_config.yaml, then just run:

    python train.py

That's it. No flags, no arguments needed.
Optionally override the config file path:

    python train.py --config my_config.yaml
"""

import os
import sys
import argparse

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    print("[ERROR] PyYAML not installed. Run:  pip install pyyaml")
    sys.exit(1)

# ── Parse optional --config flag ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Voice Shield Multilingual Fine-Tuning")
parser.add_argument(
    "--config", default="train_config.yaml",
    help="Path to YAML config file (default: train_config.yaml)"
)
args = parser.parse_args()

config_path = args.config
if not os.path.isfile(config_path):
    print(f"[ERROR] Config file not found: {config_path}")
    print(f"        Expected: {os.path.abspath(config_path)}")
    sys.exit(1)

# ── Load config ────────────────────────────────────────────────────────────────
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Validate required field
if not cfg.get("indicfake_root") or cfg["indicfake_root"] == "/path/to/IndicFake":
    print("[ERROR] Please set 'indicfake_root' in train_config.yaml to your IndicFake dataset path.")
    sys.exit(1)

if not os.path.isdir(cfg["indicfake_root"]):
    print(f"[ERROR] indicfake_root directory not found: {cfg['indicfake_root']}")
    sys.exit(1)

# ── Build args namespace for train_multilingual ─────────────────────────────────
class TrainArgs:
    indicfake_root = cfg["indicfake_root"]
    languages      = ",".join(cfg.get("languages", ["hi", "ta"]))
    epochs         = int(cfg.get("epochs", 15))
    batch_size     = int(cfg.get("batch_size", 6))
    frontend_lr    = float(cfg.get("frontend_lr", 1e-6))
    backend_lr     = float(cfg.get("backend_lr", 1e-4))
    max_per_lang   = int(cfg.get("max_per_lang", 5000))
    english_ratio  = float(cfg.get("english_ratio", 0.30))

print("\n" + "=" * 60)
print("  Voice Shield — Multilingual Training")
print(f"  Config      : {os.path.abspath(config_path)}")
print(f"  Languages   : {cfg.get('languages', ['hi', 'ta'])}")
print(f"  Epochs      : {TrainArgs.epochs}")
print(f"  Batch size  : {TrainArgs.batch_size}")
print(f"  Dataset     : {TrainArgs.indicfake_root}")
print("=" * 60)

# ── Delegate to training module ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src", "pipeline"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from train_multilingual import train
train(TrainArgs)

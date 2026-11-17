"""Central configuration for the GCI Home Credit Default Risk pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Core paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
EXP_DIR = ROOT / "experiments"
LOG_DIR = EXP_DIR / "logs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR = ROOT / "output"
FOLDS_PATH = EXP_DIR / "folds.npy"
TRACKER_PATH = EXP_DIR / "tracker.csv"

# Reproducibility & learning rates
SEED = 42
EXP_LR = 0.06
FINAL_LR = 0.02
N_SPLITS = 5

# Leave one logical core free when running heavy training jobs
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

# Noise floor measured in P2 (abs(lgb_raw - lgb_raw_s7) mean fold AUC diff)
NOISE_FLOOR = 0.000072
MIN_GAIN = max(0.0002, NOISE_FLOOR)

# Feature engineering switches (frozen at freeze-features-v1)
ALL_FAMILIES = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
FROZEN_FAMILIES: list[str] = ["F2", "F5", "F6"]
USE_TEST_FOR_STATS = False
DROPPED_FEATURES: list[str] = []

# Tuned model configurations (populated in P5 / P8)
LGB_TUNED_PARAMS: dict = {}
XGB_USE_CODES: bool = False
CATBOOST_AVAILABLE: bool = True

FINAL_SEEDS = [42, 1024, 2025]
XGB_FINAL_SEEDS = [42, 1024]

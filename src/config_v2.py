"""Central configuration for the GCI Home Credit Default Risk v2 pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Core paths (strictly isolated from v1 experiments/ and submissions/)
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
EXP_DIR = ROOT / "experiments_v2"
LOG_DIR = EXP_DIR / "logs"
SUB_DIR = ROOT / "submissions_v2"
OUTPUT_DIR = ROOT / "output"

FOLDS_GROUP_PATH = EXP_DIR / "folds_group.npy"
FOLDS_DISCOVERY_PATH = EXP_DIR / "folds_discovery.npy"
FOLDS10_PATH = EXP_DIR / "folds10_group.npy"
FOLDS_V1_PATH = ROOT / "experiments" / "folds.npy"
FOLDS_PATH = FOLDS_GROUP_PATH  # Selected in P1(e) audit_v2
TRACKER_PATH = EXP_DIR / "tracker.csv"

# Reproducibility & learning rates
SEED = 42
EXP_LR = 0.06
FINAL_LR = 0.02
N_SPLITS = 5

# Leave one logical core free when running heavy training jobs (4 logical -> 3 threads)
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

# Noise floor measured in v1 P2
NOISE_FLOOR = 0.000072
MIN_GAIN = max(0.0002, NOISE_FLOOR)

# Grey-zone customer linkage switch (HARD RULE: must stay False unless explicitly approved)
ENABLE_LINKAGE: bool = False
USE_TEST_FOR_STATS: bool = False
# Constant zero-gain column identified in train.csv (only 1 zero in train, 0 in test; v1 P4 pruning tie 0.000000)
DROPPED_FEATURES: list[str] = ["FLAG_MOBIL"]

# v1 top-3 Optuna LightGBM parameter sets (from experiments/lgb_top_params.json)
_TOP_PARAMS_PATH = ROOT / "experiments" / "lgb_top_params.json"
if _TOP_PARAMS_PATH.exists():
    _raw_top = json.loads(_TOP_PARAMS_PATH.read_text(encoding="utf-8"))
    LGB_TOP_PARAMS: list[dict] = [dict(entry["params"]) for entry in _raw_top]
else:
    LGB_TOP_PARAMS = [
        {
            "max_depth": 4,
            "num_leaves": 15,
            "min_data_in_leaf": 93,
            "feature_fraction": 0.2416696493365781,
            "bagging_fraction": 0.8439312348371466,
            "lambda_l2": 2.2453924433747887,
            "lambda_l1": 3.441802045844477,
            "min_gain_to_split": 0.123735565793759,
            "bagging_freq": 1,
        },
        {
            "max_depth": 5,
            "num_leaves": 14,
            "min_data_in_leaf": 220,
            "feature_fraction": 0.34695683056752447,
            "bagging_fraction": 0.8953095439114199,
            "lambda_l2": 3.461375049980705,
            "lambda_l1": 4.411827940483147,
            "min_gain_to_split": 0.20236242982796337,
            "bagging_freq": 1,
        },
        {
            "max_depth": 4,
            "num_leaves": 15,
            "min_data_in_leaf": 62,
            "feature_fraction": 0.30802780464053403,
            "bagging_fraction": 0.72214448973048,
            "lambda_l2": 1.3872880445265083,
            "lambda_l1": 4.76364785272666,
            "min_gain_to_split": 0.24478047812649373,
            "bagging_freq": 1,
        },
    ]

# Default v2 model configs (updated as P4/P5/P6 progress)
CAT_FAST_PARAMS: dict = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "depth": 6,
    "learning_rate": 0.06,
    "border_count": 128,
    "l2_leaf_reg": 5.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "iterations": 2500,
    "early_stopping_rounds": 150,
}

XGB_V2_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 20,
    "reg_lambda": 10.0,
    "n_estimators": 5000,
    "early_stopping_rounds": 200,
}

FINAL_SEEDS = [42, 1024, 2025]
XGB_FINAL_SEEDS = [42, 1024]
CAT_FINAL_SEEDS = [42, 1024]
MLP_FINAL_SEEDS = [42, 1024]

# Selected LightGBM configuration for v2 (v1 Optuna Trial #11 / rank-3 in experiments/lgb_top_params.json,
# chosen via 5-fold CV on experiments_v2/folds_group.npy in P5)
LGB_FINAL_PARAMS: dict = dict(LGB_TOP_PARAMS[2])

# Frozen feature views (populated at P4 gate / freeze-features-v2)
VIEW_A_FAMILIES: list[str] = ["F2", "F5_LOO", "F6", "A1", "A6_INT"]
VIEW_B_FAMILIES: list[str] = ["F0", "F1", "F2", "F3", "F5_LOO", "F6", "A1", "A5", "A6_INT"]
VIEW_C_FAMILIES: list[str] = ["F2", "F5_LOO", "F6", "A1", "A6_INT"]
VIEW_USE_CODES: bool = True
VIEW_C_TE_COLS: tuple[str, ...] = (
    "OCC_X_ORG",
    "EDU_X_INC",
    "CON_X_GEN_X_FAM",
    "REGION_ID",
    "PRODUCT_CODE",
    "TERM_X_CON",
)
# Minimum category count in inner training fold before backing off to global target mean (A4 specification)
VIEW_C_TE_MIN_COUNT: int = 30
# Neighborhood size k for pattern-isolated Euclidean KNN target mean (Addendum A.4)
VIEW_C_KNN_KS: tuple[int, ...] = (500,)
VIEW_C_KNN_MODE: str = "v2"

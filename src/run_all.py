"""One-command deterministic end-to-end training and submission pipeline.

Reproduces the final multi-seed rank-averaged ensemble from raw CSVs in input/.
Usage:
    python src/run_all.py          # Full 7-model multi-seed ensemble (matches blend_final)
    python src/run_all.py --fast   # Sub-hour 6-model mode (1 fast CatBoost seed)
    python src/run_all.py --fresh  # Ignore component cache and retrain all models
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CAT_FINAL_PARAMS,
    CAT_FINAL_SEEDS,
    DROPPED_FEATURES,
    EXP_DIR,
    FINAL_BLEND_WEIGHTS,
    FINAL_LR,
    FINAL_SEEDS,
    FOLDS_PATH,
    FROZEN_FAMILIES,
    INPUT_DIR,
    LGB_TUNED_PARAMS,
    N_SPLITS,
    OUTPUT_DIR,
    SEED,
    SUB_DIR,
    USE_TEST_FOR_STATS,
    XGB_FINAL_PARAMS,
    XGB_FINAL_SEEDS,
    XGB_USE_CODES,
)
from src.cv_harness import write_submission
from src.features import build_features, wrap_fit_predict_with_te
from src.models import cat_model, lgb_model, xgb_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end GCI Home Credit pipeline.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Sub-hour mode: 3 LGB + 2 XGB + 1 faster CatBoost seed (lr=0.06, iter=2000).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear experiments/run_all_cache/ before running.",
    )
    parser.add_argument(
        "--out",
        default=str(SUB_DIR / "sub_final.csv"),
        help="Output submission CSV path.",
    )
    return parser.parse_args()


def _get_or_create_folds(train: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Load fixed folds or deterministically generate StratifiedKFold splits."""
    if FOLDS_PATH.exists():
        folds = np.load(FOLDS_PATH)
        if len(folds) == len(train):
            return folds
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = np.full(len(train), -1, dtype=np.int32)
    for fold_id, (_, va_idx) in enumerate(skf.split(train, y)):
        folds[va_idx] = fold_id
    np.save(FOLDS_PATH, folds)
    return folds


def _run_or_load_component(
    label: str,
    cache_dir: Path,
    fit_predict: Callable[
        [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
        tuple[np.ndarray, np.ndarray, int],
    ],
    X: pd.DataFrame,
    y: np.ndarray,
    Xte: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """Run fixed-fold CV (or resume from component cache) and return predictions and AUCs."""
    comp_dir = cache_dir / label
    oof_path = comp_dir / "oof.npy"
    test_path = comp_dir / "test.npy"
    meta_path = comp_dir / "meta.json"
    if oof_path.exists() and test_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        oof = np.load(oof_path)
        test_pred = np.load(test_path)
        print(
            f"[{label}] RESUMED from cache | pooled_oof_auc={meta['pooled_oof_auc']:.6f}",
            flush=True,
        )
        return oof, test_pred, meta["fold_aucs"], float(meta["pooled_oof_auc"])

    t0 = time.monotonic()
    unique_folds = sorted(int(f) for f in np.unique(folds))
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []

    for fold_id in unique_folds:
        va_mask = folds == fold_id
        tr_mask = ~va_mask
        vp, tp, best_iter = fit_predict(
            X.iloc[tr_mask], y[tr_mask], X.iloc[va_mask], y[va_mask], Xte
        )
        oof[va_mask] = np.asarray(vp, dtype=np.float64)
        test_pred += np.asarray(tp, dtype=np.float64) / len(unique_folds)
        fold_auc = float(roc_auc_score(y[va_mask], oof[va_mask]))
        fold_aucs.append(fold_auc)
        print(f"[{label}] fold {fold_id}: AUC={fold_auc:.6f} | best_iter={best_iter}", flush=True)

    pooled = float(roc_auc_score(y, oof))
    dt = time.monotonic() - t0
    comp_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    meta_path.write_text(
        json.dumps(
            {
                "label": label,
                "fold_aucs": fold_aucs,
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "pooled_oof_auc": pooled,
                "elapsed_sec": round(dt, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[{label}] DONE ({dt:.1f}s) | mean_fold_auc={np.mean(fold_aucs):.6f} | "
        f"pooled_oof_auc={pooled:.6f}",
        flush=True,
    )
    return oof, test_pred, fold_aucs, pooled


def main() -> None:
    args = parse_args()
    t_start = time.monotonic()

    cache_dir = EXP_DIR / ("run_all_cache_fast" if args.fast else "run_all_cache")
    if args.fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    y = train["TARGET"].to_numpy(dtype=np.int32)
    folds = _get_or_create_folds(train, y)

    use_f6 = "F6" in FROZEN_FAMILIES
    build_fams = [f for f in FROZEN_FAMILIES if f != "F6"]
    X, Xte = build_features(
        train,
        test,
        families=build_fams,
        use_test_for_stats=USE_TEST_FOR_STATS,
        dropped_features=DROPPED_FEATURES,
    )
    for forbidden in ("TARGET", "SK_ID_CURR"):
        assert forbidden not in X.columns and forbidden not in Xte.columns
    print(
        f"run_all.py: X={X.shape}, Xte={Xte.shape}, families={FROZEN_FAMILIES}, "
        f"use_test_for_stats={USE_TEST_FOR_STATS}, dropped={DROPPED_FEATURES}, fast={args.fast}",
        flush=True,
    )

    lgb_seeds = list(FINAL_SEEDS)
    xgb_seeds = list(XGB_FINAL_SEEDS)
    if args.fast:
        cat_seeds = [SEED]
        cat_params = {
            **CAT_FINAL_PARAMS,
            "iterations": 2000,
            "learning_rate": 0.06,
            "early_stopping_rounds": 150,
        }
    else:
        cat_seeds = list(CAT_FINAL_SEEDS)
        cat_params = dict(CAT_FINAL_PARAMS)

    family_oofs: dict[str, np.ndarray] = {}
    family_tests: dict[str, np.ndarray] = {}
    component_metrics: dict[str, dict] = {}

    # 1. LightGBM multi-seed
    lgb_oof_ranks = np.zeros(len(X), dtype=np.float64)
    lgb_test_ranks = np.zeros(len(Xte), dtype=np.float64)
    for seed in lgb_seeds:
        fp = lgb_model(params=LGB_TUNED_PARAMS, seed=seed, lr=FINAL_LR, deterministic=True)
        if use_f6:
            fp = wrap_fit_predict_with_te(fp, seed=seed)
        label = f"lgb_s{seed}"
        oof, tp, f_aucs, pooled = _run_or_load_component(
            label, cache_dir, fp, X, y, Xte, folds
        )
        lgb_oof_ranks += (rankdata(oof) / len(oof)) / len(lgb_seeds)
        lgb_test_ranks += (rankdata(tp) / len(tp)) / len(lgb_seeds)
        component_metrics[label] = {"fold_aucs": f_aucs, "pooled_oof_auc": pooled}
    family_oofs["lgb"] = lgb_oof_ranks
    family_tests["lgb"] = lgb_test_ranks

    # 2. CatBoost multi-seed
    cat_oof_ranks = np.zeros(len(X), dtype=np.float64)
    cat_test_ranks = np.zeros(len(Xte), dtype=np.float64)
    for seed in cat_seeds:
        fp = cat_model(params=cat_params, seed=seed)
        if use_f6:
            fp = wrap_fit_predict_with_te(fp, seed=seed)
        label = f"cat_s{seed}"
        oof, tp, f_aucs, pooled = _run_or_load_component(
            label, cache_dir, fp, X, y, Xte, folds
        )
        cat_oof_ranks += (rankdata(oof) / len(oof)) / len(cat_seeds)
        cat_test_ranks += (rankdata(tp) / len(tp)) / len(cat_seeds)
        component_metrics[label] = {"fold_aucs": f_aucs, "pooled_oof_auc": pooled}
    family_oofs["cat"] = cat_oof_ranks
    family_tests["cat"] = cat_test_ranks

    # 3. XGBoost multi-seed
    xgb_oof_ranks = np.zeros(len(X), dtype=np.float64)
    xgb_test_ranks = np.zeros(len(Xte), dtype=np.float64)
    for seed in xgb_seeds:
        fp = xgb_model(params=XGB_FINAL_PARAMS, seed=seed, use_codes=XGB_USE_CODES)
        if use_f6:
            fp = wrap_fit_predict_with_te(fp, seed=seed)
        label = f"xgb_s{seed}"
        oof, tp, f_aucs, pooled = _run_or_load_component(
            label, cache_dir, fp, X, y, Xte, folds
        )
        xgb_oof_ranks += (rankdata(oof) / len(oof)) / len(xgb_seeds)
        xgb_test_ranks += (rankdata(tp) / len(tp)) / len(xgb_seeds)
        component_metrics[label] = {"fold_aucs": f_aucs, "pooled_oof_auc": pooled}
    family_oofs["xgb"] = xgb_oof_ranks
    family_tests["xgb"] = xgb_test_ranks

    # Equal-weight rank blend across families
    oof_final = np.zeros(len(X), dtype=np.float64)
    test_final = np.zeros(len(Xte), dtype=np.float64)
    for fam, w in FINAL_BLEND_WEIGHTS.items():
        oof_final += w * family_oofs[fam]
        test_final += w * family_tests[fam]

    fold_aucs = [float(roc_auc_score(y[folds == f], oof_final[folds == f])) for f in range(N_SPLITS)]
    mean_fold_auc = float(np.mean(fold_aucs))
    pooled_oof_auc = float(roc_auc_score(y, oof_final))
    elapsed = time.monotonic() - t_start

    print("=" * 60, flush=True)
    print(f"FINAL RANK BLEND ({'FAST' if args.fast else 'FULL'}) | elapsed={elapsed:.1f}s", flush=True)
    for f, auc in enumerate(fold_aucs):
        print(f"  Fold {f}: {auc:.6f}", flush=True)
    print(f"  Mean fold AUC:  {mean_fold_auc:.6f}", flush=True)
    print(f"  Pooled OOF AUC: {pooled_oof_auc:.6f}", flush=True)
    print("=" * 60, flush=True)

    out_path = Path(args.out)
    meta = {
        "mode": "fast" if args.fast else "full",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 2),
        "families": FROZEN_FAMILIES,
        "dropped_features": DROPPED_FEATURES,
        "use_test_for_stats": USE_TEST_FOR_STATS,
        "blend_weights": FINAL_BLEND_WEIGHTS,
        "fold_aucs": fold_aucs,
        "mean_fold_auc": mean_fold_auc,
        "pooled_oof_auc": pooled_oof_auc,
        "components": component_metrics,
    }
    write_submission(test, test_final, out_path, meta=meta)

    if not args.fast and out_path.resolve() == (SUB_DIR / "sub_final.csv").resolve():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_df = pd.read_csv(out_path)
        out_df.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"Copied final submission to {OUTPUT_DIR / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()

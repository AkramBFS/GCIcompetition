"""Data integrity audit and fixed stratified CV fold generation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.config import EXP_DIR, FOLDS_PATH, INPUT_DIR, N_SPLITS, N_THREADS, SEED
from src.cv_harness import align_categories


def run_audit() -> dict:
    """Audit raw CSVs, generate fixed 5-fold splits if missing, and write audit_report.txt."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

    lines: list[str] = []
    red_flags: list[str] = []

    lines.append(f"=== GCI Home Credit Data Audit ({datetime.now(timezone.utc).isoformat()}) ===")
    lines.append(f"Train shape: {train.shape}")
    lines.append(f"Test shape:  {test.shape}")
    lines.append(f"Sample sub:  {sample_sub.shape}")

    # 1. Target & schema checks
    if "TARGET" in test.columns:
        red_flags.append("TARGET column present in test.csv!")
    if "TARGET" not in train.columns:
        red_flags.append("TARGET column missing from train.csv!")
    if not np.array_equal(test["SK_ID_CURR"].to_numpy(), sample_sub["SK_ID_CURR"].to_numpy()):
        red_flags.append("test.csv SK_ID_CURR does not match sample_submission.csv order!")

    feature_cols = [c for c in test.columns if c != "SK_ID_CURR"]
    train_feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    if feature_cols != train_feature_cols:
        red_flags.append(f"Feature column mismatch: train={train_feature_cols} vs test={feature_cols}")

    y = train["TARGET"].to_numpy(dtype=np.int32)
    lines.append(
        f"Target positive rate: {y.mean():.6f} ({int(y.sum())} positives / {len(y)} rows)"
    )

    # 2. ID uniqueness and overlap
    id_overlap = len(set(train["SK_ID_CURR"]).intersection(set(test["SK_ID_CURR"])))
    train_id_dups = int(train["SK_ID_CURR"].duplicated().sum())
    test_id_dups = int(test["SK_ID_CURR"].duplicated().sum())
    lines.append(
        f"SK_ID_CURR overlap={id_overlap}, train_id_dups={train_id_dups}, test_id_dups={test_id_dups}"
    )
    if id_overlap > 0 or train_id_dups > 0 or test_id_dups > 0:
        red_flags.append(
            f"ID integrity failure: overlap={id_overlap}, train_dups={train_id_dups}, test_dups={test_id_dups}"
        )

    # 3. Exact feature duplicate checks
    tr_dup_mask = train.duplicated(subset=feature_cols, keep=False)
    te_dup_mask = test.duplicated(subset=feature_cols, keep=False)
    tr_dup_count = int(tr_dup_mask.sum())
    te_dup_count = int(te_dup_mask.sum())
    conflicting_dups = 0
    if tr_dup_count > 0:
        grp_nunique = train.loc[tr_dup_mask].groupby(feature_cols, dropna=False)["TARGET"].nunique()
        conflicting_dups = int((grp_nunique > 1).sum())
    tr_keys = set(map(tuple, train[feature_cols].astype("object").fillna("__NA__").to_numpy()))
    te_keys = set(map(tuple, test[feature_cols].astype("object").fillna("__NA__").to_numpy()))
    cross_exact_dups = len(tr_keys & te_keys)
    lines.append(
        f"Exact feature duplicates: train={tr_dup_count} (conflicting_target_groups={conflicting_dups}), "
        f"test={te_dup_count}, train_test_exact_overlap={cross_exact_dups}"
    )
    if conflicting_dups > 0:
        red_flags.append(f"Found {conflicting_dups} exact-duplicate feature groups with conflicting labels")

    # 4. Same-person heuristic key check
    person_cols = [
        "CODE_GENDER",
        "DAYS_BIRTH",
        "DAYS_ID_PUBLISH",
        "DAYS_REGISTRATION",
        "NAME_EDUCATION_TYPE",
    ]
    within_train_person_rate = float(train.duplicated(subset=person_cols, keep=False).mean())
    tr_person_set = set(map(tuple, train[person_cols].fillna("__NA__").to_numpy()))
    te_person_hits = sum(
        row in tr_person_set for row in map(tuple, test[person_cols].fillna("__NA__").to_numpy())
    )
    test_to_train_person_rate = float(te_person_hits / len(test))
    lines.append(
        f"Same-person key {person_cols}: within-train rate={within_train_person_rate:.5f}, "
        f"test-to-train hit rate={test_to_train_person_rate:.5f}"
    )

    use_group_folds = within_train_person_rate > max(0.02, 2.0 * test_to_train_person_rate)
    lines.append(
        f"Fold strategy decision: {'StratifiedGroupKFold' if use_group_folds else 'StratifiedKFold'} "
        f"(n_splits={N_SPLITS}, random_state={SEED})"
    )

    # 5. Generate fixed folds once (never regenerate silently)
    if FOLDS_PATH.exists():
        folds = np.load(FOLDS_PATH)
        lines.append(f"Loaded existing fixed folds from {FOLDS_PATH}")
    else:
        if use_group_folds:
            groups = pd.util.hash_pandas_object(train[person_cols], index=False).to_numpy()
            sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            folds = np.full(len(train), -1, dtype=np.int32)
            for fold_id, (_, va_idx) in enumerate(sgkf.split(train, y, groups)):
                folds[va_idx] = fold_id
        else:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            folds = np.full(len(train), -1, dtype=np.int32)
            for fold_id, (_, va_idx) in enumerate(skf.split(train, y)):
                folds[va_idx] = fold_id
        np.save(FOLDS_PATH, folds)
        lines.append(f"Saved fixed folds to {FOLDS_PATH}")

    fold_rates = [float(y[folds == f].mean()) for f in range(N_SPLITS)]
    fold_sizes = [int((folds == f).sum()) for f in range(N_SPLITS)]
    lines.append(f"Fold sizes: {fold_sizes}")
    lines.append(f"Fold default rates: {[round(r, 6) for r in fold_rates]}")
    max_rate_diff = max(fold_rates) - min(fold_rates)
    lines.append(f"Max fold default rate spread: {max_rate_diff:.6f}")
    if max_rate_diff > 0.001:
        red_flags.append(f"Fold target rate spread {max_rate_diff:.6f} > 0.001")

    # 6. Single-column univariate AUC check
    lines.append("\n--- Univariate Feature AUCs (train non-missing) ---")
    high_auc_cols: list[tuple[str, float]] = []
    for col in feature_cols:
        s = train[col]
        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            if valid.nunique() > 1:
                auc = float(roc_auc_score(y[valid.index], valid))
                auc = max(auc, 1.0 - auc)
            else:
                auc = 0.5
        else:
            # Out-of-fold frequency/mean rank diagnostic via smoothed target rate
            oof_enc = np.zeros(len(train), dtype=np.float64)
            global_mean = float(y.mean())
            for f in range(N_SPLITS):
                tr_m = folds != f
                va_m = folds == f
                stats = (
                    pd.DataFrame({"cat": s.iloc[tr_m].fillna("__NA__"), "y": y[tr_m]})
                    .groupby("cat")["y"]
                    .agg(["count", "sum"])
                )
                enc_map = (stats["sum"] + 50.0 * global_mean) / (stats["count"] + 50.0)
                oof_enc[va_m] = s.iloc[va_m].fillna("__NA__").map(enc_map).fillna(global_mean)
            auc = float(roc_auc_score(y, oof_enc))
            auc = max(auc, 1.0 - auc)
        lines.append(f"  {col:30s}: {auc:.5f}")
        if auc > 0.70:
            high_auc_cols.append((col, auc))

    if high_auc_cols:
        red_flags.append(f"Single columns with AUC > 0.70: {high_auc_cols}")

    # 7. 5-fold LightGBM Adversarial Validation (train vs test)
    X_adv_tr, X_adv_te = align_categories(train[feature_cols], test[feature_cols])
    X_adv = pd.concat([X_adv_tr, X_adv_te], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(train), dtype=np.int32), np.ones(len(test), dtype=np.int32)])
    adv_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_adv = np.zeros(len(X_adv), dtype=np.float64)
    adv_imp = np.zeros(len(feature_cols), dtype=np.float64)
    for tr_i, va_i in adv_skf.split(X_adv, y_adv):
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=N_THREADS,
            verbose=-1,
        )
        clf.fit(X_adv.iloc[tr_i], y_adv[tr_i])
        oof_adv[va_i] = clf.predict_proba(X_adv.iloc[va_i])[:, 1]
        adv_imp += clf.booster_.feature_importance(importance_type="gain") / N_SPLITS

    adv_auc = float(roc_auc_score(y_adv, oof_adv))
    top_adv = sorted(zip(feature_cols, adv_imp), key=lambda x: x[1], reverse=True)[:10]
    lines.append(f"\n--- Adversarial Validation (Train vs Test) ---")
    lines.append(f"Adversarial 5-fold LightGBM AUC: {adv_auc:.5f}")
    lines.append("Top adversarial features (gain):")
    for fname, val in top_adv:
        lines.append(f"  {fname:30s}: {val:.2f}")
    if adv_auc > 0.55:
        red_flags.append(f"Adversarial AUC {adv_auc:.5f} > 0.55")

    lines.append("\n--- Red Flag Summary ---")
    if red_flags:
        for rf in red_flags:
            lines.append(f"RED FLAG: {rf}")
    else:
        lines.append("NONE: All audit checks passed with zero red flags.")

    report_text = "\n".join(lines) + "\n"
    report_path = EXP_DIR / "audit_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    return {
        "red_flags": red_flags,
        "adv_auc": adv_auc,
        "fold_rates": fold_rates,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = run_audit()
    if result["red_flags"]:
        sys.exit(2)

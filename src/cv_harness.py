"""Cross-validation harness, category alignment, experiment tracking, and submission utilities."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.config import EXP_DIR, EXP_LR, FOLDS_PATH, INPUT_DIR, N_THREADS, SEED, TRACKER_PATH

LGB_START: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": EXP_LR,
    "num_leaves": 24,
    "max_depth": 6,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.35,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 5.0,
    "n_estimators": 5000,
    "early_stopping_rounds": 200,
    "random_state": SEED,
    "n_jobs": N_THREADS,
    "verbose": -1,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load train, test, target vector y, and fixed CV folds."""
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Fixed folds file missing at {FOLDS_PATH}. Run `python src/audit.py` first."
        )
    train = pd.read_csv(INPUT_DIR / "train.csv")
    test = pd.read_csv(INPUT_DIR / "test.csv")
    y = train["TARGET"].to_numpy(dtype=np.int32)
    folds = np.load(FOLDS_PATH)
    if len(folds) != len(train):
        raise ValueError(f"Folds length {len(folds)} != train length {len(train)}")
    return train, test, y, folds


def align_categories(
    train_x: pd.DataFrame, test_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert all non-numeric columns to categorical dtype with identical category sets."""
    tr = train_x.copy()
    te = test_x.copy()
    for col in tr.columns:
        if not pd.api.types.is_numeric_dtype(tr[col]):
            tr_vals = tr[col].astype("object")
            te_vals = te[col].astype("object")
            tr_cats = {str(v) for v in tr_vals.dropna().unique()}
            te_cats = {str(v) for v in te_vals.dropna().unique()}
            cats = sorted(tr_cats | te_cats)
            dtype = pd.CategoricalDtype(categories=cats, ordered=False)
            tr_str = tr_vals.where(tr_vals.isna(), tr_vals.astype(str))
            te_str = te_vals.where(te_vals.isna(), te_vals.astype(str))
            tr[col] = pd.Series(pd.Categorical(tr_str, dtype=dtype), index=tr.index)
            te[col] = pd.Series(pd.Categorical(te_str, dtype=dtype), index=te.index)
    return tr, te


def _validate_features(X: pd.DataFrame, Xte: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> None:
    """Assert zero leakage columns and aligned schema across train and test."""
    for forbidden in ("TARGET", "SK_ID_CURR"):
        assert forbidden not in X.columns, f"Forbidden column {forbidden!r} found in X"
        assert forbidden not in Xte.columns, f"Forbidden column {forbidden!r} found in Xte"
    assert list(X.columns) == list(Xte.columns), "Train and test feature columns/order mismatch"
    assert len(X) == len(y) == len(folds), "Row count mismatch among X, y, and folds"


def _append_tracker(row: dict[str, Any]) -> None:
    """Append a single experiment summary row to experiments/tracker.csv."""
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "timestamp_utc",
        "n_rows",
        "n_features",
        "fold_0_auc",
        "fold_1_auc",
        "fold_2_auc",
        "fold_3_auc",
        "fold_4_auc",
        "mean_fold_auc",
        "std_fold_auc",
        "pooled_oof_auc",
        "mean_best_iter",
        "note",
    ]
    write_header = not TRACKER_PATH.exists()
    with TRACKER_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_cv(
    name: str,
    fit_predict: Callable[
        [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
        tuple[np.ndarray, np.ndarray, int],
    ],
    X: pd.DataFrame,
    y: np.ndarray,
    Xte: pd.DataFrame,
    folds: np.ndarray,
    note: str = "",
) -> dict[str, Any]:
    """Run fixed-fold CV, refuse overwrites, persist predictions/meta, and log to tracker."""
    _validate_features(X, Xte, y, folds)
    exp_path = EXP_DIR / name
    if exp_path.exists():
        raise FileExistsError(
            f"Experiment '{name}' already exists at {exp_path}. Overwriting is prohibited."
        )

    unique_folds = sorted(int(f) for f in np.unique(folds))
    oof = np.zeros(len(X), dtype=np.float64)
    test_preds = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    best_iters: list[int] = []
    fold_importances: dict[str, np.ndarray] = {}

    for fold_id in unique_folds:
        va_mask = folds == fold_id
        tr_mask = ~va_mask
        Xtr, ytr = X.iloc[tr_mask], y[tr_mask]
        Xva, yva = X.iloc[va_mask], y[va_mask]

        val_pred, test_pred, best_iter = fit_predict(Xtr, ytr, Xva, yva, Xte)
        val_pred = np.asarray(val_pred, dtype=np.float64)
        test_pred = np.asarray(test_pred, dtype=np.float64)
        if np.isnan(val_pred).any() or np.isnan(test_pred).any():
            raise ValueError(f"NaN predictions produced on fold {fold_id} of experiment {name}")

        oof[va_mask] = val_pred
        test_preds += test_pred / len(unique_folds)
        fold_auc = float(roc_auc_score(yva, val_pred))
        fold_aucs.append(fold_auc)
        best_iters.append(int(best_iter))

        imp = getattr(fit_predict, "last_feature_importances_", None)
        if imp is not None and len(imp) == X.shape[1]:
            fold_importances[f"fold_{fold_id}"] = np.asarray(imp, dtype=np.float64)

        print(
            f"[{name}] fold {fold_id}: AUC={fold_auc:.6f} | best_iter={best_iter}",
            flush=True,
        )

    mean_fold_auc = float(np.mean(fold_aucs))
    std_fold_auc = float(np.std(fold_aucs))
    pooled_oof_auc = float(roc_auc_score(y, oof))
    mean_best_iter = float(np.mean(best_iters))

    exp_path.mkdir(parents=True, exist_ok=False)
    np.save(exp_path / "oof.npy", oof)
    np.save(exp_path / "test.npy", test_preds)

    if len(fold_importances) == len(unique_folds):
        imp_df = pd.DataFrame({"feature": list(X.columns)})
        for k, vals in fold_importances.items():
            imp_df[k] = vals
        fold_cols = [f"fold_{fid}" for fid in unique_folds]
        imp_df["mean_gain"] = imp_df[fold_cols].mean(axis=1)
        imp_df = imp_df.sort_values("mean_gain", ascending=False).reset_index(drop=True)
        imp_df.to_csv(exp_path / "importance.csv", index=False)

    meta: dict[str, Any] = {
        "name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "fold_aucs": fold_aucs,
        "mean_fold_auc": mean_fold_auc,
        "std_fold_auc": std_fold_auc,
        "pooled_oof_auc": pooled_oof_auc,
        "best_iters": best_iters,
        "mean_best_iter": mean_best_iter,
        "note": note,
    }
    (exp_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tracker_row: dict[str, Any] = {
        "name": name,
        "timestamp_utc": meta["timestamp_utc"],
        "n_rows": meta["n_rows"],
        "n_features": meta["n_features"],
        "mean_fold_auc": f"{mean_fold_auc:.6f}",
        "std_fold_auc": f"{std_fold_auc:.6f}",
        "pooled_oof_auc": f"{pooled_oof_auc:.6f}",
        "mean_best_iter": f"{mean_best_iter:.1f}",
        "note": note,
    }
    for idx, val in enumerate(fold_aucs):
        tracker_row[f"fold_{idx}_auc"] = f"{val:.6f}"
    _append_tracker(tracker_row)

    print(
        f"[{name}] DONE | mean_fold_auc={mean_fold_auc:.6f} (+/- {std_fold_auc:.6f}) | "
        f"pooled_oof_auc={pooled_oof_auc:.6f}",
        flush=True,
    )
    return meta


def load_experiment_meta(name: str) -> dict[str, Any]:
    """Read meta.json for a saved experiment."""
    meta_path = EXP_DIR / name / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def compare(
    new: str,
    ref: str,
    min_gain: float = 0.0002,
    min_folds: int = 4,
) -> dict[str, Any]:
    """Compare paired per-fold AUCs of `new` vs `ref` and return keep/reject verdict."""
    m_new = load_experiment_meta(new)
    m_ref = load_experiment_meta(ref)
    aucs_new = m_new["fold_aucs"]
    aucs_ref = m_ref["fold_aucs"]
    diffs = [a - b for a, b in zip(aucs_new, aucs_ref)]
    mean_diff = float(np.mean(diffs))
    oof_diff = float(m_new["pooled_oof_auc"] - m_ref["pooled_oof_auc"])
    folds_improved = int(sum(d > 0 for d in diffs))
    keep = (mean_diff >= min_gain) and (folds_improved >= min_folds)
    verdict = "KEEP" if keep else "REJECT"

    print(f"=== Compare: {new} vs {ref} ===")
    for i, (r_val, n_val, d) in enumerate(zip(aucs_ref, aucs_new, diffs)):
        print(f"  Fold {i}: {r_val:.6f} -> {n_val:.6f} ({d:+.6f})")
    print(
        f"  Mean fold AUC: {m_ref['mean_fold_auc']:.6f} -> {m_new['mean_fold_auc']:.6f} "
        f"({mean_diff:+.6f})"
    )
    print(
        f"  Pooled OOF AUC: {m_ref['pooled_oof_auc']:.6f} -> {m_new['pooled_oof_auc']:.6f} "
        f"({oof_diff:+.6f})"
    )
    print(
        f"  Folds improved: {folds_improved}/{len(diffs)} | Rule: mean>={min_gain:+.6f}, "
        f"folds>={min_folds} -> {verdict}"
    )
    return {
        "new": new,
        "ref": ref,
        "fold_diffs": diffs,
        "mean_diff": mean_diff,
        "oof_diff": oof_diff,
        "folds_improved": folds_improved,
        "keep": keep,
        "verdict": verdict,
    }


def rank_blend_arrays(
    names: list[str],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rank-averaged OOF and test arrays across saved experiments."""
    if not names:
        raise ValueError("names must be non-empty")
    if weights is None:
        w = np.ones(len(names), dtype=np.float64) / len(names)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(names):
            raise ValueError("weights length must match names length")
        w = w / w.sum()

    oof_blend = None
    test_blend = None
    for name, wi in zip(names, w):
        oof_arr = np.load(EXP_DIR / name / "oof.npy")
        test_arr = np.load(EXP_DIR / name / "test.npy")
        oof_rank = rankdata(oof_arr) / len(oof_arr)
        test_rank = rankdata(test_arr) / len(test_arr)
        oof_blend = wi * oof_rank if oof_blend is None else oof_blend + wi * oof_rank
        test_blend = wi * test_rank if test_blend is None else test_blend + wi * test_rank
    assert oof_blend is not None and test_blend is not None
    return oof_blend, test_blend


def rank_blend(
    names: list[str],
    y: np.ndarray,
    weights: list[float] | None = None,
) -> tuple[float, np.ndarray]:
    """Return pooled OOF AUC and blended test ranks across saved experiments."""
    oof_blend, test_blend = rank_blend_arrays(names, weights=weights)
    auc = float(roc_auc_score(y, oof_blend))
    return auc, test_blend


def _git_short_hash() -> str:
    """Return current git short commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def write_submission(
    test_df: pd.DataFrame,
    pred: np.ndarray,
    path: str | Path,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Validate predictions against sample_submission.csv and write CSV + JSON sidecar."""
    pred_arr = np.asarray(pred, dtype=np.float64)
    sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

    if len(pred_arr) != len(sample_sub):
        raise ValueError(f"Prediction length {len(pred_arr)} != sample_submission {len(sample_sub)}")
    if not np.array_equal(test_df["SK_ID_CURR"].to_numpy(), sample_sub["SK_ID_CURR"].to_numpy()):
        raise ValueError("test_df SK_ID_CURR order does not match sample_submission.csv")
    if np.isnan(pred_arr).any() or np.isinf(pred_arr).any():
        raise ValueError("Predictions contain NaN or Inf")
    if (pred_arr < 0.0).any() or (pred_arr > 1.0).any():
        raise ValueError(
            f"Predictions out of [0, 1] bounds: min={pred_arr.min()}, max={pred_arr.max()}"
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = sample_sub.copy()
    sub["TARGET"] = pred_arr
    sub.to_csv(out_path, index=False)

    # Read back and verify schema integrity
    check_df = pd.read_csv(out_path)
    assert list(check_df.columns) == ["SK_ID_CURR", "TARGET"]
    assert len(check_df) == len(sample_sub)
    assert not check_df["TARGET"].isna().any()

    sidecar = {
        "submission_file": out_path.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_short_hash(),
        "n_rows": int(len(check_df)),
        "pred_min": float(pred_arr.min()),
        "pred_max": float(pred_arr.max()),
        "pred_mean": float(pred_arr.mean()),
    }
    if meta:
        sidecar.update(meta)
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Validated submission written to {out_path} (sidecar: {sidecar_path})", flush=True)
    return out_path

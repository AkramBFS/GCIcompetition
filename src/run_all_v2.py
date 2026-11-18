"""One-command deterministic end-to-end v2 training and submission pipeline.

Reproduces the finalized v2 20-model (10 model-view group) ensemble from raw CSVs in input/.
Usage:
    python src/run_all_v2.py                  # Full 20-model pipeline (--full)
    python src/run_all_v2.py --fast           # Sub-30m LightGBM subset (VIEW_A x3 + VIEW_B x1)
    python src/run_all_v2.py --fresh          # Clear cache before running
    python src/run_all_v2.py --enable-linkage # Grey-zone linkage variant (False by default)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audit_v2 import make_fingerprint_key
from src.config_v2 import (
    CAT_FAST_PARAMS,
    CAT_FINAL_SEEDS,
    DROPPED_FEATURES,
    ENABLE_LINKAGE,
    EXP_DIR,
    FINAL_LR,
    FINAL_SEEDS,
    FOLDS10_PATH,
    FOLDS_PATH,
    INPUT_DIR,
    LGB_FINAL_PARAMS,
    N_SPLITS,
    N_THREADS,
    OUTPUT_DIR,
    SEED,
    SUB_DIR,
    VIEW_A_FAMILIES,
    VIEW_B_FAMILIES,
    VIEW_C_FAMILIES,
    VIEW_C_KNN_KS,
    VIEW_C_KNN_MODE,
    VIEW_C_TE_COLS,
    VIEW_C_TE_MIN_COUNT,
    VIEW_USE_CODES,
    XGB_FINAL_SEEDS,
    XGB_V2_PARAMS,
)
from src.cv_harness import write_submission
from src.features import (
    build_features,
    wrap_fit_predict_with_knn,
    wrap_fit_predict_with_linkage,
    wrap_fit_predict_with_te,
)
from src.models import cat_model, lgb_model, xgb_model
from src.stacking_v2 import honest_blend_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end v2 GCI Home Credit pipeline.")
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--full",
        action="store_true",
        help="Full 20-model v2 pipeline (default).",
    )
    mode_grp.add_argument(
        "--fast",
        action="store_true",
        help="Sub-30m LightGBM subset (VIEW_A 3 seeds + VIEW_B seed 42).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and clear component cache before running.",
    )
    parser.add_argument(
        "--enable-linkage",
        action="store_true",
        default=ENABLE_LINKAGE,
        help="Enable grey-zone B1 linkage features (default False).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=N_THREADS,
        help="CPU thread count override.",
    )
    parser.add_argument(
        "--out",
        default=str(SUB_DIR / "sub_final_v2.csv"),
        help="Output submission CSV path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(EXP_DIR / "manifest.json"),
        help="Output manifest JSON path.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _compute_code_hash() -> str:
    h = hashlib.sha256()
    files = [
        ROOT_DIR / "src" / "config_v2.py",
        ROOT_DIR / "src" / "audit_v2.py",
        ROOT_DIR / "src" / "features.py",
        ROOT_DIR / "src" / "models.py",
        ROOT_DIR / "src" / "stacking_v2.py",
        ROOT_DIR / "src" / "run_all_v2.py",
    ]
    for fp in files:
        h.update(fp.name.encode("utf-8"))
        h.update(fp.read_bytes())
    return h.hexdigest()


def _git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _get_or_create_group_folds(train: pd.DataFrame, y: np.ndarray, n_splits: int, path: Path) -> np.ndarray:
    if path.exists():
        folds = np.load(path)
        if len(folds) == len(train):
            return folds
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    chosen_key = make_fingerprint_key(train, ("G", "REG", "B_ID", "B_REG"))
    group_codes, _ = pd.factorize(chosen_key, sort=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = np.full(len(train), -1, dtype=np.int32)
    for fid, (_, va_idx) in enumerate(sgkf.split(train, y, group_codes)):
        folds[va_idx] = fid
    np.save(path, folds)
    return folds


def _wrap_fp(fp: Any, view: str, seed: int, use_linkage: bool, knn_cache_dir: Path | None) -> Any:
    if view == "C":
        fp = wrap_fit_predict_with_te(
            fp, cols=VIEW_C_TE_COLS, alpha=50.0, min_count=VIEW_C_TE_MIN_COUNT, seed=seed
        )
    fp = wrap_fit_predict_with_te(
        fp, cols=("ORGANIZATION_TYPE", "OCCUPATION_TYPE"), alpha=50.0, min_count=0, seed=seed
    )
    if view == "C":
        fp = wrap_fit_predict_with_knn(
            fp, k_values=VIEW_C_KNN_KS, mode=VIEW_C_KNN_MODE, seed=seed, cache_dir=knn_cache_dir
        )
    if use_linkage:
        fp = wrap_fit_predict_with_linkage(fp)
    return fp


def _run_or_load_component(
    label: str,
    comp_cfg: dict[str, Any],
    code_hash: str,
    data_hash: str,
    cache_dir: Path,
    fit_predict: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    Xte: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    key_input = json.dumps(comp_cfg, sort_keys=True) + "|" + code_hash + "|" + data_hash
    comp_hash = hashlib.sha256(key_input.encode("utf-8")).hexdigest()[:16]
    comp_dir = cache_dir / f"{label}_{comp_hash}"
    oof_p = comp_dir / "oof.npy"
    test_p = comp_dir / "test.npy"
    meta_p = comp_dir / "meta.json"
    if oof_p.exists() and test_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        print(f"[{label}] RESUMED from cache ({comp_hash}) | pooled={meta['pooled_oof_auc']:.6f}", flush=True)
        return np.load(oof_p), np.load(test_p), meta

    t0 = time.monotonic()
    unique_folds = sorted(int(f) for f in np.unique(folds))
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    fold_aucs: list[float] = []
    for fid in unique_folds:
        va_m = folds == fid
        tr_m = ~va_m
        vp, tp, best_iter = fit_predict(X.iloc[tr_m], y[tr_m], X.iloc[va_m], y[va_m], Xte)
        oof[va_m] = np.asarray(vp, dtype=np.float64)
        test_pred += np.asarray(tp, dtype=np.float64) / len(unique_folds)
        fauc = float(roc_auc_score(y[va_m], oof[va_m]))
        fold_aucs.append(fauc)
        print(f"[{label}] fold {fid}: AUC={fauc:.6f} | best_iter={best_iter}", flush=True)

    pooled = float(roc_auc_score(y, oof))
    dt = time.monotonic() - t0
    comp_dir.mkdir(parents=True, exist_ok=True)
    np.save(oof_p, oof)
    np.save(test_p, test_pred)
    meta = {
        "label": label,
        "comp_hash": comp_hash,
        "config": comp_cfg,
        "fold_aucs": fold_aucs,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "pooled_oof_auc": pooled,
        "elapsed_sec": round(dt, 2),
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{label}] DONE ({dt:.1f}s) | mean={meta['mean_fold_auc']:.6f} | pooled={pooled:.6f}", flush=True)
    return oof, test_pred, meta


def main() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    args = parse_args()
    t_start = time.monotonic()

    # Override N_THREADS cleanly inside models module if --threads specified
    import src.models as models_mod

    models_mod.N_THREADS = int(args.threads)

    cache_name = "run_all_cache_v2_fast" if args.fast else "run_all_cache_v2"
    if args.enable_linkage:
        cache_name += "_linkage"
    cache_dir = EXP_DIR / cache_name
    knn_cache_dir = cache_dir / "knn_cache"
    if args.fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_csv = INPUT_DIR / "train.csv"
    test_csv = INPUT_DIR / "test.csv"
    sub_csv = INPUT_DIR / "sample_submission.csv"
    input_hashes = {
        "train.csv": _sha256_file(train_csv),
        "test.csv": _sha256_file(test_csv),
        "sample_submission.csv": _sha256_file(sub_csv),
    }
    data_hash = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode("utf-8")).hexdigest()
    code_hash = _compute_code_hash()

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    y = train["TARGET"].to_numpy(dtype=np.int32)
    folds5 = _get_or_create_group_folds(train, y, N_SPLITS, FOLDS_PATH)
    folds10 = _get_or_create_group_folds(train, y, 10, FOLDS10_PATH) if not args.fast else None

    Xa, Xtea = build_features(
        train, test, families=[f for f in VIEW_A_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES
    )
    Xb, Xteb = build_features(
        train, test, families=[f for f in VIEW_B_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES
    )
    X_views: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {"A": (Xa, Xtea), "B": (Xb, Xteb)}
    if not args.fast:
        Xc, Xtec = build_features(
            train, test, families=[f for f in VIEW_C_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES
        )
        X_views["C"] = (Xc, Xtec)

    for v, (Xv, Xtev) in X_views.items():
        for forbidden in ("TARGET", "SK_ID_CURR"):
            assert forbidden not in Xv.columns and forbidden not in Xtev.columns
        assert list(Xv.columns) == list(Xtev.columns)

    # Component plan
    specs: list[tuple[str, str, str, str, int, int, dict[str, Any]]] = []
    if args.fast:
        for s in FINAL_SEEDS:
            specs.append((f"lgb_v2_viewA_s{s}", "LGB_VIEW_A", "A", "lgb", s, 5, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
        specs.append(("lgb_v2_viewB_s42", "LGB_VIEW_B", "B", "lgb", 42, 5, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
    elif args.enable_linkage:
        specs.append(("lgb_v2_viewA_link_s42", "LGB_VIEW_A_LINK", "A", "lgb", 42, 5, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
        specs.append(("xgb_v2_viewA_link_s42", "XGB_VIEW_A_LINK", "A", "xgb", 42, 5, dict(XGB_V2_PARAMS)))
        specs.append(("cat_v2_viewA_link_s42", "CAT_VIEW_A_LINK", "A", "cat", 42, 5, dict(CAT_FAST_PARAMS)))
    else:
        for s in FINAL_SEEDS:
            specs.append((f"lgb_v2_viewA_s{s}", "LGB_VIEW_A", "A", "lgb", s, 5, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
        for s in FINAL_SEEDS:
            specs.append((f"lgb_v2_viewB_s{s}", "LGB_VIEW_B", "B", "lgb", s, 5, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
        for s in FINAL_SEEDS:
            specs.append((f"lgb_v2_viewC_s{s}", "LGB_VIEW_C", "C", "lgb", s, 5, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
        specs.append(("lgb_v2_viewA_10fold", "LGB_VIEW_A_10F", "A", "lgb", 42, 10, {**LGB_FINAL_PARAMS, "lr": FINAL_LR}))
        for v in ("A", "B", "C"):
            for s in XGB_FINAL_SEEDS:
                specs.append((f"xgb_v2_view{v}_s{s}", f"XGB_VIEW_{v}", v, "xgb", s, 5, dict(XGB_V2_PARAMS)))
        for s in CAT_FINAL_SEEDS:
            specs.append((f"cat_v2_viewA_fast_s{s}", "CAT_VIEW_A", "A", "cat", s, 5, dict(CAT_FAST_PARAMS)))
        specs.append(("cat_v2_viewB_fast_s42", "CAT_VIEW_B", "B", "cat", 42, 5, dict(CAT_FAST_PARAMS)))
        specs.append(("cat_v2_viewC_fast_s42", "CAT_VIEW_C", "C", "cat", 42, 5, dict(CAT_FAST_PARAMS)))

    group_preds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    comp_metrics: dict[str, Any] = {}

    for label, grp, v, kind, seed, n_f, params in specs:
        Xv, Xtev = X_views[v]
        use_f = folds10 if n_f == 10 else folds5
        assert use_f is not None
        if kind == "lgb":
            p_clean = {k: val for k, val in params.items() if k != "lr"}
            fp = lgb_model(params=p_clean, seed=seed, lr=float(params.get("lr", FINAL_LR)), deterministic=True, use_codes=VIEW_USE_CODES)
        elif kind == "xgb":
            fp = xgb_model(params=params, seed=seed, use_codes=True)
        else:
            fp = cat_model(params=params, seed=seed)

        fp = _wrap_fp(fp, view=v, seed=seed, use_linkage=args.enable_linkage, knn_cache_dir=knn_cache_dir)
        cfg_hash_obj = {
            "label": label,
            "view": v,
            "kind": kind,
            "seed": seed,
            "n_folds": n_f,
            "threads": int(args.threads),
            "enable_linkage": bool(args.enable_linkage),
            "params": params,
        }
        oof, tp, meta = _run_or_load_component(
            label, cfg_hash_obj, code_hash, data_hash, cache_dir, fp, Xv, y, Xtev, use_f
        )
        comp_metrics[label] = {
            "group": grp,
            "view": v,
            "seed": seed,
            "n_folds": n_f,
            "pooled_oof_auc": meta["pooled_oof_auc"],
            "mean_fold_auc": meta["mean_fold_auc"],
            "elapsed_sec": meta["elapsed_sec"],
        }
        group_preds.setdefault(grp, []).append((oof, tp))

    oof_pool: dict[str, np.ndarray] = {}
    test_pool: dict[str, np.ndarray] = {}
    for grp, pairs in group_preds.items():
        oof_pool[grp] = sum(rankdata(o) / len(y) for o, _ in pairs) / len(pairs)
        test_pool[grp] = sum(rankdata(t) / len(test) for _, t in pairs) / len(pairs)

    blend_res = honest_blend_cv(oof_pool, test_pool, y, folds5, run_et_restacker=False, seed=SEED)
    eq_info = blend_res["equal_rank"]
    oof_final = eq_info["oof"]
    test_final = np.clip(eq_info["test"], 0.0, 1.0)
    fold_aucs = eq_info["fold_aucs"]
    mean_fold_auc = eq_info["mean_fold_auc"]
    pooled_oof_auc = eq_info["pooled_oof_auc"]
    elapsed = time.monotonic() - t_start

    print("=" * 64, flush=True)
    print(f"FINAL v2 ENSEMBLE ({'FAST' if args.fast else 'FULL'} | linkage={args.enable_linkage}) | elapsed={elapsed:.1f}s", flush=True)
    for f_i, auc in enumerate(fold_aucs):
        print(f"  Fold {f_i}: {auc:.6f}", flush=True)
    print(f"  Mean fold AUC:  {mean_fold_auc:.6f}", flush=True)
    print(f"  Pooled OOF AUC: {pooled_oof_auc:.6f}", flush=True)
    print("=" * 64, flush=True)

    import catboost
    import lightgbm
    import scipy
    import sklearn
    import xgboost

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "fast" if args.fast else "full",
        "enable_linkage": bool(args.enable_linkage),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
        },
        "n_threads": int(args.threads),
        "seeds": {
            "global": SEED,
            "lgb": FINAL_SEEDS,
            "xgb": XGB_FINAL_SEEDS,
            "cat": CAT_FINAL_SEEDS,
        },
        "input_sha256": input_hashes,
        "code_hash": code_hash,
        "final_blend_description": f"Equal-weight fold-isolated rank blend across {len(oof_pool)} model-view groups ({len(specs)} models)",
        "pool_groups": {k: len(v) for k, v in group_preds.items()},
        "fold_aucs": fold_aucs,
        "mean_fold_auc": mean_fold_auc,
        "pooled_oof_auc": pooled_oof_auc,
        "total_runtime_sec": round(elapsed, 2),
        "components": comp_metrics,
    }

    out_path = Path(args.out)
    write_submission(test, test_final, out_path, meta=manifest)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}", flush=True)

    if not args.fast and not args.enable_linkage and out_path.resolve() == (SUB_DIR / "sub_final_v2.csv").resolve():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(out_path).to_csv(OUTPUT_DIR / "submission_v2.csv", index=False)


if __name__ == "__main__":
    main()

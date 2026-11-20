"""Resumable Optuna hyperparameter tuning for LightGBM on fixed CV folds."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DROPPED_FEATURES,
    EXP_DIR,
    EXP_LR,
    FROZEN_FAMILIES,
    N_THREADS,
    SEED,
    USE_TEST_FOR_STATS,
)
from src.cv_harness import LGB_START, load_data
from src.features import build_features, wrap_fit_predict_with_te
from src.models import lgb_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters with Optuna.")
    parser.add_argument("--n-trials", type=int, default=25, help="Total target trials in study.")
    parser.add_argument("--timeout", type=int, default=7000, help="Max study runtime in seconds.")
    parser.add_argument(
        "--trial-cap-sec",
        type=int,
        default=600,
        help="Soft per-trial time cap in seconds (prunes remaining folds if exceeded).",
    )
    parser.add_argument("--study-name", default="lgb_home_credit_v1", help="Optuna study name.")
    return parser.parse_args()


def _export_results(study: optuna.Study) -> None:
    """Write Optuna trials CSV and top-3 parameter JSON."""
    df = study.trials_dataframe()
    csv_path = EXP_DIR / "optuna_lgb.csv"
    df.to_csv(csv_path, index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: float(t.value or -1.0), reverse=True)
    top3: list[dict[str, Any]] = []
    for rank_idx, trial in enumerate(completed[:3], start=1):
        params = dict(trial.params)
        params["bagging_freq"] = 1
        params["num_leaves"] = min(int(params["num_leaves"]), 2 ** int(params["max_depth"]) - 1)
        top3.append(
            {
                "rank": rank_idx,
                "trial_number": trial.number,
                "mean_fold_auc": float(trial.value),
                "pooled_oof_auc": trial.user_attrs.get("pooled_oof_auc"),
                "fold_aucs": trial.user_attrs.get("fold_aucs"),
                "mean_best_iter": trial.user_attrs.get("mean_best_iter"),
                "params": params,
            }
        )
    top_path = EXP_DIR / "lgb_top_params.json"
    top_path.write_text(json.dumps(top3, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    train, test, y, folds = load_data()
    use_f6 = "F6" in FROZEN_FAMILIES
    build_fams = [f for f in FROZEN_FAMILIES if f != "F6"]
    X, Xte = build_features(
        train,
        test,
        families=build_fams,
        use_test_for_stats=USE_TEST_FOR_STATS,
        dropped_features=DROPPED_FEATURES,
    )
    print(
        f"Optuna tuning LightGBM on shape={X.shape}, families={FROZEN_FAMILIES}, "
        f"use_test_for_stats={USE_TEST_FOR_STATS}, dropped={len(DROPPED_FEATURES)}, "
        f"n_threads={N_THREADS}",
        flush=True,
    )

    storage_url = f"sqlite:///{(EXP_DIR / 'optuna_lgb.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    existing_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining_trials = max(0, args.n_trials - existing_complete)
    print(
        f"Existing completed trials={existing_complete}, running {remaining_trials} more.",
        flush=True,
    )
    if remaining_trials == 0:
        _export_results(study)
        return

    unique_folds = sorted(int(f) for f in np.unique(folds))

    def objective(trial: optuna.Trial) -> float:
        t0 = time.monotonic()
        max_depth = trial.suggest_int("max_depth", 4, 8)
        raw_leaves = trial.suggest_int("num_leaves", 12, 48)
        eff_leaves = min(raw_leaves, 2**max_depth - 1)
        params: dict[str, Any] = {
            **LGB_START,
            "learning_rate": EXP_LR,
            "max_depth": max_depth,
            "num_leaves": eff_leaves,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 400),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.7),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 30.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
        }

        fp = lgb_model(params=params, seed=SEED, lr=EXP_LR, deterministic=False)
        if use_f6:
            fp = wrap_fit_predict_with_te(fp, seed=SEED)

        oof = np.zeros(len(X), dtype=np.float64)
        fold_aucs: list[float] = []
        best_iters: list[int] = []

        for fold_id in unique_folds:
            va_mask = folds == fold_id
            tr_mask = ~va_mask
            val_pred, _, best_iter = fp(
                X.iloc[tr_mask], y[tr_mask], X.iloc[va_mask], y[va_mask], Xte
            )
            oof[va_mask] = val_pred
            fold_auc = float(roc_auc_score(y[va_mask], val_pred))
            fold_aucs.append(fold_auc)
            best_iters.append(int(best_iter))
            trial.report(float(np.mean(fold_aucs)), step=fold_id)

            if (time.monotonic() - t0) > args.trial_cap_sec and fold_id < unique_folds[-1]:
                print(
                    f"[Trial {trial.number}] Pruned at fold {fold_id} due to time cap.",
                    flush=True,
                )
                raise optuna.TrialPruned()

        mean_auc = float(np.mean(fold_aucs))
        pooled_auc = float(roc_auc_score(y, oof))
        elapsed = time.monotonic() - t0
        trial.set_user_attr("fold_aucs", fold_aucs)
        trial.set_user_attr("pooled_oof_auc", pooled_auc)
        trial.set_user_attr("mean_best_iter", float(np.mean(best_iters)))
        trial.set_user_attr("elapsed_sec", round(elapsed, 2))
        trial.set_user_attr("effective_num_leaves", eff_leaves)
        print(
            f"[Trial {trial.number}] mean_auc={mean_auc:.6f} pooled={pooled_auc:.6f} "
            f"elapsed={elapsed:.1f}s params={params}",
            flush=True,
        )
        return mean_auc

    def _callback(study_obj: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        _export_results(study_obj)

    study.optimize(
        objective,
        n_trials=remaining_trials,
        timeout=args.timeout,
        callbacks=[_callback],
    )
    _export_results(study)
    print(f"Best trial #{study.best_trial.number}: mean_fold_auc={study.best_value:.6f}", flush=True)


if __name__ == "__main__":
    main()

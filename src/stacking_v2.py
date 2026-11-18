"""Strict 5-fold Honest Blend-CV stacking and blending module for GCI Home Credit v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config_v2 import N_THREADS, SEED


def fit_apply_rank(tr_vals: np.ndarray, eval_vals_list: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Map values to [1e-5, 1-1e-5] empirical ranks fitted strictly on training array `tr_vals`."""
    sorted_tr = np.sort(np.asarray(tr_vals, dtype=np.float64))
    n = float(len(sorted_tr))

    def _transform(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        lo = np.searchsorted(sorted_tr, a, side="left")
        hi = np.searchsorted(sorted_tr, a, side="right")
        r = 0.5 * (lo + hi) / n
        return np.clip(r, 1e-5, 1.0 - 1e-5)

    return _transform(tr_vals), [_transform(ev) for ev in eval_vals_list]


def _hill_climb_weights(ranks_tr: np.ndarray, y_tr: np.ndarray, max_steps: int = 25) -> np.ndarray:
    """Greedy forward selection with replacement (Caruana) maximizing AUC on training folds."""
    n_models = ranks_tr.shape[1]
    counts = np.zeros(n_models, dtype=np.float64)
    aucs = [float(roc_auc_score(y_tr, ranks_tr[:, j])) for j in range(n_models)]
    best_idx = int(np.argmax(aucs))
    counts[best_idx] += 1.0
    cur_sum = ranks_tr[:, best_idx].copy()
    best_auc = aucs[best_idx]

    for step in range(2, max_steps + 1):
        cand_best_j = -1
        cand_best_auc = best_auc
        for j in range(n_models):
            blend = (cur_sum + ranks_tr[:, j]) / float(step)
            auc = float(roc_auc_score(y_tr, blend))
            if auc > cand_best_auc + 1e-7:
                cand_best_auc = auc
                cand_best_j = j
        if cand_best_j < 0:
            break
        counts[cand_best_j] += 1.0
        cur_sum += ranks_tr[:, cand_best_j]
        best_auc = cand_best_auc

    return counts / counts.sum()


def honest_blend_cv(
    oof_dict: dict[str, np.ndarray],
    test_dict: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    raw_train: pd.DataFrame | None = None,
    raw_test: pd.DataFrame | None = None,
    equal_weights: dict[str, float] | None = None,
    hill_steps: int = 25,
    logreg_cs: tuple[float, ...] = (0.01, 0.1, 1.0),
    run_et_restacker: bool = True,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Evaluate blending and stacking methods via strict out-of-fold Honest Blend-CV.

    For every outer fold `f`, all empirical rank normalizers, greedy weights, logistic
    regression stackers, ExtraTrees restackers, and raw column medians are fitted ONLY
    on `folds != f` rows and scored on `folds == f`.
    """
    names = list(oof_dict.keys())
    if not names:
        raise ValueError("oof_dict must be non-empty")
    n_rows = len(y)
    n_test = len(next(iter(test_dict.values())))
    unique_folds = sorted(int(f) for f in np.unique(folds))
    n_folds = len(unique_folds)

    if equal_weights is None:
        eq_w = np.ones(len(names), dtype=np.float64) / len(names)
    else:
        eq_w = np.array([float(equal_weights.get(k, 0.0)) for k in names], dtype=np.float64)
        eq_w = eq_w / eq_w.sum()

    # Prepare optional 3 raw columns for ExtraTrees restacker (Addendum A.5)
    raw_cols_tr: np.ndarray | None = None
    raw_cols_te: np.ndarray | None = None
    if run_et_restacker and raw_train is not None and raw_test is not None:
        def _extract_3(df: pd.DataFrame) -> np.ndarray:
            ct = (
                df["AMT_CREDIT"].astype("float64")
                / df["AMT_ANNUITY"].astype("float64").replace(0.0, np.nan)
            )
            return np.column_stack(
                [
                    df["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64),
                    ct.to_numpy(dtype=np.float64),
                    df["EXT_SOURCE_2"].to_numpy(dtype=np.float64),
                ]
            )

        raw_cols_tr = _extract_3(raw_train)
        raw_cols_te = _extract_3(raw_test)

    method_names = ["equal_rank", "hill_climber"] + [f"logreg_c{c}" for c in logreg_cs]
    if raw_cols_tr is not None:
        method_names.append("et_restacker")

    oofs: dict[str, np.ndarray] = {m: np.zeros(n_rows, dtype=np.float64) for m in method_names}
    tests: dict[str, np.ndarray] = {m: np.zeros(n_test, dtype=np.float64) for m in method_names}
    fold_details: dict[str, Any] = {"hill_weights": [], "logreg_coefs": {}}

    for f in unique_folds:
        va_m = folds == f
        tr_m = ~va_m
        y_tr = y[tr_m]

        r_tr = np.zeros((int(tr_m.sum()), len(names)), dtype=np.float64)
        r_va = np.zeros((int(va_m.sum()), len(names)), dtype=np.float64)
        r_te = np.zeros((n_test, len(names)), dtype=np.float64)

        for j, name in enumerate(names):
            tr_r, (va_r, te_r) = fit_apply_rank(
                oof_dict[name][tr_m], [oof_dict[name][va_m], test_dict[name]]
            )
            r_tr[:, j] = tr_r
            r_va[:, j] = va_r
            r_te[:, j] = te_r

        # (a) Equal / configured rank average (fitted rank scale per fold)
        oofs["equal_rank"][va_m] = r_va @ eq_w
        tests["equal_rank"] += (r_te @ eq_w) / n_folds

        # (b) Hill climber (<= 25 steps)
        hw = _hill_climb_weights(r_tr, y_tr, max_steps=hill_steps)
        fold_details["hill_weights"].append({n: float(w) for n, w in zip(names, hw) if w > 0})
        oofs["hill_climber"][va_m] = r_va @ hw
        tests["hill_climber"] += (r_te @ hw) / n_folds

        # (c) Logistic stacker on logit of ranks
        l_tr = logit(r_tr)
        l_va = logit(r_va)
        l_te = logit(r_te)
        for c_val in logreg_cs:
            mname = f"logreg_c{c_val}"
            clf = LogisticRegression(C=c_val, max_iter=500, solver="lbfgs", random_state=seed)
            clf.fit(l_tr, y_tr)
            oofs[mname][va_m] = clf.predict_proba(l_va)[:, 1]
            tests[mname] += clf.predict_proba(l_te)[:, 1] / n_folds

        # (d) ExtraTrees restacker on top-7 logit-ranks + 3 raw cols (training-fold median imputed)
        if raw_cols_tr is not None and raw_cols_te is not None:
            tr_aucs = [float(roc_auc_score(y_tr, r_tr[:, j])) for j in range(len(names))]
            top7_idx = sorted(range(len(names)), key=lambda j: tr_aucs[j], reverse=True)[:7]
            meds = np.nanmedian(raw_cols_tr[tr_m], axis=0)

            def _fill_med(arr: np.ndarray) -> np.ndarray:
                out = arr.copy()
                for c_idx in range(out.shape[1]):
                    nan_m = ~np.isfinite(out[:, c_idx])
                    if nan_m.any():
                        out[nan_m, c_idx] = meds[c_idx]
                return out

            et_X_tr = np.hstack([l_tr[:, top7_idx], _fill_med(raw_cols_tr[tr_m])])
            et_X_va = np.hstack([l_va[:, top7_idx], _fill_med(raw_cols_tr[va_m])])
            et_X_te = np.hstack([l_te[:, top7_idx], _fill_med(raw_cols_te)])
            et = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=1000,
                random_state=seed,
                n_jobs=N_THREADS,
            )
            et.fit(et_X_tr, y_tr)
            oofs["et_restacker"][va_m] = et.predict_proba(et_X_va)[:, 1]
            tests["et_restacker"] += et.predict_proba(et_X_te)[:, 1] / n_folds

    # Pick best logreg C by Honest Blend-CV mean fold AUC
    best_lr_name = max(
        [f"logreg_c{c}" for c in logreg_cs],
        key=lambda m: float(np.mean([roc_auc_score(y[folds == f], oofs[m][folds == f]) for f in unique_folds])),
    )

    # (e) Meta-blends of {hill_climber, best_logreg, et_restacker} using fold-wise rank transform
    stack_cands = ["hill_climber", best_lr_name]
    if "et_restacker" in oofs:
        stack_cands.append("et_restacker")

    def _add_meta_blend(meta_name: str, members: list[str]) -> None:
        m_oof = np.zeros(n_rows, dtype=np.float64)
        m_te = np.zeros(n_test, dtype=np.float64)
        for f in unique_folds:
            va_m = folds == f
            tr_m = ~va_m
            for mem in members:
                _, (va_r, te_r) = fit_apply_rank(oofs[mem][tr_m], [oofs[mem][va_m], tests[mem]])
                m_oof[va_m] += va_r / len(members)
                m_te += (te_r / len(members)) / n_folds
        oofs[meta_name] = m_oof
        tests[meta_name] = m_te

    if len(stack_cands) >= 2:
        ranked_cands = sorted(
            stack_cands,
            key=lambda m: float(np.mean([roc_auc_score(y[folds == f], oofs[m][folds == f]) for f in unique_folds])),
            reverse=True,
        )
        _add_meta_blend("meta_top2", ranked_cands[:2])
        if len(ranked_cands) == 3:
            _add_meta_blend("meta_top3", ranked_cands[:3])

    results: dict[str, dict[str, Any]] = {}
    eq_fold_aucs = [float(roc_auc_score(y[folds == f], oofs["equal_rank"][folds == f])) for f in unique_folds]
    for mname, m_oof in oofs.items():
        f_aucs = [float(roc_auc_score(y[folds == f], m_oof[folds == f])) for f in unique_folds]
        diffs = [a - b for a, b in zip(f_aucs, eq_fold_aucs)]
        results[mname] = {
            "fold_aucs": f_aucs,
            "mean_fold_auc": float(np.mean(f_aucs)),
            "std_fold_auc": float(np.std(f_aucs)),
            "pooled_oof_auc": float(roc_auc_score(y, m_oof)),
            "diff_vs_equal": float(np.mean(diffs)),
            "folds_improved_vs_equal": int(sum(d > 0 for d in diffs)),
            "oof": m_oof,
            "test": tests[mname],
            "details": fold_details if mname == "hill_climber" else {},
        }
    return results

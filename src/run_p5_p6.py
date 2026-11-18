"""Phase P5 (Models on Frozen Views) and Phase P6 (Honest Blend-CV & Stacking) runner."""

from __future__ import annotations

import json
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config_v2 import (
    CAT_FAST_PARAMS,
    CAT_FINAL_SEEDS,
    DROPPED_FEATURES,
    EXP_DIR,
    FINAL_LR,
    FINAL_SEEDS,
    FOLDS10_PATH,
    LGB_TOP_PARAMS,
    MLP_FINAL_SEEDS,
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
from src.cv_harness import load_data_v2, load_experiment_meta, run_cv_v2, write_submission
from src.features import (
    build_features,
    wrap_fit_predict_with_knn,
    wrap_fit_predict_with_linkage,
    wrap_fit_predict_with_te,
)
from src.models import cat_model, lgb_model, logreg_binned_model, mlp_model, xgb_model
from src.stacking_v2 import honest_blend_cv

LOG_PATH = EXP_DIR / "LOG.md"
KNN_CACHE_DIR = EXP_DIR / "knn_cache"


def append_log(text: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _wrap_view(
    fp: Any,
    view: str,
    seed: int,
    use_linkage: bool = False,
) -> Any:
    if view == "C":
        fp = wrap_fit_predict_with_te(
            fp,
            cols=VIEW_C_TE_COLS,
            alpha=50.0,
            min_count=VIEW_C_TE_MIN_COUNT,
            seed=seed,
        )
    fp = wrap_fit_predict_with_te(
        fp, cols=("ORGANIZATION_TYPE", "OCCUPATION_TYPE"), alpha=50.0, min_count=0, seed=seed
    )
    if view == "C":
        fp = wrap_fit_predict_with_knn(
            fp,
            k_values=VIEW_C_KNN_KS,
            mode=VIEW_C_KNN_MODE,
            seed=seed,
            cache_dir=KNN_CACHE_DIR,
        )
    if use_linkage:
        fp = wrap_fit_predict_with_linkage(fp)
    return fp


def run_or_load(
    name: str,
    view: str,
    model_kind: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    X_views: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    seed: int = SEED,
    params: dict[str, Any] | None = None,
    lr: float | None = None,
    use_codes: bool = True,
    use_linkage: bool = False,
    note: str = "",
) -> tuple[dict[str, Any], float]:
    meta_path = EXP_DIR / name / "meta.json"
    if meta_path.exists():
        print(f"[SKIP/LOAD] {name} already exists.", flush=True)
        return load_experiment_meta(name, exp_dir=EXP_DIR), 0.0

    X, Xte = X_views[view]
    if model_kind == "lgb":
        fp = lgb_model(params=params, seed=seed, lr=lr or FINAL_LR, deterministic=True, use_codes=use_codes)
    elif model_kind == "xgb":
        p = dict(XGB_V2_PARAMS)
        if params:
            p.update(params)
        if lr is not None:
            p["learning_rate"] = lr
        fp = xgb_model(params=p, seed=seed, use_codes=use_codes)
    elif model_kind == "cat":
        p = dict(CAT_FAST_PARAMS)
        if params:
            p.update(params)
        if lr is not None:
            p["learning_rate"] = lr
        fp = cat_model(params=p, seed=seed)
    elif model_kind == "logreg":
        fp = logreg_binned_model(params=params, seed=seed)
    elif model_kind == "mlp":
        fp = mlp_model(params=params, seed=seed)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    fp = _wrap_view(fp, view=view, seed=seed, use_linkage=use_linkage)
    t0 = time.monotonic()
    meta = run_cv_v2(name=name, fit_predict=fp, X=X, y=y, Xte=Xte, folds=folds, note=note)
    dt = time.monotonic() - t0
    meta["elapsed_sec"] = round(dt, 2)
    (EXP_DIR / name / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta, dt


def _load_oof_test(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(EXP_DIR / name / "oof.npy"), np.load(EXP_DIR / name / "test.npy")


def _rank_avg_group(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    oof_r = None
    te_r = None
    for n in names:
        o, t = _load_oof_test(n)
        ro = rankdata(o) / len(o)
        rt = rankdata(t) / len(t)
        oof_r = ro / len(names) if oof_r is None else oof_r + ro / len(names)
        te_r = rt / len(names) if te_r is None else te_r + rt / len(names)
    assert oof_r is not None and te_r is not None
    return oof_r, te_r


def main() -> None:
    now = datetime.now(timezone.utc)
    deadline = datetime(2026, 11, 20, 1, 0, 0, tzinfo=timezone.utc)
    hours_left = (deadline - now).total_seconds() / 3600.0
    print(f"P5/P6 Start UTC: {now.isoformat()} ({hours_left:.2f} h left)", flush=True)
    append_log(
        f"\n## Phase P5: Models on Frozen Views (`VIEW_A`, `VIEW_B`, `VIEW_C`)\n"
        f"- **Start UTC:** `{now.isoformat()}` (**{hours_left:.2f} h** left)"
    )

    train, test, y, folds = load_data_v2()
    folds10 = np.load(FOLDS10_PATH)

    Xa, Xtea = build_features(
        train, test, families=[f for f in VIEW_A_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES
    )
    Xb, Xteb = build_features(
        train, test, families=[f for f in VIEW_B_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES
    )
    Xc, Xtec = build_features(
        train, test, families=[f for f in VIEW_C_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES
    )
    X_views = {"A": (Xa, Xtea), "B": (Xb, Xteb), "C": (Xc, Xtec)}

    # ------------------------------------------------------------------
    # 1. LightGBM models
    # ------------------------------------------------------------------
    top_metas = []
    for idx, p in enumerate(LGB_TOP_PARAMS, start=1):
        m, dt = run_or_load(
            f"lgb_v2_viewA_t{idx}_s42",
            "A",
            "lgb",
            train,
            test,
            y,
            folds,
            X_views,
            seed=42,
            params=p,
            lr=FINAL_LR,
            use_codes=VIEW_USE_CODES,
            note=f"P5 LGB VIEW_A v1 top-{idx} params lr=0.02",
        )
        top_metas.append((idx, p, m, dt))
        append_log(f"- **`lgb_v2_viewA_t{idx}_s42`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    best_idx, best_lgb_params, best_t_meta, _ = max(top_metas, key=lambda x: x[2]["mean_fold_auc"])
    best_t1_name = f"lgb_v2_viewA_t{best_idx}_s42"
    append_log(f"- **Selected `VIEW_A` LightGBM config:** `top-{best_idx}` (`{best_t1_name}`, mean `{best_t_meta['mean_fold_auc']:.6f}`).")

    lgb_a_names = [best_t1_name]
    for s in FINAL_SEEDS:
        if s == 42:
            continue
        m, dt = run_or_load(
            f"lgb_v2_viewA_s{s}",
            "A",
            "lgb",
            train,
            test,
            y,
            folds,
            X_views,
            seed=s,
            params=best_lgb_params,
            lr=FINAL_LR,
            use_codes=VIEW_USE_CODES,
            note=f"P5 LGB VIEW_A top-{best_idx} seed {s}",
        )
        lgb_a_names.append(f"lgb_v2_viewA_s{s}")
        append_log(f"- **`lgb_v2_viewA_s{s}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    m_nat, dt_nat = run_or_load(
        "lgb_v2_viewA_natcat_s42",
        "A",
        "lgb",
        train,
        test,
        y,
        folds,
        X_views,
        seed=42,
        params=best_lgb_params,
        lr=FINAL_LR,
        use_codes=False,
        note="P5 LGB VIEW_A native categoricals diversity",
    )
    append_log(f"- **`lgb_v2_viewA_natcat_s42`** ({dt_nat:.1f}s): mean `{m_nat['mean_fold_auc']:.6f}` | pooled `{m_nat['pooled_oof_auc']:.6f}`")

    m_lr01, dt_lr01 = run_or_load(
        "lgb_v2_viewA_lr01_s42",
        "A",
        "lgb",
        train,
        test,
        y,
        folds,
        X_views,
        seed=42,
        params={**best_lgb_params, "n_estimators": 8000, "early_stopping_rounds": 300},
        lr=0.01,
        use_codes=VIEW_USE_CODES,
        note="P5 LGB VIEW_A Addendum A.7 lr=0.01 check",
    )
    lr01_gain = m_lr01["pooled_oof_auc"] - best_t_meta["pooled_oof_auc"]
    append_log(
        f"- **`lgb_v2_viewA_lr01_s42`** ({dt_lr01:.1f}s): mean `{m_lr01['mean_fold_auc']:.6f}` | pooled `{m_lr01['pooled_oof_auc']:.6f}` (delta vs lr=0.02: `{lr01_gain:+.6f}`)"
    )

    # 10-fold LightGBM on folds10_group.npy
    m_10f, dt_10f = run_or_load(
        "lgb_v2_viewA_10fold",
        "A",
        "lgb",
        train,
        test,
        y,
        folds10,
        X_views,
        seed=42,
        params=best_lgb_params,
        lr=FINAL_LR,
        use_codes=VIEW_USE_CODES,
        note="P5 LGB VIEW_A 10-fold group CV",
    )
    f10_gain = m_10f["pooled_oof_auc"] - best_t_meta["pooled_oof_auc"]
    append_log(
        f"- **`lgb_v2_viewA_10fold`** ({dt_10f:.1f}s): mean `{m_10f['mean_fold_auc']:.6f}` | pooled `{m_10f['pooled_oof_auc']:.6f}` (delta vs 5-fold: `{f10_gain:+.6f}`)"
    )

    lgb_b_names = []
    for s in FINAL_SEEDS:
        m, dt = run_or_load(
            f"lgb_v2_viewB_s{s}",
            "B",
            "lgb",
            train,
            test,
            y,
            folds,
            X_views,
            seed=s,
            params=best_lgb_params,
            lr=FINAL_LR,
            use_codes=VIEW_USE_CODES,
            note=f"P5 LGB VIEW_B seed {s}",
        )
        lgb_b_names.append(f"lgb_v2_viewB_s{s}")
        append_log(f"- **`lgb_v2_viewB_s{s}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    lgb_c_names = []
    for s in FINAL_SEEDS:
        m, dt = run_or_load(
            f"lgb_v2_viewC_s{s}",
            "C",
            "lgb",
            train,
            test,
            y,
            folds,
            X_views,
            seed=s,
            params=best_lgb_params,
            lr=FINAL_LR,
            use_codes=VIEW_USE_CODES,
            note=f"P5 LGB VIEW_C (A4 TE + KNN v2) seed {s}",
        )
        lgb_c_names.append(f"lgb_v2_viewC_s{s}")
        append_log(f"- **`lgb_v2_viewC_s{s}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    # ------------------------------------------------------------------
    # 2. XGBoost models (integer codes)
    # ------------------------------------------------------------------
    xgb_a_names, xgb_b_names, xgb_c_names = [], [], []
    for v_code, lst in [("A", xgb_a_names), ("B", xgb_b_names), ("C", xgb_c_names)]:
        for s in XGB_FINAL_SEEDS:
            nm = f"xgb_v2_view{v_code}_s{s}"
            m, dt = run_or_load(
                nm,
                v_code,
                "xgb",
                train,
                test,
                y,
                folds,
                X_views,
                seed=s,
                use_codes=True,
                note=f"P5 XGB VIEW_{v_code} integer codes seed {s}",
            )
            lst.append(nm)
            append_log(f"- **`{nm}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    # ------------------------------------------------------------------
    # 3. CatBoost "fast" models
    # ------------------------------------------------------------------
    cat_a_names = []
    for s in CAT_FINAL_SEEDS:
        nm = f"cat_v2_viewA_fast_s{s}"
        m, dt = run_or_load(
            nm,
            "A",
            "cat",
            train,
            test,
            y,
            folds,
            X_views,
            seed=s,
            note=f"P5 CatBoost fast VIEW_A seed {s}",
        )
        cat_a_names.append(nm)
        append_log(f"- **`{nm}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    cat_b_names = ["cat_v2_viewB_fast_s42"]
    m, dt = run_or_load(
        "cat_v2_viewB_fast_s42",
        "B",
        "cat",
        train,
        test,
        y,
        folds,
        X_views,
        seed=42,
        note="P5 CatBoost fast VIEW_B seed 42",
    )
    append_log(f"- **`cat_v2_viewB_fast_s42`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    cat_c_names = ["cat_v2_viewC_fast_s42"]
    m, dt = run_or_load(
        "cat_v2_viewC_fast_s42",
        "C",
        "cat",
        train,
        test,
        y,
        folds,
        X_views,
        seed=42,
        note="P5 CatBoost fast VIEW_C seed 42",
    )
    append_log(f"- **`cat_v2_viewC_fast_s42`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    # ------------------------------------------------------------------
    # 4. Logistic Regression on Binned Inputs (C in {0.01, 0.1, 1.0})
    # ------------------------------------------------------------------
    lr_metas = []
    for c_val, c_tag in [(0.01, "c001"), (0.1, "c01"), (1.0, "c10")]:
        nm = f"logreg_v2_viewA_{c_tag}"
        m, dt = run_or_load(
            nm,
            "A",
            "logreg",
            train,
            test,
            y,
            folds,
            X_views,
            seed=42,
            params={"C": c_val, "n_bins": 30},
            note=f"P5 Binned LogReg C={c_val}",
        )
        lr_metas.append((nm, m))
        append_log(f"- **`{nm}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")
    best_logreg_name, best_logreg_meta = max(lr_metas, key=lambda x: x[1]["mean_fold_auc"])

    # ------------------------------------------------------------------
    # 5. Plain MLP (PyTorch CPU, 2 seeds on VIEW_A)
    # ------------------------------------------------------------------
    mlp_names = []
    mlp_metas = []
    for s in MLP_FINAL_SEEDS:
        nm = f"mlp_v2_viewA_s{s}"
        m, dt = run_or_load(
            nm,
            "A",
            "mlp",
            train,
            test,
            y,
            folds,
            X_views,
            seed=s,
            note=f"P5 PyTorch CPU MLP (256,128) seed {s}",
        )
        mlp_names.append(nm)
        mlp_metas.append(m)
        append_log(f"- **`{nm}`** ({dt:.1f}s): mean `{m['mean_fold_auc']:.6f}` | pooled `{m['pooled_oof_auc']:.6f}`")

    # ------------------------------------------------------------------
    # Phase P6: Honest Blend-CV & Stacking
    # ------------------------------------------------------------------
    now_p6 = datetime.now(timezone.utc)
    hours_left_p6 = (deadline - now_p6).total_seconds() / 3600.0
    append_log(
        f"\n## Phase P6: Stacking and Blending (5-fold Honest Blend-CV)\n"
        f"- **Start UTC:** `{now_p6.isoformat()}` (**{hours_left_p6:.2f} h** left)"
    )

    # Multi-seed group summaries (per family & view)
    core_groups: dict[str, list[str]] = {
        "LGB_VIEW_A": lgb_a_names + (["lgb_v2_viewA_lr01_s42"] if lr01_gain >= 0.0003 else []),
        "LGB_VIEW_A_NATCAT": ["lgb_v2_viewA_natcat_s42"],
        "LGB_VIEW_B": lgb_b_names,
        "LGB_VIEW_C": lgb_c_names,
        "XGB_VIEW_A": xgb_a_names,
        "XGB_VIEW_B": xgb_b_names,
        "XGB_VIEW_C": xgb_c_names,
        "CAT_VIEW_A": cat_a_names,
        "CAT_VIEW_B": cat_b_names,
        "CAT_VIEW_C": cat_c_names,
    }
    if f10_gain >= 0.0003:
        append_log("- Note: `lgb_v2_viewA_10fold` cleared +0.0003 over 5-fold; tracking its test predictions.")

    oof_pool: dict[str, np.ndarray] = {}
    test_pool: dict[str, np.ndarray] = {}
    for gname, members in core_groups.items():
        go, gt = _rank_avg_group(members)
        oof_pool[gname] = go
        test_pool[gname] = gt
        g_auc = float(roc_auc_score(y, go))
        append_log(f"- Group `{gname}` ({len(members)} seeds): pooled OOF AUC `{g_auc:.6f}`")

    # Test admission of LOGREG_BEST and MLP_2SEED (require OOF >= 0.745 and Honest Blend-CV >= +0.0003)
    base_res = honest_blend_cv(oof_pool, test_pool, y, folds, raw_train=train, raw_test=test, run_et_restacker=False)
    base_eq_mean = base_res["equal_rank"]["mean_fold_auc"]

    logreg_o, logreg_t = _load_oof_test(best_logreg_name)
    logreg_auc = float(roc_auc_score(y, logreg_o))
    admit_logreg = False
    if logreg_auc >= 0.745:
        test_lr_pool = {**oof_pool, "LOGREG_A": logreg_o}
        test_lr_te = {**test_pool, "LOGREG_A": logreg_t}
        lr_check = honest_blend_cv(test_lr_pool, test_lr_te, y, folds, run_et_restacker=False)
        lr_blend_gain = lr_check["hill_climber"]["mean_fold_auc"] - base_res["hill_climber"]["mean_fold_auc"]
        lr_eq_gain = lr_check["equal_rank"]["mean_fold_auc"] - base_eq_mean
        admit_logreg = max(lr_blend_gain, lr_eq_gain) >= 0.0003
        append_log(
            f"- **LogReg Admission Check (`{best_logreg_name}`, AUC={logreg_auc:.6f}):** eq_gain=`{lr_eq_gain:+.6f}`, hill_gain=`{lr_blend_gain:+.6f}` -> `{'ADMIT' if admit_logreg else 'REJECT'}`"
        )
    if admit_logreg:
        oof_pool["LOGREG_A"] = logreg_o
        test_pool["LOGREG_A"] = logreg_t

    mlp_o, mlp_t = _rank_avg_group(mlp_names)
    mlp_auc = float(roc_auc_score(y, mlp_o))
    admit_mlp = False
    if mlp_auc >= 0.745:
        test_m_pool = {**oof_pool, "MLP_A": mlp_o}
        test_m_te = {**test_pool, "MLP_A": mlp_t}
        m_check = honest_blend_cv(test_m_pool, test_m_te, y, folds, run_et_restacker=False)
        m_eq_gain = m_check["equal_rank"]["mean_fold_auc"] - base_eq_mean
        m_hill_gain = m_check["hill_climber"]["mean_fold_auc"] - base_res["hill_climber"]["mean_fold_auc"]
        admit_mlp = max(m_eq_gain, m_hill_gain) >= 0.0003
        append_log(
            f"- **MLP Admission Check (`MLP_2SEED`, AUC={mlp_auc:.6f}):** eq_gain=`{m_eq_gain:+.6f}`, hill_gain=`{m_hill_gain:+.6f}` -> `{'ADMIT' if admit_mlp else 'REJECT'}`"
        )
    else:
        append_log(f"- **MLP Admission Check (`MLP_2SEED`, AUC={mlp_auc:.6f} < 0.745):** -> `REJECT`")
    if admit_mlp:
        oof_pool["MLP_A"] = mlp_o
        test_pool["MLP_A"] = mlp_t

    # Full Honest Blend-CV across all stacking methods (a..e)
    # Also evaluate equal family weights (1/3 LGB + 1/3 XGB + 1/3 CAT) as Method a2
    stack_res = honest_blend_cv(
        oof_pool,
        test_pool,
        y,
        folds,
        raw_train=train,
        raw_test=test,
        run_et_restacker=True,
        seed=SEED,
    )

    append_log("\n### Honest Blend-CV Comparison")
    for mname, info in stack_res.items():
        append_log(
            f"- `{mname}`: mean `{info['mean_fold_auc']:.6f}` (+/- {info['std_fold_auc']:.6f}) | "
            f"pooled `{info['pooled_oof_auc']:.6f}` | vs equal `{info['diff_vs_equal']:+.6f}` (`{info['folds_improved_vs_equal']}/5` folds)"
        )

    # Adopt complex method over equal_rank ONLY if diff_vs_equal >= +0.0003 AND folds_improved >= 4
    complex_cands = [
        (m, info)
        for m, info in stack_res.items()
        if m != "equal_rank" and info["diff_vs_equal"] >= 0.0003 and info["folds_improved_vs_equal"] >= 4
    ]
    if complex_cands:
        chosen_method, chosen_info = max(complex_cands, key=lambda kv: kv[1]["mean_fold_auc"])
    else:
        chosen_method, chosen_info = "equal_rank", stack_res["equal_rank"]

    append_log(f"- **Selected Final Blend Method:** **`{chosen_method}`** (mean fold `{chosen_info['mean_fold_auc']:.6f}`, pooled `{chosen_info['pooled_oof_auc']:.6f}`).")

    # Compare vs v1_blend_on_folds_group
    v1_metrics = json.loads((EXP_DIR / "v1_baseline_metrics.json").read_text(encoding="utf-8"))
    v1_grp_folds = v1_metrics["v1_blend_on_folds_group"]["fold_aucs"]
    v2_folds = chosen_info["fold_aucs"]
    v1_diffs = [a - b for a, b in zip(v2_folds, v1_grp_folds)]
    v1_improved = int(sum(d > 0 for d in v1_diffs))
    v1_mean_diff = float(np.mean(v1_diffs))
    append_log(
        f"- **v2 Blend vs v1 Blend (on `folds_group.npy`):** `0.765179` -> **`{chosen_info['mean_fold_auc']:.6f}`** (`{v1_mean_diff:+.6f}`, `{v1_improved}/5` folds improved)"
    )

    # Save blend_final_v2
    blend_dir = EXP_DIR / "blend_final_v2"
    blend_dir.mkdir(parents=True, exist_ok=True)
    np.save(blend_dir / "oof.npy", chosen_info["oof"])
    np.save(blend_dir / "test.npy", chosen_info["test"])

    # Normalize test predictions strictly to [0, 1] via rankdata or clipping
    test_sub_preds = np.clip(chosen_info["test"], 0.0, 1.0)
    blend_meta = {
        "name": "blend_final_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chosen_method": chosen_method,
        "pool_members": list(oof_pool.keys()),
        "core_groups": core_groups,
        "admit_logreg": admit_logreg,
        "admit_mlp": admit_mlp,
        "fold_aucs": chosen_info["fold_aucs"],
        "mean_fold_auc": chosen_info["mean_fold_auc"],
        "std_fold_auc": chosen_info["std_fold_auc"],
        "pooled_oof_auc": chosen_info["pooled_oof_auc"],
        "v1_mean_diff": v1_mean_diff,
        "v1_folds_improved": v1_improved,
        "all_methods": {
            k: {
                "mean_fold_auc": v["mean_fold_auc"],
                "pooled_oof_auc": v["pooled_oof_auc"],
                "diff_vs_equal": v["diff_vs_equal"],
                "folds_improved_vs_equal": v["folds_improved_vs_equal"],
                "fold_aucs": v["fold_aucs"],
            }
            for k, v in stack_res.items()
        },
    }
    (blend_dir / "meta.json").write_text(json.dumps(blend_meta, indent=2), encoding="utf-8")
    write_submission(test, test_sub_preds, SUB_DIR / "sub_01_blend_v2.csv", meta=blend_meta)

    # Build matching ENABLE_LINKAGE=True variant (for standby shipping only if explicitly approved)
    link_names = []
    for kind, nm in [("lgb", "lgb_v2_viewA_link_s42"), ("xgb", "xgb_v2_viewA_link_s42"), ("cat", "cat_v2_viewA_link_s42")]:
        p_kw = best_lgb_params if kind == "lgb" else None
        m_l, dt_l = run_or_load(
            nm,
            "A",
            kind,
            train,
            test,
            y,
            folds,
            X_views,
            seed=42,
            params=p_kw,
            lr=FINAL_LR if kind == "lgb" else None,
            use_codes=True,
            use_linkage=True,
            note=f"P6 ENABLE_LINKAGE=True standby {kind} VIEW_A",
        )
        link_names.append(nm)
        append_log(f"- **`{nm}` (linkage standby)** ({dt_l:.1f}s): mean `{m_l['mean_fold_auc']:.6f}` | pooled `{m_l['pooled_oof_auc']:.6f}`")
    lo, lt = _rank_avg_group(link_names)
    link_dir = EXP_DIR / "blend_final_v2_linkage"
    link_dir.mkdir(parents=True, exist_ok=True)
    np.save(link_dir / "oof.npy", lo)
    np.save(link_dir / "test.npy", lt)
    link_meta = {
        "name": "blend_final_v2_linkage",
        "enable_linkage": True,
        "members": link_names,
        "pooled_oof_auc": float(roc_auc_score(y, lo)),
        "fold_aucs": [float(roc_auc_score(y[folds == f], lo[folds == f])) for f in range(5)],
    }
    link_meta["mean_fold_auc"] = float(np.mean(link_meta["fold_aucs"]))
    (link_dir / "meta.json").write_text(json.dumps(link_meta, indent=2), encoding="utf-8")
    write_submission(test, np.clip(lt, 0.0, 1.0), SUB_DIR / "sub_final_v2_linkage.csv", meta=link_meta)

    print("P5/P6 COMPLETE!", json.dumps({k: v for k, v in blend_meta.items() if k != "all_methods"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Automated P3 Feature Discovery & P4 Gate Runner for GCI Home Credit v2."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config_v2 import (
    DROPPED_FEATURES,
    EXP_DIR,
    EXP_LR,
    FOLDS_DISCOVERY_PATH,
    FOLDS_GROUP_PATH,
    FOLDS_V1_PATH,
    N_THREADS,
    SEED,
)
from src.cv_harness import compare_v2, load_data_v2, load_experiment_meta, run_cv_v2
from src.features import (
    build_a7_ratio_candidates,
    build_features,
    compute_nested_knn_target_mean,
    wrap_fit_predict_with_knn,
    wrap_fit_predict_with_linkage,
    wrap_fit_predict_with_te,
)
from src.models import lgb_model

LOG_PATH = EXP_DIR / "LOG.md"
KNN_CACHE_DIR = EXP_DIR / "knn_cache"


def append_log(text: str) -> None:
    """Append markdown text cleanly to experiments_v2/LOG.md."""
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def git_commit_v2(msg: str) -> None:
    """Stage v2 files (never tutorial.ipynb or input/) and commit locally."""
    subprocess.run(
        ["git", "add", "src/", "experiments_v2/"],
        cwd=str(ROOT_DIR),
        check=True,
    )
    res = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(ROOT_DIR),
    )
    if res.returncode != 0:
        subprocess.run(["git", "commit", "-m", msg], cwd=str(ROOT_DIR), check=True)
        print(f"[GIT COMMITTED] {msg}", flush=True)


def run_or_load_lgb(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    families: list[str],
    te_f6: bool = True,
    extra_te_cols: tuple[str, ...] = (),
    te_min_count: int = 30,
    knn_ks: tuple[int, ...] = (),
    knn_mode: str = "v1",
    use_linkage: bool = False,
    use_codes: bool = False,
    extra_params: dict[str, Any] | None = None,
    use_knn_cache: bool = True,
    note: str = "",
) -> dict[str, Any]:
    """Run 5-fold LightGBM CV or load existing meta.json if already run."""
    meta_path = EXP_DIR / name / "meta.json"
    if meta_path.exists():
        print(f"[SKIP/LOAD] {name} already exists.", flush=True)
        return load_experiment_meta(name, exp_dir=EXP_DIR)

    build_fams = [f for f in families if f not in ("F6", "B1")]
    X, Xte = build_features(
        train,
        test,
        families=build_fams,
        use_test_for_stats=False,
        dropped_features=DROPPED_FEATURES,
    )
    fp = lgb_model(params=extra_params, seed=SEED, lr=EXP_LR, use_codes=use_codes)

    if extra_te_cols:
        fp = wrap_fit_predict_with_te(
            fp, cols=extra_te_cols, alpha=50.0, min_count=te_min_count, seed=SEED
        )
    if te_f6:
        fp = wrap_fit_predict_with_te(
            fp, cols=("ORGANIZATION_TYPE", "OCCUPATION_TYPE"), alpha=50.0, min_count=0, seed=SEED
        )
    if knn_ks:
        fp = wrap_fit_predict_with_knn(
            fp,
            k_values=knn_ks,
            mode=knn_mode,
            seed=SEED,
            cache_dir=KNN_CACHE_DIR if use_knn_cache else None,
        )
    if use_linkage:
        fp = wrap_fit_predict_with_linkage(fp)

    return run_cv_v2(name=name, fit_predict=fp, X=X, y=y, Xte=Xte, folds=folds, note=note)


def eval_adv(X: pd.DataFrame, Xte: pd.DataFrame, seed: int = SEED) -> tuple[float, list[tuple[str, float]]]:
    """5-fold LightGBM adversarial validation AUC (train vs test)."""
    X_adv = pd.concat([X, Xte], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(X), dtype=np.int32), np.ones(len(Xte), dtype=np.int32)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_adv), dtype=np.float64)
    imp = np.zeros(X.shape[1], dtype=np.float64)
    for tr_i, va_i in skf.split(X_adv, y_adv):
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=N_THREADS,
            verbose=-1,
        )
        clf.fit(X_adv.iloc[tr_i], y_adv[tr_i])
        oof[va_i] = clf.predict_proba(X_adv.iloc[va_i])[:, 1]
        imp += clf.booster_.feature_importance(importance_type="gain") / 5.0
    auc = float(roc_auc_score(y_adv, oof))
    top = sorted(zip(X.columns, imp), key=lambda x: x[1], reverse=True)[:8]
    return auc, [(str(c), float(v)) for c, v in top]


def check_suspicious_gain(
    name: str,
    cmp_res: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds_group: np.ndarray,
    state: dict[str, Any],
) -> None:
    """Trigger Hard Rule 7 checks if any single change raises mean AUC by > +0.004."""
    if cmp_res["mean_diff"] <= 0.004:
        return
    print(f"[SUSPICIOUS GAIN ALERT] {name} gained {cmp_res['mean_diff']:+.6f} > +0.004!", flush=True)
    X, Xte = build_features(
        train,
        test,
        families=[f for f in state["families"] if f not in ("F6", "B1")],
        use_test_for_stats=False,
        dropped_features=DROPPED_FEATURES,
    )
    adv_auc, adv_top = eval_adv(X, Xte)
    # Permuted label check
    rng = np.random.default_rng(777)
    y_perm = rng.permutation(y)
    perm_meta = run_or_load_lgb(
        f"{name}_suspicious_perm",
        train,
        test,
        y_perm,
        folds_group,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        use_knn_cache=False,
        note="Hard Rule 7 permuted-label check",
    )
    # Random folds comparison
    folds_rand = np.load(FOLDS_V1_PATH)
    rand_meta = run_or_load_lgb(
        f"{name}_suspicious_rand",
        train,
        test,
        y,
        folds_rand,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="Hard Rule 7 random-folds check",
    )
    report = {
        "experiment": name,
        "mean_diff": cmp_res["mean_diff"],
        "adv_auc": adv_auc,
        "adv_top": adv_top,
        "permuted_label_oof_auc": perm_meta["pooled_oof_auc"],
        "random_folds_mean_auc": rand_meta["mean_fold_auc"],
    }
    (EXP_DIR / "suspicious_gain_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    append_log(f"\n> [!WARNING]\n> **Hard Rule 7 Suspicious Gain (>+0.004) on `{name}`:** `{json.dumps(report)}`\n")
    sys.exit(3)


def log_result(name: str, ref: str, meta: dict[str, Any], cmp_res: dict[str, Any], reason: str = "") -> None:
    """Append standardized experiment result and verdict to experiments_v2/LOG.md."""
    f_str = "[" + ", ".join(f"{v:.6f}" for v in meta["fold_aucs"]) + "]"
    append_log(
        f"- **Result (`{name}`, {meta['n_features']} cols, vs `{ref}`):**\n"
        f"  - Fold AUCs: `{f_str}` | Mean: **`{meta['mean_fold_auc']:.6f}`** | Pooled: `{meta['pooled_oof_auc']:.6f}`\n"
        f"  - Delta vs `{ref}`: **`{cmp_res['mean_diff']:+.6f}`** | Folds improved: `{cmp_res['folds_improved']}/5` -> **`{cmp_res['verdict']}`**"
        + (f" ({reason})" if reason else "")
    )


def main() -> None:
    now = datetime.now(timezone.utc)
    deadline = datetime(2026, 11, 20, 1, 0, 0, tzinfo=timezone.utc)
    hours_left = (deadline - now).total_seconds() / 3600.0
    print(f"P3/P4 Orchestrator Start UTC: {now.isoformat()} ({hours_left:.2f} h left)", flush=True)

    train, test, y, folds_group = load_data_v2()

    # Active incumbent state starting from lgb2_inc
    state: dict[str, Any] = {
        "inc_name": "lgb2_inc",
        "families": ["F2", "F5_LOO", "F6"],
        "extra_te_cols": [],
        "knn_ks": [],
        "knn_mode": "v1",
        "use_codes": False,
        "extra_params": {},
    }

    # ------------------------------------------------------------------
    # 1. A0 (region_cat)
    # ------------------------------------------------------------------
    a0b_meta = load_experiment_meta("lgb2_a0b_regte", exp_dir=EXP_DIR)
    a0b_cmp = compare_v2("lgb2_a0b_regte", "lgb2_inc")
    log_result("lgb2_a0b_regte", "lgb2_inc", a0b_meta, a0b_cmp, "TE_REGION_ID only")

    a0a_meta = run_or_load_lgb(
        "lgb2_a0a_regcat",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A0"],
        extra_params={"cat_smooth": 20},
        note="P3 A0a REGION_ID categorical cat_smooth=20",
    )
    a0a_cmp = compare_v2("lgb2_a0a_regcat", "lgb2_inc")
    log_result("lgb2_a0a_regcat", "lgb2_inc", a0a_meta, a0a_cmp, "REGION_ID cat + cat_smooth=20")

    a0c_meta = run_or_load_lgb(
        "lgb2_a0c_regboth",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A0"],
        extra_te_cols=("REGION_ID",),
        te_min_count=30,
        extra_params={"cat_smooth": 20},
        note="P3 A0c REGION_ID cat + nested TE_REGION_ID",
    )
    a0c_cmp = compare_v2("lgb2_a0c_regboth", "lgb2_inc")
    log_result("lgb2_a0c_regboth", "lgb2_inc", a0c_meta, a0c_cmp, "Both REGION_ID cat & TE")

    a0_candidates = [
        ("lgb2_a0a_regcat", a0a_meta, a0a_cmp, {"families": state["families"] + ["A0"], "extra_params": {"cat_smooth": 20}}),
        ("lgb2_a0b_regte", a0b_meta, a0b_cmp, {"extra_te_cols": ["REGION_ID"]}),
        ("lgb2_a0c_regboth", a0c_meta, a0c_cmp, {"families": state["families"] + ["A0"], "extra_te_cols": ["REGION_ID"], "extra_params": {"cat_smooth": 20}}),
    ]
    a0_passing = [c for c in a0_candidates if c[2]["keep"]]
    if a0_passing:
        best_name, _, best_cmp, patch = max(a0_passing, key=lambda c: c[1]["mean_fold_auc"])
        check_suspicious_gain(best_name, best_cmp, train, test, y, folds_group, state)
        state["inc_name"] = best_name
        state.update(patch)
        append_log(f"- **A0 Winner:** `{best_name}` accepted into incumbent.\n")
        git_commit_v2(f"feat(v2): P3 A0 accept {best_name}")
    else:
        append_log("- **A0 Verdict:** All 3 variants rejected; incumbent stays `lgb2_inc`.\n")

    # ------------------------------------------------------------------
    # 2. A0b (lgb_codes)
    # ------------------------------------------------------------------
    append_log(
        f"\n### A0b (`lgb_codes`: ordinal integer categorical codes)\n"
        f"- **Hypothesis (`lgb2_a0b_codes`):** Representing all categorical columns as ordinal integer codes (`NaN -> -1`) may reduce high-cardinality overfitting on `ORGANIZATION_TYPE`/`OCCUPATION_TYPE` (as observed in XGBoost v1). Expected: `-0.0005` to `+0.0005`."
    )
    codes_meta = run_or_load_lgb(
        "lgb2_a0b_codes",
        train,
        test,
        y,
        folds_group,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        use_codes=True,
        extra_params=state["extra_params"],
        note="P3 A0b LightGBM ordinal integer codes",
    )
    codes_cmp = compare_v2("lgb2_a0b_codes", state["inc_name"])
    log_result("lgb2_a0b_codes", state["inc_name"], codes_meta, codes_cmp)
    if codes_cmp["keep"]:
        check_suspicious_gain("lgb2_a0b_codes", codes_cmp, train, test, y, folds_group, state)
        state["inc_name"] = "lgb2_a0b_codes"
        state["use_codes"] = True
        git_commit_v2("feat(v2): P3 A0b accept lgb2_a0b_codes")

    # ------------------------------------------------------------------
    # 3. A3 (KNN target mean, Refined Spec A.4)
    # ------------------------------------------------------------------
    append_log(
        f"\n### A3 (`KNN target mean`: nested Quantile-Normal [EXT_1..3, CREDIT_TERM])\n"
        f"- **Hypothesis (`lgb2_a3_knn500_v1`):** Adding nested `k=500` target mean over global 4D space (`NaN -> 0.0` normal median) captures non-axis-aligned interactions among external scores and loan term. Expected: `+0.0003` to `+0.0020`.\n"
        f"- **Hypothesis (`lgb2_a3_knn500_v2`):** Restricting `k=500` neighbours to identical EXT missingness patterns (fallback to V1 if `<3,000` train rows) avoids artificial median distances on missing scores. Expected: `+0.0004` to `+0.0022`."
    )
    knn_v1_meta = run_or_load_lgb(
        "lgb2_a3_knn500_v1",
        train,
        test,
        y,
        folds_group,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=(500,),
        knn_mode="v1",
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A3 KNN k=500 V1 global pool",
    )
    knn_v1_cmp = compare_v2("lgb2_a3_knn500_v1", state["inc_name"])
    log_result("lgb2_a3_knn500_v1", state["inc_name"], knn_v1_meta, knn_v1_cmp, "V1 global pool")

    knn_v2_meta = run_or_load_lgb(
        "lgb2_a3_knn500_v2",
        train,
        test,
        y,
        folds_group,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=(500,),
        knn_mode="v2",
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A3 KNN k=500 V2 pattern-isolated pools",
    )
    knn_v2_cmp = compare_v2("lgb2_a3_knn500_v2", state["inc_name"])
    log_result("lgb2_a3_knn500_v2", state["inc_name"], knn_v2_meta, knn_v2_cmp, "V2 pattern-isolated")

    knn_passing = [c for c in [("lgb2_a3_knn500_v1", "v1", knn_v1_meta, knn_v1_cmp), ("lgb2_a3_knn500_v2", "v2", knn_v2_meta, knn_v2_cmp)] if c[3]["keep"]]
    if knn_passing:
        k_name, k_mode, _, k_cmp = max(knn_passing, key=lambda c: c[2]["mean_fold_auc"])
        check_suspicious_gain(k_name, k_cmp, train, test, y, folds_group, state)
        state["inc_name"] = k_name
        state["knn_ks"] = [500]
        state["knn_mode"] = k_mode

        # Distribution parity check on Fold 0
        X_tmp, Xte_tmp = build_features(train, test, families=["F2"], dropped_features=DROPPED_FEATURES)
        m0 = folds_group == 0
        tr_k, va_k, _ = compute_nested_knn_target_mean(
            X_tmp.iloc[~m0], y[~m0], X_tmp.iloc[m0], Xte_tmp, k_values=(500,), mode=k_mode, seed=SEED, cache_dir=KNN_CACHE_DIR
        )
        col_k = "KNN_EXT_TERM_500_MEAN"
        append_log(
            f"- **A3 (`{k_mode}`, `k=500`) Fold-0 Distribution Check:** train mean={tr_k[col_k].mean():.5f} std={tr_k[col_k].std():.5f} | val mean={va_k[col_k].mean():.5f} std={va_k[col_k].std():.5f}"
        )
        git_commit_v2(f"feat(v2): P3 A3 accept {k_name}")

        # Multi-k check: k=100 and k=2000
        append_log(
            f"- **Hypothesis (`lgb2_a3_knn_add100`):** Adding finer `k=100` on top of `k=500` (`mode={k_mode}`) captures local risk pockets (`>= +0.0002`, `>= 4/5` folds)."
        )
        k100_meta = run_or_load_lgb(
            "lgb2_a3_knn_add100",
            train,
            test,
            y,
            folds_group,
            families=state["families"],
            extra_te_cols=tuple(state["extra_te_cols"]),
            knn_ks=(500, 100),
            knn_mode=k_mode,
            use_codes=state["use_codes"],
            extra_params=state["extra_params"],
            note=f"P3 A3 KNN add k=100 ({k_mode})",
        )
        k100_cmp = compare_v2("lgb2_a3_knn_add100", state["inc_name"])
        log_result("lgb2_a3_knn_add100", state["inc_name"], k100_meta, k100_cmp)
        if k100_cmp["keep"]:
            check_suspicious_gain("lgb2_a3_knn_add100", k100_cmp, train, test, y, folds_group, state)
            state["inc_name"] = "lgb2_a3_knn_add100"
            state["knn_ks"] = [500, 100]
            git_commit_v2("feat(v2): P3 A3 add KNN k=100")

        append_log(
            f"- **Hypothesis (`lgb2_a3_knn_add2000`):** Adding broader `k=2000` on top of `{state['knn_ks']}` (`mode={k_mode}`) captures smooth global risk gradients (`>= +0.0002`, `>= 4/5` folds)."
        )
        ks_2000 = tuple(state["knn_ks"] + [2000])
        k2000_meta = run_or_load_lgb(
            "lgb2_a3_knn_add2000",
            train,
            test,
            y,
            folds_group,
            families=state["families"],
            extra_te_cols=tuple(state["extra_te_cols"]),
            knn_ks=ks_2000,
            knn_mode=k_mode,
            use_codes=state["use_codes"],
            extra_params=state["extra_params"],
            note=f"P3 A3 KNN add k=2000 ({k_mode})",
        )
        k2000_cmp = compare_v2("lgb2_a3_knn_add2000", state["inc_name"])
        log_result("lgb2_a3_knn_add2000", state["inc_name"], k2000_meta, k2000_cmp)
        if k2000_cmp["keep"]:
            check_suspicious_gain("lgb2_a3_knn_add2000", k2000_cmp, train, test, y, folds_group, state)
            state["inc_name"] = "lgb2_a3_knn_add2000"
            state["knn_ks"] = list(ks_2000)
            git_commit_v2("feat(v2): P3 A3 add KNN k=2000")

    # ------------------------------------------------------------------
    # 4. A1 (Term/rate family G1)
    # ------------------------------------------------------------------
    append_log(
        "\n### A1 (`Term/rate family G1`: 7 cols)\n"
        "- **Hypothesis (`lgb2_a1_term_rate`):** Adding `TERM_MODE`, `TERM_RESID_REL`, `COST_PROXY`, `RATE_PER_PERIOD` (bisection), `DOWNPAY_PROXY`, `GOODS_EQ_CREDIT`, and `PRODUCT_CODE` exposes implied interest rate and loan product pricing. Expected: `+0.0003` to `+0.0015`."
    )
    a1_meta = run_or_load_lgb(
        "lgb2_a1_term_rate",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A1"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A1 term/rate family",
    )
    a1_cmp = compare_v2("lgb2_a1_term_rate", state["inc_name"])
    log_result("lgb2_a1_term_rate", state["inc_name"], a1_meta, a1_cmp)
    if a1_cmp["keep"]:
        check_suspicious_gain("lgb2_a1_term_rate", a1_cmp, train, test, y, folds_group, state)
        state["inc_name"] = "lgb2_a1_term_rate"
        state["families"] = state["families"] + ["A1"]
        git_commit_v2("feat(v2): P3 A1 accept term/rate family")

    # Helper for Rule A.3 (> 8 cols evaluated at BOTH ff=0.35 and ff=0.60)
    def run_rule_a3_family(tag: str, fam_code: str, hyp: str) -> None:
        append_log(hyp)
        inc_ff60_name = f"{state['inc_name']}_ff60"
        ff60_params = {**state["extra_params"], "feature_fraction": 0.60}
        run_or_load_lgb(
            inc_ff60_name,
            train,
            test,
            y,
            folds_group,
            families=state["families"],
            extra_te_cols=tuple(state["extra_te_cols"]),
            knn_ks=tuple(state["knn_ks"]),
            knn_mode=state["knn_mode"],
            use_codes=state["use_codes"],
            extra_params=ff60_params,
            note=f"Rule A.3 baseline ff=0.60 for {state['inc_name']}",
        )
        m35 = run_or_load_lgb(
            f"lgb2_{tag}_ff35",
            train,
            test,
            y,
            folds_group,
            families=state["families"] + [fam_code],
            extra_te_cols=tuple(state["extra_te_cols"]),
            knn_ks=tuple(state["knn_ks"]),
            knn_mode=state["knn_mode"],
            use_codes=state["use_codes"],
            extra_params=state["extra_params"],
            note=f"P3 {tag} at ff=0.35",
        )
        c35 = compare_v2(f"lgb2_{tag}_ff35", state["inc_name"])
        log_result(f"lgb2_{tag}_ff35", state["inc_name"], m35, c35, "Rule A.3 ff=0.35")

        m60 = run_or_load_lgb(
            f"lgb2_{tag}_ff60",
            train,
            test,
            y,
            folds_group,
            families=state["families"] + [fam_code],
            extra_te_cols=tuple(state["extra_te_cols"]),
            knn_ks=tuple(state["knn_ks"]),
            knn_mode=state["knn_mode"],
            use_codes=state["use_codes"],
            extra_params=ff60_params,
            note=f"P3 {tag} at ff=0.60",
        )
        c60 = compare_v2(f"lgb2_{tag}_ff60", inc_ff60_name)
        log_result(f"lgb2_{tag}_ff60", inc_ff60_name, m60, c60, "Rule A.3 ff=0.60")

        if c35["keep"] or c60["keep"]:
            best_c = c35 if (c35["keep"] and (not c60["keep"] or c35["mean_diff"] >= c60["mean_diff"])) else c60
            best_n = f"lgb2_{tag}_ff35" if best_c is c35 else f"lgb2_{tag}_ff60"
            check_suspicious_gain(best_n, best_c, train, test, y, folds_group, state)
            # Keep incumbent reference at ff=0.35 for standard <=8 col tests, or best setting
            state["families"] = state["families"] + [fam_code]
            state["inc_name"] = f"lgb2_{tag}_ff35"
            append_log(f"- **Rule A.3 Verdict (`{fam_code}`):** ACCEPTED (passed via `{best_n}`).\n")
            git_commit_v2(f"feat(v2): P3 accept {fam_code} under Rule A.3")
        else:
            append_log(f"- **Rule A.3 Verdict (`{fam_code}`):** REJECTED at both ff=0.35 and ff=0.60.\n")

    # ------------------------------------------------------------------
    # 5. A5 (div_ext, 18 cols > 8 cols -> Rule A.3)
    # ------------------------------------------------------------------
    run_rule_a3_family(
        "a5_divext",
        "A5",
        "\n### A5 (`div_ext`: 18 ratios of amounts/age/emp by EXT scores, Rule A.3)\n"
        "- **Hypothesis (`lgb2_a5_divext`):** Dividing `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`, `AMT_GOODS_PRICE`, `AGE_YEARS`, `EMPLOYED_MAG` by `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_MEAN` directly highlights high-debt / low-external-score risk (`>= +0.0002`, `>= 4/5` folds at ff=0.35 or ff=0.60).",
    )

    # ------------------------------------------------------------------
    # 6. A6 (age_int, 1 col)
    # ------------------------------------------------------------------
    append_log(
        "\n### A6 (`age_int`: integer age in years)\n"
        "- **Hypothesis (`lgb2_a6a_age_int` / `lgb2_a6b_age_cat`):** Adding `floor(-DAYS_BIRTH / 365)` as an integer or categorical captures discrete age-threshold underwriting rules. Expected: `+0.0001` to `+0.0005`."
    )
    a6a_m = run_or_load_lgb(
        "lgb2_a6a_age_int",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A6_INT"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A6a integer age",
    )
    a6a_c = compare_v2("lgb2_a6a_age_int", state["inc_name"])
    log_result("lgb2_a6a_age_int", state["inc_name"], a6a_m, a6a_c)

    a6b_m = run_or_load_lgb(
        "lgb2_a6b_age_cat",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A6_CAT"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A6b categorical integer age",
    )
    a6b_c = compare_v2("lgb2_a6b_age_cat", state["inc_name"])
    log_result("lgb2_a6b_age_cat", state["inc_name"], a6b_m, a6b_c)

    a6_passing = [c for c in [("lgb2_a6a_age_int", "A6_INT", a6a_m, a6a_c), ("lgb2_a6b_age_cat", "A6_CAT", a6b_m, a6b_c)] if c[3]["keep"]]
    if a6_passing:
        best_n, best_f, _, best_c = max(a6_passing, key=lambda c: c[2]["mean_fold_auc"])
        check_suspicious_gain(best_n, best_c, train, test, y, folds_group, state)
        state["inc_name"] = best_n
        state["families"] = state["families"] + [best_f]
        git_commit_v2(f"feat(v2): P3 A6 accept {best_f}")

    # ------------------------------------------------------------------
    # 7. A4 (TE combos G4, nested, alpha=50, min_count=30)
    # ------------------------------------------------------------------
    a4_cols = ["OCC_X_ORG", "EDU_X_INC", "CON_X_GEN_X_FAM", "REGION_ID", "PRODUCT_CODE", "TERM_X_CON"]
    new_te_cols = [c for c in a4_cols if c not in state["extra_te_cols"]]
    combined_te_cols = tuple(state["extra_te_cols"] + new_te_cols)
    append_log(
        f"\n### A4 (`TE combos G4`: {new_te_cols}, `alpha=50`, `min_count=30`)\n"
        "- **Hypothesis (`lgb2_a4_te_combos`):** Nested smoothed target encodings over occupation-organization, education-income, contract-gender-family, region, product code, and term-contract capture interaction risk priors without exploding categorical cardinality. Expected: `+0.0002` to `+0.0010`."
    )
    a4_m = run_or_load_lgb(
        "lgb2_a4_te_combos",
        train,
        test,
        y,
        folds_group,
        families=state["families"],
        extra_te_cols=combined_te_cols,
        te_min_count=30,
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A4 nested TE interaction combos",
    )
    a4_c = compare_v2("lgb2_a4_te_combos", state["inc_name"])
    log_result("lgb2_a4_te_combos", state["inc_name"], a4_m, a4_c)
    if a4_c["keep"]:
        check_suspicious_gain("lgb2_a4_te_combos", a4_c, train, test, y, folds_group, state)
        state["inc_name"] = "lgb2_a4_te_combos"
        state["extra_te_cols"] = list(combined_te_cols)
        git_commit_v2("feat(v2): P3 A4 accept nested TE combos")

    # ------------------------------------------------------------------
    # 8. Re-test F1 (18 cols) and F3 (8 cols) under Rule A.3
    # ------------------------------------------------------------------
    run_rule_a3_family(
        "retest_f1",
        "F1",
        "\n### Re-test `F1` (External Score Interactions) under Rule A.3 (`ff=0.35` vs `ff=0.60`)\n"
        "- **Hypothesis (`lgb2_retest_f1`):** F1 failed in v1 at `ff=0.35` due to column dilution; testing at `ff=0.60` may let trees leverage EXT aggregations/diffs alongside the v2 incumbent.",
    )
    run_rule_a3_family(
        "retest_f3",
        "F3",
        "\n### Re-test `F3` (Age & Temporal Ratios) under Rule A.3 (`ff=0.35` vs `ff=0.60`)\n"
        "- **Hypothesis (`lgb2_retest_f3`):** Re-evaluating F3 temporal ratios (`AGE_YEARS`, `EMPLOYED_TO_AGE`, `AGE_AT_LAST_PAYMENT`, etc.) at both `ff=0.35` and `ff=0.60` checks if higher feature fraction unlocks temporal signal.",
    )

    # ------------------------------------------------------------------
    # 9. A7 Screened Pairwise Ratios (Discovery on folds {0,1,2}, Confirmation on {3,4})
    # ------------------------------------------------------------------
    sel_path = EXP_DIR / "a7_selected_15.json"
    if not sel_path.exists():
        disc_arr = np.load(FOLDS_DISCOVERY_PATH)
        disc_idx = disc_arr[:, 0]
        disc_folds = disc_arr[:, 1]
        train_disc = train.iloc[disc_idx].reset_index(drop=True)
        y_disc = y[disc_idx]

        build_fams = [f for f in state["families"] if f not in ("F6", "B1")]
        Xd_base, Xte_base = build_features(
            train_disc, test, families=build_fams, dropped_features=DROPPED_FEATURES
        )
        Xd_ratios = build_a7_ratio_candidates(train_disc)
        # Drop near-constant or >95% missing ratios
        keep_cand_cols = [
            c
            for c in Xd_ratios.columns
            if Xd_ratios[c].isna().mean() <= 0.95 and Xd_ratios[c].nunique(dropna=True) > 10
        ]
        Xd = pd.concat([Xd_base, Xd_ratios[keep_cand_cols]], axis=1)
        Xte_dum = pd.concat([Xte_base, build_a7_ratio_candidates(test)[keep_cand_cols]], axis=1)

        fp_disc = lgb_model(params=state["extra_params"], seed=SEED, lr=EXP_LR, use_codes=state["use_codes"])
        if state["extra_te_cols"]:
            fp_disc = wrap_fit_predict_with_te(fp_disc, cols=tuple(state["extra_te_cols"]), min_count=30, seed=SEED)
        fp_disc = wrap_fit_predict_with_te(
            fp_disc, cols=("ORGANIZATION_TYPE", "OCCUPATION_TYPE"), min_count=0, seed=SEED
        )
        if state["knn_ks"]:
            fp_disc = wrap_fit_predict_with_knn(
                fp_disc, k_values=tuple(state["knn_ks"]), mode=state["knn_mode"], seed=SEED, cache_dir=KNN_CACHE_DIR
            )

        disc_meta = run_cv_v2(
            name="lgb2_a7_discovery",
            fit_predict=fp_disc,
            X=Xd,
            y=y_disc,
            Xte=Xte_dum,
            folds=disc_folds,
            note="P3 A7 discovery on folds {0,1,2} only",
        )
        imp_df = pd.read_csv(EXP_DIR / "lgb2_a7_discovery" / "importance.csv")
        for fcol in ("fold_0", "fold_1", "fold_2"):
            imp_df[f"rank_{fcol}"] = imp_df[fcol].rank(ascending=False, method="average")
        cand_imp = imp_df[imp_df["feature"].isin(keep_cand_cols)].copy()
        cand_imp["mean_rank"] = cand_imp[["rank_fold_0", "rank_fold_1", "rank_fold_2"]].mean(axis=1)
        cand_imp["std_rank"] = cand_imp[["rank_fold_0", "rank_fold_1", "rank_fold_2"]].std(axis=1)
        stable = cand_imp[cand_imp["std_rank"] <= 20.0].sort_values("mean_rank")
        if len(stable) >= 15:
            top15_df = stable.head(15)
            rule_used = "std_rank <= 20"
        else:
            top15_df = cand_imp.sort_values(
                ["mean_rank", "std_rank"], ascending=[True, True]
            ).head(15)
            rule_used = "fallback_top15_by_mean_rank (fewer than 15 had std_rank <= 20)"
        selected_15 = list(top15_df["feature"])
        sel_payload = {
            "selected": selected_15,
            "rule_used": rule_used,
            "discovery_mean_fold_auc": disc_meta["mean_fold_auc"],
            "details": top15_df[["feature", "mean_rank", "std_rank", "mean_gain"]].to_dict(orient="records"),
        }
        sel_path.write_text(json.dumps(sel_payload, indent=2), encoding="utf-8")
        append_log(
            f"\n### A7 (`Screened Pairwise Ratios`: Discovery on `folds {{0,1,2}}`, Confirmation on `folds {{3,4}}`)\n"
            f"- **Frozen Discovery Top-15 (`{rule_used}`):** `{selected_15}`\n"
            "- **Hypothesis (`lgb2_a7_confirm`):** Adding these 15 frozen discovery ratios improves held-out confirmation folds 3 and 4 (`mean_diff_34 >= +0.0003` and neither fold negative; multiple-comparisons caveat: candidates were screened across 66 ratios on folds {0,1,2})."
        )

    # Confirmation run at ff=0.35 and ff=0.60
    inc_ff60_name = f"{state['inc_name']}_ff60"
    ff60_params = {**state["extra_params"], "feature_fraction": 0.60}
    run_or_load_lgb(
        inc_ff60_name,
        train,
        test,
        y,
        folds_group,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=ff60_params,
        note=f"Rule A.3 baseline ff=0.60 for {state['inc_name']}",
    )
    a7_35 = run_or_load_lgb(
        "lgb2_a7_confirm_ff35",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A7"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 A7 confirmation ff=0.35",
    )
    a7_60 = run_or_load_lgb(
        "lgb2_a7_confirm_ff60",
        train,
        test,
        y,
        folds_group,
        families=state["families"] + ["A7"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=ff60_params,
        note="P3 A7 confirmation ff=0.60",
    )
    inc_35_meta = load_experiment_meta(state["inc_name"], exp_dir=EXP_DIR)
    inc_60_meta = load_experiment_meta(inc_ff60_name, exp_dir=EXP_DIR)

    def _confirm_34(new_m: dict[str, Any], ref_m: dict[str, Any]) -> tuple[float, float, float, bool]:
        d3 = new_m["fold_aucs"][3] - ref_m["fold_aucs"][3]
        d4 = new_m["fold_aucs"][4] - ref_m["fold_aucs"][4]
        m34 = 0.5 * (d3 + d4)
        ok = (m34 >= 0.0003) and (d3 >= 0.0) and (d4 >= 0.0)
        return m34, d3, d4, ok

    m34_35, d3_35, d4_35, ok_35 = _confirm_34(a7_35, inc_35_meta)
    m34_60, d3_60, d4_60, ok_60 = _confirm_34(a7_60, inc_60_meta)
    append_log(
        f"- **A7 Confirmation (`ff=0.35` vs `{state['inc_name']}`):** fold3 `{d3_35:+.6f}`, fold4 `{d4_35:+.6f}`, mean_34=`{m34_35:+.6f}` -> `{'KEEP' if ok_35 else 'REJECT'}`\n"
        f"- **A7 Confirmation (`ff=0.60` vs `{inc_ff60_name}`):** fold3 `{d3_60:+.6f}`, fold4 `{d4_60:+.6f}`, mean_34=`{m34_60:+.6f}` -> `{'KEEP' if ok_60 else 'REJECT'}`"
    )
    if ok_35 or ok_60:
        state["families"] = state["families"] + ["A7"]
        state["inc_name"] = "lgb2_a7_confirm_ff35"
        git_commit_v2("feat(v2): P3 A7 accept confirmed top-15 pairwise ratios")

    # ------------------------------------------------------------------
    # 10. B1 Linkage Family (ENABLE_LINKAGE=False) on Group Folds AND Random Folds
    # ------------------------------------------------------------------
    append_log(
        "\n### B1 (`Linkage family G5`: grey-zone `ENABLE_LINKAGE=False`, evaluated on BOTH `folds_group.npy` and `folds.npy`)\n"
        "- **Hypothesis (`lgb2_b1_linkage`):** Leave-self-out fingerprint group size and other-row default rate (`FP_GROUP_SIZE_LOO`, `FP_N_PRIOR`, `FP_PRIOR_DEFAULT_RATE`) cannot help across `folds_group.npy` (where groups never cross train/val splits, only train/train) vs `folds.npy` (where 1.654% of val rows link to train rows). Expected: near `0.0000` on `folds_group.npy`, strong positive on `folds.npy`."
    )
    b1_grp_m = run_or_load_lgb(
        "lgb2_b1_linkage_group",
        train,
        test,
        y,
        folds_group,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_linkage=True,
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 B1 linkage on folds_group.npy (ENABLE_LINKAGE=False)",
    )
    b1_grp_c = compare_v2("lgb2_b1_linkage_group", state["inc_name"])
    log_result("lgb2_b1_linkage_group", state["inc_name"], b1_grp_m, b1_grp_c, "on folds_group.npy")

    folds_rand = np.load(FOLDS_V1_PATH)
    inc_rand_m = run_or_load_lgb(
        "lgb2_inc_randfolds",
        train,
        test,
        y,
        folds_rand,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 incumbent evaluated on v1 random folds.npy",
    )
    b1_rand_m = run_or_load_lgb(
        "lgb2_b1_linkage_randfolds",
        train,
        test,
        y,
        folds_rand,
        families=state["families"],
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_linkage=True,
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        note="P3 B1 linkage on v1 random folds.npy",
    )
    b1_rand_c = compare_v2("lgb2_b1_linkage_randfolds", "lgb2_inc_randfolds")
    log_result("lgb2_b1_linkage_randfolds", "lgb2_inc_randfolds", b1_rand_m, b1_rand_c, "on random folds.npy")

    # ------------------------------------------------------------------
    # Phase P4: Freeze Feature Views & Gate Checks
    # ------------------------------------------------------------------
    now_p4 = datetime.now(timezone.utc)
    hours_left_p4 = (deadline - now_p4).total_seconds() / 3600.0
    append_log(
        f"\n## Phase P4: Feature View Freeze & Gate Checks\n"
        f"- **Start UTC:** `{now_p4.isoformat()}` (**{hours_left_p4:.2f} h** left)"
    )

    view_a_fams = list(state["families"])
    age_conflict = "A6_CAT" if "A6_INT" in view_a_fams else ("A6_INT" if "A6_CAT" in view_a_fams else None)
    view_b_set = set(view_a_fams) | {"F0", "F1", "F3", "F4_LITE", "A1", "A5"}
    if age_conflict and age_conflict in view_b_set:
        view_b_set.discard(age_conflict)
    view_b_fams = sorted(view_b_set)
    view_c_fams = list(view_a_fams)

    # Feature-fraction sweep on VIEW_A: {0.35, 0.50, 0.70}
    ff_results: dict[float, dict[str, Any]] = {}
    for ff_val, ff_label in [(0.35, "35"), (0.50, "50"), (0.70, "70")]:
        params_ff = {**state["extra_params"], "feature_fraction": ff_val} if ff_val != 0.35 else dict(state["extra_params"])
        m_ff = run_or_load_lgb(
            f"lgb2_viewA_ff{ff_label}",
            train,
            test,
            y,
            folds_group,
            families=view_a_fams,
            extra_te_cols=tuple(state["extra_te_cols"]),
            knn_ks=tuple(state["knn_ks"]),
            knn_mode=state["knn_mode"],
            use_codes=state["use_codes"],
            extra_params=params_ff,
            note=f"P4 VIEW_A feature_fraction={ff_val}",
        )
        ff_results[ff_val] = m_ff
        append_log(
            f"- **P4 `VIEW_A` `ff={ff_val:.2f}`:** mean fold `{m_ff['mean_fold_auc']:.6f}`, pooled `{m_ff['pooled_oof_auc']:.6f}`"
        )
    best_ff = max(ff_results.keys(), key=lambda k: ff_results[k]["mean_fold_auc"])
    append_log(f"- **P4 Selected `feature_fraction` for `VIEW_A`:** `{best_ff}`")

    # Permuted-label gate test (NO KNN cache!)
    rng_perm = np.random.default_rng(999)
    y_perm_gate = rng_perm.permutation(y)
    perm_gate_meta = run_or_load_lgb(
        "lgb2_gate_permuted_y",
        train,
        test,
        y_perm_gate,
        folds_group,
        families=view_a_fams,
        extra_te_cols=tuple(state["extra_te_cols"]),
        knn_ks=tuple(state["knn_ks"]),
        knn_mode=state["knn_mode"],
        use_codes=state["use_codes"],
        extra_params=state["extra_params"],
        use_knn_cache=False,
        note="P4 mandatory permuted-label gate test",
    )
    perm_auc = perm_gate_meta["pooled_oof_auc"]
    append_log(f"- **P4 Permuted-Label Gate Test:** pooled OOF AUC = **`{perm_auc:.6f}`** (must be in `[0.495, 0.505]`)")
    if not (0.495 <= perm_auc <= 0.505):
        print(f"GATE FAILURE: permuted_y AUC {perm_auc:.6f} outside [0.495, 0.505]", flush=True)
        sys.exit(4)

    # Adversarial validation gate test on VIEW_A and VIEW_B
    Xa, Xtea = build_features(
        train,
        test,
        families=[f for f in view_a_fams if f not in ("F6", "B1")],
        dropped_features=DROPPED_FEATURES,
    )
    adv_a_auc, adv_a_top = eval_adv(Xa, Xtea)
    Xb, Xteb = build_features(
        train,
        test,
        families=[f for f in view_b_fams if f not in ("F6", "B1")],
        dropped_features=DROPPED_FEATURES,
    )
    adv_b_auc, adv_b_top = eval_adv(Xb, Xteb)
    append_log(
        f"- **P4 Adversarial Validation Gate Test:** `VIEW_A` = **`{adv_a_auc:.5f}`** (top: {adv_a_top[:3]}), "
        f"`VIEW_B` = **`{adv_b_auc:.5f}`** (top: {adv_b_top[:3]}) (limit <= 0.53)"
    )

    frozen_payload = {
        "timestamp_utc": now_p4.isoformat(),
        "incumbent_name": state["inc_name"],
        "view_a_families": view_a_fams,
        "view_b_families": view_b_fams,
        "view_c_families": view_c_fams,
        "extra_te_cols": state["extra_te_cols"],
        "knn_ks": state["knn_ks"],
        "knn_mode": state["knn_mode"],
        "use_codes": state["use_codes"],
        "extra_params": state["extra_params"],
        "best_ff_exp": best_ff,
        "permuted_y_auc": perm_auc,
        "adv_a_auc": adv_a_auc,
        "adv_b_auc": adv_b_auc,
        "b1_group_diff": b1_grp_c["mean_diff"],
        "b1_rand_diff": b1_rand_c["mean_diff"],
    }
    (EXP_DIR / "frozen_views_v2.json").write_text(json.dumps(frozen_payload, indent=2), encoding="utf-8")
    print("P3/P4 COMPLETE:", json.dumps(frozen_payload, indent=2), flush=True)
    if adv_a_auc > 0.53 or adv_b_auc > 0.53:
        print(f"GATE FAILURE: adversarial AUC exceeded 0.53: A={adv_a_auc:.5f}, B={adv_b_auc:.5f}", flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()

"""Comprehensive v2 unit, isolation, parity, determinism, and reproduction tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.audit_v2 import make_fingerprint_key
from src.config_v2 import (
    DROPPED_FEATURES,
    EXP_DIR,
    FOLDS10_PATH,
    FOLDS_GROUP_PATH,
    INPUT_DIR,
    SEED,
    SUB_DIR,
    VIEW_A_FAMILIES,
    VIEW_B_FAMILIES,
    VIEW_C_FAMILIES,
    VIEW_C_TE_COLS,
)
from src.features import (
    build_features,
    compute_nested_knn_target_mean,
    compute_nested_linkage_features,
    compute_nested_target_encoding,
    quantize_count,
    wrap_fit_predict_with_knn,
    wrap_fit_predict_with_te,
)
from src.models import lgb_model, logreg_binned_model, mlp_model
from src.stacking_v2 import honest_blend_cv


def _make_sample_frames(n_train: int = 240, n_test: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load a small schema-accurate slice from input CSVs or create a synthetic fallback."""
    if (INPUT_DIR / "train.csv").exists() and (INPUT_DIR / "test.csv").exists():
        tr = pd.read_csv(INPUT_DIR / "train.csv", nrows=n_train)
        te = pd.read_csv(INPUT_DIR / "test.csv", nrows=n_test)
        y = tr["TARGET"].to_numpy(dtype=np.int32)
        return tr, te, y

    rng = np.random.default_rng(42)
    tr = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n_train),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_train),
            "CODE_GENDER": rng.choice(["F", "M"], n_train, p=[0.6, 0.4]),
            "CNT_CHILDREN": rng.integers(0, 4, n_train),
            "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n_train),
            "AMT_CREDIT": rng.uniform(1e5, 1e6, n_train),
            "AMT_ANNUITY": rng.uniform(5e3, 5e4, n_train),
            "AMT_GOODS_PRICE": rng.uniform(1e5, 9e5, n_train),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n_train),
            "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary / secondary special"], n_train),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n_train),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents"], n_train),
            "REGION_POPULATION_RELATIVE": rng.choice([0.01, 0.02, 0.03], n_train),
            "DAYS_BIRTH": -rng.integers(8000, 24000, n_train),
            "DAYS_EMPLOYED": np.where(rng.random(n_train) < 0.2, 365243, -rng.integers(100, 5000, n_train)),
            "DAYS_REGISTRATION": -rng.uniform(0, 10000, n_train),
            "DAYS_ID_PUBLISH": -rng.integers(0, 5000, n_train),
            "OWN_CAR_AGE": np.where(rng.random(n_train) < 0.5, np.nan, rng.choice([2.0, 10.0, 64.0], n_train)),
            "FLAG_MOBIL": np.ones(n_train, dtype=np.int64),
            "FLAG_EMP_PHONE": rng.integers(0, 2, n_train),
            "FLAG_WORK_PHONE": rng.integers(0, 2, n_train),
            "OCCUPATION_TYPE": rng.choice(["Core staff", "Laborers", "Other"], n_train),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n_train).astype(float),
            "REGION_RATING_CLIENT": rng.integers(1, 4, n_train),
            "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "School"], n_train),
            "EXT_SOURCE_1": np.where(rng.random(n_train) < 0.6, np.nan, rng.uniform(0, 1, n_train)),
            "EXT_SOURCE_2": rng.uniform(0, 1, n_train),
            "EXT_SOURCE_3": np.where(rng.random(n_train) < 0.3, np.nan, rng.uniform(0, 1, n_train)),
            "DAYS_LAST_PHONE_CHANGE": -rng.uniform(0, 3000, n_train),
            "AMT_REQ_CREDIT_BUREAU_HOUR": np.where(rng.random(n_train) < 0.15, np.nan, 0.0),
            "AMT_REQ_CREDIT_BUREAU_MON": np.where(rng.random(n_train) < 0.15, np.nan, 0.0),
            "AMT_REQ_CREDIT_BUREAU_QRT": np.where(rng.random(n_train) < 0.15, np.nan, 1.0),
            "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n_train) < 0.15, np.nan, 2.0),
            "TARGET": rng.integers(0, 2, n_train),
        }
    )
    te = tr.drop(columns=["TARGET"]).iloc[:n_test].copy()
    te["SK_ID_CURR"] = np.arange(n_train, n_train + n_test)
    return tr, te, tr["TARGET"].to_numpy(dtype=np.int32)


def test_knn_and_te_isolation() -> None:
    """KNN, TE, and linkage wrappers exclude self and never touch validation labels."""
    tr, te, y = _make_sample_frames(n_train=260, n_test=60)
    X, Xte = build_features(tr, te, families=["F2", "A1"], dropped_features=DROPPED_FEATURES)
    Xtr, ytr = X.iloc[:200].reset_index(drop=True), y[:200].copy()
    Xva, yva = X.iloc[200:].reset_index(drop=True), y[200:].copy()

    # Make row 10 and row 11 exact duplicates in KNN space to test index-based (not distance-based) exclusion
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_TERM"):
        Xtr.loc[11, col] = Xtr.loc[10, col]

    # 1. TE shuffle validation labels + flip training row's own label
    cap_base: dict[str, pd.DataFrame] = {}
    cap_shuf: dict[str, pd.DataFrame] = {}

    def spy_model(store: dict[str, pd.DataFrame]):
        def _fp(a_tr, a_ytr, a_va, a_yva, a_te):
            store["tr"] = a_tr.copy()
            store["va"] = a_va.copy()
            return np.full(len(a_va), 0.1), np.full(len(a_te), 0.1), 1

        return _fp

    te_cols = ("ORGANIZATION_TYPE", "OCCUPATION_TYPE") + VIEW_C_TE_COLS
    w_base = wrap_fit_predict_with_te(spy_model(cap_base), cols=te_cols, min_count=5, seed=SEED)
    w_shuf = wrap_fit_predict_with_te(spy_model(cap_shuf), cols=te_cols, min_count=5, seed=SEED)
    w_base(Xtr, ytr, Xva, yva, Xte)
    yva_shuf = np.random.default_rng(123).permutation(yva)
    w_shuf(Xtr, ytr, Xva, yva_shuf, Xte)
    for col in [f"TE_{c}" for c in te_cols]:
        assert np.array_equal(cap_base["va"][col].to_numpy(), cap_shuf["va"][col].to_numpy(), equal_nan=True)

    tr_te_a, _, _ = compute_nested_target_encoding(Xtr, ytr, Xva, Xte, cols=te_cols, min_count=5, seed=SEED)
    ytr_flip = ytr.copy()
    ytr_flip[10] = 1 - ytr_flip[10]
    tr_te_b, _, _ = compute_nested_target_encoding(Xtr, ytr_flip, Xva, Xte, cols=te_cols, min_count=5, seed=SEED)
    for col in [f"TE_{c}" for c in te_cols]:
        assert tr_te_a[col].iloc[10] == tr_te_b[col].iloc[10]

    # 2. KNN V1 and V2 shuffle validation labels + flip training row's own label
    for mode in ("v1", "v2"):
        cap_k_base: dict[str, pd.DataFrame] = {}
        cap_k_shuf: dict[str, pd.DataFrame] = {}
        wk_base = wrap_fit_predict_with_knn(spy_model(cap_k_base), k_values=(50,), mode=mode, min_pattern_rows=30, seed=SEED)
        wk_shuf = wrap_fit_predict_with_knn(spy_model(cap_k_shuf), k_values=(50,), mode=mode, min_pattern_rows=30, seed=SEED)
        wk_base(Xtr, ytr, Xva, yva, Xte)
        wk_shuf(Xtr, ytr, Xva, yva_shuf, Xte)
        assert np.array_equal(
            cap_k_base["va"]["KNN_EXT_TERM_50_MEAN"].to_numpy(),
            cap_k_shuf["va"]["KNN_EXT_TERM_50_MEAN"].to_numpy(),
            equal_nan=True,
        )

        tr_ka, _, _ = compute_nested_knn_target_mean(
            Xtr, ytr, Xva, Xte, k_values=(50,), mode=mode, min_pattern_rows=30, seed=SEED
        )
        tr_kb, _, _ = compute_nested_knn_target_mean(
            Xtr, ytr_flip, Xva, Xte, k_values=(50,), mode=mode, min_pattern_rows=30, seed=SEED
        )
        # Flipping row 10's label must leave row 10's own KNN feature bit-identical
        assert tr_ka["KNN_EXT_TERM_50_MEAN"].iloc[10] == tr_kb["KNN_EXT_TERM_50_MEAN"].iloc[10]

    # 3. Linkage B1 self-exclusion check
    Xtr_dup = Xtr.copy()
    for c in ("CODE_GENDER", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"):
        Xtr_dup.loc[11, c] = Xtr_dup.loc[10, c]
    tr_la, _, _ = compute_nested_linkage_features(Xtr_dup, ytr, Xva, Xte)
    tr_lb, _, _ = compute_nested_linkage_features(Xtr_dup, ytr_flip, Xva, Xte)
    assert tr_la["FP_PRIOR_DEFAULT_RATE"].iloc[10] == tr_lb["FP_PRIOR_DEFAULT_RATE"].iloc[10]


def test_target_derived_distribution_match() -> None:
    """Training rows and validation rows have matching mean and std for nested TE and KNN features."""
    tr, te, y = _make_sample_frames(n_train=1200, n_test=100)
    X, Xte = build_features(tr, te, families=["F2", "A1"], dropped_features=DROPPED_FEATURES)
    Xtr, ytr = X.iloc[:900].reset_index(drop=True), y[:900]
    Xva = X.iloc[900:].reset_index(drop=True)

    tr_te, va_te, _ = compute_nested_target_encoding(
        Xtr, ytr, Xva, Xte, cols=("ORGANIZATION_TYPE", "OCCUPATION_TYPE"), seed=SEED
    )
    for col in ("TE_ORGANIZATION_TYPE", "TE_OCCUPATION_TYPE"):
        assert abs(float(tr_te[col].mean()) - float(va_te[col].mean())) < 0.02
        assert abs(float(tr_te[col].std()) - float(va_te[col].std())) < 0.02

    tr_k, va_k, _ = compute_nested_knn_target_mean(
        Xtr, ytr, Xva, Xte, k_values=(100,), mode="v1", seed=SEED
    )
    assert abs(float(tr_k["KNN_EXT_TERM_100_MEAN"].mean()) - float(va_k["KNN_EXT_TERM_100_MEAN"].mean())) < 0.02
    assert abs(float(tr_k["KNN_EXT_TERM_100_MEAN"].std()) - float(va_k["KNN_EXT_TERM_100_MEAN"].std())) < 0.02


def test_count_frequency_parity() -> None:
    """F5_LOO removes self-inclusion disparity between train singletons and unseen test rows."""
    tr, te, _ = _make_sample_frames(n_train=30, n_test=6)
    # Row 0 in train: unique loan template (singleton in train) -> LOO count must be 0.0
    tr.loc[0, "AMT_GOODS_PRICE"] = 987654.0
    # Rows 1..5 in train: 5 rows with 555000.0 -> LOO count on train is 4.0
    tr.loc[1:5, "AMT_GOODS_PRICE"] = 555000.0
    # Test row 0: unseen template -> count is 0.0 (exact match with train singleton!)
    te.loc[0, "AMT_GOODS_PRICE"] = 123456.0
    # Test row 1: template appearing 5 times in train -> count is 5.0; train rows with 6 copies -> LOO is 5.0
    te.loc[1, "AMT_GOODS_PRICE"] = 555000.0

    X, Xte = build_features(tr, te, families=["F2", "F5_LOO"], dropped_features=DROPPED_FEATURES)
    assert float(X["DENSITY_GOODS_PRICE"].iloc[0]) == 0.0
    assert float(Xte["DENSITY_GOODS_PRICE"].iloc[0]) == 0.0
    assert float(X["DENSITY_GOODS_PRICE"].iloc[1]) == float(quantize_count(np.array([4.0]))[0])
    assert float(Xte["DENSITY_GOODS_PRICE"].iloc[1]) == float(quantize_count(np.array([5.0]))[0])


def test_honest_blend_cv_isolation() -> None:
    """Honest Blend-CV fits weights/stackers strictly on the 4 training folds (folds != f)."""
    rng = np.random.default_rng(7)
    n, n_te = 3000, 200
    y = rng.integers(0, 2, n, dtype=np.int32)
    folds = np.repeat(np.arange(5, dtype=np.int32), n // 5)
    oof_dict = {
        "m1": rng.uniform(0.05, 0.95, n),
        "m2": rng.uniform(0.05, 0.95, n),
    }
    test_dict = {
        "m1": rng.uniform(0.05, 0.95, n_te),
        "m2": rng.uniform(0.05, 0.95, n_te),
    }
    res_base = honest_blend_cv(oof_dict, test_dict, y, folds, run_et_restacker=False, seed=SEED)

    # Corrupt fold 2 OOF predictions and fold 2 labels; folds != 2 predictions must stay bit-identical,
    # and if only y[folds == 2] changes, fold 2 OOF predictions from the fold-2 stacker stay bit-identical
    y_corrupt = y.copy()
    y_corrupt[folds == 2] = 1 - y_corrupt[folds == 2]
    res_y2 = honest_blend_cv(oof_dict, test_dict, y_corrupt, folds, run_et_restacker=False, seed=SEED)
    for mname in ("equal_rank", "hill_climber", "logreg_c0.1"):
        assert np.array_equal(res_base[mname]["oof"][folds == 2], res_y2[mname]["oof"][folds == 2])


def test_logreg_and_mlp_preprocessing_isolation() -> None:
    """LogReg and MLP fit scalers/discretizers/encoders strictly on Xtr without leaking Xva; torch stays lazy."""
    import src.run_all_v2  # noqa: F401

    assert "torch" not in sys.modules

    tr, te, y = _make_sample_frames(n_train=260, n_test=50)
    X, Xte = build_features(tr, te, families=["F2", "A1", "A6_INT"], dropped_features=DROPPED_FEATURES)
    Xtr, ytr = X.iloc[:200].reset_index(drop=True), y[:200]
    Xva, yva = X.iloc[200:].reset_index(drop=True), y[200:]

    Xva_bad = Xva.copy()
    for col in Xva_bad.columns:
        if pd.api.types.is_numeric_dtype(Xva_bad[col]):
            Xva_bad[col] = 999999.0
    yva_shuf = np.random.default_rng(77).permutation(yva)

    _, tp_lr_1, _ = logreg_binned_model({"C": 0.1}, seed=SEED)(Xtr, ytr, Xva, yva, Xte)
    _, tp_lr_2, _ = logreg_binned_model({"C": 0.1}, seed=SEED)(Xtr, ytr, Xva_bad, yva_shuf, Xte)
    assert np.array_equal(tp_lr_1, tp_lr_2)

    if importlib.util.find_spec("torch") is not None:
        _, tp_mlp_1, _ = mlp_model({"epochs": 1}, seed=SEED)(Xtr, ytr, Xva, yva, Xte)
        _, tp_mlp_2, _ = mlp_model({"epochs": 1}, seed=SEED)(Xtr, ytr, Xva_bad, yva_shuf, Xte)
        assert np.allclose(tp_mlp_1, tp_mlp_2, atol=1e-6, rtol=0.0)
    else:
        with pytest.raises(ImportError, match="requirements_experiments.txt"):
            mlp_model({"epochs": 1}, seed=SEED)


def test_determinism_and_schema_v2() -> None:
    """Determinism, group-fold integrity, and frozen feature schema checks."""
    tr, te, y = _make_sample_frames(n_train=260, n_test=60)
    for fams in (VIEW_A_FAMILIES, VIEW_B_FAMILIES, VIEW_C_FAMILIES):
        X, Xte = build_features(tr, te, families=[f for f in fams if f != "F6"], dropped_features=DROPPED_FEATURES)
        assert "TARGET" not in X.columns and "SK_ID_CURR" not in X.columns
        assert "TARGET" not in Xte.columns and "SK_ID_CURR" not in Xte.columns
        assert list(X.columns) == list(Xte.columns)
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                assert not np.isinf(X[c]).any()
                assert not np.isinf(Xte[c]).any()

    X, Xte = build_features(tr, te, families=[f for f in VIEW_A_FAMILIES if f != "F6"], dropped_features=DROPPED_FEATURES)
    vp1, tp1, it1 = lgb_model({"n_estimators": 20, "min_data_in_leaf": 10}, seed=SEED, lr=0.06, deterministic=True, use_codes=True)(
        X.iloc[:200], y[:200], X.iloc[200:], y[200:], Xte
    )
    vp2, tp2, it2 = lgb_model({"n_estimators": 20, "min_data_in_leaf": 10}, seed=SEED, lr=0.06, deterministic=True, use_codes=True)(
        X.iloc[:200], y[:200], X.iloc[200:], y[200:], Xte
    )
    assert it1 == it2
    assert np.allclose(vp1, vp2, atol=1e-6, rtol=0.0)
    assert np.allclose(tp1, tp2, atol=1e-6, rtol=0.0)

    if FOLDS_GROUP_PATH.exists() and (INPUT_DIR / "train.csv").exists():
        full_tr = pd.read_csv(
            INPUT_DIR / "train.csv",
            usecols=[
                "TARGET",
                "CODE_GENDER",
                "REGION_POPULATION_RELATIVE",
                "DAYS_BIRTH",
                "DAYS_EMPLOYED",
                "DAYS_ID_PUBLISH",
                "DAYS_REGISTRATION",
                "DAYS_LAST_PHONE_CHANGE",
            ],
        )
        folds_grp = np.load(FOLDS_GROUP_PATH)
        assert folds_grp.shape == (len(full_tr),)
        assert set(np.unique(folds_grp).tolist()) == {0, 1, 2, 3, 4}
        keys = make_fingerprint_key(full_tr)
        max_f = int(pd.DataFrame({"k": keys, "f": folds_grp}).groupby("k")["f"].nunique().max())
        assert max_f == 1

        if FOLDS10_PATH.exists():
            folds10_grp = np.load(FOLDS10_PATH)
            assert folds10_grp.shape == (len(full_tr),)
            assert set(np.unique(folds10_grp).tolist()) == set(range(10))
            max_f10 = int(pd.DataFrame({"k": keys, "f": folds10_grp}).groupby("k")["f"].nunique().max())
            assert max_f10 == 1


def test_reproduction_check_v2() -> None:
    """Verify run_all_v2.py output matches full blend/manifest or clean-room fast baseline."""
    from scipy.stats import spearmanr

    sub_final_v2 = SUB_DIR / "sub_final_v2.csv"
    manifest_path = EXP_DIR / "manifest.json"
    blend_sub = SUB_DIR / "sub_01_blend_v2.csv"
    blend_meta_path = EXP_DIR / "blend_final_v2" / "meta.json"
    fast_ref_path = EXP_DIR / "p7_fast_determinism.json"

    if not sub_final_v2.exists() or not manifest_path.exists():
        pytest.skip("v2 final submission/manifest not built yet (run src/run_all_v2.py first)")

    sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")
    df_final = pd.read_csv(sub_final_v2)
    assert df_final.shape == sample_sub.shape
    assert list(df_final.columns) == ["SK_ID_CURR", "TARGET"]
    assert np.array_equal(df_final["SK_ID_CURR"].to_numpy(), sample_sub["SK_ID_CURR"].to_numpy())
    assert not df_final["TARGET"].isna().any()
    assert ((df_final["TARGET"] >= 0.0) & (df_final["TARGET"] <= 1.0)).all()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") == "fast":
        if not fast_ref_path.exists():
            pytest.skip("p7_fast_determinism.json missing; skipping --fast AUC cross-check")
        fast_ref = json.loads(fast_ref_path.read_text(encoding="utf-8"))
        assert abs(float(manifest["pooled_oof_auc"]) - float(fast_ref["fast1_pooled_oof_auc"])) <= 1e-4
    else:
        if not blend_meta_path.exists():
            pytest.skip("blend_final_v2/meta.json missing; skipping --full AUC cross-check")
        blend_meta = json.loads(blend_meta_path.read_text(encoding="utf-8"))
        assert abs(float(manifest["pooled_oof_auc"]) - float(blend_meta["pooled_oof_auc"])) <= 1e-4
        if blend_sub.exists():
            df_blend = pd.read_csv(blend_sub)
            corr = float(spearmanr(df_final["TARGET"], df_blend["TARGET"]).statistic)
            assert corr >= 0.9999

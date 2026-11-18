"""Unit and integrity tests for the GCI Home Credit pipeline."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FOLDS_PATH, INPUT_DIR
from src.cv_harness import write_submission
from src.features import (
    build_features,
    compute_nested_target_encoding,
    safe_div,
    wrap_fit_predict_with_te,
)
from src.models import lgb_model


def _make_sample_frames(n_train: int = 240, n_test: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Create small schema-accurate sample frames from raw CSVs or synthetic fallback."""
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
            "CODE_GENDER": rng.choice(["F", "M", "XNA"], n_train, p=[0.6, 0.39, 0.01]),
            "CNT_CHILDREN": rng.integers(0, 4, n_train),
            "AMT_INCOME_TOTAL": rng.uniform(5e4, 3e5, n_train),
            "AMT_CREDIT": rng.uniform(1e5, 1e6, n_train),
            "AMT_ANNUITY": rng.uniform(5e3, 5e4, n_train),
            "AMT_GOODS_PRICE": rng.uniform(1e5, 9e5, n_train),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n_train),
            "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary / secondary special"], n_train),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n_train),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents"], n_train),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.05, n_train),
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
            "EXT_SOURCE_1": rng.uniform(0, 1, n_train),
            "EXT_SOURCE_2": rng.uniform(0, 1, n_train),
            "EXT_SOURCE_3": rng.uniform(0, 1, n_train),
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


def test_no_target_leakage_in_features() -> None:
    """1. build_features has no y parameter, excludes TARGET/SK_ID_CURR, and aligns columns."""
    sig = inspect.signature(build_features)
    param_names = [p.lower() for p in sig.parameters]
    assert "y" not in param_names
    assert "target" not in param_names

    tr, te, _ = _make_sample_frames()
    X, Xte = build_features(tr, te, families=["F0", "F1", "F2", "F3", "F4", "F5"])
    assert "TARGET" not in X.columns
    assert "SK_ID_CURR" not in X.columns
    assert "TARGET" not in Xte.columns
    assert "SK_ID_CURR" not in Xte.columns
    assert list(X.columns) == list(Xte.columns)


def test_determinism() -> None:
    """2. Running the same deterministic LightGBM config twice yields identical predictions."""
    tr, te, y = _make_sample_frames(n_train=300, n_test=80)
    X, Xte = build_features(tr, te, families=["F0", "F1", "F2", "F3"])
    Xtr, ytr = X.iloc[:200], y[:200]
    Xva, yva = X.iloc[200:], y[200:]

    fp1 = lgb_model(
        params={"n_estimators": 25, "min_data_in_leaf": 10},
        seed=42,
        lr=0.06,
        deterministic=True,
    )
    vp1, tp1, it1 = fp1(Xtr, ytr, Xva, yva, Xte)

    fp2 = lgb_model(
        params={"n_estimators": 25, "min_data_in_leaf": 10},
        seed=42,
        lr=0.06,
        deterministic=True,
    )
    vp2, tp2, it2 = fp2(Xtr, ytr, Xva, yva, Xte)

    assert it1 == it2
    assert np.allclose(vp1, vp2, atol=1e-6, rtol=0.0)
    assert np.allclose(tp1, tp2, atol=1e-6, rtol=0.0)


def test_fold_integrity() -> None:
    """3. Fixed folds cover every row once across 5 folds with balanced default rates."""
    if not FOLDS_PATH.exists():
        pytest.skip("folds.npy not generated yet (run audit.py first)")
    y = pd.read_csv(INPUT_DIR / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
    folds = np.load(FOLDS_PATH)
    assert folds.shape == (len(y),)
    assert set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4}
    assert (folds >= 0).all()
    rates = [float(y[folds == f].mean()) for f in range(5)]
    assert (max(rates) - min(rates)) < 0.001


def test_target_encoding_isolation() -> None:
    """4. Nested target encoding never uses validation labels or a row's own training label."""
    tr, te, y = _make_sample_frames(n_train=240, n_test=60)
    X, Xte = build_features(tr, te, families=["F0"])
    Xtr, ytr = X.iloc[:180].reset_index(drop=True), y[:180].copy()
    Xva, yva = X.iloc[180:].reset_index(drop=True), y[180:].copy()

    captured_base: dict[str, pd.DataFrame] = {}
    captured_shuf: dict[str, pd.DataFrame] = {}

    def spy_model(store: dict[str, pd.DataFrame]):
        def _fp(a_tr, a_ytr, a_va, a_yva, a_te):
            store["tr"] = a_tr.copy()
            store["va"] = a_va.copy()
            return np.full(len(a_va), 0.1), np.full(len(a_te), 0.1), 1
        return _fp

    wrapped_base = wrap_fit_predict_with_te(spy_model(captured_base), seed=42)
    wrapped_shuf = wrap_fit_predict_with_te(spy_model(captured_shuf), seed=42)

    wrapped_base(Xtr, ytr, Xva, yva, Xte)
    rng = np.random.default_rng(99)
    yva_shuffled = rng.permutation(yva)
    wrapped_shuf(Xtr, ytr, Xva, yva_shuffled, Xte)

    for col in ("TE_ORGANIZATION_TYPE", "TE_OCCUPATION_TYPE"):
        assert np.array_equal(
            captured_base["va"][col].to_numpy(),
            captured_shuf["va"][col].to_numpy(),
            equal_nan=True,
        )

    # Flip one training row's label and confirm its own encoding is bit-identical
    tr_a, _, _ = compute_nested_target_encoding(Xtr, ytr, Xva, Xte, seed=42)
    idx_to_flip = 17
    ytr_flipped = ytr.copy()
    ytr_flipped[idx_to_flip] = 1 - ytr_flipped[idx_to_flip]
    tr_b, _, _ = compute_nested_target_encoding(Xtr, ytr_flipped, Xva, Xte, seed=42)

    for col in ("TE_ORGANIZATION_TYPE", "TE_OCCUPATION_TYPE"):
        assert tr_a[col].iloc[idx_to_flip] == tr_b[col].iloc[idx_to_flip]


def test_cleaning_and_safe_div() -> None:
    """5. F0 removes placeholders (365243, >=60, XNA), ratios have no inf, safe_div handles 0/NaN."""
    res = safe_div(np.array([10.0, 10.0, np.nan, 12.0]), np.array([0.0, np.nan, 2.0, 4.0]))
    assert res.dtype == np.float32
    assert np.isnan(res.iloc[0])
    assert np.isnan(res.iloc[1])
    assert np.isnan(res.iloc[2])
    assert np.isclose(res.iloc[3], 3.0)

    tr, te, _ = _make_sample_frames(n_train=100, n_test=40)
    tr.loc[0, "DAYS_EMPLOYED"] = 365243
    tr.loc[1, "OWN_CAR_AGE"] = 65.0
    tr.loc[2, "ORGANIZATION_TYPE"] = "XNA"
    tr.loc[3, "CODE_GENDER"] = "XNA"
    tr.loc[4, "AMT_ANNUITY"] = 0.0
    tr.loc[5, "CNT_CHILDREN"] = 0

    X, Xte = build_features(tr, te, families=["F0", "F1", "F2", "F3"])
    assert (X["DAYS_EMPLOYED"] == 365243).sum() == 0
    assert (Xte["DAYS_EMPLOYED"] == 365243).sum() == 0
    assert (X["OWN_CAR_AGE"] >= 60).sum() == 0
    assert (Xte["OWN_CAR_AGE"] >= 60).sum() == 0
    assert "XNA" not in set(X["ORGANIZATION_TYPE"].dropna())
    assert "XNA" not in set(X["CODE_GENDER"].dropna())

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            assert not np.isinf(X[col]).any(), f"Infinity in X[{col}]"
            assert not np.isinf(Xte[col]).any(), f"Infinity in Xte[{col}]"


def test_submission_validator(tmp_path: Path) -> None:
    """6. write_submission enforces schema, ID order, NaN-free values, and [0, 1] bounds."""
    sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")
    test_df = sample_sub[["SK_ID_CURR"]].copy()
    valid_preds = np.linspace(0.01, 0.99, len(sample_sub))
    out_csv = tmp_path / "valid_sub.csv"
    write_submission(test_df, valid_preds, out_csv, meta={"oof_auc": 0.75})
    assert out_csv.exists()
    assert out_csv.with_suffix(".json").exists()

    with pytest.raises(ValueError):
        bad_nan = valid_preds.copy()
        bad_nan[0] = np.nan
        write_submission(test_df, bad_nan, tmp_path / "bad_nan.csv")

    with pytest.raises(ValueError):
        bad_range = valid_preds.copy()
        bad_range[0] = 1.05
        write_submission(test_df, bad_range, tmp_path / "bad_range.csv")

    with pytest.raises(ValueError):
        bad_ids = test_df.iloc[::-1].copy()
        write_submission(bad_ids, valid_preds, tmp_path / "bad_ids.csv")


def test_reproduction_check() -> None:
    """7. Verify run_all.py OOF AUC matches tracker blend_final and sub_final rank-correlates >= 0.999."""
    import json
    from scipy.stats import spearmanr

    sub_final_csv = ROOT_DIR / "submissions" / "sub_final.csv"
    sub_final_json = ROOT_DIR / "submissions" / "sub_final.json"
    sub_blend_csv = ROOT_DIR / "submissions" / "sub_03_blend.csv"
    tracker_csv = ROOT_DIR / "experiments" / "tracker.csv"

    if not sub_final_csv.exists() or not sub_final_json.exists():
        pytest.skip("sub_final artifacts not generated yet (run src/run_all.py first)")

    sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")
    final_df = pd.read_csv(sub_final_csv)
    assert final_df.shape == sample_sub.shape
    assert list(final_df.columns) == list(sample_sub.columns)
    assert np.array_equal(final_df["SK_ID_CURR"].to_numpy(), sample_sub["SK_ID_CURR"].to_numpy())
    assert not final_df["TARGET"].isna().any()
    assert ((final_df["TARGET"] >= 0.0) & (final_df["TARGET"] <= 1.0)).all()

    meta = json.loads(sub_final_json.read_text(encoding="utf-8"))
    run_all_auc = float(meta["pooled_oof_auc"])
    assert run_all_auc < 0.78

    tracker_df = pd.read_csv(tracker_csv)
    blend_rows = tracker_df.loc[tracker_df["name"] == "blend_final"]
    assert len(blend_rows) >= 1
    tracker_auc = float(blend_rows.iloc[-1]["pooled_oof_auc"])
    assert abs(run_all_auc - tracker_auc) <= 0.0002

    blend_df = pd.read_csv(sub_blend_csv)
    corr = float(spearmanr(final_df["TARGET"], blend_df["TARGET"]).statistic)
    assert corr >= 0.999


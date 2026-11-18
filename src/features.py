"""Deterministic, leak-free feature engineering pipeline for GCI Home Credit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

from src.cv_harness import align_categories

# 25 dominant annuity-term peaks (local maxima with count >= 500 of AMT_CREDIT / AMT_ANNUITY rounded to 0.1)
# observed on input/train.csv in src/audit_v2.py P1(d) (documented in experiments_v2/audit_v2.md).
_DOMINANT_TERM_MODES = np.array(
    [
        8.4, 9.2, 9.4, 10.1, 12.6, 13.7, 14.9, 15.6, 16.5, 17.3,
        18.4, 19.5, 20.0, 20.7, 21.3, 22.6, 23.5, 25.1, 26.3, 27.7,
        30.1, 30.8, 34.1, 36.4, 37.8,
    ],
    dtype=np.float64,
)

_BUREAU_CANDIDATES = (
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)


def safe_div(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    """Divide numerator by denominator, returning float32 with NaN for zero/NaN/inf."""
    num = pd.Series(numerator, dtype="float64")
    den = pd.Series(denominator, dtype="float64")
    den_clean = den.where((den != 0.0) & (~den.isna()), np.nan)
    res = num / den_clean
    res = res.replace([np.inf, -np.inf], np.nan)
    return res.astype(np.float32)


def _clean_days_employed(df: pd.DataFrame) -> pd.Series:
    """Private helper: return DAYS_EMPLOYED with 365243 placeholder replaced by NaN."""
    # 365243 is the official Home Credit pensioner/unemployed sentinel (30,898 train rows, tutorial.ipynb / data_profile.md).
    s = df["DAYS_EMPLOYED"].astype("float64")
    return s.where(s != 365243.0, np.nan)


def _clean_own_car_age(df: pd.DataFrame) -> pd.Series:
    """Private helper: return OWN_CAR_AGE with >= 60 outliers replaced by NaN."""
    # >= 60 outlier threshold identified in input/train.csv EDA (1,826 rows at 64/65/91 years, experiments/data_profile.md).
    s = df["OWN_CAR_AGE"].astype("float64")
    return s.where(s < 60.0, np.nan)


def _age_years(df: pd.DataFrame) -> pd.Series:
    """Private helper: positive age in years."""
    return (-df["DAYS_BIRTH"].astype("float64") / 365.25).astype(np.float32)


def _credit_term(df: pd.DataFrame) -> pd.Series:
    """Private helper: loan duration proxy AMT_CREDIT / AMT_ANNUITY."""
    return safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])


def _ext_mean(df: pd.DataFrame) -> pd.Series:
    """Private helper: row-wise mean of available raw EXT_SOURCE_1/2/3 scores."""
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
    return df[ext_cols].astype("float64").mean(axis=1, skipna=True).astype(np.float32)


def _clean_cat_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known placeholder tokens in categorical columns with NaN."""
    # Placeholder strings documented in experiments/data_profile.md ("XNA" in ORGANIZATION_TYPE/CODE_GENDER; "Unknown" in NAME_FAMILY_STATUS).
    out = df.copy()
    for col in ("ORGANIZATION_TYPE", "CODE_GENDER"):
        if col in out.columns:
            s = out[col].astype("object")
            out[col] = s.where(s != "XNA", np.nan)
    if "NAME_FAMILY_STATUS" in out.columns:
        s = out["NAME_FAMILY_STATUS"].astype("object")
        out["NAME_FAMILY_STATUS"] = s.where(s != "Unknown", np.nan)
    return out


def _add_f0(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    raw_cols: list[str],
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F0: Missingness counts, placeholder flags, and anomaly replacement with NaN."""
    tr_nan_cnt = tr[raw_cols].isna().sum(axis=1).astype(np.float32)
    te_nan_cnt = te[raw_cols].isna().sum(axis=1).astype(np.float32)

    for df, nan_cnt in ((tr, tr_nan_cnt), (te, te_nan_cnt)):
        df["NAN_COUNT"] = nan_cnt
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.float32)
            df["DAYS_EMPLOYED"] = _clean_days_employed(df).astype(np.float32)
        else:
            skipped.append("F0:DAYS_EMPLOYED_ANOM")

        if "OWN_CAR_AGE" in df.columns:
            df["OWN_CAR_AGE_ANOM"] = (df["OWN_CAR_AGE"] >= 60).fillna(False).astype(np.float32)
            df["OWN_CAR_AGE"] = _clean_own_car_age(df).astype(np.float32)
        else:
            skipped.append("F0:OWN_CAR_AGE_ANOM")

    tr = _clean_cat_placeholders(tr)
    te = _clean_cat_placeholders(te)

    bureau_cols = [c for c in _BUREAU_CANDIDATES if c in raw_cols]
    missing_bureau = [c for c in _BUREAU_CANDIDATES if c not in raw_cols]
    for mc in missing_bureau:
        skipped.append(f"F0:missing_input:{mc}")

    if bureau_cols:
        tr_all_na = tr[bureau_cols].isna().all(axis=1)
        tr_any_na = tr[bureau_cols].isna().any(axis=1)
        te_all_na = te[bureau_cols].isna().all(axis=1)
        te_any_na = te[bureau_cols].isna().any(axis=1)
        tr["BUREAU_MISSING"] = tr_all_na.astype(np.float32)
        te["BUREAU_MISSING"] = te_all_na.astype(np.float32)
        if not ((tr_all_na == tr_any_na).all() and (te_all_na == te_any_na).all()):
            tr["BUREAU_MISSING_COUNT"] = tr[bureau_cols].isna().sum(axis=1).astype(np.float32)
            te["BUREAU_MISSING_COUNT"] = te[bureau_cols].isna().sum(axis=1).astype(np.float32)
    else:
        skipped.append("F0:BUREAU_MISSING")

    return tr, te


def _add_f1(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F1: External credit score aggregations, weighted mean, products, and ratios."""
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in tr.columns]
    if len(ext_cols) < 3:
        skipped.append("F1:incomplete_EXT_SOURCE_columns")
        return tr, te

    # Domain-prior weights ordering EXT_SOURCE_3 > EXT_SOURCE_1 > EXT_SOURCE_2, matching univariate train AUCs
    # (0.6796 > 0.6647 > 0.6582 in experiments/data_profile.md); used only in VIEW_B F1.
    weights = np.array([2.0, 1.0, 3.0], dtype=np.float64)

    for df in (tr, te):
        ext_vals = df[ext_cols].astype("float64")
        arr = ext_vals.to_numpy()
        valid_mask = ~np.isnan(arr)

        df["EXT_MEAN"] = ext_vals.mean(axis=1, skipna=True).astype(np.float32)
        df["EXT_STD"] = ext_vals.std(axis=1, skipna=True, ddof=1).astype(np.float32)
        df["EXT_MIN"] = ext_vals.min(axis=1, skipna=True).astype(np.float32)
        df["EXT_MAX"] = ext_vals.max(axis=1, skipna=True).astype(np.float32)
        df["EXT_MEDIAN"] = ext_vals.median(axis=1, skipna=True).astype(np.float32)
        df["EXT_PROD"] = ext_vals.prod(axis=1, skipna=True, min_count=1).astype(np.float32)
        df["EXT_NMISS"] = ext_vals.isna().sum(axis=1).astype(np.float32)

        w_sum = (valid_mask * weights).sum(axis=1)
        wx_sum = np.where(valid_mask, arr * weights, 0.0).sum(axis=1)
        wmean = np.where(w_sum > 0.0, wx_sum / np.where(w_sum > 0.0, w_sum, 1.0), np.nan)
        df["EXT_WMEAN"] = wmean.astype(np.float32)

        s1 = ext_vals["EXT_SOURCE_1"]
        s2 = ext_vals["EXT_SOURCE_2"]
        s3 = ext_vals["EXT_SOURCE_3"]
        df["EXT_1_X_2"] = (s1 * s2).astype(np.float32)
        df["EXT_1_MINUS_2"] = (s1 - s2).astype(np.float32)
        df["EXT_1_X_3"] = (s1 * s3).astype(np.float32)
        df["EXT_1_MINUS_3"] = (s1 - s3).astype(np.float32)
        df["EXT_2_X_3"] = (s2 * s3).astype(np.float32)
        df["EXT_2_MINUS_3"] = (s2 - s3).astype(np.float32)

        if "DAYS_BIRTH" in df.columns:
            age = _age_years(df)
            df["EXT_1_DIV_AGE"] = safe_div(s1, age)
            df["EXT_2_DIV_AGE"] = safe_div(s2, age)
            df["EXT_3_DIV_AGE"] = safe_div(s3, age)
        else:
            skipped.append("F1:EXT_k_DIV_AGE")

        if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
            term = _credit_term(df)
            df["EXT_MEAN_X_CREDIT_TERM"] = (
                df["EXT_MEAN"].astype("float64") * term.astype("float64")
            ).astype(np.float32)
        else:
            skipped.append("F1:EXT_MEAN_X_CREDIT_TERM")

    return tr, te


def _add_f2(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F2: Loan affordability and debt-burden ratios (annuity confirmed annual-scale)."""
    for df in (tr, te):
        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
        else:
            skipped.append("F2:ANNUITY_INCOME_RATIO")

        if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
        else:
            skipped.append("F2:CREDIT_INCOME_RATIO")

        if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
            df["CREDIT_TERM"] = _credit_term(df)
        else:
            skipped.append("F2:CREDIT_TERM")

        if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
            df["CREDIT_GOODS_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
        else:
            skipped.append("F2:CREDIT_GOODS_RATIO")

        if {"AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            df["GOODS_INCOME_RATIO"] = safe_div(df["AMT_GOODS_PRICE"], df["AMT_INCOME_TOTAL"])
        else:
            skipped.append("F2:GOODS_INCOME_RATIO")

        if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
            df["CREDIT_MINUS_GOODS"] = (
                df["AMT_CREDIT"].astype("float64") - df["AMT_GOODS_PRICE"].astype("float64")
            ).astype(np.float32)
        else:
            skipped.append("F2:CREDIT_MINUS_GOODS")

        if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
            df["INCOME_PER_PERSON"] = safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
        else:
            skipped.append("F2:INCOME_PER_PERSON")

        if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(df.columns):
            df["INCOME_PER_CHILD"] = safe_div(df["AMT_INCOME_TOTAL"], df["CNT_CHILDREN"])
        else:
            skipped.append("F2:INCOME_PER_CHILD")

        if {"AMT_ANNUITY", "CNT_FAM_MEMBERS"}.issubset(df.columns):
            df["ANNUITY_PER_PERSON"] = safe_div(df["AMT_ANNUITY"], df["CNT_FAM_MEMBERS"])
        else:
            skipped.append("F2:ANNUITY_PER_PERSON")

        if {"AMT_INCOME_TOTAL", "AMT_ANNUITY"}.issubset(df.columns):
            df["INCOME_MINUS_ANNUITY"] = (
                df["AMT_INCOME_TOTAL"].astype("float64") - df["AMT_ANNUITY"].astype("float64")
            ).astype(np.float32)
        else:
            skipped.append("F2:INCOME_MINUS_ANNUITY")

    return tr, te


def _add_f3(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F3: Age and temporal ratios using positive magnitudes."""
    for df in (tr, te):
        if "DAYS_BIRTH" not in df.columns:
            skipped.append("F3:DAYS_BIRTH_missing")
            continue

        age = _age_years(df)
        df["AGE_YEARS"] = age
        age_days = -df["DAYS_BIRTH"].astype("float64")
        emp_days = -_clean_days_employed(df)
        car_age = _clean_own_car_age(df)

        if "DAYS_EMPLOYED" in df.columns:
            df["EMPLOYED_TO_AGE"] = safe_div(emp_days, age_days)
        else:
            skipped.append("F3:EMPLOYED_TO_AGE")

        if "DAYS_REGISTRATION" in df.columns:
            df["REGISTRATION_TO_AGE"] = safe_div(-df["DAYS_REGISTRATION"].astype("float64"), age_days)
        else:
            skipped.append("F3:REGISTRATION_TO_AGE")

        if "DAYS_ID_PUBLISH" in df.columns:
            df["ID_PUBLISH_TO_AGE"] = safe_div(-df["DAYS_ID_PUBLISH"].astype("float64"), age_days)
        else:
            skipped.append("F3:ID_PUBLISH_TO_AGE")

        if "OWN_CAR_AGE" in df.columns:
            df["CAR_AGE_TO_AGE"] = safe_div(car_age, age)
        else:
            skipped.append("F3:CAR_AGE_TO_AGE")

        if "DAYS_LAST_PHONE_CHANGE" in df.columns:
            phone_days = -df["DAYS_LAST_PHONE_CHANGE"].astype("float64")
            df["PHONE_CHANGE_TO_AGE"] = safe_div(phone_days, age_days)
            df["PHONE_CHANGE_TO_EMPLOYED"] = safe_div(phone_days, emp_days)
        else:
            skipped.append("F3:PHONE_CHANGE")

        if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
            term = _credit_term(df)
            df["AGE_AT_LAST_PAYMENT"] = (
                age.astype("float64") + term.astype("float64")
            ).astype(np.float32)
        else:
            skipped.append("F3:AGE_AT_LAST_PAYMENT")

    return tr, te


def _add_f4(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    cat_cols: list[str],
    use_test_for_stats: bool,
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F4: Peer-relative group differences (25 cols, min count 30) and category frequencies."""
    tr_aux = pd.DataFrame(index=tr.index)
    te_aux = pd.DataFrame(index=te.index)

    for aux, df in ((tr_aux, tr), (te_aux, te)):
        aux["EXT_MEAN"] = _ext_mean(df).astype("float64")
        aux["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].astype("float64")
        aux["CREDIT_TERM"] = _credit_term(df).astype("float64")
        aux["AGE_YEARS"] = _age_years(df).astype("float64")

        cleaned = _clean_cat_placeholders(df)
        for col, alias in (
            ("ORGANIZATION_TYPE", "ORG"),
            ("OCCUPATION_TYPE", "OCC"),
            ("NAME_EDUCATION_TYPE", "EDU"),
            ("NAME_INCOME_TYPE", "INC"),
        ):
            if col in cleaned.columns:
                aux[alias] = cleaned[col].astype("object")
        if {"OCCUPATION_TYPE", "NAME_EDUCATION_TYPE"}.issubset(cleaned.columns):
            occ = cleaned["OCCUPATION_TYPE"].astype("object")
            edu = cleaned["NAME_EDUCATION_TYPE"].astype("object")
            pair = occ.astype(str) + "__" + edu.astype(str)
            pair = pair.where(~(occ.isna() | edu.isna()), np.nan)
            aux["OCC_EDU"] = pair

    ref_aux = (
        pd.concat([tr_aux, te_aux], ignore_index=True) if use_test_for_stats else tr_aux
    )
    ref_raw = pd.concat([tr, te], ignore_index=True) if use_test_for_stats else tr

    val_cols = ["EXT_MEAN", "AMT_INCOME_TOTAL", "CREDIT_TERM", "AGE_YEARS"]
    grp_aliases = [g for g in ("ORG", "OCC", "EDU", "INC", "OCC_EDU") if g in ref_aux.columns]

    for g in grp_aliases:
        counts = ref_aux.groupby(g, dropna=True)["EXT_MEAN"].size()
        valid_groups = counts[counts >= 30].index
        means = ref_aux.groupby(g, dropna=True)[val_cols].mean()
        means = means.loc[means.index.isin(valid_groups)]

        ext_grp_map = means["EXT_MEAN"].to_dict()
        tr[f"GRP_{g}_EXT_MEAN"] = tr_aux[g].map(ext_grp_map).astype(np.float32)
        te[f"GRP_{g}_EXT_MEAN"] = te_aux[g].map(ext_grp_map).astype(np.float32)

        for v in val_cols:
            grp_map = means[v].to_dict()
            tr[f"{v}_MINUS_GRP_{g}"] = (tr_aux[v] - tr_aux[g].map(grp_map)).astype(np.float32)
            te[f"{v}_MINUS_GRP_{g}"] = (te_aux[v] - te_aux[g].map(grp_map)).astype(np.float32)

    for col in cat_cols:
        if col not in ref_raw.columns:
            skipped.append(f"F4:FREQ_{col}")
            continue
        ref_s = ref_raw[col].astype("object").fillna("__NA__").astype(str)
        freq_map = (ref_s.value_counts() / float(len(ref_s))).to_dict()
        tr_s = tr[col].astype("object").fillna("__NA__").astype(str)
        te_s = te[col].astype("object").fillna("__NA__").astype(str)
        tr[f"FREQ_{col}"] = tr_s.map(freq_map).fillna(0.0).astype(np.float32)
        te[f"FREQ_{col}"] = te_s.map(freq_map).fillna(0.0).astype(np.float32)

    return tr, te


def _add_f5(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    use_test_for_stats: bool,
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F5: Duplicated loan-structure density counts."""
    tr_term_r = _credit_term(tr).round(1)
    te_term_r = _credit_term(te).round(1)

    tr_keys = pd.DataFrame(
        {
            "AMT_CREDIT": tr["AMT_CREDIT"],
            "AMT_ANNUITY": tr["AMT_ANNUITY"],
            "AMT_GOODS_PRICE": tr["AMT_GOODS_PRICE"],
            "TERM_R": tr_term_r,
        },
        index=tr.index,
    )
    te_keys = pd.DataFrame(
        {
            "AMT_CREDIT": te["AMT_CREDIT"],
            "AMT_ANNUITY": te["AMT_ANNUITY"],
            "AMT_GOODS_PRICE": te["AMT_GOODS_PRICE"],
            "TERM_R": te_term_r,
        },
        index=te.index,
    )
    ref_keys = (
        pd.concat([tr_keys, te_keys], ignore_index=True) if use_test_for_stats else tr_keys
    )

    ca_counts = ref_keys.groupby(["AMT_CREDIT", "AMT_ANNUITY"], dropna=False).size()
    gp_counts = ref_keys["AMT_GOODS_PRICE"].value_counts(dropna=False)
    tr_counts = ref_keys["TERM_R"].value_counts(dropna=False)

    tr_ca_idx = pd.MultiIndex.from_frame(tr_keys[["AMT_CREDIT", "AMT_ANNUITY"]])
    te_ca_idx = pd.MultiIndex.from_frame(te_keys[["AMT_CREDIT", "AMT_ANNUITY"]])

    tr["DENSITY_CREDIT_ANNUITY"] = (
        tr_ca_idx.map(ca_counts).to_numpy(dtype=np.float64, na_value=0.0).astype(np.float32)
    )
    te["DENSITY_CREDIT_ANNUITY"] = (
        te_ca_idx.map(ca_counts).to_numpy(dtype=np.float64, na_value=0.0).astype(np.float32)
    )

    tr["DENSITY_GOODS_PRICE"] = (
        tr_keys["AMT_GOODS_PRICE"].map(gp_counts).fillna(0.0).astype(np.float32)
    )
    te["DENSITY_GOODS_PRICE"] = (
        te_keys["AMT_GOODS_PRICE"].map(gp_counts).fillna(0.0).astype(np.float32)
    )

    tr["DENSITY_CREDIT_TERM_ROUND"] = (
        tr_keys["TERM_R"].map(tr_counts).fillna(0.0).astype(np.float32)
    )
    te["DENSITY_CREDIT_TERM_ROUND"] = (
        te_keys["TERM_R"].map(tr_counts).fillna(0.0).astype(np.float32)
    )

    return tr, te


def quantize_count(arr: np.ndarray | pd.Series) -> np.ndarray:
    """Coarsen high-frequency counts to avoid exact off-by-1 train-vs-test split separation."""
    # Step cutoffs (<10 exact, 10..99 nearest 5, >=100 nearest 50) selected in P1(c) (experiments_v2/audit_v2.md)
    # to eliminate off-by-1 train-vs-test count separation (engineered adv AUC 0.93734 -> 0.52464).
    s = np.asarray(arr, dtype=np.float64)
    return np.where(
        s < 10.0,
        s,
        np.where(s < 100.0, np.round(s / 5.0) * 5.0, np.round(s / 50.0) * 50.0),
    ).astype(np.float32)


def _add_f5_loo(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    use_test_for_stats: bool,
    skipped: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F5_LOO: Leave-self-out loan-structure density counts with parity quantization."""
    tr_term_r = _credit_term(tr).round(1)
    te_term_r = _credit_term(te).round(1)

    tr_keys = pd.DataFrame(
        {
            "AMT_CREDIT": tr["AMT_CREDIT"],
            "AMT_ANNUITY": tr["AMT_ANNUITY"],
            "AMT_GOODS_PRICE": tr["AMT_GOODS_PRICE"],
            "TERM_R": tr_term_r,
        },
        index=tr.index,
    )
    te_keys = pd.DataFrame(
        {
            "AMT_CREDIT": te["AMT_CREDIT"],
            "AMT_ANNUITY": te["AMT_ANNUITY"],
            "AMT_GOODS_PRICE": te["AMT_GOODS_PRICE"],
            "TERM_R": te_term_r,
        },
        index=te.index,
    )
    ref_keys = (
        pd.concat([tr_keys, te_keys], ignore_index=True) if use_test_for_stats else tr_keys
    )

    ca_counts = ref_keys.groupby(["AMT_CREDIT", "AMT_ANNUITY"], dropna=False).size()
    gp_counts = ref_keys["AMT_GOODS_PRICE"].value_counts(dropna=False)
    tr_counts = ref_keys["TERM_R"].value_counts(dropna=False)

    tr_ca_idx = pd.MultiIndex.from_frame(tr_keys[["AMT_CREDIT", "AMT_ANNUITY"]])
    te_ca_idx = pd.MultiIndex.from_frame(te_keys[["AMT_CREDIT", "AMT_ANNUITY"]])

    tr_ca_raw = tr_ca_idx.map(ca_counts).to_numpy(dtype=np.float64, na_value=0.0) - 1.0
    te_ca_raw = te_ca_idx.map(ca_counts).to_numpy(dtype=np.float64, na_value=0.0)
    if use_test_for_stats:
        te_ca_raw = te_ca_raw - 1.0
    tr["DENSITY_CREDIT_ANNUITY"] = quantize_count(np.clip(tr_ca_raw, 0.0, None))
    te["DENSITY_CREDIT_ANNUITY"] = quantize_count(np.clip(te_ca_raw, 0.0, None))

    tr_gp_raw = tr_keys["AMT_GOODS_PRICE"].map(gp_counts).fillna(0.0).to_numpy(dtype=np.float64) - 1.0
    te_gp_raw = te_keys["AMT_GOODS_PRICE"].map(gp_counts).fillna(0.0).to_numpy(dtype=np.float64)
    if use_test_for_stats:
        te_gp_raw = te_gp_raw - 1.0
    tr["DENSITY_GOODS_PRICE"] = quantize_count(np.clip(tr_gp_raw, 0.0, None))
    te["DENSITY_GOODS_PRICE"] = quantize_count(np.clip(te_gp_raw, 0.0, None))

    tr_tr_raw = tr_keys["TERM_R"].map(tr_counts).fillna(0.0).to_numpy(dtype=np.float64) - 1.0
    te_tr_raw = te_keys["TERM_R"].map(tr_counts).fillna(0.0).to_numpy(dtype=np.float64)
    if use_test_for_stats:
        te_tr_raw = te_tr_raw - 1.0
    tr["DENSITY_CREDIT_TERM_ROUND"] = quantize_count(np.clip(tr_tr_raw, 0.0, None))
    te["DENSITY_CREDIT_TERM_ROUND"] = quantize_count(np.clip(te_tr_raw, 0.0, None))

    return tr, te


def _nearest_term_mode(ct: pd.Series) -> np.ndarray:
    """Map each row's CREDIT_TERM to the nearest dominant peak from P1(d)."""
    vals = ct.to_numpy(dtype=np.float64)
    valid = np.isfinite(vals)
    out = np.full(len(vals), np.nan, dtype=np.float64)
    if valid.any():
        diffs = np.abs(vals[valid, None] - _DOMINANT_TERM_MODES[None, :])
        idx = np.argmin(diffs, axis=1)
        out[valid] = _DOMINANT_TERM_MODES[idx]
    return out


def _product_code_series(df: pd.DataFrame) -> pd.Series:
    """Build PRODUCT_CODE = (AMT_GOODS_PRICE rounded to thousands, TERM_MODE) as string series."""
    goods = df["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    ct = _credit_term(df)
    tmode = _nearest_term_mode(ct)
    # Round goods price to nearest 1,000 currency units (catalog product price bucket per A1 specification).
    goods_k = np.where(np.isfinite(goods), np.round(goods / 1000.0).astype(np.int64), -1)
    t_str = np.where(np.isfinite(tmode), np.round(tmode, 1).astype(str), "NA")
    return pd.Series([f"{g}_{t}" for g, t in zip(goods_k, t_str)], index=df.index, dtype="object")


def _solve_rate_per_period(
    annuity: np.ndarray, principal: np.ndarray, n_periods: np.ndarray
) -> np.ndarray:
    """Vectorised 30-step bisection solving r / (1 - (1+r)^(-n)) = annuity / principal."""
    out = np.full(len(annuity), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(annuity)
        & np.isfinite(principal)
        & np.isfinite(n_periods)
        & (annuity > 0.0)
        & (principal > 0.0)
        & (n_periods > 1.0)
    )
    if not valid.any():
        return out

    target = annuity[valid] / principal[valid]
    n = n_periods[valid]

    def _annuity_factor(r: np.ndarray) -> np.ndarray:
        small = np.abs(r) < 1e-8
        r_safe = np.where(small, 1e-8, r)
        factor = r_safe / (1.0 - np.power(1.0 + r_safe, -n))
        limit = (1.0 / n) + r * (n + 1.0) / (2.0 * n)
        return np.where(small, limit, factor)

    lo = np.full(len(target), -0.15, dtype=np.float64)
    hi = np.full(len(target), 0.50, dtype=np.float64)
    f_lo = _annuity_factor(lo) - target
    f_hi = _annuity_factor(hi) - target
    bracketed = (f_lo <= 0.0) & (f_hi >= 0.0)

    if bracketed.any():
        l_b = lo[bracketed].copy()
        h_b = hi[bracketed].copy()
        t_b = target[bracketed]
        n_b = n[bracketed]

        def _f_b(r: np.ndarray) -> np.ndarray:
            small = np.abs(r) < 1e-8
            r_safe = np.where(small, 1e-8, r)
            factor = r_safe / (1.0 - np.power(1.0 + r_safe, -n_b))
            limit = (1.0 / n_b) + r * (n_b + 1.0) / (2.0 * n_b)
            return np.where(small, limit, factor) - t_b

        for _ in range(30):
            mid = 0.5 * (l_b + h_b)
            f_mid = _f_b(mid)
            go_right = f_mid < 0.0
            l_b = np.where(go_right, mid, l_b)
            h_b = np.where(go_right, h_b, mid)
        sol = np.full(len(target), np.nan, dtype=np.float64)
        sol[bracketed] = 0.5 * (l_b + h_b)
        out[valid] = sol
    return out


def _add_a0_region(
    tr: pd.DataFrame, te: pd.DataFrame, skipped: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A0: Categorical region identifier derived from REGION_POPULATION_RELATIVE."""
    if "REGION_POPULATION_RELATIVE" not in tr.columns:
        skipped.append("A0:REGION_POPULATION_RELATIVE")
        return tr, te
    for df in (tr, te):
        df["REGION_ID"] = df["REGION_POPULATION_RELATIVE"].astype("float64").round(6).astype(str)
    return tr, te


def _add_a1_term_rate(
    tr: pd.DataFrame, te: pd.DataFrame, skipped: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A1: Loan term mode, residuals, implied annuity interest rate, downpayment proxy, product code."""
    req = {"AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}
    if not req.issubset(tr.columns):
        skipped.append("A1:missing_inputs")
        return tr, te

    for df in (tr, te):
        ct = _credit_term(df)
        tmode = _nearest_term_mode(ct)
        cred = df["AMT_CREDIT"].to_numpy(dtype=np.float64)
        ann = df["AMT_ANNUITY"].to_numpy(dtype=np.float64)
        goods = df["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)

        df["TERM_MODE"] = tmode.astype(np.float32)
        df["TERM_RESID_REL"] = (safe_div(ct, tmode) - 1.0).astype(np.float32)
        df["COST_PROXY"] = (safe_div(tmode * ann, cred) - 1.0).astype(np.float32)
        df["RATE_PER_PERIOD"] = _solve_rate_per_period(ann, cred, tmode).astype(np.float32)
        df["DOWNPAY_PROXY"] = safe_div(goods - cred, goods)
        df["GOODS_EQ_CREDIT"] = (df["AMT_GOODS_PRICE"] == df["AMT_CREDIT"]).astype(np.float32)
        df["PRODUCT_CODE"] = _product_code_series(df)
    return tr, te


def _add_a5_div_ext(
    tr: pd.DataFrame, te: pd.DataFrame, skipped: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A5: Amount, age, and employment divided by EXT_SOURCE_3, EXT_SOURCE_2, and EXT_MEAN (18 cols)."""
    if not {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(tr.columns):
        skipped.append("A5:missing_EXT_SOURCE")
        return tr, te
    for df in (tr, te):
        ext_mean = _ext_mean(df)
        nums = {
            "AMT_CREDIT": df["AMT_CREDIT"],
            "AMT_ANNUITY": df["AMT_ANNUITY"],
            "AMT_INCOME_TOTAL": df["AMT_INCOME_TOTAL"],
            "AMT_GOODS_PRICE": df["AMT_GOODS_PRICE"],
            "AGE_YEARS": _age_years(df),
            "EMPLOYED_MAG": -_clean_days_employed(df),
        }
        dens = {
            "EXT3": df["EXT_SOURCE_3"],
            "EXT2": df["EXT_SOURCE_2"],
            "EXTMEAN": ext_mean,
        }
        for xn, xval in nums.items():
            for en, evals in dens.items():
                df[f"DIV_{xn}_BY_{en}"] = safe_div(xval, evals)
    return tr, te


def _add_a6_age_int(
    tr: pd.DataFrame, te: pd.DataFrame, skipped: list[str], as_cat: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A6: Integer age floor(-DAYS_BIRTH / 365) as numeric or categorical."""
    if "DAYS_BIRTH" not in tr.columns:
        skipped.append("A6:DAYS_BIRTH")
        return tr, te
    for df in (tr, te):
        age_i = np.floor(-df["DAYS_BIRTH"].astype("float64") / 365.0).astype(np.int32)
        if as_cat:
            df["AGE_CAT"] = pd.Series(age_i.astype(str), index=df.index, dtype="object")
        else:
            df["AGE_INT"] = age_i.astype(np.float32)
    return tr, te


def _add_f4_lite(
    tr: pd.DataFrame, te: pd.DataFrame, skipped: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """F4_LITE: Capped 12-column LOO peer-relative differences (no FREQ_* counts) for VIEW_B."""
    tr_aux = pd.DataFrame(index=tr.index)
    te_aux = pd.DataFrame(index=te.index)
    for aux, df in ((tr_aux, tr), (te_aux, te)):
        aux["EXT_MEAN"] = _ext_mean(df).astype("float64")
        aux["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].astype("float64")
        aux["CREDIT_TERM"] = _credit_term(df).astype("float64")
        aux["AGE_YEARS"] = _age_years(df).astype("float64")
        cleaned = _clean_cat_placeholders(df)
        aux["ORG"] = cleaned["ORGANIZATION_TYPE"].astype("object")
        aux["OCC"] = cleaned["OCCUPATION_TYPE"].astype("object")
        occ = aux["OCC"]
        edu = cleaned["NAME_EDUCATION_TYPE"].astype("object")
        pair = occ.astype(str) + "__" + edu.astype(str)
        aux["OCC_EDU"] = pair.where(~(occ.isna() | edu.isna()), np.nan)

    specs = [
        ("ORG", ["EXT_MEAN", "AMT_INCOME_TOTAL", "CREDIT_TERM", "AGE_YEARS"], True),
        ("OCC", ["EXT_MEAN", "AMT_INCOME_TOTAL", "CREDIT_TERM", "AGE_YEARS"], True),
        ("OCC_EDU", ["EXT_MEAN"], True),
    ]
    for g, vals, emit_grp_ext in specs:
        for v in vals:
            s = tr_aux[v]
            valid_v = ~s.isna()
            df_g = pd.DataFrame({"g": tr_aux[g], "v": s}).loc[valid_v]
            agg = df_g.groupby("g", dropna=True)["v"].agg(["count", "sum"])
            agg = agg[agg["count"] >= 30]
            cnt_map = agg["count"].to_dict()
            sum_map = agg["sum"].to_dict()
            full_mean_map = (agg["sum"] / agg["count"]).to_dict()

            tr_cnt = tr_aux[g].map(cnt_map).to_numpy(dtype=np.float64)
            tr_sum = tr_aux[g].map(sum_map).to_numpy(dtype=np.float64)
            tr_val = tr_aux[v].to_numpy(dtype=np.float64)
            # Leave-self-out mean on train rows where val is finite
            has_val = np.isfinite(tr_val)
            eff_cnt = np.where(has_val, tr_cnt - 1.0, tr_cnt)
            eff_sum = np.where(has_val, tr_sum - np.nan_to_num(tr_val, nan=0.0), tr_sum)
            tr_grp_mean = np.where(eff_cnt >= 29.0, eff_sum / np.where(eff_cnt >= 29.0, eff_cnt, 1.0), np.nan)
            te_grp_mean = te_aux[g].map(full_mean_map).to_numpy(dtype=np.float64)

            if emit_grp_ext and v == "EXT_MEAN":
                tr[f"GRP_{g}_EXT_MEAN"] = tr_grp_mean.astype(np.float32)
                te[f"GRP_{g}_EXT_MEAN"] = te_grp_mean.astype(np.float32)
            tr[f"{v}_MINUS_GRP_{g}"] = (tr_val - tr_grp_mean).astype(np.float32)
            te[f"{v}_MINUS_GRP_{g}"] = (te_aux[v].to_numpy(dtype=np.float64) - te_grp_mean).astype(np.float32)
    return tr, te


def build_a7_ratio_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Build all pairwise ratios among 12 key numeric columns for A7 discovery screening."""
    base = pd.DataFrame(
        {
            "EXT_SOURCE_1": df["EXT_SOURCE_1"].astype("float64"),
            "EXT_SOURCE_2": df["EXT_SOURCE_2"].astype("float64"),
            "EXT_SOURCE_3": df["EXT_SOURCE_3"].astype("float64"),
            "AMT_CREDIT": df["AMT_CREDIT"].astype("float64"),
            "AMT_ANNUITY": df["AMT_ANNUITY"].astype("float64"),
            "AMT_GOODS_PRICE": df["AMT_GOODS_PRICE"].astype("float64"),
            "AMT_INCOME_TOTAL": df["AMT_INCOME_TOTAL"].astype("float64"),
            "CREDIT_TERM": _credit_term(df).astype("float64"),
            "AGE_YEARS": _age_years(df).astype("float64"),
            "EMPLOYED_MAG": (-_clean_days_employed(df)).astype("float64"),
            "ID_PUBLISH_MAG": (-df["DAYS_ID_PUBLISH"].astype("float64")),
            "REGISTRATION_MAG": (-df["DAYS_REGISTRATION"].astype("float64")),
        },
        index=df.index,
    )
    cols = list(base.columns)
    out: dict[str, pd.Series] = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            out[f"A7_RATIO_{c1}_DIV_{c2}"] = safe_div(base[c1], base[c2])
    return pd.DataFrame(out, index=df.index)


def _add_a7_selected(
    tr: pd.DataFrame, te: pd.DataFrame, skipped: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A7: Add frozen screened pairwise ratios from experiments_v2/a7_selected_15.json."""
    sel_path = Path(__file__).resolve().parent.parent / "experiments_v2" / "a7_selected_15.json"
    if not sel_path.exists():
        skipped.append("A7:missing_a7_selected_15.json")
        return tr, te
    selected = json.loads(sel_path.read_text(encoding="utf-8")).get("selected", [])
    tr_cand = build_a7_ratio_candidates(tr)
    te_cand = build_a7_ratio_candidates(te)
    for col in selected:
        if col in tr_cand.columns:
            tr[col] = tr_cand[col]
            te[col] = te_cand[col]
        else:
            skipped.append(f"A7:missing_{col}")
    return tr, te


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    families: Iterable[str],
    use_test_for_stats: bool = False,
    dropped_features: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic train/test feature matrices without ever receiving y."""
    fam_set = set(families)
    raw_cols = [c for c in test.columns if c != "SK_ID_CURR"]
    cat_cols = [c for c in raw_cols if not pd.api.types.is_numeric_dtype(train[c])]

    tr = train[raw_cols].copy()
    te = test[raw_cols].copy()
    skipped: list[str] = []

    if "F0" in fam_set:
        tr, te = _add_f0(tr, te, raw_cols, skipped)
    if "F1" in fam_set:
        tr, te = _add_f1(tr, te, skipped)
    if "F2" in fam_set:
        tr, te = _add_f2(tr, te, skipped)
    if "F3" in fam_set:
        tr, te = _add_f3(tr, te, skipped)
    if "F4" in fam_set:
        tr, te = _add_f4(tr, te, cat_cols, use_test_for_stats, skipped)
    if "F4_LITE" in fam_set:
        tr, te = _add_f4_lite(tr, te, skipped)
    if "F5_LOO" in fam_set:
        tr, te = _add_f5_loo(tr, te, use_test_for_stats, skipped)
    elif "F5" in fam_set:
        tr, te = _add_f5(tr, te, use_test_for_stats, skipped)
    if "A0" in fam_set:
        tr, te = _add_a0_region(tr, te, skipped)
    if "A1" in fam_set:
        tr, te = _add_a1_term_rate(tr, te, skipped)
    if "A5" in fam_set:
        tr, te = _add_a5_div_ext(tr, te, skipped)
    if "A6_INT" in fam_set:
        tr, te = _add_a6_age_int(tr, te, skipped, as_cat=False)
    if "A6_CAT" in fam_set:
        tr, te = _add_a6_age_int(tr, te, skipped, as_cat=True)
    if "A7" in fam_set:
        tr, te = _add_a7_selected(tr, te, skipped)

    if dropped_features:
        drop_cols = [c for c in dropped_features if c in tr.columns]
        if drop_cols:
            tr = tr.drop(columns=drop_cols)
            te = te.drop(columns=drop_cols)

    # Ensure zero infinities in any numeric column
    for col in tr.columns:
        if pd.api.types.is_numeric_dtype(tr[col]):
            tr[col] = tr[col].replace([np.inf, -np.inf], np.nan)
            te[col] = te[col].replace([np.inf, -np.inf], np.nan)

    tr, te = align_categories(tr, te)
    build_features.last_skipped_ = skipped  # type: ignore[attr-defined]
    return tr, te


def _get_te_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Resolve an existing column or construct virtual A0/A4 interaction keys cleanly."""
    if col in df.columns:
        return df[col].astype("object").fillna("__NA__").astype(str)
    if col == "REGION_ID" and "REGION_POPULATION_RELATIVE" in df.columns:
        return df["REGION_POPULATION_RELATIVE"].astype("float64").round(6).astype(str)
    if col == "OCC_X_ORG" and {"OCCUPATION_TYPE", "ORGANIZATION_TYPE"}.issubset(df.columns):
        a = df["OCCUPATION_TYPE"].astype("object").fillna("__NA__").astype(str)
        b = df["ORGANIZATION_TYPE"].astype("object").fillna("__NA__").astype(str)
        return a + "__" + b
    if col == "EDU_X_INC" and {"NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"}.issubset(df.columns):
        a = df["NAME_EDUCATION_TYPE"].astype("object").fillna("__NA__").astype(str)
        b = df["NAME_INCOME_TYPE"].astype("object").fillna("__NA__").astype(str)
        return a + "__" + b
    if col == "CON_X_GEN_X_FAM" and {
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_FAMILY_STATUS",
    }.issubset(df.columns):
        a = df["NAME_CONTRACT_TYPE"].astype("object").fillna("__NA__").astype(str)
        b = df["CODE_GENDER"].astype("object").fillna("__NA__").astype(str)
        c = df["NAME_FAMILY_STATUS"].astype("object").fillna("__NA__").astype(str)
        return a + "__" + b + "__" + c
    if col == "PRODUCT_CODE" and {"AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        return _product_code_series(df).astype("object").fillna("__NA__").astype(str)
    if col == "TERM_X_CON" and {"AMT_CREDIT", "AMT_ANNUITY", "NAME_CONTRACT_TYPE"}.issubset(df.columns):
        tmode = (
            df["TERM_MODE"].to_numpy(dtype=np.float64)
            if "TERM_MODE" in df.columns
            else _nearest_term_mode(_credit_term(df))
        )
        t_str = pd.Series(
            np.where(np.isfinite(tmode), np.round(tmode, 1).astype(str), "NA"),
            index=df.index,
        )
        con = df["NAME_CONTRACT_TYPE"].astype("object").fillna("__NA__").astype(str)
        return t_str + "__" + con
    return None


def compute_nested_target_encoding(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    cols: tuple[str, ...] = ("ORGANIZATION_TYPE", "OCCUPATION_TYPE"),
    alpha: float = 50.0,  # Pre-registered smoothing prior weight (F6/A4 specification)
    min_count: int = 0,  # 0 for raw categoricals (F6); 30 for interaction combos (VIEW_C_TE_MIN_COUNT)
    n_inner_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute leak-free inner-OOF smoothed target encodings for training and full-train encodings for val/test."""
    tr_out = Xtr.copy()
    va_out = Xva.copy()
    te_out = Xte.copy()

    # Deterministic KFold (independent of ytr) ensures changing one row's label never changes its own encoding
    kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    y_arr = np.asarray(ytr, dtype=np.float64)
    global_mean = float(y_arr.mean())

    for col in cols:
        tr_raw_s = _get_te_series(Xtr, col)
        if tr_raw_s is None:
            continue
        va_s = _get_te_series(Xva, col)
        te_s = _get_te_series(Xte, col)
        assert va_s is not None and te_s is not None

        feat_name = f"TE_{col}"
        tr_s = tr_raw_s.reset_index(drop=True)

        oof_vals = np.full(len(Xtr), global_mean, dtype=np.float32)
        for in_tr_idx, in_va_idx in kf.split(np.arange(len(Xtr))):
            in_y = y_arr[in_tr_idx]
            in_global = float(in_y.mean())
            df_in = pd.DataFrame({"cat": tr_s.iloc[in_tr_idx].to_numpy(), "y": in_y})
            stats = df_in.groupby("cat")["y"].agg(["count", "sum"])
            if min_count > 0:
                stats = stats[stats["count"] >= min_count]
            enc_map = ((stats["sum"] + alpha * in_global) / (stats["count"] + alpha)).to_dict()
            oof_vals[in_va_idx] = (
                tr_s.iloc[in_va_idx].map(enc_map).fillna(in_global).to_numpy(dtype=np.float32)
            )

        full_df = pd.DataFrame({"cat": tr_s.to_numpy(), "y": y_arr})
        full_stats = full_df.groupby("cat")["y"].agg(["count", "sum"])
        if min_count > 0:
            full_stats = full_stats[full_stats["count"] >= min_count]
        full_map = (
            (full_stats["sum"] + alpha * global_mean) / (full_stats["count"] + alpha)
        ).to_dict()

        tr_out[feat_name] = oof_vals
        va_out[feat_name] = va_s.map(full_map).fillna(global_mean).to_numpy(dtype=np.float32)
        te_out[feat_name] = te_s.map(full_map).fillna(global_mean).to_numpy(dtype=np.float32)

    return tr_out, va_out, te_out


def wrap_fit_predict_with_te(
    base_fit_predict: Callable[
        [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
        tuple[np.ndarray, np.ndarray, int],
    ],
    cols: tuple[str, ...] = ("ORGANIZATION_TYPE", "OCCUPATION_TYPE"),
    alpha: float = 50.0,
    min_count: int = 0,
    n_inner_splits: int = 5,
    seed: int = 42,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Wrap any model fit_predict to inject nested fold-isolated target encodings (F6/A0/A4)."""

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tr_te, va_te, te_te = compute_nested_target_encoding(
            Xtr,
            ytr,
            Xva,
            Xte,
            cols=cols,
            alpha=alpha,
            min_count=min_count,
            n_inner_splits=n_inner_splits,
            seed=seed,
        )
        val_pred, test_pred, best_iter = base_fit_predict(tr_te, ytr, va_te, yva, te_te)
        imp = getattr(base_fit_predict, "last_feature_importances_", None)
        if imp is not None:
            fit_predict.last_feature_importances_ = imp  # type: ignore[attr-defined]
            fit_predict.last_feature_names_ = list(tr_te.columns)  # type: ignore[attr-defined]
        return val_pred, test_pred, best_iter

    return fit_predict


def _knn_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract float64 [EXT_1, EXT_2, EXT_3, CREDIT_TERM] matrix and 3-bit EXT missingness pattern."""
    ct = df["CREDIT_TERM"] if "CREDIT_TERM" in df.columns else _credit_term(df)
    mat = np.column_stack(
        [
            df["EXT_SOURCE_1"].to_numpy(dtype=np.float64),
            df["EXT_SOURCE_2"].to_numpy(dtype=np.float64),
            df["EXT_SOURCE_3"].to_numpy(dtype=np.float64),
            ct.to_numpy(dtype=np.float64),
        ]
    )
    pat = (
        np.isnan(mat[:, 0]).astype(np.int32) * 4
        + np.isnan(mat[:, 1]).astype(np.int32) * 2
        + np.isnan(mat[:, 2]).astype(np.int32)
    )
    return mat, pat


def compute_nested_knn_target_mean(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    k_values: tuple[int, ...] = (500,),
    mode: str = "v1",
    min_pattern_rows: int = 3000,  # Fallback threshold: train patterns 0,1,4,5 have 12.8k-64.2k rows/fold vs 23-132 for 2,3,6,7
    chunk_size: int = 5000,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute nested fold-isolated KNN target means in Quantile-Normal [EXT_1..3, CREDIT_TERM] space."""
    import hashlib

    tr_out = Xtr.copy()
    va_out = Xva.copy()
    te_out = Xte.copy()

    y_arr = np.asarray(ytr, dtype=np.float64)
    raw_tr, pat_tr = _knn_feature_matrix(Xtr)
    raw_va, pat_va = _knn_feature_matrix(Xva)
    raw_te, pat_te = _knn_feature_matrix(Xte)

    cache_file: Path | None = None
    if cache_dir is not None:
        h = hashlib.sha256()
        h.update(f"{k_values}|{mode}|{min_pattern_rows}|{seed}".encode("utf-8"))
        h.update(np.ascontiguousarray(raw_tr).tobytes())
        h.update(np.ascontiguousarray(raw_va).tobytes())
        h.update(np.ascontiguousarray(raw_te).tobytes())
        h.update(np.ascontiguousarray(y_arr).tobytes())
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"knn_{h.hexdigest()[:24]}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            for k in k_values:
                col_name = f"KNN_EXT_TERM_{k}_MEAN"
                tr_out[col_name] = data[f"tr_{k}"]
                va_out[col_name] = data[f"va_{k}"]
                te_out[col_name] = data[f"te_{k}"]
            return tr_out, va_out, te_out

    # Fit per-column QuantileTransformer strictly on non-missing training values; NaN -> 0.0 (normal median)
    Ztr = np.zeros_like(raw_tr, dtype=np.float64)
    Zva = np.zeros_like(raw_va, dtype=np.float64)
    Zte = np.zeros_like(raw_te, dtype=np.float64)
    for j in range(4):
        obs_tr = np.isfinite(raw_tr[:, j])
        if obs_tr.sum() >= 2:
            n_q = min(1000, int(obs_tr.sum()))
            qt = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution="normal",
                random_state=seed + j,
                subsample=50000,
            )
            qt.fit(raw_tr[obs_tr, j].reshape(-1, 1))
            Ztr[obs_tr, j] = qt.transform(raw_tr[obs_tr, j].reshape(-1, 1)).ravel()
            obs_va = np.isfinite(raw_va[:, j])
            if obs_va.any():
                Zva[obs_va, j] = qt.transform(raw_va[obs_va, j].reshape(-1, 1)).ravel()
            obs_te = np.isfinite(raw_te[:, j])
            if obs_te.any():
                Zte[obs_te, j] = qt.transform(raw_te[obs_te, j].reshape(-1, 1)).ravel()

    max_k = max(k_values)
    eff_max_k = min(max_k, max(1, len(Xtr) - 1))
    tr_res = {k: np.empty(len(Xtr), dtype=np.float32) for k in k_values}
    va_res = {k: np.empty(len(Xva), dtype=np.float32) for k in k_values}
    te_res = {k: np.empty(len(Xte), dtype=np.float32) for k in k_values}

    def _query_loo(nn_obj: NearestNeighbors, q_mat: np.ndarray, q_local_idx: np.ndarray, pool_y: np.ndarray, cap_k: int) -> dict[int, np.ndarray]:
        out_d = {k: np.empty(len(q_mat), dtype=np.float32) for k in k_values}
        for s in range(0, len(q_mat), chunk_size):
            e = min(s + chunk_size, len(q_mat))
            ind = nn_obj.kneighbors(q_mat[s:e], n_neighbors=cap_k + 1, return_distance=False)
            q_col = q_local_idx[s:e, None]
            mask = ind != q_col
            has_self = (~mask).any(axis=1)
            mask[~has_self, -1] = False
            neigh = ind[mask].reshape(e - s, cap_k)
            for k in k_values:
                kk = min(k, cap_k)
                out_d[k][s:e] = pool_y[neigh[:, :kk]].mean(axis=1).astype(np.float32)
        return out_d

    def _query_eval(nn_obj: NearestNeighbors, q_mat: np.ndarray, pool_y: np.ndarray, cap_k: int) -> dict[int, np.ndarray]:
        out_d = {k: np.empty(len(q_mat), dtype=np.float32) for k in k_values}
        for s in range(0, len(q_mat), chunk_size):
            e = min(s + chunk_size, len(q_mat))
            ind = nn_obj.kneighbors(q_mat[s:e], n_neighbors=cap_k, return_distance=False)
            for k in k_values:
                kk = min(k, cap_k)
                out_d[k][s:e] = pool_y[ind[:, :kk]].mean(axis=1).astype(np.float32)
        return out_d

    nn_global = NearestNeighbors(n_neighbors=eff_max_k + 1, algorithm="kd_tree", leaf_size=40, n_jobs=1)
    nn_global.fit(Ztr)

    if mode == "v1":
        tr_d = _query_loo(nn_global, Ztr, np.arange(len(Ztr)), y_arr, eff_max_k)
        va_d = _query_eval(nn_global, Zva, y_arr, eff_max_k)
        te_d = _query_eval(nn_global, Zte, y_arr, eff_max_k)
        for k in k_values:
            tr_res[k][:] = tr_d[k]
            va_res[k][:] = va_d[k]
            te_res[k][:] = te_d[k]
    else:
        # V2: per-missingness-pattern pools when pattern count >= min_pattern_rows; fallback to V1 otherwise
        fallback_pats: list[int] = []
        for p in range(8):
            tr_idx = np.where(pat_tr == p)[0]
            va_idx = np.where(pat_va == p)[0]
            te_idx = np.where(pat_te == p)[0]
            if len(tr_idx) == 0 and len(va_idx) == 0 and len(te_idx) == 0:
                continue
            if len(tr_idx) < min_pattern_rows or len(tr_idx) <= eff_max_k:
                fallback_pats.append(p)
                continue

            active_dims = [j for j in (0, 1, 2) if ((p >> (2 - j)) & 1) == 0] + [3]
            nn_p = NearestNeighbors(
                n_neighbors=eff_max_k + 1, algorithm="kd_tree", leaf_size=40, n_jobs=1
            )
            nn_p.fit(Ztr[tr_idx][:, active_dims])
            y_p = y_arr[tr_idx]

            tr_d = _query_loo(nn_p, Ztr[tr_idx][:, active_dims], np.arange(len(tr_idx)), y_p, eff_max_k)
            for k in k_values:
                tr_res[k][tr_idx] = tr_d[k]
            if len(va_idx) > 0:
                va_d = _query_eval(nn_p, Zva[va_idx][:, active_dims], y_p, eff_max_k)
                for k in k_values:
                    va_res[k][va_idx] = va_d[k]
            if len(te_idx) > 0:
                te_d = _query_eval(nn_p, Zte[te_idx][:, active_dims], y_p, eff_max_k)
                for k in k_values:
                    te_res[k][te_idx] = te_d[k]

        if fallback_pats:
            fb_tr = np.where(np.isin(pat_tr, fallback_pats))[0]
            fb_va = np.where(np.isin(pat_va, fallback_pats))[0]
            fb_te = np.where(np.isin(pat_te, fallback_pats))[0]
            if len(fb_tr) > 0:
                tr_d = _query_loo(nn_global, Ztr[fb_tr], fb_tr, y_arr, eff_max_k)
                for k in k_values:
                    tr_res[k][fb_tr] = tr_d[k]
            if len(fb_va) > 0:
                va_d = _query_eval(nn_global, Zva[fb_va], y_arr, eff_max_k)
                for k in k_values:
                    va_res[k][fb_va] = va_d[k]
            if len(fb_te) > 0:
                te_d = _query_eval(nn_global, Zte[fb_te], y_arr, eff_max_k)
                for k in k_values:
                    te_res[k][fb_te] = te_d[k]

    save_dict: dict[str, np.ndarray] = {}
    for k in k_values:
        col_name = f"KNN_EXT_TERM_{k}_MEAN"
        tr_out[col_name] = tr_res[k]
        va_out[col_name] = va_res[k]
        te_out[col_name] = te_res[k]
        if cache_file is not None:
            save_dict[f"tr_{k}"] = tr_res[k]
            save_dict[f"va_{k}"] = va_res[k]
            save_dict[f"te_{k}"] = te_res[k]
    if cache_file is not None and save_dict:
        np.savez(cache_file, **save_dict)
    return tr_out, va_out, te_out


def wrap_fit_predict_with_knn(
    base_fit_predict: Callable[
        [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
        tuple[np.ndarray, np.ndarray, int],
    ],
    k_values: tuple[int, ...] = (500,),
    mode: str = "v1",
    min_pattern_rows: int = 3000,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Wrap any model fit_predict to inject nested fold-isolated KNN target mean features (A3)."""

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tr_k, va_k, te_k = compute_nested_knn_target_mean(
            Xtr,
            ytr,
            Xva,
            Xte,
            k_values=k_values,
            mode=mode,
            min_pattern_rows=min_pattern_rows,
            seed=seed,
            cache_dir=cache_dir,
        )
        val_pred, test_pred, best_iter = base_fit_predict(tr_k, ytr, va_k, yva, te_k)
        imp = getattr(base_fit_predict, "last_feature_importances_", None)
        if imp is not None:
            fit_predict.last_feature_importances_ = imp  # type: ignore[attr-defined]
            fit_predict.last_feature_names_ = list(tr_k.columns)  # type: ignore[attr-defined]
        return val_pred, test_pred, best_iter

    return fit_predict


def _fp_key_from_df(df: pd.DataFrame) -> pd.Series:
    """Construct K3 customer fingerprint key from feature columns present in X."""
    g = df["CODE_GENDER"].astype("object").fillna("__NA__").astype(str)
    reg = df["REGION_POPULATION_RELATIVE"].astype("float64").round(6).astype(str)
    b_id = (
        (df["DAYS_BIRTH"].astype("float64") - df["DAYS_ID_PUBLISH"].astype("float64"))
        .round()
        .astype("Int64")
        .astype(str)
    )
    b_reg = (
        (df["DAYS_BIRTH"].astype("float64") - df["DAYS_REGISTRATION"].astype("float64"))
        .round()
        .astype("Int64")
        .astype(str)
    )
    return g + "|" + reg + "|" + b_id + "|" + b_reg


def compute_nested_linkage_features(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute nested leave-self-out fingerprint linkage features (B1, gated by ENABLE_LINKAGE)."""
    tr_out = Xtr.copy()
    va_out = Xva.copy()
    te_out = Xte.copy()

    y_arr = np.asarray(ytr, dtype=np.float64)
    tr_k = _fp_key_from_df(Xtr)
    va_k = _fp_key_from_df(Xva)
    te_k = _fp_key_from_df(Xte)

    stats = pd.DataFrame({"k": tr_k.to_numpy(), "y": y_arr}).groupby("k")["y"].agg(["count", "sum"])
    cnt_map = stats["count"].to_dict()
    sum_map = stats["sum"].to_dict()

    tr_cnt = tr_k.map(cnt_map).fillna(0.0).to_numpy(dtype=np.float64)
    tr_sum = tr_k.map(sum_map).fillna(0.0).to_numpy(dtype=np.float64)
    loo_cnt = np.clip(tr_cnt - 1.0, 0.0, None)
    loo_sum = tr_sum - y_arr
    tr_out["FP_GROUP_SIZE_LOO"] = loo_cnt.astype(np.float32)
    tr_out["FP_N_PRIOR"] = loo_cnt.astype(np.float32)
    tr_out["FP_PRIOR_DEFAULT_RATE"] = np.where(
        loo_cnt >= 1.0, loo_sum / np.where(loo_cnt >= 1.0, loo_cnt, 1.0), np.nan
    ).astype(np.float32)

    for df_out, keys in ((va_out, va_k), (te_out, te_k)):
        cnt = keys.map(cnt_map).fillna(0.0).to_numpy(dtype=np.float64)
        sm = keys.map(sum_map).fillna(0.0).to_numpy(dtype=np.float64)
        df_out["FP_GROUP_SIZE_LOO"] = cnt.astype(np.float32)
        df_out["FP_N_PRIOR"] = cnt.astype(np.float32)
        df_out["FP_PRIOR_DEFAULT_RATE"] = np.where(
            cnt >= 1.0, sm / np.where(cnt >= 1.0, cnt, 1.0), np.nan
        ).astype(np.float32)

    return tr_out, va_out, te_out


def wrap_fit_predict_with_linkage(
    base_fit_predict: Callable[
        [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
        tuple[np.ndarray, np.ndarray, int],
    ],
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Wrap any model fit_predict to inject nested B1 customer linkage features (ENABLE_LINKAGE)."""

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tr_l, va_l, te_l = compute_nested_linkage_features(Xtr, ytr, Xva, Xte)
        val_pred, test_pred, best_iter = base_fit_predict(tr_l, ytr, va_l, yva, te_l)
        imp = getattr(base_fit_predict, "last_feature_importances_", None)
        if imp is not None:
            fit_predict.last_feature_importances_ = imp  # type: ignore[attr-defined]
            fit_predict.last_feature_names_ = list(tr_l.columns)  # type: ignore[attr-defined]
        return val_pred, test_pred, best_iter

    return fit_predict

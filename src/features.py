"""Deterministic, leak-free feature engineering pipeline for GCI Home Credit."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.cv_harness import align_categories

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
    s = df["DAYS_EMPLOYED"].astype("float64")
    return s.where(s != 365243.0, np.nan)


def _clean_own_car_age(df: pd.DataFrame) -> pd.Series:
    """Private helper: return OWN_CAR_AGE with >= 60 outliers replaced by NaN."""
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
    if "F5" in fam_set:
        tr, te = _add_f5(tr, te, use_test_for_stats, skipped)

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


def compute_nested_target_encoding(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    cols: tuple[str, ...] = ("ORGANIZATION_TYPE", "OCCUPATION_TYPE"),
    alpha: float = 50.0,
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
        if col not in Xtr.columns:
            continue
        feat_name = f"TE_{col}"
        tr_s = Xtr[col].astype("object").fillna("__NA__").astype(str).reset_index(drop=True)
        va_s = Xva[col].astype("object").fillna("__NA__").astype(str)
        te_s = Xte[col].astype("object").fillna("__NA__").astype(str)

        oof_vals = np.full(len(Xtr), global_mean, dtype=np.float32)
        for in_tr_idx, in_va_idx in kf.split(np.arange(len(Xtr))):
            in_y = y_arr[in_tr_idx]
            in_global = float(in_y.mean())
            df_in = pd.DataFrame({"cat": tr_s.iloc[in_tr_idx].to_numpy(), "y": in_y})
            stats = df_in.groupby("cat")["y"].agg(["count", "sum"])
            enc_map = ((stats["sum"] + alpha * in_global) / (stats["count"] + alpha)).to_dict()
            oof_vals[in_va_idx] = (
                tr_s.iloc[in_va_idx].map(enc_map).fillna(in_global).to_numpy(dtype=np.float32)
            )

        full_df = pd.DataFrame({"cat": tr_s.to_numpy(), "y": y_arr})
        full_stats = full_df.groupby("cat")["y"].agg(["count", "sum"])
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
    n_inner_splits: int = 5,
    seed: int = 42,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Wrap any model fit_predict to inject nested fold-isolated target encodings (F6)."""

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

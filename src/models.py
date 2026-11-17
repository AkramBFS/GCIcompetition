"""Model wrappers adhering to the cv_harness fit_predict signature."""

from __future__ import annotations

from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import N_THREADS, SEED
from src.cv_harness import LGB_START


def lgb_model(
    params: dict[str, Any] | None = None,
    seed: int = SEED,
    lr: float | None = None,
    deterministic: bool = False,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Return a LightGBM fit_predict callable matching the CV harness signature."""
    cfg = dict(LGB_START)
    if params:
        cfg.update(params)
    if lr is not None:
        cfg["learning_rate"] = lr
    cfg["random_state"] = seed
    cfg["bagging_seed"] = seed
    cfg["feature_fraction_seed"] = seed
    cfg["data_random_seed"] = seed
    cfg["n_jobs"] = N_THREADS
    cfg["verbose"] = -1
    if deterministic:
        cfg["deterministic"] = True
        cfg["force_row_wise"] = True

    stopping_rounds = int(cfg.pop("early_stopping_rounds", 200))

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        model = lgb.LGBMClassifier(**cfg)
        model.fit(
            Xtr,
            ytr,
            eval_X=Xva,
            eval_y=yva,
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iter = int(model.best_iteration_ or cfg.get("n_estimators", 5000))
        val_pred = model.predict_proba(Xva, num_iteration=best_iter)[:, 1]
        test_pred = model.predict_proba(Xte, num_iteration=best_iter)[:, 1]
        fit_predict.last_feature_importances_ = model.booster_.feature_importance(  # type: ignore[attr-defined]
            importance_type="gain"
        )
        return val_pred, test_pred, best_iter

    return fit_predict


def xgb_model(
    params: dict[str, Any] | None = None,
    seed: int = SEED,
    use_codes: bool = False,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Return an XGBoost fit_predict callable matching the CV harness signature."""
    cfg: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "min_child_weight": 20,
        "reg_lambda": 10.0,
        "n_estimators": 5000,
        "early_stopping_rounds": 200,
        "n_jobs": N_THREADS,
        "random_state": seed,
        "enable_categorical": not use_codes,
    }
    if params:
        cfg.update(params)
    cfg["random_state"] = seed
    cfg["n_jobs"] = N_THREADS
    cfg["enable_categorical"] = not use_codes

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if not use_codes:
            return df
        out = df.copy()
        for col in out.columns:
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                codes = out[col].cat.codes.astype(np.float32)
                out[col] = codes.where(codes >= 0, np.nan)
        return out

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tr_df = _prep(Xtr)
        va_df = _prep(Xva)
        te_df = _prep(Xte)
        model = xgb.XGBClassifier(**cfg)
        model.fit(
            tr_df,
            ytr,
            eval_set=[(va_df, yva)],
            verbose=False,
        )
        best_iter = int(getattr(model, "best_iteration", cfg.get("n_estimators", 5000) - 1)) + 1
        val_pred = model.predict_proba(va_df)[:, 1]
        test_pred = model.predict_proba(te_df)[:, 1]
        fit_predict.last_feature_importances_ = model.feature_importances_  # type: ignore[attr-defined]
        return val_pred, test_pred, best_iter

    return fit_predict


def cat_model(
    params: dict[str, Any] | None = None,
    seed: int = SEED,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Return a CatBoost fit_predict callable matching the CV harness signature."""
    from catboost import CatBoostClassifier, Pool

    cfg: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "iterations": 6000,
        "early_stopping_rounds": 300,
        "thread_count": N_THREADS,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    if params:
        cfg.update(params)
    cfg["random_seed"] = seed
    cfg["thread_count"] = N_THREADS
    cfg["allow_writing_files"] = False

    def _prep_cat(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = df.copy()
        cat_cols: list[str] = []
        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):
                cat_cols.append(col)
                s = out[col].astype("object")
                out[col] = s.where(~s.isna(), "Missing").astype(str)
        return out, cat_cols

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tr_df, cat_cols = _prep_cat(Xtr)
        va_df, _ = _prep_cat(Xva)
        te_df, _ = _prep_cat(Xte)
        tr_pool = Pool(tr_df, ytr, cat_features=cat_cols)
        va_pool = Pool(va_df, yva, cat_features=cat_cols)
        te_pool = Pool(te_df, cat_features=cat_cols)

        model = CatBoostClassifier(**cfg)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        best_iter = int(model.get_best_iteration())
        val_pred = model.predict_proba(va_pool)[:, 1]
        test_pred = model.predict_proba(te_pool)[:, 1]
        fit_predict.last_feature_importances_ = model.get_feature_importance()  # type: ignore[attr-defined]
        return val_pred, test_pred, best_iter

    return fit_predict

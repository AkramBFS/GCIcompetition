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
    use_codes: bool = False,
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

    stopping_rounds = cfg.pop("early_stopping_rounds", 200)

    def _prep_codes(df: pd.DataFrame) -> pd.DataFrame:
        if not use_codes:
            return df
        out = df.copy()
        for col in out.columns:
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].cat.codes.astype(np.float32)
        return out

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        tr_df = _prep_codes(Xtr)
        va_df = _prep_codes(Xva)
        te_df = _prep_codes(Xte)
        model = lgb.LGBMClassifier(**cfg)
        if stopping_rounds is not None and int(stopping_rounds) > 0:
            model.fit(
                tr_df,
                ytr,
                eval_X=va_df,
                eval_y=yva,
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(int(stopping_rounds), verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or cfg.get("n_estimators", 5000))
        else:
            model.fit(tr_df, ytr)
            best_iter = int(cfg.get("n_estimators", 5000))
        val_pred = model.predict_proba(va_df, num_iteration=best_iter)[:, 1]
        test_pred = model.predict_proba(te_df, num_iteration=best_iter)[:, 1]
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


def logreg_binned_model(
    params: dict[str, Any] | None = None,
    seed: int = SEED,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Return a quantile-binned L2 LogisticRegression fit_predict callable (strict train-only fit)."""
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

    cfg: dict[str, Any] = {
        "C": 0.1,
        "n_bins": 30,
        "max_iter": 500,
        "min_frequency": 20,
    }
    if params:
        cfg.update(params)
    c_val = float(cfg["C"])
    n_bins = int(cfg["n_bins"])
    max_iter = int(cfg["max_iter"])
    min_freq = int(cfg["min_frequency"])

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
        cat_cols = [c for c in Xtr.columns if c not in num_cols]

        parts_tr: list[sparse.csr_matrix] = []
        parts_va: list[sparse.csr_matrix] = []
        parts_te: list[sparse.csr_matrix] = []

        if num_cols:
            tr_num = Xtr[num_cols].to_numpy(dtype=np.float64)
            va_num = Xva[num_cols].to_numpy(dtype=np.float64)
            te_num = Xte[num_cols].to_numpy(dtype=np.float64)

            tr_na = np.isnan(tr_num).astype(np.float64)
            va_na = np.isnan(va_num).astype(np.float64)
            te_na = np.isnan(te_num).astype(np.float64)

            meds = np.nanmedian(tr_num, axis=0)
            meds = np.where(np.isfinite(meds), meds, 0.0)
            for j in range(tr_num.shape[1]):
                tr_num[tr_na[:, j] > 0.5, j] = meds[j]
                va_num[va_na[:, j] > 0.5, j] = meds[j]
                te_num[te_na[:, j] > 0.5, j] = meds[j]

            kbd = KBinsDiscretizer(
                n_bins=n_bins,
                encode="onehot",
                strategy="quantile",
                subsample=min(50000, len(Xtr)),
                random_state=seed,
            )
            parts_tr.append(sparse.csr_matrix(kbd.fit_transform(tr_num)))
            parts_va.append(sparse.csr_matrix(kbd.transform(va_num)))
            parts_te.append(sparse.csr_matrix(kbd.transform(te_num)))

            parts_tr.append(sparse.csr_matrix(tr_na))
            parts_va.append(sparse.csr_matrix(va_na))
            parts_te.append(sparse.csr_matrix(te_na))

        if cat_cols:
            def _cat_arr(df: pd.DataFrame) -> np.ndarray:
                arr = np.empty((len(df), len(cat_cols)), dtype=object)
                for j, c in enumerate(cat_cols):
                    arr[:, j] = df[c].astype("object").fillna("__NA__").astype(str).to_numpy()
                return arr

            ohe = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min(min_freq, max(1, len(Xtr) // 10)),
                sparse_output=True,
            )
            parts_tr.append(sparse.csr_matrix(ohe.fit_transform(_cat_arr(Xtr))))
            parts_va.append(sparse.csr_matrix(ohe.transform(_cat_arr(Xva))))
            parts_te.append(sparse.csr_matrix(ohe.transform(_cat_arr(Xte))))

        X_tr_mat = sparse.hstack(parts_tr, format="csr")
        X_va_mat = sparse.hstack(parts_va, format="csr")
        X_te_mat = sparse.hstack(parts_te, format="csr")

        clf = LogisticRegression(C=c_val, max_iter=max_iter, solver="lbfgs", random_state=seed)
        clf.fit(X_tr_mat, ytr)
        val_pred = clf.predict_proba(X_va_mat)[:, 1]
        test_pred = clf.predict_proba(X_te_mat)[:, 1]
        best_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else max_iter
        return val_pred, test_pred, best_iter

    return fit_predict


def mlp_model(
    params: dict[str, Any] | None = None,
    seed: int = SEED,
) -> Callable[
    [pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame],
    tuple[np.ndarray, np.ndarray, int],
]:
    """Return a deterministic PyTorch CPU (256, 128) MLP fit_predict callable."""
    import time

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import QuantileTransformer
    from torch.utils.data import DataLoader, TensorDataset

    cfg: dict[str, Any] = {
        "epochs": 12,
        "patience": 3,
        "batch_size": 1024,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "max_seconds_per_fold": 360.0,
    }
    if params:
        cfg.update(params)

    class _TabularMLP(nn.Module):
        def __init__(self, n_num: int, cat_dims: list[tuple[int, int]], dropout: float) -> None:
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(num_cat, emb_dim) for num_cat, emb_dim in cat_dims])
            emb_total = sum(d for _, d in cat_dims)
            in_dim = n_num + emb_total
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            if len(self.embs) > 0:
                emb_outs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embs)]
                x = torch.cat([x_num] + emb_outs, dim=1)
            else:
                x = x_num
            return self.net(x).squeeze(1)

    def fit_predict(
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xva: pd.DataFrame,
        yva: np.ndarray,
        Xte: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        t0 = time.monotonic()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(N_THREADS)

        num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
        cat_cols = [c for c in Xtr.columns if c not in num_cols]

        # Numeric preprocessing strictly fitted on Xtr
        tr_num_raw = Xtr[num_cols].to_numpy(dtype=np.float64)
        va_num_raw = Xva[num_cols].to_numpy(dtype=np.float64)
        te_num_raw = Xte[num_cols].to_numpy(dtype=np.float64)

        tr_na = np.isnan(tr_num_raw).astype(np.float32)
        va_na = np.isnan(va_num_raw).astype(np.float32)
        te_na = np.isnan(te_num_raw).astype(np.float32)

        tr_scaled = np.zeros_like(tr_num_raw, dtype=np.float32)
        va_scaled = np.zeros_like(va_num_raw, dtype=np.float32)
        te_scaled = np.zeros_like(te_num_raw, dtype=np.float32)

        for j in range(len(num_cols)):
            obs_tr = np.isfinite(tr_num_raw[:, j])
            if obs_tr.sum() >= 2:
                qt = QuantileTransformer(
                    n_quantiles=min(1000, int(obs_tr.sum())),
                    output_distribution="normal",
                    random_state=seed + j,
                    subsample=50000,
                )
                qt.fit(tr_num_raw[obs_tr, j].reshape(-1, 1))
                tr_scaled[obs_tr, j] = qt.transform(tr_num_raw[obs_tr, j].reshape(-1, 1)).ravel().astype(np.float32)
                obs_va = np.isfinite(va_num_raw[:, j])
                if obs_va.any():
                    va_scaled[obs_va, j] = qt.transform(va_num_raw[obs_va, j].reshape(-1, 1)).ravel().astype(np.float32)
                obs_te = np.isfinite(te_num_raw[:, j])
                if obs_te.any():
                    te_scaled[obs_te, j] = qt.transform(te_num_raw[obs_te, j].reshape(-1, 1)).ravel().astype(np.float32)

        Xtr_num = np.hstack([tr_scaled, tr_na]).astype(np.float32)
        Xva_num = np.hstack([va_scaled, va_na]).astype(np.float32)
        Xte_num = np.hstack([te_scaled, te_na]).astype(np.float32)

        # Categorical integer indexing strictly fitted on Xtr (0 = unknown/missing)
        cat_dims: list[tuple[int, int]] = []
        tr_cat_idx = np.zeros((len(Xtr), len(cat_cols)), dtype=np.int64)
        va_cat_idx = np.zeros((len(Xva), len(cat_cols)), dtype=np.int64)
        te_cat_idx = np.zeros((len(Xte), len(cat_cols)), dtype=np.int64)
        for j, col in enumerate(cat_cols):
            tr_s = Xtr[col].astype("object").fillna("__NA__").astype(str)
            uniq = sorted(tr_s.unique().tolist())
            mapping = {v: i + 1 for i, v in enumerate(uniq)}
            tr_cat_idx[:, j] = tr_s.map(mapping).fillna(0).to_numpy(dtype=np.int64)
            va_cat_idx[:, j] = Xva[col].astype("object").fillna("__NA__").astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
            te_cat_idx[:, j] = Xte[col].astype("object").fillna("__NA__").astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
            n_cat = len(uniq) + 1
            emb_dim = min(16, max(2, (n_cat + 1) // 2))
            cat_dims.append((n_cat, emb_dim))

        tr_ds = TensorDataset(
            torch.from_numpy(Xtr_num),
            torch.from_numpy(tr_cat_idx),
            torch.from_numpy(np.asarray(ytr, dtype=np.float32)),
        )
        g = torch.Generator().manual_seed(seed)
        tr_loader = DataLoader(tr_ds, batch_size=int(cfg["batch_size"]), shuffle=True, generator=g)

        model = _TabularMLP(Xtr_num.shape[1], cat_dims, float(cfg["dropout"]))
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        loss_fn = nn.BCEWithLogitsLoss()

        xva_n_t = torch.from_numpy(Xva_num)
        xva_c_t = torch.from_numpy(va_cat_idx)
        xte_n_t = torch.from_numpy(Xte_num)
        xte_c_t = torch.from_numpy(te_cat_idx)

        def _predict_batches(xn: torch.Tensor, xc: torch.Tensor) -> np.ndarray:
            model.eval()
            preds: list[np.ndarray] = []
            with torch.no_grad():
                for s in range(0, len(xn), 4096):
                    e = min(s + 4096, len(xn))
                    logits = model(xn[s:e], xc[s:e])
                    preds.append(torch.sigmoid(logits).cpu().numpy())
            return np.concatenate(preds)

        best_auc = -1.0
        best_epoch = 1
        best_state: dict[str, torch.Tensor] | None = None
        no_imp = 0

        for epoch in range(1, int(cfg["epochs"]) + 1):
            model.train()
            for bx_n, bx_c, by in tr_loader:
                if len(bx_n) < 2:
                    continue
                opt.zero_grad()
                logits = model(bx_n, bx_c)
                loss = loss_fn(logits, by)
                loss.backward()
                opt.step()

            vp = _predict_batches(xva_n_t, xva_c_t)
            auc = float(roc_auc_score(yva, vp))
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= int(cfg["patience"]) or (time.monotonic() - t0) >= float(cfg["max_seconds_per_fold"]):
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_pred = _predict_batches(xva_n_t, xva_c_t).astype(np.float64)
        test_pred = _predict_batches(xte_n_t, xte_c_t).astype(np.float64)
        return val_pred, test_pred, best_epoch

    return fit_predict

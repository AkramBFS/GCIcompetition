# GCI Global 2026 — Home Credit Default Risk Solution

End-to-end, leak-free, reproducible tabular machine learning pipeline for the **GCI Global 2026 Home Credit Default Risk** competition.

- **Metric:** ROC-AUC (`sklearn.metrics.roc_auc_score`)
- **Dataset:** `input/train.csv` (`171,202` rows x `34` columns, `8.0729%` positive `TARGET`), `input/test.csv` (`61,500` rows x `33` columns)
- **Final Multi-Seed Rank Blend CV:** **`0.765206`** mean 5-fold AUC | **`0.765177`** pooled OOF AUC (vs tutorial LightGBM `0.7443` / tutorial 4-model blend `0.7531` on 70/30 holdout)

---

## 1. Repository Structure

| Path | Purpose |
| --- | --- |
| `src/config.py` | Fixed seeds (`42`), thread count (`N_THREADS = 3`), frozen feature switches, tuned model configs, blend weights |
| `src/audit.py` | Data integrity checks, adversarial validation, univariate AUCs, and fixed fold generation (`experiments/folds.npy`) |
| `src/cv_harness.py` | Unified 5-fold CV runner (`run_cv`), category alignment, paired fold comparison (`compare`), rank blending, submission validation |
| `src/features.py` | Modular feature families (`F0`–`F5`), `safe_div`, and nested fold-isolated target encoding (`F6`) |
| `src/models.py` | Standardized `fit_predict` wrappers for LightGBM, XGBoost, and CatBoost |
| `src/run_experiment.py` | CLI to run and compare single named experiments on fixed folds |
| `src/tune_lgb.py` | Resumable Optuna TPE tuning script for LightGBM (`experiments/optuna_lgb.db`) |
| `src/run_all.py` | One-command deterministic end-to-end training and submission script (`--fast` and `--fresh` flags supported) |
| `tests/test_pipeline.py` | Automated unit, isolation, determinism, cleaning, fold integrity, submission, and reproduction tests (`pytest`) |
| `make_zip.py` | Packages reproducible source archive `competition_code.zip` for Omnicampus code audit |
| `experiments/` | `RULES_DIGEST.md`, `data_profile.md`, `audit_report.txt`, `LOG.md`, `tracker.csv`, `folds.npy`, and per-experiment `meta.json` / `importance.csv` |

---

## 2. Validation Strategy, Audit & Decision Rule

1. **Data Audit (`src/audit.py`):**
   - Verified `0` `SK_ID_CURR` overlap, `0` exact duplicate rows in train or test, `TARGET` absent from test, and exact `SK_ID_CURR` alignment with `sample_submission.csv`.
   - Same-person heuristic duplicate rate within train (`0.00235`) matched the test-to-train hit rate (`0.00203`).
   - 5-fold LightGBM adversarial validation AUC: `0.51673` (< `0.55`). Max univariate feature AUC: `EXT_SOURCE_3` (`0.67957` < `0.70`).
2. **Fixed CV Splits:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` saved once to `experiments/folds.npy` (fold target rate spread `0.000029`). `SK_ID_CURR` and `TARGET` are asserted absent from all model inputs.
3. **Seed Noise Floor & Keep/Reject Rule:**
   - Measured model-seed noise floor between `lgb_raw` (seed 42: `0.751362`) and `lgb_raw_s7` (seed 7: `0.751434`): `0.000072`.
   - Feature additions were kept only if mean fold-AUC improved by `>= +0.0002` **and** at least `4/5` folds improved on identical splits (`compare()`). Feature removals required `>= -0.0001`.

---

## 3. Statement on Test-Set Statistics

**All feature transformations and statistics in the final pipeline are computed strictly from the training data (`USE_TEST_FOR_STATS = False`).**

In Phase P3 (F4 peer-relative and frequency A/B experiment), computing unsupervised group/frequency statistics on `train + test` vs `train` only yielded `-0.000068` mean fold AUC (`2/5` folds improved, below the `+0.0002` threshold). Therefore, transductive test-set statistics are disabled, and only label-free `align_categories` category union (needed for schema alignment) touches `test.csv` values before prediction.

---

## 4. Feature Engineering & Progression

- **Raw Baseline (`lgb_raw`, 32 features, `lr=0.06`):** `0.751362` mean fold AUC (`0.751310` pooled OOF) using native LightGBM `category` dtypes for all 8 text columns (stored as pandas 3 `str`) and native `NaN` handling.
- **Kept Feature Families (46 effective features after P4 pruning):**
  1. **`F2` Affordability & Debt Ratios (`+0.008755` mean AUC, `5/5` folds -> `0.760117`):** `ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `CREDIT_TERM` (`AMT_CREDIT / AMT_ANNUITY`), `CREDIT_GOODS_RATIO`, `GOODS_INCOME_RATIO`, `CREDIT_MINUS_GOODS`, `INCOME_PER_PERSON`, `INCOME_PER_CHILD`, `ANNUITY_PER_PERSON`, `INCOME_MINUS_ANNUITY` (verified annual-scale annuity in `experiments/data_profile.md`: median `AMT_ANNUITY / AMT_INCOME_TOTAL = 0.1630`).
  2. **`F5` Loan-Structure Density (`+0.001273` mean AUC, `4/5` folds -> `0.761389`):** Train-only counts sharing `(AMT_CREDIT, AMT_ANNUITY)`, `AMT_GOODS_PRICE`, and rounded `CREDIT_TERM` (`DENSITY_CREDIT_ANNUITY`, `DENSITY_GOODS_PRICE`, `DENSITY_CREDIT_TERM_ROUND`).
  3. **`F6` Nested Fold-Isolated Target Encoding (`+0.000455` mean AUC, `4/5` folds -> `0.761844`):** Smoothed (`alpha = 50`) mean `TARGET` for `ORGANIZATION_TYPE` and `OCCUPATION_TYPE`. Computed strictly inside each outer training fold via an inner 5-fold deterministic `KFold(random_state=seed)` out-of-fold scheme; validation/test rows use the full outer-training-fold mapping (unseen categories receive global training mean).
  4. **P4 Pruning (`-0.000000` tie -> KEEP):** Dropped `FLAG_MOBIL` (`0.0` gain importance across all 5 folds).
- **Rejected Families (logged in `experiments/LOG.md` & `experiments/tracker.csv`):**
  - `F0` cleaning/flags (`+0.000186`, `3/5` folds), `F1` external score block (`+0.000126`, `3/5` folds) and compact `F1` core (`-0.000609`, `1/5` folds), `F3` age/time ratios (`-0.000039`, `2/5` folds), `F4` peer-relative/frequency (`-0.001741`, `0/5` folds), P4 bottom-30% pruning (`-0.000365`, `1/5` folds).

---

## 5. Models, Tuning, Seeds & Ensemble

| Component / Experiment | Config Highlights | Seeds | Mean Fold AUC | Pooled OOF AUC |
| --- | --- | --- | ---: | ---: |
| `lgb_raw` | `LGB_START` (`lr=0.06`), 32 raw cols | `42` | `0.751362` | `0.751310` |
| `lgb_f2` | + F2 affordability (42 cols) | `42` | `0.760117` | `0.760110` |
| `lgb_f2_f5` | + F5 density (45 cols) | `42` | `0.761389` | `0.761348` |
| `lgb_f2_f5_f6` | + F6 nested TE (47 cols) | `42` | `0.761844` | `0.761783` |
| `lgb_frozen_lr02` | Frozen features at `lr=0.02`, deterministic | `42` | `0.762914` | `0.762892` |
| `lgb_tuned_lr02` | Optuna Trial #12 (`depth=4, leaves=15, ff=0.242, l1=3.44, l2=2.25`), `lr=0.02` | `42` | `0.763523` | `0.763490` |
| `xgb_frozen_codes` | `XGBClassifier(hist, depth=5, lr=0.03, colsample=0.5, mcw=20, l2=10)` + int codes | `42` | `0.763183` | `0.763140` |
| `cat_frozen` | `CatBoostClassifier(depth=6, lr=0.04, l2=5, Bernoulli=0.8)` + native `cat_features` | `42` | `0.763174` | `0.763124` |
| `LGB_3SEED` | Rank average of `lgb_tuned_lr02` across 3 seeds | `42, 1024, 2025` | `0.763910` | `0.763889` |
| `CAT_2SEED` | Rank average of `cat_frozen` across 2 seeds | `42, 1024` | `0.763862` | `0.763828` |
| `XGB_2SEED` | Rank average of `xgb_frozen_codes` across 2 seeds | `42, 1024` | `0.763846` | `0.763817` |
| **`blend_final`** | **Equal family rank blend (`1/3 LGB + 1/3 CAT + 1/3 XGB`)** | Multi | **`0.765206`** | **`0.765177`** |

*(A diversity `extra_trees=True` LightGBM run scored `0.758940` and was rejected from the blend. Coarse 0.1 weight grid search scored `+0.000026` over equal weights across `3/5` folds, below the `0.0002` P8 threshold, so equal family weights `1/3, 1/3, 1/3` were retained.)*

---

## 6. Reproducibility & Usage

### Environment
Pinned dependencies in `requirements.txt` (`python==3.11.9`, `numpy==2.4.6`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `lightgbm==4.7.0`, `xgboost==3.2.0`, `catboost==1.2.10`, `optuna==5.1.0`, `pytest==9.2.0`). All LightGBM final models run with `deterministic=True` and `force_row_wise=True`.

### One-Command End-to-End Reproduction
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/run_all.py
```
Outputs validated `submissions/sub_final.csv`, `submissions/sub_final.json`, and `output/submission.csv` directly from `input/*.csv`.
For a sub-hour single-CatBoost-seed verification run (`~35-45 min`):
```powershell
.\.venv\Scripts\python.exe src/run_all.py --fast
```

### Run Unit & Integrity Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

### Build Submission Code Archive
```powershell
.\.venv\Scripts\python.exe make_zip.py
```

---

## 7. Citations & References

- **Official Course Materials (GCI Global 2026, Matsuo-Iwasawa Lab, University of Tokyo):** `README.ipynb`, `tutorial.ipynb`, and `gci_competition_master_context_guideline.md` (problem framing, outlier/placeholder definitions for `DAYS_EMPLOYED == 365243`, `OWN_CAR_AGE >= 60`, `ORGANIZATION_TYPE == "XNA"`, and submission schema).
- **Library Documentation:**
  - LightGBM Python API & Parameters: https://lightgbm.readthedocs.io/
  - XGBoost Python API & Histogram Categorical Support: https://xgboost.readthedocs.io/
  - CatBoost Python API: https://catboost.ai/en/docs/
  - Optuna TPE Sampler & RDB Storage: https://optuna.readthedocs.io/
  - scikit-learn `StratifiedKFold`, `KFold`, `roc_auc_score`: https://scikit-learn.org/
  - SciPy `rankdata`, `spearmanr`: https://docs.scipy.org/

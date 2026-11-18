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

---

# Iteration 2 (`v2`) — Group-Validated Multi-View Pipeline

> **Pre-Registered Improvement Summary:** `blend_final_v2` improves mean 5-fold group-CV AUC from **`0.765179`** (v1 blend scored on `folds_group.npy`) to **`0.766841`** (**`+0.001662`**, **`5/5` folds improved**), with pooled OOF AUC **`0.766777`** (vs v1 `0.765177`).

## 8. v2 Audit, Group Folds & Count Parity (`src/audit_v2.py`)

1. **Customer-Fingerprint & Group Folds (`experiments_v2/folds_group.npy`):**
   - Repeat-applicant fingerprint `K3 = (CODE_GENDER, round(REGION_POPULATION_RELATIVE, 6), DAYS_BIRTH - DAYS_ID_PUBLISH, round(DAYS_BIRTH - DAYS_REGISTRATION))` has a `0.018%` chance-collision null rate (`< 1%`), identifies `3,472` multi-application rows (`2.028%` of train, `1,748` pairs), and matches `1.933%` of test rows (`1.654%` of validation rows under v1's random `folds.npy`).
   - Within train multi-row groups, repeat applications exhibit strong label concordance: $P(y=1 \mid \text{other member } y=1) = 0.4151$ vs $P(y=1 \mid \text{other } y=0) = 0.0371$ against the `0.0807` global rate (`5.14x` lift).
   - **CV Switch:** v2 switches from random `StratifiedKFold` to `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `K3` (`experiments_v2/folds_group.npy`, max folds per group = `1`, target rate spread `0.000029`), plus a 10-fold `StratifiedGroupKFold` (`experiments_v2/folds10_group.npy`).
2. **Count-Parity Fix (`F5_LOO`):**
   - v1's `F5` raw counts included self on train (`>= 1`) while unseen test rows received `0`, yielding a 5-fold adversarial AUC of `0.59309`. Subtracting `1` on train without coarsening high counts created exact off-by-1 separation (`0.93734`).
   - `F5_LOO` applies leave-self-out (`count - 1`, clipped at `0`) on train and deterministic `quantize_count` (`<10` exact, `10..99` nearest `5`, `>=100` nearest `50`) on both train and test, cutting adversarial AUC to **`0.52464`** (`<= 0.53`) and improving LightGBM CV by **`+0.000792`** (`5/5` folds) on `folds_group.npy`.
3. **Train-Only Statistics & Grey-Zone Linkage Policy:**
   - **Train-only statistics:** `USE_TEST_FOR_STATS = False` across the entire v2 pipeline.
   - **Grey-zone flag (`ENABLE_LINKAGE = False`):** Nested leave-self-out customer-linkage features (`B1`: `FP_GROUP_SIZE_LOO`, `FP_N_PRIOR`, `FP_PRIOR_DEFAULT_RATE`) produced `+0.001428` (`5/5` folds) on v1 random `folds.npy` (where groups leak across folds) vs `-0.000340` (`3/5` folds) on group-isolated `folds_group.npy`. `ENABLE_LINKAGE` remains **`False`** in the shipped v2 submission (`submissions_v2/sub_final_v2.csv`); a standby linkage variant (`submissions_v2/sub_final_v2_linkage.csv`) was built only as an unshipped counterfactual.

## 9. v2 Feature Views, Nested Target Features & Gate Checks

- **Accepted P3 Families on `folds_group.npy` (`EXP_LR = 0.06`, starting from `lgb2_inc = 0.761828`):**
  1. `F5_LOO` parity-fixed density (`+0.000792` vs raw `F5`, `5/5` folds).
  2. `A0b` ordinal integer categorical codes for LightGBM (`use_codes=True`, `NaN -> -1`, `+0.000439`, `4/5` folds -> `0.762267`).
  3. `A1` loan term/rate family (`TERM_MODE` from 25 dominant `CREDIT_TERM` peaks, `TERM_RESID_REL`, `COST_PROXY`, `RATE_PER_PERIOD` via 30-step vectorised bisection, `DOWNPAY_PROXY`, `GOODS_EQ_CREDIT`, `PRODUCT_CODE`, `+0.001199`, `5/5` folds -> `0.763467`).
  4. `A6_INT` integer age `floor(-DAYS_BIRTH / 365)` (`+0.000601`, `4/5` folds -> `0.764068`).
- **Frozen Feature Views (`freeze-features-v2`):**
  - **`VIEW_A` (54 effective cols, adv AUC `0.51703`):** `["F2", "F5_LOO", "F6", "A1", "A6_INT"]` with `use_codes=True`.
  - **`VIEW_B` (102 effective cols, adv AUC `0.52580`):** `["F0", "F1", "F2", "F3", "F5_LOO", "F6", "A1", "A5", "A6_INT"]` (diverse broad tree view; `F4_LITE` excluded after failing the `<= 0.53` adversarial check).
  - **`VIEW_C` (61 effective cols):** `VIEW_A` + nested `A4` smoothed interaction target encodings (`OCC_X_ORG`, `EDU_X_INC`, `CON_X_GEN_X_FAM`, `REGION_ID`, `PRODUCT_CODE`, `TERM_X_CON`, `alpha=50`, `min_count=30`) + nested `A3` `k=500` KNN target mean (`mode="v2"` missingness-pattern isolated Euclidean distance over quantile-normal `[EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, CREDIT_TERM]`, falling back to global `v1` when pattern count `< 3,000`).
- **Nested Isolation & Permuted-Label Gate:**
  - Every target-derived feature (`F6` TE, `A4` TE combos, `A3` KNN target mean, `B1` linkage) is computed strictly nested inside each outer CV fold (training rows exclude self via inner deterministic 5-fold KFold or exact row-index exclusion; validation/test rows query only the outer training split). Verified via label-shuffle and single-row label-flip unit tests in `tests/test_pipeline_v2.py`.
  - **Permuted-label gate test (`lgb2_gate_permuted_y_fixed300`, fixed `n_estimators=300` without early-stopping validation peeking):** pooled OOF AUC = **`0.500894`** (`[0.494907, 0.495749, 0.502668, 0.508644, 0.503385]`), inside `[0.495, 0.505]`. *(Note: running early stopping on random validation labels induces ~`+0.005` upward selection bias even on the raw 32 columns, which scored `0.506547` vs `VIEW_A`'s `0.506376`.)*

## 10. v2 Model Suite, Honest Blend-CV & Reproducibility

- **Final Ensemble (`blend_final_v2`, 10 model-view groups / 20 models, equal-weight fold-isolated rank average):**
  - `LGB_VIEW_A` (3 seeds `42, 1024, 2025`, Optuna top-3 `lr=0.02`, pooled `0.765648`), `LGB_VIEW_B` (3 seeds, `0.764476`), `LGB_VIEW_C` (3 seeds, `0.765653`), `LGB_VIEW_A_10F` (10-fold LightGBM, `0.765708`, `+0.000370` over 5-fold), `XGB_VIEW_A` (2 seeds, `0.764770`), `XGB_VIEW_B` (2 seeds, `0.763538`), `XGB_VIEW_C` (2 seeds, `0.764298`), `CAT_VIEW_A` (2 fast seeds `lr=0.06, depth=6, border_count=128`, `0.764087`), `CAT_VIEW_B` (`0.762397`), `CAT_VIEW_C` (`0.761534`).
  - *Rejected from blend under pre-registered rules:* binned L2 `LogisticRegression` (best `C=0.01`, `0.754377`, `-0.000030` blend gain), PyTorch CPU 2-layer `(256, 128)` `MLP` (`0.747779` 2-seed, `-0.000162` blend gain), native-categorical LightGBM (`0.756174`), and `lr=0.01` LightGBM (`+0.000078 < +0.0003`).
  - *Honest Blend-CV stacker selection:* On the 10 admitted groups, `equal_rank` scored **`0.766841`** mean fold AUC (`0.766777` pooled OOF). `meta_top2` (`0.767103`, `+0.000262`, `4/5` folds) and `logreg_c0.01` (`0.767094`, `+0.000254`, `4/5` folds) fell short of the `+0.0003` complexity gate, so `equal_rank` was retained.
- **Seeds, Versions, Runtimes & Tolerances:**
  - Seeds: global `42`; LightGBM `[42, 1024, 2025]`; XGBoost/CatBoost `[42, 1024]`; `PYTHONHASHSEED=42`; LightGBM `deterministic=True, force_row_wise=True`; `N_THREADS = 3` (2 physical / 4 logical cores).
  - Pinned environment in `requirements_v2.txt` (adds `torch==2.15.0+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu`).
  - Expected runtimes: `--fast` (4 LightGBM models, `0.765773` pooled OOF) **`~25.5–27 min`** (`1530.8s` / `1628.6s`, bit-identical SHA256 across fresh runs); `--full` (20 models from scratch) **`~5.1 hours`** (`~63 min` when resuming cached components).
  - Cross-machine thread tolerance (`lgb_v2_viewA_s42` with `2` vs `3` vs `4` threads): **`0.000000`** AUC difference.
- **Reproduce & Package v2:**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements_v2.txt
  .\.venv\Scripts\python.exe src/run_all_v2.py --full
  .\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_pipeline_v2.py -v
  .\.venv\Scripts\python.exe make_zip_v2.py
  ```
- **Idea-Source Credits (v2):** All code was implemented independently from scratch; feature-engineering and validation ideas were inspired by the Kaggle Home Credit 1st-place team's post-mortem write-up (feature restriction, implied annuity interest rates, KNN target features, neural net vs GBDT gap, stacking expectations), public Kaggle Home Credit write-ups (EXT/credit ratios and annuity-term peaks), and a public GCI competition repository README's structured experiment discipline.

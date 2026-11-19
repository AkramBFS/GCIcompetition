# GCI Global 2026 — Home Credit Default Risk Solution

End-to-end, leak-free, reproducible tabular machine learning pipeline for the **GCI Global 2026 Home Credit Default Risk** competition.

## How to reproduce

> **Submitted Version:** **`v2`** ([`submissions_v2/sub_final_v2.csv`](submissions_v2/sub_final_v2.csv), `ENABLE_LINKAGE = False`) is the submitted competition solution (`0.766841` mean 5-fold group-CV AUC | `0.766777` pooled OOF AUC). All `v1` files (`src/run_all.py`, `src/config.py`, `experiments/`, `submissions/`, `requirements.txt`, Sections 1–7 below) and grey-zone linkage files (`ENABLE_LINKAGE = True` / `experiments_v2/blend_final_v2_linkage/`) are preserved for history only.

- **Shipped Submission CSV:** `submissions_v2/sub_final_v2.csv`
- **Shipped CSV SHA256:** `51e1cac2e1ba739496f3a5a6d0fe52b05ef352dbeca1dc3763ddf3e376831f96`
- **Python & Dependencies:** Python `3.11.9` — install the 8 cross-platform direct packages via `pip install -r requirements_v2.txt` (no `torch` or Windows-only packages needed).
- **Data Location:** Place `train.csv`, `test.csv`, and `sample_submission.csv` inside `input/` (see `input/README.txt`).
- **Fast Verification (`--fast`):** Runs 4 LightGBM models (`VIEW_A` x3 + `VIEW_B` x1) in **~20–27 min** (`1,218.1s` in a clean minimal venv), achieving **`0.765773` pooled OOF AUC** (`0.765830` mean fold AUC, bit-identical SHA256 `60bd178560f83dd5589dfb53658237ac0fc12a580fae13b5bb807c48e9060780`):
  ```bash
  PYTHONHASHSEED=42 python src/run_all_v2.py --fast --fresh
  ```
  Or execute [`solution_v2.ipynb`](solution_v2.ipynb) cell-by-cell (`--fast` default with `--full` commented).
- **Full End-to-End Reproduction (`--full`):** Runs all 20 models across 10 model-view groups from scratch in **~4.2 h** (`14,983.5s` uncached on 2 physical / 4 logical CPU cores), reproducing **`0.766777` pooled OOF AUC** (`0.766841` mean fold AUC) and exact SHA256 `51e1cac2e1ba739496f3a5a6d0fe52b05ef352dbeca1dc3763ddf3e376831f96`:
  ```bash
  PYTHONHASHSEED=42 python src/run_all_v2.py --full --fresh
  ```
- **Seeds Used:** Global & hash seed `42`; `StratifiedGroupKFold` (5-fold & 10-fold) `42`; LightGBM `[42, 1024, 2025]` (+ 10-fold `42`); XGBoost `[42, 1024]`; CatBoost `VIEW_A` `[42, 1024]` & `VIEW_B`/`VIEW_C` `42`; nested TE/KNN matching model seed; `honest_blend_cv` `42`.
- **Automated Tests:** `pytest tests/test_pipeline.py tests/test_pipeline_v2.py -v`

---

- **Metric:** ROC-AUC (`sklearn.metrics.roc_auc_score`)
- **Dataset:** `input/train.csv` (`171,202` rows x `34` columns, `8.0729%` positive `TARGET`), `input/test.csv` (`61,500` rows x `33` columns)
- **v1 Historical Blend CV:** `0.765206` mean 5-fold random-CV AUC | `0.765177` pooled OOF AUC (Omnicampus public `0.76876`; vs tutorial LightGBM `0.7443` / tutorial 4-model blend `0.7531` on 70/30 holdout)

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

> **Pre-Registered Improvement Summary:** On identical group-isolated CV (`experiments_v2/folds_group.npy`), `blend_final_v2` improves mean 5-fold AUC from **`0.765179`** (`v1` blend) to **`0.766841`** (**`+0.001662`**, **`5/5` folds improved**), with pooled OOF AUC **`0.766777`** (vs `v1` `0.765177`). `submissions_v2/sub_final_v2.csv` is generated **100% automatically** by `src/run_all_v2.py --full` from `input/*.csv` with zero manual edits or hand-labeled predictions.

---

## Validation design

### 1. Group-Isolated CV (`StratifiedGroupKFold` on Customer Fingerprint `K3`)
- **Why v1 random folds leaked:** Customer-fingerprint auditing (`src/audit_v2.py`) showed that `K3 = (CODE_GENDER, round(REGION_POPULATION_RELATIVE, 6), DAYS_BIRTH - DAYS_ID_PUBLISH, round(DAYS_BIRTH - DAYS_REGISTRATION))` identifies `3,472` repeat-applicant rows in `train.csv` (`2.028%`, `1,748` pairs) with a **`0.018%`** chance-collision null rate (`< 1%`) and **`1.933%`** test-to-train match rate. Within multi-application train groups, default labels show strong concordance: $P(y=1 \mid \text{other member } y=1) = 0.4151$ vs $P(y=1 \mid \text{other } y=0) = 0.0371$ against the `0.0807` base rate (`5.14x` lift). Under v1's random `StratifiedKFold` (`experiments/folds.npy`), `1.654%` of validation rows shared a fingerprint with training-fold rows.
- **v2 group-split solution:** v2 replaces random splits with `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` saved to `experiments_v2/folds_group.npy` (`169,462` groups / `167,730` singletons, **`max_folds_per_group = 1`**, target-rate spread `0.000029`) plus a verified 10-fold `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` saved to `experiments_v2/folds10_group.npy` (**`max_folds_per_group = 1`**, target-rate spread `0.000058`). Both invariants are enforced by `tests/test_pipeline_v2.py::test_determinism_and_schema_v2`.

### 2. v1 vs v2 Comparison on the Same Footing (`experiments_v2/folds_group.npy`)

| Pipeline / Mode | CV Split Basis | Mean Fold AUC | Pooled OOF AUC | Delta vs v1 (same footing) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `v1` (`blend_final`, 7 models) | `folds.npy` (random) | `0.765206` | `0.765177` | — | Omnicampus public score `0.76876` |
| `v1` (`blend_final`, 7 models) | `folds_group.npy` (group) | `0.765179` | `0.765177` | Baseline | Per-fold `[0.765393, 0.774380, 0.760458, 0.765747, 0.759917]` |
| `v2 --fast` (4 LGB models) | `folds_group.npy` (group) | `0.765830` | `0.765773` | `+0.000651` | `VIEW_A` x3 + `VIEW_B` x1; sub-30 min verified alternative |
| **`v2 --full` (`blend_final_v2`, 20 models)** | **`folds_group.npy` (group)** | **`0.766841`** | **`0.766777`** | **`+0.001662` (`5/5` folds)** | **Shipped (`ENABLE_LINKAGE=False`); `[0.766895, 0.775647, 0.760755, 0.767735, 0.763171]`** |

### 3. Uncached Full-Run Runtime, `--fast` Alternative & Cross-Machine Tolerance
- **Uncached `--full --fresh` wall time (minimal `requirements_v2.txt` venv, no cache):** **`14,983.48s` (`4h 10m` / `4.16h`)** end-to-end on 2 physical / 4 logical CPU cores (`N_THREADS = 3`), with `14,928.96s` in model training/evaluation across the 20 components (`10` LightGBM `4,126.96s` / `68.8m`; `6` XGBoost `3,145.89s` / `52.4m`; `4` CatBoost `7,656.11s` / `127.6m`) and **`0 / 20` OOF/test SHA256 mismatches (`max_abs_diff = 0.0`)** against cached/P5 runs (`experiments_v2/p7_uncached_comparison.json`). Resuming warm component caches completes in `~63 min`.
- **`--fast` alternative (`--fast --fresh`):** Runs 4 deterministic LightGBM models (`VIEW_A` seeds `42, 1024, 2025` + `VIEW_B` seed `42`) in **`1,218.1s` (`20.3 min`)** in the minimal venv (`1,530.8s` / `1,628.6s` in the dev venv), achieving `0.765830` mean fold AUC / `0.765773` pooled OOF AUC with bit-identical CSV output (`SHA256 = 60bd178560f83dd5589dfb53658237ac0fc12a580fae13b5bb807c48e9060780`, `experiments_v2/p7_fast_determinism.json`).
- **Expected cross-machine tolerance:** Changing thread count (`2` vs `3` vs `4` threads) on `lgb_v2_viewA_s42` produces **`0.000000`** AUC difference (`experiments_v2/p7_thread_tolerance.json`). Across different OS, CPU ISA, or BLAS builds, `LightGBM` (`deterministic=True, force_row_wise=True`), `XGBoost` (`hist`), and `CatBoost` remain reproducible within `<= 1e-4` pooled OOF AUC and `>= 0.9999` Spearman rank correlation (`tests/test_pipeline_v2.py::test_reproduction_check_v2`).

### 4. Grey-Zone Customer Linkage (`ENABLE_LINKAGE = False`) & Test-Statistics Policy
- **What `ENABLE_LINKAGE` does:** `B1` (`compute_nested_linkage_features` in `src/features.py`) computes nested leave-self-out group summary features (`FP_GROUP_SIZE_LOO`, `FP_N_PRIOR`, `FP_PRIOR_DEFAULT_RATE`) over fingerprint `K3` using **training labels only** (never validation or test labels; validation/test rows look up the outer training split, and each training row excludes itself via label-independent leave-self-out counts).
- **Disabled in shipped submission:** `ENABLE_LINKAGE` is **`False`** in `src/config_v2.py` and in `submissions_v2/sub_final_v2.csv`. While enabling it raised random-fold CV by `+0.001428` (`5/5` folds: `0.764422 -> 0.765849`) by exploiting cross-fold repeat-applicant leakage, under leak-free group isolation (`folds_group.npy`) it hurt LightGBM CV (`-0.000340`: `0.764068 -> 0.763728`, `3/5` folds) and yielded `0.765785` mean / `0.765726` pooled in an unshipped 3-model standby blend (`submissions_v2/sub_final_v2_linkage.csv`).
- **Train-only statistics:** `USE_TEST_FOR_STATS = False` across the entire v2 pipeline.

---

## 8. v2 Count Parity (`F5_LOO`), Feature Views & Gate Checks

1. **Count-Parity Fix (`F5_LOO`):**
   - `v1`'s `F5` counts included self on train (`>= 1`) while unseen test rows received `0`, yielding an adversarial validation AUC of `0.59309`. Naive `count - 1` on train without coarsening high counts created exact off-by-1 integer separation (`0.93734`).
   - `F5_LOO` subtracts `1` on train (clipped at `0`) and applies deterministic `quantize_count` (`< 10` exact, `10..99` nearest `5`, `>= 100` nearest `50`), cutting engineered adversarial AUC to **`0.52464`** (`<= 0.53`) and improving LightGBM CV by **`+0.000792`** (`5/5` folds) on `folds_group.npy`.
2. **Accepted P3 Families on `folds_group.npy` (`EXP_LR = 0.06`, starting from `lgb2_inc = 0.761828`):**
   - `A0b` ordinal integer categorical codes for LightGBM (`use_codes=True`, `NaN -> -1`, `+0.000439`, `4/5` folds -> `0.762267`).
   - `A1` loan term/rate family (`TERM_MODE` from 25 dominant `CREDIT_TERM` peaks, `TERM_RESID_REL`, `COST_PROXY`, `RATE_PER_PERIOD` via 30-step vectorised bisection, `DOWNPAY_PROXY`, `GOODS_EQ_CREDIT`, `PRODUCT_CODE`, `+0.001199`, `5/5` folds -> `0.763467`).
   - `A6_INT` integer age `floor(-DAYS_BIRTH / 365)` (`+0.000601`, `4/5` folds -> `0.764068`).
3. **Frozen Feature Views (`freeze-features-v2`):**
   - **`VIEW_A` (54 effective cols, adv AUC `0.51703`):** `["F2", "F5_LOO", "F6", "A1", "A6_INT"]` with `use_codes=True`.
   - **`VIEW_B` (102 effective cols, adv AUC `0.52580`):** `["F0", "F1", "F2", "F3", "F5_LOO", "F6", "A1", "A5", "A6_INT"]` (`F4_LITE` excluded after failing the `<= 0.53` adversarial check).
   - **`VIEW_C` (61 effective cols):** `VIEW_A` + nested `A4` smoothed interaction target encodings (`OCC_X_ORG`, `EDU_X_INC`, `CON_X_GEN_X_FAM`, `REGION_ID`, `PRODUCT_CODE`, `TERM_X_CON`, `alpha=50`, `min_count=30`) + nested `A3` `k=500` KNN target mean (`mode="v2"` missingness-pattern isolated Euclidean distance over quantile-normal `[EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, CREDIT_TERM]`, falling back to global `v1` when pattern count `< 3,000`).
4. **Permuted-Label Gate Test:**
   - Fixed-iteration `lgb2_gate_permuted_y_fixed300` (`n_estimators=300`, no early-stopping validation peeking) scored pooled OOF AUC **`0.500894`** (`[0.494907, 0.495749, 0.502668, 0.508644, 0.503385]`), inside `[0.495, 0.505]`. *(Early stopping on random validation labels induces ~`+0.005` selection bias even on the raw 32 columns, which scored `0.506547` vs `VIEW_A`'s `0.506376`.)*

---

## 9. v2 Model Suite, Honest Blend-CV & Environment Hygiene

- **Final Ensemble (`blend_final_v2`, 10 model-view groups / 20 models, equal-weight fold-isolated rank average):**
  - `LGB_VIEW_A` (3 seeds `42, 1024, 2025`, Optuna Trial #11 / top-3 `lr=0.02`, pooled `0.765648`), `LGB_VIEW_B` (3 seeds, `0.764476`), `LGB_VIEW_C` (3 seeds, `0.765653`), `LGB_VIEW_A_10F` (10-fold LightGBM on `folds10_group.npy`, `0.765708`, `+0.000370` pooled over 5-fold seed 42), `XGB_VIEW_A` (2 seeds, `0.764770`), `XGB_VIEW_B` (2 seeds, `0.763538`), `XGB_VIEW_C` (2 seeds, `0.764298`), `CAT_VIEW_A` (2 fast seeds `lr=0.06, depth=6, border_count=128`, `0.764087`), `CAT_VIEW_B` (`0.762397`), `CAT_VIEW_C` (`0.761534`).
  - *Rejected from blend under pre-registered gates:* binned L2 `LogisticRegression` (best `C=0.01`, `0.754377`, `-0.000030` blend gain), PyTorch CPU 2-layer `(256, 128)` `MLP` (`0.747779` 2-seed, `-0.000162` blend gain), native-categorical LightGBM (`0.756174`), and `lr=0.01` LightGBM (`+0.000078 < +0.0003`).
  - *Honest Blend-CV stacker selection:* On the 10 admitted groups, `equal_rank` scored **`0.766841`** mean fold AUC (`0.766777` pooled OOF). `meta_top2` (`0.767103`, `+0.000262`, `4/5` folds) and `logreg_c0.01` (`0.767094`, `+0.000254`, `4/5` folds) fell short of the `+0.0003` complexity gate, so `equal_rank` was retained.
- **Minimal Cross-Platform Dependencies vs Experiment Extras:**
  - `requirements_v2.txt`: minimal 8 cross-platform direct dependencies only (`numpy==2.4.6`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `lightgbm==4.7.0`, `xgboost==3.2.0`, `catboost==1.2.10`, `pytest==9.2.0`). No Windows-only packages and no `torch`; verified in a fresh clean venv to run `--fast --fresh`, `--full --fresh`, and all 14 `pytest` tests (`14 passed`). `torch` is imported lazily inside `src/models.py::mlp_model` only when running exploratory neural experiments.
  - `requirements_experiments.txt`: adds optional exploratory packages (`torch==2.15.0+cpu`, `optuna==5.1.0`, visualization/notebook tools).
  - `requirements_lock_windows.txt`: full frozen Windows development environment lockfile.
- **Reproduce & Package v2:**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements_v2.txt
  .\.venv\Scripts\python.exe src/run_all_v2.py --full
  .\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_pipeline_v2.py -v
  .\.venv\Scripts\python.exe make_zip_v2.py
  ```

---

## 10. Random Seeds, Automation & Rules Compliance

### Automated Execution & Post-Processing Audit
- **100% Automated Model Generation:** `submissions_v2/sub_final_v2.csv` is generated completely automatically by `src/run_all_v2.py --full` from `input/train.csv`, `input/test.csv`, and `input/sample_submission.csv` with **zero manual steps**, zero hand-labeling, and zero external data.
- **Post-Processing Audit:** Code audit of `src/*.py` confirms zero per-row or per-group prediction overrides, threshold edits, or EDA-derived prediction rules after model inference. The only post-model operations are empirical rank averaging (`scipy.stats.rankdata(x) / len(x)`), equal-weight averaging across the 10 model-view groups, `[1e-5, 1 - 1e-5]` numeric logit-safety bounds inside `src/stacking_v2.py::fit_apply_rank` (unused by `equal_rank` inference beyond floating-point `[0.0, 1.0]` clipping), and schema/NaN/range assertions in `write_submission`.

### Complete Inventory of Random Seeds

| Component / Stage | Seed(s) | Where Set |
| --- | --- | --- |
| Global default & process hash seed | `42` | `SEED = 42` (`src/config.py`, `src/config_v2.py`), `PYTHONHASHSEED = "42"` (`src/run_all_v2.py`) |
| 5-fold & 10-fold Group CV (`StratifiedGroupKFold`) | `42` | `experiments_v2/folds_group.npy`, `folds10_group.npy` (`src/audit_v2.py`, `src/run_all_v2.py`) |
| v1 5-fold Stratified CV (`StratifiedKFold`) | `42` | `experiments/folds.npy` (`src/audit.py`) |
| Nested TE (`KFold`) & KNN (`QuantileTransformer` `seed + j`) | Model seed (`42, 1024, 2025`) | `src/features.py`, `src/run_all_v2.py` (`42` during discovery/audit) |
| Final `LGB_VIEW_A`, `LGB_VIEW_B`, `LGB_VIEW_C` | `42, 1024, 2025` | `FINAL_SEEDS` (`src/config_v2.py`, `deterministic=True`, `force_row_wise=True`) |
| Final `LGB_VIEW_A_10F` (10-fold LightGBM) | `42` | `src/run_all_v2.py` |
| Final `XGB_VIEW_A`, `XGB_VIEW_B`, `XGB_VIEW_C` | `42, 1024` | `XGB_FINAL_SEEDS` (`src/config_v2.py`) |
| Final `CAT_VIEW_A` / `CAT_VIEW_B`, `CAT_VIEW_C` | `42, 1024` / `42` | `CAT_FINAL_SEEDS` (`src/config_v2.py`) |
| `honest_blend_cv` & counterfactual stackers | `42` | `src/stacking_v2.py`, `src/run_all_v2.py` |
| Exploratory / rejected / audit checks (not in shipped CSV) | `42`, `7` (v1 noise floor), `777` (val-label shuffle), `999` (permuted-`y` gate), `42, 1024` (MLP), `42` (LogReg/Optuna/linkage) | `src/audit_v2.py`, `src/run_p3_p4.py`, `src/run_p5_p6.py`, `src/tune_lgb.py` |

### Inventory of Domain & EDA-Derived Constants in `src/`

| Constant | Value / Rule | Location | Provenance |
| --- | --- | --- | --- |
| `DAYS_EMPLOYED` sentinel | `365243` -> `NaN` | `src/features.py:52,105`, `src/audit_v2.py:42` | Official Home Credit placeholder (`tutorial.ipynb`, `experiments/data_profile.md`: 30,898 train rows) |
| `OWN_CAR_AGE` outlier cutoff | `>= 60.0` -> `NaN` | `src/features.py:59,111` | `input/train.csv` EDA (`experiments/data_profile.md`: 1,826 rows at `64/65/91`) |
| Categorical placeholders | `"XNA"`, `"Unknown"` -> `NaN` | `src/features.py:85,88` | `input/train.csv` EDA (`experiments/data_profile.md`: `ORGANIZATION_TYPE`, `CODE_GENDER`, `NAME_FAMILY_STATUS`) |
| `DROPPED_FEATURES` | `["FLAG_MOBIL"]` | `src/config.py:35`, `src/config_v2.py:41` | `input/train.csv` near-constant (`0` gain importance across all folds, v1 P4 tie `0.000000`) |
| `_DOMINANT_TERM_MODES` | 25 peaks `[8.4 ... 37.8]` | `src/features.py:19-26` | Prominent local maxima (`count >= 500`) of `CREDIT_TERM.round(1)` on `input/train.csv` computed in `src/audit_v2.py` P1(d) (`experiments_v2/audit_v2.md`) |
| `quantize_count` bins | `<10` exact; `10..99` nearest `5`; `>=100` nearest `50` | `src/features.py:459-463` | Chosen in P1(c) (`experiments_v2/audit_v2.md`) to remove off-by-1 train-vs-test count separation (`adv AUC 0.93734 -> 0.52464`) |
| `AMT_GOODS_PRICE` bucket | `round(goods / 1000.0)` | `src/features.py:547` | Domain A1 `PRODUCT_CODE` bucketing to nearest 1,000 currency units |
| `EXT_WMEAN` weights | `[2.0, 1.0, 3.0]` for `EXT_1, EXT_2, EXT_3` | `src/features.py:153` | Domain ordering (`EXT_3 > EXT_1 > EXT_2`, matching train univariate AUCs `0.6796 > 0.6647 > 0.6582`); used only in `VIEW_B` |
| Nested TE smoothing | `alpha = 50.0`, `min_count = 30` (combos) | `src/features.py:900`, `src/config_v2.py:136` | Pre-registered F6/A4 regularization hyperparameters |
| Nested KNN parameters | `k = 500`, `min_pattern_rows = 3000` | `src/features.py:1024`, `src/config_v2.py:138` | Pre-registered A3 neighborhood size; `3000` separates populous missingness patterns (`12.8k–64.2k`/fold) from sparse patterns (`23–132`) |
| `LGB_FINAL_PARAMS` | `LGB_TOP_PARAMS[2]` (Trial #11) | `src/config_v2.py:120` | Selected from v1 Optuna search (`experiments/lgb_top_params.json`) via P5 5-fold CV on `folds_group.npy` |

---

## REFERENCES

**Code Independence Statement:** All code in `src/` and `tests/` was written **independently from scratch** specifically for this competition. No external solution code, Kaggle kernel code, or third-party pipeline code was downloaded, copied, or adapted. Where any future snippet were ever adapted from an external source, its URL would be cited in an inline comment at that exact spot.

**Idea Sources & Official Documentation Credited:**
- **Kaggle Home Credit Default Risk 1st-Place Team Post-Mortem Write-Up (Ideas Only):** Curated feature-view restriction against noisy collinear features, implied loan annuity interest rates (`RATE_PER_PERIOD` / `COST_PROXY`), pattern-isolated KNN target-mean features over normalized external credit scores, empirical observation of neural-network vs GBDT performance gaps on tabular missingness patterns, and conservative expectations for linear/tree stacking gains over rank blends.
- **Public Kaggle Home Credit Default Risk Write-Ups (Ideas Only):** External-score interaction ratios/products, rounded credit-term annuity contract peaks (`TERM_MODE`), and nested out-of-fold smoothed target encoding.
- **Official Course Materials (GCI Global 2026, Matsuo-Iwasawa Lab, University of Tokyo):** `README.ipynb`, `tutorial.ipynb`, and `gci_competition_master_context_guideline.md` (competition rules, schema, and known raw data anomaly `DAYS_EMPLOYED = 365243`).
- **Official Library Documentation:** LightGBM (`https://lightgbm.readthedocs.io/`), XGBoost (`https://xgboost.readthedocs.io/`), CatBoost (`https://catboost.ai/en/docs/`), scikit-learn (`https://scikit-learn.org/`), SciPy (`https://docs.scipy.org/`), NumPy (`https://numpy.org/doc/`), pandas (`https://pandas.pydata.org/docs/`), Optuna (`https://optuna.readthedocs.io/`), and PyTorch (`https://pytorch.org/docs/`).

# Experiment Log — GCI Home Credit Default Risk

Deadline: **2026-11-20 01:00:00 UTC**

## Phase P0: Environment, Hardware & Speed Benchmark
- **Start UTC:** `2026-11-17T19:31:46Z`
- **End UTC:** `2026-11-17T19:56:33Z`
- **Hardware:** 2 physical / 4 logical CPU cores, 11.9 GB RAM (`N_THREADS = 3`, leaving 1 logical core free). Single heavy training job at a time.
- **Python & Key Libraries:**
  - `python==3.11.9`
  - `numpy==2.4.6`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `scipy==1.17.1`
  - `lightgbm==4.7.0`, `xgboost==3.2.0`, `catboost==1.2.10` (all verified compatible with pandas 3 / numpy 2.4)
  - `optuna==5.1.0`, `openpyxl==3.1.5`, `pytest==9.2.0`
- **LightGBM 1-Fold Speed Benchmark (`LGB_START`, `EXP_LR=0.06`, `threads=3`, 32 raw features):**
  - Wall time: `15.03s`, `best_iter=257`, `fold0_auc=0.747543` (~75s projected for 5-fold CV at `EXP_LR`).
- **Gate P0:** PASSED (`experiments/RULES_DIGEST.md`, `.gitignore`, `requirements.txt` frozen, imports and fits verified).

## Phase P1: Data Audit & Profile
- **Start UTC:** `2026-11-17T19:56:51Z`
- **Results:**
  - `train.csv` (`171,202 x 34`), `test.csv` (`61,500 x 33`), `SK_ID_CURR` overlap `0`, exact duplicates `0`.
  - Same-person key within-train rate (`0.00235`) matches test-to-train hit rate (`0.00203`) -> fixed 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` saved to `experiments/folds.npy` (default rate spread `0.000029`).
  - Adversarial 5-fold LightGBM AUC: `0.51673` (< 0.55); max univariate AUC: `EXT_SOURCE_3` (`0.67957` < 0.70).
  - `AMT_ANNUITY / AMT_INCOME_TOTAL` median `0.1630` (annual-scale confirmed -> enable `INCOME_MINUS_ANNUITY` in F2 and `AGE_AT_LAST_PAYMENT` in F3).
  - Missing inputs skipped cleanly in F0: `AMT_REQ_CREDIT_BUREAU_DAY`, `AMT_REQ_CREDIT_BUREAU_WEEK`.
- **Gate P1:** PASSED (`audit_report.txt`, `data_profile.md`, `folds.npy`, all 6 unit tests pass, zero red flags).

---

## Experiment Log (Hypotheses, Results, Verdicts)

### P2: Raw Baseline & Noise Floor
- **Start UTC:** `2026-11-17T19:58:17Z`
- **Hypothesis (`lgb_raw`):** Training `LGB_START` (`EXP_LR=0.06`, seed `42`, native `category` dtypes for all 8 string columns, raw `NaN` untouched) on all 32 raw columns across fixed 5-fold CV should substantially beat the tutorial LightGBM baseline (`0.7443`) by preserving categorical features and early stopping, landing around `0.748–0.753` OOF AUC.
- **Result (`lgb_raw`, 32 raw features):**
  - Fold AUCs: `[0.748225, 0.751686, 0.756469, 0.746328, 0.754102]` (mean best iter `369.8`)
  - Mean Fold AUC: **`0.751362`** (`+/- 0.003711`) | Pooled OOF AUC: **`0.751310`**
- **Hypothesis (`lgb_raw_s7`):** Re-running `lgb_raw` identically with `seed=7` measures pure model-seed variance (noise floor `abs(delta)`), expected `< 0.0003`.
- **Result (`lgb_raw_s7`, 32 raw features):**
  - Fold AUCs: `[0.748042, 0.752491, 0.757150, 0.745964, 0.753525]` (mean best iter `360.2`)
  - Mean Fold AUC: **`0.751434`** (`+/- 0.003990`) | Pooled OOF AUC: **`0.751391`**
  - Delta vs `lgb_raw`: `+0.000072` mean fold AUC (`2/5` folds improved).
- **Noise Floor:** `abs(delta) = 0.000072` (`< 0.0002`). Acceptance threshold remains `max(0.0002, NOISE_FLOOR) = +0.0002` mean fold AUC and `>= 4/5` folds improved.
- **Submission:** `submissions/sub_01_lgb_raw.csv` (with `submissions/sub_01_lgb_raw.json`) validated against `sample_submission.csv`.
- **Gate P2:** PASSED. Current incumbent: `lgb_raw` (`0.751362`).

---

### P3: Feature Families (EXP_LR = 0.06, Incumbent starts at `lgb_raw`)
- **Hypothesis (`lgb_f0` — F0 Cleaning & Flags):** Adding `NAN_COUNT`, `DAYS_EMPLOYED_ANOM` (and replacing `365243 -> NaN`), `OWN_CAR_AGE_ANOM` (and replacing `>=60 -> NaN`), `BUREAU_MISSING`, and replacing `"XNA"`/`"Unknown"` placeholders with `NaN` cleanly separates anomaly regimes from true distributions. Expected effect: `+0.0003` to `+0.0010` mean AUC gain.
- **Result (`lgb_f0`, 36 features, skipped `AMT_REQ_CREDIT_BUREAU_DAY`, `AMT_REQ_CREDIT_BUREAU_WEEK`):**
  - Fold AUCs: `[0.748127, 0.751286, 0.756947, 0.746941, 0.754440]` (mean best iter `339.6`)
  - Mean Fold AUC: `0.751548` (`+0.000186` vs `lgb_raw`) | Pooled OOF AUC: `0.751449`
  - Folds improved: `3/5` -> **REJECT** (below `+0.0002` and `< 4/5` folds; incumbent stays `lgb_raw`; downstream F1–F4 clean placeholders internally via private helpers).
- **Hypothesis (`lgb_f1` — F1 External Scores):** Adding `EXT_MEAN`, `EXT_STD`, `EXT_MIN`, `EXT_MAX`, `EXT_MEDIAN`, `EXT_PROD`, `EXT_NMISS`, `EXT_WMEAN`, pairwise products/diffs, `EXT_k_DIV_AGE`, and `EXT_MEAN_X_CREDIT_TERM` captures the strongest credit-bureau signal interactions before any imputation. Expected effect: `+0.003` to `+0.006` mean AUC gain (`5/5` folds).
- **Result (`lgb_f1`, 50 features):**
  - Fold AUCs: `[0.747501, 0.752495, 0.757140, 0.748304, 0.752002]` (mean best iter `320.0`)
  - Mean Fold AUC: `0.751488` (`+0.000126` vs `lgb_raw`) | Pooled OOF AUC: `0.751443`
  - Folds improved: `3/5` -> **REJECT** (adding 18 correlated EXT columns diluted `feature_fraction=0.35` and failed `mean>=+0.0002`, `folds>=4`; incumbent stays `lgb_raw`).
- **Hypothesis (`lgb_f2` — F2 Affordability Ratios):** Adding `CREDIT_TERM`, `ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `CREDIT_GOODS_RATIO`, `GOODS_INCOME_RATIO`, `CREDIT_MINUS_GOODS`, `INCOME_PER_PERSON`, `INCOME_PER_CHILD`, `ANNUITY_PER_PERSON`, and `INCOME_MINUS_ANNUITY` supplies nonlinear debt-to-income and loan-to-value ratios that axis-aligned trees cannot easily approximate from raw columns. Expected effect: `+0.003` to `+0.008` mean AUC gain (`5/5` folds).
- **Result (`lgb_f2`, 42 features):**
  - Fold AUCs: `[0.757543, 0.760985, 0.764673, 0.755538, 0.761844]` (mean best iter `356.4`)
  - Mean Fold AUC: **`0.760117`** (`+0.008755` vs `lgb_raw`) | Pooled OOF AUC: **`0.760110`**
  - Folds improved: `5/5` -> **KEEP** (new incumbent: `lgb_f2`, active families: `["F2"]`).
- **Hypothesis (`lgb_f2_f3` — F3 Age & Temporal Ratios):** Adding `AGE_YEARS`, `EMPLOYED_TO_AGE`, `REGISTRATION_TO_AGE`, `ID_PUBLISH_TO_AGE`, `CAR_AGE_TO_AGE`, `PHONE_CHANGE_TO_AGE`, `PHONE_CHANGE_TO_EMPLOYED`, and `AGE_AT_LAST_PAYMENT` captures career/document/phone stability relative to age and retirement-horizon loan maturity (`AGE_AT_LAST_PAYMENT`). Expected effect: `+0.0005` to `+0.0020` mean AUC gain (`5/5` folds).
- **Result (`lgb_f2_f3`, 50 features):**
  - Fold AUCs: `[0.757866, 0.760734, 0.764719, 0.755379, 0.761691]` (mean best iter `377.4`)
  - Mean Fold AUC: `0.760078` (`-0.000039` vs `lgb_f2`) | Pooled OOF AUC: `0.760021`
  - Folds improved: `2/5` -> **REJECT** (incumbent remains `lgb_f2`).
- **Hypothesis (`lgb_f2_f4_tr` — F4 Peer-Relative & Frequency, Train-Only):** Adding 25 peer-relative group features (mean/diff of `EXT_MEAN`, `AMT_INCOME_TOTAL`, `CREDIT_TERM`, `AGE_YEARS` across `ORG`, `OCC`, `EDU`, `INC`, `OCC_EDU`, min group size 30) plus 8 categorical `FREQ_*` columns computed strictly on `train` should provide peer-context signal (`+0.0005` to `+0.0020` mean AUC).
- **Result (`lgb_f2_f4_tr`, 75 features):**
  - Fold AUCs: `[0.755522, 0.760590, 0.761617, 0.755114, 0.759036]` (mean best iter `399.6`)
  - Mean Fold AUC: `0.758376` (`-0.001741` vs `lgb_f2`) | Pooled OOF AUC: `0.758364`
  - Folds improved: `0/5` -> **REJECT** (33 collinear group/freq columns severely diluted `feature_fraction=0.35` trees).
- **Hypothesis (`lgb_f2_f4_trte` — F4 A/B Test Train+Test Stats):** Re-running F4 with `use_test_for_stats=True` (label-free transductive statistics) tests whether larger sample counts improve peer means/frequencies by `>= +0.0002` over `lgb_f2_f4_tr` (kept only if gain `>= +0.0002`).
- **Result (`lgb_f2_f4_trte`, 75 features):**
  - Fold AUCs: `[0.756226, 0.760366, 0.762502, 0.754139, 0.758303]` (mean best iter `338.4`)
  - Mean Fold AUC: `0.758307` (`-0.000068` vs `lgb_f2_f4_tr`, `2/5` folds improved) | Pooled OOF AUC: `0.758261`
  - A/B Decision: delta `-0.000068 < 0.0002` -> keep **train-only** (`USE_TEST_FOR_STATS = False`, zero rule risk); F4 overall **REJECT**, incumbent stays `lgb_f2`.
- **Hypothesis (`lgb_f2_f5` — F5 Loan-Structure Density):** Adding 3 train-only density count features (`DENSITY_CREDIT_ANNUITY`, `DENSITY_GOODS_PRICE`, `DENSITY_CREDIT_TERM_ROUND`) captures popular retail loan templates without diluting `feature_fraction` (`45` features total). Expected effect: `+0.0002` to `+0.0008` mean AUC.
- **Result (`lgb_f2_f5`, 45 features):**
  - Fold AUCs: `[0.759308, 0.761988, 0.766030, 0.758110, 0.761511]` (mean best iter `359.2`)
  - Mean Fold AUC: **`0.761389`** (`+0.001273` vs `lgb_f2`) | Pooled OOF AUC: **`0.761348`**
  - Folds improved: `4/5` -> **KEEP** (new incumbent: `lgb_f2_f5`, active families: `["F2", "F5"]`).
- **Hypothesis (`lgb_f2_f5_f6` — F6 Nested Target Encoding):** Adding fold-isolated 5-fold inner-OOF smoothed target encodings (`alpha=50`) for `ORGANIZATION_TYPE` and `OCCUPATION_TYPE` (`47` features total) provides smoothed numeric risk priors for the 58-category `ORGANIZATION_TYPE` and 19-category `OCCUPATION_TYPE`. Expected effect: `+0.0002` to `+0.0006` mean AUC (`>= 4/5` folds).
- **Result (`lgb_f2_f5_f6`, 47 effective features):**
  - Fold AUCs: `[0.759344, 0.762811, 0.765261, 0.759736, 0.762068]` (mean best iter `367.4`)
  - Mean Fold AUC: **`0.761844`** (`+0.000455` vs `lgb_f2_f5`) | Pooled OOF AUC: **`0.761783`**
  - Folds improved: `4/5` -> **KEEP** (new incumbent: `lgb_f2_f5_f6`, active families: `["F2", "F5", "F6"]`).
- **Hypothesis (`lgb_f2_f5_f6_f1core` — Compact Core F1 on `F2+F5+F6`):** Earlier full `F1` (18 columns) diluted `feature_fraction=0.35`. Keeping only the 6 core F1 summaries (`EXT_MEAN`, `EXT_STD`, `EXT_PROD`, `EXT_NMISS`, `EXT_WMEAN`, `EXT_MEAN_X_CREDIT_TERM`) on top of `F2+F5+F6` (`51` effective features) should capture cross-bureau agreement/interaction without feature dilution. Expected effect: `+0.0005` to `+0.0020` mean AUC (`>= 4/5` folds).
- **Result (`lgb_f2_f5_f6_f1core`, 51 effective features):**
  - Fold AUCs: `[0.757660, 0.762605, 0.766332, 0.758872, 0.760707]` (mean best iter `394.4`)
  - Mean Fold AUC: `0.761235` (`-0.000609` vs `lgb_f2_f5_f6`) | Pooled OOF AUC: `0.761212`
  - Folds improved: `1/5` -> **REJECT** (even core F1 summaries hurt when trees already have raw `EXT_SOURCE_1/2/3`, `F2`, `F5`, `F6`).

#### Feature Freeze Gate (`freeze-features-v1`)
- **Frozen Families:** `["F2", "F5", "F6"]` (`USE_TEST_FOR_STATS = False`, `47` effective columns inside folds).
- **Hypothesis (`lgb_frozen_lr02`):** Lowering LightGBM learning rate from `EXP_LR=0.06` to `FINAL_LR=0.02` with `deterministic=True` and `force_row_wise=True` on the frozen feature set allows finer tree steps and reduces variance. Expected effect: `+0.0008` to `+0.0020` mean AUC over `lgb_f2_f5_f6` (`0.761844`).
- **Result (`lgb_frozen_lr02`, 47 effective features, `lr=0.02`, deterministic):**
  - Fold AUCs: `[0.760258, 0.763800, 0.767796, 0.759578, 0.763137]` (mean best iter `1131.6`)
  - Mean Fold AUC: **`0.762914`** (`+0.001070` vs `lgb_f2_f5_f6`) | Pooled OOF AUC: **`0.762892`**
  - Folds improved: `4/5` -> **KEEP**
- **Submission:** `submissions/sub_02_lgb_frozen_lr02.csv` (+ `.json`) validated against `sample_submission.csv`.
- **Gate P3:** PASSED (`freeze-features-v1` tagged, 6/6 unit tests pass).

---

### P4: Feature Pruning (`EXP_LR = 0.06`, baseline `lgb_f2_f5_f6` = `0.761844`)
- **Hypothesis (`lgb_prune_zero`):** `FLAG_MOBIL` had `0.0` gain across all 5 folds (`0` only occurs once in train and `0` times in test). Dropping `FLAG_MOBIL` (`46` effective features) should match or slightly improve AUC (`>= -0.0001`).
- **Hypothesis (`lgb_prune_bot30`):** Dropping the bottom 30% (14 lowest-gain columns: `AMT_INCOME_TOTAL`, `INCOME_PER_CHILD`, `NAME_FAMILY_STATUS`, `FLAG_WORK_PHONE`, `AMT_REQ_CREDIT_BUREAU_QRT`, `NAME_HOUSING_TYPE`, `NAME_INCOME_TYPE`, `NAME_CONTRACT_TYPE`, `AMT_REQ_CREDIT_BUREAU_MON`, `CNT_FAM_MEMBERS`, `CNT_CHILDREN`, `AMT_REQ_CREDIT_BUREAU_HOUR`, `FLAG_EMP_PHONE`, `FLAG_MOBIL`) tests whether removing low-signal features reduces `feature_fraction` noise (`33` effective features). Keep only if mean AUC delta `>= -0.0001`.

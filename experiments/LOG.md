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

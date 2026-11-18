# Experiment Log v2 — GCI Home Credit Default Risk

- **Deadline:** `2026-11-20 01:00:00+00:00`
- **v1 Incumbent Fallback:** tag `final` (`94121ab`), `competition_code.zip`, `submissions/sub_final.csv` (pooled OOF AUC `0.765177`, mean fold AUC `0.765206`, Omnicampus public score `0.76876`).

## Phase P0: Setup
- **Start UTC:** `2026-11-18T11:44:45+00:00` (**37.25 h** remaining until deadline)
- **Git branch:** `v2` branched from tag `final` (`94121ab`)
- **Hardware:** 2 physical / 4 logical cores, `N_THREADS = 3`, single heavy job at a time.
- **Environment Versions:**
  - `python==3.11.9` (`Windows-10-10.0.19045-SP0`)
  - `numpy==2.4.6`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `scipy==1.17.1`
  - `lightgbm==4.7.0`, `xgboost==3.2.0`, `catboost==1.2.10`, `optuna==5.1.0`, `pytest==9.2.0`
- **Isolation:** All v2 configs, logs, folds, experiments, and submissions live in `src/config_v2.py`, `experiments_v2/`, `submissions_v2/`, `requirements_v2.txt`, `competition_code_v2.zip`. `ENABLE_LINKAGE = False`.

- **Added dependency for P5 MLP:** `torch==2.15.0+cpu` installed cleanly from PyTorch CPU index (`requirements_v2.txt` frozen with `--extra-index-url https://download.pytorch.org/whl/cpu`).

## Phase P1: Audit v2
- **Start UTC:** `2026-11-18T11:46:37Z` (**37.22 h** left)
- **Findings (`experiments_v2/audit_v2.md`):**
  - **Customer fingerprint (`K3 = G | REG | B_ID | B_REG`):** `2.028%` train multi-row groups (`1,748` pairs), `1.933%` test-to-train match rate, `1.654%` validation-to-train match rate on `folds.npy`, chance-collision null `0.018%` (`< 1%`).
  - **Escalation Flag P1(f):** Strong label concordance among linked rows: `P(y=1 | other y=1) = 0.4151` (up to `0.4524` on `K5`) vs `P(y=1 | other y=0) = 0.0371` vs `0.0807` base rate (**5.14x lift**). `ENABLE_LINKAGE` remains `False`.
  - **Folds Decision P1(e):** Generated **`experiments_v2/folds_group.npy`** (`StratifiedGroupKFold`, 5 folds, seed 42, `K3` groups, max folds per group = `1`, target rate spread `0.000029`), `experiments_v2/folds_discovery.npy`, and `experiments_v2/folds10_group.npy`.
  - **Engineered Adversarial AUC & Count Parity Fix:** v1 frozen `F2 + F5` adversarial AUC was `0.59309` (`> 0.53`). Naive LOO (`count - 1`) yielded `0.93734` due to constant off-by-1 separation on frequent templates. Adding `F5_LOO` (leave-self-out on train + `quantize_count`: `<10` exact, `10..99` nearest 5, `>=100` nearest 50) reduces adversarial AUC to **`0.52464`** (`<= 0.53` PASS; and `0.52008` when including fold-mapped `F6_TE`).
  - **Profile:** `REGION_POPULATION_RELATIVE` has `81` unique train / `80` test values (80/80 overlap); `AMT_CREDIT == AMT_GOODS_PRICE` is `34.99%` train / `34.65%` test; `CREDIT_TERM` shows 48 clear local peaks (top mode `20.0` at `16,200` rows, `34.1` at `5,757`, `23.5` at `4,841`).
- **Commit:** P0 + P1 committed cleanly on branch `v2`.

## Phase P2: Incumbent Re-Baseline on `folds_group.npy`
- **Start UTC:** `2026-11-18T11:58:58Z` (**37.02 h** left)
- **Hypothesis (`lgb2_v1_f5raw`):** Evaluating v1's frozen feature set (`F2, F5, F6`, `FLAG_MOBIL` dropped, 46 features) on `folds_group.npy` with `LGB_START` (`EXP_LR=0.06`) establishes the group-fold baseline prior to the LOO density parity fix (expected ~`0.7615`).
- **Hypothesis (`lgb2_inc`):** Replacing `F5` with `F5_LOO` (leave-self-out train counts + `quantize_count` so adversarial AUC drops from `0.59309` to `0.52464`) removes train/test parity bias while retaining loan-template density signal (~`0.0000 +/- 0.0003` vs `lgb2_v1_f5raw`; preferred as cleaner parity-verified incumbent).
- **Result (`lgb2_v1_f5raw`, 44 features on `folds_group.npy`):**
  - Fold AUCs: `[0.761350, 0.770297, 0.754673, 0.762041, 0.756817]` (mean best iter `380.4`)
  - Mean Fold AUC: `0.761036` (`+/- 0.005388`) | Pooled OOF AUC: `0.761015`
- **Result (`lgb2_inc`, 44 features with `F5_LOO` on `folds_group.npy`):**
  - Fold AUCs: `[0.761475, 0.771278, 0.755634, 0.763283, 0.757470]` (mean best iter `359.0`)
  - Mean Fold AUC: **`0.761828`** (`+/- 0.005457`) | Pooled OOF AUC: **`0.761748`**
  - Compare `lgb2_inc` vs `lgb2_v1_f5raw`: **`+0.000792`** mean fold AUC, **`5/5`** folds improved -> **KEEP** (new incumbent: `lgb2_inc`, active families: `["F2", "F5_LOO", "F6"]`).
- **v1 `blend_final` OOF Evaluated on `folds_group.npy` (stored `experiments/blend_final/oof.npy`):**
  - On `experiments/folds.npy`: `[0.762728, 0.766599, 0.770005, 0.761713, 0.764984]`, mean `0.765206`, pooled `0.765177`.
  - On `experiments_v2/folds_group.npy`: `[0.765393, 0.774380, 0.760458, 0.765747, 0.759917]`, mean **`0.765179`**, pooled **`0.765177`**.

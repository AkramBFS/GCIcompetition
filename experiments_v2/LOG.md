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

## Phase P3: Feature Discovery (`folds_group.npy`, `EXP_LR = 0.06`, Keep Rule: `mean >= +0.0002` AND `>= 4/5` folds)
- **Start UTC:** `2026-11-18T12:05:10Z` (**~36.91 h** left)
- **Starting Incumbent:** `lgb2_inc` (`["F2", "F5_LOO", "F6"]`, `44` base cols + `2` TE cols, mean fold AUC `0.761828`)

### A0 (`region_cat`: `REGION_POPULATION_RELATIVE` 81 unique values)
- **Hypothesis (`lgb2_a0a_regcat`):** Adding `REGION_ID = round(REGION_POPULATION_RELATIVE, 6)` as a LightGBM categorical with `cat_smooth=20` captures discrete regional default risk better than numeric splits alone. Expected: `+0.0002` to `+0.0008` mean AUC.
- **Hypothesis (`lgb2_a0b_regte`):** Adding nested smoothed target encoding `TE_REGION_ID` (`alpha=50`, `min_count=30`) provides a smoothed numeric regional default prior without categorical tree overfitting. Expected: `+0.0002` to `+0.0008` mean AUC.
- **Hypothesis (`lgb2_a0c_regboth`):** Adding both `REGION_ID` (`cat_smooth=20`) and `TE_REGION_ID` combines discrete splits and smoothed prior. Expected: `+0.0002` to `+0.0008` mean AUC.
- **Result (`lgb2_a0b_regte`, 44 cols, vs `lgb2_inc`):**
  - Fold AUCs: `[0.761275, 0.770004, 0.756885, 0.760993, 0.758265]` | Mean: **`0.761484`** | Pooled: `0.761445`
  - Delta vs `lgb2_inc`: **`-0.000344`** | Folds improved: `2/5` -> **`REJECT`** (TE_REGION_ID only)
- **Result (`lgb2_a0a_regcat`, 45 cols, vs `lgb2_inc`):**
  - Fold AUCs: `[0.759861, 0.769997, 0.753610, 0.760561, 0.757687]` | Mean: **`0.760343`** | Pooled: `0.760276`
  - Delta vs `lgb2_inc`: **`-0.001485`** | Folds improved: `1/5` -> **`REJECT`** (REGION_ID cat + cat_smooth=20)
- **Result (`lgb2_a0c_regboth`, 45 cols, vs `lgb2_inc`):**
  - Fold AUCs: `[0.760783, 0.769963, 0.754065, 0.760585, 0.756429]` | Mean: **`0.760365`** | Pooled: `0.760302`
  - Delta vs `lgb2_inc`: **`-0.001463`** | Folds improved: `0/5` -> **`REJECT`** (Both REGION_ID cat & TE)
- **A0 Verdict:** All 3 variants rejected; incumbent stays `lgb2_inc`.

### A0b (`lgb_codes`: ordinal integer categorical codes)
- **Hypothesis (`lgb2_a0b_codes`):** Representing all categorical columns as ordinal integer codes (`NaN -> -1`) may reduce high-cardinality overfitting on `ORGANIZATION_TYPE`/`OCCUPATION_TYPE` (as observed in XGBoost v1). Expected: `-0.0005` to `+0.0005`.
- **Result (`lgb2_a0b_codes`, 44 cols, vs `lgb2_inc`):**
  - Fold AUCs: `[0.762555, 0.771595, 0.756114, 0.763139, 0.757934]` | Mean: **`0.762267`** | Pooled: `0.762141`
  - Delta vs `lgb2_inc`: **`+0.000439`** | Folds improved: `4/5` -> **`KEEP`**

### A3 (`KNN target mean`: nested Quantile-Normal [EXT_1..3, CREDIT_TERM])
- **Hypothesis (`lgb2_a3_knn500_v1`):** Adding nested `k=500` target mean over global 4D space (`NaN -> 0.0` normal median) captures non-axis-aligned interactions among external scores and loan term. Expected: `+0.0003` to `+0.0020`.
- **Hypothesis (`lgb2_a3_knn500_v2`):** Restricting `k=500` neighbours to identical EXT missingness patterns (fallback to V1 if `<3,000` train rows) avoids artificial median distances on missing scores. Expected: `+0.0004` to `+0.0022`.
- **Result (`lgb2_a3_knn500_v1`, 44 cols, vs `lgb2_a0b_codes`):**
  - Fold AUCs: `[0.762587, 0.770571, 0.756232, 0.761511, 0.757346]` | Mean: **`0.761649`** | Pooled: `0.761569`
  - Delta vs `lgb2_a0b_codes`: **`-0.000618`** | Folds improved: `2/5` -> **`REJECT`** (V1 global pool)
- **Result (`lgb2_a3_knn500_v2`, 44 cols, vs `lgb2_a0b_codes`):**
  - Fold AUCs: `[0.762167, 0.770479, 0.756939, 0.762321, 0.757378]` | Mean: **`0.761857`** | Pooled: `0.761713`
  - Delta vs `lgb2_a0b_codes`: **`-0.000410`** | Folds improved: `1/5` -> **`REJECT`** (V2 pattern-isolated)

### A1 (`Term/rate family G1`: 7 cols)
- **Hypothesis (`lgb2_a1_term_rate`):** Adding `TERM_MODE`, `TERM_RESID_REL`, `COST_PROXY`, `RATE_PER_PERIOD` (bisection), `DOWNPAY_PROXY`, `GOODS_EQ_CREDIT`, and `PRODUCT_CODE` exposes implied interest rate and loan product pricing. Expected: `+0.0003` to `+0.0015`.
- **Result (`lgb2_a1_term_rate`, 51 cols, vs `lgb2_a0b_codes`):**
  - Fold AUCs: `[0.764257, 0.773159, 0.757392, 0.763815, 0.758712]` | Mean: **`0.763467`** | Pooled: `0.763360`
  - Delta vs `lgb2_a0b_codes`: **`+0.001199`** | Folds improved: `5/5` -> **`KEEP`**

### A5 (`div_ext`: 18 ratios of amounts/age/emp by EXT scores, Rule A.3)
- **Hypothesis (`lgb2_a5_divext`):** Dividing `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`, `AMT_GOODS_PRICE`, `AGE_YEARS`, `EMPLOYED_MAG` by `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_MEAN` directly highlights high-debt / low-external-score risk (`>= +0.0002`, `>= 4/5` folds at ff=0.35 or ff=0.60).
- **Result (`lgb2_a5_divext_ff35`, 69 cols, vs `lgb2_a1_term_rate`):**
  - Fold AUCs: `[0.763767, 0.770606, 0.757010, 0.762795, 0.758476]` | Mean: **`0.762531`** | Pooled: `0.762485`
  - Delta vs `lgb2_a1_term_rate`: **`-0.000936`** | Folds improved: `0/5` -> **`REJECT`** (Rule A.3 ff=0.35)
- **Result (`lgb2_a5_divext_ff60`, 69 cols, vs `lgb2_a1_term_rate_ff60`):**
  - Fold AUCs: `[0.762652, 0.771111, 0.757519, 0.763164, 0.757743]` | Mean: **`0.762438`** | Pooled: `0.762323`
  - Delta vs `lgb2_a1_term_rate_ff60`: **`-0.000653`** | Folds improved: `1/5` -> **`REJECT`** (Rule A.3 ff=0.60)
- **Rule A.3 Verdict (`A5`):** REJECTED at both ff=0.35 and ff=0.60.

### A6 (`age_int`: integer age in years)
- **Hypothesis (`lgb2_a6a_age_int` / `lgb2_a6b_age_cat`):** Adding `floor(-DAYS_BIRTH / 365)` as an integer or categorical captures discrete age-threshold underwriting rules. Expected: `+0.0001` to `+0.0005`.
- **Result (`lgb2_a6a_age_int`, 52 cols, vs `lgb2_a1_term_rate`):**
  - Fold AUCs: `[0.765004, 0.773492, 0.757723, 0.763169, 0.760951]` | Mean: **`0.764068`** | Pooled: `0.763981`
  - Delta vs `lgb2_a1_term_rate`: **`+0.000601`** | Folds improved: `4/5` -> **`KEEP`**
- **Result (`lgb2_a6b_age_cat`, 52 cols, vs `lgb2_a1_term_rate`):**
  - Fold AUCs: `[0.765004, 0.773492, 0.757723, 0.763169, 0.760951]` | Mean: **`0.764068`** | Pooled: `0.763981`
  - Delta vs `lgb2_a1_term_rate`: **`+0.000601`** | Folds improved: `4/5` -> **`KEEP`**

### A4 (`TE combos G4`: ['OCC_X_ORG', 'EDU_X_INC', 'CON_X_GEN_X_FAM', 'REGION_ID', 'PRODUCT_CODE', 'TERM_X_CON'], `alpha=50`, `min_count=30`)
- **Hypothesis (`lgb2_a4_te_combos`):** Nested smoothed target encodings over occupation-organization, education-income, contract-gender-family, region, product code, and term-contract capture interaction risk priors without exploding categorical cardinality. Expected: `+0.0002` to `+0.0010`.
- **Result (`lgb2_a4_te_combos`, 52 cols, vs `lgb2_a6a_age_int`):**
  - Fold AUCs: `[0.765856, 0.772301, 0.756038, 0.766368, 0.761524]` | Mean: **`0.764417`** | Pooled: `0.764377`
  - Delta vs `lgb2_a6a_age_int`: **`+0.000350`** | Folds improved: `3/5` -> **`REJECT`**

### Re-test `F1` (External Score Interactions) under Rule A.3 (`ff=0.35` vs `ff=0.60`)
- **Hypothesis (`lgb2_retest_f1`):** F1 failed in v1 at `ff=0.35` due to column dilution; testing at `ff=0.60` may let trees leverage EXT aggregations/diffs alongside the v2 incumbent.
- **Result (`lgb2_retest_f1_ff35`, 70 cols, vs `lgb2_a6a_age_int`):**
  - Fold AUCs: `[0.762778, 0.770716, 0.755673, 0.764141, 0.759151]` | Mean: **`0.762492`** | Pooled: `0.762375`
  - Delta vs `lgb2_a6a_age_int`: **`-0.001576`** | Folds improved: `1/5` -> **`REJECT`** (Rule A.3 ff=0.35)
- **Result (`lgb2_retest_f1_ff60`, 70 cols, vs `lgb2_a6a_age_int_ff60`):**
  - Fold AUCs: `[0.762740, 0.771920, 0.755622, 0.763547, 0.757710]` | Mean: **`0.762308`** | Pooled: `0.762166`
  - Delta vs `lgb2_a6a_age_int_ff60`: **`-0.000881`** | Folds improved: `0/5` -> **`REJECT`** (Rule A.3 ff=0.60)
- **Rule A.3 Verdict (`F1`):** REJECTED at both ff=0.35 and ff=0.60.

### Re-test `F3` (Age & Temporal Ratios) under Rule A.3 (`ff=0.35` vs `ff=0.60`)
- **Hypothesis (`lgb2_retest_f3`):** Re-evaluating F3 temporal ratios (`AGE_YEARS`, `EMPLOYED_TO_AGE`, `AGE_AT_LAST_PAYMENT`, etc.) at both `ff=0.35` and `ff=0.60` checks if higher feature fraction unlocks temporal signal.
- **Result (`lgb2_retest_f3_ff35`, 60 cols, vs `lgb2_a6a_age_int`):**
  - Fold AUCs: `[0.764047, 0.773158, 0.757422, 0.765125, 0.759978]` | Mean: **`0.763946`** | Pooled: `0.763945`
  - Delta vs `lgb2_a6a_age_int`: **`-0.000122`** | Folds improved: `1/5` -> **`REJECT`** (Rule A.3 ff=0.35)
- **Result (`lgb2_retest_f3_ff60`, 60 cols, vs `lgb2_a6a_age_int_ff60`):**
  - Fold AUCs: `[0.763405, 0.771449, 0.757996, 0.762776, 0.759484]` | Mean: **`0.763022`** | Pooled: `0.763007`
  - Delta vs `lgb2_a6a_age_int_ff60`: **`-0.000166`** | Folds improved: `2/5` -> **`REJECT`** (Rule A.3 ff=0.60)
- **Rule A.3 Verdict (`F3`):** REJECTED at both ff=0.35 and ff=0.60.

### A7 (`Screened Pairwise Ratios`: Discovery on `folds {0,1,2}`, Confirmation on `folds {3,4}`)
- **Frozen Discovery Top-15 (`std_rank <= 20`):** `['A7_RATIO_EXT_SOURCE_2_DIV_AGE_YEARS', 'A7_RATIO_EXT_SOURCE_3_DIV_CREDIT_TERM', 'A7_RATIO_EXT_SOURCE_2_DIV_AMT_ANNUITY', 'A7_RATIO_EXT_SOURCE_1_DIV_AGE_YEARS', 'A7_RATIO_EXT_SOURCE_3_DIV_AGE_YEARS', 'A7_RATIO_AMT_ANNUITY_DIV_AMT_GOODS_PRICE', 'A7_RATIO_EXT_SOURCE_2_DIV_CREDIT_TERM', 'A7_RATIO_AMT_ANNUITY_DIV_EMPLOYED_MAG', 'A7_RATIO_EXT_SOURCE_2_DIV_EXT_SOURCE_3', 'A7_RATIO_AMT_CREDIT_DIV_AMT_ANNUITY', 'A7_RATIO_AGE_YEARS_DIV_EMPLOYED_MAG', 'A7_RATIO_EXT_SOURCE_2_DIV_AMT_CREDIT', 'A7_RATIO_EXT_SOURCE_2_DIV_AMT_INCOME_TOTAL', 'A7_RATIO_EXT_SOURCE_3_DIV_AMT_ANNUITY', 'A7_RATIO_CREDIT_TERM_DIV_EMPLOYED_MAG']`
- **Hypothesis (`lgb2_a7_confirm`):** Adding these 15 frozen discovery ratios improves held-out confirmation folds 3 and 4 (`mean_diff_34 >= +0.0003` and neither fold negative; multiple-comparisons caveat: candidates were screened across 66 ratios on folds {0,1,2}).
- **A7 Confirmation (`ff=0.35` vs `lgb2_a6a_age_int`):** fold3 `+0.002936`, fold4 `-0.003177`, mean_34=`-0.000121` -> `REJECT`
- **A7 Confirmation (`ff=0.60` vs `lgb2_a6a_age_int_ff60`):** fold3 `-0.001081`, fold4 `-0.001917`, mean_34=`-0.001499` -> `REJECT`

### B1 (`Linkage family G5`: grey-zone `ENABLE_LINKAGE=False`, evaluated on BOTH `folds_group.npy` and `folds.npy`)
- **Hypothesis (`lgb2_b1_linkage`):** Leave-self-out fingerprint group size and other-row default rate (`FP_GROUP_SIZE_LOO`, `FP_N_PRIOR`, `FP_PRIOR_DEFAULT_RATE`) cannot help across `folds_group.npy` (where groups never cross train/val splits, only train/train) vs `folds.npy` (where 1.654% of val rows link to train rows). Expected: near `0.0000` on `folds_group.npy`, strong positive on `folds.npy`.
- **Result (`lgb2_b1_linkage_group`, 52 cols, vs `lgb2_a6a_age_int`):**
  - Fold AUCs: `[0.764573, 0.773498, 0.757860, 0.763727, 0.758984]` | Mean: **`0.763728`** | Pooled: `0.763577`
  - Delta vs `lgb2_a6a_age_int`: **`-0.000340`** | Folds improved: `3/5` -> **`REJECT`** (on folds_group.npy)
- **Result (`lgb2_b1_linkage_randfolds`, 52 cols, vs `lgb2_inc_randfolds`):**
  - Fold AUCs: `[0.763789, 0.766003, 0.770284, 0.764488, 0.764683]` | Mean: **`0.765849`** | Pooled: `0.765806`
  - Delta vs `lgb2_inc_randfolds`: **`+0.001428`** | Folds improved: `5/5` -> **`KEEP`** (on random folds.npy)

## Phase P4: Feature View Freeze & Gate Checks
- **Start UTC:** `2026-11-18T12:58:10.421195+00:00` (**36.03 h** left)
- **P4 `VIEW_A` `ff=0.35`:** mean fold `0.764068`, pooled `0.763981`
- **P4 `VIEW_A` `ff=0.50`:** mean fold `0.763623`, pooled `0.763543`
- **P4 `VIEW_A` `ff=0.70`:** mean fold `0.762747`, pooled `0.762699`
- **P4 Selected `feature_fraction` for `VIEW_A`:** `0.35`
- **P4 Permuted-Label Gate Test:** pooled OOF AUC = **`0.506376`** (must be in `[0.495, 0.505]`)
- **P4 Permuted-Label Diagnostic & Fixed-Iteration Gate (`lgb2_gate_permuted_y_fixed300`):**
  - Under early stopping (`early_stopping_rounds=200` on `yva_perm`), Fold 3 ran 1,111 iterations and peaked at `0.514448` (`pooled=0.506376`). Raw 32 features under the identical permutation & early stopping scored `0.506547`, confirming this is early-stopping selection bias on validation noise (1-SE of AUC on 34k rows is `0.0057`), not leakage.
  - Running the full nested `VIEW_A` pipeline with fixed `n_estimators=300` (no early-stopping peeking on `yva_perm`) yields: Fold AUCs `[0.494907, 0.495749, 0.502668, 0.508644, 0.503385]`, mean `0.501071`, pooled OOF AUC **`0.500894`** (**PASS**, within `[0.495, 0.505]`).
- **P4 Adversarial Validation Gate (`train` vs `test`):**
  - `VIEW_A` (`["F2", "F5_LOO", "F6", "A1", "A6_INT"]`, 54 effective cols): **`0.51703`** (**PASS**, `<= 0.53`).
  - `VIEW_B` (`["F0", "F1", "F2", "F3", "F5_LOO", "F6", "A1", "A5", "A6_INT"]`, 102 effective cols; `F4_LITE` excluded because LOO group means caused `0.9997` train-vs-test cardinality artifact): **`0.52580`** (**PASS**, `<= 0.53`).
  - `VIEW_C` (`VIEW_A` + `A4` TE combos `min_count=30` + `A3` KNN `k=500` `mode="v2"`, 61 effective cols): label-free adversarial AUC **`0.51703`** (**PASS**, `<= 0.53`).
- **Gate P4:** PASSED (`freeze-features-v2` tagged).

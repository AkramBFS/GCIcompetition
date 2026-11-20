# Audit v2 — GCI Home Credit Default Risk

- **Audit Timestamp (UTC):** `2026-11-18T11:46:37.186909+00:00` (**37.22 h** before deadline)
- **Base Target Rate (`train.csv`):** `0.080729` (`13821` positives / `171202` rows)

## (a) Customer-Fingerprint Audit

Date-like columns present: `DAYS_BIRTH`, `DAYS_EMPLOYED` (365,243 = anomaly -> excluded), `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`, `OWN_CAR_AGE`.

Candidate keys built from `G = CODE_GENDER`, `REG = round(REGION_POPULATION_RELATIVE, 6)`, `B_ID = DAYS_BIRTH - DAYS_ID_PUBLISH`, `B_REG = round(DAYS_BIRTH - DAYS_REGISTRATION)`, `B_EMP = round(DAYS_BIRTH - DAYS_EMPLOYED_clean)`, `B_PH = round(DAYS_BIRTH - DAYS_LAST_PHONE_CHANGE)`:

| Key | Train Dup Rate | Test->Train Match | Val->Train Match (`folds.npy`) | Null Collision (`B_ID` perm) | Pairs | `P(y=1 \| other y=1)` | `P(y=1 \| other y=0)` | Train Group Sizes (1 / 2 / 3 / 4+) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `K1_G_BID_BREG` | 0.03190 (5462) | 0.03011 | 0.02609 | 0.00628 | 2935 | **0.2500** | 0.0482 | 165740 / 2592 / 61 / 22 (max 5) |
| `K2_G_BID_BREG_BEMP` | 0.01680 (2877) | 0.01548 | 0.01381 | 0.00019 | 1446 | **0.4385** | 0.0383 | 168325 / 1431 / 5 / 0 (max 3) |
| `K3_G_REG_BID_BREG` | 0.02028 (3472) | 0.01933 | 0.01654 | 0.00018 | 1748 | **0.4151** | 0.0371 | 167730 / 1724 / 8 / 0 (max 3) |
| `K4_G_REG_BID_BREG_BEMP` | 0.01647 (2820) | 0.01527 | 0.01354 | 0.00004 | 1416 | **0.4409** | 0.0387 | 168382 / 1404 / 4 / 0 (max 3) |
| `K5_G_BID_BREG_BEMP_BPH` | 0.01513 (2590) | 0.01416 | 0.01250 | 0.00000 | 1301 | **0.4524** | 0.0372 | 168612 / 1289 / 4 / 0 (max 3) |

### Train, Test, and Train+Test Group Size Distributions

- **`K1_G_BID_BREG`**: Train `{'n_groups': 168415, 'size_1': 165740, 'size_2': 2592, 'size_3': 61, 'size_4plus': 22, 'max_size': 5, 'rows_in_multi': 5462}` | Test `{'n_groups': 61142, 'size_1': 60797, 'size_2': 334, 'size_3': 10, 'size_4plus': 1, 'max_size': 5, 'rows_in_multi': 703}` | Train+Test `{'n_groups': 227753, 'size_1': 223052, 'size_2': 4527, 'size_3': 131, 'size_4plus': 43, 'max_size': 7, 'rows_in_multi': 9650}`
- **`K2_G_BID_BREG_BEMP`**: Train `{'n_groups': 169761, 'size_1': 168325, 'size_2': 1431, 'size_3': 5, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 2877}` | Test `{'n_groups': 61320, 'size_1': 61140, 'size_2': 180, 'size_3': 0, 'size_4plus': 0, 'max_size': 2, 'rows_in_multi': 360}` | Train+Test `{'n_groups': 230134, 'size_1': 227592, 'size_2': 2517, 'size_3': 24, 'size_4plus': 1, 'max_size': 4, 'rows_in_multi': 5110}`
- **`K3_G_REG_BID_BREG`**: Train `{'n_groups': 169462, 'size_1': 167730, 'size_2': 1724, 'size_3': 8, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 3472}` | Test `{'n_groups': 61276, 'size_1': 61052, 'size_2': 224, 'size_3': 0, 'size_4plus': 0, 'max_size': 2, 'rows_in_multi': 448}` | Train+Test `{'n_groups': 229553, 'size_1': 226432, 'size_2': 3093, 'size_3': 28, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 6270}`
- **`K4_G_REG_BID_BREG_BEMP`**: Train `{'n_groups': 169790, 'size_1': 168382, 'size_2': 1404, 'size_3': 4, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 2820}` | Test `{'n_groups': 61324, 'size_1': 61148, 'size_2': 176, 'size_3': 0, 'size_4plus': 0, 'max_size': 2, 'rows_in_multi': 352}` | Train+Test `{'n_groups': 230179, 'size_1': 227679, 'size_2': 2477, 'size_3': 23, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 5023}`
- **`K5_G_BID_BREG_BEMP_BPH`**: Train `{'n_groups': 169905, 'size_1': 168612, 'size_2': 1289, 'size_3': 4, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 2590}` | Test `{'n_groups': 61339, 'size_1': 61178, 'size_2': 161, 'size_3': 0, 'size_4plus': 0, 'max_size': 2, 'rows_in_multi': 322}` | Train+Test `{'n_groups': 230376, 'size_1': 228069, 'size_2': 2288, 'size_3': 19, 'size_4plus': 0, 'max_size': 3, 'rows_in_multi': 4633}`

> [!IMPORTANT]
> **Escalation Flag P1(f) — Strong Label Concordance:** Across all keys with `< 1%` chance collision, repeat applications sharing a fingerprint exhibit **$P(y=1 \mid \text{other member } y=1) = 0.4151\text{–}0.4524$** vs **$P(y=1 \mid \text{other } y=0) = 0.0371\text{–}0.0387$** (compared to the `0.0807` global default rate — a **5.14x–5.60x lift**). Consequently, we adopt **StratifiedGroupKFold (`folds_group.npy`)** for v2 evaluation and keep any target-linkage feature strictly behind `ENABLE_LINKAGE = False`.

## (b) & (c) Engineered-Feature Adversarial Validation & Count Parity Fix

- **Raw 32 features (v1 reference):** `0.51673`
- **v1 frozen `F2 + F5` (label-free, 44 cols):** **`0.59309`** (`> 0.53` FAIL)
- **v1 frozen `F2 + F5 + F6_TE` (inner-OOF train vs full-train test artifact):** `0.99998` (because inner-OOF on train adds fold noise inside each category while test gets exact full-train category values); when evaluated properly with identical fold mapping (`val fold 0` vs `test` mapped from the 4 training folds, after `F5_LOO` fix): **`0.52008`** (`<= 0.53` PASS).
  - Top adversarial gain drivers in v1: [('DENSITY_CREDIT_ANNUITY', 129542.37464775112), ('DENSITY_GOODS_PRICE', 10831.779783772157), ('DENSITY_CREDIT_TERM_ROUND', 4100.561613587541), ('ORGANIZATION_TYPE', 4015.5750266075133), ('EXT_SOURCE_2', 1614.0268831879994)]
- **Root Cause & Parity Fix (`F5_LOO`):**
  1. In v1 `F5`, train density counts included self (`>= 1`), whereas unseen test templates received `0`.
  2. Subtracting `1` from train counts *without* coarsening large counts (`adv_naive_loo_auc = 0.93734`) causes exact off-by-1 separation on frequent templates (e.g., `C-1` vs `C` for identical `AMT_GOODS_PRICE`).
  3. Applying leave-self-out (`count - 1` on train, `count` on test) **plus deterministic count quantization** (`<10` exact, `10..99` nearest `5`, `>=100` nearest `50`) lowers 5-fold engineered-feature adversarial AUC to **`0.52464`** (`<= 0.53` PASS). Top features: [('ORGANIZATION_TYPE', 6530.970355176926), ('DENSITY_CREDIT_ANNUITY', 2274.2985642910003), ('EXT_SOURCE_3', 1948.5939162991242), ('EXT_SOURCE_2', 1856.8834782792724), ('OCCUPATION_TYPE', 1260.816798233986)]

## (d) Data Profile Addendum (Low-Cardinality Numerics, `CREDIT_TERM` Modes, Annuity Scale)

### Numeric Columns with <= 500 Unique Values

| Column | Train Unique | Test Unique | Overlap |
| --- | ---: | ---: | ---: |
| `CNT_CHILDREN` | 13 | 12 | 10 |
| `REGION_POPULATION_RELATIVE` | 81 | 80 | 80 |
| `OWN_CAR_AGE` | 54 | 56 | 53 |
| `FLAG_MOBIL` | 2 | 1 | 1 |
| `FLAG_EMP_PHONE` | 2 | 2 | 2 |
| `FLAG_WORK_PHONE` | 2 | 2 | 2 |
| `CNT_FAM_MEMBERS` | 15 | 13 | 11 |
| `REGION_RATING_CLIENT` | 3 | 3 | 3 |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | 5 | 4 | 4 |
| `AMT_REQ_CREDIT_BUREAU_MON` | 21 | 20 | 18 |
| `AMT_REQ_CREDIT_BUREAU_QRT` | 11 | 9 | 9 |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | 23 | 20 | 19 |

- **Top Region Category Candidate:** `REGION_POPULATION_RELATIVE` (81 unique values in train, 80 in test, 80/80 overlap; strong univariate AUC `0.5350`).

- **Share `AMT_CREDIT == AMT_GOODS_PRICE`:** Train `34.9902%` | Test `34.6520%`.
- **`AMT_ANNUITY / AMT_INCOME_TOTAL` quantiles:** `{'q01': 0.03911457142857143, 'q10': 0.08, 'q50': 0.16296076923076924, 'q90': 0.3020434782608696, 'q99': 0.4839}` (annual-scale confirmed).
- **Top-20 modes of `CREDIT_TERM.round(1)`:** `{20.0: 16200, 34.1: 5757, 23.5: 4841, 19.5: 4467, 34.0: 4277, 20.7: 3971, 34.2: 3820, 12.6: 3460, 30.8: 3223, 25.1: 2948, 18.4: 2344, 10.1: 2218, 18.3: 2197, 14.9: 2124, 10.2: 2019, 20.4: 2011, 31.0: 1957, 19.6: 1954, 27.7: 1895, 13.7: 1880}`
- **Top-20 integer modes of `CREDIT_TERM.round(0)`:** `{20.0: 24960, 34.0: 14369, 21.0: 10919, 10.0: 8540, 18.0: 8478, 31.0: 7691, 15.0: 7449, 13.0: 7423, 19.0: 7402, 9.0: 6592, 24.0: 6262, 23.0: 5367, 14.0: 5345, 25.0: 5259, 26.0: 5211, 28.0: 5138, 16.0: 5013, 36.0: 3873, 17.0: 3739, 30.0: 3407}`
- **Local peaks (`count >= 500`) of `CREDIT_TERM.round(1)` (48 peaks):** `[(8.4, 1391), (8.8, 782), (9.2, 1522), (9.4, 1643), (9.7, 940), (10.1, 2218), (10.5, 646), (10.8, 731), (12.4, 970), (12.6, 3460), (12.9, 548), (13.3, 518), (13.7, 1880), (14.0, 691), (14.3, 825), (14.7, 967), (14.9, 2124), (15.3, 691), (15.6, 1199), (16.3, 1265), (16.5, 1278), (17.3, 1604), (17.8, 1363), (18.4, 2344), (18.7, 1294), (19.0, 840), (19.5, 4467), (20.0, 16200), (20.4, 2011), (20.7, 3971), (21.3, 1470), (21.5, 1359), (22.6, 1793), (23.5, 4841), (25.1, 2948), (25.7, 1039), (26.3, 1451), (26.5, 535), (27.7, 1895), (28.1, 1014), (28.7, 698), (30.1, 1817), (30.3, 690), (30.8, 3223), (31.0, 1957), (34.1, 5757), (36.4, 1255), (37.8, 1240)]`

## (e) CV Fold Decision (`folds_group.npy`)

- **Decision:** Switch from random `StratifiedKFold` (`experiments/folds.npy`) to **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`** grouped by `K3_G_REG_BID_BREG` (chance collision `0.077% < 1%`, 167,679 unique groups) saved to `experiments_v2/folds_group.npy`.
- **Group Isolation Check:** max folds spanned by any group = `1` (strictly 1).
- **Fold Sizes:** `[34240, 34241, 34241, 34239, 34241]`
- **Fold Default Rates:** `[0.080724, 0.080751, 0.080722, 0.080727, 0.080722]` (spread `0.000029`)
- **Discovery / 10-Fold Splits:** Saved `experiments_v2/folds_discovery.npy` (`folds_group in {0, 1, 2}`, shape `(102722, 2)`) and `experiments_v2/folds10_group.npy`.

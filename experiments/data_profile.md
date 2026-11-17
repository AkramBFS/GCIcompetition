# Data Profile — GCI Home Credit Default Risk

- **Train shape:** `171,202` rows x `34` columns (`SK_ID_CURR` `0..171201`, `TARGET` + `32` features)
- **Test shape:** `61,500` rows x `33` columns (`SK_ID_CURR` `171202..232701`, `32` features, no `TARGET`)
- **Sample submission:** `61,500` rows x `2` columns (`SK_ID_CURR`, `TARGET`)
- **Target balance:** `13,821` positives (`8.0729%`) / `157,381` negatives (`91.9271%`)

## 1. Column-by-Column Schema & Missingness

*Note: pandas 3.1.1 loads text columns as dtype `str`. Univariate AUC for numeric columns is `max(auc, 1 - auc)` on non-missing training rows.*

| Column | Dtype | Train Missing % | Test Missing % | Cardinality / Range & Univariate AUC | Description |
| --- | --- | ---: | ---: | --- | --- |
| `SK_ID_CURR` | `int64` | 0.00% | 0.00% | `[0, 171201]` (excluded ID) | ID of loan in sample |
| `TARGET` | `int64` | 0.00% | N/A | `{0, 1}` (8.07% pos) | Binary default indicator |
| `NAME_CONTRACT_TYPE` | `str` | 0.00% | 0.00% | `2 / 2` (`Cash loans`, `Revolving loans`) | Cash or revolving loan |
| `CODE_GENDER` | `str` | 0.00% | 0.00% | `3 / 2` (`F`, `M`, `XNA`: 4/0) | Client gender |
| `CNT_CHILDREN` | `int64` | 0.00% | 0.00% | `[0, 14]`, AUC `0.5167` | Number of children |
| `AMT_INCOME_TOTAL` | `float64` | 0.00% | 0.00% | `[2.61e4, 1.35e7]`, AUC `0.5182` | Client total income |
| `AMT_CREDIT` | `float64` | 0.00% | 0.00% | `[4.5e4, 4.05e6]`, AUC `0.5212` | Credit amount of loan |
| `AMT_ANNUITY` | `float64` | 0.00% | 0.00% | `[1980, 1.71e5]`, AUC `0.5023` | Loan annuity |
| `AMT_GOODS_PRICE` | `float64` | 0.00% | 0.00% | `[4.5e4, 2.254e6]`, AUC `0.5359` | Price of goods for loan |
| `NAME_INCOME_TYPE` | `str` | 0.00% | 0.00% | `8 / 8` (`Working`, `Pensioner`, etc.) | Client income type |
| `NAME_EDUCATION_TYPE` | `str` | 0.00% | 0.00% | `5 / 5` (`Secondary...`, `Higher ed`, etc.) | Highest education level |
| `NAME_FAMILY_STATUS` | `str` | 0.00% | 0.00% | `6 / 5` (`Unknown`: 2/0) | Family status |
| `NAME_HOUSING_TYPE` | `str` | 0.00% | 0.00% | `6 / 6` (`House / apartment`, etc.) | Housing situation |
| `REGION_POPULATION_RELATIVE` | `float64` | 0.00% | 0.00% | `[0.00029, 0.07251]`, AUC `0.5350` | Normalized region population |
| `DAYS_BIRTH` | `int64` | 0.00% | 0.00% | `[-25229, -7673]`, AUC `0.5844` | Age in days (negative) |
| `DAYS_EMPLOYED` | `int64` | 0.00% | 0.00% | `[-17910, 365243]`, AUC `0.5256` | Days employed (`365243` = anom) |
| `DAYS_REGISTRATION` | `float64` | 0.00% | 0.00% | `[-24672, 0]`, AUC `0.5426` | Days since registration change |
| `DAYS_ID_PUBLISH` | `int64` | 0.00% | 0.00% | `[-7197, 0]`, AUC `0.5568` | Days since ID document change |
| `OWN_CAR_AGE` | `float64` | 66.00% | 66.52% | `[0, 91]`, AUC `0.5595` | Car age in years (`>=60` outlier) |
| `FLAG_MOBIL` | `int64` | 0.00% | 0.00% | `[0, 1]` (only 1 zero in train), AUC `0.5000` | Provided mobile phone |
| `FLAG_EMP_PHONE` | `int64` | 0.00% | 0.00% | `[0, 1]`, AUC `0.5322` | Provided work phone |
| `FLAG_WORK_PHONE` | `int64` | 0.00% | 0.00% | `[0, 1]`, AUC `0.5214` | Provided home/work phone |
| `OCCUPATION_TYPE` | `str` | 0.00% | 0.00% | `19 / 19` (uses `'Other'` for 53,745/18,734) | Client occupation |
| `CNT_FAM_MEMBERS` | `float64` | 0.0012% (2) | 0.00% | `[1, 16]`, AUC `0.5062` | Number of family members |
| `REGION_RATING_CLIENT` | `int64` | 0.00% | 0.00% | `[1, 3]`, AUC `0.5481` | Region rating (1, 2, 3) |
| `ORGANIZATION_TYPE` | `str` | 0.00% | 0.00% | `58 / 58` (`XNA`: 30,898/10,565) | Employer organization type |
| `EXT_SOURCE_1` | `float64` | 69.47% | 69.78% | `[0.01709, 0.9476]`, AUC `0.6647` | External normalized score 1 |
| `EXT_SOURCE_2` | `float64` | 0.22% | 0.21% | `[8.17e-8, 0.855]`, AUC `0.6582` | External normalized score 2 |
| `EXT_SOURCE_3` | `float64` | 31.88% | 32.02% | `[0.000527, 0.896]`, AUC `0.6796` | External normalized score 3 |
| `DAYS_LAST_PHONE_CHANGE` | `float64` | 0.00% | 0.00% | `[-4173, 0]`, AUC `0.5565` | Days since last phone change |
| `AMT_REQ_CREDIT_BUREAU_HOUR` | `float64` | 13.50% | 13.84% | `[0, 4]`, AUC `0.5001` | Bureau inquiries 1h prior |
| `AMT_REQ_CREDIT_BUREAU_MON` | `float64` | 13.50% | 13.84% | `[0, 23]`, AUC `0.5060` | Bureau inquiries 1m prior |
| `AMT_REQ_CREDIT_BUREAU_QRT` | `float64` | 13.50% | 13.84% | `[0, 261]`, AUC `0.5070` | Bureau inquiries 3m prior |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | `float64` | 13.50% | 13.84% | `[0, 23]`, AUC `0.5205` | Bureau inquiries 1y prior |

## 2. Specific Diagnostic Findings (P1.2 & P1.3)

### 2.1 Co-Missingness of `AMT_REQ_CREDIT_BUREAU_*`
- The 4 present bureau columns (`HOUR`, `MON`, `QRT`, `YEAR`) are **100% co-missing**:
  - Train: `23,116` rows (`13.5022%`) have all 4 missing (`all_na == any_na == 23,116`).
  - Test: `8,513` rows (`13.8423%`) have all 4 missing (`all_na == any_na == 8,513`).
- Because they are perfectly co-missing, F0 adds `BUREAU_MISSING` only (no redundant `BUREAU_MISSING_COUNT`).

### 2.2 Missing Planned Formula Inputs
- In the full Kaggle Home Credit schema, 6 `AMT_REQ_CREDIT_BUREAU_*` columns exist (`HOUR`, `DAY`, `WEEK`, `MON`, `QRT`, `YEAR`).
- In this GCI subset, **`AMT_REQ_CREDIT_BUREAU_DAY`** and **`AMT_REQ_CREDIT_BUREAU_WEEK`** do not exist. `src/features.py` safely skips missing bureau inputs and logs them. All other F0–F6 inputs exist.

### 2.3 Scale of `AMT_ANNUITY` (`AMT_ANNUITY / AMT_INCOME_TOTAL`)
- Quantiles on `train.csv`:
  - `1%`: `0.0391`
  - `10%`: `0.0800`
  - `25%`: `0.1148`
  - `50%` (median): **`0.1630`**
  - `75%`: `0.2293`
  - `90%`: `0.3020`
  - `99%`: `0.4839`
  - Share `> 1.0`: `0.000117` (`0.0117%`)
- **Conclusion:** `AMT_ANNUITY` is **annual-scale** (median ~0.163, rarely > 1). Both `INCOME_MINUS_ANNUITY` (F2) and `AGE_AT_LAST_PAYMENT = AGE_YEARS + CREDIT_TERM` (F3) are valid and retained.

### 2.4 Placeholder & Anomaly Counts
- `DAYS_EMPLOYED == 365243`: **`30,898`** in train (`18.05%`), **`10,565`** in test (`17.18%`). No other `DAYS_*` column has any positive value (`DAYS_BIRTH`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE` are all `<= 0`).
- `OWN_CAR_AGE >= 60`: **`1,826`** in train (`64.0`: `1,331`, `65.0`: `494`, `91.0`: `1`), **`673`** in test.
- `ORGANIZATION_TYPE == "XNA"`: **`30,898`** in train, **`10,565`** in test (corresponds 1-to-1 with `DAYS_EMPLOYED == 365243`). Replaced with `NaN` in F0.
- `CODE_GENDER == "XNA"`: **`4`** in train, **`0`** in test. Replaced with `NaN` in F0.
- `NAME_FAMILY_STATUS == "Unknown"`: **`2`** in train, **`0`** in test. Replaced with `NaN` in F0.
- `OCCUPATION_TYPE == "Other"` (`53,745` train / `18,734` test) and `ORGANIZATION_TYPE == "Other"` (`9,272` train / `3,414` test) are legitimate large categories and retained.

### 2.5 Audit Decisions
- **Duplicates:** `0` exact duplicate feature rows within `train.csv` or `test.csv`; `0` `SK_ID_CURR` overlap. No rows need to be dropped.
- **Same-person heuristic:** `(CODE_GENDER, DAYS_BIRTH, DAYS_ID_PUBLISH, DAYS_REGISTRATION, NAME_EDUCATION_TYPE)` has within-train duplicate rate `0.00235` vs test-to-train hit rate `0.00203` (comparable). Standard 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` is appropriate and fixed in `experiments/folds.npy`.
- **Distribution shift & leakage:** 5-fold adversarial AUC is `0.51673` (< 0.55) and maximum single-column AUC is `0.67957` (`EXT_SOURCE_3` < 0.70). `SK_ID_CURR` is excluded from all models. Zero red flags.

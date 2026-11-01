# GCI World (Matsuo-Iwasawa Lab, UTokyo) — In-Class Competition Master Context & Technical Specification

> **SYSTEM PROMPT / AI CONTEXT INSTRUCTION:**  
> You are assisting a participant in the **GCI World In-Class Machine Learning Competition** hosted by the Matsuo-Iwasawa Lab at the University of Tokyo. Your goal is to help the user achieve a **Top 1% / #1 Leaderboard** ranking (at minimum Top 10–20% for Honors/Outstanding completion) while strictly adhering to all competition rules, file schemas, and anti-leakage constraints detailed below.

---

## 1. Competition Overview & Administrative Rules

* **Competition Problem:** A modified, streamlined version of the Kaggle **"Home Credit Default Risk"** competition (originally 2018).
* **Objective:** Predict the probability (`TARGET` $\in [0, 1]$) that a loan applicant will default on their loan based on customer attributes.
* **Evaluation Metric:** **ROC-AUC** (`sklearn.metrics.roc_auc_score`).
* **Submission Platform:** **Omnicampus** (GCI World portal).
* **Deadlines:**
  * **Submissions Open:** October 2, 11:00 AM UTC
  * **Submission Deadline:** **November 20, 1:00 AM UTC** *(Strict deadline; no extensions).*
* **Leaderboard & Final Evaluation Mechanics (CRITICAL):**
  * **Public vs. Private Leaderboard:** During the competition, the visible Public Leaderboard is evaluated on only a **subset** of the test dataset. Final rankings are determined after the deadline on a separate **Private Leaderboard** using the entire test dataset.
  * **Last Submission Rule:** **Your LAST submission on Omnicampus is the one that counts** for the final evaluation—NOT your highest-scoring submission. You must re-submit your best `submission.csv` file before the deadline.
* **Course Completion & Honors/Outstanding Requirements:**
  * **Standard Completion:** 7+ Attendance Surveys (out of 14), 14+ Homework points (out of 24 max), Passing Final Assignment, Competition is *optional*.
  * **Honors / Outstanding Student Selection:** 7+ Attendance Surveys, 14+ Homework points, **Top 10% on Final Assignment**, AND **Top 20% on the Competition Private Leaderboard**.
  * **Code Audit Requirement:** Students aiming for Honors/Outstanding **MUST submit their source code** as a `.zip` file (`tutorial.zip` or equivalent) to the `"Competition (.zip)"` assignment slot on Omnicampus. The grading team runs the code internally to verify reproducibility and check for rule violations.
* **Strict Rules & Prohibitions:**
  * **NO External Data:** Using external datasets or pre-trained weights from full Kaggle Home Credit tables not provided in the `input/` folder is strictly prohibited.
  * **NO Data Leakage / Leaderboard Probing:** Scores near `1.0` indicate target leakage or rule violations.
  * **Private Repositories Only:** Local IDEs (VS Code, PyCharm) and Git repositories are permitted, but repositories **MUST be kept strictly private**.
  * **Allowed Models:** Any machine learning model or technique (LightGBM, XGBoost, CatBoost, AdaBoost, Random Forest, Logistic Regression, Neural Networks/Deep Learning, TabPFN, Stacking/Blending Ensembles, Optuna) is permitted.

---

## 2. Directory Structure, Dataset Schema & Data Quirks

### 2.1 Expected File & Directory Structure
```text
competition/
├── input/
│   ├── train.csv                  # Training dataset (171,202 rows, 34 columns)
│   ├── test.csv                   # Test dataset (33 columns; excludes TARGET)
│   └── sample_submission.csv      # Submission template (SK_ID_CURR, TARGET)
├── output/
│   └── submission.csv             # Generated prediction output (MUST match sample_submission schema)
├── HomeCredit_columns_description.xlsx
├── README.ipynb
└── tutorial.ipynb                 # (or custom notebook/script compressed into tutorial.zip)
```

### 2.2 Dataset Specifications
* **Training Data (`train.csv`):** `171,202` rows $\times$ `34` columns:
  * `SK_ID_CURR`: Unique applicant ID (integer).
  * `TARGET`: Binary target (`1` = client had payment difficulties / defaulted, `0` = all other cases). Highly imbalanced/skewed toward `0`.
  * **32 Explanatory Features:** Mix of numerical (`int64`, `float64`) and categorical (`object`/string) columns.
* **Test Data (`test.csv`):** Contains `SK_ID_CURR` and the exact same **32 explanatory features** (`TARGET` omitted).

### 2.3 Known Data Anomalies, Missing Values & Quirks
1. **Missing Values (`NaN`) Identified in Tutorials:**
   * `OWN_CAR_AGE`: Massive missingness (`112,992` missing out of `171,202` rows in `train.csv` — represents applicants who do not own a car or did not report age).
   * `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`: Normalized external credit bureau risk scores between `0` and `1`. Contains missing values (`EXT_SOURCE_1` has high missingness; `2` and `3` have fewer).
   * `CNT_FAM_MEMBERS`: Small number of missing values.
   * `AMT_REQ_CREDIT_BUREAU_*` (`HOUR`, `DAY`, `WEEK`, `MON`, `QRT`, `YEAR`): Number of inquiries to Credit Bureau about the client before application. Missing values represent `0` inquiries.
2. **Known Outliers & Hidden Missing Values:**
   * `ORGANIZATION_TYPE`: Contains string category `"XNA"` (second most frequent category), which represents a **missing value** (often associated with pensioners/unemployed).
   * `OWN_CAR_AGE`: Natural distribution is `0` to `40` years, but there is an unnatural spike around `60` to `70` years. Values $\ge 60$ are treated as **outliers/missing (`np.nan`)**.
   * `AMT_INCOME_TOTAL`: Heavily right-skewed with extreme high-income outliers and compiled from rounded/discrete numbers (only `1,641` unique values across `171,202` rows).
   * `DAYS_EMPLOYED` *(Classic Home Credit Quirk — verify in EDA)*: In Home Credit data, `DAYS_EMPLOYED` often contains a massive positive outlier (`365243` days $\approx 1000$ years) for pensioners/unemployed, while normal values are negative days relative to application. Always inspect `DAYS_*` columns (`DAYS_BIRTH`, `DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`) for positive placeholder anomalies.
3. **Unhandled Categorical Columns in Official Tutorials (Major Opportunity):**
   * The official tutorials manually encode `NAME_CONTRACT_TYPE` and `ORGANIZATION_TYPE`, one-hot encode `OWN_CAR_AGE`, and then **drop all remaining string/categorical columns** via `.select_dtypes(include=['number'])` in Section 6.2 (e.g., `CODE_GENDER` ["F"/"M"], `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_TYPE_SUITE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `WEEKDAY_APPR_PROCESS_START`, etc.). Properly encoding these dropped categorical columns is one of the easiest ways to beat the tutorial benchmarks.

---

## 3. Official Progression & Benchmark Scores

The course provides a 5-step iterative progression across official notebooks and Office Hours:

1. **Step 1 (`tutorial.ipynb` — Baseline):** Uses only 5 manual features (`NAME_CONTRACT_TYPE`, `AMT_INCOME_TOTAL`, `EXT_SOURCE_2`, `OWN_CAR_AGE`, `ORGANIZATION_TYPE`). Public LB Score $\approx 0.66819$.
2. **Step 2 (`tutorial_adding_features.ipynb` — Day 1 Office Hour by Yui):** Expands scope to all shared numeric columns (`num_cols_train.intersection(num_cols_test)`).
   * **Holdout Validation AUC (70/30 split, `random_state=0`):** `0.732457`
3. **Step 5 (`tutorial_feature_engineering.ipynb` — Day 2 Office Hour by Minh):** Adds 4 domain features (`LOG_AMT_INCOME_TOTAL`, `EXT_MEAN`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`).
   * **Holdout Validation AUC:** `0.733938` (`+0.001481` over Step 2)
4. **Step 3 (`tutorial_model_ensembling.ipynb` — Advanced Notebook):** Trains 4 Boosting models on the engineered feature set (`70/30` holdout split, `random_state=0`):
   * **AdaBoost Valid AUC:** `0.721236` *(Over-relies on `EXT_MEAN`)*
   * **LightGBM Valid AUC:** `0.744335`
   * **CatBoost Valid AUC:** `0.750127`
   * **XGBoost Valid AUC:** `0.752677`
   * **Equal-Weight Ensemble (`[0.25, 0.25, 0.25, 0.25]`) Valid AUC:** `0.751853`
   * **Weighted Ensemble (`[0.0, 0.6, 0.1, 0.3]` for `[Ada, XGB, LGBM, Cat]`) Valid AUC:** **`0.753147`** (`+0.020690` gain over Step 2 baseline)
5. **Step 4 (`tutorial_hyperparameter_tuning.ipynb` — Day 3 Office Hour by Ibtissam):** Uses **Optuna + CatBoost** with `iterations=4000`, `early_stopping_rounds=200`, `use_best_model=True`, and `n_trials=15` ($\approx 30$ minutes runtime) tuning 8 hyperparameters: `depth`, `learning_rate`, `l2_leaf_reg`, `rsm`, `random_strength`, `border_count`, `leaf_estimation_iterations`, and `bootstrap_type`.

---

## 4. Reference Code: Official Advanced Tutorial Pipeline

Below is the consolidated Python pipeline from `tutorial_model_ensembling.ipynb`, including the exact preprocessing, feature engineering, model configurations, submission saving, and `.zip` packaging required by Omnicampus.

```python
import os
import zipfile
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ==========================================
# 1. Setup & Data Loading
# ==========================================
current_dir = Path(os.getcwd())
train_file = current_dir / "input" / "train.csv"
test_file = current_dir / "input" / "test.csv"
sample_sub_file = current_dir / "input" / "sample_submission.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
sample_sub = pd.read_csv(sample_sub_file)

# ==========================================
# 2. Preprocessing (Official Tutorial Baseline)
# ==========================================
# Label Encoding NAME_CONTRACT_TYPE
train["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}, inplace=True)
test["NAME_CONTRACT_TYPE"].replace({'Cash loans': 0, 'Revolving loans': 1}, inplace=True)

# Count Encoding ORGANIZATION_TYPE (using train frequencies to prevent leakage)
organization_ce = train["ORGANIZATION_TYPE"].value_counts()
train["ORGANIZATION_TYPE"] = train["ORGANIZATION_TYPE"].map(organization_ce)
test["ORGANIZATION_TYPE"] = test["ORGANIZATION_TYPE"].map(organization_ce)

# Impute EXT_SOURCE_1/2/3 & CNT_FAM_MEMBERS with Train Mean
for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CNT_FAM_MEMBERS"]:
    mean_val = train[col].mean()
    train[col].fillna(mean_val, inplace=True)
    test[col].fillna(mean_val, inplace=True)

# Fill AMT_REQ_CREDIT_BUREAU_* with 0
for col in ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
    if col in train.columns:
        train[col].fillna(0, inplace=True)
        test[col].fillna(0, inplace=True)

# Clean OWN_CAR_AGE outliers (>= 60 -> NaN) and One-Hot Encode by decade
train.loc[train["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan
test.loc[test["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan

train["OWN_CAR_AGE"] = train["OWN_CAR_AGE"] // 10
test["OWN_CAR_AGE"] = test["OWN_CAR_AGE"] // 10

train_car_age_ohe = pd.get_dummies(train["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_")
test_car_age_ohe = pd.get_dummies(test["OWN_CAR_AGE"]).add_prefix("OWN_CAR_AGE_")

train = pd.concat([train, train_car_age_ohe], axis=1)
test = pd.concat([test, test_car_age_ohe], axis=1)
train.drop('OWN_CAR_AGE', axis=1, inplace=True)
test.drop('OWN_CAR_AGE', axis=1, inplace=True)

# ==========================================
# 3. Feature Engineering (Official 4 Features)
# ==========================================
def add_effective_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    tr, te = train.copy(), test.copy()
    new_cols: List[str] = []

    def add_feature(name: str, tr_v: pd.Series, te_v: pd.Series):
        tr[name] = tr_v
        te[name] = te_v
        new_cols.append(name)

    if 'AMT_INCOME_TOTAL' in tr.columns:
        add_feature('LOG_AMT_INCOME_TOTAL', np.log1p(tr['AMT_INCOME_TOTAL']), np.log1p(te['AMT_INCOME_TOTAL']))

    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in tr.columns]
    if ext_cols:
        add_feature('EXT_MEAN', tr[ext_cols].mean(axis=1, skipna=True), te[ext_cols].mean(axis=1, skipna=True))

    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(tr.columns):
        tr_raw = tr['AMT_ANNUITY'] / tr['AMT_INCOME_TOTAL'].where(tr['AMT_INCOME_TOTAL'] > 0, np.nan)
        te_raw = te['AMT_ANNUITY'] / te['AMT_INCOME_TOTAL'].where(te['AMT_INCOME_TOTAL'] > 0, np.nan)
        add_feature('ANNUITY_INCOME_RATIO', tr_raw, te_raw)

    if {'AMT_CREDIT', 'AMT_ANNUITY'}.issubset(tr.columns):
        term_tr = tr['AMT_CREDIT'] / tr['AMT_ANNUITY'].where(tr['AMT_ANNUITY'] > 0, np.nan)
        term_te = te['AMT_CREDIT'] / te['AMT_ANNUITY'].where(te['AMT_ANNUITY'] > 0, np.nan)
        add_feature('CREDIT_TERM', term_tr, term_te)

    return tr, te, new_cols

train, test, _new_cols = add_effective_features(train, test)

# ==========================================
# 4. Feature Selection, Scaling & Split
# ==========================================
num_cols_train = train.drop(columns=['TARGET']).select_dtypes(include=['number']).columns
num_cols_test  = test.select_dtypes(include=['number']).columns
num_cols = num_cols_train.intersection(num_cols_test)

X = train[num_cols].values
y = train['TARGET'].values
X_test = test[num_cols].values

sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)
X_test_std = sc.transform(X_test)

X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.3, stratify=y, random_state=0)

# ==========================================
# 5. Model Training & Ensembling
# ==========================================
xgb = XGBClassifier(
    objective="binary:logistic", eval_metric="auc",
    n_estimators=600, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method="hist", n_jobs=-1, random_state=0
)
lgbm = LGBMClassifier(
    objective="binary", n_estimators=1000, learning_rate=0.03,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=40, class_weight="balanced", n_jobs=-1, random_state=0
)
cat = CatBoostClassifier(
    loss_function="Logloss", eval_metric="AUC",
    iterations=1000, learning_rate=0.03, depth=6,
    l2_leaf_reg=3.0, auto_class_weights="Balanced",
    random_seed=0, verbose=False, allow_writing_files=False
)

# ==========================================
# 6. Submission Generation [DO NOT CHANGE FORMAT]
# ==========================================
# Assuming pred is the 1D array of predicted probabilities for test set:
# sample_sub['TARGET'] = pred
# output_dir = current_dir / "output"
# os.makedirs(output_dir, exist_ok=True)
# sample_sub.to_csv(output_dir / "submission.csv", index=False)

# ==========================================
# 7. Code Zipping for Honors/Outstanding [REQUIRED]
# ==========================================
notebook_filename = "tutorial.ipynb"
zip_filename = "tutorial.zip"
notebook_path = current_dir / notebook_filename
zip_output_path = current_dir / zip_filename

if notebook_path.exists():
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(notebook_path, arcname=notebook_filename)
```

---

## 5. Strategic Roadmap & Flaws in the Official Baseline (How to Win)

The official tutorials deliberately leave major performance gaps for students to discover. To push your score from `~0.753` to **`0.770+`**, systematically address these exact bottlenecks:

### 5.1 Fix the Validation Strategy (Move from 70/30 Holdout to 5-Fold Stratified CV)
* **The Flaw:** The tutorial uses a single `70/30` `train_test_split`, which discards 30% of your training data and is noisy for small `+0.001` improvements.
* **The Fix:** Implement **5-Fold (or 10-Fold) `StratifiedKFold` Cross-Validation**.
  * Train each model across all 5 folds.
  * Compute Out-of-Fold (OOF) validation predictions to measure true CV ROC-AUC.
  * Average the 5 fold models' predictions on `X_test` (Fold-Averaging alone boosts test AUC by `+0.002` to `+0.004`).
  * **Golden Rule from Lecture:** Only keep a new feature or hyperparameter change if it improves **Mean CV AUC** AND improves **at least 4 out of 5 folds**.

### 5.2 Stop Dropping Categorical Columns & Stop Mean-Imputing `EXT_SOURCE_*` Before FE
* **The Flaw 1 (Dropped Categoricals):** Section 6.2 filters `select_dtypes(include=['number'])`, silently throwing away critical features like `CODE_GENDER` (women have significantly lower default rates in Home Credit), `NAME_EDUCATION_TYPE` (Higher education vs. Secondary), `NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, `FLAG_OWN_CAR`, and `FLAG_OWN_REALTY`.
  * **Fix:** Encode all categorical columns! Use Out-of-Fold Target Encoding, Frequency/Count Encoding, Native Categorical support in LightGBM (`astype('category')`) and CatBoost (`cat_features`), or One-Hot Encoding for low-cardinality columns.
* **The Flaw 2 (Premature Mean Imputation of `EXT_SOURCE`):** In Section 4, `tutorial_model_ensembling.ipynb` fills `NaN` in `EXT_SOURCE_1`, `2`, and `3` with the global mean **before** computing `EXT_MEAN` in Section 5, and before passing data to XGBoost/LightGBM/CatBoost.
  * **Fix:** Compute `EXT_MEAN`, `EXT_STD`, `EXT_MIN`, `EXT_MAX`, and `EXT_PROD` on the **raw** `EXT_SOURCE` columns (`skipna=True`) *before* any imputation! Furthermore, GBDT models (XGBoost, LightGBM, CatBoost) handle `NaN` natively and learn whether "missingness" itself carries risk signal. If you impute for linear/MLP models, create explicit binary missingness flags (`EXT_SOURCE_1_IS_NAN`, `OWN_CAR_AGE_IS_NAN`, `AMT_REQ_BUREAU_IS_NAN`) first.
* **The Flaw 3 (Overfitting in LightGBM/XGBoost):** In the tutorial, LightGBM (`num_leaves=63`, `n_estimators=1000`, no early stopping) achieves `0.7443` while XGBoost (`max_depth=5`) achieves `0.7526`. `num_leaves=63` is too complex for noisy credit data without early stopping.
  * **Fix:** Use `early_stopping_rounds=100` (or `200`) inside each CV fold and restrict LightGBM `num_leaves` to `15–31` with higher regularization (`reg_alpha`, `reg_lambda`, `colsample_bytree=0.6–0.8`). Also test removing `class_weight='balanced'` or tuning `scale_pos_weight`, as ROC-AUC is rank-based and extreme class weights can sometimes distort tree splits.

### 5.3 High-Value Domain Feature Engineering Checklist
Always apply the exact same transformation to `train` and `test` (replacing $0$ denominators with `np.nan` to avoid `inf`):

1. **`EXT_SOURCE` Interactions (Highest Signal):**
   * `EXT_MEAN`, `EXT_STD`, `EXT_MIN`, `EXT_MAX` across `['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']`.
   * `EXT_PROD` = `EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3`.
   * Pairwise products & ratios: `EXT_1 * EXT_2`, `EXT_2 * EXT_3`, `EXT_1 * EXT_3`, and weighted combinations (since `EXT_SOURCE_2` and `EXT_SOURCE_3` have fewer missing values).
2. **Financial Burden & Loan Structure Ratios:**
   * `CREDIT_TERM` = `AMT_CREDIT / AMT_ANNUITY` (Loan duration in payment periods).
   * `ANNUITY_INCOME_RATIO` = `AMT_ANNUITY / AMT_INCOME_TOTAL` (Debt-to-income service burden).
   * `CREDIT_TO_GOODS_RATIO` = `AMT_CREDIT / AMT_GOODS_PRICE` (Loan-to-value; $>1.0$ means financing insurance/fees on top of the item price, a strong default indicator).
   * `CREDIT_TO_INCOME_RATIO` = `AMT_CREDIT / AMT_INCOME_TOTAL`.
   * `GOODS_PRICE_DIFF` = `AMT_CREDIT - AMT_GOODS_PRICE`.
   * `INCOME_PER_PERSON` = `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`.
   * `DISPOSABLE_INCOME` = `AMT_INCOME_TOTAL - AMT_ANNUITY`.
3. **Demographic & Temporal Ratios (`DAYS_*` columns):**
   * Fix `DAYS_EMPLOYED == 365243` $\rightarrow$ create binary flag `DAYS_EMPLOYED_ANOM = (df['DAYS_EMPLOYED'] == 365243).astype(int)` and set `365243` to `np.nan`.
   * Convert negative days to positive years (`AGE_YEARS = -DAYS_BIRTH / 365.25`, `EMPLOYED_YEARS = -DAYS_EMPLOYED / 365.25`).
   * `EMPLOYED_TO_AGE_RATIO` = `DAYS_EMPLOYED / DAYS_BIRTH` (Fraction of life employed).
   * `CAR_TO_AGE_RATIO` = `OWN_CAR_AGE / AGE_YEARS` and `CAR_TO_EMPLOYED_RATIO` = `OWN_CAR_AGE / EMPLOYED_YEARS`.
   * `ID_PUBLISH_TO_AGE` = `DAYS_ID_PUBLISH / DAYS_BIRTH`.
4. **Groupby Aggregations (Contextual Peer Comparisons):**
   * Group by `['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']` or `['NAME_INCOME_TYPE', 'CODE_GENDER']` and compute the group `mean` and `std` of `EXT_MEAN`, `AMT_INCOME_TOTAL`, and `AMT_CREDIT`.
   * Compute relative features: `INCOME_RATIO_TO_GROUP = AMT_INCOME_TOTAL / GROUP_MEAN_INCOME` and `EXT_MEAN_DIFF_FROM_GROUP = EXT_MEAN - GROUP_MEAN_EXT`.

### 5.4 Ensembling, Optuna Tuning & Rank Averaging
1. **Model Diversity:** Train 5-Fold CV models of **LightGBM**, **CatBoost** (passing categorical columns natively via `cat_features`), **XGBoost**, and optionally a **LogisticRegression / MLP / Random Forest** pipeline on scaled/imputed features.
2. **Optuna Optimization:**
   * Step 1: Use Optuna (`TPE` sampler) to tune hyperparameters for each GBDT family using 5-Fold CV AUC as the objective.
   * Step 2: Use `scipy.optimize.minimize` or Optuna on the **Out-of-Fold (OOF)** probability matrix to find optimal blending weights $[w_1, w_2, w_3, \dots]$, or use **Rank Averaging** (`scipy.stats.rankdata(preds) / len(preds)`) before blending to eliminate probability calibration mismatches between models.
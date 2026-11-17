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

---

## Experiment Log (Hypotheses, Results, Verdicts)

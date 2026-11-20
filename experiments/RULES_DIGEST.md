# Competition Rules Digest — GCI Global 2026 Home Credit Default Risk

Sources reviewed: `README.ipynb` and `gci_competition_master_context_guideline.md`.

## 1. Timeline, Evaluation, and Submission Limits
- **Deadline:** November 20, 2026 at **01:00 UTC** (strict; no extensions).
- **Platform:** Omnicampus (GCI World portal).
- **Metric:** ROC-AUC (`sklearn.metrics.roc_auc_score`).
- **Submission Format:** CSV matching `input/sample_submission.csv` (`SK_ID_CURR,TARGET`, 61,500 rows, exact ID order, no NaN/Inf, predicted default probabilities in `[0, 1]`). Invalid formats receive score `-1`.
- **Daily / Total Submission Limit:** No daily limit stated ("You can submit as many times as you want").
- **Public vs. Private Leaderboard:** Public LB evaluates a subset of `test.csv`. Final Private LB evaluates the **entire** `test.csv`.
- **CRITICAL — Last Submission Counts:** Final evaluation uses the **LAST submitted file** on Omnicampus, not the highest Public LB submission. Always upload the `.zip` first and the final CSV (`submissions/sub_final.csv`) **LAST**.

## 2. Honors / Outstanding & Code Audit Requirements (.zip)
- **Honors Qualification:** Top 20% on Private Leaderboard + code audit verification.
- **Required Zip Upload:** Upload a `.zip` archive to the `"Competition (.zip)"` assignment slot on Omnicampus. Organizers execute the submitted code to confirm reproducibility of the final CSV and verify rule compliance.
- **Required Contents in Our `.zip` (built via `make_zip.py`):**
  - `README.md` (methodology, seeds, library versions, test-statistics note, citations)
  - `requirements.txt` (pinned Python packages)
  - `make_zip.py`
  - `src/` (`config.py`, `audit.py`, `cv_harness.py`, `features.py`, `models.py`, `run_experiment.py`, `tune_lgb.py`, `run_all.py`)
  - `tests/` (`test_pipeline.py`)
  - `experiments/` (`RULES_DIGEST.md`, `LOG.md`, `data_profile.md`, `audit_report.txt`, `tracker.csv`, `folds.npy`, experiment `meta.json` and `importance.csv`, `optuna_lgb.csv`, `lgb_top_params.json`)
  - `submissions/*.json` (sidecar metadata; CSVs excluded or regenerated via `src/run_all.py`)
  - `tutorial.ipynb` (preserved original/baseline notebook)
- **Excluded from `.zip`:** `.venv*/`, `input/*.csv` (organizers supply `input/`), `.git/`, large `experiments/*/*.npy` arrays, `experiments/logs/`, `experiments/*.db`, `submissions/*.csv`, `output/`.

## 3. Data, External Resources, and Test-Set Statistics Policy
- **No External Data:** Strictly prohibited. Only `input/train.csv` and `input/test.csv` may be used. No pretrained weights or full Kaggle Home Credit auxiliary tables.
- **No Hand-Labeling / Manual Edits:** All predictions must be generated automatically by a reproducible ML pipeline. Manual overrides or rule-based test edits are prohibited.
- **Test-Set Statistics:** Using `input/test.csv` without labels is allowed within the provided files, as both `train.csv` and `test.csv` are provided inputs. However, to ensure zero rule ambiguity and strong generalization to unseen samples, we test `use_test_for_stats=False` (train-only) vs `use_test_for_stats=True` (train+test) in F4:
  - Keep **train-only** statistics unless `train+test` improves mean fold-AUC by `>= +0.0002` across `>= 4/5` folds.
  - If transductive (`train+test`) statistics were ever kept, document it prominently in `README.md`.
- **Private Repository Rule:** Git repositories must remain strictly private.

## 4. Reproducibility & Seed Requirements
- Set fixed random seeds everywhere (`SEED = 42` default; deterministic LightGBM flags `deterministic=True`, `force_row_wise=True` in final runs).
- Pin exact library versions in `requirements.txt`.
- Provide one-command end-to-end reproduction from raw CSVs (`python src/run_all.py`).

## 5. Code-Citation Rule
- External reference code or snippets used must be explicitly cited with their original URL both in a code comment at the site of use and in `README.md`.
- Uncited external code discovered during code audit risks revocation of Honors/prizes.
- All pipeline modules in `src/` are written cleanly from scratch referencing only the official competition `tutorial.ipynb` / `gci_competition_master_context_guideline.md` and official library documentation.

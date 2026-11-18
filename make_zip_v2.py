"""Build the reproducible source archive competition_code_v2.zip for Omnicampus code audit."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "competition_code_v2.zip"

INCLUDE_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "requirements_v2.txt",
    "make_zip.py",
    "make_zip_v2.py",
    "tutorial.ipynb",
]

INCLUDE_V1_EXP_FILES = {
    "RULES_DIGEST.md",
    "LOG.md",
    "data_profile.md",
    "audit_report.txt",
    "tracker.csv",
    "folds.npy",
    "optuna_lgb.csv",
    "lgb_top_params.json",
}

INCLUDE_V2_EXP_FILES = {
    "LOG.md",
    "audit_v2.md",
    "audit_v2_summary.json",
    "tracker.csv",
    "folds_group.npy",
    "folds_discovery.npy",
    "folds10_group.npy",
    "v1_baseline_metrics.json",
    "a7_selected_15.json",
    "frozen_views_v2.json",
    "manifest.json",
    "params_cat_smooth20.json",
    "params_ff50.json",
    "params_ff60.json",
    "params_ff70.json",
}


def _pack_exp_dir(exp_dir: Path, include_files: set[str], files_to_pack: list[Path]) -> None:
    if not exp_dir.exists():
        return
    for name in sorted(include_files):
        p = exp_dir / name
        if p.exists():
            files_to_pack.append(p)
    skip_dirs = {"logs", "run_all_cache", "run_all_cache_fast", "run_all_cache_v2", "run_all_cache_v2_fast", "knn_cache"}
    for sub in sorted(exp_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith("run_all_cache") and sub.name not in skip_dirs:
            for fname in ("meta.json", "importance.csv"):
                fp = sub / fname
                if fp.exists():
                    files_to_pack.append(fp)


def build_zip_v2(output_path: Path = ZIP_PATH) -> Path:
    """Create competition_code_v2.zip excluding venvs, raw CSVs, git metadata, and large binaries."""
    files_to_pack: list[Path] = []

    for rel in INCLUDE_ROOT_FILES:
        p = ROOT / rel
        if p.exists():
            files_to_pack.append(p)

    for folder in ("src", "tests"):
        folder_path = ROOT / folder
        if folder_path.exists():
            for p in sorted(folder_path.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    files_to_pack.append(p)

    _pack_exp_dir(ROOT / "experiments", INCLUDE_V1_EXP_FILES, files_to_pack)
    _pack_exp_dir(ROOT / "experiments_v2", INCLUDE_V2_EXP_FILES, files_to_pack)

    for sub_folder in ("submissions", "submissions_v2"):
        sub_dir = ROOT / sub_folder
        if sub_dir.exists():
            for p in sorted(sub_dir.glob("*.json")):
                files_to_pack.append(p)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files_to_pack:
            arcname = fp.relative_to(ROOT).as_posix()
            zf.write(fp, arcname=arcname)

    size_kb = output_path.stat().st_size / 1024.0
    print(f"Built {output_path.name}: {len(files_to_pack)} files, {size_kb:.1f} KB")
    return output_path


if __name__ == "__main__":
    build_zip_v2()

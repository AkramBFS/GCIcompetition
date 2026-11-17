"""Build the reproducible source archive for Omnicampus Competition (.zip) submission."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "competition_code.zip"

INCLUDE_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "make_zip.py",
    "tutorial.ipynb",
]

INCLUDE_EXPERIMENT_FILES = {
    "RULES_DIGEST.md",
    "LOG.md",
    "data_profile.md",
    "audit_report.txt",
    "tracker.csv",
    "folds.npy",
    "optuna_lgb.csv",
    "lgb_top_params.json",
}


def build_zip(output_path: Path = ZIP_PATH) -> Path:
    """Create competition_code.zip excluding venvs, raw CSVs, git metadata, and large binaries."""
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

    exp_dir = ROOT / "experiments"
    if exp_dir.exists():
        for name in sorted(INCLUDE_EXPERIMENT_FILES):
            p = exp_dir / name
            if p.exists():
                files_to_pack.append(p)
        for sub in sorted(exp_dir.iterdir()):
            if sub.is_dir() and sub.name != "logs":
                for fname in ("meta.json", "importance.csv"):
                    fp = sub / fname
                    if fp.exists():
                        files_to_pack.append(fp)

    sub_dir = ROOT / "submissions"
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
    build_zip()

"""CLI to run one named cross-validation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DROPPED_FEATURES, EXP_LR, SEED
from src.cv_harness import compare, load_data, run_cv
from src.features import build_features, wrap_fit_predict_with_te
from src.models import cat_model, lgb_model, xgb_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one named GCI Home Credit CV experiment.")
    parser.add_argument("--name", required=True, help="Unique experiment name.")
    parser.add_argument(
        "--model",
        choices=["lgb", "xgb", "cat"],
        default="lgb",
        help="Model family to train.",
    )
    parser.add_argument(
        "--families",
        default="",
        help="Comma-separated feature families, e.g. F0,F1,F2 (empty for raw).",
    )
    parser.add_argument(
        "--use-test-for-stats",
        action="store_true",
        help="Include test rows when computing F4/F5 unsupervised statistics.",
    )
    parser.add_argument(
        "--use-f6",
        action="store_true",
        help="Enable nested fold-isolated target encoding (F6).",
    )
    parser.add_argument(
        "--drop-features",
        default="",
        help="Comma-separated features to drop, or 'config' to use config.DROPPED_FEATURES.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic/force_row_wise flags for LightGBM.",
    )
    parser.add_argument(
        "--xgb-codes",
        action="store_true",
        help="Use integer category codes instead of enable_categorical=True for XGBoost.",
    )
    parser.add_argument(
        "--extra-params",
        default="",
        help="JSON dict or path to JSON file of model hyperparameters.",
    )
    parser.add_argument(
        "--compare-to",
        default="",
        help="Reference experiment name to compare against.",
    )
    parser.add_argument("--min-gain", type=float, default=0.0002, help="Min mean AUC gain.")
    parser.add_argument("--min-folds", type=int, default=4, help="Min folds improved.")
    parser.add_argument("--note", default="", help="Experiment note saved to meta/tracker.")
    return parser.parse_args()


def _load_extra_params(raw: str) -> dict:
    if not raw:
        return {}
    p = Path(raw)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(raw)


def main() -> None:
    args = parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    use_f6 = args.use_f6 or ("F6" in families)
    build_fams = [f for f in families if f != "F6"]

    if args.drop_features == "config":
        dropped = list(DROPPED_FEATURES)
    elif args.drop_features:
        dropped = [c.strip() for c in args.drop_features.split(",") if c.strip()]
    else:
        dropped = []

    extra_params = _load_extra_params(args.extra_params)

    train, test, y, folds = load_data()
    X, Xte = build_features(
        train,
        test,
        families=build_fams,
        use_test_for_stats=args.use_test_for_stats,
        dropped_features=dropped,
    )
    skipped = getattr(build_features, "last_skipped_", [])
    print(
        f"Features built for {args.name}: shape={X.shape}, families={families or ['RAW']}, "
        f"use_test_for_stats={args.use_test_for_stats}, dropped={len(dropped)}, skipped={skipped}",
        flush=True,
    )

    if args.model == "lgb":
        lr = args.lr if args.lr is not None else EXP_LR
        fp = lgb_model(
            params=extra_params,
            seed=args.seed,
            lr=lr,
            deterministic=args.deterministic,
        )
    elif args.model == "xgb":
        params = dict(extra_params)
        if args.lr is not None:
            params["learning_rate"] = args.lr
        fp = xgb_model(params=params, seed=args.seed, use_codes=args.xgb_codes)
    else:
        params = dict(extra_params)
        if args.lr is not None:
            params["learning_rate"] = args.lr
        fp = cat_model(params=params, seed=args.seed)

    if use_f6:
        fp = wrap_fit_predict_with_te(fp, seed=args.seed)

    run_cv(
        name=args.name,
        fit_predict=fp,
        X=X,
        y=y,
        Xte=Xte,
        folds=folds,
        note=args.note,
    )

    if args.compare_to:
        compare(
            new=args.name,
            ref=args.compare_to,
            min_gain=args.min_gain,
            min_folds=args.min_folds,
        )


if __name__ == "__main__":
    main()

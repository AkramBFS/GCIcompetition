"""CLI to run one named v2 cross-validation experiment into experiments_v2/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config_v2 import DROPPED_FEATURES, EXP_LR, SEED
from src.cv_harness import compare_v2, load_data_v2, run_cv_v2
from src.features import build_features, wrap_fit_predict_with_te
from src.models import cat_model, lgb_model, xgb_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one named v2 CV experiment.")
    parser.add_argument("--name", required=True, help="Unique experiment name in experiments_v2/.")
    parser.add_argument(
        "--model",
        choices=["lgb", "xgb", "cat", "logreg", "mlp"],
        default="lgb",
        help="Model family to train.",
    )
    parser.add_argument(
        "--families",
        default="",
        help="Comma-separated feature families, e.g. F2,F5_LOO,F6.",
    )
    parser.add_argument("--use-test-for-stats", action="store_true")
    parser.add_argument("--use-f6", action="store_true", help="Enable nested TE (F6).")
    parser.add_argument(
        "--te-cols",
        default="",
        help="Additional comma-separated columns for nested TE (in addition to F6 cols when F6 active).",
    )
    parser.add_argument("--te-min-count", type=int, default=0, help="Min row count for TE cells.")
    parser.add_argument("--knn-k", default="", help="Comma-separated KNN k values, e.g. 500.")
    parser.add_argument("--knn-mode", choices=["v1", "v2"], default="v1")
    parser.add_argument("--use-linkage", action="store_true", help="Enable B1 nested linkage wrapper.")
    parser.add_argument("--use-codes", action="store_true", help="Ordinal integer codes for LGB/XGB.")
    parser.add_argument("--drop-features", default="config")
    parser.add_argument("--folds-path", default="", help="Optional override folds .npy path.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--extra-params", default="")
    parser.add_argument("--compare-to", default="")
    parser.add_argument("--min-gain", type=float, default=0.0002)
    parser.add_argument("--min-folds", type=int, default=4)
    parser.add_argument("--permuted-y-seed", type=int, default=None, help="Permute y once for leak test.")
    parser.add_argument("--note", default="")
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
    build_fams = [f for f in families if f not in ("F6", "B1")]

    if args.drop_features == "config":
        dropped = list(DROPPED_FEATURES)
    elif args.drop_features:
        dropped = [c.strip() for c in args.drop_features.split(",") if c.strip()]
    else:
        dropped = []

    extra_params = _load_extra_params(args.extra_params)

    train, test, y, folds = load_data_v2()
    if args.folds_path:
        folds = np.load(args.folds_path)
    if args.permuted_y_seed is not None:
        rng = np.random.default_rng(args.permuted_y_seed)
        y = rng.permutation(y)

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
        f"dropped={len(dropped)}, skipped={skipped}",
        flush=True,
    )

    if args.model == "lgb":
        lr = args.lr if args.lr is not None else EXP_LR
        fp = lgb_model(
            params=extra_params,
            seed=args.seed,
            lr=lr,
            deterministic=args.deterministic,
            use_codes=args.use_codes,
        ) if hasattr(lgb_model, "__code__") and "use_codes" in lgb_model.__code__.co_varnames else lgb_model(
            params=extra_params,
            seed=args.seed,
            lr=lr,
            deterministic=args.deterministic,
        )
    elif args.model == "xgb":
        params = dict(extra_params)
        if args.lr is not None:
            params["learning_rate"] = args.lr
        fp = xgb_model(params=params, seed=args.seed, use_codes=args.use_codes)
    elif args.model == "cat":
        params = dict(extra_params)
        if args.lr is not None:
            params["learning_rate"] = args.lr
        fp = cat_model(params=params, seed=args.seed)
    elif args.model == "logreg":
        from src.models import logreg_binned_model

        fp = logreg_binned_model(params=extra_params, seed=args.seed)
    else:
        from src.models import mlp_model

        fp = mlp_model(params=extra_params, seed=args.seed)

    # Optional nested wrappers (innermost executed closest to model, outermost produces features first)
    if use_f6 or args.te_cols:
        te_cols_list: list[str] = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"] if use_f6 else []
        if args.te_cols:
            for c in args.te_cols.split(","):
                c = c.strip()
                if c and c not in te_cols_list:
                    te_cols_list.append(c)
        fp = wrap_fit_predict_with_te(
            fp,
            cols=tuple(te_cols_list),
            alpha=50.0,
            min_count=args.te_min_count,
            seed=args.seed,
        )

    if args.knn_k:
        from src.features import wrap_fit_predict_with_knn

        ks = tuple(int(k.strip()) for k in args.knn_k.split(",") if k.strip())
        fp = wrap_fit_predict_with_knn(fp, k_values=ks, mode=args.knn_mode, seed=args.seed)

    if args.use_linkage or ("B1" in families):
        from src.features import wrap_fit_predict_with_linkage

        fp = wrap_fit_predict_with_linkage(fp, train_df=train, test_df=test)

    run_cv_v2(
        name=args.name,
        fit_predict=fp,
        X=X,
        y=y,
        Xte=Xte,
        folds=folds,
        note=args.note,
    )

    if args.compare_to:
        compare_v2(
            new=args.name,
            ref=args.compare_to,
            min_gain=args.min_gain,
            min_folds=args.min_folds,
        )


if __name__ == "__main__":
    main()

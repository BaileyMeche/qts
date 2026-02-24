from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .impact_config import ImpactConfig, require_parquet_engine
from .impact_core import collapse_multihit_trades, robustness_sweep, run_exchange_and_pooled
from .io import find_trade_files, read_trades_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporary market impact estimation pipeline")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--max-rows-per-file", type=int, default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--ts-floor", type=str, default="1ms")
    p.add_argument("--outdir", type=Path, default=Path("outputs"))
    p.add_argument("--run-robustness", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_parquet_engine()

    cfg = ImpactConfig(
        M=args.M,
        ts_floor=None if str(args.ts_floor).lower() == "none" else args.ts_floor,
        max_rows_per_file=args.max_rows_per_file,
        sample_frac=args.sample_frac,
    )

    files = find_trade_files(args.repo_root)
    if not files:
        raise FileNotFoundError(
            "No parquet trade files found. Place parquet files under repo root or set "
            "TRADE_FILES_GLOB/TRADE_FILES_LIST environment variables."
        )

    pieces = []
    for f in files:
        try:
            pieces.append(read_trades_parquet(f, cfg.max_rows_per_file, cfg.sample_frac))
        except ValueError as exc:
            raise ValueError(f"Schema mismatch in file {f}: {exc}") from exc

    trades = pd.concat(pieces, ignore_index=True)
    parent = collapse_multihit_trades(trades, cfg.ts_floor)
    results = run_exchange_and_pooled(parent, cfg)

    args.outdir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.outdir / "impact_results.parquet", index=False)
    results.to_csv(args.outdir / "impact_results.csv", index=False)

    if args.run_robustness:
        robust = robustness_sweep(parent, cfg)
        robust.to_parquet(args.outdir / "impact_robustness.parquet", index=False)
        robust.to_csv(args.outdir / "impact_robustness.csv", index=False)

    print(f"Saved outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

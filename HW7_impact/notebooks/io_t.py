from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from impact_config import require_parquet_engine


_ALIAS_MAP = {
    "trade price": "price",
    "trade_price": "price",
    "price": "price",
    "timestamp": "ts",
    "time": "ts",
    "ts": "ts",
    "quantity": "qty",
    "size": "qty",
    "qty": "qty",
    "Exchange": "exchange",
    "exchange": "exchange",
    "side": "side",
}
_REQUIRED_COLS = ["ts", "side", "qty", "price", "exchange"]


def find_trade_files(repo_root: Path) -> list[Path]:
    env_list = os.getenv("TRADE_FILES_LIST")
    if env_list:
        files = [Path(item.strip()).expanduser() for item in env_list.split(",") if item.strip()]
        return sorted({p.resolve() for p in files if p.exists()})

    env_glob = os.getenv("TRADE_FILES_GLOB")
    if env_glob:
        matches = [p for p in repo_root.glob(env_glob) if p.is_file()]
        return sorted({p.resolve() for p in matches})

    patterns = [
        "**/*SOL*USDT*trade*.parquet",
        "**/*ETH*USDT*trade*.parquet",
        "**/*trades*.parquet",
    ]
    files: set[Path] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            if path.is_file():
                files.add(path.resolve())
    return sorted(files)


def _standardize_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    rename_map = {col: _ALIAS_MAP[col] for col in df.columns if col in _ALIAS_MAP}
    std = df.rename(columns=rename_map)
    missing = [c for c in _REQUIRED_COLS if c not in std.columns]
    if missing:
        raise ValueError(f"File {path} missing required standardized columns: {missing}")
    return std[_REQUIRED_COLS].copy()


def read_trades_parquet(path: Path, max_rows: Optional[int], sample_frac: Optional[float]) -> pd.DataFrame:
    require_parquet_engine()
    df = pd.read_parquet(path)
    df = _standardize_columns(df, path)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["side"] = df["side"].astype("string")
    df["exchange"] = df["exchange"].astype("string")

    df = df.dropna(subset=_REQUIRED_COLS)
    df = df[(df["qty"] > 0) & (df["price"] > 0)]

    if max_rows is not None and max_rows > 0:
        df = df.iloc[:max_rows]

    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42)

    return df.reset_index(drop=True)


def explore_trades(df: pd.DataFrame) -> None:
    if df.empty:
        print("Empty dataframe")
        return

    ts_min = df["ts"].min()
    ts_max = df["ts"].max()
    print(f"rows={len(df):,}")
    print(f"exchanges={df['exchange'].nunique()}")
    print(f"date span={ts_min} -> {ts_max}")
    print("top exchanges:")
    print(df["exchange"].value_counts().head(10))
    print("side labels:")
    print(df["side"].value_counts())
    print("qty quantiles:")
    print(df["qty"].quantile([0.5, 0.9, 0.99, 0.9975, 0.999]))
    print("price quantiles:")
    print(df["price"].quantile([0.0, 0.5, 0.9, 0.99, 1.0]))

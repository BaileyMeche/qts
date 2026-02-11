from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_cds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"], utc=False)
    if not (df["tenor"].nunique() == 1 and df["tenor"].iloc[0] == "5Y"):
        raise ValueError("Expected only 5Y tenor")
    if not (df["currency"].nunique() == 1 and df["currency"].iloc[0] == "USD"):
        raise ValueError("Expected only USD currency")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def _download_equities(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    symbols = sorted(set(tickers) | {"SPY"})
    data = yf.download(
        tickers=symbols,
        start=(start - pd.Timedelta(days=7)).date().isoformat(),
        end=(end + pd.Timedelta(days=7)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No data returned from yfinance")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(1):
            px = data.xs("Adj Close", axis=1, level=1)
        else:
            px = data.xs("Close", axis=1, level=1)
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        px = data[[col]].rename(columns={col: symbols[0]})

    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index()


def load_or_fetch_equity_adj_close(
    cache_path: Path,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load cached equity adjusted closes, or download and cache.

    Notes
    -----
    - Prefer local cache if available.
    - If the cache is parquet but the parquet engine (pyarrow/fastparquet) is
      missing, fall back to a yfinance download (if available) rather than
      crashing.
    """

    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df.sort_index()
        except ImportError:
            # Parquet engine missing; attempt online fallback below.
            pass

    try:
        df = _download_equities(tickers=tickers, start=start, end=end)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load equity cache at {cache_path} and failed to download with yfinance."
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Cache when possible. If parquet engine is unavailable, write CSV fallback.
    try:
        df.to_parquet(cache_path)
    except ImportError:
        csv_path = cache_path.with_suffix(".csv")
        df.to_csv(csv_path)
    return df

from __future__ import annotations

import pandas as pd


def to_weekly_wed_last(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-WED").last()


def simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(float).pct_change()


def cds_wide_parspread(cds_df: pd.DataFrame) -> pd.DataFrame:
    wide = cds_df.pivot(index="date", columns="ticker", values="parspread").sort_index()
    wide.index = pd.to_datetime(wide.index).tz_localize(None)
    return wide


def align_weekly_returns(
    cds_ret: pd.DataFrame,
    eq_ret: pd.DataFrame,
    market_ret: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = cds_ret.index.intersection(eq_ret.index).intersection(market_ret.index)
    return cds_ret.loc[idx], eq_ret.loc[idx], market_ret.loc[idx]

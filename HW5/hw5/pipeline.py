from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .io import load_cds, load_or_fetch_equity_adj_close
from .rolling import (
    rolling_ols_no_intercept,
    rolling_slope_no_intercept_boxcar,
    rolling_slope_no_intercept_ew,
)
from .transforms import align_weekly_returns, cds_wide_parspread, simple_returns, to_weekly_wed_last


@dataclass
class PipelineResult:
    panel: pd.DataFrame
    cds_weekly_ret: pd.DataFrame
    eq_weekly_ret: pd.DataFrame
    market_weekly_ret: pd.Series


def compute_panel(
    cds_ret: pd.DataFrame,
    eq_ret: pd.DataFrame,
    market_ret: pd.Series,
    boxcar_window: int,
    ew_window: int,
    ew_half_life: float,
) -> pd.DataFrame:
    index_ret = cds_ret.mean(axis=1, skipna=True).rename("r_index")
    rows: list[pd.DataFrame] = []

    for ticker in cds_ret.columns:
        y_eq = eq_ret[ticker]
        y_cds = cds_ret[ticker]
        X_capm = pd.DataFrame({"m": market_ret})
        gamma = rolling_ols_no_intercept(y_eq, X_capm, window=boxcar_window)["m"].rename("gamma")

        X_cds = pd.DataFrame({"equity": y_eq, "index": index_ret})
        betas = rolling_ols_no_intercept(y_cds, X_cds, window=boxcar_window)

        f = betas["equity"] * y_eq + betas["index"] * index_ret
        rho = y_cds - f
        c = y_eq - gamma * market_ret

        mu_box = rolling_slope_no_intercept_boxcar(rho, c.shift(1), window=ew_window)
        mu_ew = rolling_slope_no_intercept_ew(rho, c.shift(1), window=ew_window, half_life=ew_half_life)

        pred_box = mu_box.shift(1) * c.shift(1)
        pred_ew = mu_ew.shift(1) * c.shift(1)
        q_box = rho - pred_box
        q_ew = rho - pred_ew

        tdf = pd.DataFrame(
            {
                "date": cds_ret.index,
                "ticker": ticker,
                "r_cds": y_cds,
                "r_equity": y_eq,
                "m": market_ret,
                "r_index": index_ret,
                "gamma": gamma,
                "beta_equity": betas["equity"],
                "beta_index": betas["index"],
                "f": f,
                "rho": rho,
                "c": c,
                "mu_boxcar": mu_box,
                "mu_ew": mu_ew,
                "pred_boxcar": pred_box,
                "pred_ew": pred_ew,
                "q_boxcar": q_box,
                "q_ew": q_ew,
            }
        )
        rows.append(tdf)
    return pd.concat(rows, axis=0, ignore_index=True)


def run_pipeline(config: Config) -> PipelineResult:
    cds_df = load_cds(config.cds_path)
    tickers = sorted(cds_df["ticker"].unique().tolist())
    cds_px = cds_wide_parspread(cds_df)

    eq_px = load_or_fetch_equity_adj_close(
        config.equity_cache_path,
        tickers=tickers,
        start=cds_px.index.min(),
        end=cds_px.index.max(),
    )
    if "SPY" not in eq_px.columns:
        raise ValueError("Equity adjusted close data must include SPY")

    cds_w = to_weekly_wed_last(cds_px)
    eq_w = to_weekly_wed_last(eq_px[tickers + ["SPY"]])

    cds_ret = simple_returns(cds_w)
    eq_ret = simple_returns(eq_w[tickers])
    m_ret = simple_returns(eq_w[["SPY"]])["SPY"].rename("m")

    cds_ret, eq_ret, m_ret = align_weekly_returns(cds_ret, eq_ret, m_ret)

    panel = compute_panel(
        cds_ret=cds_ret,
        eq_ret=eq_ret,
        market_ret=m_ret,
        boxcar_window=config.boxcar_window,
        ew_window=config.ew_window,
        ew_half_life=config.ew_half_life,
    )
    return PipelineResult(panel=panel, cds_weekly_ret=cds_ret, eq_weekly_ret=eq_ret, market_weekly_ret=m_ret)

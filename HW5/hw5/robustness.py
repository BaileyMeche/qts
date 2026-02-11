from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .metrics import pooled_metrics
from .pipeline import compute_panel


def run_robustness(
    config: Config,
    cds_ret: pd.DataFrame,
    eq_ret: pd.DataFrame,
    market_ret: pd.Series,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = compute_panel(
        cds_ret=cds_ret,
        eq_ret=eq_ret,
        market_ret=market_ret,
        boxcar_window=config.boxcar_window,
        ew_window=config.ew_window,
        ew_half_life=config.ew_half_life,
    )
    baseline_pool = pooled_metrics(baseline)
    base_ew_rmse = float(baseline_pool.loc[baseline_pool["model"] == "ew", "rmse"].iloc[0])
    base_ew_r2 = float(baseline_pool.loc[baseline_pool["model"] == "ew", "oos_r2"].iloc[0])

    rows = []
    for window in config.robustness_windows:
        base_panel_w = compute_panel(
            cds_ret=cds_ret,
            eq_ret=eq_ret,
            market_ret=market_ret,
            boxcar_window=config.boxcar_window,
            ew_window=window,
            ew_half_life=config.ew_half_life,
        )
        base_ticker_delta = (
            base_panel_w.groupby("ticker")[["q_ew", "q_boxcar"]].apply(lambda g: (g["q_ew"].pow(2).mean() ** 0.5) - (g["q_boxcar"].pow(2).mean() ** 0.5))
        )

        for half_life in config.robustness_half_lives:
            panel = compute_panel(
                cds_ret=cds_ret,
                eq_ret=eq_ret,
                market_ret=market_ret,
                boxcar_window=config.boxcar_window,
                ew_window=window,
                ew_half_life=half_life,
            )
            pool = pooled_metrics(panel)
            ew_row = pool.loc[pool["model"] == "ew"].iloc[0]
            ticker_delta = (
                panel.groupby("ticker")[["q_ew", "q_boxcar"]].apply(lambda g: (g["q_ew"].pow(2).mean() ** 0.5) - (g["q_boxcar"].pow(2).mean() ** 0.5))
            )
            stab = ticker_delta.rank().corr(base_ticker_delta.rank(), method="spearman")
            rows.append(
                {
                    "window": window,
                    "half_life": half_life,
                    "ew_rmse": float(ew_row["rmse"]),
                    "ew_oos_r2": float(ew_row["oos_r2"]),
                    "delta_rmse_vs_baseline": float(ew_row["rmse"] - base_ew_rmse),
                    "delta_r2_vs_baseline": float(ew_row["oos_r2"] - base_ew_r2),
                    "stability_rank_corr": float(stab),
                }
            )

    table = pd.DataFrame(rows).sort_values(["window", "half_life"]).reset_index(drop=True)
    table.to_csv(out_dir / "robustness_sweep.csv", index=False)
    return table

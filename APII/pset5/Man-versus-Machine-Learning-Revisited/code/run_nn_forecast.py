#!/usr/bin/env python3
"""
Run rolling neural-network (MLP) earnings forecasts and evaluate against the RF baseline.

Usage (from repo_root/code):
  python run_nn_forecast.py --mode woLAB
  python run_nn_forecast.py --mode wLAB
  python run_nn_forecast.py --mode woLAB --evaluate

Assumes that the replication has already produced:
  ../data/Results/df_train_new.parquet
and (for evaluation vs RF):
  ../data/Results/RF_wo_lookahead_raw_005.parquet  (or with_lookahead)

Outputs:
  ../data/Results/NN_wo_lookahead_raw.parquet  (or NN_with_lookahead_raw.parquet)
  ../data/Results/NN_vs_RF_MSE_<mode>.csv      (if --evaluate)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from functions.nn_forecast import MLPSpec, MLPParams, run_mlp_forecasts, summarize_mse_comparison


def build_specs(mode: str):
    """
    Match x_cols construction in 02_EarningsForecasts.ipynb.

    mode:
      - "wLAB": with look-ahead bias (uses EPS_true_l1_q2 etc.)
      - "woLAB": without look-ahead bias (replaces certain lagged EPS_true_l1_* as in notebook)
    """
    # ratio_chars are defined in preprocessing; in df_train_new they should already exist.
    # We reference them by name; if you change preprocessing, update this list.
    ratio_chars = ['CAPEI', 'bm', 'evm', 'pe_exi', 'pe_inc', 'ps', 'pcf', 'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa', 'roe', 'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity', 'pretret_noa', 'pretret_earnat', 'GProf', 'equity_invcap', 'debt_invcap', 'totdebt_invcap', 'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt', 'invt_act', 'rect_act', 'debt_at', 'debt_ebitda', 'short_debt', 'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf', 'lt_ppent', 'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio', 'intcov', 'intcov_ratio', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', 'inv_turn', 'at_turn', 'rect_turn', 'pay_turn', 'sale_invcap', 'sale_equity', 'sale_nwc', 'rd_sale', 'adv_sale', 'staff_sale', 'accrual', 'ptb', 'PEG_trailing', 'divyield']

    macro = ["RGDP", "RCON", "INDPROD", "UNEMP"]

    # Helper to choose lagged EPS_true columns under woLAB vs wLAB
    def lag(col_wlab: str, col_wolab: str) -> str:
        return col_wlab if mode == "wLAB" else col_wolab

    specs = [
        MLPSpec(
            horizon="q1",
            y_col="EPS_true_q1",
            af_col="EPS_ana_q1",
            ann_date_col="ANNDATS_q1",
            x_cols=ratio_chars + ["ret", "prc", "EPS_true_l1_q1", "EPS_ana_q1"] + macro
        ),
        MLPSpec(
            horizon="q2",
            y_col="EPS_true_q2",
            af_col="EPS_ana_q2",
            ann_date_col="ANNDATS_q2",
            x_cols=ratio_chars + ["ret", "prc", lag("EPS_true_l1_q2", "EPS_true_l1_q1"), "EPS_ana_q2"] + macro
        ),
        MLPSpec(
            horizon="q3",
            y_col="EPS_true_q3",
            af_col="EPS_ana_q3",
            ann_date_col="ANNDATS_q3",
            x_cols=ratio_chars + ["ret", "prc", lag("EPS_true_l1_q3", "EPS_true_l1_q1"), "EPS_ana_q3"] + macro
        ),
        MLPSpec(
            horizon="y1",
            y_col="EPS_true_y1",
            af_col="EPS_ana_y1",
            ann_date_col="ANNDATS_y1",
            x_cols=ratio_chars + ["ret", "prc", "EPS_true_l1_y1", "EPS_ana_y1"] + macro
        ),
        MLPSpec(
            horizon="y2",
            y_col="EPS_true_y2",
            af_col="EPS_ana_y2",
            ann_date_col="ANNDATS_y2",
            x_cols=ratio_chars + ["ret", "prc", lag("EPS_true_l1_y2", "EPS_true_l1_y1"), "EPS_ana_y2"] + macro
        ),
    ]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wLAB", "woLAB"], default="woLAB",
                    help="Match the RF benchmark's look-ahead-bias variant.")
    ap.add_argument("--train_window_months", type=int, default=12)
    ap.add_argument("--start_date", type=str, default="1986-01-31")
    ap.add_argument("--evaluate", action="store_true",
                    help="Also compute MSE summary vs RF and analyst forecasts.")
    ap.add_argument("--out", type=str, default="",
                    help="Optional custom output parquet path.")
    args = ap.parse_args()

    repo_code = Path(__file__).resolve().parent
    data_results = (repo_code / ".." / "data" / "Results").resolve()
    data_results.mkdir(parents=True, exist_ok=True)

    df_path = data_results / "df_train_new.parquet"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing {df_path}. Run 01_Preprocess.ipynb first.")

    df = pd.read_parquet(df_path)
    # Ensure timestamps are MonthEnd
    df["YearMonth"] = pd.to_datetime(df["YearMonth"]) + MonthEnd(0)

    specs = build_specs(args.mode)

    # Simple, low-tuning MLP
    params = MLPParams(hidden_layer_sizes=(64, 32, 16), alpha=1e-4, max_iter=200,
                       early_stopping=True, random_state=0)

    nn_forecast = run_mlp_forecasts(
        df=df,
        specs=specs,
        params=params,
        train_window_months=args.train_window_months,
        start_date=args.start_date,
        verbose_every=12
    )

    out_path = Path(args.out) if args.out else (
        data_results / ("NN_with_lookahead_raw.parquet" if args.mode == "wLAB" else "NN_wo_lookahead_raw.parquet")
    )
    nn_forecast.to_parquet(out_path, index=False)
    print(f"Saved NN forecasts to: {out_path}")

    if args.evaluate:
        rf_path = data_results / ("RF_with_lookahead_raw_005.parquet" if args.mode == "wLAB" else "RF_wo_lookahead_raw_005.parquet")
        if not rf_path.exists():
            raise FileNotFoundError(f"Missing {rf_path}. Run 02_EarningsForecasts.ipynb first (RF benchmark).")
        rf = pd.read_parquet(rf_path)

        summary = summarize_mse_comparison(rf_df=rf, nn_df=nn_forecast)
        out_csv = data_results / f"NN_vs_RF_MSE_{args.mode}.csv"
        summary.to_csv(out_csv, index=False)
        print(f"Saved MSE comparison to: {out_csv}")
        print(summary)


if __name__ == "__main__":
    main()

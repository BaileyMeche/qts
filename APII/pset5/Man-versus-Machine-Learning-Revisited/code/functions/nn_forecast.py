"""
Neural-network earnings forecast replication utilities.

This module implements a simple MLP benchmark to compare against the
random-forest (BHL-style) earnings forecasts constructed in 02_EarningsForecasts.ipynb.

Design goals
------------
- Match the replication's *rolling-window* evaluation protocol:
  For each month t, fit the model on data in [t-12m, t) subject to the
  same announcement-date filters used in the random-forest code, and
  predict for the cross-section at t.
- Keep the model simple and tuning light: a small MLP with
  (i) feature standardization, (ii) L2 weight decay, and (iii) early stopping.

Outputs
-------
The main entry point `run_mlp_forecasts(...)` returns a DataFrame with:
  permno, YearMonth, NN_<h>, AF_<h>, AE_<h> for horizons h in {q1,q2,q3,y1,y2}.

This mirrors the RF forecast parquet structure so that evaluation code can reuse
the same downstream workflow.

Notes
-----
- Uses scikit-learn only (already in the repo requirements).
- The model is re-fit each month t, consistent with the RF benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MLPSpec:
    """One forecasting task specification."""
    horizon: str                # e.g., "q1"
    y_col: str                  # e.g., "EPS_true_q1"
    af_col: str                 # e.g., "EPS_ana_q1"
    ann_date_col: str           # e.g., "ANNDATS_q1"
    x_cols: List[str]           # feature list


@dataclass(frozen=True)
class MLPParams:
    """Hyperparameters for the MLPRegressor."""
    hidden_layer_sizes: Tuple[int, ...] = (64, 32, 16)
    activation: str = "relu"
    alpha: float = 1e-4                 # L2 penalty (weight decay)
    learning_rate_init: float = 1e-3
    batch_size: int = 1024
    max_iter: int = 200
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    random_state: int = 0


def _make_mlp(params: MLPParams) -> Pipeline:
    """Standardize features, then fit an MLP regressor."""
    mlp = MLPRegressor(
        hidden_layer_sizes=params.hidden_layer_sizes,
        activation=params.activation,
        solver="adam",
        alpha=params.alpha,
        batch_size=params.batch_size,
        learning_rate_init=params.learning_rate_init,
        max_iter=params.max_iter,
        early_stopping=params.early_stopping,
        validation_fraction=params.validation_fraction,
        n_iter_no_change=params.n_iter_no_change,
        random_state=params.random_state,
        verbose=False,
    )
    return Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("mlp", mlp)])


def _train_test_split_month(
    df: pd.DataFrame,
    t: pd.Timestamp,
    ann_date_col: str,
    x_cols: Sequence[str],
    y_col: str,
    train_window_months: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match the RF notebook's rolling-window logic:

    Train set:
      YearMonth in [t-12m, t) AND (ANNDATS_h + MonthEnd(0) < t), then dropna on X,Y.

    Test set:
      YearMonth == t AND (ANNDATS_h > YearMonth), then dropna on X,Y.
    """
    t0 = t - MonthEnd(train_window_months)

    df_train = df[(df["YearMonth"] < t) &
                  (df["YearMonth"] >= t0) &
                  (df[ann_date_col] + MonthEnd(0) < t)].copy()

    df_test = df[(df[ann_date_col] > df["YearMonth"]) &
                 (df["YearMonth"] == t)].copy()

    # enforce same missing-data policy as RF notebook
    df_train = df_train.dropna(subset=list(x_cols) + [y_col])
    df_test = df_test.dropna(subset=list(x_cols) + [y_col])

    return df_train, df_test


def run_mlp_forecasts(
    df: pd.DataFrame,
    specs: Sequence[MLPSpec],
    params: Optional[MLPParams] = None,
    train_window_months: int = 12,
    start_date: str = "1986-01-31",
    verbose_every: int = 12,
) -> pd.DataFrame:
    """
    Run the rolling MLP forecasts for each horizon, then merge into a single panel.

    Parameters
    ----------
    df:
      The preprocessed training panel, typically `../data/Results/df_train_new.parquet`.
      Must include at least: permno, YearMonth, and columns referenced in each MLPSpec.
    specs:
      List of horizon specifications. Each horizon is trained separately, matching the RF code.
    params:
      MLP hyperparameters. If None, defaults are used.
    train_window_months:
      Rolling training window length in months (default 12, as in the RF notebook).
    start_date:
      First forecasting month (inclusive). Matches the RF notebook's cutoff (> 1986-01-01).
    verbose_every:
      Print progress every N months if >0.

    Returns
    -------
    DataFrame with columns:
      permno, YearMonth, NN_<h>, AF_<h>, AE_<h> for each horizon h.
    """
    if params is None:
        params = MLPParams()

    df = df.copy()
    if not np.issubdtype(df["YearMonth"].dtype, np.datetime64):
        df["YearMonth"] = pd.to_datetime(df["YearMonth"]) + MonthEnd(0)

    time_idx = sorted(df["YearMonth"].unique())
    time_idx = [t for t in time_idx if t >= pd.to_datetime(start_date)]

    out_by_h: List[pd.DataFrame] = []

    for spec in specs:
        rows = []
        model = _make_mlp(params)

        for j, t in enumerate(time_idx):
            df_train, df_test = _train_test_split_month(
                df, t, spec.ann_date_col, spec.x_cols, spec.y_col, train_window_months
            )
            if df_train.empty or df_test.empty:
                continue

            model.fit(df_train[spec.x_cols], df_train[spec.y_col])
            y_pred = model.predict(df_test[spec.x_cols])

            tmp = pd.DataFrame({
                "permno": df_test["permno"].values,
                "YearMonth": df_test["YearMonth"].values,
                f"NN_{spec.horizon}": y_pred,
                f"AF_{spec.horizon}": df_test[spec.af_col].values,
                f"AE_{spec.horizon}": df_test[spec.y_col].values,
            })
            rows.append(tmp)

            if verbose_every and (j % verbose_every == 0) and (j > 0):
                print(f"[{spec.horizon}] processed {j}/{len(time_idx)} months; last={t.date()}")

        if not rows:
            raise RuntimeError(f"No forecasts produced for horizon {spec.horizon}. "
                               "Check date filters and missing-data patterns.")
        out_by_h.append(pd.concat(rows, axis=0, ignore_index=True))

    # Merge horizons horizontally (outer join on permno,YearMonth)
    out = out_by_h[0]
    for other in out_by_h[1:]:
        out = out.merge(other, on=["permno", "YearMonth"], how="outer")

    out = out.sort_values(["YearMonth", "permno"]).reset_index(drop=True)
    return out


def compute_monthly_mse(
    forecast_df: pd.DataFrame,
    model_prefix: str,                  # "NN" or "RF"
    horizons: Sequence[str] = ("q1","q2","q3","y1","y2"),
    group_col: str = "YearMonth",
) -> pd.DataFrame:
    """
    Compute monthly mean squared errors E[(pred - actual)^2] where the cross-sectional
    mean is taken within each month (matching the replication's approach).
    """
    rows = []
    for h in horizons:
        pred = f"{model_prefix}_{h}"
        act = f"AE_{h}"
        tmp = forecast_df.dropna(subset=[pred, act]).copy()
        mse_month = ((tmp[pred] - tmp[act])**2).groupby(tmp[group_col]).mean()
        rows.append(mse_month.rename(h))
    mse = pd.concat(rows, axis=1)
    return mse


def summarize_mse_comparison(
    rf_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    horizons: Sequence[str] = ("q1","q2","q3","y1","y2"),
) -> pd.DataFrame:
    """
    Produce a compact summary comparing NN, RF, and analyst forecasts in terms of
    average monthly MSE.
    """
    # Align on common (permno,YearMonth) to ensure apples-to-apples
    key = ["permno","YearMonth"]
    common = rf_df[key].merge(nn_df[key], on=key, how="inner")
    rf_df = rf_df.merge(common, on=key, how="inner")
    nn_df = nn_df.merge(common, on=key, how="inner")

    out = []
    for h in horizons:
        # MSEs as averages of monthly cross-sectional MSEs
        rf_mse = compute_monthly_mse(rf_df, "RF", [h])[h].mean()
        nn_mse = compute_monthly_mse(nn_df, "NN", [h])[h].mean()

        # analyst benchmark
        af_col = f"AF_{h}"
        ae_col = f"AE_{h}"
        tmp = rf_df.dropna(subset=[af_col, ae_col]).copy()
        af_mse = (((tmp[af_col] - tmp[ae_col])**2)
                  .groupby(tmp["YearMonth"]).mean()).mean()

        out.append({
            "horizon": h,
            "MSE_Analyst": af_mse,
            "MSE_RF": rf_mse,
            "MSE_NN": nn_mse,
            "NN_minus_RF": nn_mse - rf_mse,
            "RF_minus_Analyst": rf_mse - af_mse,
            "NN_minus_Analyst": nn_mse - af_mse,
        })
    return pd.DataFrame(out)

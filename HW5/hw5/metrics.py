from __future__ import annotations

import numpy as np
import pandas as pd


def error_stats(err: pd.Series) -> dict[str, float]:
    e = err.dropna().astype(float)
    if e.empty:
        return {k: np.nan for k in ["rmse", "mean", "std", "q01", "q05", "q95", "q99", "skew", "kurtosis", "n"]}
    return {
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "mean": float(e.mean()),
        "std": float(e.std(ddof=1)),
        "q01": float(e.quantile(0.01)),
        "q05": float(e.quantile(0.05)),
        "q95": float(e.quantile(0.95)),
        "q99": float(e.quantile(0.99)),
        "skew": float(e.skew()),
        "kurtosis": float(e.kurtosis()),
        "n": float(e.shape[0]),
    }


def oos_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true, y_pred], axis=1).dropna()
    if df.empty:
        return np.nan
    y = df.iloc[:, 0].astype(float)
    yhat = df.iloc[:, 1].astype(float)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return np.nan
    return 1.0 - sse / sst


def metrics_by_ticker(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for ticker, g in results.groupby("ticker"):
        err_box = g["q_boxcar"]
        err_ew = g["q_ew"]
        pred_box = g["pred_boxcar"]
        pred_ew = g["pred_ew"]
        y = g["rho"]
        r1 = error_stats(err_box)
        r2 = error_stats(err_ew)
        rows.append({"ticker": ticker, "model": "boxcar", **r1, "oos_r2": oos_r2(y, pred_box)})
        rows.append({"ticker": ticker, "model": "ew", **r2, "oos_r2": oos_r2(y, pred_ew)})
    return pd.DataFrame(rows)


def pooled_metrics(results: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model, err_col, pred_col in [
        ("boxcar", "q_boxcar", "pred_boxcar"),
        ("ew", "q_ew", "pred_ew"),
    ]:
        stats = error_stats(results[err_col])
        stats["model"] = model
        stats["oos_r2"] = oos_r2(results["rho"], results[pred_col])
        out.append(stats)
    df = pd.DataFrame(out)
    box_rmse = float(df.loc[df["model"] == "boxcar", "rmse"].iloc[0])
    box_r2 = float(df.loc[df["model"] == "boxcar", "oos_r2"].iloc[0])
    df["rmse_delta_vs_boxcar"] = df["rmse"] - box_rmse
    df["oos_r2_delta_vs_boxcar"] = df["oos_r2"] - box_r2
    return df


def event_windows(results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    weekly = results.groupby("date")[["q_boxcar", "q_ew"]].apply(
        lambda g: pd.Series(
            {
                "rmse_boxcar": float(np.sqrt(np.nanmean(np.square(g["q_boxcar"])))),
                "rmse_ew": float(np.sqrt(np.nanmean(np.square(g["q_ew"])))),
            }
        )
    )
    weekly["gap_ew_minus_boxcar"] = weekly["rmse_ew"] - weekly["rmse_boxcar"]
    return weekly.reindex(weekly["gap_ew_minus_boxcar"].abs().sort_values(ascending=False).head(top_n).index)

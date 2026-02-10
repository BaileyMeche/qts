from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ols_no_intercept(y: pd.Series, X: pd.DataFrame, window: int) -> pd.DataFrame:
    y = y.astype(float)
    X = X.astype(float)
    idx = y.index
    cols = list(X.columns)
    out = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)

    yv = y.to_numpy()
    xv = X.to_numpy()
    p = xv.shape[1]

    for i in range(window, len(idx)):
        yw = yv[i - window : i]
        xw = xv[i - window : i, :]
        mask = np.isfinite(yw) & np.isfinite(xw).all(axis=1)
        if mask.sum() <= p:
            continue
        beta, *_ = np.linalg.lstsq(xw[mask], yw[mask], rcond=None)
        out.iloc[i] = beta
    return out


def ew_weights(window: int, half_life: float) -> np.ndarray:
    lam = 0.5 ** (1.0 / half_life)
    ages = np.arange(window - 1, -1, -1)
    w = lam**ages
    return w / w.sum()


def rolling_slope_no_intercept_boxcar(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    out = np.full_like(yv, np.nan, dtype=float)
    for i in range(window, len(yv)):
        yw = yv[i - window : i]
        xw = xv[i - window : i]
        mask = np.isfinite(yw) & np.isfinite(xw)
        if mask.sum() < 3:
            continue
        den = np.dot(xw[mask], xw[mask])
        if den <= 0:
            continue
        out[i] = np.dot(xw[mask], yw[mask]) / den
    return pd.Series(out, index=y.index, name="mu_boxcar")


def rolling_slope_no_intercept_ew(y: pd.Series, x: pd.Series, window: int, half_life: float) -> pd.Series:
    w_full = ew_weights(window, half_life)
    yv, xv = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    out = np.full_like(yv, np.nan, dtype=float)
    for i in range(window, len(yv)):
        yw = yv[i - window : i]
        xw = xv[i - window : i]
        mask = np.isfinite(yw) & np.isfinite(xw)
        if mask.sum() < 3:
            continue
        w = w_full[mask]
        w = w / w.sum()
        den = np.dot(w * xw[mask], xw[mask])
        if den <= 0:
            continue
        out[i] = np.dot(w * xw[mask], yw[mask]) / den
    return pd.Series(out, index=y.index, name="mu_ew")

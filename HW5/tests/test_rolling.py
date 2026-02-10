from __future__ import annotations

import numpy as np
import pandas as pd

from hw5.rolling import ew_weights, rolling_ols_no_intercept, rolling_slope_no_intercept_ew


def test_rolling_ols_matches_synthetic_truth() -> None:
    n = 40
    x1 = np.arange(n, dtype=float)
    x2 = np.linspace(1.0, 3.0, n)
    beta = np.array([2.0, -0.5])
    y = x1 * beta[0] + x2 * beta[1]

    out = rolling_ols_no_intercept(
        pd.Series(y), pd.DataFrame({"x1": x1, "x2": x2}), window=16
    )
    last = out.iloc[-1].to_numpy()
    assert np.allclose(last, beta, atol=1e-10)


def test_ew_slope_matches_weighted_formula() -> None:
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    x = pd.Series([2.0, 1.0, 1.5, 2.5, 3.5, 4.0])
    w = ew_weights(window=5, half_life=12.0)

    out = rolling_slope_no_intercept_ew(y, x, window=5, half_life=12.0)
    yw = y.iloc[0:5].to_numpy()
    xw = x.iloc[0:5].to_numpy()
    expected = np.dot(w * xw, yw) / np.dot(w * xw, xw)
    assert np.isclose(out.iloc[5], expected)

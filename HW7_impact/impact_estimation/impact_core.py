from __future__ import annotations

from dataclasses import asdict
import importlib

import numpy as np
import pandas as pd

from .impact_config import ImpactConfig


RESULT_COLUMNS = [
    "label", "date", "qbin", "mu0", "mu0_se_hac", "n", "r2", "mu1", "mu2", "mu1_se", "mu2_se", "success", "cost"
]


def collapse_multihit_trades(df: pd.DataFrame, ts_floor: str | None) -> pd.DataFrame:
    d = df.copy()
    d["ts_g"] = d["ts"].dt.floor(ts_floor) if ts_floor is not None else d["ts"]
    d = d.sort_values(["exchange", "ts_g"], kind="mergesort")
    d["pxq"] = d["price"] * d["qty"]
    grouped = (
        d.groupby(["exchange", "ts_g", "side"], observed=True, sort=False)
        .agg(qty=("qty", "sum"), pxq=("pxq", "sum"))
        .reset_index()
    )
    grouped["price"] = grouped["pxq"] / grouped["qty"]
    out = grouped.rename(columns={"ts_g": "ts"})[["ts", "qty", "price", "exchange", "side"]]
    out = out.dropna(subset=["price"])
    out = out[(out["qty"] > 0) & (out["price"] > 0)].copy()
    return out.reset_index(drop=True)


def infer_side_signs(df: pd.DataFrame, M: int) -> dict[str, int]:
    d = df.sort_values("ts").reset_index(drop=True)
    sides = [str(s) for s in d["side"].dropna().unique()]
    if len(sides) == 2:
        d["p_fwd"] = d["price"].shift(-M)
        d = d.dropna(subset=["p_fwd"])
        d["r"] = d["p_fwd"] / d["price"] - 1.0
        s0, s1 = sides[0], sides[1]
        map_a, map_b = {s0: 1, s1: -1}, {s0: -1, s1: 1}
        ya = d["side"].map(map_a) * d["r"]
        yb = d["side"].map(map_b) * d["r"]
        return map_a if ya.mean() >= yb.mean() else map_b

    mapping: dict[str, int] = {}
    for s in sides:
        u = s.upper()
        if any(tok in u for tok in ["ASK", "BUY"]) or u == "A":
            mapping[s] = 1
        elif any(tok in u for tok in ["BID", "SELL"]) or u == "B":
            mapping[s] = -1
        else:
            mapping[s] = 1
    return mapping


def add_markouts(df: pd.DataFrame, M: int, side_sign: dict[str, int]) -> pd.DataFrame:
    d = df.sort_values("ts").copy()
    d["sgn"] = d["side"].map(side_sign).fillna(1).astype(int)
    d["p_fwd"] = d["price"].shift(-M)
    d = d.dropna(subset=["p_fwd"])
    d["r"] = d["p_fwd"] / d["price"] - 1.0
    d["y"] = d["sgn"] * d["r"]
    d["V"] = d["qty"]
    return d[["ts", "exchange", "side", "sgn", "price", "p_fwd", "r", "y", "V"]].reset_index(drop=True)


def trim_outliers_and_bin(df: pd.DataFrame, q_outlier: float, q_bins: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    d = df.copy()
    cut = d["V"].quantile(q_outlier)
    d = d[d["V"] <= cut].copy()
    levels = sorted(set(x for pair in q_bins for x in pair))
    qvals = d["V"].quantile(levels).to_dict()
    d["qbin"] = pd.NA
    for lo, hi in q_bins:
        vlo, vhi = qvals[lo], qvals[hi]
        d.loc[(d["V"] >= vlo) & (d["V"] < vhi), "qbin"] = f"Q{lo:.4f}_{hi:.4f}"
    return d.dropna(subset=["qbin"]).reset_index(drop=True)


def fit_sqrt_model(df: pd.DataFrame, hac_lags: int) -> dict[str, float | int]:
    x = np.sqrt(df["V"].to_numpy(float))
    y = df["y"].to_numpy(float)
    xx = np.dot(x, x)
    mu = float(np.dot(x, y) / xx) if xx > 0 else 0.0
    resid = y - mu * x
    n = len(y)
    r2 = 1.0 - (np.dot(resid, resid) / np.dot(y, y)) if np.dot(y, y) > 0 else 0.0

    se = np.nan
    sm_mod = importlib.util.find_spec("statsmodels")
    if sm_mod is not None:
        sm = importlib.import_module("statsmodels.api")
        model = sm.OLS(y, x, hasconst=False)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
        mu = float(res.params[0])
        se = float(res.bse[0])
        r2 = float(res.rsquared)
    elif n > 1 and xx > 0:
        sigma2 = float(np.dot(resid, resid) / max(n - 1, 1))
        se = float(np.sqrt(sigma2 / xx))

    return {"mu0": max(mu, 0.0), "mu0_se_hac": se, "n": n, "r2": r2}


def fit_power_model(df: pd.DataFrame) -> dict[str, float | int | bool]:
    v = df["V"].to_numpy(float)
    y = df["y"].to_numpy(float)

    scipy_mod = importlib.util.find_spec("scipy")
    if scipy_mod is not None:
        sco = importlib.import_module("scipy.optimize")

        def residuals(theta: np.ndarray) -> np.ndarray:
            return theta[0] * np.power(v, theta[1]) - y

        x0 = np.array([max(np.median(np.abs(y)), 1e-6), 0.5], float)
        res = sco.least_squares(
            residuals,
            x0=x0,
            bounds=([1e-12, 1e-6], [np.inf, 1 - 1e-6]),
            loss="huber",
            f_scale=1.0,
            max_nfev=2000,
        )
        mu1, mu2 = float(res.x[0]), float(res.x[1])
        n, dof = len(v), max(len(v) - 2, 1)
        s2 = float((2 * res.cost) / dof)
        cov = s2 * np.linalg.pinv(res.jac.T @ res.jac)
        return {
            "mu1": mu1,
            "mu2": mu2,
            "mu1_se": float(np.sqrt(max(cov[0, 0], 0.0))),
            "mu2_se": float(np.sqrt(max(cov[1, 1], 0.0))),
            "n": n,
            "success": bool(res.success),
            "cost": float(res.cost),
        }

    # fallback: log-linear approximation
    mask = (v > 0) & (y > 0)
    vv, yy = v[mask], y[mask]
    if len(vv) < 5:
        return {"mu1": 1e-12, "mu2": 0.5, "mu1_se": np.nan, "mu2_se": np.nan, "n": len(v), "success": False, "cost": np.nan}

    X = np.c_[np.ones(len(vv)), np.log(vv)]
    beta, *_ = np.linalg.lstsq(X, np.log(yy), rcond=None)
    mu1 = float(np.exp(beta[0]))
    mu2 = float(np.clip(beta[1], 1e-6, 1 - 1e-6))
    pred = mu1 * np.power(vv, mu2)
    cost = 0.5 * float(np.sum((pred - yy) ** 2))
    return {"mu1": mu1, "mu2": mu2, "mu1_se": np.nan, "mu2_se": np.nan, "n": len(v), "success": True, "cost": cost}


def fit_grid(d: pd.DataFrame, cfg: ImpactConfig, label: str) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    x = d.copy()
    x["date"] = x["ts"].dt.date
    rows = []
    for (date, qbin), g in x.groupby(["date", "qbin"], observed=True):
        if len(g) < cfg.min_n:
            continue
        row = {"label": label, "date": date, "qbin": qbin}
        row.update(fit_sqrt_model(g, cfg.hac_lags))
        row.update(fit_power_model(g))
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def run_exchange_and_pooled(d_parent: pd.DataFrame, cfg: ImpactConfig) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for ex, g in d_parent.groupby("exchange", observed=True):
        m = infer_side_signs(g, cfg.M)
        d = trim_outliers_and_bin(add_markouts(g, cfg.M, m), cfg.q_outlier, cfg.q_bins)
        r = fit_grid(d, cfg, f"EX:{ex}")
        if not r.empty:
            outputs.append(r)

    m_all = infer_side_signs(d_parent, cfg.M)
    d_all = trim_outliers_and_bin(add_markouts(d_parent, cfg.M, m_all), cfg.q_outlier, cfg.q_bins)
    r_all = fit_grid(d_all, cfg, "ALL")
    if not r_all.empty:
        outputs.append(r_all)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=RESULT_COLUMNS)


def robustness_sweep(d_parent: pd.DataFrame, base_cfg: ImpactConfig | None = None) -> pd.DataFrame:
    cfg = base_cfg or ImpactConfig()
    outs: list[pd.DataFrame] = []
    for M in [1, 5, 10, 25, 50, 100]:
        for qout in [0.995, 0.9975, 0.999]:
            local = ImpactConfig(**asdict(cfg))
            local.M = M
            local.q_outlier = qout
            local.hac_lags = max(10, M)
            r = run_exchange_and_pooled(d_parent, local)
            if not r.empty:
                r["M"] = M
                r["q_outlier"] = qout
                outs.append(r)
    if outs:
        return pd.concat(outs, ignore_index=True)
    return pd.DataFrame(columns=RESULT_COLUMNS + ['M', 'q_outlier'])

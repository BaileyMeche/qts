from __future__ import annotations

from dataclasses import asdict
import importlib

import numpy as np
import pandas as pd

from impact_config import ImpactConfig


RESULT_COLUMNS = [
    "label", "date", "qbin", "mu0", "mu0_se_hac", "n", "r2", "mu1", "mu2", "mu1_se", "mu2_se", "success", "cost"
]

# impact_core.py (add near top)
import re
from typing import Optional

# impact_core.py (replace infer_side_signs with this)

def _semantic_side_sign(label: str) -> Optional[int]:
    u = str(label).strip().upper()
    if u in {"A", "ASK", "BUY"}:
        return +1
    if u in {"B", "BID", "SELL"}:
        return -1
    if ("ASK" in u) or ("BUY" in u):
        return +1
    if ("BID" in u) or ("SELL" in u):
        return -1
    return None


def infer_side_signs(df: pd.DataFrame, M: int, mode: str = "infer") -> dict[str, int]:
    """
    mode="semantic": use semantic mapping only (unknowns -> +1)
    mode="infer": semantic first; if exactly 2 labels and unresolved, infer by maximizing mean signed markout
    """
    d = df.sort_values("ts").reset_index(drop=True)
    sides = [str(s) for s in d["side"].dropna().unique()]

    mapping: dict[str, int] = {}
    unknown: list[str] = []
    for s in sides:
        sg = _semantic_side_sign(s)
        if sg is None:
            unknown.append(s)
        else:
            mapping[s] = sg

    if not unknown:
        return mapping

    if mode != "infer":
        for s in unknown:
            mapping[s] = +1
        return mapping

    # infer only if exactly 2 labels
    if len(sides) == 2:
        d["p_fwd"] = d["price"].shift(-M)
        d = d.dropna(subset=["p_fwd"])
        r = d["p_fwd"] / d["price"] - 1.0

        s0, s1 = sides[0], sides[1]
        map_a, map_b = {s0: +1, s1: -1}, {s0: -1, s1: +1}
        ya = d["side"].map(map_a) * r
        yb = d["side"].map(map_b) * r
        return map_a if float(ya.mean()) >= float(yb.mean()) else map_b

    for s in unknown:
        mapping[s] = +1
    return mapping

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


# impact_core.py (replace fit_grid)
def fit_grid(d: pd.DataFrame, cfg: ImpactConfig, label: str) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    x = d.copy()
    x["date"] = x["ts"].dt.date
    rows = []

    for date, gd in x.groupby("date", observed=True):
        # stage 1: mu2 from whole day (all qbins combined)
        if len(gd) < cfg.min_n:
            continue
        pwr_day = fit_power_exponent_binned(gd, n_bins=30, min_bin_n=200)
        mu2_hat = float(pwr_day["mu2"])
        mu2_se  = float(pwr_day.get("mu2_se", np.nan))

        # stage 2: per qbin with mu2 fixed
        for qbin, g in gd.groupby("qbin", observed=True):
            if len(g) < cfg.min_n:
                continue

            row = {"label": label, "date": date, "qbin": qbin}
            row.update(fit_sqrt_model(g, cfg.hac_lags))

            pwr_bin = fit_mu1_given_mu2(g, mu2_hat)
            row.update({
                "mu1": float(pwr_bin["mu1"]),
                "mu2": mu2_hat,
                "mu1_se": float(pwr_bin["mu1_se"]),
                "mu2_se": mu2_se,
                "success": bool(pwr_bin["success"]),
                "cost": float(pwr_bin["cost"]),
                "n": int(pwr_bin["n"]),
            })
            rows.append(row)

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def run_exchange_and_pooled(d_parent: pd.DataFrame, cfg: ImpactConfig) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []

    for ex, g in d_parent.groupby("exchange", observed=True):
        # NEW: infer at cfg.sign_ref_M, apply at cfg.M
        m = infer_side_signs(g, cfg.sign_ref_M, mode=cfg.side_map_mode)
        d = trim_outliers_and_bin(add_markouts(g, cfg.M, m), cfg.q_outlier, cfg.q_bins)
        r = fit_grid(d, cfg, f"EX:{ex}")
        if not r.empty:
            outputs.append(r)

    m_all = infer_side_signs(d_parent, cfg.sign_ref_M, mode=cfg.side_map_mode)
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
            # sign mapping remains anchored by local.sign_ref_M (copied from cfg)
            r = run_exchange_and_pooled(d_parent, local)
            if not r.empty:
                r["M"] = M
                r["q_outlier"] = qout
                outs.append(r)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=RESULT_COLUMNS + ["M", "q_outlier"])

# impact_core.py (new)
def fit_power_exponent_binned(df: pd.DataFrame, *, n_bins: int = 30, min_bin_n: int = 200) -> dict:
    """
    Fit y ~ mu1 * V^mu2 using binned means (reduces noise, stabilizes mu2).
    Returns mu1_global, mu2, and approximate SEs.
    """
    x = df[["V", "y"]].dropna().copy()
    x = x[(x["V"] > 0)]
    if x.empty:
        return {"mu1": 1e-12, "mu2": 0.5, "mu1_se": np.nan, "mu2_se": np.nan, "success": False, "cost": np.nan}

    q = min(n_bins, int(x["V"].nunique()))
    if q < 5:
        # fallback to raw fit
        return fit_power_model(df)

    x["vbin"] = pd.qcut(x["V"], q=q, duplicates="drop")
    agg = (
        x.groupby("vbin", observed=True)
        .agg(vb=("V", "median"), yb=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    agg = agg[(agg["n"] >= min_bin_n) & (agg["yb"] > 0) & (agg["vb"] > 0)].copy()
    if len(agg) < 6:
        return fit_power_model(df)

    vb = agg["vb"].to_numpy(float)
    yb = agg["yb"].to_numpy(float)
    w  = np.sqrt(agg["n"].to_numpy(float))

    # log-linear start
    X = np.c_[np.ones(len(vb)), np.log(vb)]
    beta, *_ = np.linalg.lstsq(X, np.log(yb), rcond=None)
    mu1_0 = float(np.exp(beta[0]))
    mu2_0 = float(np.clip(beta[1], 1e-6, 1 - 1e-6))
    x0 = np.array([max(mu1_0, 1e-12), mu2_0], float)

    scipy_mod = importlib.util.find_spec("scipy")
    if scipy_mod is None:
        # no scipy: return log-linear
        pred = x0[0] * np.power(vb, x0[1])
        cost = 0.5 * float(np.sum((w * (pred - yb)) ** 2))
        return {"mu1": x0[0], "mu2": x0[1], "mu1_se": np.nan, "mu2_se": np.nan, "success": True, "cost": cost}

    sco = importlib.import_module("scipy.optimize")

    def resid(theta: np.ndarray) -> np.ndarray:
        return w * (theta[0] * np.power(vb, theta[1]) - yb)

    res = sco.least_squares(
        resid,
        x0=x0,
        bounds=([1e-12, 1e-6], [np.inf, 1 - 1e-6]),
        loss="huber",
        f_scale=1.0,
        max_nfev=4000,
    )

    mu1, mu2 = float(res.x[0]), float(res.x[1])

    dof = max(len(vb) - 2, 1)
    s2 = float((2 * res.cost) / dof)
    cov = s2 * np.linalg.pinv(res.jac.T @ res.jac)

    return {
        "mu1": mu1,
        "mu2": mu2,
        "mu1_se": float(np.sqrt(max(cov[0, 0], 0.0))),
        "mu2_se": float(np.sqrt(max(cov[1, 1], 0.0))),
        "success": bool(res.success),
        "cost": float(res.cost),
    }


# impact_core.py (new)
def fit_mu1_given_mu2(df: pd.DataFrame, mu2: float) -> dict:
    v = df["V"].to_numpy(float)
    y = df["y"].to_numpy(float)
    x = np.power(v, mu2)

    xx = float(np.dot(x, x))
    mu1 = float(np.dot(x, y) / xx) if xx > 0 else 0.0
    mu1 = max(mu1, 0.0)

    resid = y - mu1 * x
    n = len(y)
    dof = max(n - 1, 1)
    sigma2 = float(np.dot(resid, resid) / dof) if n > 1 else np.nan
    se = float(np.sqrt(sigma2 / xx)) if (xx > 0 and np.isfinite(sigma2)) else np.nan
    cost = 0.5 * float(np.dot(resid, resid))
    return {"mu1": mu1, "mu1_se": se, "n": n, "success": True, "cost": cost}
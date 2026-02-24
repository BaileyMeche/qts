from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fit(d: pd.DataFrame, mu0: float, mu1: float, mu2: float, title: str):
    x = d[["V", "y"]].dropna().copy()
    x = x[x["V"] > 0]
    x["vbin"] = pd.qcut(x["V"], q=min(20, x["V"].nunique()), duplicates="drop")
    agg = x.groupby("vbin", observed=True).agg(vb=("V", "median"), yb=("y", "mean")).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(agg["vb"], agg["yb"], label="Binned mean impact")

    vmin, vmax = float(max(x["V"].min(), 1e-12)), float(x["V"].max())
    vgrid = np.geomspace(vmin, vmax, 200)
    ax.plot(vgrid, mu0 * np.sqrt(vgrid), label="Square-root")
    ax.plot(vgrid, mu1 * np.power(vgrid, mu2), label="Power-law")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trade size V")
    ax.set_ylabel("Signed markout y")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig, outdir: Path, stem: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    path = outdir / f"{safe}.png"
    fig.savefig(path, dpi=140)
    return path

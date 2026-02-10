from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rolling_rmse(results: pd.DataFrame, out_path: Path, window: int = 12) -> None:
    weekly = results.groupby("date")[["q_boxcar", "q_ew"]].apply(
        lambda g: pd.Series(
            {
                "boxcar": float(np.sqrt(np.nanmean(np.square(g["q_boxcar"])))),
                "ew": float(np.sqrt(np.nanmean(np.square(g["q_ew"])))),
            }
        )
    )
    roll = weekly.rolling(window).mean()
    plt.figure(figsize=(11, 5))
    plt.plot(roll.index, roll["boxcar"], label="Boxcar")
    plt.plot(roll.index, roll["ew"], label="EWLS")
    plt.title(f"Rolling pooled RMSE ({window}-week mean)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_error_gap(results: pd.DataFrame, out_path: Path) -> None:
    weekly = results.groupby("date")[["q_boxcar", "q_ew"]].apply(
        lambda g: pd.Series(
            {
                "rmse_boxcar": float(np.sqrt(np.nanmean(np.square(g["q_boxcar"])))),
                "rmse_ew": float(np.sqrt(np.nanmean(np.square(g["q_ew"])))),
            }
        )
    )
    weekly["gap"] = weekly["rmse_ew"] - weekly["rmse_boxcar"]
    plt.figure(figsize=(11, 4))
    plt.plot(weekly.index, weekly["gap"], label="EW - Boxcar")
    plt.axhline(0.0, color="black", lw=1)
    plt.title("Weekly pooled RMSE gap")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tail_comparison(results: pd.DataFrame, out_path: Path) -> None:
    e1 = results["q_boxcar"].dropna().to_numpy()
    e2 = results["q_ew"].dropna().to_numpy()
    if e1.size == 0 or e2.size == 0:
        return
    qs = np.linspace(0.01, 0.99, 99)
    q1 = np.quantile(e1, qs)
    q2 = np.quantile(e2, qs)
    plt.figure(figsize=(6, 6))
    plt.plot(q1, q2, lw=1.5)
    lim = [min(q1.min(), q2.min()), max(q1.max(), q2.max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Boxcar error quantiles")
    plt.ylabel("EWLS error quantiles")
    plt.title("Tail comparison: error quantile-quantile")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_robustness_heatmap(table: pd.DataFrame, value_col: str, out_path: Path) -> None:
    piv = table.pivot(index="window", columns="half_life", values=value_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(piv.shape[1]), labels=[str(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), labels=[str(r) for r in piv.index])
    ax.set_xlabel("Half-life")
    ax.set_ylabel("Window")
    ax.set_title(value_col)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

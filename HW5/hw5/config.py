from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIT_INTERCEPT = False  # match HW definitions (no constant term in f, c, q)
RUN_INTERCEPT_ROBUSTNESS = True

@dataclass(frozen=True)
class Config:
    root: Path
    cds_path: Path
    equity_cache_path: Path
    output_dir: Path
    boxcar_window: int = 16
    ew_window: int = 16
    ew_half_life: float = 12.0
    robustness_windows: tuple[int, ...] = (12, 16, 26)
    robustness_half_lives: tuple[float, ...] = (8.0, 12.0, 20.0)



def default_config(root: Path | None = None) -> Config:
    hw_root = (root or Path(__file__).resolve().parents[1]).resolve()
    return Config(
        root=hw_root,
        cds_path=hw_root / "Liq5YCDS.delim",
        equity_cache_path=hw_root / "equity_adj_close.parquet",
        output_dir=hw_root / "output",
    )

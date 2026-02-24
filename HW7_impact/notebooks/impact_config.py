from __future__ import annotations


from dataclasses import dataclass, field
from typing import Optional

# impact_config.py  (edit in place)

@dataclass(slots=True)
class ImpactConfig:
    M: int = 50
    ts_floor: Optional[str] = "1ms"
    q_outlier: float = 0.9975
    q_bins: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: (
            (0.80, 0.90),
            (0.90, 0.95),
            (0.95, 0.97),
            (0.97, 0.99),
            (0.99, 0.9975),
        )
    )
    hac_lags: int = 50
    max_rows_per_file: Optional[int] = None
    sample_frac: Optional[float] = None
    min_n: int = 5000
    random_state: int = 42

    # NEW (add)
    side_map_mode: str = "semantic"   # {"semantic","infer"}
    sign_ref_M: int = 1              # infer side mapping at this fixed horizon, not cfg.M

def require_parquet_engine() -> str:
    """Require a parquet engine and return the chosen one."""
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        pass

    try:
        import fastparquet  # noqa: F401

        return "fastparquet"
    except Exception as exc:
        raise RuntimeError(
            "No parquet engine found. Install one with `pip install pyarrow` "
            "or `pip install fastparquet` before running the pipeline."
        ) from exc

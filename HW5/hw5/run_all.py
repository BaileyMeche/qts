from __future__ import annotations

from pathlib import Path

from .config import default_config
from .metrics import event_windows, metrics_by_ticker, pooled_metrics
from .pipeline import run_pipeline
from .plots import plot_error_gap, plot_robustness_heatmap, plot_rolling_rmse, plot_tail_comparison
from .robustness import run_robustness


def main() -> None:
    config = default_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(config)
    panel = result.panel
    panel.to_csv(config.output_dir / "panel_results.csv", index=False)

    per_ticker = metrics_by_ticker(panel)
    per_ticker.to_csv(config.output_dir / "metrics_per_ticker.csv", index=False)

    pooled = pooled_metrics(panel)
    pooled.to_csv(config.output_dir / "metrics_pooled.csv", index=False)

    events = event_windows(panel)
    events.to_csv(config.output_dir / "event_windows.csv")

    plot_rolling_rmse(panel, config.output_dir / "rolling_rmse.png")
    plot_error_gap(panel, config.output_dir / "error_gap.png")
    plot_tail_comparison(panel, config.output_dir / "tail_qq.png")

    robustness_dir = config.output_dir / "robustness"
    robust = run_robustness(
        config=config,
        cds_ret=result.cds_weekly_ret,
        eq_ret=result.eq_weekly_ret,
        market_ret=result.market_weekly_ret,
        out_dir=robustness_dir,
    )
    plot_robustness_heatmap(robust, "delta_rmse_vs_baseline", robustness_dir / "delta_rmse_heatmap.png")
    plot_robustness_heatmap(robust, "stability_rank_corr", robustness_dir / "stability_heatmap.png")

    report = Path(config.output_dir / "summary.md")
    report.write_text(
        "\n".join(
            [
                "# HW5 Predictive Regression Summary",
                "",
                "## Generated outputs",
                "- panel_results.csv",
                "- metrics_per_ticker.csv",
                "- metrics_pooled.csv",
                "- event_windows.csv",
                "- rolling_rmse.png",
                "- error_gap.png",
                "- tail_qq.png",
                "- robustness/robustness_sweep.csv",
                "- robustness/delta_rmse_heatmap.png",
                "- robustness/stability_heatmap.png",
                "",
                "## Interpretation scaffold",
                "See event_windows.csv for largest model gaps, and compare pooled/tail metrics between boxcar and EW.",
            ]
        )
    )


if __name__ == "__main__":
    main()

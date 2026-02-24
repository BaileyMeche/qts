# Temporary Market Impact Parameter Estimation

This directory contains an end-to-end Python pipeline for estimating temporary impact parameters from tick/trade parquet data.

## Contents

- `impact_estimation/`: reusable module (IO, core estimation, plotting, CLI)
- `tests/`: fast pytest coverage for key invariants
- `notebooks/TemporaryMarketImpactParamEstimation.ipynb`: runbook notebook
- `outputs/`: generated tables and figures

## Setup

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pytest HW7_impact/tests -q
```

## Run CLI

```bash
python -m impact_estimation.cli --repo-root . --outdir HW7_impact/outputs --run-robustness
```

Optional environment overrides:
- `TRADE_FILES_GLOB='**/*trades*.parquet'`
- `TRADE_FILES_LIST='/path/a.parquet,/path/b.parquet'`

## Notebook

Open and run top-to-bottom:

```bash
jupyter notebook HW7_impact/notebooks/TemporaryMarketImpactParamEstimation.ipynb
```

## Outputs

- `outputs/impact_results.parquet`, `outputs/impact_results.csv`
- `outputs/impact_robustness.parquet`, `outputs/impact_robustness.csv` (if enabled)
- `outputs/figures/*.png`

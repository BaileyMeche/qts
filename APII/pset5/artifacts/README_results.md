# PS5 Part 2 Outputs

## Produced files
- `artifacts/mse_comparison_table.csv`
- `artifacts/mse_comparison_table.md`
- `artifacts/mse_improvement.csv`
- `RUNLOG.txt`
- `writeup.md`

## Reproduction commands attempted
```bash
python -m venv .venv
source .venv/bin/activate
# requirements.txt is missing in this copy of the repo
python -c "import sys; sys.path.append('code'); import functions.nn_forecast as nf; print('ok')"
jupyter nbconvert --execute --to notebook --inplace code/00_DataDownload.ipynb
jupyter nbconvert --execute --to notebook --inplace code/01_Preprocess.ipynb
jupyter nbconvert --execute --to notebook --inplace code/02_EarningsForecasts.ipynb
cd code && python run_nn_forecast.py --mode woLAB --evaluate
```

## Blocker
This environment does not contain the upstream replication notebooks/data and cannot reach external hosts (proxy 403), so `df_train_new.parquet` and RF benchmark parquet files could not be generated/downloaded.

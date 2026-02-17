Deliverables for Problem Set 5 — Part 2 (Man vs Machine Learning Revisited)

Included files:
- PS5_Part2_Writeup.pdf
- code/functions/nn_forecast.py
- code/run_nn_forecast.py
- code/07_NeuralNetwork.ipynb

How to run (from repository root):
1) Follow upstream replication:
   - run code/00_DataDownload.ipynb (WRDS credentials required)
   - run code/01_Preprocess.ipynb (creates data/Results/df_train_new.parquet)
   - run code/02_EarningsForecasts.ipynb (creates RF_* parquets)
2) Run NN:
   - Notebook: open code/07_NeuralNetwork.ipynb
   - or CLI: cd code && python run_nn_forecast.py --mode woLAB --evaluate

Outputs created by NN code:
- data/Results/NN_wo_lookahead_raw.parquet (or NN_with_lookahead_raw.parquet)
- data/Results/NN_vs_RF_MSE_<mode>.csv (if --evaluate)

Note: In the sandbox environment used to generate these deliverables, WRDS data were not available,
so numeric replication outputs were not computed here.

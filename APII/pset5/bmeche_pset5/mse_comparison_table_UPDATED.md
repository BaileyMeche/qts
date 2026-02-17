# MSE Comparison (woLAB)

Computed from `data/Results/NN_vs_RF_MSE_woLAB.csv`.

> Note: Quarterly-horizon NN MSE is extremely large; this typically indicates an implementation or scaling issue for q1–q3 and should be diagnosed before final submission.


| horizon | MSE_Analyst | MSE_RF | MSE_NN | NN_minus_RF | RF_minus_Analyst | NN_minus_Analyst |
| --- | --- | --- | --- | --- | --- | --- |
| q1 | 0.122559 | 0.109678 | 1.038e+13 | 1.038e+13 | -0.012881 | 1.038e+13 |
| q2 | 0.121711 | 0.109841 | 1.516e+13 | 1.516e+13 | -0.011870 | 1.516e+13 |
| q3 | 0.121570 | 0.108771 | 3.706e+13 | 3.706e+13 | -0.012799 | 3.706e+13 |
| y1 | 0.121792 | 0.111036 | 0.087926 | -0.023110 | -0.010756 | -0.033866 |
| y2 | 0.123895 | 0.112001 | 0.090127 | -0.021874 | -0.011894 | -0.033768 |

## Percent Improvement (NN vs RF)

| horizon | pct_improvement_nn_vs_rf |
| --- | --- |
| q1 | -9.468e+13 |
| q2 | -1.380e+14 |
| q3 | -3.408e+14 |
| y1 | 0.208134 |
| y2 | 0.195301 |
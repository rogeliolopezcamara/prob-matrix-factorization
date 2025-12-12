# Forecast Analysis Results

| RMSE | MAE | MSE | R2 | Model |
| --- | --- | --- | --- | --- |
| 1.0893774284665239 | 0.6543302102378422 | 1.1867431816523364 | -0.07421396538114666 | gaussian_mf |
| 1.3012214863769103 | 0.9848710637563297 | 1.6931773566089359 | -0.5326271011762103 | poisson_mf |
| 1.0993831189851402 | 0.7396132238934334 | 1.208643242309495 | -0.09403742117539804 | hpf_cavi |
| 1.4445542457558118 | 1.215466145262309 | 2.0867369689311426 | -0.8888686522571052 | hpf_pytorch |


## Plots
![RMSE Comparison](figures/forecast_analysis/rmse_comparison.png)
![Preds vs True](figures/forecast_analysis/preds_vs_true_box.png)
![Residuals](figures/forecast_analysis/residuals.png)

### Predictions by True Value
#### gaussian_mf
![gaussian_mf Histograms](figures/forecast_analysis/gaussian_mf_pred_hist_by_true.png)
#### poisson_mf
![poisson_mf Histograms](figures/forecast_analysis/poisson_mf_pred_hist_by_true.png)
#### hpf_cavi
![hpf_cavi Histograms](figures/forecast_analysis/hpf_cavi_pred_hist_by_true.png)
#### hpf_pytorch
![hpf_pytorch Histograms](figures/forecast_analysis/hpf_pytorch_pred_hist_by_true.png)

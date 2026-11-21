# Machine Learning for Volatility Forecasting Under Distribution Shift

## Executive summary

The evidence is **partially supportive and metric-dependent**. XGBoost reduced pooled out-of-sample MAE on all five ETF series, but its RMSE was worse than the historical-volatility baseline on all five. The gain in typical absolute error therefore did not translate into better control of large errors. Performance and event discrimination differ substantially by asset, so the experiment does not support a claim of universal stable predictive information.

## Data and evaluation

- Five ETFs: SPY, QQQ, IWM, TLT, GLD; adjusted daily OHLCV from Yahoo Finance via yfinance.
- Retrieved 2026-11-21T20:17:11.346675+00:00; first forecast dates: {'SPY': '2010-04-13 00:00:00', 'QQQ': '2010-04-13 00:00:00', 'IWM': '2010-09-03 00:00:00', 'TLT': '2012-10-31 00:00:00', 'GLD': '2015-02-25 00:00:00'}; last labeled forecast origins: {'SPY': '2026-11-19 00:00:00', 'QQQ': '2026-11-19 00:00:00', 'IWM': '2026-11-19 00:00:00', 'TLT': '2026-11-19 00:00:00', 'GLD': '2026-11-19 00:00:00'}.
- Expanding walk-forward validation; initial train=2520 sessions; test blocks=252 sessions; seed=17.
- Target is next-session absolute close-to-close log return. This is a noisy daily volatility proxy, not latent or intraday realized volatility.

## Regression results

Positive skill means lower error than historical volatility.

| Asset | XGBoost MAE skill | XGBoost RMSE skill | XGBoost annual MAE wins |
|---|---:|---:|---:|
| SPY | +6.1% | -2.5% | 17/17 |
| QQQ | +4.1% | -6.6% | 13/17 |
| IWM | +2.9% | -4.6% | 14/17 |
| TLT | -0.2% | -7.6% | 8/15 |
| GLD | +4.4% | -6.0% | 11/12 |

## Probability forecasts

The logistic model predicts whether next-session volatility exceeds a cutoff fixed at the training-window 80th percentile. Its cutoff and base-rate reference are recomputed within each fold. See the PDF report for log loss, Brier score, ROC-AUC, average precision, calibration error, and reliability curves.

## Distribution shift and inference

Yearly performance is uneven even where average MAE improves. High-volatility observations are materially harder to forecast. Paired absolute-error tests use a 21-session HAC standard error and Holm correction over the 15 asset/model comparisons; these are descriptive comparisons on one historical sample, not proof of future stability.

## Sensitivity and limitations

SPY was also evaluated with six-month rather than one-year test blocks, preserving the same initial training size and first forecast date. See the report table for the paired comparison.

The study uses five ETF proxies from one vendor and daily adjusted close prices. It does not include transaction costs or test a trading strategy. No LSTM or attention model was included: the modest single-asset sample and noisy daily target do not justify extra model capacity before larger multi-asset and later-period tests. Yahoo/yfinance terms should be reviewed before redistribution or commercial use.

## Reproduction

```powershell
pip install -e ".[download,notebooks,test]"
pytest -q
python -m src.run_experiment --local-data-dir data --min-train-size 2520 --test-size 252
python -m src.generate_report
```

## References

1. Qiu, Z., Kownatzki, C., Scalzo, F., and Cha, E.S. (2025). Historical Perspectives in Volatility Forecasting Methods with Machine Learning. *Risks*, 13(5), 98. https://doi.org/10.3390/risks13050098
2. yfinance documentation: https://ranaroussi.github.io/yfinance/
3. Hudson River Trading, Trading Machine Learning Benchmarks: https://www.hudsonrivertrading.com/hrtbeat/trading-machine-learning/

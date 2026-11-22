# FI-2010 Limit Order Book Companion Experiment

## Research question

Do order-book features retain short-horizon predictive information when the distribution changes across trading days?

## Results at a glance

The XGBoost model had the best mean macro-F1 over the nine held-out days at **0.478** (day-to-day standard deviation 0.050). Random forest was close at **0.474**; logistic regression averaged **0.395**. The XGBoost model led on some days and fell behind random forest on others. Thus the features carry predictive information in this benchmark, while model ranking and score still move across days.

These are classification results, not a trading-strategy result. The FI-2010 release used here supplies normalized features and labels. This experiment does not estimate fills, spread capture, fees, latency, slippage, turnover, inventory risk, or P&L.

## Test-day scores

Macro-F1 is the unweighted average of class-specific F1 scores, so each direction/stationary class contributes equally.

| Test day | Prior | Logistic | Random forest | XGBoost |
|---:|---:|---:|---:|---:|
| 2 | 0.198 | 0.396 | 0.414 | 0.432 |
| 3 | 0.185 | 0.355 | 0.424 | 0.402 |
| 4 | 0.192 | 0.405 | 0.474 | 0.469 |
| 5 | 0.199 | 0.435 | 0.487 | 0.504 |
| 6 | 0.184 | 0.378 | 0.529 | 0.514 |
| 7 | 0.196 | 0.414 | 0.515 | 0.533 |
| 8 | 0.166 | 0.346 | 0.485 | 0.417 |
| 9 | 0.166 | 0.380 | 0.496 | 0.526 |
| 10 | 0.185 | 0.443 | 0.443 | 0.505 |

## Pooled held-out metrics

Pooled metrics combine all 354,825 test observations from days 2 through 10. The fold-mean macro-F1 above gives each test day equal weight; pooled metrics give each event equal weight.

| Model | Mean fold macro-F1 (SD) | Pooled macro-F1 | Pooled balanced accuracy | Pooled log loss | Pooled multiclass Brier |
|---|---:|---:|---:|---:|---:|
| xgboost | 0.478 (0.050) | 0.496 | 0.496 | 0.959 | 0.578 |
| random forest | 0.474 (0.039) | 0.502 | 0.500 | 0.988 | 0.596 |
| logistic regression | 0.395 (0.034) | 0.402 | 0.423 | 1.042 | 0.628 |
| training class prior | 0.186 (0.012) | 0.184 | 0.333 | 1.087 | 0.659 |

Probabilities are scored with log loss and multiclass Brier score. Lower is better for those metrics. They do not replace a full calibration analysis by day or stock.

## Experimental design

- **Data:** official FI-2010 NoAuction Z-score representation; five Nasdaq Nordic stocks, ten consecutive trading days, 144 provided features and five mid-price movement labels.
- **Target:** the 10-event-ahead label (source row 149): 1 = up, 2 = stationary, 3 = down. The source defines the labels from the average relative mid-price change over the next event horizon.
- **Validation:** the published day-based anchored folds CF_1 through CF_9. Each model trains on the fold's expanding past window and tests on the next day. The training windows range from 39,512 to 362,400 observations.
- **Models:** training-window class-prior baseline; logistic regression (C=0.1); random forest (100 trees, depth 18, leaf size 30, half-sample per tree); XGBoost (240 trees, depth 5, learning rate 0.06). Hyperparameters were fixed before evaluating these folds; no test-fold tuning was performed.
- **Metrics:** accuracy, balanced accuracy, macro precision/recall/F1, log loss and multiclass Brier score. Per-event out-of-sample predictions are retained for independent review.
- **Shift diagnostic:** decile-bin population stability index (PSI), with bin boundaries estimated on each training fold. Across folds, median feature PSI ranged 0.010–0.034; the 90th percentile across features ranged 2.689–13.978. The feature-wise tail matters: averages can be dominated by a small number of features. PSI is descriptive and does not identify a causal market regime.

## Day-block uncertainty and shift association

Paired two-day moving-block bootstrap intervals resample the nine ordered test days, not individual events. Positive improvement means higher macro-F1 or lower log loss than the training class-prior baseline.

| Model | Metric | Mean improvement vs class prior | 95% two-day block-bootstrap interval |
|---|---|---:|---:|
| logistic regression | log_loss | 0.043 | [0.033, 0.052] |
| logistic regression | macro_f1 | 0.209 | [0.192, 0.222] |
| random forest | log_loss | 0.091 | [0.071, 0.126] |
| random forest | macro_f1 | 0.289 | [0.261, 0.322] |
| xgboost | log_loss | 0.124 | [0.101, 0.145] |
| xgboost | macro_f1 | 0.292 | [0.257, 0.325] |

These are descriptive uncertainty intervals, not significance tests. There are only nine folds and the expanding training sets overlap. No untouched final test period was reserved after results were explored. The PSI/performance rank associations are likewise exploratory and do not establish that measured feature shift causes performance changes:

- logistic regression: PSI vs macro-F1 Spearman 0.00; PSI vs log-loss Spearman 0.38.
- random forest: PSI vs macro-F1 Spearman -0.13; PSI vs log-loss Spearman 0.43.
- xgboost: PSI vs macro-F1 Spearman -0.02; PSI vs log-loss Spearman 0.42.

## Interpretation and limitations

1. The benchmark contains ten trading days from June 2010. Nine chronological test days are useful for a focused walk-forward exercise, but they cannot establish persistence across months, years, venues, or current market structure.
2. The public feature matrices aggregate the five stocks without a per-row security identifier in the released files used here. Results are therefore pooled and cannot support a per-stock stability claim.
3. Features are the benchmark's precomputed Z-score representation. We use the official anchored folds and do not fit an alternate raw-data feature pipeline. Normalization details inherited from the source should be considered when comparing against other datasets or production feeds.
4. Adjacent event labels overlap in their forward horizons, so event-level observations are not independent. Fold variability is descriptive; this report does not claim a formal significance test.
5. No raw order messages, executable order simulation, or full cost model is evaluated here. No profitability claim is made.
6. The longer daily ETF study remains the main multi-year distribution-shift evidence. This FI-2010 experiment adds microstructure relevance but does not replace that broader test.

## Reproducibility

The official Fairdata package is identified by dataset ID `73eb48d7-4dbc-4a10-a52a-da745b47a649` and SHA-256 `bcc89a5aa7d8067dda98374393444eb885a4283a41fd33e323496380e057e1a6`. It is licensed CC BY 4.0. Download and verify it, then run:

```powershell
python -m src.download_fi2010
python -m src.market_microstructure
python -m src.microstructure_inference
python -m src.generate_microstructure_report
```

The raw package and extracted matrices are kept out of Git. Metrics, metadata, figures, compressed held-out predictions, this report, and the PDF are written under `results/market_microstructure/`.

## Sources

- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., and Iosifidis, A. (2018). “Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods.” *Journal of Forecasting*, 37(8), 852–866. [Paper](https://doi.org/10.1002/for.2543); [arXiv](https://arxiv.org/abs/1705.03233).
- Official dataset record: [Fairdata Etsin, FI-2010](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649).

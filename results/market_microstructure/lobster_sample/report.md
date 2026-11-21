# LOBSTER raw-event sample study

## Scope

This engineering and modeling demonstration uses LOBSTER's public five-level sample files for five US stocks on 2012-06-21. It is one session, so it cannot establish persistence across days or regimes.

The C++17 extractor processed 1,747,025 aligned message/book events in total. It derives top-of-book spread, queue imbalance, order-flow imbalance, and signed event flow. Raw sample archives are kept outside Git; their SHA-256 hashes are in `run_metadata.json`.

## Evaluation

The target is the mid-price direction 100 book events ahead; moves within half a cent are labeled stationary. Each ticker uses a chronological 70/30 split. The last 100 training events are purged so training labels cannot extend into the test period.

| Ticker | Model | Train events | Test events | Macro-F1 | Balanced accuracy | Log loss |
|---|---|---:|---:|---:|---:|---:|
| AAPL | engineered_logistic | 211,010 | 90,377 | 0.382 | 0.385 | 1.077 |
| AAPL | queue_imbalance_logistic | 211,010 | 90,377 | 0.345 | 0.346 | 1.096 |
| AAPL | random_forest | 211,010 | 90,377 | 0.389 | 0.389 | 1.033 |
| AAPL | training_class_prior | 211,010 | 90,377 | 0.219 | 0.333 | 0.927 |
| AAPL | xgboost | 211,010 | 90,377 | 0.368 | 0.385 | 0.912 |
| AMZN | engineered_logistic | 109,054 | 46,681 | 0.392 | 0.406 | 1.094 |
| AMZN | queue_imbalance_logistic | 109,054 | 46,681 | 0.351 | 0.377 | 1.094 |
| AMZN | random_forest | 109,054 | 46,681 | 0.392 | 0.395 | 1.041 |
| AMZN | training_class_prior | 109,054 | 46,681 | 0.216 | 0.333 | 0.939 |
| AMZN | xgboost | 109,054 | 46,681 | 0.363 | 0.382 | 0.922 |
| GOOG | engineered_logistic | 78,771 | 33,702 | 0.397 | 0.397 | 1.027 |
| GOOG | queue_imbalance_logistic | 78,771 | 33,702 | 0.302 | 0.363 | 1.088 |
| GOOG | random_forest | 78,771 | 33,702 | 0.391 | 0.395 | 0.933 |
| GOOG | training_class_prior | 78,771 | 33,702 | 0.223 | 0.333 | 0.844 |
| GOOG | xgboost | 78,771 | 33,702 | 0.376 | 0.388 | 0.818 |
| INTC | engineered_logistic | 406,621 | 174,209 | 0.292 | 0.768 | 0.970 |
| INTC | queue_imbalance_logistic | 406,621 | 174,209 | 0.287 | 0.754 | 0.945 |
| INTC | random_forest | 406,621 | 174,209 | 0.377 | 0.781 | 0.759 |
| INTC | training_class_prior | 406,621 | 174,209 | 0.326 | 0.333 | 0.224 |
| INTC | xgboost | 406,621 | 174,209 | 0.402 | 0.381 | 0.173 |
| MSFT | engineered_logistic | 416,960 | 178,640 | 0.344 | 0.760 | 0.927 |
| MSFT | queue_imbalance_logistic | 416,960 | 178,640 | 0.332 | 0.726 | 0.933 |
| MSFT | random_forest | 416,960 | 178,640 | 0.427 | 0.771 | 0.738 |
| MSFT | training_class_prior | 416,960 | 178,640 | 0.322 | 0.333 | 0.311 |
| MSFT | xgboost | 416,960 | 178,640 | 0.504 | 0.462 | 0.224 |

![Walk-forward classification scores by ticker](chronological_macro_f1.png)

## Taker-touch cost sensitivity

Predicted directional classes are entered at the next event's displayed best price and exited at the opposite best price 100 events after the signal. Trades are one share, sequential, and skipped if displayed top-level size is below one share. Fees are scenario assumptions per side; spread crossing is included in gross P&L.

This is a counterfactual benchmark, not a full simulator: it does not model queue priority, market impact, fees specific to a participant, or latency beyond the next event. It must not be read as a profitable strategy result.

| Model | Fee per side (bps) | Trades | Mean net bps/trade | Sum of sequential trade bps/share |
|---|---:|---:|---:|---:|
| engineered_logistic | 0.0 | 4,634 | -3.236 | -14994.8 |
| engineered_logistic | 5.0 | 4,634 | -13.236 | -61334.8 |
| engineered_logistic | 10.0 | 4,634 | -23.236 | -107674.8 |
| queue_imbalance_logistic | 0.0 | 4,352 | -3.454 | -15032.0 |
| queue_imbalance_logistic | 5.0 | 4,352 | -13.454 | -58552.0 |
| queue_imbalance_logistic | 10.0 | 4,352 | -23.454 | -102072.0 |
| random_forest | 0.0 | 4,296 | -3.149 | -13526.6 |
| random_forest | 5.0 | 4,296 | -13.149 | -56486.6 |
| random_forest | 10.0 | 4,296 | -23.149 | -99446.6 |
| training_class_prior | 0.0 | 1,692 | -3.129 | -5294.0 |
| training_class_prior | 5.0 | 1,692 | -13.129 | -22214.0 |
| training_class_prior | 10.0 | 1,692 | -23.129 | -39134.0 |
| xgboost | 0.0 | 2,262 | -2.665 | -6027.9 |
| xgboost | 5.0 | 2,262 | -12.665 | -28647.9 |
| xgboost | 10.0 | 2,262 | -22.665 | -51267.9 |

![Mean net result per trade under fee scenarios](taker_touch_fee_sensitivity.png)

## Reproduction

```powershell
pip install -e ".[microstructure]"
python -m src.run_lobster_sample
```

The runner downloads the official sample ZIPs, verifies SHA-256, compiles the C++ feature extractor, and writes metrics and figures under `results/market_microstructure/lobster_sample/`.

## Source and data handling

- LOBSTER public sample files: [official sample page](https://php.lobsterdata.com/info/DataSamples.php).
- LOBSTER access and terms: [access options](https://php.lobsterdata.com/info/AccessOptions.php) and [terms](https://php.lobsterdata.com/termsAndConditions.php).
- The raw sample files are not committed or redistributed. Check the source terms before using the data outside this demonstration.

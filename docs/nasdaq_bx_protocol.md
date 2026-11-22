# Multi-session Nasdaq BX study protocol

## Research question

Do short-horizon limit-order-book features provide repeatable predictive information across separate market sessions, and does the information survive a simple order-level execution check?

## Data and splits

The study uses the official Nasdaq BX TotalView-ITCH 5.0 sample files. It focuses on AAPL, MSFT, and INTC and reconstructs the public order book from the event stream. Raw archives remain local under the git-ignored `data/` directory and are not redistributed by this repository. Nasdaq's public sample directory does not state an open data license; users must follow Nasdaq's applicable data-use terms. Nasdaq identifies this endpoint as its ITCH sample source in its [official FAQ](https://www.nasdaqtrader.com/Content/TechnicalSupport/FAQs/ITCH_FAQ.pdf). The message format follows Nasdaq's [TotalView-ITCH 5.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification_5.0.pdf).

| Partition | Sessions | Use |
|---|---|---|
| Train | 2019-01-30, 2019-05-30, 2019-07-30 | Fit model parameters |
| Validation | 2019-08-30, 2019-10-30 | Select one model by mean per-day log loss; macro-F1 breaks ties |
| Final test | 2019-12-30, 2020-01-30, 2020-03-02 | Evaluate the frozen model choices once |

Dates are assigned before downloading or inspecting the held-out prediction results. The three final dates are not used for feature or model selection. The files are exchange sample sessions, not a continuous daily panel; conclusions must be limited to these sessions and the BX venue.

## Sampling, features, and target

The reconstructed book is sampled at 100 ms intervals during the regular session. Features include quoted spread, top-level and five-level queue imbalance, microprice displacement, mid-price returns over 100 ms, one second, and five seconds, one-second realized volatility, and time of day. Symbol indicators are included. Rolling features and labels are computed within each symbol-session; the one-second forward horizon never crosses a session boundary.

The target is the one-second mid-price move: down or up when the move exceeds one cent in ITCH price units, otherwise flat. The final ten snapshots in each symbol-session are excluded because their full target horizon is unavailable.

## Models and selection

The predeclared candidates are a training class-prior baseline, logistic regression on queue imbalance, logistic regression on the full feature set, random forest, and XGBoost. Model parameters are fixed in the run metadata. The validation winner is chosen by mean log loss across date-symbol groups, with mean macro-F1 as the tie-break. The selected model is refit using training and validation dates only. No final-test outcome is used to change its features, parameters, or selection rule.

Reported metrics are macro-F1, log loss, multiclass Brier score, and ten-bin top-label expected calibration error. Final results are shown per session and symbol, with the mean and range across the three held-out dates. Three test sessions are too few for a meaningful claim of statistical significance; no event-level confidence interval is presented as if events were independent.

## Passive execution stress

On the final sessions, the frozen model and the queue-imbalance baseline generate signals at each 100 ms snapshot. A 10-share hypothetical order joins the decision-time best bid for an up prediction or best ask for a down prediction. Entry delays are 1, 10, and 50 ms; the quote price remains fixed during delay. Orders expire after one second, and positions do not overlap within a symbol.

The queue begins behind every displayed order already present at that price. Order-ID executions, partial executions, cancellations, deletions, and replacements update queue-ahead volume; later same-price orders join behind the simulated order. The first five displayed levels are swept at exit, with any residual quantity marked to mid. Fees are stress-tested at 0, 1, and 5 basis points per side.

This remains a counterfactual. It omits hidden liquidity, market impact from the hypothetical order, exchange acknowledgements and clock uncertainty, participant-specific rebates and fees, and interaction among simultaneous orders. Observed later-order executions while the modeled queue-ahead remains are counted as priority anomalies and do not fill the virtual order. The output is a queue-aware stress diagnostic, not a production-grade fill model or evidence of profitability.

## Reproduction

Install the optional dependencies and run development before fetching the final-test files:

```powershell
python -m pip install -e ".[nasdaq_itch]"
python -m src.nasdaq_itch_study download --split development
python -m src.nasdaq_itch_study extract --split development
python -m src.nasdaq_itch_study development
python -m src.nasdaq_itch_study extract --split validation --include-order-events
python -m src.nasdaq_queue_simulation --split validation
```

After inspecting validation results and retaining the generated `protocol_lock.json`, retrieve and process the final period:

```powershell
python -m src.nasdaq_itch_study download --split test
python -c "from src.nasdaq_itch_study import SESSIONS, extract_session; [extract_session(s, include_order_events=True) for s in SESSIONS if s.split == 'test']"
python -m src.nasdaq_itch_study final
python -m src.nasdaq_queue_simulation
```

The compressed raw file sizes and SHA-256 hashes are written to `results/nasdaq_bx_itch/data_provenance.json`; feature snapshots and order-event replay files are generated locally and remain untracked. MeatPy 0.5.0 (BSD-3-Clause) is used for ITCH parsing and order-book reconstruction.

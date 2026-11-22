"""Build the public-facing report and figures for the Nasdaq BX ITCH study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "nasdaq_bx_itch"
FIGURES = RESULTS / "figures"
PDF = ROOT / "output" / "pdf" / "nasdaq_bx_multisession_report.pdf"
MODEL_ORDER = ["training_class_prior", "queue_imbalance_logistic", "engineered_logistic", "random_forest", "xgboost"]
MODEL_LABELS = {
    "training_class_prior": "Class prior", "queue_imbalance_logistic": "Queue logistic",
    "engineered_logistic": "Feature logistic", "random_forest": "Random forest", "xgboost": "XGBoost",
}
COLORS = dict(zip(MODEL_ORDER, ["#8a94a6", "#3b82f6", "#14b8a6", "#f59e0b", "#7c3aed"]))


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby("model").agg(
        macro_f1=("macro_f1", "mean"), log_loss=("log_loss", "mean"),
        brier=("multiclass_brier", "mean"), ece=("top_label_ece_10bin", "mean"),
        log_loss_day_min=("log_loss", "min"), log_loss_day_max=("log_loss", "max"),
    ).reindex(MODEL_ORDER)


def _queue_summary(queue: pd.DataFrame, selected: str) -> pd.DataFrame:
    chosen = queue[queue.model.isin([selected, "queue_imbalance_logistic"]) &
                   (queue.fee_bps_per_side == 0)].copy()
    chosen["gross_notional"] = chosen.mean_gross_bps_per_filled_trade.fillna(0) * chosen.n_orders_with_entry_fill
    grouped = chosen.groupby(["model", "delay_ms"], as_index=False).agg(
        n_signals=("n_signals", "sum"), fills=("n_orders_with_entry_fill", "sum"),
        gross_notional=("gross_notional", "sum"),
    )
    grouped["fill_probability"] = grouped.fills / grouped.n_signals.clip(lower=1)
    grouped["gross_bps"] = grouped.gross_notional / grouped.fills.clip(lower=1)
    return grouped


def _make_figures(metrics: pd.DataFrame, shift: pd.DataFrame, queue: pd.DataFrame,
                  selected: str) -> dict[str, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics_day = metrics.groupby(["session_date", "model"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), log_loss=("log_loss", "mean")
    )
    dates = sorted(metrics_day.session_date.unique())
    x = np.arange(len(dates))
    width = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for j, model in enumerate(MODEL_ORDER):
        vals = metrics_day[metrics_day.model == model].set_index("session_date").reindex(dates)
        offset = (j - (len(MODEL_ORDER) - 1) / 2) * width
        axes[0].bar(x + offset, vals.macro_f1, width, label=MODEL_LABELS[model], color=COLORS[model])
        axes[1].bar(x + offset, vals.log_loss, width, label=MODEL_LABELS[model], color=COLORS[model])
    axes[0].set_title("Macro-F1 by held-out session")
    axes[0].set_ylabel("Higher is better")
    axes[1].set_title("Log loss by held-out session")
    axes[1].set_ylabel("Lower is better")
    for ax in axes:
        ax.set_xticks(x, [d[2:] for d in dates])
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="best")
    fig.suptitle("Frozen models on the three-session final period", fontsize=14, fontweight="bold")
    metric_figure = FIGURES / "final_test_model_scores.png"
    fig.savefig(metric_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    calibration = metrics[metrics.model == selected].groupby(
        ["session_date", "ticker"], as_index=False
    ).top_label_ece_10bin.mean()
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    labels = [f"{d[2:]} {ticker}" for d, ticker in zip(calibration.session_date, calibration.ticker)]
    ax.bar(np.arange(len(calibration)), calibration.top_label_ece_10bin, color="#0891b2")
    ax.set_xticks(np.arange(len(calibration)), labels, rotation=35, ha="right")
    ax.set_ylabel("Ten-bin top-label ECE (lower is better)")
    ax.set_title(f"Calibration error by session and symbol: {MODEL_LABELS[selected]}")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    calibration_figure = FIGURES / "final_test_calibration.png"
    fig.savefig(calibration_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    shift_day = shift.groupby("session_date", as_index=False).agg(
        median_psi=("median_feature_psi", "median"), p90_psi=("p90_feature_psi", "median")
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    x = np.arange(len(shift_day))
    ax.bar(x - .18, shift_day.median_psi, .36, color="#2563eb", label="Median feature PSI")
    ax.bar(x + .18, shift_day.p90_psi, .36, color="#f97316", label="Median across groups of p90 PSI")
    ax.set_xticks(x, shift_day.session_date)
    ax.set_ylabel("Population stability index (descriptive)")
    ax.set_title("Feature-distribution change versus training data")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    shift_figure = FIGURES / "final_test_feature_shift.png"
    fig.savefig(shift_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)

    qsummary = _queue_summary(queue, selected)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for model in list(dict.fromkeys([selected, "queue_imbalance_logistic"])):
        g = qsummary[qsummary.model == model].sort_values("delay_ms")
        axes[0].plot(g.delay_ms, g.fill_probability * 100, marker="o", label=MODEL_LABELS[model], color=COLORS[model])
        axes[1].plot(g.delay_ms, g.gross_bps, marker="o", label=MODEL_LABELS[model], color=COLORS[model])
    axes[0].set_title("Any entry fill")
    axes[0].set_ylabel("Filled orders / considered signals (%)")
    axes[1].set_title("Marked result on filled orders")
    axes[1].set_ylabel("Gross basis points per filled trade")
    for ax in axes:
        ax.set_xlabel("Entry delay (ms)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    queue_figure = FIGURES / "final_test_queue_stress.png"
    fig.savefig(queue_figure, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"scores": metric_figure, "calibration": calibration_figure,
            "shift": shift_figure, "queue": queue_figure}


def _write_markdown(metrics: pd.DataFrame, shift: pd.DataFrame, queue: pd.DataFrame,
                    lock: dict, figures: dict[str, Path]) -> Path:
    summary = _summarize(metrics)
    day = metrics.groupby(["session_date", "model"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), log_loss=("log_loss", "mean"),
        multiclass_brier=("multiclass_brier", "mean"),
    )
    selected = lock["selected_model"]
    xgb_day = day[day.model == selected].set_index("session_date")
    prior_day = day[day.model == "training_class_prior"].set_index("session_date")
    win_count = int((xgb_day.log_loss < prior_day.log_loss).sum())
    shift_summary = shift.groupby("session_date").agg(
        median_psi=("median_feature_psi", "median"), p90_psi=("p90_feature_psi", "median"),
        max_psi=("max_feature_psi", "max"),
    )
    queue_zero = queue[(queue.model == selected) & (queue.fee_bps_per_side == 0)].copy()
    total_signals = int(queue_zero.n_signals.sum())
    total_fills = int(queue_zero.n_orders_with_entry_fill.sum())
    weighted_gross = float((queue_zero.mean_gross_bps_per_filled_trade.fillna(0) *
                            queue_zero.n_orders_with_entry_fill).sum() / max(total_fills, 1))
    table = ["| Model | Mean macro-F1 | Mean log loss | Mean Brier | Mean top-label ECE | Log-loss range by group |",
             "|---|---:|---:|---:|---:|---:|"]
    for model, row in summary.iterrows():
        if model not in summary.dropna(how="all").index:
            continue
        table.append(f"| {MODEL_LABELS[model]} | {row.macro_f1:.3f} | {row.log_loss:.3f} | {row.brier:.3f} | {row.ece:.3f} | {row.log_loss_day_min:.3f}-{row.log_loss_day_max:.3f} |")
    day_rows = ["| Final date | Selected-model log loss | Class-prior log loss | Selected macro-F1 | Prior macro-F1 |",
                "|---|---:|---:|---:|---:|"]
    for date in sorted(xgb_day.index):
        day_rows.append(f"| {date} | {xgb_day.loc[date, 'log_loss']:.3f} | {prior_day.loc[date, 'log_loss']:.3f} | {xgb_day.loc[date, 'macro_f1']:.3f} | {prior_day.loc[date, 'macro_f1']:.3f} |")
    selected_calibration = metrics[metrics.model == selected].groupby(
        ["session_date", "ticker"], as_index=False
    ).agg(top_label_ece_10bin=("top_label_ece_10bin", "mean"))
    calibration_rows = ["| Final date | Symbol | Top-label ECE (10 bins) |", "|---|---|---:|"]
    for row in selected_calibration.itertuples(index=False):
        calibration_rows.append(f"| {row.session_date} | {row.ticker} | {row.top_label_ece_10bin:.3f} |")
    selected_queue = queue[queue.model == selected].copy()
    selected_queue["gross_notional"] = selected_queue.mean_gross_bps_per_filled_trade.fillna(0) * selected_queue.n_orders_with_entry_fill
    selected_queue["net_notional"] = selected_queue.mean_net_bps_per_filled_trade.fillna(0) * selected_queue.n_orders_with_entry_fill
    fee_summary = selected_queue.groupby(["delay_ms", "fee_bps_per_side"], as_index=False).agg(
        signals=("n_signals", "sum"), fills=("n_orders_with_entry_fill", "sum"),
        gross_notional=("gross_notional", "sum"), net_notional=("net_notional", "sum"),
    )
    fee_rows = ["| Delay (ms) | Fee per side (bps) | Fill probability | Filled orders | Gross bps / fill | Net bps / fill |",
                "|---:|---:|---:|---:|---:|---:|"]
    for row in fee_summary.itertuples(index=False):
        fee_rows.append(f"| {row.delay_ms:g} | {row.fee_bps_per_side:g} | {row.fills / max(row.signals, 1):.3%} | {row.fills} | {row.gross_notional / max(row.fills, 1):.2f} | {row.net_notional / max(row.fills, 1):.2f} |")
    psi_rows = ["| Final date | Median of group median PSI | Median of group p90 PSI | Largest group PSI |",
                "|---|---:|---:|---:|"]
    for date, row in shift_summary.iterrows():
        psi_rows.append(f"| {date} | {row.median_psi:.3f} | {row.p90_psi:.3f} | {row.max_psi:.3f} |")
    queue_fill = total_fills / total_signals if total_signals else 0.0
    text = f"""# Multi-session Nasdaq BX order-book study

## Research question

Do short-horizon limit-order-book features provide repeatable predictive information across separate market sessions, and does that information survive a queue-aware execution stress?

## Executive result

The model was selected before the final period: **{MODEL_LABELS[selected]}** had the lowest validation log loss on the two validation sessions. On the three predeclared final sessions ({', '.join(sorted(metrics.session_date.unique()))}), it had lower average log loss than the training class-prior reference on **{win_count} of 3 dates**. Its average macro-F1 was {summary.loc[selected, 'macro_f1']:.3f}, versus {summary.loc['training_class_prior', 'macro_f1']:.3f} for the class prior. The result supports measurable short-horizon probabilistic structure in these samples; it does not establish a persistent market signal or trading profitability.

The passive replay produced **{total_fills:,} filled 10-share orders from {total_signals:,} considered signals ({queue_fill:.3%})** for the selected model at zero fees, across the specified delay scenarios. The fill-weighted gross marked result was {weighted_gross:.2f} bps per filled order. This counterfactual is conditional on the observed fills and is not an estimate of deployable strategy performance.

## Data and frozen protocol

- Venue/source: official Nasdaq BX TotalView-ITCH 5.0 public sample archives. BX is a Nasdaq venue and this is not a consolidated US market feed.
- Instruments: AAPL, MSFT, and INTC. Training sessions: 2019-01-30, 2019-05-30, 2019-07-30. Validation: 2019-08-30 and 2019-10-30. Final test: 2019-12-30, 2020-01-30, and 2020-03-02.
- The book is sampled every 100 ms; the target is the three-class mid-price move one second later, with moves within one cent labeled flat. Rolling features and labels stay within each ticker-session.
- The fixed candidates were a class-prior reference, queue-imbalance logistic regression, engineered logistic regression, random forest, and XGBoost. The winner was chosen by mean validation log loss across date-symbol groups, with macro-F1 as a tie-break. The selected candidate and all comparisons were refit on train plus validation; the final period was evaluated once.
- Metrics: macro-F1, log loss, multiclass Brier score, and ten-bin top-label expected calibration error. The three dates are the evaluation units; per-date results are descriptive and no event-level significance test is claimed.

## Final-period model scores

{chr(10).join(table)}

This table averages nine date-symbol cells equally. The log-loss range is the minimum and maximum across those cells. The adjacent one-second labels overlap, so sample counts are not independent observations.

![Final-period scores by session](figures/{figures['scores'].name})

### Probability calibration

Ten-bin top-label expected calibration error (ECE) summarizes the gap between confidence and observed correctness. It is descriptive; adjacent predictions are dependent.

{chr(10).join(calibration_rows)}

![Calibration error by date and ticker](figures/{figures['calibration'].name})

### Selected model and class-prior comparison by date

{chr(10).join(day_rows)}

## Distribution shift

PSI compares each final date's feature distributions with reference bins fit on the combined training and validation snapshots. It is a diagnostic, not a causal explanation of model performance. Values in the table summarize across three symbols; the largest PSI calls attention to one feature/group and is sensitive to binning.

{chr(10).join(psi_rows)}

![Final-period feature shift](figures/{figures['shift'].name})

## Queue-aware passive execution stress

The simulation posts a 10-share order at the decision-time best quote, tests 1, 10, and 50 ms entry delays and 0, 1, and 5 bps per-side fees, and expires each order after one second. The queue starts behind the displayed order IDs at that price. Public executions, partial executions, cancellations, deletions, and replacements update queue-ahead volume; later orders join behind. The exit sweeps up to five displayed levels and marks any residual at mid. Positions do not overlap by ticker.

![Fill and gross marked result by delay at zero fees](figures/{figures['queue'].name})

### Fee sensitivity for the selected model

{chr(10).join(fee_rows)}

These results are a deliberately limited counterfactual. They omit hidden liquidity, exchange acknowledgements and clock error, impact from the hypothetical order, participant-specific fees/rebates, and interactions with other orders. A negative conditional gross result and low fill rate are evidence that classification scores alone are insufficient; they do not prove a strategy cannot be designed.

## Data access and reproduction

Raw exchange files are not redistributed here. Download them only under Nasdaq's applicable data-use terms. The public sample directory does not state an open-data license. File sizes and SHA-256 digests are recorded in `data_provenance.json`; raw archives and generated order-event logs remain in the git-ignored `data/` directory. The protocol lock records model settings, package versions, and the test evaluation hashes.

```powershell
python -m pip install -e ".[nasdaq_itch]"
python -m pip install "reportlab>=4.0"
python -m src.nasdaq_itch_study download --split development
python -m src.nasdaq_itch_study extract --split train
python -m src.nasdaq_itch_study extract --split validation --include-order-events
python -m src.nasdaq_itch_study development
python -m src.nasdaq_queue_simulation --split validation
python -m src.nasdaq_itch_study download --split test
python -m src.nasdaq_itch_study extract --split test --include-order-events
python -m src.nasdaq_itch_study final
python -c "from pathlib import Path; from src.nasdaq_queue_simulation import run_queue_sensitivity; run_queue_sensitivity(predictions_path=Path('results/nasdaq_bx_itch/final_test_predictions.parquet'))"
python -m src.generate_nasdaq_bx_report
```

Source: [Nasdaq BX ITCH sample directory](https://emi.nasdaq.com/ITCH/Nasdaq%20BX%20ITCH/), [Nasdaq ITCH FAQ](https://www.nasdaqtrader.com/Content/TechnicalSupport/FAQs/ITCH_FAQ.pdf), and [ITCH 5.0 specification](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification_5.0.pdf). MeatPy 0.5.0 is used for message parsing and book reconstruction; its license and data notes are in `THIRD_PARTY_NOTICES.md`.
"""
    out = RESULTS / "report.md"
    out.write_text(text, encoding="utf-8")
    return out


def _make_pdf(metrics: pd.DataFrame, shift: pd.DataFrame, queue: pd.DataFrame,
              lock: dict, figures: dict[str, Path]) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image, PageBreak, Paragraph,
                                   SimpleDocTemplate, Spacer, Table, TableStyle)

    summary = _summarize(metrics)
    selected = lock["selected_model"]
    day = metrics.groupby(["session_date", "model"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), log_loss=("log_loss", "mean")
    )
    date_rows = [["Date", "Selected log loss", "Prior log loss", "Selected macro-F1", "Prior macro-F1"]]
    for date in sorted(day.session_date.unique()):
        s = day[(day.session_date == date) & (day.model == selected)].iloc[0]
        b = day[(day.session_date == date) & (day.model == "training_class_prior")].iloc[0]
        date_rows.append([date, f"{s.log_loss:.3f}", f"{b.log_loss:.3f}", f"{s.macro_f1:.3f}", f"{b.macro_f1:.3f}"])
    score_rows = [["Frozen model", "Macro-F1", "Log loss", "Brier", "Top-label ECE"]]
    for model, row in summary.iterrows():
        score_rows.append([MODEL_LABELS[model], f"{row.macro_f1:.3f}", f"{row.log_loss:.3f}", f"{row.brier:.3f}", f"{row.ece:.3f}"])
    shift_rows = [["Date", "Median PSI", "Group p90 PSI", "Largest PSI"]]
    for date, row in shift.groupby("session_date").agg(
        median=("median_feature_psi", "median"), p90=("p90_feature_psi", "median"),
        maximum=("max_feature_psi", "max"),
    ).iterrows():
        shift_rows.append([date, f"{row['median']:.3f}", f"{row['p90']:.3f}", f"{row['maximum']:.3f}"])

    doc = SimpleDocTemplate(str(PDF), pagesize=letter, rightMargin=.62*inch,
                            leftMargin=.62*inch, topMargin=.62*inch, bottomMargin=.6*inch,
                            title="Multi-session Nasdaq BX order-book study",
                            author="FUTURE research project")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=23, leading=27, textColor=colors.HexColor("#14213d"), alignment=TA_LEFT, spaceAfter=9))
    styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontSize=10.5, leading=15,
                              textColor=colors.HexColor("#4b5563"), spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=14, leading=17,
                              textColor=colors.HexColor("#14213d"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.8, leading=12.2, spaceAfter=6))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8, leading=10,
                              textColor=colors.HexColor("#5b6472"), spaceAfter=8))
    story = [Paragraph("Multi-session order-book study", styles["ReportTitle"]),
             Paragraph("FUTURE | Financial uncertainty testing under regime evolution", styles["Deck"]),
             Paragraph("Question: do short-horizon order-book features carry predictive information across separate market sessions, and does it survive an order-level execution stress?", styles["BodySmall"]),
             Paragraph("Result in brief", styles["Section"])]
    story.append(Paragraph(
        f"The validation protocol selected <b>{MODEL_LABELS[selected]}</b> by mean log loss before the final period was opened. On the three reserved sessions, its mean macro-F1 was {summary.loc[selected, 'macro_f1']:.3f}, compared with {summary.loc['training_class_prior', 'macro_f1']:.3f} for a training class-prior forecast. The result demonstrates measurable forecast structure in this limited sample, not a durable signal or trading edge.", styles["BodySmall"]))
    story.extend([Image(str(figures["scores"]), width=7.15*inch, height=2.86*inch),
                  Paragraph("Scores are averaged over three symbols within each date. Macro-F1 and log loss answer different questions; read both panels.", styles["Caption"]),
                  Paragraph("Final-period comparison", styles["Section"])])
    score_table = Table(score_rows, colWidths=[2.25*inch, .9*inch, .9*inch, .8*inch, 1.15*inch], repeatRows=1)
    score_table.setStyle(_table_style(colors))
    story.extend([score_table, Spacer(1, 8), Paragraph("Selected model versus the class-prior reference by date", styles["Section"])])
    date_table = Table(date_rows, colWidths=[1.2*inch, 1.15*inch, 1.15*inch, 1.25*inch, 1.15*inch], repeatRows=1)
    date_table.setStyle(_table_style(colors))
    selected_queue = queue[queue.model == selected].copy()
    selected_zero_fee = selected_queue[selected_queue.fee_bps_per_side == 0]
    selected_fills = int(selected_zero_fee.n_orders_with_entry_fill.sum())
    selected_signals = int(selected_zero_fee.n_signals.sum())
    selected_fill_rate = selected_fills / max(selected_signals, 1)
    selected_gross = float((selected_zero_fee.mean_gross_bps_per_filled_trade.fillna(0) *
                            selected_zero_fee.n_orders_with_entry_fill).sum() / max(selected_fills, 1))
    selected_queue["gross_total"] = selected_queue.mean_gross_bps_per_filled_trade.fillna(0) * selected_queue.n_orders_with_entry_fill
    selected_queue["net_total"] = selected_queue.mean_net_bps_per_filled_trade.fillna(0) * selected_queue.n_orders_with_entry_fill
    fee_table_rows = [["Delay", "Fee/side", "Fill rate", "Fills", "Gross bps", "Net bps"]]
    fee_table = selected_queue.groupby(["delay_ms", "fee_bps_per_side"], as_index=False).agg(
        signals=("n_signals", "sum"), fills=("n_orders_with_entry_fill", "sum"),
        gross=("gross_total", "sum"), net=("net_total", "sum"),
    )
    for row in fee_table.itertuples(index=False):
        fee_table_rows.append([f"{row.delay_ms:g} ms", f"{row.fee_bps_per_side:g} bps",
                               f"{row.fills / max(row.signals, 1):.2%}", f"{row.fills:,}",
                               f"{row.gross / max(row.fills, 1):.2f}", f"{row.net / max(row.fills, 1):.2f}"])
    fee_table_obj = Table(fee_table_rows, colWidths=[.85*inch, 1.0*inch, .9*inch, .75*inch, .9*inch, .85*inch], repeatRows=1)
    fee_table_obj.setStyle(_table_style(colors))
    story.extend([date_table, Spacer(1, 6), Paragraph(
        "The final dates were assigned before model selection. No final-period metric was used to choose models, features, thresholds, or settings.", styles["Caption"]),
        PageBreak(), Paragraph("Calibration and distribution shift", styles["ReportTitle"]),
        Paragraph("Probability calibration", styles["Section"]),
        Paragraph("Ten-bin top-label ECE measures the difference between confidence and observed correctness. It is a compact descriptive diagnostic; adjacent predictions are dependent.", styles["BodySmall"]),
        Image(str(figures["calibration"]), width=6.6*inch, height=2.78*inch),
        Paragraph("Feature shift", styles["Section"]),
        Paragraph("PSI compares each final session's feature distributions against bins fit on training plus validation snapshots. It is descriptive and depends on the selected binning; it does not identify the cause of a performance change.", styles["BodySmall"]),
        Image(str(figures["shift"]), width=6.6*inch, height=3.12*inch)])
    psi_table = Table(shift_rows, colWidths=[1.6*inch, 1.35*inch, 1.35*inch, 1.35*inch], repeatRows=1)
    psi_table.setStyle(_table_style(colors))
    story.extend([psi_table, PageBreak(), Paragraph("Queue-aware passive execution stress", styles["ReportTitle"]),
                  Paragraph("The simulation posts a 10-share limit order at the decision-time best quote. Public order IDs are replayed to maintain queue-ahead volume through executions, cancellations, deletions, and replacements. It tests 1, 10, and 50 ms entry delays, one-second expiry, and a five-level exit sweep with residual inventory marked to mid.", styles["BodySmall"]),
                  Image(str(figures["queue"]), width=6.8*inch, height=2.85*inch),
                  Paragraph("Selected-model fee and delay sensitivity", styles["Section"]), fee_table_obj,
                  Paragraph(f"The selected model generated {selected_fills:,} fills from {selected_signals:,} considered signals ({selected_fill_rate:.3%}); the fill-weighted gross marked result was {selected_gross:.2f} bps per filled order at zero fees. This is a counterfactual, not a production fill model. It omits hidden liquidity, exchange acknowledgements and clock error, market impact, participant-specific fees and rebates, and interactions with other orders.", styles["BodySmall"]),
                  Paragraph("Scope", styles["Section"]),
                  Paragraph("Nasdaq BX ITCH 5.0 samples: three training dates, two validation dates, and three untouched final dates from January 2019 to March 2020. The study covers AAPL, MSFT, and INTC at 100 ms snapshots and predicts the three-class one-second mid-price move. The final sample contains only three exchange sessions; date-level comparisons are descriptive, and no event-level significance claim is made.", styles["BodySmall"]),
                  Paragraph("Raw archives are not included. SHA-256 data provenance and locked settings are in the repository results; Nasdaq's public sample directory does not state an open-data license. Follow Nasdaq's applicable terms. Sources: <link href='https://emi.nasdaq.com/ITCH/Nasdaq%20BX%20ITCH/' color='#2563eb'>Nasdaq BX sample directory</link>, <link href='https://www.nasdaqtrader.com/Content/TechnicalSupport/FAQs/ITCH_FAQ.pdf' color='#2563eb'>ITCH FAQ</link>, and <link href='https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification_5.0.pdf' color='#2563eb'>ITCH 5.0 specification</link>. Reproduction commands and complete caveats are in the repository README.", styles["BodySmall"])])

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7dde5"))
        canvas.line(.62*inch, .46*inch, 7.88*inch, .46*inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5b6472"))
        canvas.drawString(.62*inch, .29*inch, "FUTURE | Nasdaq BX ITCH multi-session study")
        canvas.drawRightString(7.88*inch, .29*inch, f"{document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF


def _table_style(colors):
    from reportlab.platypus import TableStyle
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14213d")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d7dde5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])


def main() -> None:
    metrics = pd.read_csv(RESULTS / "final_test_by_day.csv")
    shift = pd.read_csv(RESULTS / "final_test_feature_shift.csv")
    queue = pd.read_csv(RESULTS / "passive_queue_sensitivity.csv")
    lock = json.loads((RESULTS / "protocol_lock.json").read_text(encoding="utf-8"))
    if not (RESULTS / "final_test_evaluated.json").exists():
        raise FileNotFoundError("Run the locked final evaluation once before building this report")
    figures = _make_figures(metrics, shift, queue, lock["selected_model"])
    markdown = _write_markdown(metrics, shift, queue, lock, figures)
    pdf = _make_pdf(metrics, shift, queue, lock, figures)
    print(f"Wrote {markdown.relative_to(ROOT)}")
    print(f"Wrote {pdf.relative_to(ROOT)}")
    for name, path in figures.items():
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

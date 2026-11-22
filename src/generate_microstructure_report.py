"""Create a reproducible report and figures from FI-2010 experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from src.market_microstructure import CLASS_LABELS, evaluate, feature_psi_by_fold, load_matrix
from src.microstructure_inference import build_inference


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/market_microstructure"
DATA = ROOT / "data/market_microstructure/processed"


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "training_class_prior": ("training_prior_pred", ["p_prior_up", "p_prior_stationary", "p_prior_down"]),
        "logistic_regression": ("logistic_regression_pred", ["logistic_regression_p_up", "logistic_regression_p_stationary", "logistic_regression_p_down"]),
        "random_forest": ("random_forest_pred", ["random_forest_p_up", "random_forest_p_stationary", "random_forest_p_down"]),
        "xgboost": ("xgboost_pred", ["xgboost_p_up", "xgboost_p_stationary", "xgboost_p_down"]),
    }
    target = predictions["y_true"].to_numpy(dtype=int)
    results = []
    for name, (pred_col, probability_cols) in columns.items():
        values = evaluate(target, predictions[pred_col].to_numpy(dtype=int), predictions[probability_cols].to_numpy(dtype=float))
        results.append({"model": name, "n_test": len(target), **values})
    return pd.DataFrame(results)


def refresh_shift_diagnostics() -> pd.DataFrame:
    rows = []
    for fold in range(1, 10):
        train_x, _ = load_matrix(DATA / f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt")
        test_x, _ = load_matrix(DATA / f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt")
        psi = feature_psi_by_fold(train_x, test_x)
        rows.append({"fold": fold, "test_day": fold + 1,
                     "mean_feature_psi": float(np.mean(psi)),
                     "median_feature_psi": float(np.median(psi)),
                     "p90_feature_psi": float(np.quantile(psi, 0.90)),
                     "max_feature_psi": float(np.max(psi))})
        del train_x, test_x
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "feature_shift.csv", index=False)
    return frame


def make_figures(fold_metrics: pd.DataFrame, shift: pd.DataFrame, predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors_by_model = {"training_class_prior": "#64748b", "logistic_regression": "#2563eb", "random_forest": "#d97706", "xgboost": "#15803d"}
    for model, group in fold_metrics.groupby("model"):
        group = group.sort_values("test_day")
        ax.plot(group["test_day"], group["macro_f1"], marker="o", linewidth=2, label=model.replace("_", " "), color=colors_by_model[model])
    ax.set(title="Held-out macro-F1 across anchored test days", xlabel="Test day (benchmark order)", ylabel="Macro-F1", xticks=range(2, 11))
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "walk_forward_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    dist = predictions.groupby("test_day")["y_true"].value_counts(normalize=True).unstack(fill_value=0).reindex(columns=CLASS_LABELS, fill_value=0)
    ax.stackplot(dist.index, dist[1], dist[2], dist[3], labels=["Up", "Stationary", "Down"], colors=["#16a34a", "#94a3b8", "#dc2626"], alpha=0.88)
    ax.set(title="Held-out label mix changes across days", xlabel="Test day (benchmark order)", ylabel="Share of events", xticks=range(2, 11), ylim=(0, 1))
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / "test_label_mix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)
    axes[0].plot(shift["test_day"], shift["median_feature_psi"], marker="o", color="#2563eb")
    axes[0].set_ylabel("Median PSI")
    axes[1].plot(shift["test_day"], shift["p90_feature_psi"], marker="s", color="#b45309")
    axes[1].set_ylabel("90th percentile PSI")
    axes[1].set_xlabel("Test day (benchmark order)")
    axes[1].set_xticks(range(2, 11))
    fig.suptitle("Train-to-test feature shift by anchored fold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_psi_by_day.png", dpi=180)
    plt.close(fig)


def write_markdown(fold_metrics: pd.DataFrame, pooled: pd.DataFrame, shifts: pd.DataFrame,
                   metadata: dict, inference: pd.DataFrame,
                   shift_association: pd.DataFrame) -> str:
    mean_scores = fold_metrics.groupby("model").agg(mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"), mean_log_loss=("log_loss", "mean")).reset_index()
    joined = mean_scores.merge(pooled, on="model", suffixes=("_daymean", "_pooled"))
    joined = joined.sort_values("mean_macro_f1", ascending=False)
    day_table = fold_metrics.pivot(index="test_day", columns="model", values="macro_f1").reset_index()
    shift_summary = shifts[["median_feature_psi", "p90_feature_psi"]].agg(["median", "min", "max"])
    table_lines = ["| Model | Mean fold macro-F1 (SD) | Pooled macro-F1 | Pooled balanced accuracy | Pooled log loss | Pooled multiclass Brier |", "|---|---:|---:|---:|---:|---:|"]
    for r in joined.itertuples(index=False):
        table_lines.append(f"| {r.model.replace('_',' ')} | {r.mean_macro_f1:.3f} ({r.sd_macro_f1:.3f}) | {r.macro_f1:.3f} | {r.balanced_accuracy:.3f} | {r.log_loss:.3f} | {r.multiclass_brier:.3f} |")
    day_lines = ["| Test day | Prior | Logistic | Random forest | XGBoost |", "|---:|---:|---:|---:|---:|"]
    for row in day_table.itertuples(index=False):
        day_lines.append(f"| {row.test_day} | {row.training_class_prior:.3f} | {row.logistic_regression:.3f} | {row.random_forest:.3f} | {row.xgboost:.3f} |")
    inference_lines = ["| Model | Metric | Mean improvement vs class prior | 95% two-day block-bootstrap interval |",
                       "|---|---|---:|---:|"]
    for row in inference.sort_values(["model", "metric"]).itertuples(index=False):
        inference_lines.append(f"| {row.model.replace('_', ' ')} | {row.metric} | {row.mean_delta_favoring_model:.3f} | [{row.block_bootstrap_95_low:.3f}, {row.block_bootstrap_95_high:.3f}] |")
    association_lines = [f"- {r.model.replace('_', ' ')}: PSI vs macro-F1 Spearman {r.spearman_psi_macro_f1:.2f}; PSI vs log-loss Spearman {r.spearman_psi_log_loss:.2f}."
                         for r in shift_association.itertuples(index=False)]
    text = f"""# FI-2010 Limit Order Book Companion Experiment

## Research question

Do order-book features retain short-horizon predictive information when the distribution changes across trading days?

## Results at a glance

The XGBoost model had the best mean macro-F1 over the nine held-out days at **{joined.iloc[0].mean_macro_f1:.3f}** (day-to-day standard deviation {joined.iloc[0].sd_macro_f1:.3f}). Random forest was close at **{joined.loc[joined.model=='random_forest','mean_macro_f1'].iloc[0]:.3f}**; logistic regression averaged **{joined.loc[joined.model=='logistic_regression','mean_macro_f1'].iloc[0]:.3f}**. The XGBoost model led on some days and fell behind random forest on others. Thus the features carry predictive information in this benchmark, while model ranking and score still move across days.

These are classification results, not a trading-strategy result. The FI-2010 release used here supplies normalized features and labels. This experiment does not estimate fills, spread capture, fees, latency, slippage, turnover, inventory risk, or P&L.

## Test-day scores

Macro-F1 is the unweighted average of class-specific F1 scores, so each direction/stationary class contributes equally.

{chr(10).join(day_lines)}

## Pooled held-out metrics

Pooled metrics combine all {int(pooled.n_test.max()):,} test observations from days 2 through 10. The fold-mean macro-F1 above gives each test day equal weight; pooled metrics give each event equal weight.

{chr(10).join(table_lines)}

Probabilities are scored with log loss and multiclass Brier score. Lower is better for those metrics. They do not replace a full calibration analysis by day or stock.

## Experimental design

- **Data:** official FI-2010 NoAuction Z-score representation; five Nasdaq Nordic stocks, ten consecutive trading days, 144 provided features and five mid-price movement labels.
- **Target:** the 10-event-ahead label (source row 149): 1 = up, 2 = stationary, 3 = down. The source defines the labels from the average relative mid-price change over the next event horizon.
- **Validation:** the published day-based anchored folds CF_1 through CF_9. Each model trains on the fold's expanding past window and tests on the next day. The training windows range from {int(fold_metrics.n_train.min()):,} to {int(fold_metrics.n_train.max()):,} observations.
- **Models:** training-window class-prior baseline; logistic regression (C=0.1); random forest (100 trees, depth 18, leaf size 30, half-sample per tree); XGBoost (240 trees, depth 5, learning rate 0.06). Hyperparameters were fixed before evaluating these folds; no test-fold tuning was performed.
- **Metrics:** accuracy, balanced accuracy, macro precision/recall/F1, log loss and multiclass Brier score. Per-event out-of-sample predictions are retained for independent review.
- **Shift diagnostic:** decile-bin population stability index (PSI), with bin boundaries estimated on each training fold. Across folds, median feature PSI ranged {shifts.median_feature_psi.min():.3f}–{shifts.median_feature_psi.max():.3f}; the 90th percentile across features ranged {shifts.p90_feature_psi.min():.3f}–{shifts.p90_feature_psi.max():.3f}. The feature-wise tail matters: averages can be dominated by a small number of features. PSI is descriptive and does not identify a causal market regime.

## Day-block uncertainty and shift association

Paired two-day moving-block bootstrap intervals resample the nine ordered test days, not individual events. Positive improvement means higher macro-F1 or lower log loss than the training class-prior baseline.

{chr(10).join(inference_lines)}

These are descriptive uncertainty intervals, not significance tests. There are only nine folds and the expanding training sets overlap. No untouched final test period was reserved after results were explored. The PSI/performance rank associations are likewise exploratory and do not establish that measured feature shift causes performance changes:

{chr(10).join(association_lines)}

## Interpretation and limitations

1. The benchmark contains ten trading days from June 2010. Nine chronological test days are useful for a focused walk-forward exercise, but they cannot establish persistence across months, years, venues, or current market structure.
2. The public feature matrices aggregate the five stocks without a per-row security identifier in the released files used here. Results are therefore pooled and cannot support a per-stock stability claim.
3. Features are the benchmark's precomputed Z-score representation. We use the official anchored folds and do not fit an alternate raw-data feature pipeline. Normalization details inherited from the source should be considered when comparing against other datasets or production feeds.
4. Adjacent event labels overlap in their forward horizons, so event-level observations are not independent. Fold variability is descriptive; this report does not claim a formal significance test.
5. No raw order messages, executable order simulation, or full cost model is evaluated here. No profitability claim is made.
6. The longer daily ETF study remains the main multi-year distribution-shift evidence. This FI-2010 experiment adds microstructure relevance but does not replace that broader test.

## Reproducibility

The official Fairdata package is identified by dataset ID `{metadata['dataset_id']}` and SHA-256 `{metadata['package_sha256']}`. It is licensed CC BY 4.0. Download and verify it, then run:

```powershell
python -m src.download_fi2010
python -m src.market_microstructure
python -m src.microstructure_inference
python -m src.generate_microstructure_report
```

The raw package and extracted matrices are kept out of Git. Metrics, metadata, figures, compressed held-out predictions, this report, and the PDF are written under `results/market_microstructure/`.

## Sources

- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., and Iosifidis, A. (2018). “Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods.” *Journal of Forecasting*, 37(8), 852–866. [Paper](https://doi.org/10.1002/for.2543); [arXiv](https://arxiv.org/abs/1705.03233).
- Official dataset record: [Fairdata Etsin, FI-2010](https://etsin.fairdata.fi/dataset/{metadata['dataset_id']}).

"""
    (RESULTS / "report.md").write_text(text.rstrip() + "\n", encoding="utf-8")
    return text


def markdown_to_story(md: str, pooled: pd.DataFrame, fold_metrics: pd.DataFrame,
                      inference: pd.DataFrame) -> list:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=27, textColor=colors.HexColor("#17324d"), spaceAfter=18))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontSize=14, leading=18, textColor=colors.HexColor("#17324d"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=6))
    story = [Paragraph("FI-2010 Limit Order Book<br/>Companion Experiment", styles["ReportTitle"]),
             Paragraph("A chronological evaluation of short-horizon mid-price classification under day-to-day market change", styles["BodyCustom"]), Spacer(1, 0.08 * inch)]
    summary = pooled.merge(fold_metrics.groupby("model").agg(mean_fold_macro_f1=("macro_f1", "mean"), sd_fold_macro_f1=("macro_f1", "std")).reset_index(), on="model")
    summary = summary.sort_values("mean_fold_macro_f1", ascending=False)
    data = [["Model", "Fold mean F1", "Pooled F1", "Bal. acc.", "Log loss", "Brier"]]
    for r in summary.itertuples(index=False):
        data.append([r.model.replace("_", " "), f"{r.mean_fold_macro_f1:.3f} +/- {r.sd_fold_macro_f1:.3f}", f"{r.macro_f1:.3f}", f"{r.balanced_accuracy:.3f}", f"{r.log_loss:.3f}", f"{r.multiclass_brier:.3f}"])
    table = Table(data, repeatRows=1, colWidths=[1.45*inch, 1.25*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324d")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6), ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    sections = {
        "Results at a glance": "XGBoost had the best average macro-F1 over the nine held-out days, with random forest close behind. The day-level ranking changes, and all models remain imperfect. This is evidence of predictive structure in this historical benchmark, not evidence of a trading edge.",
        "Test-day scores": "The figure shows day-level macro-F1 for every anchored fold. Because each test day is kept visible, the chart exposes variation that a single pooled score would hide.",
        "Held-out label mix": "Class proportions change across the test days. A training-prior baseline is included to show the challenge created by class imbalance and changing labels.",
        "Feature distribution shift": "PSI compares each held-out day's feature distribution with training-derived decile bins. Median and 90th-percentile PSI are shown across the 144 features. This is a descriptive diagnostic; it does not assign named regimes or establish a cause.",
        "Day-block uncertainty": "The table summarizes paired model improvements over the class-prior baseline with a two-day moving-block bootstrap over the nine ordered test folds. These descriptive intervals do not account for every dependency from overlapping training windows and are not significance tests.",
        "Design and limits": "The FI-2010 dataset covers five Nasdaq Nordic stocks across ten consecutive trading days in June 2010. The experiment uses 144 precomputed Z-score features, a 10-event target, and the official anchored folds CF_1 through CF_9. Models use fixed hyperparameters and each fold's training window only. The release used here does not support per-stock results, current-market validation, latency analysis, realistic execution-cost estimates, or a P&L backtest. Adjacent event labels overlap, so event rows are not independent.",
        "Reproducibility and attribution": "Official dataset record: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649. Dataset ID 73eb48d7-4dbc-4a10-a52a-da745b47a649; Fairdata package SHA-256 bcc89a5aa7d8067dda98374393444eb885a4283a41fd33e323496380e057e1a6; license CC BY 4.0. See report.md and run_metadata.json for the complete protocol, per-fold metrics, and command lines.",
    }
    for title, body in sections.items():
        block = [Paragraph(title, styles["H2Custom"]), Paragraph(body.replace("&", "&amp;"), styles["BodyCustom"])]
        if title == "Test-day scores":
            block.append(Image(str(RESULTS / "walk_forward_macro_f1.png"), width=6.75*inch, height=3.6*inch))
        elif title == "Held-out label mix":
            block.append(Image(str(RESULTS / "test_label_mix.png"), width=6.75*inch, height=3.15*inch))
        elif title == "Feature distribution shift":
            block.append(Image(str(RESULTS / "feature_psi_by_day.png"), width=6.75*inch, height=3.4*inch))
        elif title == "Day-block uncertainty":
            data = [["Model", "Metric", "Mean improvement", "95% interval"]]
            for r in inference.sort_values(["model", "metric"]).itertuples(index=False):
                data.append([r.model.replace("_", " "), r.metric,
                             f"{r.mean_delta_favoring_model:.3f}",
                             f"[{r.block_bootstrap_95_low:.3f}, {r.block_bootstrap_95_high:.3f}]"])
            table = Table(data, repeatRows=1, colWidths=[1.6*inch, 1.05*inch, 1.6*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            block.append(table)
        story.append(KeepTogether(block))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Reference: Ntakaris et al. (2018), Journal of Forecasting, 37(8), 852-866, doi:10.1002/for.2543.", styles["BodyCustom"]))
    return story


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(0.65*inch, 0.55*inch, 7.85*inch, 0.55*inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(0.7*inch, 0.35*inch, "FI-2010 limit order book companion experiment")
    canvas.drawRightString(7.8*inch, 0.35*inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.read_csv(RESULTS / "fold_metrics.csv")
    predictions = pd.read_csv(RESULTS / "test_predictions.csv.gz")
    with (RESULTS / "run_metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    pooled = pooled_metrics(predictions)
    pooled.to_csv(RESULTS / "pooled_summary_metrics.csv", index=False)
    shifts = refresh_shift_diagnostics()
    inference, shift_association = build_inference(fold_metrics, shifts)
    inference.to_csv(RESULTS / "day_block_inference.csv", index=False, float_format="%.6f")
    shift_association.to_csv(RESULTS / "shift_performance_association.csv", index=False,
                             float_format="%.6f")
    make_figures(fold_metrics, shifts, predictions)
    md = write_markdown(fold_metrics, pooled, shifts, metadata, inference, shift_association)
    pdf_path = RESULTS / "report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.62*inch, bottomMargin=0.72*inch, title="FI-2010 Limit Order Book Companion Experiment", author="Financial Market ML Research")
    doc.build(markdown_to_story(md, pooled, fold_metrics, inference), onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Wrote {pdf_path}")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()

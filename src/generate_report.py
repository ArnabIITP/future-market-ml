"""Create a reproducible Markdown and PDF research report from experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
NAVY = "#17324D"
TEAL = "#168C8C"
ORANGE = "#D78345"
GREY = "#64748B"


def load_data():
    pred = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"])
    event = pd.read_csv(RESULTS / "high_volatility_predictions.csv", parse_dates=["date"])
    annual = pd.read_csv(RESULTS / "annual_metrics.csv")
    regime = pd.read_csv(RESULTS / "regime_metrics.csv")
    loss_tests = pd.read_csv(RESULTS / "paired_loss_tests.csv")
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    sensitivity_dir = RESULTS / "sensitivity_half_year"
    sensitivity = pd.read_csv(sensitivity_dir / "predictions.csv", parse_dates=["date"]) if (sensitivity_dir / "predictions.csv").exists() else None
    return pred, event, annual, regime, loss_tests, metadata, sensitivity


def _regression_summary(pred):
    rows = []
    for ticker, subset in pred.groupby("ticker"):
        by_model = {}
        for model, g in subset.groupby("model"):
            by_model[model] = {
                "n": len(g),
                "mae": mean_absolute_error(g.actual_volatility, g.forecast_volatility),
                "rmse": mean_squared_error(g.actual_volatility, g.forecast_volatility) ** 0.5,
            }
        base = by_model["historical_volatility"]
        for model in ("linear", "random_forest", "xgboost"):
            metric = by_model[model]
            rows.append({
                "ticker": ticker, "model": model, "n": metric["n"],
                "mae": metric["mae"], "mae_skill": (base["mae"] - metric["mae"]) / base["mae"],
                "rmse": metric["rmse"], "rmse_skill": (base["rmse"] - metric["rmse"]) / base["rmse"],
                "baseline_mae": base["mae"], "baseline_rmse": base["rmse"],
            })
    return pd.DataFrame(rows)


def _event_summary(event):
    rows = []
    for (ticker, model), g in event.groupby(["ticker", "model"]):
        y, p = g.actual_event.to_numpy(), g.probability.to_numpy()
        rows.append({
            "ticker": ticker, "model": model, "n": len(g),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "brier": brier_score_loss(y, p),
            "roc_auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
            "average_precision": average_precision_score(y, p),
            "ece_10_bins": _ece(y, p),
            "event_rate": float(y.mean()),
        })
    return pd.DataFrame(rows)


def _ece(truth, probability, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    return float(sum(
        np.mean(bucket == i) * abs(np.mean(truth[bucket == i]) - np.mean(probability[bucket == i]))
        for i in range(bins) if np.any(bucket == i)
    ))


def make_figures(pred, event, annual, regression, event_summary):
    FIGURES.mkdir(exist_ok=True)
    models = ["linear", "random_forest", "xgboost"]
    tickers = list(pred.ticker.drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.19
    x = np.arange(len(tickers))
    palette = ["#5F83A3", "#168C8C", "#D78345"]
    for i, model in enumerate(models):
        rows = regression.loc[regression.model == model].set_index("ticker").reindex(tickers)
        axes[0].bar(x + (i - 1) * width, rows.mae_skill * 100, width, label=model.replace("_", " "), color=palette[i])
        axes[1].bar(x + (i - 1) * width, rows.rmse_skill * 100, width, label=model.replace("_", " "), color=palette[i])
    for ax, title in zip(axes, ["MAE skill vs historical volatility", "RMSE skill vs historical volatility"]):
        ax.axhline(0, color="#344054", linewidth=0.9)
        ax.set_xticks(x, tickers)
        ax.set_ylabel("Improvement (%)")
        ax.set_title(title, color=NAVY, fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Volatility forecast errors are metric-dependent", color=NAVY, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "error_skill.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for ticker in tickers:
        g = annual.loc[(annual.ticker == ticker) & (annual.model == "xgboost")].sort_values("year")
        b = annual.loc[(annual.ticker == ticker) & (annual.model == "historical_volatility")].set_index("year").mae
        skill = 100 * (1 - g.mae.to_numpy() / g.year.map(b).to_numpy())
        ax.plot(g.year, skill, marker="o", markersize=2.8, linewidth=1.3, label=ticker)
    ax.axhline(0, color="#344054", linewidth=0.9)
    ax.set_ylabel("XGBoost MAE improvement (%)")
    ax.set_xlabel("Calendar year")
    ax.set_title("Year-by-year performance: positive values favor XGBoost", color=NAVY, weight="bold")
    ax.grid(alpha=0.2)
    ax.legend(ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "annual_xgboost_skill.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    order = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    for model, color, label in [("logistic_regression", TEAL, "Logistic regression"), ("training_base_rate", ORANGE, "Training base rate")]:
        g = event_summary.loc[event_summary.model == model].set_index("ticker").reindex(order)
        axes[0].plot(order, g.roc_auc, marker="o", color=color, label=label)
        axes[1].plot(order, g.log_loss, marker="o", color=color, label=label)
    axes[0].axhline(0.5, color=GREY, linestyle="--", linewidth=1, label="No-rank reference")
    axes[0].set_ylabel("ROC-AUC (higher is better)")
    axes[1].set_ylabel("Log loss (lower is better)")
    for ax, title in zip(axes, ["High-volatility ranking", "Probability forecast score"]):
        ax.set_title(title, color=NAVY, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("High-volatility event forecasts", color=NAVY, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "event_classification.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    bins = np.linspace(0, 1, 11)
    for ticker in order:
        g = event.loc[(event.ticker == ticker) & (event.model == "logistic_regression")]
        if g.empty:
            continue
        ids = np.minimum(np.digitize(g.probability, bins[1:-1]), 9)
        obs, forecast = [], []
        for i in range(10):
            s = g.loc[ids == i]
            if len(s):
                obs.append(s.actual_event.mean())
                forecast.append(s.probability.mean())
        ax.plot(forecast, obs, marker="o", markersize=3, linewidth=1.2, label=ticker)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set(xlim=(0, 0.6), ylim=(0, 0.6), xlabel="Mean predicted probability", ylabel="Observed event frequency")
    ax.set_title("Pooled out-of-sample reliability", color=NAVY, weight="bold")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "event_calibration.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def fmt_pct(value):
    return f"{100 * value:+.1f}%"


def make_markdown(path, regression, event_summary, annual, regime, loss_tests, metadata, sensitivity):
    xgb = regression.loc[regression.model == "xgboost"].set_index("ticker")
    rows = ["# Machine Learning for Volatility Forecasting Under Distribution Shift", "", "## Executive summary", ""]
    rows.append("The evidence is **partially supportive and metric-dependent**. XGBoost reduced pooled out-of-sample MAE on all five ETF series, but its RMSE was worse than the historical-volatility baseline on all five. The gain in typical absolute error therefore did not translate into better control of large errors. Performance and event discrimination differ substantially by asset, so the experiment does not support a claim of universal stable predictive information.")
    rows += ["", "## Data and evaluation", "", f"- Five ETFs: {', '.join(metadata['tickers'])}; adjusted daily OHLCV from Yahoo Finance via yfinance.", f"- Retrieved {metadata['retrieved_at_utc']}; first forecast dates: {metadata['first_forecast_date_by_ticker']}; last labeled forecast origins: {metadata['last_observation_date_by_ticker']}.", f"- Expanding walk-forward validation; initial train={metadata['initial_train_size']} sessions; test blocks={metadata['test_size_sessions']} sessions; seed={metadata['random_seed']}.", "- Target is next-session absolute close-to-close log return. This is a noisy daily volatility proxy, not latent or intraday realized volatility.", "", "## Regression results", "", "Positive skill means lower error than historical volatility.", "", "| Asset | XGBoost MAE skill | XGBoost RMSE skill | XGBoost annual MAE wins |", "|---|---:|---:|---:|"]
    for ticker in metadata["tickers"]:
        g = annual.loc[(annual.ticker == ticker) & (annual.model == "xgboost")]
        b = annual.loc[(annual.ticker == ticker) & (annual.model == "historical_volatility")].set_index("year").mae
        wins = int((g.mae.to_numpy() < g.year.map(b).to_numpy()).sum())
        rows.append(f"| {ticker} | {fmt_pct(xgb.loc[ticker, 'mae_skill'])} | {fmt_pct(xgb.loc[ticker, 'rmse_skill'])} | {wins}/{len(g)} |")
    rows += ["", "## Probability forecasts", "", "The logistic model predicts whether next-session volatility exceeds a cutoff fixed at the training-window 80th percentile. Its cutoff and base-rate reference are recomputed within each fold. See the PDF report for log loss, Brier score, ROC-AUC, average precision, calibration error, and reliability curves.", "", "## Distribution shift and inference", "", "Yearly performance is uneven even where average MAE improves. High-volatility observations are materially harder to forecast. Paired absolute-error tests use a 21-session HAC standard error and Holm correction over the 15 asset/model comparisons; these are descriptive comparisons on one historical sample, not proof of future stability.", "", "## Sensitivity and limitations", ""]
    if sensitivity is not None:
        rows.append("SPY was also evaluated with six-month rather than one-year test blocks, preserving the same initial training size and first forecast date. See the report table for the paired comparison.")
    rows += ["", "The study uses five ETF proxies from one vendor and daily adjusted close prices. It does not include transaction costs or test a trading strategy. No LSTM or attention model was included: the modest single-asset sample and noisy daily target do not justify extra model capacity before larger multi-asset and later-period tests. Yahoo/yfinance terms should be reviewed before redistribution or commercial use.", "", "## Reproduction", "", "```powershell", "pip install -e \".[download,notebooks,test]\"", "pytest -q", "python -m src.run_experiment --local-data-dir data --min-train-size 2520 --test-size 252", "python -m src.generate_report", "```", "", "## References", "", "1. Qiu, Z., Kownatzki, C., Scalzo, F., and Cha, E.S. (2025). Historical Perspectives in Volatility Forecasting Methods with Machine Learning. *Risks*, 13(5), 98. https://doi.org/10.3390/risks13050098", "2. yfinance documentation: https://ranaroussi.github.io/yfinance/", "3. Hudson River Trading, Trading Machine Learning Benchmarks: https://www.hudsonrivertrading.com/hrtbeat/trading-machine-learning/", ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def build_pdf(path, pred, event, annual, regime, loss_tests, metadata, regression, event_summary, sensitivity):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=colors.HexColor(NAVY), alignment=TA_LEFT, spaceAfter=14))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor(NAVY), spaceBefore=10, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.2, textColor=colors.HexColor("#253548"), spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.3, leading=9.5, textColor=colors.HexColor(GREY)))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=colors.HexColor(NAVY), backColor=colors.HexColor("#EAF3F4"), borderColor=colors.HexColor(TEAL), borderWidth=1, borderPadding=10, spaceBefore=7, spaceAfter=12))
    doc = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=.65*inch, leftMargin=.65*inch, topMargin=.65*inch, bottomMargin=.62*inch, title="ML for Volatility Forecasting Under Distribution Shift", author="Financial Market ML Research")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E2EA"))
        canvas.line(.65*inch, .48*inch, 7.85*inch, .48*inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawString(.65*inch, .32*inch, "Financial Market ML Research | Data retrieved 22 Nov 2026")
        canvas.drawRightString(7.85*inch, .32*inch, f"{document.page}")
        canvas.restoreState()

    story = []
    story += [Paragraph("Machine Learning for Volatility Forecasting Under Distribution Shift", styles["TitleCustom"]),
              Paragraph("A walk-forward study of five liquid ETF proxies, 2000-2026", styles["BodyCustom"]),
              Spacer(1, 9),
              Paragraph("Research question: Can machine-learning models extract predictive information from financial time series that persists as market distributions change?", styles["Callout"]),
              Paragraph("Executive summary", styles["Section"])]
    xgb = regression.loc[regression.model == "xgboost"].set_index("ticker")
    rf = regression.loc[regression.model == "random_forest"].set_index("ticker")
    xgb_mae_wins = int((xgb.mae_skill > 0).sum())
    xgb_rmse_wins = int((xgb.rmse_skill > 0).sum())
    significant = int(((loss_tests.candidate == "xgboost") & (loss_tests.p_value_holm < .05)).sum())
    story.append(Paragraph(
        f"The result is <b>partially supportive, but not universal</b>. XGBoost lowered pooled out-of-sample MAE versus the 21-session historical-volatility baseline on {xgb_mae_wins} of 5 assets, while lowering RMSE on {xgb_rmse_wins} of 5. Random forest showed the same MAE/RMSE split. The paired HAC tests mark {significant} of the five XGBoost comparisons as significant after Holm correction across 15 model/asset comparisons. A lower MAE alongside a higher RMSE means forecasts improve on typical days but are less reliable on large-error days. Asset and year breakdowns show the gains are not stable in every regime.", styles["BodyCustom"]))
    story += [Paragraph("Study design", styles["Section"])]
    first = min(pd.to_datetime(list(metadata["first_forecast_date_by_ticker"].values())))
    last = max(pd.to_datetime(list(metadata["last_observation_date_by_ticker"].values())))
    design = [
        ["Assets", "SPY (US large cap), QQQ (Nasdaq), IWM (small cap), TLT (long Treasuries), GLD (gold)"],
        ["Coverage", f"Forecast origins {first:%d %b %Y} to {last:%d %b %Y}; latest target uses the next available close"],
        ["Observations", "2,953-4,179 out-of-sample forecasts per asset; 5,473-6,699 usable history rows"],
        ["Validation", "2,520-session initial train; expanding history; 252-session non-overlapping test blocks"],
        ["Regression target", "Next-session absolute close-to-close log return; fit on log squared return"],
        ["Models", "Historical volatility; linear regression; random forest; XGBoost; logistic high-volatility event model"],
        ["Features", "Lagged return/magnitude, 5/10/21/63-session rolling volatility and mean absolute return, volume change"],
    ]
    table = Table([[Paragraph(f"<b>{a}</b>", styles["SmallCustom"]), Paragraph(b, styles["SmallCustom"])] for a,b in design], colWidths=[1.12*inch, 5.95*inch])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#EFF4F7")), ("VALIGN",(0,0),(-1,-1),"TOP"), ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#D9E2EA")), ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [table, Spacer(1,7), Paragraph("Data are daily adjusted OHLCV downloaded from Yahoo Finance using yfinance. Raw CSV snapshots, retrieval timestamp, package versions, and SHA-256 hashes are retained locally. The data source is an unofficial research feed, not an open redistribution license.", styles["SmallCustom"]), PageBreak()]

    story += [Paragraph("Forecast accuracy", styles["Section"]), Paragraph("Errors are expressed in daily return units. One basis point (bp) is 0.0001. Skill is defined as (historical-volatility error - model error) / historical-volatility error; positive values favor the model. Historical volatility is the trailing 21-session return standard deviation multiplied by sqrt(2/pi), the Gaussian reference for expected absolute return.", styles["BodyCustom"])]
    story.append(Image(str(FIGURES / "error_skill.png"), width=7.05*inch, height=2.35*inch))
    headers = ["Asset", "Model", "MAE (bp)", "MAE skill", "RMSE (bp)", "RMSE skill"]
    rows = [headers]
    for ticker in metadata["tickers"]:
        for model in ["historical_volatility", "linear", "random_forest", "xgboost"]:
            g = regression.loc[(regression.ticker == ticker) & (regression.model == model)].iloc[0] if model != "historical_volatility" else None
            if model == "historical_volatility":
                baseline_rows = pred.loc[(pred.ticker == ticker) & (pred.model == model)]
                mae = mean_absolute_error(baseline_rows.actual_volatility, baseline_rows.forecast_volatility)
                rmse = mean_squared_error(baseline_rows.actual_volatility, baseline_rows.forecast_volatility) ** .5
                ms = rs = 0.0
            else:
                mae, rmse, ms, rs = g.mae, g.rmse, g.mae_skill, g.rmse_skill
            rows.append([ticker, model.replace("_", " "), f"{mae*10000:.2f}", f"{ms*100:+.1f}%", f"{rmse*10000:.2f}", f"{rs*100:+.1f}%"])
    t = Table(rows, colWidths=[.63*inch, 1.55*inch, .89*inch, .83*inch, .95*inch, .88*inch], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.1),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FA")]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D9E2EA")),("ALIGN",(2,1),(-1,-1),"RIGHT"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [t, Spacer(1,4), Paragraph("No parameters were selected using the final test years.", styles["SmallCustom"]), PageBreak()]

    story += [Paragraph("Stability across time and regimes", styles["Section"]), Paragraph("Calendar-year MAE skill makes temporal variation visible. The 10-year training window means evaluation begins at different dates across ETFs; the first and last partial years are included.", styles["BodyCustom"]), Image(str(FIGURES / "annual_xgboost_skill.png"), width=7.05*inch, height=2.82*inch)]
    annual_rows = []
    for ticker in metadata["tickers"]:
        g = annual.loc[(annual.ticker == ticker) & (annual.model == "xgboost")]
        b = annual.loc[(annual.ticker == ticker) & (annual.model == "historical_volatility")].set_index("year").mae
        wins = int((g.mae.to_numpy() < g.year.map(b).to_numpy()).sum())
        annual_rows.append([ticker, f"{wins}/{len(g)}", f"{100*wins/len(g):.0f}%", f"{100*(1-g.mae.mean()/g.year.map(b).mean()):+.1f}% mean annual MAE skill"])
    t = Table([["Asset","Years XGB MAE < baseline","Share","Mean annual skill"]] + annual_rows, colWidths=[.8*inch,1.8*inch, .7*inch,2.9*inch], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FA")]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D9E2EA")),("ALIGN",(1,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [t, Spacer(1,8), Paragraph("A high-volatility event is defined by a training-only 80th-percentile absolute-return cutoff. Fold-specific cutoffs prevent future labels from setting the event threshold. MAE and RMSE are markedly larger in the high-volatility regime; high-volatility samples are less frequent, so their errors have wider uncertainty.", styles["BodyCustom"])]
    high_rows=[]
    for ticker in metadata["tickers"]:
        for regime_name in ["low","high"]:
            g=regime.loc[(regime.ticker==ticker)&(regime.model=="xgboost")&(regime.regime==regime_name)]
            b=regime.loc[(regime.ticker==ticker)&(regime.model=="historical_volatility")&(regime.regime==regime_name)]
            if len(g) and len(b):
                mae=float(np.average(g.mae,weights=g.n)); bmae=float(np.average(b.mae,weights=b.n))
                high_rows.append([ticker,regime_name,int(g.n.sum()),f"{mae*10000:.1f}",f"{100*(bmae-mae)/bmae:+.1f}%"])
    t=Table([["Asset","Regime","Forecasts","XGB MAE (bp)","MAE skill"]]+high_rows,colWidths=[.75*inch,.8*inch,1.0*inch,1.2*inch,1.05*inch],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.2),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FA")]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D9E2EA")),("ALIGN",(2,1),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story += [t, PageBreak()]

    story += [Paragraph("Probabilistic high-volatility forecasts", styles["Section"]), Paragraph("The logistic model forecasts a top-quintile volatility event using lagged returns, rolling volatility, and volume features. Its probabilities are unweighted so they remain comparable to the training base-rate reference. All scores are pooled over out-of-sample predictions.", styles["BodyCustom"]), Image(str(FIGURES / "event_classification.png"), width=6.5*inch, height=2.20*inch), Image(str(FIGURES / "event_calibration.png"), width=3.3*inch, height=2.5*inch)]
    class_rows=[["Asset","Model","Log loss","Brier","ROC-AUC","Avg precision","ECE-10"]]
    for ticker in metadata["tickers"]:
        for model in ["training_base_rate","logistic_regression"]:
            g=event_summary.loc[(event_summary.ticker==ticker)&(event_summary.model==model)].iloc[0]
            class_rows.append([ticker,model.replace("_"," "),f"{g.log_loss:.3f}",f"{g.brier:.3f}",f"{g.roc_auc:.3f}",f"{g.average_precision:.3f}",f"{g.ece_10_bins:.3f}"])
    t=Table(class_rows,colWidths=[.55*inch,1.45*inch,.72*inch,.65*inch,.72*inch,.9*inch,.68*inch],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),6.7),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FA")]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D9E2EA")),("ALIGN",(2,1),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    story += [t, Spacer(1,7), Paragraph("The reliability curves pool folds and use equal-width probability bins. Expected calibration error is descriptive and bin-dependent. The classifier's AUC is strongest for SPY and weakest for GLD/TLT; a probability score can improve on a constant reference without ranking extremes reliably.", styles["BodyCustom"]), PageBreak()]

    story += [Paragraph("Statistical comparison and robustness", styles["Section"]), Paragraph("Paired daily absolute losses compare each candidate with historical volatility. Standard errors use a 21-session Bartlett/Newey-West HAC estimate; p-values are Holm-adjusted across all 15 asset/model comparisons.", styles["BodyCustom"])]
    rows=[["Asset","Candidate","MAE reduction","HAC z","Raw p","Holm p"]]
    for _,r in loss_tests.iterrows():
        p_raw = "<0.0001" if r.p_value_two_sided < .0001 else f"{r.p_value_two_sided:.4f}"
        p_holm = "<0.0001" if r.p_value_holm < .0001 else f"{r.p_value_holm:.4f}"
        rows.append([r.ticker,r.candidate.replace("_"," "),f"{r.mean_mae_reduction*10000:+.2f} bp",f"{r.z_stat:.2f}",p_raw,p_holm])
    t=Table(rows,colWidths=[.65*inch,1.2*inch,1.22*inch,.7*inch,.7*inch,.7*inch],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.0),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FA")]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D9E2EA")),("ALIGN",(2,1),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story += [t, Spacer(1,10), Paragraph("Half-year test-block sensitivity (SPY)", styles["Section"])]
    if sensitivity is not None:
        sens = sensitivity.loc[sensitivity.ticker == "SPY"]
        base = pred.loc[(pred.ticker == "SPY") & (pred.model == "historical_volatility")].set_index("date")
        sens_rows=[["Model","1-year blocks MAE","6-month blocks MAE","Change"]]
        for model in ["historical_volatility","linear","random_forest","xgboost"]:
            a=pred.loc[(pred.ticker=="SPY")&(pred.model==model)]
            b=sens.loc[sens.model==model]
            am=mean_absolute_error(a.actual_volatility,a.forecast_volatility)
            bm=mean_absolute_error(b.actual_volatility,b.forecast_volatility)
            sens_rows.append([model.replace("_"," "),f"{am*10000:.2f} bp",f"{bm*10000:.2f} bp",f"{(bm-am)*10000:+.2f} bp"])
        t=Table(sens_rows,colWidths=[1.7*inch,1.5*inch,1.5*inch,1.1*inch],repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(NAVY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FA")]),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#D9E2EA")),("ALIGN",(1,1),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story += [t, Paragraph("The initial training size and first forecast date are unchanged; only retraining/test block cadence differs. The six-month run uses 126-session test blocks and refits at twice the cadence.", styles["SmallCustom"])]
    else:
        story.append(Paragraph("Sensitivity results were not found. Run the documented half-year SPY command to include this table.",styles["BodyCustom"]))
    story += [Paragraph("Interpretation", styles["Section"]), Paragraph("The evidence supports extracting some forecast information in selected assets and metrics, not a market-invariant signal. The best follow-up is a preregistered replication on a separate vendor and later time span, with realized volatility from intraday data and parameter choices frozen before evaluation. Avoid using these results as a trading claim.", styles["BodyCustom"]), PageBreak()]

    story += [Paragraph("Limitations and reproducibility", styles["Section"]), Paragraph("<b>Target noise.</b> One-day absolute returns are a noisy proxy for conditional volatility. Intraday realized variance or a multi-day target could reduce label noise. <b>Data limitations.</b> Adjusted close history can be revised by the vendor; five ETFs are not a representative universe of all markets. <b>Model selection.</b> One feature set and modest configurations were used; no nested hyperparameter search was performed. <b>Inference.</b> HAC comparisons are approximate and do not erase selection bias or establish future performance. <b>Deep learning.</b> LSTM and attention models were omitted because the single-series daily sample is modest and no tabular baseline evidence justifies adding high-capacity models yet.", styles["BodyCustom"])]
    story += [Paragraph("Reproduce from cached data", styles["Section"]), Paragraph("The repository retains raw inputs locally (not for redistribution) and run metadata includes per-file SHA-256 hashes and package versions. Run <font name='Courier'>pytest -q</font>, <font name='Courier'>python -m src.run_experiment --local-data-dir data --min-train-size 2520 --test-size 252</font>, then <font name='Courier'>python -m src.generate_report</font>. The six-month SPY sensitivity uses <font name='Courier'>--input-csv data/SPY_2000-01-01_latest.csv --price-column Close --min-train-size 2520 --test-size 126 --output-dir results/sensitivity_half_year</font>.", styles["BodyCustom"])]
    story += [Paragraph("References", styles["Section"]), Paragraph("1. Qiu, Z., Kownatzki, C., Scalzo, F., and Cha, E.S. (2025). Historical Perspectives in Volatility Forecasting Methods with Machine Learning. <i>Risks</i>, 13(5), 98. DOI: 10.3390/risks13050098.<br/>2. yfinance documentation: ranaroussi.github.io/yfinance/ (unofficial downloader; see linked Yahoo terms for use rights).<br/>3. Hudson River Trading. Trading Machine Learning Benchmarks: hudsonrivertrading.com/hrtbeat/trading-machine-learning/.", styles["SmallCustom"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    pred, event, annual, regime, loss_tests, metadata, sensitivity = load_data()
    regression = _regression_summary(pred)
    event_summary = _event_summary(event)
    make_figures(pred, event, annual, regression, event_summary)
    make_markdown(ROOT / "research_report.md", regression, event_summary, annual, regime, loss_tests, metadata, sensitivity)
    build_pdf(ROOT / "research_report.pdf", pred, event, annual, regime, loss_tests, metadata, regression, event_summary, sensitivity)
    print(f"Wrote {ROOT / 'research_report.pdf'}")


if __name__ == "__main__":
    main()

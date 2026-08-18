# ── evaluate.py ──────────────────────────────────────────────────────────────
"""
All metrics and plots.

Functions
─────────
metrics()                — MAE, RMSE, MAPE
print_table()            — pretty terminal table
plot_single()            — one model vs actual
plot_all()               — all models vs actual on one chart
plot_residuals()         — error analysis
plot_loss_curves()  ← NEW — training vs validation loss over epochs
plot_feature_importance() ← NEW — which features RFR relied on most
plot_rmse_bar()          — cross-ticker RMSE comparison
save_csv()               — save all metrics to CSV

"""
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend, no display needed
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from typing import Dict


import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config as cfg

log = logging.getLogger(__name__)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"ANN": "#1f77b4", "RFR": "#2ca02c", "LSTM": "#d62728"}
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI"]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    MAE  — average dollar error. Easy to explain: "off by $X on average"
    RMSE — like MAE but punishes large errors more heavily
    MAPE — error as a percentage. "off by X% on average"
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def print_table(ticker: str, results: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {ticker}  —  Results")
    print(f"{'─'*50}")
    print(f"  {'Model':<8}  {'MAE':>7}  {'RMSE':>7}  {'MAPE':>8}")
    print(f"  {'─'*38}")
    for m, v in results.items():
        print(f"  {m:<8}  {v['MAE']:>7.2f}  {v['RMSE']:>7.2f}  {v['MAPE']:>7.2f}%")
    print(f"{'─'*50}")


# ─── Helper ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(cfg.OUTPUT_DIR, name)
    fig.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved  %s", path)


# ─── Prediction Plots ────────────────────────────────────────────────────────

def plot_single(actual, predicted, model_name, ticker):
    """
    Draws two lines — actual price (black dashed) and predicted price (coloured).
    The closer the lines, the better the model performed.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(actual,    color="black", linestyle="--", lw=1.2, label="Actual")
    ax.plot(predicted, color=COLORS[model_name], lw=1.4, label=f"{model_name} Predicted")
    ax.set_title(f"{ticker}  —  {model_name} Predictions")
    ax.set_xlabel("Trading days (test set)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    _save(fig, f"{ticker}_{model_name}.png")


def plot_all(actual, preds: Dict[str, np.ndarray], ticker):
    """All 3 models on one chart so you can compare them visually."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, color="black", linestyle="--", lw=1.4, label="Actual", alpha=0.9)
    for name, pred in preds.items():
        ax.plot(pred, color=COLORS[name], lw=1.2, label=name, alpha=0.85)
    ax.set_title(f"{ticker}  —  All Models vs Actual")
    ax.set_xlabel("Trading days (test set)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    _save(fig, f"{ticker}_comparison.png")


def plot_residuals(actual, preds: Dict[str, np.ndarray], ticker):
    """
    Residual = actual - predicted.
    A perfect model has all residuals at 0.
    If residuals drift above/below 0, the model is consistently wrong in one direction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, pred) in zip(axes, preds.items()):
        ax.plot(actual - pred, color=COLORS[name], lw=0.8, alpha=0.7)
        ax.axhline(0, color="black", lw=1, linestyle="--")
        ax.set_title(f"{name}")
        ax.set_xlabel("Trading days")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    fig.suptitle(f"{ticker}  —  Residual Analysis", fontsize=13)
    _save(fig, f"{ticker}_residuals.png")


# ─── Loss Curves ─────────────────────────────────────────────────────────────

def plot_loss_curves(history, model_name: str, ticker: str):
    """
    Shows how training loss and validation loss changed over each epoch.

    What to look for
    ────────────────
    Good:  both lines decrease together and level off close to each other.
    Bad:   train loss keeps dropping but val loss starts rising — this is
           OVERFITTING (model is memorising training data, not learning).

    EarlyStopping automatically stops training when this happens.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history.history["loss"],     label="Train Loss", color="#1f77b4", lw=1.5)
    ax.plot(history.history["val_loss"], label="Val Loss",   color="#d62728", lw=1.5,
            linestyle="--")

    ax.set_title(f"{ticker}  —  {model_name} Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()

    _save(fig, f"{ticker}_{model_name}_loss_curve.png")


# ─── Feature Importance ──────────────────────────────────────────────────────

def plot_feature_importance(rfr, ticker: str, best_params: dict | None = None):
    """
    Random Forest tracks how much each feature contributed to its predictions.
    Higher bar = more important feature.

    This is a great talking point in interviews:
    'Close price and SMA were the most important features,
     meaning recent price trend matters more than volume for next-day prediction.'
    """
    importance = rfr.feature_importances_

    # Sort features from most to least important
    sorted_idx = np.argsort(importance)
    sorted_features = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    best_params_text = ""
    if best_params:
        best_params_text = json.dumps(best_params, sort_keys=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(sorted_features, sorted_importance, color="#2ca02c", alpha=0.85)

    # Add value labels on each bar
    for bar, val in zip(bars, sorted_importance):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    title = f"{ticker}  —  Feature Importance (Random Forest)"
    ax.set_title(title)
    ax.set_xlabel("Importance Score")

    if best_params_text:
        ax.text(
            0.99, 0.01,
            f"Best params: {best_params_text}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
        )

    _save(fig, f"{ticker}_feature_importance.png")


# ─── Summary Plots ───────────────────────────────────────────────────────────

def plot_rmse_bar(all_results: Dict[str, Dict[str, Dict[str, float]]]):
    """Grouped bar chart comparing RMSE of all models across all tickers."""
    tickers = list(all_results.keys())
    models  = ["ANN", "RFR", "LSTM"]
    x, w    = np.arange(len(tickers)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        vals = [all_results[t][m]["RMSE"] for t in tickers]
        ax.bar(x + i * w, vals, w, label=m, color=COLORS[m], alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("RMSE Comparison — All Tickers & Models")
    ax.legend()
    _save(fig, "rmse_all_tickers.png")


def plot_all_companies_comparison(all_results: Dict[str, Dict[str, Dict[str, float]]]):
    """
    Compare ANN, RFR, and LSTM across every company using MAE, RMSE, and MAPE.

    The lower the bar, the better the model. This plot is useful for explaining
    the overall winner instead of only discussing one stock at a time.
    """
    rows = [
        {"Ticker": ticker, "Model": model, **values}
        for ticker, result in all_results.items()
        for model, values in result.items()
    ]
    df = pd.DataFrame(rows)
    models = ["ANN", "RFR", "LSTM"]
    metrics_to_plot = ["MAE", "RMSE", "MAPE"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric_name in zip(axes, metrics_to_plot):
        averages = (
            df.groupby("Model")[metric_name]
            .mean()
            .reindex(models)
        )
        bars = ax.bar(
            averages.index,
            averages.values,
            color=[COLORS[m] for m in averages.index],
            alpha=0.85,
        )
        ax.set_title(f"Average {metric_name}")
        ax.set_ylabel(f"{metric_name} (lower is better)")
        ax.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, averages.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    best_model = (
        df.groupby("Model")["RMSE"]
        .mean()
        .idxmin()
    )
    fig.suptitle(
        f"All Companies Comparison - Best Overall: {best_model}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "all_companies_model_comparison.png")


def save_csv(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    all_best_params: Dict[str, dict] | None = None,
):
    """Save all metrics to a CSV file for easy sharing and reporting."""
    all_best_params = all_best_params or {}

    rows = []
    for ticker, res in all_results.items():
        best_params = all_best_params.get(ticker, {})
        best_params_str = json.dumps(best_params, sort_keys=True)
        for model, values in res.items():
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model,
                    **values,
                    "RFR_Best_Params": best_params_str if model == "RFR" else "",
                }
            )

    df   = pd.DataFrame(rows)
    path = os.path.join(cfg.OUTPUT_DIR, "metrics.csv")
    df.to_csv(path, index=False)
    log.info("Metrics CSV saved  →  %s", path)
    print("\n", df.to_string(index=False))


def save_rfr_gridsearch_params(all_best_params: Dict[str, dict]) -> None:
    """Save the best RandomForestGridSearch parameters found for each ticker."""
    rows = [
        {"Ticker": ticker, **params}
        for ticker, params in all_best_params.items()
    ]
    df   = pd.DataFrame(rows)
    path = os.path.join(cfg.OUTPUT_DIR, "rfr_gridsearch_best_params.csv")
    df.to_csv(path, index=False)
    log.info("RFR GridSearch best params saved  →  %s", path)

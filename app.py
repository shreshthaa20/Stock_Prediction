import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from data_loader import load_ticker, make_sequences, prepare_data
from evaluate import metrics


st.set_page_config(
    page_title="StockSight",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0A192F;
    }

    [data-testid="stHeader"] {
        background-color: #64B5F6;
    }

    [data-testid="stSidebar"] {
        background-color: #112240;
    }

    .stApp,
    .stApp p,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stCaptionContainer"] {
        color: #FFFFFF !important;
    }

    [data-baseweb="select"],
    [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    div[role="combobox"] *,
    .stSelectbox * {
        color: #000080 !important;
        background-color: #FFFFFF !important;
    }

    /* ensure the displayed selected value is readable */
    div[role="combobox"] {
        color: #000080 !important;
        background-color: #FFFFFF !important;
    }

    /* make selectbox labels in the sidebar navy */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox > label {
        color: #000080 !important;
    }

    /* make sidebar buttons navy and readable */
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton button *,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] button * {
        color: #000080 !important;
        background-color: #FFFFFF !important;
        border-color: #000080 !important;
    }

    /* target button internals used by different Streamlit versions */
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button > div > span,
    [data-testid="stSidebar"] button > div,
    [data-testid="stSidebar"] button > div > span,
    [data-testid="stSidebar"] button[role="button"] > div > span {
        color: #000080 !important;
        background-color: #FFFFFF !important;
    }

    /* make all sidebar labels (including 'Select stock') navy */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label * {
        color: #000080 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_COLORS = {
    "Actual": "#6B7280",
    "ANN": "#059669",
    "RFR": "#2563EB",
    "LSTM": "#7C3AED",
}
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI"]


@st.cache_data(show_spinner=False)
def get_prepared_data(ticker: str):
    df = load_ticker(ticker, cfg.START_DATE, cfg.END_DATE)
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_data(df)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test, cfg.SEQUENCE_LEN)
    return df, X_train, X_test, y_train, y_test, x_scaler, y_scaler, X_test_seq, y_test_seq


@st.cache_resource(show_spinner=False)
def get_saved_models(ticker: str):
    from models import load_models
    return load_models(ticker)


def inverse_scale(y_scaler, values):
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).flatten()


def model_files_exist(ticker: str) -> bool:
    folder = os.path.join("saved_models", ticker)
    return all(
        os.path.exists(os.path.join(folder, name))
        for name in ["rfr.pkl", "ann.keras", "lstm.keras"]
    )


@st.cache_data(show_spinner=False)
def load_metrics_summary():
    path = os.path.join(cfg.OUTPUT_DIR, "metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def show_all_companies_comparison():
    st.markdown(
        '<h2 style="color:#000080;text-align:center;font-weight:700;margin:8px 0">All Companies Comparison</h2>',
        unsafe_allow_html=True,
    )
    summary_df = load_metrics_summary()
    if summary_df is None or summary_df.empty:
        st.info("Run `python main.py` to create `outputs/metrics.csv` and calculate the overall winner.")
        return

    tickers = list(summary_df["Ticker"].drop_duplicates())
    models = ["ANN", "RFR", "LSTM"]
    metric_names = ["MAE", "RMSE", "MAPE"]
    x = np.arange(len(tickers))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, metric_name in zip(axes, metric_names):
        for index, model in enumerate(models):
            values = []
            for ticker_symbol in tickers:
                row = summary_df[
                    (summary_df["Ticker"] == ticker_symbol)
                    & (summary_df["Model"] == model)
                ]
                values.append(float(row[metric_name].iloc[0]) if not row.empty else np.nan)

            ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=model,
                color=MODEL_COLORS[model],
                alpha=0.85,
            )

        ax.set_title(metric_name)
        ax.set_ylabel(f"{metric_name} (lower is better)")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=0)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].legend(loc="upper right")
    fig.suptitle("All Companies and All Models Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "This graph compares ANN, Random Forest, and LSTM for every company. "
        "Lower MAE, RMSE, and MAPE values mean better prediction accuracy."
    )

    avg_metrics = (
        summary_df.groupby("Model")[["MAE", "RMSE", "MAPE"]]
        .mean()
        .sort_values("RMSE")
    )
    best_model = avg_metrics.index[0]
    best_mae = avg_metrics.loc[best_model, "MAE"]
    best_rmse = avg_metrics.loc[best_model, "RMSE"]
    best_mape = avg_metrics.loc[best_model, "MAPE"]

    st.success(
        f"Best overall model: {best_model}. Average MAE: {best_mae:.2f}, "
        f"average RMSE: {best_rmse:.2f}, average MAPE: {best_mape:.2f}%."
    )
    st.dataframe(
        avg_metrics.reset_index(),
        use_container_width=True,
        hide_index=True,
    )


if "view" not in st.session_state:
    st.session_state.view = "home"


with st.sidebar:
    st.divider()

    ticker = st.selectbox("Select Stock", cfg.TICKERS, index=None, placeholder="Choose a stock")

    selections_ready = ticker is not None
    load_dashboard = st.button(
        "Load Dashboard",
        type="primary",
        use_container_width=True,
        disabled=not selections_ready,
    )
    all_companies_comparison = st.button(
        "All Companies Comparison",
        use_container_width=True,
    )

    if load_dashboard:
        st.session_state.view = "stock"
    if all_companies_comparison:
        st.session_state.view = "all_companies"

    st.divider()
    st.caption(f"Period: {cfg.START_DATE} to {cfg.END_DATE}")
    st.caption("Split: 80% Train / 20% Test")
    st.caption(f"LSTM Window: {cfg.SEQUENCE_LEN} days")


st.title("StockSight")


if st.session_state.view == "all_companies":
    show_all_companies_comparison()
    st.stop()

col1 = st.columns(1)[0]
col1.metric("Selected Stock", ticker or "Not selected")

if not selections_ready:
    st.info("Choose a Stock in the sidebar to continue.")
    st.stop()

if not model_files_exist(ticker):
    st.error(f"No saved models found for {ticker}. Run `python main.py` first.")
    st.stop()

if not load_dashboard and st.session_state.view != "stock":
    st.info("Click **Load Dashboard** in the sidebar to load predictions.")
    st.stop()


with st.spinner("Loading Data and Saved Models..."):
    try:
        (
            df,
            X_train,
            X_test,
            y_train,
            y_test,
            x_scaler,
            y_scaler,
            X_test_seq,
            y_test_seq,
        ) = get_prepared_data(ticker)
        rfr, ann, lstm = get_saved_models(ticker)
    except Exception as exc:
        st.error(f"Could not load dashboard: {exc}")
        st.stop()


offset = cfg.SEQUENCE_LEN
actual = inverse_scale(y_scaler, y_test[offset:])

with st.spinner("Running Predictions..."):
    ann_pred = inverse_scale(y_scaler, ann.predict(X_test[offset:], verbose=0).flatten())
    rfr_pred = inverse_scale(y_scaler, rfr.predict(X_test[offset:]))
    lstm_pred = inverse_scale(y_scaler, lstm.predict(X_test_seq, verbose=0).flatten())

predictions = {
    "ANN": ann_pred,
    "RFR": rfr_pred,
    "LSTM": lstm_pred,
}

# Create tabs for structured navigation
tab_comp, tab_ann, tab_rfr, tab_lstm = st.tabs([
    "📈 Comparison (All Models)",
    "🧠 ANN Model",
    "🌲 RFR Model",
    "🔄 LSTM Model"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: COMPARISON (ALL MODELS VS ACTUAL)
# ─────────────────────────────────────────────────────────────────────────────
with tab_comp:
    st.subheader(f"All Models vs Actual Price for {ticker}")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual, color=MODEL_COLORS["Actual"], linestyle="--", label="Actual", linewidth=1.5)
    for name, pred in predictions.items():
        ax.plot(pred, color=MODEL_COLORS[name], label=f"{name} Predicted", linewidth=1.5)
    ax.set_xlabel("Trading Days (Test Set)")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    st.markdown("""
    **Graph Explanation:**
    This comparison chart plots the actual stock closing prices alongside predictions from all three models (ANN, Random Forest, and LSTM) over the testing period.
    - The **Actual** price is represented by the **dashed grey line**.
    - The **ANN (Artificial Neural Network)** prediction is the **green line**.
    - The **RFR (Random Forest Regressor)** prediction is the **blue line**.
    - The **LSTM (Long Short-Term Memory)** prediction is the **purple line**.
    
    *Look for which model line tracks the actual price line most closely, especially during sharp trends or sudden reversals.*
    """)

    st.subheader("Performance Metrics Summary Table")
    rows = []
    for name, pred in predictions.items():
        model_metrics = metrics(actual, pred)
        rows.append(
            {
                "Model": name,
                "MAE (Mean Absolute Error)": f"${model_metrics['MAE']:.2f}",
                "RMSE (Root Mean Squared Error)": f"${model_metrics['RMSE']:.2f}",
                "MAPE (Mean Absolute Percentage Error)": f"{model_metrics['MAPE']:.2f}%",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("""
    **Metrics Explanation:**
    - **MAE (Mean Absolute Error):** On average, how many dollars the predictions deviate from actual values. Lower is better.
    - **RMSE (Root Mean Squared Error):** Similiar to MAE, but penalizes larger errors more heavily. Lower is better.
    - **MAPE (Mean Absolute Percentage Error):** The average percentage deviation of the predictions from actual prices. Lower is better.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANN MODEL
# ─────────────────────────────────────────────────────────────────────────────
with tab_ann:
    st.subheader(f"ANN Model vs Actual - {ticker}")
    
    m1, m2, m3 = st.columns(3)
    ann_metrics = metrics(actual, predictions["ANN"])
    m1.metric("ANN MAE", f"${ann_metrics['MAE']:.2f}")
    m2.metric("ANN RMSE", f"${ann_metrics['RMSE']:.2f}")
    m3.metric("ANN MAPE", f"{ann_metrics['MAPE']:.2f}%")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, color=MODEL_COLORS["Actual"], linestyle="--", label="Actual", linewidth=1.5)
    ax.plot(predictions["ANN"], color=MODEL_COLORS["ANN"], label="ANN Predicted", linewidth=1.6)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    st.markdown("""
    **Graph Explanation:**
    This graph displays the actual stock price (dashed line) versus the Artificial Neural Network (ANN) predictions (green line). The ANN uses dense feed-forward connections to map features to prices, indicating how well simple neural mapping captures daily stock trends.
    """)
    
    st.subheader("Residual Analysis (ANN)")
    residuals = actual - predictions["ANN"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(len(residuals)), residuals, color=np.where(residuals >= 0, "#059669", "#DC2626"), alpha=0.8)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Actual - Predicted")
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
    **Residuals Explanation:**
    Residuals represent the prediction error (Actual minus Predicted). Green bars indicate the model under-predicted (actual was higher), while red bars indicate over-prediction (actual was lower). The ideal model has small bars hovering close to zero.
    """)

    st.subheader("Training vs Validation Loss Curve")
    loss_curve_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_ANN_loss_curve.png")
    if os.path.exists(loss_curve_path):
        st.image(loss_curve_path, use_container_width=True)
        st.markdown("""
        **Loss Curve Explanation:**
        The training vs. validation loss curve shows how the neural network learned over training epochs. A healthy training run features both training loss (solid line) and validation loss (dashed line) declining steadily and converging. If validation loss increases while training loss decreases, the model is overfitting.
        """)
    else:
        st.warning(f"No Saved Loss Curve Found at `{loss_curve_path}`.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: RFR MODEL
# ─────────────────────────────────────────────────────────────────────────────
with tab_rfr:
    st.subheader(f"RFR Model vs Actual - {ticker}")
    
    m1, m2, m3 = st.columns(3)
    rfr_metrics = metrics(actual, predictions["RFR"])
    m1.metric("RFR MAE", f"${rfr_metrics['MAE']:.2f}")
    m2.metric("RFR RMSE", f"${rfr_metrics['RMSE']:.2f}")
    m3.metric("RFR MAPE", f"{rfr_metrics['MAPE']:.2f}%")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, color=MODEL_COLORS["Actual"], linestyle="--", label="Actual", linewidth=1.5)
    ax.plot(predictions["RFR"], color=MODEL_COLORS["RFR"], label="RFR Predicted", linewidth=1.6)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    st.markdown("""
    **Graph Explanation:**
    This graph displays the actual stock price (dashed line) versus the Random Forest Regressor (RFR) predictions (blue line). RFR uses an ensemble of decision trees, which are less prone to overfitting but may predict in steps rather than smooth curves.
    """)
    
    st.subheader("Residual Analysis (RFR)")
    residuals = actual - predictions["RFR"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(len(residuals)), residuals, color=np.where(residuals >= 0, "#059669", "#DC2626"), alpha=0.8)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Actual - Predicted")
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
    **Residuals Explanation:**
    Residuals represent the prediction error (Actual minus Predicted). Green bars indicate the model under-predicted, while red bars indicate over-prediction.
    """)

    st.subheader("Feature Importance Plot")
    feature_importance_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_feature_importance.png")
    if os.path.exists(feature_importance_path):
        st.image(feature_importance_path, use_container_width=True)
        st.markdown("""
        **Feature Importance Explanation:**
        This chart ranks the engineered features by their predictive importance in the Random Forest model. Higher scores indicate the model relied more heavily on that feature (e.g., SMA, Close price, Volume) to make its decisions.
        """)
    else:
        st.warning(f"No Saved Feature-Importance Plot Found at `{feature_importance_path}`. Run `python main.py` to generate it.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────
with tab_lstm:
    st.subheader(f"LSTM Model vs Actual - {ticker}")
    
    m1, m2, m3 = st.columns(3)
    lstm_metrics = metrics(actual, predictions["LSTM"])
    m1.metric("LSTM MAE", f"${lstm_metrics['MAE']:.2f}")
    m2.metric("LSTM RMSE", f"${lstm_metrics['RMSE']:.2f}")
    m3.metric("LSTM MAPE", f"{lstm_metrics['MAPE']:.2f}%")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, color=MODEL_COLORS["Actual"], linestyle="--", label="Actual", linewidth=1.5)
    ax.plot(predictions["LSTM"], color=MODEL_COLORS["LSTM"], label="LSTM Predicted", linewidth=1.6)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    st.markdown("""
    **Graph Explanation:**
    This graph displays the actual stock price (dashed line) versus the Long Short-Term Memory (LSTM) network predictions (purple line). LSTM is a recurrent neural network designed for sequence processing, allowing it to capture historical price trends and momentum.
    """)
    
    st.subheader("Residual Analysis (LSTM)")
    residuals = actual - predictions["LSTM"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(range(len(residuals)), residuals, color=np.where(residuals >= 0, "#059669", "#DC2626"), alpha=0.8)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Actual - Predicted")
    ax.grid(True, axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("""
    **Residuals Explanation:**
    Residuals represent the prediction error (Actual minus Predicted). Green bars indicate the model under-predicted, while red bars indicate over-prediction.
    """)

    st.subheader("Training vs Validation Loss Curve")
    loss_curve_path = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_LSTM_loss_curve.png")
    if os.path.exists(loss_curve_path):
        st.image(loss_curve_path, use_container_width=True)
        st.markdown("""
        **Loss Curve Explanation:**
        The training vs. validation loss curve shows how the neural network learned over training epochs. A healthy training run features both training loss (solid line) and validation loss (dashed line) declining steadily and converging. If validation loss increases while training loss decreases, the model is overfitting.
        """)
    else:
        st.warning(f"No Saved Loss Curve Found at `{loss_curve_path}`.")

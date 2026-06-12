import matplotlib
matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from data_loader import load_ticker, prepare_data, make_sequences
from evaluate import metrics

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight — ML Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {background: transparent;}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {visibility: hidden;}

[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stVerticalBlock"],
[data-testid="stWidgetLabel"],
[data-testid="stSelectbox"],
[data-testid="stButton"] {
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] {
    display: block !important;
    min-width: 280px !important;
}

/* ── Base ── */
.stApp { background-color: #0E1117; color: #F8FAFC; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] hr { border-color: #1E293B !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #161B22 0%, #0d2137 50%, #161B22 100%);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 44px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -30%; right: 5%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(0,200,150,0.10) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40%; left: 10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 11px; font-weight: 700;
    letter-spacing: 3px; color: #00C896;
    text-transform: uppercase; margin-bottom: 14px;
}
.hero-title {
    font-size: 38px; font-weight: 700;
    color: #F8FAFC; line-height: 1.15; margin-bottom: 10px;
}
.hero-title span { color: #00C896; }
.hero-subtitle { font-size: 15px; color: #64748b; }

/* ── Section headers ── */
.section-header {
    font-size: 10px; font-weight: 700;
    letter-spacing: 3px; color: #00C896;
    text-transform: uppercase;
    margin: 36px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E293B;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 6px;
    margin-right: 8px;
}
.badge-ticker { background: rgba(0,200,150,0.12);  color: #00C896; border: 1px solid rgba(0,200,150,0.3);  }
.badge-ann    { background: rgba(0,200,150,0.10);  color: #00C896; border: 1px solid rgba(0,200,150,0.25); }
.badge-rfr    { background: rgba(59,130,246,0.10); color: #3B82F6; border: 1px solid rgba(59,130,246,0.25);}
.badge-lstm   { background: rgba(248,250,252,0.08);color: #F8FAFC; border: 1px solid rgba(248,250,252,0.2);}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #263548 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 10px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #00C896 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    color: #00C896;
    border: 1px solid #00C896;
    border-radius: 8px;
    font-weight: 600; font-size: 13px;
    padding: 10px 28px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: rgba(0,200,150,0.12);
    color: #F8FAFC;
}

/* ── Rank cards ── */
.rank-card { border-radius: 12px; padding: 22px; }
.rank-1 { background: rgba(0,200,150,0.07);  border: 1px solid rgba(0,200,150,0.3);  }
.rank-2 { background: rgba(59,130,246,0.07); border: 1px solid rgba(59,130,246,0.3); }
.rank-3 { background: rgba(100,116,139,0.07);border: 1px solid rgba(100,116,139,0.25);}
.rank-num   { font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 800; opacity: 0.15; float: right; color: #F8FAFC; }
.rank-model { font-size: 20px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px; }
.rank-stat  { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #64748b; line-height: 1.8; }

/* ── Verdict box ── */
.verdict-box {
    background: linear-gradient(135deg, rgba(0,200,150,0.07), rgba(59,130,246,0.07));
    border: 1px solid rgba(0,200,150,0.35);
    border-radius: 14px;
    padding: 28px 32px; margin: 24px 0;
}
.verdict-eyebrow { font-size: 10px; font-weight: 700; letter-spacing: 3px; color: #00C896; text-transform: uppercase; margin-bottom: 10px; }
.verdict-main    { font-size: 20px; font-weight: 700; color: #F8FAFC; }
.verdict-sub     { font-size: 13px; color: #64748b; margin-top: 6px; line-height: 1.6; }

/* ── Info card ── */
.info-card {
    background: #1E293B;
    border: 1px solid #263548;
    border-left: 3px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 8px 0;
    font-size: 13px; color: #94a3b8; line-height: 1.6;
}

/* ── Divider ── */
.custom-divider { border: none; border-top: 1px solid #1E293B; margin: 36px 0; }

/* ── Dataframe ── */
.stDataFrame { background: #1E293B !important; border: 1px solid #263548 !important; border-radius: 10px !important; }
[data-testid="stDataFrame"] * { color: #F8FAFC !important; background-color: #1E293B !important; }


/* ── Spinner ── */
.stSpinner > div { border-top-color: #00C896 !important; }

p, .stMarkdown p { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Chart style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1E293B",
    "axes.facecolor":   "#1E293B",
    "axes.edgecolor":   "#263548",
    "axes.labelcolor":  "#64748b",
    "axes.titlecolor":  "#F8FAFC",
    "xtick.color":      "#475569",
    "ytick.color":      "#475569",
    "grid.color":       "#1a2640",
    "grid.linewidth":   0.8,
    "text.color":       "#94a3b8",
    "legend.facecolor": "#1E293B",
    "legend.edgecolor": "#263548",
    "legend.labelcolor":"#F8FAFC",
    "font.family":      "sans-serif",
})

MODEL_COLORS = {
    "Actual": "#475569",
    "ANN":    "#00C896",
    "RFR":    "#3B82F6",
    "LSTM":   "#F8FAFC",
}

# ── Chart style ───────────────────────────────────────────────────────────────
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size:18px;font-weight:700;color:#F8FAFC;margin-bottom:4px'>StockSight</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#94a3b8;margin-bottom:20px;letter-spacing:1px'>ML PREDICTION DASHBOARD</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#00C896;font-weight:700;margin-bottom:8px'>STOCK</div>", unsafe_allow_html=True)
    ticker = st.selectbox("Select stock", cfg.TICKERS, key="stock_selector")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#00C896;font-weight:700;margin-top:16px;margin-bottom:8px'>MODEL</div>", unsafe_allow_html=True)
    model_name = st.selectbox("Select model", ["ANN", "RFR", "LSTM"], key="model_selector")
    if st.button("Load dashboard", type="primary", use_container_width=True):
        st.session_state.dashboard_loaded = True
    st.markdown("---")
    st.markdown(f"<div style='font-size:12px;color:#94a3b8;line-height:2'>Period<br><span style='color:#cbd5e1;font-family:JetBrains Mono,monospace;font-size:11px'>{cfg.START_DATE} — {cfg.END_DATE}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:#94a3b8;line-height:2;margin-top:8px'>Split<br><span style='color:#cbd5e1'>80% train / 20% test</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:#94a3b8;line-height:2;margin-top:8px'>LSTM window<br><span style='color:#cbd5e1'>{cfg.SEQUENCE_LEN} days</span></div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.title("StockSight Dashboard")
st.write("Choose a stock and model, then load the dashboard.")
top_col1, top_col2, top_col3 = st.columns([1, 1, 1])
with top_col1:
    ticker = st.selectbox("Stock", cfg.TICKERS, key="stock_selector_main")
with top_col2:
    model_name = st.selectbox("Model", ["ANN", "RFR", "LSTM"], key="model_selector_main")
with top_col3:
    st.write("")
    st.write("")
    if st.button("Load dashboard", type="primary", use_container_width=True, key="load_dashboard_main"):
        st.session_state.dashboard_loaded = True

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">ML Stock Prediction</div>
    <div class="hero-title">Stock<span>Sight</span> Dashboard</div>
    <div class="hero-subtitle">
        Comparing ANN &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; LSTM
        &nbsp;&nbsp;|&nbsp;&nbsp; {', '.join(cfg.TICKERS)}
        &nbsp;&nbsp;|&nbsp;&nbsp; {cfg.START_DATE} to {cfg.END_DATE}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Check models ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_prepared_data(selected_ticker: str):
    df = load_ticker(selected_ticker, cfg.START_DATE, cfg.END_DATE)
    X_tr, X_te, y_tr, y_te, x_scaler, y_scaler = prepare_data(df)
    X_te_seq, y_te_seq = make_sequences(X_te, y_te, cfg.SEQUENCE_LEN)
    return df, X_tr, X_te, y_tr, y_te, x_scaler, y_scaler, X_te_seq, y_te_seq


@st.cache_resource(show_spinner=False)
def get_saved_models(selected_ticker: str):
    from models import load_models
    return load_models(selected_ticker)


model_path = os.path.join("saved_models", ticker, "rfr.pkl")
if not os.path.exists(model_path):
    st.error(f"No saved models found for **{ticker}**. Run `python main.py` first.")
    st.stop()

# ── Load data + models ────────────────────────────────────────────────────────
if not st.session_state.get("dashboard_loaded", False):
    st.info("Select a stock and model from the sidebar, then click **Load dashboard**.")
    st.stop()

with st.spinner("Loading data and models ..."):
    try:
        df, X_tr, X_te, y_tr, y_te, x_scaler, y_scaler, X_te_seq, y_te_seq = get_prepared_data(ticker)
        rfr, ann, lstm = get_saved_models(ticker)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ── Predictions ───────────────────────────────────────────────────────────────
offset = cfg.SEQUENCE_LEN

def inv(arr):
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

actual    = inv(y_te[offset:])
ann_pred  = inv(ann.predict(X_te[offset:],  verbose=0).flatten())
rfr_pred  = inv(rfr.predict(X_te[offset:]))
lstm_pred = inv(lstm.predict(X_te_seq, verbose=0).flatten())

pred = {"ANN": ann_pred, "RFR": rfr_pred, "LSTM": lstm_pred}[model_name]
m    = metrics(actual, pred)

# ── Header badges ─────────────────────────────────────────────────────────────
badge_class = {"ANN": "badge-ann", "RFR": "badge-rfr", "LSTM": "badge-lstm"}[model_name]
st.markdown(f"""
<div style="margin-bottom:24px">
    <span class="badge badge-ticker">{ticker}</span>
    <span class="badge {badge_class}">{model_name}</span>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE",       f"${m['MAE']}",     help="Average dollar error")
c2.metric("RMSE",      f"${m['RMSE']}",    help="Penalises large errors more")
c3.metric("MAPE",      f"{m['MAPE']}%",    help="Average percentage error")
c4.metric("Test Days", f"{len(actual)}",   help="Days in the test set")

# ── Prediction chart ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Price Prediction</div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(actual, color=MODEL_COLORS["Actual"], linestyle="--", lw=1.2, label="Actual", alpha=0.6)
ax.plot(pred,   color=MODEL_COLORS[model_name], lw=1.8, label=f"{model_name} Predicted")
ax.fill_between(range(len(actual)), actual, pred,
                alpha=0.05, color=MODEL_COLORS[model_name])
ax.set_xlabel("Trading Days (Test Set)", fontsize=11)
ax.set_ylabel("Price (USD)", fontsize=11)
ax.set_title(f"{ticker}  —  {model_name}  |  MAPE {m['MAPE']}%", fontsize=12, pad=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Residuals ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Residual Analysis</div>', unsafe_allow_html=True)
st.markdown("<div class='info-card'>Residual = Actual minus Predicted. A good model has bars scattered randomly around zero with no clear upward or downward drift.</div>", unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(13, 3))
residuals = actual - pred
colors_r  = [MODEL_COLORS[model_name] if r >= 0 else "#EF4444" for r in residuals]
ax2.bar(range(len(residuals)), residuals, color=colors_r, alpha=0.65, width=1.0)
ax2.axhline(0, color="#00C896", lw=1.2, linestyle="--", alpha=0.6)
ax2.set_xlabel("Trading Days", fontsize=11)
ax2.set_ylabel("Error (USD)", fontsize=11)
ax2.set_title(f"{ticker} — {model_name} Residuals", fontsize=12, pad=12)
ax2.grid(True, alpha=0.4)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ── Feature importance / Loss curve ───────────────────────────────────────────
if model_name == "RFR":
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI"]
    importance   = rfr.feature_importances_
    sorted_idx   = np.argsort(importance)
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    bars = ax3.barh([FEATURE_COLS[i] for i in sorted_idx],
                    importance[sorted_idx],
                    color="#3B82F6", alpha=0.85, height=0.55)
    for bar, val in zip(bars, importance[sorted_idx]):
        ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=10, color="#cbd5e1")
    ax3.set_xlabel("Importance Score", fontsize=11)
    ax3.set_title(f"{ticker} — Feature Importance (Random Forest)", fontsize=12, pad=12)
    ax3.grid(True, alpha=0.4, axis="x")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

if model_name in ["ANN", "LSTM"]:
    st.markdown('<div class="section-header">Training Loss Curve</div>', unsafe_allow_html=True)
    st.markdown("<div class='info-card'>Both lines going down together = healthy training. Validation loss rising while training loss falls = overfitting.</div>", unsafe_allow_html=True)
    loss_img = os.path.join(cfg.OUTPUT_DIR, f"{ticker}_{model_name}_loss_curve.png")
    if os.path.exists(loss_img):
        st.image(loss_img)
    else:
        st.info("Run main.py again to generate loss curves.")

# ── All models comparison ─────────────────────────────────────────────────────
st.markdown(f'<div class="section-header">All Models Comparison — {ticker}</div>', unsafe_allow_html=True)

fig4, ax4 = plt.subplots(figsize=(13, 4))
ax4.plot(actual,    color=MODEL_COLORS["Actual"], linestyle="--", lw=1.2, label="Actual",  alpha=0.6)
ax4.plot(ann_pred,  color=MODEL_COLORS["ANN"],  lw=1.5, label="ANN",  alpha=0.9)
ax4.plot(rfr_pred,  color=MODEL_COLORS["RFR"],  lw=1.5, label="RFR",  alpha=0.9)
ax4.plot(lstm_pred, color=MODEL_COLORS["LSTM"], lw=1.5, label="LSTM", alpha=0.9)
ax4.set_xlabel("Trading Days (Test Set)", fontsize=11)
ax4.set_ylabel("Price (USD)", fontsize=11)
ax4.set_title(f"{ticker} — All Models vs Actual", fontsize=12, pad=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.4)
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

# Metrics table
rows = []
for name, p in [("ANN", ann_pred), ("RFR", rfr_pred), ("LSTM", lstm_pred)]:
    m_ = metrics(actual, p)
    rows.append({"Model": name, "MAE ($)": m_["MAE"],
                 "RMSE ($)": m_["RMSE"], "MAPE (%)": m_["MAPE"]})
df_single = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    df_single.style
        .highlight_min(axis=0, color="#052e16")
        .set_properties(**{"color": "#F8FAFC", "background-color": "#1E293B", "border": "1px solid #263548"}),
    use_container_width=True,
)
st.markdown("<div style='font-size:12px;color:#94a3b8;margin-top:6px'>Green highlight = best (lowest error) for that metric</div>", unsafe_allow_html=True)

# RMSE bar chart
fig5, ax5 = plt.subplots(figsize=(7, 3))
model_names_list = ["ANN", "RFR", "LSTM"]
rmse_vals   = [metrics(actual, ann_pred)["RMSE"],
               metrics(actual, rfr_pred)["RMSE"],
               metrics(actual, lstm_pred)["RMSE"]]
bar_colors  = [MODEL_COLORS[mn] for mn in model_names_list]
bars = ax5.bar(model_names_list, rmse_vals, color=bar_colors, alpha=0.85, width=0.4)
for bar, val in zip(bars, rmse_vals):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.15,
             f"${val}", ha="center", fontsize=11,
             fontweight="600", color="#F8FAFC")
ax5.set_ylabel("RMSE — lower is better", fontsize=11)
ax5.set_title(f"{ticker} — RMSE by Model", fontsize=12, pad=12)
ax5.grid(True, alpha=0.4, axis="y")
plt.tight_layout()
st.pyplot(fig5)
plt.close(fig5)

# ── Cross-stock section ───────────────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Cross-Stock Analysis — All Companies</div>', unsafe_allow_html=True)

col_b1, col_b2 = st.columns(2)
with col_b1:
    show_graphs = st.button("Show Prediction Graphs — All Stocks")
with col_b2:
    show_summary = st.button("Generate Full Summary")

# ── All company graphs ────────────────────────────────────────────────────────
if show_graphs:
    for t in cfg.TICKERS:
        with st.spinner(f"Loading {t} ..."):
            try:
                _, _, X_te_t, _, y_te_t, _, y_sc_t, X_te_seq_t, _ = get_prepared_data(t)
                rfr_t, ann_t, lstm_t = get_saved_models(t)
                off = cfg.SEQUENCE_LEN

                def inv_t(arr):
                    return y_sc_t.inverse_transform(arr.reshape(-1,1)).flatten()

                act_t       = inv_t(y_te_t[off:])
                ann_t_pred  = inv_t(ann_t.predict(X_te_t[off:],  verbose=0).flatten())
                rfr_t_pred  = inv_t(rfr_t.predict(X_te_t[off:]))
                lstm_t_pred = inv_t(lstm_t.predict(X_te_seq_t,   verbose=0).flatten())

                st.markdown(f'<div class="section-header">{t}</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(13, 4))
                ax.plot(act_t,       color=MODEL_COLORS["Actual"], linestyle="--", lw=1.2, label="Actual",  alpha=0.6)
                ax.plot(ann_t_pred,  color=MODEL_COLORS["ANN"],  lw=1.5, label="ANN",  alpha=0.9)
                ax.plot(rfr_t_pred,  color=MODEL_COLORS["RFR"],  lw=1.5, label="RFR",  alpha=0.9)
                ax.plot(lstm_t_pred, color=MODEL_COLORS["LSTM"], lw=1.5, label="LSTM", alpha=0.9)
                ax.set_xlabel("Trading Days", fontsize=11)
                ax.set_ylabel("Price (USD)", fontsize=11)
                ax.set_title(f"{t} — All Models vs Actual", fontsize=12, pad=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                c1, c2, c3 = st.columns(3)
                for col, name, p in zip([c1, c2, c3],
                                        ["ANN", "RFR", "LSTM"],
                                        [ann_t_pred, rfr_t_pred, lstm_t_pred]):
                    m_ = metrics(act_t, p)
                    col.metric(f"{name} RMSE", f"${m_['RMSE']}")

            except Exception as e:
                st.warning(f"Could not load {t}: {e}")

# ── Full summary ──────────────────────────────────────────────────────────────
if show_summary:
    all_rows = []
    progress = st.progress(0, text="Calculating across all companies ...")

    for i, t in enumerate(cfg.TICKERS):
        with st.spinner(f"Processing {t} ..."):
            try:
                _, _, X_te_t, _, y_te_t, _, y_sc_t, X_te_seq_t, _ = get_prepared_data(t)
                rfr_t, ann_t, lstm_t = get_saved_models(t)
                off = cfg.SEQUENCE_LEN

                def inv_t(arr):
                    return y_sc_t.inverse_transform(arr.reshape(-1,1)).flatten()

                act_t       = inv_t(y_te_t[off:])
                ann_t_pred  = inv_t(ann_t.predict(X_te_t[off:],  verbose=0).flatten())
                rfr_t_pred  = inv_t(rfr_t.predict(X_te_t[off:]))
                lstm_t_pred = inv_t(lstm_t.predict(X_te_seq_t,   verbose=0).flatten())

                for name, p in [("ANN", ann_t_pred), ("RFR", rfr_t_pred), ("LSTM", lstm_t_pred)]:
                    m_ = metrics(act_t, p)
                    all_rows.append({"Ticker": t, "Model": name,
                                     "MAE": m_["MAE"], "RMSE": m_["RMSE"], "MAPE": m_["MAPE"]})
            except Exception as e:
                st.warning(f"Could not load {t}: {e}")

        progress.progress((i + 1) / len(cfg.TICKERS),
                          text=f"Processed {t} ({i+1}/{len(cfg.TICKERS)})")

    if all_rows:
        df_all = pd.DataFrame(all_rows)

        # Full metrics table
        st.markdown('<div class="section-header">Full Metrics — All Companies and Models</div>', unsafe_allow_html=True)
        df_display = df_all.rename(columns={"MAE": "MAE ($)", "RMSE": "RMSE ($)", "MAPE": "MAPE (%)"})
        st.dataframe(
            df_display.style
                .highlight_min(subset=["MAE ($)", "RMSE ($)", "MAPE (%)"], axis=0, color="#052e16")
                .set_properties(**{"color": "#F8FAFC", "background-color": "#1E293B", "border": "1px solid #263548"}),
            use_container_width=True,
        )

        # RMSE grouped bar
        st.markdown('<div class="section-header">RMSE — All Companies</div>', unsafe_allow_html=True)
        tickers_list = list(df_all["Ticker"].unique())
        x = np.arange(len(tickers_list))
        w = 0.25

        fig6, ax6 = plt.subplots(figsize=(13, 5))
        for i, m_name in enumerate(["ANN", "RFR", "LSTM"]):
            vals = [df_all[(df_all["Ticker"]==t) & (df_all["Model"]==m_name)]["RMSE"].values[0]
                    for t in tickers_list]
            bars = ax6.bar(x + i*w, vals, w, label=m_name,
                           color=MODEL_COLORS[m_name], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax6.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.15, f"{val}",
                         ha="center", fontsize=9, color="#F8FAFC")
        ax6.set_xticks(x + w)
        ax6.set_xticklabels(tickers_list, fontsize=12)
        ax6.set_ylabel("RMSE — lower is better", fontsize=11)
        ax6.set_title("RMSE Comparison — All Companies", fontsize=12, pad=12)
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.4, axis="y")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        # MAPE grouped bar
        st.markdown('<div class="section-header">MAPE — All Companies</div>', unsafe_allow_html=True)
        fig7, ax7 = plt.subplots(figsize=(13, 5))
        for i, m_name in enumerate(["ANN", "RFR", "LSTM"]):
            vals = [df_all[(df_all["Ticker"]==t) & (df_all["Model"]==m_name)]["MAPE"].values[0]
                    for t in tickers_list]
            bars = ax7.bar(x + i*w, vals, w, label=m_name,
                           color=MODEL_COLORS[m_name], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax7.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.05, f"{val}%",
                         ha="center", fontsize=9, color="#F8FAFC")
        ax7.set_xticks(x + w)
        ax7.set_xticklabels(tickers_list, fontsize=12)
        ax7.set_ylabel("MAPE % — lower is better", fontsize=11)
        ax7.set_title("MAPE Comparison — All Companies", fontsize=12, pad=12)
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.4, axis="y")
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

        # Average per model
        summary = df_all.groupby("Model")[["MAE","RMSE","MAPE"]].mean().round(2)
        summary.columns = ["Avg MAE ($)", "Avg RMSE ($)", "Avg MAPE (%)"]

        st.markdown('<div class="section-header">Average Performance — All 4 Stocks</div>', unsafe_allow_html=True)
        st.dataframe(
            summary.style
                .highlight_min(axis=0, color="#052e16")
                .set_properties(**{"color": "#F8FAFC", "background-color": "#1E293B", "border": "1px solid #263548"}),
            use_container_width=True,
        )

        # Best model
        best_model   = summary["Avg RMSE ($)"].idxmin()
        worst_model  = summary["Avg RMSE ($)"].idxmax()
        second_model = [m for m in ["ANN","RFR","LSTM"]
                        if m != best_model and m != worst_model][0]
        best_rmse = summary.loc[best_model, "Avg RMSE ($)"]
        best_mape = summary.loc[best_model, "Avg MAPE (%)"]

        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-eyebrow">Best Overall Model</div>
            <div class="verdict-main">{best_model} &nbsp;—&nbsp; Avg RMSE ${best_rmse} &nbsp;|&nbsp; Avg MAPE {best_mape}%</div>
            <div class="verdict-sub">
                {best_model} achieved the lowest average prediction error
                across {', '.join(cfg.TICKERS)} from {cfg.START_DATE} to {cfg.END_DATE}.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rank cards
        st.markdown('<div class="section-header">Model Rankings</div>', unsafe_allow_html=True)
        ranked = summary["Avg RMSE ($)"].rank().to_dict()
        c1, c2, c3 = st.columns(3)
        for col, model in zip([c1, c2, c3], ["ANN", "RFR", "LSTM"]):
            rank  = int(ranked[model])
            rmse  = summary.loc[model, "Avg RMSE ($)"]
            mape  = summary.loc[model, "Avg MAPE (%)"]
            mae   = summary.loc[model, "Avg MAE ($)"]
            rank_class = {1: "rank-1", 2: "rank-2", 3: "rank-3"}[rank]
            rank_label = {1: "1st", 2: "2nd", 3: "3rd"}[rank]
            with col:
                st.markdown(f"""
                <div class="rank-card {rank_class}">
                    <div class="rank-num">{rank_label}</div>
                    <div class="rank-model">{model}</div>
                    <div class="rank-stat">RMSE &nbsp; ${rmse}</div>
                    <div class="rank-stat">MAPE &nbsp; {mape}%</div>
                    <div class="rank-stat">MAE &nbsp;&nbsp; ${mae}</div>
                </div>
                """, unsafe_allow_html=True)

        # Final verdict table
        st.markdown('<div class="section-header">Final Verdict</div>', unsafe_allow_html=True)
        verdict_df = pd.DataFrame([
            {"Rank": "1st", "Model": best_model,
             "Avg RMSE ($)": summary.loc[best_model,   "Avg RMSE ($)"],
             "Avg MAPE (%)": summary.loc[best_model,   "Avg MAPE (%)"],
             "Verdict": "Best — lowest error"},
            {"Rank": "2nd", "Model": second_model,
             "Avg RMSE ($)": summary.loc[second_model, "Avg RMSE ($)"],
             "Avg MAPE (%)": summary.loc[second_model, "Avg MAPE (%)"],
             "Verdict": "Middle"},
            {"Rank": "3rd", "Model": worst_model,
             "Avg RMSE ($)": summary.loc[worst_model,  "Avg RMSE ($)"],
             "Avg MAPE (%)": summary.loc[worst_model,  "Avg MAPE (%)"],
             "Verdict": "Worst — highest error"},
        ])
        st.dataframe(verdict_df.set_index("Rank"), use_container_width=True)

        st.markdown(f"""
        <div style="margin-top:20px;padding:22px 28px;background:#161B22;
                    border:1px solid #1E293B;border-radius:12px;
                    font-size:14px;color:#64748b;line-height:1.8">
            <strong style="color:#F8FAFC">Conclusion:</strong>
            Based on real trained models and Yahoo Finance data from
            {cfg.START_DATE} to {cfg.END_DATE},
            <strong style="color:#00C896">{best_model}</strong>
            is the most accurate model for predicting next-day closing prices
            across {', '.join(cfg.TICKERS)},
            achieving an average RMSE of ${best_rmse} and MAPE of {best_mape}%.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:12px;color:#1E293B;padding:8px 0;letter-spacing:1px">
    PYTHON &nbsp;·&nbsp; TENSORFLOW &nbsp;·&nbsp; SCIKIT-LEARN &nbsp;·&nbsp;
    STREAMLIT &nbsp;·&nbsp; YAHOO FINANCE
</div>
""", unsafe_allow_html=True)

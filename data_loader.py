# ── data_loader.py ───────────────────────────────────────────────────────────
"""
Downloads stock data, computes SMA + RSI, normalises, and splits.

Critical fixes vs naïve implementations
────────────────────────────────────────
1. Scaler is fit ONLY on training data, then applied to test data.
   Fitting on the full dataset lets future statistics leak into training.

2. Separate scalers for features (X) and target (y) so inverse-transforming
   predictions is unambiguous — no column-index guessing required.

3. Chronological split (shuffle=False) — mandatory for time-series.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config as cfg

log = logging.getLogger(__name__)


# ─── Technical indicators ────────────────────────────────────────────────────

def _sma(close: pd.Series, w: int) -> pd.Series:
    window = max(2, min(w, len(close)))
    return close.rolling(window, min_periods=1).mean().bfill().ffill()


def _rsi(close: pd.Series, w: int) -> pd.Series:
    window = max(2, min(w, len(close)))
    delta = close.diff().fillna(0)
    gain  = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs    = gain / (loss + 1e-9)          # avoid division by zero
    rsi   = 100 - 100 / (1 + rs)
    return rsi.bfill().ffill()


# ─── Main loader ─────────────────────────────────────────────────────────────

def load_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance, add SMA + RSI, add Target column.

    Returns a clean DataFrame ready for model ingestion.
    Raises ValueError if the download is empty.
    """
    import yfinance as yf
    log.info("Downloading %s from Yahoo Finance ...", ticker)
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}. "
                         f"Check ticker symbol and internet connection.")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["SMA"]    = _sma(df["Close"], cfg.SMA_WINDOW)
    df["RSI"]    = _rsi(df["Close"], cfg.RSI_WINDOW)
    df["Target"] = df["Close"].shift(-1).bfill()   # predict next-day close
    df.dropna(inplace=True)

    log.info("%s -> %d rows  |  Close %.2f - %.2f",
             ticker, len(df), df["Close"].min(), df["Close"].max())
    return df


# ─── Split + scale ───────────────────────────────────────────────────────────

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume", "SMA", "RSI"]


def prepare_data(df: pd.DataFrame):
    """
    Chronological split followed by leak-free Min-Max scaling.

    Returns
    -------
    X_train, X_test  : scaled feature arrays  (n, 7)
    y_train, y_test  : scaled target arrays   (n,)
    x_scaler         : fitted MinMaxScaler for X  (needed to rescale LSTM seqs)
    y_scaler         : fitted MinMaxScaler for y  (needed to inverse-transform preds)
    """
    X = df[FEATURE_COLS].values
    y = df["Target"].values.reshape(-1, 1)

    cut = int(len(X) * (1 - cfg.TEST_SIZE))
    if cut < 2:
        cut = max(1, len(X) // 2)

    X_tr, X_te = X[:cut], X[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr)   # fit on train only ← no leakage
    X_te = x_scaler.transform(X_te)

    y_scaler = MinMaxScaler()
    y_tr = y_scaler.fit_transform(y_tr).ravel()
    y_te = y_scaler.transform(y_te).ravel()

    log.info("Split -> train %d | test %d", len(X_tr), len(X_te))
    return X_tr, X_te, y_tr, y_te, x_scaler, y_scaler


# ─── LSTM sequences ──────────────────────────────────────────────────────────

def make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping look-back windows for LSTM.
    If X is shorter than seq_len, uses edge padding so sequences are always valid.

    Returns X_seq (n, seq_len, features) and y_seq (n,).
    """
    Xs, ys = [], []
    if len(X) <= seq_len:
        for i in range(len(X)):
            seq = X[: i + 1]
            pad_len = seq_len - len(seq)
            padded = np.pad(seq, ((pad_len, 0), (0, 0)), mode="edge")
            Xs.append(padded)
            ys.append(y[i])
    else:
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len : i])
            ys.append(y[i])
    return np.array(Xs), np.array(ys)

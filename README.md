# 📈 StockSight — Stock Price Prediction using Machine Learning

<p align="center">
  <a href="https://shreshthaa20-stock-prediction-app-lzbq5n.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-CPU-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/yfinance-Data-0A66C2" />
</p>

---

## 🌐 Live App

> **👉 [https://shreshthaa20-stock-prediction-app-lzbq5n.streamlit.app/](https://shreshthaa20-stock-prediction-app-lzbq5n.streamlit.app/)**

---

## 📖 Overview

**StockSight** is an interactive stock price prediction web app built with Streamlit. It trains and compares three machine learning models — **ANN**, **Random Forest Regressor (RFR)**, and **LSTM** — on real historical stock data fetched via `yfinance`, and visualises predictions alongside actual prices with performance metrics.

---

## ✨ Features

- 📊 **Multi-model comparison** — ANN, RFR, and LSTM trained and evaluated side-by-side
- 📉 **Real stock data** — fetched live using `yfinance` with technical indicators (SMA, RSI)
- 🏆 **Performance metrics** — MAE, RMSE, and MAPE for each model per ticker
- 🏢 **All-companies comparison** — bar chart breakdown across all tracked stocks
- 🎨 **Dark themed UI** — custom navy/dark blue Streamlit theme
- 💾 **Cached predictions** — saved models loaded instantly without retraining

---

## 🤖 Models

| Model | Type | Description |
|---|---|---|
| **ANN** | Artificial Neural Network | Feedforward neural network trained on feature sequences |
| **RFR** | Random Forest Regressor | Ensemble tree-based model for tabular feature prediction |
| **LSTM** | Long Short-Term Memory | Recurrent neural network for sequential time-series prediction |

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| **UI** | Streamlit |
| **Data** | yfinance, pandas, numpy |
| **ML / DL** | TensorFlow (CPU), scikit-learn |
| **Visualisation** | Matplotlib |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

```
Stock_Prediction_using-_machine_learning/
├── app.py              # Streamlit web app (main entry point)
├── main.py             # Trains all models and exports metrics
├── models.py           # Model definitions (ANN, RFR, LSTM)
├── data_loader.py      # Data fetching, feature engineering & sequencing
├── evaluate.py         # Metric calculations (MAE, RMSE, MAPE)
├── config.py           # Global config (dates, sequence length, tickers)
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python runtime version for deployment
├── saved_models/       # Pre-trained model files (per ticker)
└── outputs/            # metrics.csv and prediction charts
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shreshthaa20/Stock_Prediction_using-_machine_learning.git
cd Stock_Prediction_using-_machine_learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models (optional — pre-trained models are included)

```bash
python main.py
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Usage

1. Open the app (live or local)
2. Select a **stock ticker** from the sidebar
3. Click **Predict** to load model predictions
4. View the **prediction chart** comparing Actual vs ANN vs RFR vs LSTM
5. Check **MAE / RMSE / MAPE** metrics for each model
6. Switch to **All Companies Comparison** to see which model performs best overall

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using Streamlit & TensorFlow</p>

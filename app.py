import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# API CoinMarketCap
API_KEY = "8ecffa96-9482-44ac-93e2-dd37d007c639"  # Ganti dengan API Key Anda
CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# Fungsi untuk mengambil data BTC dari CoinMarketCap API
def get_btc_data():
    params = {"symbol": "BTC", "convert": "IDR"}
    headers = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": API_KEY}
    response = requests.get(CMC_URL, headers=headers, params=params)
    
    if response.status_code != 200:
        st.error(f"Error mengambil data dari CoinMarketCap API: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    try:
        price = data["data"]["BTC"]["quote"]["IDR"]["price"]
        timestamp = pd.to_datetime(data["status"]["timestamp"])
        df = pd.DataFrame({
            "timestamp": [timestamp],
            "close": [price],
            "open": [price],
            "high": [price],
            "low": [price],
            "volume": [np.nan]
        })
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan data API: {e}")
        return pd.DataFrame()
    
    return df

# Fungsi untuk menghitung indikator teknikal
def add_indicators(df):
    if df.empty or "close" not in df.columns:
        return df

    df["SMA"] = SMAIndicator(df["close"], window=14).sma_indicator()
    df["EMA"] = EMAIndicator(df["close"], window=14).ema_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    return df

# Fungsi untuk model prediksi menggunakan Random Forest
def train_model(df):
    if df.empty:
        return df

    X = df[["SMA", "EMA", "RSI", "MACD", "MACD_Signal"]]
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)

    if X.isnull().sum().sum() > 0 or np.isnan(y).sum() > 0:
        st.error("Terdapat nilai NaN dalam dataset setelah preprocessing.")
        return df
    
    X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    df["Prediksi"] = model.predict(X)
    
    return df

# Streamlit UI
st.title("Aplikasi Sinyal BTC dengan Analisis Teknikal dan Machine Learning")
df = get_btc_data()

if not df.empty:
    df = add_indicators(df)
    df = train_model(df)

    if not df.empty:
        # Visualisasi Candlestick & Indikator
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], 
                                     low=df["low"], close=df["close"], name="Candlestick"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SMA"], mode="lines", name="SMA"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA"], mode="lines", name="EMA"))

        # Tambahkan sinyal beli dan jual
        buy_signals = df[df["Prediksi"] == 1]
        sell_signals = df[df["Prediksi"] == 0]
        fig.add_trace(go.Scatter(x=buy_signals["timestamp"], y=buy_signals["close"], mode="markers", 
                                 marker=dict(color="green", size=10), name="Beli"))
        fig.add_trace(go.Scatter(x=sell_signals["timestamp"], y=sell_signals["close"], mode="markers", 
                                 marker=dict(color="red", size=10), name="Jual"))

        st.plotly_chart(fig)

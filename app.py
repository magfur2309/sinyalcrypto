import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Fungsi untuk mengambil data BTC dari Tokocrypto API
def get_btc_data():
    url = "https://api.tokocrypto.com/v1/market/kline?symbol=BTCIDR&interval=1d&limit=90"
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"Error mengambil data dari Tokocrypto API: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    if "data" not in data or not data["data"]:
        st.error("API Tokocrypto tidak mengembalikan data yang valid.")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
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

    df.fillna(method="bfill", inplace=True)  # Isi NaN dengan nilai sebelumnya
    df.fillna(method="ffill", inplace=True)  # Isi NaN dengan nilai setelahnya
    df.dropna(inplace=True)  # Hapus baris yang masih memiliki NaN

    return df

# Fungsi untuk model prediksi menggunakan Random Forest
def train_model(df):
    if df.empty:
        return df

    X = df[["SMA", "EMA", "RSI", "MACD", "MACD_Signal"]]
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)  # 1 = beli, 0 = jual

    if X.isnull().sum().sum() > 0 or np.isnan(y).sum() > 0:
        st.error("Terdapat nilai NaN dalam dataset setelah preprocessing.")
        return df

    # Pisahkan Data Training & Testing
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

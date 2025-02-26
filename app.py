import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Fungsi untuk mengambil data BTC dari Binance API
def get_btc_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=100"
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                         "close_time", "quote_asset_volume", "trades", 
                                         "taker_base_vol", "taker_quote_vol", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# Fungsi untuk menghitung indikator teknikal
def add_indicators(df):
    df["SMA"] = SMAIndicator(df["close"], window=14).sma_indicator()
    df["EMA"] = EMAIndicator(df["close"], window=14).ema_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    return df

# Fungsi untuk model prediksi menggunakan Random Forest
def train_model(df):
    df = df.dropna()
    X = df[["SMA", "EMA", "RSI", "MACD", "MACD_Signal"]]
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)  # 1 = beli, 0 = jual

    # Periksa apakah ada nilai NaN
    if X.isnull().sum().sum() > 0 or np.isnan(y).sum() > 0:
        raise ValueError("Terdapat nilai NaN dalam dataset setelah preprocessing.")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df["Prediksi"] = model.predict(X)
    return df

# Streamlit UI
st.title("Aplikasi Sinyal BTC dengan Analisis Teknikal dan Machine Learning")
df = get_btc_data()
df = add_indicators(df)

# Logging jumlah data sebelum training
st.write("Jumlah data sebelum training:", len(df))
st.write("Jumlah data setelah dropna:", len(df.dropna()))

df = train_model(df)

# Visualisasi data
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], 
                             low=df["low"], close=df["close"], name="Candlestick"))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SMA"], mode="lines", name="SMA"))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA"], mode="lines", name="EMA"))

buy_signals = df[df["Prediksi"] == 1]
sell_signals = df[df["Prediksi"] == 0]
fig.add_trace(go.Scatter(x=buy_signals["timestamp"], y=buy_signals["close"], mode="markers", 
                         marker=dict(color="green", size=10), name="Beli"))
fig.add_trace(go.Scatter(x=sell_signals["timestamp"], y=sell_signals["close"], mode="markers", 
                         marker=dict(color="red", size=10), name="Jual"))

st.plotly_chart(fig)

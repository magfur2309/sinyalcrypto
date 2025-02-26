import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Konfigurasi Telegram
TELEGRAM_TOKEN = "t.me/ngepirbot"
CHAT_ID = "7692585926:AAF0Wxcaco0-tc5n_n41oe6lKUB-bEg4-ic"

# Fungsi untuk mengirim notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    response = requests.get(url, params=params)
    return response.json()

# Fungsi untuk mengambil data BTC dari CoinGecko API
def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=idr&days=30&interval=daily"
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error(f"Error mengambil data dari CoinGecko API: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    if "prices" not in data:
        st.error("API CoinGecko tidak mengembalikan data yang valid.")
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = np.nan  # CoinGecko tidak menyediakan volume pada endpoint ini
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan data API: {e}")
        return pd.DataFrame()
    
    if df.empty:
        st.error("Dataset dari API kosong.")
    
    return df

# Fungsi untuk menghitung indikator teknikal
def add_indicators(df):
    if "close" not in df.columns:
        st.error("Kolom 'close' tidak ditemukan dalam dataset. Pastikan API mengembalikan data yang benar.")
        return pd.DataFrame()
    
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
    
    # Jika dataset kosong setelah dropna, hentikan proses
    if df.empty:
        st.error("Dataset kosong setelah preprocessing. Periksa data yang diambil dari API.")
        return df
    
    X = df[["SMA", "EMA", "RSI", "MACD", "MACD_Signal"]]
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)  # 1 = beli, 0 = jual
    
    # Isi NaN jika masih ada setelah dropna
    X = X.fillna(method='bfill').fillna(method='ffill')
    
    # Periksa apakah ada nilai NaN
    if X.isnull().sum().sum() > 0 or np.isnan(y).sum() > 0:
        st.error("Terdapat nilai NaN dalam dataset setelah preprocessing.")
        return df
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df["Prediksi"] = model.predict(X)
    
    # Kirim sinyal ke Telegram
    latest_signal = df.iloc[-1]
    if latest_signal["Prediksi"] == 1:
        message = f"🚀 Sinyal BELI BTC! Harga saat ini: {latest_signal['close']:.2f} IDR"
    else:
        message = f"⚠️ Sinyal JUAL BTC! Harga saat ini: {latest_signal['close']:.2f} IDR"
    
    response = send_telegram_message(message)
    st.write("Respon Telegram:", response)
    
    return df

# Streamlit UI
st.title("Aplikasi Sinyal BTC dengan Analisis Teknikal dan Machine Learning")
df = get_btc_data()

# Logging jumlah data setelah pengambilan API
st.write("Jumlah data dari API:", len(df))

if not df.empty:
    df = add_indicators(df)

    # Logging jumlah data sebelum training
    st.write("Jumlah data sebelum training:", len(df))
    st.write("Jumlah data setelah dropna:", len(df.dropna()))

    df = train_model(df)

    if not df.empty:
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

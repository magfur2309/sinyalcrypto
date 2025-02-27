import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
import datetime
import plotly.io as pio

# Telegram Bot
TELEGRAM_TOKEN = "7692585926:AAF0Wxcaco0-tc5n_n41oe6lKUB-bEg4-ic"
TELEGRAM_CHAT_ID = "5590432269"
import streamlit as st

# Fungsi untuk mengirim notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

# Fungsi untuk mengirim gambar ke Telegram
def send_telegram_photo(photo_path, caption="📊 Sinyal BTC Terbaru!"):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        files = {"photo": photo}
        payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        response = requests.post(url, data=payload, files=files)
    
    if response.status_code == 200:
        print("✅ Gambar terkirim ke Telegram!")
    else:
        print(f"🚨 Gagal mengirim gambar! Error: {response.text}")

# Fungsi untuk mengambil data BTC dari CoinGecko API
def get_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=idr&days=90&interval=daily"
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
        df["volume"] = np.nan
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan data API: {e}")
        return pd.DataFrame()
    
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
    
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)
    
    return df

# Fungsi untuk model prediksi menggunakan Random Forest
def train_model(df):
    X = df[["SMA", "EMA", "RSI", "MACD", "MACD_Signal"]]
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    
    if X.isnull().sum().sum() > 0 or np.isnan(y).sum() > 0:
        st.error("Terdapat nilai NaN dalam dataset setelah preprocessing.")
        return df
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df["Prediksi"] = model.predict(X)
    
    return df

# Streamlit UI
st.title("Aplikasi Sinyal BTC dengan Analisis Teknikal dan Machine Learning")
df = get_btc_data()

if not df.empty:
    df = add_indicators(df)
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
        
        # Simpan dan kirim grafik ke Telegram
        chart_path = "chart.png"
        pio.write_image(fig, chart_path)
        send_telegram_photo(chart_path, "📊 Sinyal BTC dengan Analisis Teknikal 📈")
        
        # Kirim notifikasi Telegram
        if not buy_signals.empty:
            send_telegram_message(f"🚀 Sinyal BELI: Harga BTC saat ini Rp{buy_signals['close'].iloc[-1]:,.0f}\nLONG {buy_signals['close'].iloc[-1] - 500} - {buy_signals['close'].iloc[-1] - 1000}\nSL {buy_signals['close'].iloc[-1] - 1500} (-4%)")
        if not sell_signals.empty:
            send_telegram_message(f"⚠️ Sinyal JUAL: Harga BTC saat ini Rp{sell_signals['close'].iloc[-1]:,.0f}\nSHORT {sell_signals['close'].iloc[-1] + 500} - {sell_signals['close'].iloc[-1] + 1000}\nSL {sell_signals['close'].iloc[-1] + 1500} (-4%)")

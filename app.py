import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ta as talib


# Konfigurasi Bot Telegram
TOKEN = "t.me/ngepirbot"
CHAT_ID = "7692585926:AAF0Wxcaco0-tc5n_n41oe6lKUB-bEg4-ic"

def send_signal(signal):
    message = f"🚀 Sinyal BTC: {signal}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    requests.get(url, params=params)

# Simulasi Data Harga BTC
def get_btc_data():
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 50000  # Simulasi harga
    df = pd.DataFrame({"Close": prices})
    df["SMA"] = talib.SMA(df["Close"], timeperiod=14)
    df["EMA"] = talib.EMA(df["Close"], timeperiod=14)
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)
    return df

# Model Machine Learning
model = RandomForestClassifier()

def train_model(df):
    df = df.dropna()
    X = df[["SMA", "EMA", "RSI"]]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    model.fit(X, y)

def predict_signal(df):
    last_data = df.dropna().iloc[-1]
    features = last_data[["SMA", "EMA", "RSI"]].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Beli" if prediction == 1 else "Jual"

# Streamlit UI
st.title("📊 Aplikasi Sinyal BTC")
data = get_btc_data()
st.line_chart(data["Close"], use_container_width=True)

if st.button("Prediksi Sinyal"):
    train_model(data)
    signal = predict_signal(data)
    st.write(f"Sinyal: **{signal}**")
    send_signal(signal)  # Kirim sinyal ke Telegram

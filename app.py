import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from sklearn.metrics import mean_squared_error
import os
import joblib

# 1. Load Data BTC dari Yahoo Finance
def load_btc_data(start='2022-01-01', end='2025-01-01'):
    try:
        df = yf.download('BTC-USD', start=start, end=end)
        return df[['Close']]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

data = load_btc_data()
if data is None:
    raise ValueError("Failed to fetch BTC data")

# 2. Normalisasi Data
scaler_path = "scaler.pkl"
scaler = MinMaxScaler(feature_range=(0,1))
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    data_scaled = scaler.transform(data)
else:
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, scaler_path)

# 3. Membuat Dataset untuk LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50
X, y = create_sequences(data_scaled, seq_length)

# 4. Split Data Train & Test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Membangun Model LSTM
model_path = "btc_lstm_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
else:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 6. Training Model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    model.save(model_path)
    print("Model saved successfully!")

# 7. Prediksi Harga BTC
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Evaluasi Model
mse = mean_squared_error(y_test_actual, predictions)
rmse = np.sqrt(mse)
print(f'MSE: {mse}, RMSE: {rmse}')

# 9. Plot Hasil Prediksi
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.legend()
plt.show()

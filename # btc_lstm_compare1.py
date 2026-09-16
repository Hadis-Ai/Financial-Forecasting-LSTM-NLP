# btc_lstm_compare.py
"""
مقایسه مدل‌ها — LSTM
- دانلود داده BTC-USD (Close) روزانه 10 سال
- ساخت دنباله‌ها، آموزش LSTM
- ارزیابی روی داده تست (MSE, RMSE, MAE)
- ذخیره نتایج actual vs predicted
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------- تنظیمات ----------
TICKER = "BTC-USD"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"
SEQ_LEN = 30
TEST_RATIO = 0.1
VAL_RATIO = 0.1
BATCH_SIZE = 16
EPOCHS = 25
RANDOM_SEED = 42
# -----------------------------

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# دانلود داده
df = yf.download(TICKER, start=START_DATE, end=END_DATE, interval="wkl", progress=False)
df = df[['Close']].dropna()
df['log-close'] = np.log(df['Close'])

# فقط ستون لگاریتمی رو برای آموزش مدل انتخاب کن
# نرمال‌سازی
scaler = MinMaxScaler(feature_range=(0,1))

scaled = scaler.fit_transform(df[['log-close']].values)

# ساخت دنباله‌ها
def create_sequences(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

# تقسیم train/val/test
n_total = X.shape[0]
n_test = int(n_total * TEST_RATIO)
n_train_val = n_total - n_test
n_val = int(n_train_val * VAL_RATIO)
n_train = n_train_val - n_val

X_train = X[:n_train]
y_train = y[:n_train]
X_val = X[n_train:n_train+n_val]
y_val = y[n_train:n_train+n_val]
X_test = X[n_train_val:]
y_test = y[n_train_val:]

# مدل LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, 1)))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))


model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# آموزش
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

# پیش‌بینی روی تست
# ۱. پیش‌بینی روی داده‌های تست
y_pred_scaled = model.predict(X_test)

# ۲. برگرداندن به حالت لگاریتمی (بدون تبدیل به قیمت واقعی)
y_test_log = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

# ۳. محاسبه معیارها روی لگاریتم (دقیقاً مثل ایویوز)
mse = mean_squared_error(y_test_log, y_pred_log)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_log, y_pred_log)

# حالا چاپ کن؛ می‌بینی که عددها کوچک شدند!
print(f"Test -> MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

# ۴. حالا برای رسم نمودار، به قیمت واقعی (دلار) برگرد
y_test_inv = np.exp(y_test_log)
y_pred_inv = np.exp(y_pred_log)
# ذخیره نتایج actual vs predicted
results_df = pd.DataFrame({
    'date': df.index[-len(y_test_inv):],
    'actual': y_test_inv.flatten(),
    'predicted': y_pred_inv.flatten()
})
results_df.to_csv("btc_lstm_results_test.csv", index=False)
print("Saved btc_lstm_results_test.csv")

# رسم نمودار actual vs predicted
plt.figure(figsize=(12,6))
plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual')
plt.plot(df.index[-len(y_pred_inv):], y_pred_inv, label='Predicted')
plt.title('BTC Close - Actual vs Predicted (LSTM)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
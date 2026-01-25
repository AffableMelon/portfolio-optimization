import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pmdarima as pm

logging.basicConfig(level=logging.INFO)

def split_data(data, train_size=0.8):
    """
    Splits data into training and testing sets chronologically.
    """
    split_idx = int(len(data) * train_size)
    train, test = data[:split_idx], data[split_idx:]
    return train, test

def train_arima(train_data, order=(5,1,0)):
    """
    Trains an ARIMA model.
    """
    logging.info(f"Training ARIMA with order {order}...")
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps):
    """
    Forecasts with ARIMA model.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(train_data, seq_length=60, epochs=10, batch_size=32):
    """
    Trains an LSTM model.
    Expected data shape: (samples, features) meaning 2D array.
    """
    logging.info("Training LSTM...")
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    X_train, y_train = create_sequences(scaled_train, seq_length)
    
    # Reshape for LSTM (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, scaler, history

def forecast_lstm(model, data, scaler, seq_length, steps):
    """
    Forecasts next 'steps' days using LSTM.
    Need the last 'seq_length' days from data to start prediction.
    """
    # Start with the last sequence from data
    input_seq = data[-seq_length:].values.reshape(-1, 1)
    scaled_input_seq = scaler.transform(input_seq)
    
    curr_seq = scaled_input_seq.reshape(1, seq_length, 1)
    predictions = []
    
    for _ in range(steps):
        pred_scaled = model.predict(curr_seq)
        predictions.append(pred_scaled[0, 0])
        
        # Update sequence with prediction
        curr_seq = np.append(curr_seq[:, 1:, :], [[pred_scaled[0]]], axis=1)
        
    predictions_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions_unscaled

def evaluate_forecast(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mape = np.mean(np.abs((true - predicted) / true)) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

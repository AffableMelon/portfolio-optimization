import pandas as pd
import numpy as np
import logging
from pmdarima import auto_arima
from pmdarima.arima import ARIMA as PMArima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Setup logging
logging.basicConfig(level=logging.INFO)

def train_test_split_series(series, split_date):
    """Split a time series into train/test sets based on a date."""
    series = series.copy()

    # Ensure datetime index and remove timezone info so downstream libs stay happy
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, utc=True)
    
    # Handle the case where it's already a DatetimeIndex but might be tz-aware or mixed
    if series.index.tz is not None:
        series.index = series.index.tz_convert(None)
    elif series.index.dtype == 'object':
        # Re-verify if it's still object after to_datetime (could happen if mixed)
        series.index = pd.to_datetime(series.index, utc=True).tz_convert(None)

    # Normalize split_date to naive Timestamp for consistent comparisons
    split_date = pd.to_datetime(split_date, utc=True).tz_convert(None)

    # Chronological split
    train = series.loc[series.index < split_date].dropna()
    test = series.loc[series.index >= split_date].dropna()

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    return train, test


def fit_arima_model(train, seasonal=True, m=1):
    logging.info("Fitting Auto ARIMA model...")
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=m,
        stepwise=False,        # Exhaustive search for better patterns
        with_intercept=True,   # Enforce a trend component
        allowdrift=True,       # Allow the line to slope up/down
        error_action='ignore',
        suppress_warnings=True
    )
    
    # If the model is still too simple to be useful, force a more dynamic order
    order = getattr(model, 'order', (0,0,0))
    if order in [(0, 1, 0), (1, 1, 0), (0, 1, 1)]:
        logging.info(f"Model {order} is too simple; forcing dynamic ARIMA(2,1,2).")
        model = PMArima(order=(2, 1, 2), with_intercept=True).fit(train)
        
    return model

def fit_arima_model2(train, seasonal=True, m=1):
    """Fit an ARIMA/SARIMA model using auto_arima."""
    logging.info("Fitting Auto ARIMA model...")
    model = auto_arima(
        train,
        start_p=1, start_q=1,
        max_p=15, max_q=15,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None,
        max_d=2,
        D=None,
        max_D=1,
        seasonal=seasonal,
        m=m,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        with_intercept=True,
        allowdrift=True,
        allowmean=True,
        stationary=False
    )

    def _force_dynamic_model(train_series):
        logging.info("Auto ARIMA produced a random-walk forecast; forcing ARIMA(1,1,1) with intercept.")
        seasonal_order = (0, 0, 0, m) if seasonal and m > 1 else (0, 0, 0, 0)
        forced = PMArima(
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            with_intercept=True,
            suppress_warnings=True
        )
        forced.fit(train_series)
        return forced

    def _get_model_order(arima_model):
        order = None
        candidate_attrs = ['order', 'model_', 'arima_res_']
        for attr in candidate_attrs:
            value = getattr(arima_model, attr, None)
            if value is None:
                continue
            if attr == 'order':
                order = value() if callable(value) else value
                if order is not None:
                    break
            else:
                order = getattr(value, 'order', None)
                if callable(order):
                    order = order()
                if order is not None:
                    break
        return order

    # If auto_arima falls back to a random walk (flat forecast), refit with enforced AR/MA dynamics
    detected_order = _get_model_order(model)
    if detected_order == (0, 1, 0):
        model = _force_dynamic_model(train)
    else:
        preview_horizon = min(10, len(train))
        if preview_horizon > 1:
            preview = np.asarray(model.predict(n_periods=preview_horizon))
            if np.allclose(preview, preview[0]):
                model = _force_dynamic_model(train)

    return model

def forecast_and_evaluate(model, test, return_conf_int=True, alpha=0.05):
    """Forecast on the test set, evaluate metrics, and return confidence intervals."""
    
    actual_series = test.dropna()
    n_periods = len(actual_series)

    # Generate the forecast and optionally confidence intervals
    try:
        pred_out = model.predict(n_periods=n_periods, return_conf_int=return_conf_int, alpha=alpha)
    except:
        # Some models or versions might not accept alpha or return_conf_int the same way, but pmdarima usually does
        pred_out = model.predict(n_periods=n_periods)
        return_conf_int = False

    # model.predict may return just an array or (array, conf_int_array)
    if return_conf_int:
        try:
            forecast_vals, conf_int_array = pred_out
        except Exception:
            # unexpected format — try to coerce
            forecast_vals = np.asarray(pred_out)
            conf_int_array = None
    else:
        forecast_vals = np.asarray(pred_out)
        conf_int_array = None

    forecast_vals = np.asarray(forecast_vals)
    
    # Primary alignment: label-based using the test index (positional subset)
    test_index = actual_series.index
    if forecast_vals.shape[0] >= len(test_index):
        # take first len(test_index) predictions
        forecast_series = pd.Series(forecast_vals[: len(test_index)], index=test_index, name="forecast")
    else:
        # fewer predictions than test rows: align to the first N positions
        forecast_series = pd.Series(forecast_vals, index=test_index[: forecast_vals.shape[0]], name="forecast")

    # Build conf_int_df if available and match to the same index used for forecast_series
    if conf_int_array is not None:
        conf_arr = np.asarray(conf_int_array)
        # match rows to forecast_series length
        conf_len = conf_arr.shape[0]
        conf_idx = forecast_series.index[:conf_len]
        try:
            conf_int_df = pd.DataFrame(conf_arr[:conf_len], index=conf_idx, columns=['lower_ci', 'upper_ci'])
        except Exception:
            # fallback: create numeric columns without column names
            # If 1D array?
            if conf_arr.ndim == 1:
                conf_int_df = pd.DataFrame(conf_arr[:conf_len], index=conf_idx)
            else:
                conf_int_df = pd.DataFrame(conf_arr[:conf_len], index=conf_idx)
                if conf_int_df.shape[1] >= 2:
                    conf_int_df.columns = ['lower_ci', 'upper_ci']
    else:
        conf_int_df = pd.DataFrame(index=forecast_series.index)

    # Create evaluation DataFrame by joining on index
    df_eval = pd.concat([actual_series, forecast_series], axis=1)
    df_eval.columns = ["actual", "forecast"]
    df_eval.dropna(inplace=True)

    # If joining by labels produced an empty DataFrame, fallback to positional alignment
    if df_eval.empty:
        k = min(len(actual_series), len(forecast_vals))
        if k == 0:
            raise ValueError("No overlapping data to evaluate: actual or predicted series is empty.")
        pos_index = actual_series.index[:k]
        df_eval = pd.DataFrame({
            'actual': actual_series.values[:k],
            'forecast': forecast_vals[:k]
        }, index=pos_index)
        # update conf_int_df index to pos_index if possible
        if conf_int_array is not None:
            conf_int_df = conf_int_df.reindex(pos_index)

    # --- Metric Calculation ---
    non_zero_actuals = df_eval["actual"] != 0
    
    mae = mean_absolute_error(df_eval["actual"], df_eval["forecast"])
    rmse = np.sqrt(mean_squared_error(df_eval["actual"], df_eval["forecast"]))
    mape = np.mean(np.abs((df_eval["actual"][non_zero_actuals] - df_eval["forecast"][non_zero_actuals]) / df_eval["actual"][non_zero_actuals])) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    
    return forecast_series, conf_int_df, metrics


# --- LSTM ---- 
def create_lstm_sequences(data, sequence_length):
    """Create input sequences and corresponding labels for LSTM."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=32):
    """Builds, compiles, and trains an LSTM model."""
    logging.info("Training LSTM...")
    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = lstm_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    return lstm_model, history

def forecast_lstm(model, all_data, test_data, scaler, sequence_length):
    """Generate a forecast using a trained LSTM model."""
    
    # Need to access values properly
    dataset = all_data.values # Assuming pandas object
    inputs = dataset[len(dataset) - len(test_data) - sequence_length:]
    
    inputs = inputs.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)

    X_test = []
    # Logic in user snippet:
    for i in range(sequence_length, len(inputs_scaled)):
        X_test.append(inputs_scaled[i-sequence_length:i, 0])
    
    X_test = np.array(X_test)
    if X_test.shape[0] == 0:
        logging.warning("X_test is empty. Check data length.")
        return pd.Series(dtype=float)
        
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    forecast_series = pd.Series(predictions.flatten(), index=test_data.index, name="LSTM_Forecast")
    return forecast_series

def evaluate_forecast(actual, forecast, model_name="Model"):
    """Calculate and print evaluation metrics for a forecast."""
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        if np.isinf(mape): mape = np.nan
    
    print(f"\n--- {model_name} Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

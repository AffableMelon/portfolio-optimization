import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_future_forecast_arima(model, steps, alpha=0.05):
    """
    Generates forecast with confidence intervals using pmdarima or statsmodels ARIMA.
    """
    # Try pmdarima style
    try:
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
        # Convert to DataFrame for consistency
        conf_int_df = pd.DataFrame(conf_int, columns=['lower_ci', 'upper_ci'])
        return pd.Series(forecast), conf_int_df
    except:
        # Fallback to statsmodels style
        forecast_result = model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        return forecast, conf_int

def plot_forecast(history, forecast, conf_int=None, title="Forecast"):
    """
    Plots historical data and forecast with confidence intervals.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(history.index, history, label='History')
    plt.plot(forecast.index, forecast, label='Forecast', color='red')
    
    if conf_int is not None:
        plt.fill_between(forecast.index, 
                         conf_int.iloc[:, 0], 
                         conf_int.iloc[:, 1], 
                         color='pink', alpha=0.3, label='Confidence Interval')
    
    plt.title(title)
    plt.legend()
    plt.show()

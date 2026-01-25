import yfinance as yf
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical data for the given tickers from yfinance.

    Args:
        tickers (list): List of ticker symbols (e.g., ['TSLA', 'BND', 'SPY']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Combined DataFrame with all assets, or dictionary of DataFrames.
    """
    logging.info(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        
        # Determine structure based on number of tickers
        if len(tickers) == 1:
            # If single ticker, just return it but maybe add a Ticker column or keep simple
            data.columns = [f"{col}" for col in data.columns] # Flatten if needed or keep MultiIndex
            logging.info("Data fetched successfully.")
            return data
        
        # If multiple tickers, yfinance returns MultiIndex columns (Ticker, OHLCV)
        # We might want to stack it or keep it wide. 
        # For this project, stacking or having a clean structure is better.
        
        logging.info("Data fetched successfully.")
        return data

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def clean_data(data):
    """
    Cleans the financial data.
    
    Args:
        data (pd.DataFrame): Raw data from yfinance.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    logging.info("Cleaning data...")
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        logging.warning("Missing values found. Filling with forward fill.")
        data = data.ffill().bfill()
        
    # Ensure correct data types (yfinance usually gives floats)
    data = data.astype(float)
    
    logging.info("Data cleaning complete.")
    return data

def save_data(data, filepath):
    """
    Saves the data to a CSV file.
    
    Args:
        data (pd.DataFrame): Data to save.
        filepath (str): Path to save the CSV.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath)
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

if __name__ == "__main__":
    assets = ['TSLA', 'BND', 'SPY']
    start = '2015-01-01'
    end = '2026-01-15'
    
    raw_data = fetch_data(assets, start, end)
    cleaned_data = clean_data(raw_data)
    save_data(cleaned_data, "data/processed/historical_data.csv")

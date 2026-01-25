# Portfolio Optimization and Financial Forecasting

This project implements a comprehensive pipeline for financial time series analysis, forecasting, and portfolio optimization. It leverages historical stock data to build predictive models using statistical (ARIMA) and deep learning (LSTM) techniques, ultimately aiming to construct an optimized portfolio strategy.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results and Metrics](#results)
- [Usage](#usage)

## Overview

The primary objective of this repository is to apply advanced data science techniques to financial markets. The workflow is divided into five distinct tasks:

1.  **Exploratory Data Analysis (EDA)**: Understanding data distributions, trends, and volatility.
2.  **Modeling**: Building and training ARIMA and LSTM models for price prediction.
3.  **Forecasting**: Generating future price predictions.
4.  **Optimization**: allocating portfolio weights to maximize returns or minimize risk.
5.  **Backtesting**: Evaluating the strategy against historical data.

The analysis focuses on a portfolio consisting of high-growth stocks (e.g., **TSLA**) and market indices/bonds (e.g., **SPY**, **BND**).

## Project Structure

```text
portfolio-optimization/
│
├── data/
│   └── processed/          # Stores cleaned and preprocessed data (CSV)
│
├── notebooks/              # Jupyter notebooks for interactive analysis
│   ├── task_1_eda.ipynb           # Data fetching and initial visualization
│   ├── task_2_modeling.ipynb      # ARIMA and LSTM model training
│   ├── task_3_forecasting.ipynb   # Future price forecasting
│   ├── task_4_optimization.ipynb  # Portfolio weight optimization
│   └── task_5_backtesting.ipynb   # Strategy performance evaluation
│
├── src/                    # Source code and utility modules
│   ├── data_loader.py      # Functions for fetching and cleaning data
│   ├── forecasting.py      # Logic for generating forecasts
│   ├── models.py           # Model definitions (ARIMA, LSTM) and training loops
│   └── portfolio_optimization.py # Optimization algorithms
│
├── scripts/                # Standalone execution scripts
│
└── requirements.txt        # Project dependencies
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/AffableMelon/portfolio-optimization.git
    cd portfolio-optimization
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Methodology

### 1. Data Preparation
Historical data is fetched, cleaned, and normalized. Missing values are handled, and the time series data is set to a daily frequency.

### 2. Time Series Modeling
Two distinct approaches are used to model the stock price movements:

-   **ARIMA (AutoRegressive Integrated Moving Average)**: A statistical model that captures linear trends and seasonality. It is well-suited for short-term forecasting where the data shows stationarity or can be made stationary.
-   **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) capable of learning long-term dependencies and non-linear patterns in the data.

### 3. Portfolio Optimization
Modern Portfolio Theory (MPT) principles are applied to determine the optimal asset allocation. The goal is to maximize the Sharpe Ratio (risk-adjusted return).

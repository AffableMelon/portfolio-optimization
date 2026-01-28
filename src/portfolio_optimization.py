import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt

def calculate_expected_returns(data, forecast_data=None):
    """
    Calculates expected annual returns.
    If forecast_data is provided (dict of ticker -> annualized_return), overrides historical mean.
    """
    # Calculate historical mean returns (annualized)
    mu = expected_returns.mean_historical_return(data)
    
    if forecast_data:
        for ticker, ret in forecast_data.items():
            mu[ticker] = ret
            
    return mu

def calculate_covariance_matrix(data):
    """
    Calculates the covariance matrix of daily returns, annualized.
    """
    S = risk_models.sample_cov(data)
    return S

def optimize_portfolio(mu, S):
    """
    Optimizes portfolio for Max Sharpe and Min Volatility.
    """
    ef = EfficientFrontier(mu, S)
    
    # Max Sharpe
    weights_max_sharpe = ef.max_sharpe()
    weights_max_sharpe = dict(weights_max_sharpe)
    perf_max_sharpe = ef.portfolio_performance(verbose=True)
    
    # Reset for Min Volatility
    ef = EfficientFrontier(mu, S)
    weights_min_vol = ef.min_volatility()
    weights_min_vol = dict(weights_min_vol)
    perf_min_vol = ef.portfolio_performance(verbose=True)
    
    return {
        "max_sharpe": {"weights": weights_max_sharpe, "performance": perf_max_sharpe},
        "min_vol": {"weights": weights_min_vol, "performance": perf_min_vol}
    }

def plot_efficient_frontier(mu, S, optimal_portfolios):
    ef = EfficientFrontier(mu, S)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate random portfolios for plotting
    n_samples = 10000
    w = np.random.dirichlet(np.ones(len(mu)), n_samples)
    rets = w.dot(mu)
    stds = np.sqrt(np.diag(w @ S @ w.T))
    sharpes = rets / stds
    
    ax.scatter(stds, rets, c=sharpes, cmap='viridis', marker='.', alpha=0.5)
    
    # Plot Efficient Frontier line (approximate or using pypfopt plotting)
    # Using pypfopt's plotting is cleaner but requires 'plotting' module import or manual
    
    # Plot Optimal Points
    max_sharpe = optimal_portfolios['max_sharpe']['performance']
    min_vol = optimal_portfolios['min_vol']['performance']
    
    ax.scatter(max_sharpe[1], max_sharpe[0], marker='*', color='r', s=200, label='Max Sharpe')
    ax.scatter(min_vol[1], min_vol[0], marker='*', color='b', s=200, label='Min Volatility')
    
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility (Std. Dev)')
    ax.set_ylabel('Expected Annual Return')
    ax.legend()
    plt.show()

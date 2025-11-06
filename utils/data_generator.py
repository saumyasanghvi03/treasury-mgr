import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate sample data for the treasury management system"""
    np.random.seed(42)
    
    # Dashboard metrics
    data = {
        'liquidity_total': 150_000_000,
        'liquidity_change': 3.2,
        'lcr_ratio': 125.5,
        'var_95': 2_500_000,
        'var_change': -1.8,
        'nim': 2.85,
        'nim_change': 0.15
    }
    
    # Cash flow forecast
    dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
    base_cashflow = 5_000_000
    trend = np.linspace(0, 1_000_000, 30)
    seasonal = 500_000 * np.sin(np.linspace(0, 4*np.pi, 30))
    noise = np.random.normal(0, 200_000, 30)
    forecast = base_cashflow + trend + seasonal + noise
    
    data['cashflow_forecast'] = pd.DataFrame({
        'date': dates,
        'forecast': forecast,
        'lower_bound': forecast - 500_000,
        'upper_bound': forecast + 500_000
    })
    
    # ALM Gap data
    buckets = ['0-1M', '1-3M', '3-6M', '6-12M', '1-2Y', '2-5Y', '5Y+']
    assets = np.array([25, 30, 35, 40, 45, 50, 55])
    liabilities = np.array([30, 35, 32, 38, 42, 48, 45])
    
    data['alm_gap'] = pd.DataFrame({
        'bucket': buckets,
        'assets': assets,
        'liabilities': liabilities,
        'gap': assets - liabilities
    })
    
    # Historical cash flows for ML training
    historical_dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='D'
    )
    historical_cashflow = 4_000_000 + np.cumsum(np.random.normal(10000, 100000, len(historical_dates)))
    
    data['historical_cashflow'] = pd.DataFrame({
        'date': historical_dates,
        'amount': historical_cashflow,
        'inflows': np.abs(np.random.normal(3_000_000, 500_000, len(historical_dates))),
        'outflows': np.abs(np.random.normal(2_800_000, 600_000, len(historical_dates)))
    })
    
    # Portfolio data for VaR
    data['portfolio'] = pd.DataFrame({
        'asset': ['Bonds', 'Equities', 'Money Market', 'Forex', 'Commodities'],
        'value': [50_000_000, 30_000_000, 40_000_000, 15_000_000, 10_000_000],
        'weight': [0.34, 0.21, 0.28, 0.10, 0.07],
        'volatility': [0.05, 0.20, 0.01, 0.15, 0.25]
    })
    
    # Generate returns for VaR calculation
    n_days = 252
    returns_data = {}
    for asset in data['portfolio']['asset']:
        vol = data['portfolio'][data['portfolio']['asset'] == asset]['volatility'].values[0]
        returns_data[asset] = np.random.normal(0.0001, vol/np.sqrt(252), n_days)
    
    data['historical_returns'] = pd.DataFrame(returns_data)
    
    # Basel III LCR components
    data['lcr_components'] = {
        'hqla_level1': 80_000_000,
        'hqla_level2a': 15_000_000,
        'hqla_level2b': 10_000_000,
        'total_hqla': 105_000_000,
        'total_net_outflows': 83_600_000,
        'lcr_ratio': 125.5
    }
    
    # Intraday liquidity data
    hours = pd.date_range(
        start=datetime.now().replace(hour=9, minute=0, second=0),
        periods=8,
        freq='H'
    )
    
    data['intraday_liquidity'] = pd.DataFrame({
        'time': hours,
        'available': [95, 88, 82, 78, 81, 85, 90, 93],
        'required': [80, 80, 80, 80, 80, 80, 80, 80],
        'buffer': [15, 8, 2, -2, 1, 5, 10, 13]
    })
    
    # Investment universe for optimizer
    data['investment_universe'] = pd.DataFrame({
        'instrument': [
            'US Treasury 2Y', 'US Treasury 5Y', 'US Treasury 10Y',
            'Corporate Bond AAA', 'Corporate Bond AA', 'Corporate Bond A',
            'Money Market Fund', 'Commercial Paper', 'Certificate of Deposit'
        ],
        'expected_return': [0.045, 0.042, 0.040, 0.055, 0.060, 0.070, 0.035, 0.048, 0.050],
        'risk': [0.02, 0.04, 0.06, 0.05, 0.07, 0.09, 0.01, 0.03, 0.03],
        'liquidity_score': [100, 95, 90, 70, 65, 60, 100, 85, 80],
        'min_investment': [1_000_000, 1_000_000, 1_000_000, 500_000, 500_000, 
                          500_000, 100_000, 250_000, 250_000],
        'max_investment': [50_000_000, 50_000_000, 50_000_000, 20_000_000, 
                          20_000_000, 15_000_000, 30_000_000, 25_000_000, 25_000_000]
    })
    
    return data

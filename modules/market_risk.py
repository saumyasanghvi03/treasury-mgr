"""
Market Risk Analytics Module

This module implements various market risk measurement techniques including
Historical VaR, Monte Carlo VaR, and shock scenario analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta


class MarketRiskAnalyzer:
    """
    Comprehensive market risk analyzer implementing multiple VaR methodologies
    and stress testing frameworks.
    """
    
    def __init__(self, confidence_level=0.95):
        """
        Initialize the market risk analyzer.
        
        Args:
            confidence_level (float): Confidence level for VaR calculation (default 0.95)
        """
        self.confidence_level = confidence_level
        self.historical_returns = None
        
    def calculate_returns(self, price_data):
        """
        Calculate returns from price data.
        
        Args:
            price_data (pd.DataFrame or pd.Series): Historical price data
            
        Returns:
            pd.DataFrame or pd.Series: Calculated returns
        """
        if isinstance(price_data, pd.Series):
            returns = price_data.pct_change().dropna()
        else:
            returns = price_data.pct_change().dropna()
        
        self.historical_returns = returns
        return returns
    
    def historical_var(self, returns=None, portfolio_value=1_000_000):
        """
        Calculate Historical Value at Risk (VaR).
        
        Historical VaR uses the actual distribution of historical returns
        to estimate potential losses.
        
        Args:
            returns (pd.Series or np.array): Historical returns (optional if already set)
            portfolio_value (float): Current portfolio value
            
        Returns:
            dict: VaR metrics including VaR, CVaR, and distribution statistics
        """
        if returns is None:
            if self.historical_returns is None:
                raise ValueError("Returns data must be provided.")
            returns = self.historical_returns
        
        returns_array = np.array(returns)
        
        # Calculate VaR at specified confidence level
        var_percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns_array, var_percentile)
        var_dollar = abs(var_return * portfolio_value)
        
        # Calculate Conditional VaR (Expected Shortfall)
        # CVaR is the expected loss given that VaR threshold is breached
        cvar_return = returns_array[returns_array <= var_return].mean()
        cvar_dollar = abs(cvar_return * portfolio_value)
        
        # Distribution statistics
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        
        results = {
            'method': 'Historical VaR',
            'confidence_level': self.confidence_level,
            'var_return': var_return,
            'var_dollar': var_dollar,
            'cvar_return': cvar_return,
            'cvar_dollar': cvar_dollar,
            'portfolio_value': portfolio_value,
            'mean_return': mean_return,
            'volatility': std_return,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'worst_return': returns_array.min(),
            'best_return': returns_array.max()
        }
        
        return results
    
    def parametric_var(self, returns=None, portfolio_value=1_000_000):
        """
        Calculate Parametric VaR (Variance-Covariance method).
        
        Assumes returns follow a normal distribution.
        
        Args:
            returns (pd.Series or np.array): Historical returns
            portfolio_value (float): Current portfolio value
            
        Returns:
            dict: Parametric VaR metrics
        """
        if returns is None:
            if self.historical_returns is None:
                raise ValueError("Returns data must be provided.")
            returns = self.historical_returns
        
        returns_array = np.array(returns)
        
        # Calculate mean and standard deviation
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var_return = mean_return + z_score * std_return
        var_dollar = abs(var_return * portfolio_value)
        
        # CVaR for normal distribution
        cvar_return = mean_return - std_return * (stats.norm.pdf(z_score) / (1 - self.confidence_level))
        cvar_dollar = abs(cvar_return * portfolio_value)
        
        results = {
            'method': 'Parametric VaR',
            'confidence_level': self.confidence_level,
            'var_return': var_return,
            'var_dollar': var_dollar,
            'cvar_return': cvar_return,
            'cvar_dollar': cvar_dollar,
            'portfolio_value': portfolio_value,
            'mean_return': mean_return,
            'volatility': std_return,
            'assumption': 'Normal Distribution'
        }
        
        return results
    
    def monte_carlo_var(self, returns=None, portfolio_value=1_000_000, 
                       num_simulations=10000, time_horizon=1, random_seed=None):
        """
        Calculate Monte Carlo VaR through simulation.
        
        Simulates future returns based on historical distribution parameters.
        
        Args:
            returns (pd.Series or np.array): Historical returns
            portfolio_value (float): Current portfolio value
            num_simulations (int): Number of Monte Carlo simulations
            time_horizon (int): Time horizon in days
            random_seed (int): Random seed for reproducibility (optional)
            
        Returns:
            dict: Monte Carlo VaR results with simulation details
        """
        if returns is None:
            if self.historical_returns is None:
                raise ValueError("Returns data must be provided.")
            returns = self.historical_returns
        
        returns_array = np.array(returns)
        
        # Calculate distribution parameters
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        
        # Run Monte Carlo simulations
        if random_seed is not None:
            np.random.seed(random_seed)
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            num_simulations
        )
        
        # Calculate simulated portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        simulated_losses = portfolio_value - simulated_values
        
        # Calculate VaR from simulations
        var_percentile = (1 - self.confidence_level) * 100
        var_dollar = np.percentile(simulated_losses, 100 - var_percentile)
        var_return = var_dollar / portfolio_value
        
        # Calculate CVaR
        cvar_dollar = simulated_losses[simulated_losses >= var_dollar].mean()
        cvar_return = cvar_dollar / portfolio_value
        
        results = {
            'method': 'Monte Carlo VaR',
            'confidence_level': self.confidence_level,
            'var_return': -var_return,
            'var_dollar': var_dollar,
            'cvar_return': -cvar_return,
            'cvar_dollar': cvar_dollar,
            'portfolio_value': portfolio_value,
            'num_simulations': num_simulations,
            'time_horizon': time_horizon,
            'mean_simulated_loss': simulated_losses.mean(),
            'max_simulated_loss': simulated_losses.max(),
            'simulated_values': simulated_values
        }
        
        return results
    
    def shock_scenario_analysis(self, portfolio_positions, shock_scenarios):
        """
        Perform shock scenario analysis on portfolio positions.
        
        Args:
            portfolio_positions (pd.DataFrame): Portfolio with columns
                ['asset', 'position_size', 'current_price']
            shock_scenarios (dict): Dictionary of shock scenarios
                Example: {
                    'equity_crash': {'equity': -0.30, 'bonds': 0.05},
                    'rates_spike': {'bonds': -0.15, 'equity': -0.10}
                }
                
        Returns:
            pd.DataFrame: Portfolio impact under each scenario
        """
        results = []
        
        # Calculate base portfolio value
        portfolio_positions['position_value'] = (
            portfolio_positions['position_size'] * portfolio_positions['current_price']
        )
        base_value = portfolio_positions['position_value'].sum()
        
        for scenario_name, shocks in shock_scenarios.items():
            scenario_positions = portfolio_positions.copy()
            
            # Apply shocks to each asset
            for asset_class, shock in shocks.items():
                mask = scenario_positions['asset_class'] == asset_class
                scenario_positions.loc[mask, 'shocked_price'] = (
                    scenario_positions.loc[mask, 'current_price'] * (1 + shock)
                )
            
            # Fill shocked prices for assets not in scenario
            scenario_positions['shocked_price'].fillna(
                scenario_positions['current_price'], inplace=True
            )
            
            # Calculate shocked values
            scenario_positions['shocked_value'] = (
                scenario_positions['position_size'] * scenario_positions['shocked_price']
            )
            shocked_value = scenario_positions['shocked_value'].sum()
            
            # Calculate P&L
            pnl = shocked_value - base_value
            pnl_pct = (pnl / base_value) * 100
            
            results.append({
                'scenario': scenario_name,
                'base_value': base_value,
                'shocked_value': shocked_value,
                'pnl': pnl,
                'pnl_percentage': pnl_pct,
                'max_drawdown': abs(min(pnl, 0))
            })
        
        return pd.DataFrame(results)
    
    def stress_test_correlation_breakdown(self, returns_df, normal_correlation):
        """
        Stress test assuming correlation breakdown (flight to quality).
        
        Args:
            returns_df (pd.DataFrame): Multi-asset returns data
            normal_correlation (float): Normal correlation coefficient
            
        Returns:
            dict: Stress test results
        """
        # Calculate actual correlation matrix
        actual_correlation = returns_df.corr()
        
        # Simulate stressed correlation (e.g., increased correlation in crisis)
        stressed_correlation = actual_correlation * 1.5
        stressed_correlation = stressed_correlation.clip(-1, 1)
        
        results = {
            'normal_correlation_matrix': actual_correlation,
            'stressed_correlation_matrix': stressed_correlation,
            'correlation_increase': (stressed_correlation - actual_correlation).mean().mean()
        }
        
        return results
    
    def calculate_volatility_metrics(self, returns=None, window=30):
        """
        Calculate various volatility metrics.
        
        Args:
            returns (pd.Series): Returns data
            window (int): Rolling window for calculations
            
        Returns:
            pd.DataFrame: Volatility metrics over time
        """
        if returns is None:
            if self.historical_returns is None:
                raise ValueError("Returns data must be provided.")
            returns = self.historical_returns
        
        # Historical volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        # EWMA volatility
        ewma_vol = returns.ewm(span=window).std() * np.sqrt(252)
        
        # Create metrics dataframe
        vol_metrics = pd.DataFrame({
            'date': returns.index if isinstance(returns, pd.Series) else range(len(returns)),
            'returns': returns.values,
            'rolling_volatility': rolling_vol.values,
            'ewma_volatility': ewma_vol.values
        })
        
        return vol_metrics
    
    def back_test_var(self, returns, var_estimates, portfolio_value=1_000_000):
        """
        Back-test VaR model accuracy.
        
        Args:
            returns (pd.Series): Actual returns
            var_estimates (pd.Series): VaR estimates for each period
            portfolio_value (float): Portfolio value
            
        Returns:
            dict: Back-testing statistics
        """
        returns_array = np.array(returns)
        var_array = np.array(var_estimates)
        
        # Calculate actual losses
        actual_losses = -returns_array * portfolio_value
        
        # Count VaR breaches
        breaches = actual_losses > var_array
        num_breaches = breaches.sum()
        total_observations = len(returns_array)
        
        # Expected breaches at confidence level
        expected_breaches = total_observations * (1 - self.confidence_level)
        
        # Calculate breach rate
        breach_rate = num_breaches / total_observations
        
        # Statistical test (Kupiec POF test)
        if num_breaches > 0:
            likelihood_ratio = -2 * (
                np.log((1 - self.confidence_level) ** num_breaches * 
                       self.confidence_level ** (total_observations - num_breaches)) -
                np.log((num_breaches / total_observations) ** num_breaches * 
                       (1 - num_breaches / total_observations) ** (total_observations - num_breaches))
            )
        else:
            likelihood_ratio = 0
        
        results = {
            'total_observations': total_observations,
            'num_breaches': num_breaches,
            'expected_breaches': expected_breaches,
            'breach_rate': breach_rate,
            'expected_breach_rate': 1 - self.confidence_level,
            'likelihood_ratio': likelihood_ratio,
            'model_acceptable': abs(breach_rate - (1 - self.confidence_level)) < 0.05
        }
        
        return results


def generate_sample_returns_data(num_days=252, num_assets=3):
    """
    Generate sample returns data for market risk analysis.
    
    Args:
        num_days (int): Number of days of returns
        num_assets (int): Number of assets
        
    Returns:
        pd.DataFrame: Sample returns data
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=num_days, freq='D')
    
    # Generate correlated returns
    mean_returns = [0.0005, 0.0003, 0.0004][:num_assets]
    volatilities = [0.02, 0.015, 0.018][:num_assets]
    
    # Correlation matrix
    correlation = np.array([
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.5],
        [0.4, 0.5, 1.0]
    ])[:num_assets, :num_assets]
    
    # Covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
    
    asset_names = ['Equity_Portfolio', 'Fixed_Income', 'Commodities'][:num_assets]
    
    returns_df = pd.DataFrame(returns, columns=asset_names, index=dates)
    
    return returns_df


def generate_sample_portfolio():
    """
    Generate sample portfolio positions for shock analysis.
    
    Returns:
        pd.DataFrame: Sample portfolio
    """
    portfolio = [
        {'asset': 'US_Equities', 'asset_class': 'equity', 'position_size': 10000, 'current_price': 150.0},
        {'asset': 'EU_Equities', 'asset_class': 'equity', 'position_size': 8000, 'current_price': 120.0},
        {'asset': 'Treasury_Bonds', 'asset_class': 'bonds', 'position_size': 5000, 'current_price': 95.0},
        {'asset': 'Corporate_Bonds', 'asset_class': 'bonds', 'position_size': 6000, 'current_price': 98.0},
        {'asset': 'Gold', 'asset_class': 'commodities', 'position_size': 500, 'current_price': 1800.0}
    ]
    
    return pd.DataFrame(portfolio)


def define_shock_scenarios():
    """
    Define standard shock scenarios for stress testing.
    
    Returns:
        dict: Dictionary of shock scenarios
    """
    scenarios = {
        'equity_crash': {
            'equity': -0.30,
            'bonds': 0.05,
            'commodities': -0.15
        },
        'rates_spike': {
            'equity': -0.10,
            'bonds': -0.15,
            'commodities': 0.02
        },
        'flight_to_quality': {
            'equity': -0.20,
            'bonds': 0.10,
            'commodities': -0.10
        },
        'stagflation': {
            'equity': -0.15,
            'bonds': -0.08,
            'commodities': 0.25
        },
        'market_turmoil': {
            'equity': -0.25,
            'bonds': -0.10,
            'commodities': -0.20
        }
    }
    
    return scenarios

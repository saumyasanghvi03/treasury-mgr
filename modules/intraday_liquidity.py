"""
Intraday Liquidity Monitoring Module

This module monitors intraday liquidity positions from timestamped payment data,
tracking real-time cash positions and liquidity coverage throughout the day.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta


class IntradayLiquidityMonitor:
    """
    Monitor and analyze intraday liquidity positions from timestamped payments.
    
    Tracks cash positions throughout the trading day to ensure adequate
    liquidity for settlement obligations and regulatory compliance.
    """
    
    def __init__(self, opening_balance=1_000_000_000):
        """
        Initialize the intraday liquidity monitor.
        
        Args:
            opening_balance (float): Opening cash balance for the day
        """
        self.opening_balance = opening_balance
        self.intraday_data = None
        self.peak_usage = None
        self.minimum_balance = None
        
    def process_payment_data(self, payments_df):
        """
        Process timestamped payment data to calculate running balances.
        
        Args:
            payments_df (pd.DataFrame): Payment data with columns
                ['timestamp', 'payment_type', 'amount', 'counterparty']
                payment_type: 'INFLOW' or 'OUTFLOW'
                
        Returns:
            pd.DataFrame: Processed intraday liquidity data
        """
        df = payments_df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply sign convention
        df['net_amount'] = df.apply(
            lambda row: row['amount'] if row['payment_type'] == 'INFLOW' else -row['amount'],
            axis=1
        )
        
        # Calculate running balance
        df['balance'] = self.opening_balance + df['net_amount'].cumsum()
        
        # Extract time components for analysis
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['time_of_day'] = df['timestamp'].dt.time
        
        # Calculate time since start
        start_time = df['timestamp'].min()
        df['minutes_elapsed'] = (df['timestamp'] - start_time).dt.total_seconds() / 60
        
        # Identify peaks and troughs
        df['is_peak'] = df['balance'] == df['balance'].max()
        df['is_trough'] = df['balance'] == df['balance'].min()
        
        self.intraday_data = df
        self.peak_usage = df['balance'].max()
        self.minimum_balance = df['balance'].min()
        
        return df
    
    def calculate_liquidity_metrics(self):
        """
        Calculate key intraday liquidity metrics.
        
        Returns:
            dict: Intraday liquidity metrics
        """
        if self.intraday_data is None:
            raise ValueError("Payment data must be processed first.")
        
        df = self.intraday_data
        
        # Basic metrics
        total_inflows = df[df['payment_type'] == 'INFLOW']['amount'].sum()
        total_outflows = df[df['payment_type'] == 'OUTFLOW']['amount'].sum()
        net_flow = total_inflows - total_outflows
        closing_balance = self.opening_balance + net_flow
        
        # Intraday metrics
        peak_balance = df['balance'].max()
        minimum_balance = df['balance'].min()
        intraday_range = peak_balance - minimum_balance
        
        # Liquidity usage
        max_liquidity_needed = self.opening_balance - minimum_balance
        liquidity_utilization_rate = (max_liquidity_needed / self.opening_balance * 100) if self.opening_balance > 0 else 0
        
        # Timing analysis
        peak_time = df.loc[df['balance'].idxmax(), 'timestamp']
        trough_time = df.loc[df['balance'].idxmin(), 'timestamp']
        
        # Volatility
        balance_volatility = df['balance'].std()
        
        # Count of transactions
        num_inflows = len(df[df['payment_type'] == 'INFLOW'])
        num_outflows = len(df[df['payment_type'] == 'OUTFLOW'])
        
        metrics = {
            'opening_balance': self.opening_balance,
            'closing_balance': closing_balance,
            'total_inflows': total_inflows,
            'total_outflows': total_outflows,
            'net_flow': net_flow,
            'peak_balance': peak_balance,
            'minimum_balance': minimum_balance,
            'intraday_range': intraday_range,
            'max_liquidity_needed': max_liquidity_needed,
            'liquidity_utilization_rate': liquidity_utilization_rate,
            'balance_volatility': balance_volatility,
            'peak_time': peak_time,
            'trough_time': trough_time,
            'num_inflows': num_inflows,
            'num_outflows': num_outflows,
            'total_transactions': len(df)
        }
        
        return metrics
    
    def identify_liquidity_stress_periods(self, threshold_pct=0.80):
        """
        Identify time periods where liquidity utilization exceeds threshold.
        
        Args:
            threshold_pct (float): Threshold percentage of opening balance (0-1)
            
        Returns:
            pd.DataFrame: Stress periods with timestamps
        """
        if self.intraday_data is None:
            raise ValueError("Payment data must be processed first.")
        
        threshold_balance = self.opening_balance * threshold_pct
        
        # Find periods below threshold
        stress_df = self.intraday_data[self.intraday_data['balance'] < threshold_balance].copy()
        
        if len(stress_df) > 0:
            stress_df['utilization_rate'] = (
                (self.opening_balance - stress_df['balance']) / self.opening_balance * 100
            )
            stress_df['shortfall'] = threshold_balance - stress_df['balance']
        
        return stress_df
    
    def aggregate_by_time_bucket(self, bucket_minutes=30):
        """
        Aggregate intraday activity by time buckets.
        
        Args:
            bucket_minutes (int): Size of time bucket in minutes
            
        Returns:
            pd.DataFrame: Aggregated data by time bucket
        """
        if self.intraday_data is None:
            raise ValueError("Payment data must be processed first.")
        
        df = self.intraday_data.copy()
        
        # Create time buckets
        df['time_bucket'] = (df['minutes_elapsed'] // bucket_minutes) * bucket_minutes
        
        # Aggregate by time bucket
        bucket_agg = df.groupby('time_bucket').agg({
            'net_amount': 'sum',
            'balance': 'last',  # Ending balance for bucket
            'amount': 'count',
            'timestamp': 'last'
        }).reset_index()
        
        bucket_agg.rename(columns={
            'net_amount': 'net_flow',
            'amount': 'transaction_count',
            'timestamp': 'bucket_end_time'
        }, inplace=True)
        
        # Calculate average balance for bucket
        bucket_agg['avg_balance'] = df.groupby('time_bucket')['balance'].mean().values
        
        return bucket_agg
    
    def calculate_hourly_patterns(self):
        """
        Analyze hourly patterns in payment activity.
        
        Returns:
            pd.DataFrame: Hourly aggregated statistics
        """
        if self.intraday_data is None:
            raise ValueError("Payment data must be processed first.")
        
        hourly = self.intraday_data.groupby('hour').agg({
            'net_amount': ['sum', 'mean'],
            'amount': 'count',
            'balance': ['min', 'max', 'mean']
        }).reset_index()
        
        hourly.columns = ['hour', 'total_net_flow', 'avg_net_flow', 
                         'transaction_count', 'min_balance', 'max_balance', 'avg_balance']
        
        return hourly
    
    def forecast_intraday_requirements(self, historical_data, target_date=None):
        """
        Forecast intraday liquidity requirements based on historical patterns.
        
        Args:
            historical_data (list): List of historical intraday DataFrames
            target_date (datetime): Date to forecast (default: today)
            
        Returns:
            dict: Forecasted liquidity requirements
        """
        # Calculate historical statistics
        min_balances = [df['balance'].min() for df in historical_data]
        max_liquidity_needs = [
            self.opening_balance - df['balance'].min() for df in historical_data
        ]
        
        # Statistical forecast
        avg_min_balance = np.mean(min_balances)
        std_min_balance = np.std(min_balances)
        
        # Conservative estimate (95th percentile)
        forecast_min_balance = avg_min_balance - (1.65 * std_min_balance)
        forecast_liquidity_need = self.opening_balance - forecast_min_balance
        
        # Recommended buffer
        recommended_opening_balance = forecast_liquidity_need * 1.2  # 20% buffer
        
        forecast = {
            'forecast_date': target_date or datetime.now().date(),
            'avg_min_balance': avg_min_balance,
            'std_min_balance': std_min_balance,
            'forecast_min_balance': forecast_min_balance,
            'forecast_liquidity_need': forecast_liquidity_need,
            'recommended_opening_balance': recommended_opening_balance,
            'confidence_level': 0.95,
            'historical_periods': len(historical_data)
        }
        
        return forecast
    
    def generate_liquidity_alerts(self, warning_threshold=0.70, critical_threshold=0.85):
        """
        Generate liquidity alerts based on utilization thresholds.
        
        Args:
            warning_threshold (float): Warning level (% of opening balance used)
            critical_threshold (float): Critical level (% of opening balance used)
            
        Returns:
            list: List of alert dictionaries
        """
        if self.intraday_data is None:
            raise ValueError("Payment data must be processed first.")
        
        alerts = []
        df = self.intraday_data
        
        warning_balance = self.opening_balance * (1 - warning_threshold)
        critical_balance = self.opening_balance * (1 - critical_threshold)
        
        # Check for warning level breaches
        warning_breaches = df[df['balance'] < warning_balance]
        if len(warning_breaches) > 0:
            alerts.append({
                'level': 'WARNING',
                'message': f'Liquidity utilization exceeded {warning_threshold*100}%',
                'timestamp': warning_breaches.iloc[0]['timestamp'],
                'balance': warning_breaches.iloc[0]['balance'],
                'utilization': ((self.opening_balance - warning_breaches.iloc[0]['balance']) / 
                              self.opening_balance * 100)
            })
        
        # Check for critical level breaches
        critical_breaches = df[df['balance'] < critical_balance]
        if len(critical_breaches) > 0:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'Liquidity utilization exceeded {critical_threshold*100}%',
                'timestamp': critical_breaches.iloc[0]['timestamp'],
                'balance': critical_breaches.iloc[0]['balance'],
                'utilization': ((self.opening_balance - critical_breaches.iloc[0]['balance']) / 
                              self.opening_balance * 100)
            })
        
        # Check minimum balance
        if self.minimum_balance < 0:
            alerts.append({
                'level': 'CRITICAL',
                'message': 'Negative balance detected - insufficient liquidity',
                'timestamp': df.loc[df['balance'].idxmin(), 'timestamp'],
                'balance': self.minimum_balance,
                'shortfall': abs(self.minimum_balance)
            })
        
        return alerts


def generate_sample_intraday_payments(date=None, opening_balance=1_000_000_000):
    """
    Generate sample intraday payment data for demonstration.
    
    Args:
        date (datetime): Date for the payments (default: today)
        opening_balance (float): Opening balance
        
    Returns:
        pd.DataFrame: Sample intraday payment data
    """
    if date is None:
        date = datetime.now().date()
    
    np.random.seed(42)
    
    # Generate timestamps throughout the trading day (9 AM - 5 PM)
    start_time = datetime.combine(date, time(9, 0))
    end_time = datetime.combine(date, time(17, 0))
    
    # Generate 200-400 payments throughout the day
    num_payments = np.random.randint(200, 401)
    
    # Random timestamps
    time_range = (end_time - start_time).total_seconds()
    random_seconds = np.random.uniform(0, time_range, num_payments)
    timestamps = [start_time + timedelta(seconds=s) for s in sorted(random_seconds)]
    
    payments = []
    
    for timestamp in timestamps:
        # Payment patterns vary by time of day
        hour = timestamp.hour
        
        # More outflows in morning, more inflows in afternoon
        if hour < 12:
            payment_type = np.random.choice(['INFLOW', 'OUTFLOW'], p=[0.35, 0.65])
        else:
            payment_type = np.random.choice(['INFLOW', 'OUTFLOW'], p=[0.55, 0.45])
        
        # Amount distribution (log-normal)
        amount = np.random.lognormal(mean=14, sigma=1.5)
        
        payments.append({
            'timestamp': timestamp,
            'payment_type': payment_type,
            'amount': round(amount, 2),
            'counterparty': f'CP_{np.random.randint(1000, 9999)}',
            'payment_system': np.random.choice(['FEDWIRE', 'CHIPS', 'ACH'])
        })
    
    df = pd.DataFrame(payments)
    
    return df


def save_intraday_sample_data(output_path, num_days=5):
    """
    Generate and save multiple days of intraday payment data.
    
    Args:
        output_path (str): Path to save the CSV file
        num_days (int): Number of days to generate
        
    Returns:
        pd.DataFrame: Combined sample data
    """
    all_payments = []
    
    start_date = datetime.now().date() - timedelta(days=num_days-1)
    
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        daily_payments = generate_sample_intraday_payments(date=date)
        daily_payments['date'] = date
        all_payments.append(daily_payments)
    
    combined_df = pd.concat(all_payments, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    
    return combined_df

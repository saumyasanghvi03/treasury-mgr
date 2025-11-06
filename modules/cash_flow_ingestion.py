"""
Cash Flow Ingestion and Normalization Module

This module handles the ingestion of cash flow data from various sources
and normalizes it into a standardized format for downstream analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_transaction_data(file_path):
    """
    Load transaction data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing transaction data
        
    Returns:
        pd.DataFrame: Raw transaction data
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        return df
    except Exception as e:
        raise ValueError(f"Error loading transaction data: {str(e)}")


def normalize_cash_flows(df):
    """
    Normalize cash flow data into standardized format.
    
    Standardization includes:
    - Date format conversion
    - Category standardization
    - Currency normalization
    - Sign convention (inflows positive, outflows negative)
    
    Args:
        df (pd.DataFrame): Raw transaction dataframe
        
    Returns:
        pd.DataFrame: Normalized cash flow dataframe
    """
    normalized_df = df.copy()
    
    # Ensure date column is datetime
    if 'date' in normalized_df.columns:
        normalized_df['date'] = pd.to_datetime(normalized_df['date'])
    
    # Standardize transaction types
    if 'type' in normalized_df.columns:
        normalized_df['type'] = normalized_df['type'].str.upper()
    
    # Apply sign convention: inflows positive, outflows negative
    if 'amount' in normalized_df.columns and 'type' in normalized_df.columns:
        normalized_df['net_amount'] = normalized_df.apply(
            lambda row: row['amount'] if row['type'] == 'INFLOW' else -row['amount'],
            axis=1
        )
    
    # Add period classification
    normalized_df['year_month'] = normalized_df['date'].dt.to_period('M')
    normalized_df['quarter'] = normalized_df['date'].dt.quarter
    normalized_df['year'] = normalized_df['date'].dt.year
    
    return normalized_df


def aggregate_daily_cash_flows(df):
    """
    Aggregate cash flows by date to get daily net cash positions.
    
    Args:
        df (pd.DataFrame): Normalized cash flow dataframe
        
    Returns:
        pd.DataFrame: Daily aggregated cash flows
    """
    if 'date' not in df.columns or 'net_amount' not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'net_amount' columns")
    
    daily_cf = df.groupby('date').agg({
        'net_amount': 'sum',
        'amount': 'sum'
    }).reset_index()
    
    daily_cf.rename(columns={
        'net_amount': 'net_cash_flow',
        'amount': 'gross_amount'
    }, inplace=True)
    
    # Calculate cumulative cash position
    daily_cf['cumulative_position'] = daily_cf['net_cash_flow'].cumsum()
    
    return daily_cf


def categorize_cash_flows(df):
    """
    Categorize cash flows by type and business line.
    
    Args:
        df (pd.DataFrame): Normalized cash flow dataframe
        
    Returns:
        pd.DataFrame: Cash flows with category breakdown
    """
    if 'category' not in df.columns:
        # If no category, create a default one
        df['category'] = 'GENERAL'
    
    category_summary = df.groupby(['date', 'category']).agg({
        'net_amount': 'sum',
        'amount': ['sum', 'count']
    }).reset_index()
    
    category_summary.columns = ['date', 'category', 'net_amount', 'gross_amount', 'transaction_count']
    
    return category_summary


def generate_sample_transaction_data(output_path, num_days=365):
    """
    Generate sample transaction data for demonstration purposes.
    
    Args:
        output_path (str): Path to save the generated CSV file
        num_days (int): Number of days of historical data to generate
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=num_days)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    categories = ['OPERATIONS', 'INVESTMENTS', 'DEBT_SERVICE', 'TREASURY_OPS', 'WHOLESALE']
    types = ['INFLOW', 'OUTFLOW']
    
    records = []
    
    for date in dates:
        # Generate 5-15 transactions per day
        num_transactions = np.random.randint(5, 16)
        
        for _ in range(num_transactions):
            transaction = {
                'date': date.strftime('%Y-%m-%d'),
                'category': np.random.choice(categories),
                'type': np.random.choice(types, p=[0.48, 0.52]),  # Slightly more outflows
                'amount': np.random.lognormal(mean=11, sigma=1.5),  # Log-normal distribution
                'counterparty': f"CP_{np.random.randint(1000, 9999)}",
                'currency': 'USD'
            }
            records.append(transaction)
    
    df = pd.DataFrame(records)
    df['amount'] = df['amount'].round(2)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df


def validate_cash_flow_data(df):
    """
    Validate cash flow data for completeness and consistency.
    
    Args:
        df (pd.DataFrame): Cash flow dataframe to validate
        
    Returns:
        dict: Validation results with any issues found
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }
    
    # Check required columns
    required_columns = ['date', 'amount', 'type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Missing required columns: {missing_columns}")
    
    # Check for null values
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        validation_results['warnings'].append(f"Null values found in columns: {null_cols}")
    
    # Check for negative amounts
    if 'amount' in df.columns and (df['amount'] < 0).any():
        validation_results['warnings'].append("Negative amounts found. Please verify data.")
    
    # Check date range
    if 'date' in df.columns:
        date_range = (df['date'].max() - df['date'].min()).days
        validation_results['date_range_days'] = date_range
    
    return validation_results

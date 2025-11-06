"""
Example: Extending the Treasury Management System with Custom Data

This example shows how to:
1. Load your own data into the system
2. Customize calculations and models
3. Add new features or modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example 1: Loading Custom Historical Cash Flow Data
def load_custom_cashflow_data(csv_file_path):
    """
    Load historical cash flow data from a CSV file
    
    Expected CSV format:
    date,amount,inflows,outflows
    2024-01-01,5000000,3000000,2000000
    2024-01-02,5100000,3200000,2100000
    ...
    """
    df = pd.read_csv(csv_file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Validate data
    assert all(col in df.columns for col in ['date', 'amount', 'inflows', 'outflows']), \
        "CSV must contain: date, amount, inflows, outflows"
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

# Example 2: Custom Portfolio Data
def create_custom_portfolio(positions_dict):
    """
    Create portfolio DataFrame from a dictionary of positions
    
    Example:
    positions = {
        'US Treasury 10Y': {'value': 50_000_000, 'volatility': 0.05},
        'Corporate Bonds': {'value': 30_000_000, 'volatility': 0.07},
        'Money Market': {'value': 20_000_000, 'volatility': 0.01}
    }
    """
    portfolio_data = []
    total_value = sum(p['value'] for p in positions_dict.values())
    
    for asset, data in positions_dict.items():
        portfolio_data.append({
            'asset': asset,
            'value': data['value'],
            'weight': data['value'] / total_value,
            'volatility': data['volatility']
        })
    
    return pd.DataFrame(portfolio_data)

# Example 3: Custom ALM Gap Data
def create_alm_gap_from_balance_sheet(assets_by_maturity, liabilities_by_maturity):
    """
    Create ALM gap analysis from balance sheet maturity buckets
    
    Example:
    assets = {
        '0-1M': 25_000_000,
        '1-3M': 30_000_000,
        '3-6M': 35_000_000,
        ...
    }
    """
    buckets = list(assets_by_maturity.keys())
    
    alm_data = []
    for bucket in buckets:
        assets = assets_by_maturity.get(bucket, 0)
        liabilities = liabilities_by_maturity.get(bucket, 0)
        
        alm_data.append({
            'bucket': bucket,
            'assets': assets / 1_000_000,  # Convert to millions
            'liabilities': liabilities / 1_000_000,
            'gap': (assets - liabilities) / 1_000_000
        })
    
    return pd.DataFrame(alm_data)

# Example 4: Custom LCR Calculation
def calculate_custom_lcr(hqla_components, outflow_components, inflow_components):
    """
    Calculate LCR with custom HQLA and cash flow components
    
    Example:
    hqla = {
        'cash': 50_000_000,
        'central_bank_reserves': 30_000_000,
        'level1_securities': 20_000_000,
        'level2a_securities': 15_000_000,
        'level2b_securities': 10_000_000
    }
    """
    # Apply haircuts
    total_hqla = (
        hqla_components.get('cash', 0) +
        hqla_components.get('central_bank_reserves', 0) +
        hqla_components.get('level1_securities', 0) +
        hqla_components.get('level2a_securities', 0) * 0.85 +  # 15% haircut
        hqla_components.get('level2b_securities', 0) * 0.50    # 50% haircut
    )
    
    # Calculate stressed outflows
    total_outflows = sum(outflow_components.values())
    
    # Calculate stressed inflows (capped at 75% of outflows)
    total_inflows = min(sum(inflow_components.values()), total_outflows * 0.75)
    
    # Net cash outflows
    net_outflows = total_outflows - total_inflows
    
    # LCR ratio
    lcr_ratio = (total_hqla / net_outflows * 100) if net_outflows > 0 else float('inf')
    
    return {
        'total_hqla': total_hqla,
        'total_outflows': total_outflows,
        'total_inflows': total_inflows,
        'net_outflows': net_outflows,
        'lcr_ratio': lcr_ratio
    }

# Example 5: Integrating Custom Data into Streamlit App
def integrate_custom_data_example():
    """
    Example of how to modify app.py to use custom data
    
    In app.py, replace the data generation section with:
    """
    example_code = """
    # In app.py, modify the data loading section:
    
    if 'data' not in st.session_state:
        # Option 1: Load from files
        if st.sidebar.checkbox("Load custom data"):
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                st.session_state.data = load_custom_cashflow_data(uploaded_file)
        else:
            # Option 2: Use sample data
            st.session_state.data = data_generator.generate_sample_data()
    """
    
    return example_code

# Example 6: Custom VaR Calculation with Real Returns
def calculate_var_with_real_data(returns_csv_path, portfolio_value, confidence_level=95):
    """
    Calculate VaR using real historical returns data
    
    CSV format:
    date,asset1_return,asset2_return,asset3_return,...
    2024-01-01,0.001,-0.002,0.0015,...
    """
    # Load returns
    returns_df = pd.read_csv(returns_csv_path)
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    
    # Get return columns (exclude date)
    return_cols = [col for col in returns_df.columns if col != 'date']
    
    # Calculate portfolio returns (assuming equal weights)
    portfolio_returns = returns_df[return_cols].mean(axis=1)
    
    # Calculate VaR
    var_percentile = 100 - confidence_level
    var_value = abs(np.percentile(portfolio_returns, var_percentile) * portfolio_value)
    
    return var_value

# Example 7: Custom Investment Universe
def create_custom_investment_universe(instruments_data):
    """
    Create investment universe for optimizer
    
    Example:
    instruments = [
        {
            'instrument': 'US Treasury 2Y',
            'expected_return': 0.045,
            'risk': 0.02,
            'liquidity_score': 100,
            'min_investment': 1_000_000,
            'max_investment': 50_000_000
        },
        ...
    ]
    """
    return pd.DataFrame(instruments_data)

# Example 8: Running Custom Analysis
def custom_analysis_example():
    """Complete example of running custom analysis"""
    
    print("Custom Treasury Analysis Example\n")
    
    # 1. Create custom portfolio
    positions = {
        'Government Bonds': {'value': 60_000_000, 'volatility': 0.04},
        'Corporate Bonds': {'value': 30_000_000, 'volatility': 0.08},
        'Money Market': {'value': 10_000_000, 'volatility': 0.005}
    }
    
    portfolio = create_custom_portfolio(positions)
    print("Custom Portfolio:")
    print(portfolio)
    print()
    
    # 2. Create ALM gap analysis
    assets_by_maturity = {
        '0-1M': 20_000_000,
        '1-3M': 25_000_000,
        '3-6M': 30_000_000,
        '6-12M': 25_000_000
    }
    
    liabilities_by_maturity = {
        '0-1M': 30_000_000,
        '1-3M': 28_000_000,
        '3-6M': 22_000_000,
        '6-12M': 20_000_000
    }
    
    alm_gap = create_alm_gap_from_balance_sheet(assets_by_maturity, liabilities_by_maturity)
    print("ALM Gap Analysis:")
    print(alm_gap)
    print()
    
    # 3. Calculate custom LCR
    hqla = {
        'cash': 40_000_000,
        'central_bank_reserves': 30_000_000,
        'level1_securities': 25_000_000,
        'level2a_securities': 10_000_000
    }
    
    outflows = {
        'retail_deposits': 50_000_000,
        'wholesale_funding': 30_000_000
    }
    
    inflows = {
        'maturing_assets': 15_000_000,
        'committed_facilities': 5_000_000
    }
    
    lcr_result = calculate_custom_lcr(hqla, outflows, inflows)
    print("LCR Calculation:")
    print(f"Total HQLA: ${lcr_result['total_hqla']/1e6:.1f}M")
    print(f"Net Outflows: ${lcr_result['net_outflows']/1e6:.1f}M")
    print(f"LCR Ratio: {lcr_result['lcr_ratio']:.1f}%")
    print()
    
    # 4. Create custom investment universe
    instruments = [
        {
            'instrument': 'Treasury Bond 5Y',
            'expected_return': 0.042,
            'risk': 0.04,
            'liquidity_score': 100,
            'min_investment': 5_000_000,
            'max_investment': 40_000_000
        },
        {
            'instrument': 'Corporate Bond AA',
            'expected_return': 0.055,
            'risk': 0.06,
            'liquidity_score': 80,
            'min_investment': 1_000_000,
            'max_investment': 25_000_000
        },
        {
            'instrument': 'Money Market Fund',
            'expected_return': 0.035,
            'risk': 0.01,
            'liquidity_score': 100,
            'min_investment': 1_000_000,
            'max_investment': 30_000_000
        }
    ]
    
    universe = create_custom_investment_universe(instruments)
    print("Investment Universe:")
    print(universe[['instrument', 'expected_return', 'risk', 'liquidity_score']])
    print()

# Example 9: Database Integration
def load_data_from_database_example():
    """
    Example of loading data from a database
    
    This shows the structure - adapt to your database
    """
    example_code = """
    import psycopg2  # or your database driver
    
    def load_portfolio_from_db(connection_string):
        conn = psycopg2.connect(connection_string)
        
        query = '''
        SELECT 
            instrument_name as asset,
            market_value as value,
            weight,
            volatility
        FROM portfolio_positions
        WHERE as_of_date = CURRENT_DATE
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def load_cashflows_from_db(connection_string, start_date, end_date):
        conn = psycopg2.connect(connection_string)
        
        query = '''
        SELECT 
            transaction_date as date,
            SUM(amount) as amount,
            SUM(CASE WHEN direction = 'IN' THEN amount ELSE 0 END) as inflows,
            SUM(CASE WHEN direction = 'OUT' THEN amount ELSE 0 END) as outflows
        FROM cash_flows
        WHERE transaction_date BETWEEN %s AND %s
        GROUP BY transaction_date
        ORDER BY transaction_date
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        return df
    """
    
    return example_code

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("Treasury Management System - Custom Data Integration Examples")
    print("="*70)
    print()
    
    custom_analysis_example()
    
    print("="*70)
    print("For more examples, see the individual function docstrings above.")
    print("To integrate with your Streamlit app, modify app.py accordingly.")
    print("="*70)

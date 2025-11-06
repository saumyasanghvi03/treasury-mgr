import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

def show():
    st.header("📉 Market Risk - Value at Risk (VaR) Modeling")
    st.write("Quantify potential portfolio losses using multiple VaR methodologies")
    
    # Get data
    data = st.session_state.data
    portfolio_df = data['portfolio'].copy()
    returns_df = data['historical_returns'].copy()
    
    # Overview
    st.subheader("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    portfolio_value = portfolio_df['value'].sum()
    
    with col1:
        st.metric("Total Portfolio Value", f"${portfolio_value/1e6:.1f}M")
    
    with col2:
        weighted_vol = (portfolio_df['weight'] * portfolio_df['volatility']).sum()
        st.metric("Weighted Avg Volatility", f"{weighted_vol*100:.2f}%")
    
    with col3:
        var_95 = data['var_95']
        st.metric("VaR (95%)", f"${var_95/1e6:.2f}M")
    
    with col4:
        var_pct = (var_95 / portfolio_value) * 100
        st.metric("VaR as % of Portfolio", f"{var_pct:.2f}%")
    
    # Portfolio composition
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Portfolio Composition")
        
        fig = go.Figure(data=[go.Pie(
            labels=portfolio_df['asset'],
            values=portfolio_df['value'],
            hole=0.4,
            textinfo='label+percent'
        )])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Asset Details")
        
        display_df = portfolio_df.copy()
        st.dataframe(
            display_df.style.format({
                'value': '${:,.0f}',
                'weight': '{:.1%}',
                'volatility': '{:.1%}'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    # VaR Calculation Methods
    st.divider()
    st.subheader("VaR Calculation Methods")
    
    # Sidebar parameters
    st.sidebar.subheader("VaR Parameters")
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        [90, 95, 99],
        index=1
    )
    time_horizon = st.sidebar.selectbox(
        "Time Horizon (days)",
        [1, 5, 10],
        index=0
    )
    
    # Calculate VaR using different methods
    var_results = calculate_var(returns_df, portfolio_df, confidence_level, time_horizon)
    
    # Display VaR results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Historical VaR",
            f"${var_results['historical']/1e6:.2f}M",
            help="Based on historical return distribution"
        )
    
    with col2:
        st.metric(
            "Parametric VaR",
            f"${var_results['parametric']/1e6:.2f}M",
            help="Assumes normal distribution"
        )
    
    with col3:
        st.metric(
            "Monte Carlo VaR",
            f"${var_results['monte_carlo']/1e6:.2f}M",
            help="Simulation-based approach"
        )
    
    # VaR Comparison
    st.write("**VaR Method Comparison**")
    
    var_comparison = pd.DataFrame({
        'Method': ['Historical', 'Parametric', 'Monte Carlo'],
        'VaR ($M)': [
            var_results['historical']/1e6,
            var_results['parametric']/1e6,
            var_results['monte_carlo']/1e6
        ],
        'VaR (%)': [
            (var_results['historical']/portfolio_value)*100,
            (var_results['parametric']/portfolio_value)*100,
            (var_results['monte_carlo']/portfolio_value)*100
        ]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=var_comparison['Method'],
        y=var_comparison['VaR ($M)'],
        marker_color=['#3498db', '#e74c3c', '#2ecc71'],
        text=var_comparison['VaR ($M)'].apply(lambda x: f"${x:.2f}M"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"VaR at {confidence_level}% Confidence ({time_horizon}-day)",
        yaxis_title="VaR ($M)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical Simulation Details
    st.divider()
    st.subheader("Historical Simulation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate portfolio returns
        portfolio_returns = calculate_portfolio_returns(returns_df, portfolio_df)
        
        # Distribution plot
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=portfolio_returns * 100,
            nbinsx=50,
            name='Returns Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add VaR line
        var_threshold = np.percentile(portfolio_returns, 100 - confidence_level)
        
        fig.add_vline(
            x=var_threshold * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR {confidence_level}%",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="Portfolio Returns Distribution",
            xaxis_title="Daily Returns (%)",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Distribution Statistics**")
        
        stats_df = pd.DataFrame({
            'Metric': [
                'Mean',
                'Std Dev',
                'Skewness',
                'Kurtosis',
                'Min',
                'Max',
                f'{confidence_level}% VaR'
            ],
            'Value': [
                f"{portfolio_returns.mean()*100:.4f}%",
                f"{portfolio_returns.std()*100:.4f}%",
                f"{stats.skew(portfolio_returns):.4f}",
                f"{stats.kurtosis(portfolio_returns):.4f}",
                f"{portfolio_returns.min()*100:.4f}%",
                f"{portfolio_returns.max()*100:.4f}%",
                f"{var_threshold*100:.4f}%"
            ]
        })
        
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    # Conditional VaR (Expected Shortfall)
    st.divider()
    st.subheader("Conditional VaR (Expected Shortfall)")
    
    cvar = calculate_cvar(portfolio_returns, portfolio_value, confidence_level)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CVaR / Expected Shortfall", f"${cvar/1e6:.2f}M")
    
    with col2:
        cvar_pct = (cvar / portfolio_value) * 100
        st.metric("CVaR as % of Portfolio", f"{cvar_pct:.2f}%")
    
    with col3:
        excess = cvar - var_results['historical']
        st.metric("Excess over VaR", f"${excess/1e6:.2f}M")
    
    st.info("""
    **Conditional VaR (CVaR)** or **Expected Shortfall** represents the average loss 
    in the worst (100 - confidence_level)% of cases. It provides additional insight 
    into tail risk beyond standard VaR.
    """)
    
    # Stress Testing
    st.divider()
    st.subheader("Stress Testing & Scenario Analysis")
    
    tab1, tab2 = st.tabs(["Market Shocks", "Component VaR"])
    
    with tab1:
        st.write("Impact of hypothetical market shocks on portfolio value")
        
        scenarios = pd.DataFrame({
            'Scenario': [
                'Market Crash (-20%)',
                'Moderate Decline (-10%)',
                'Minor Correction (-5%)',
                'Volatility Spike (+50% vol)',
                'Interest Rate Shock (+200bps)'
            ],
            'Description': [
                'Severe equity market decline',
                'Moderate market correction',
                'Minor market pullback',
                'Significant increase in volatility',
                'Sharp interest rate increase'
            ],
            'Impact ($M)': [
                -portfolio_value * 0.20 / 1e6,
                -portfolio_value * 0.10 / 1e6,
                -portfolio_value * 0.05 / 1e6,
                -portfolio_value * 0.08 / 1e6,
                -portfolio_value * 0.12 / 1e6
            ]
        })
        
        scenarios['Impact (%)'] = (scenarios['Impact ($M)'] / (portfolio_value/1e6)) * 100
        
        fig = go.Figure()
        
        colors = ['darkred', 'red', 'orange', 'coral', 'lightcoral']
        
        fig.add_trace(go.Bar(
            y=scenarios['Scenario'],
            x=scenarios['Impact ($M)'],
            orientation='h',
            marker_color=colors,
            text=scenarios['Impact ($M)'].apply(lambda x: f"${x:.1f}M"),
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Stress Test Scenario Impacts",
            xaxis_title="Impact on Portfolio Value ($M)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            scenarios[['Scenario', 'Description', 'Impact ($M)', 'Impact (%)']].style.format({
                'Impact ($M)': '${:.2f}M',
                'Impact (%)': '{:.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with tab2:
        st.write("Contribution of each asset to total portfolio VaR")
        
        component_var = calculate_component_var(returns_df, portfolio_df, confidence_level)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=component_var['asset'],
                y=component_var['contribution'],
                marker_color='steelblue',
                text=component_var['contribution'].apply(lambda x: f"${x/1e6:.2f}M"),
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Component VaR by Asset",
                xaxis_title="Asset",
                yaxis_title="VaR Contribution ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                component_var.style.format({
                    'contribution': '${:,.0f}',
                    'contribution_pct': '{:.2f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    # Risk Metrics & Recommendations
    st.divider()
    st.subheader("Risk Assessment & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk rating
        var_pct = (var_results['historical'] / portfolio_value) * 100
        
        if var_pct < 1:
            st.success("✓ **Low Risk**: VaR < 1% of portfolio")
        elif var_pct < 3:
            st.warning("⚠ **Moderate Risk**: VaR 1-3% of portfolio")
        else:
            st.error("✗ **High Risk**: VaR > 3% of portfolio")
    
    with col2:
        st.info("""
        **Monitoring Actions:**
        - Daily VaR calculation
        - Backtesting VaR model
        - Limit monitoring
        - Breach reporting
        """)
    
    with col3:
        st.info("""
        **Risk Mitigation:**
        - Diversification
        - Hedging strategies
        - Position limits
        - Stop-loss orders
        """)

def calculate_var(returns_df, portfolio_df, confidence_level, time_horizon):
    """Calculate VaR using multiple methods"""
    portfolio_value = portfolio_df['value'].sum()
    portfolio_returns = calculate_portfolio_returns(returns_df, portfolio_df)
    
    # 1. Historical VaR
    var_percentile = 100 - confidence_level
    historical_var = np.percentile(portfolio_returns, var_percentile)
    historical_var_dollar = abs(historical_var * portfolio_value * np.sqrt(time_horizon))
    
    # 2. Parametric VaR (assumes normal distribution)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    z_score = stats.norm.ppf(1 - confidence_level/100)
    parametric_var_dollar = abs((mean_return + z_score * std_return) * portfolio_value * np.sqrt(time_horizon))
    
    # 3. Monte Carlo VaR
    np.random.seed(42)
    n_simulations = 10000
    simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
    mc_var = np.percentile(simulated_returns, var_percentile)
    mc_var_dollar = abs(mc_var * portfolio_value * np.sqrt(time_horizon))
    
    return {
        'historical': historical_var_dollar,
        'parametric': parametric_var_dollar,
        'monte_carlo': mc_var_dollar
    }

def calculate_portfolio_returns(returns_df, portfolio_df):
    """Calculate portfolio returns based on weights"""
    portfolio_returns = np.zeros(len(returns_df))
    
    for idx, row in portfolio_df.iterrows():
        asset = row['asset']
        weight = row['weight']
        if asset in returns_df.columns:
            portfolio_returns += returns_df[asset].values * weight
    
    return portfolio_returns

def calculate_cvar(portfolio_returns, portfolio_value, confidence_level):
    """Calculate Conditional VaR (Expected Shortfall)"""
    var_percentile = 100 - confidence_level
    var_threshold = np.percentile(portfolio_returns, var_percentile)
    
    # Average of returns below VaR threshold
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
    expected_shortfall = abs(tail_returns.mean() * portfolio_value)
    
    return expected_shortfall

def calculate_component_var(returns_df, portfolio_df, confidence_level):
    """Calculate component VaR for each asset"""
    portfolio_returns = calculate_portfolio_returns(returns_df, portfolio_df)
    
    var_percentile = 100 - confidence_level
    total_var = abs(np.percentile(portfolio_returns, var_percentile) * portfolio_df['value'].sum())
    
    component_vars = []
    
    for idx, row in portfolio_df.iterrows():
        asset = row['asset']
        weight = row['weight']
        
        # Simplified component VaR calculation
        # More sophisticated methods would use marginal VaR
        component = total_var * weight
        
        component_vars.append({
            'asset': asset,
            'contribution': component,
            'contribution_pct': weight * 100
        })
    
    return pd.DataFrame(component_vars)

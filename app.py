"""
Treasury Management Solution - Main Application

A comprehensive Streamlit-based treasury management solution providing:
- Cash flow analysis and forecasting
- Basel III LCR calculations
- ALM GAP modeling
- Market risk analytics
- Intraday liquidity monitoring
- Portfolio optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.cash_flow_ingestion import (
    load_transaction_data, normalize_cash_flows, aggregate_daily_cash_flows,
    categorize_cash_flows, validate_cash_flow_data
)
from modules.ml_forecasting import CashFlowForecaster, analyze_forecast_accuracy
from modules.basel_lcr import (
    BaselIIILCRCalculator, generate_sample_hqla_data, generate_sample_cashflows
)
from modules.alm_gap import (
    ALMGAPAnalyzer, generate_sample_alm_data, generate_duration_data
)
from modules.market_risk import (
    MarketRiskAnalyzer, generate_sample_returns_data, generate_sample_portfolio,
    define_shock_scenarios
)
from modules.intraday_liquidity import (
    IntradayLiquidityMonitor, generate_sample_intraday_payments
)
from modules.portfolio_optimizer import (
    PortfolioOptimizer, generate_sample_assets, define_sector_constraints
)


# Page configuration
st.set_page_config(
    page_title="Treasury Management Solution",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styles."""
    css_file = os.path.join(os.path.dirname(__file__), 'styles.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()


# Sidebar navigation
def sidebar_navigation():
    """Create sidebar navigation menu."""
    st.sidebar.title("💰 Treasury Management")
    st.sidebar.markdown("---")
    
    menu_options = {
        "🏠 Home": "home",
        "📊 Cash Flow Analysis": "cashflow",
        "🔮 ML Forecasting": "forecasting",
        "🏦 Basel III LCR": "lcr",
        "⚖️ ALM GAP Analysis": "alm",
        "📈 Market Risk": "market_risk",
        "⏰ Intraday Liquidity": "intraday",
        "💼 Portfolio Optimizer": "portfolio"
    }
    
    selection = st.sidebar.radio("Navigation", list(menu_options.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Treasury Management Solution**\n\n"
        "A comprehensive platform for enterprise treasury operations, "
        "risk management, and regulatory compliance."
    )
    
    return menu_options[selection]


# Home Page
def show_home():
    """Display home page with overview."""
    st.title("Treasury Management Solution")
    st.markdown("### Enterprise-Grade Treasury & Risk Management Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modules", "7", delta="Fully Integrated")
    with col2:
        st.metric("Data Sources", "Multiple", delta="Real-time Ready")
    with col3:
        st.metric("Compliance", "Basel III", delta="Compliant")
    
    st.markdown("---")
    
    # Feature overview
    st.markdown("## 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Cash Flow Management
        - Transaction data ingestion and normalization
        - Historical analysis and categorization
        - Daily and periodic aggregation
        
        ### 🔮 ML-Driven Forecasting
        - Random Forest regression models
        - Feature engineering and lag analysis
        - Confidence intervals and accuracy metrics
        
        ### 🏦 Basel III LCR Calculator
        - HQLA classification and haircuts
        - Liquidity stress testing
        - Sensitivity analysis
        
        ### ⚖️ ALM GAP Modeling
        - RSA/RSL bucket analysis
        - Duration gap calculations
        - Interest rate sensitivity
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Market Risk Analytics
        - Historical VaR calculation
        - Monte Carlo simulations
        - Shock scenario analysis
        
        ### ⏰ Intraday Liquidity Monitoring
        - Real-time position tracking
        - Payment flow analysis
        - Liquidity alerts and thresholds
        
        ### 💼 Portfolio Optimization
        - Linear programming optimization
        - Efficient frontier analysis
        - Sector and constraint management
        
        ### 🔐 Risk & Compliance
        - Regulatory reporting
        - Stress testing frameworks
        - Audit trail capabilities
        """)
    
    st.markdown("---")
    st.info("👈 Select a module from the sidebar to get started")


# Cash Flow Analysis Page
def show_cashflow_analysis():
    """Display cash flow analysis page."""
    st.title("📊 Cash Flow Analysis")
    
    # File upload or use sample data
    use_sample = st.checkbox("Use sample data", value=True)
    
    if use_sample:
        df = load_transaction_data('data/sample/transactions.csv')
        st.success("✅ Sample data loaded successfully!")
    else:
        uploaded_file = st.file_uploader("Upload transaction CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, parse_dates=['date'])
        else:
            st.info("Please upload a CSV file or use sample data")
            return
    
    # Normalize cash flows
    normalized_df = normalize_cash_flows(df)
    daily_cf = aggregate_daily_cash_flows(normalized_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_inflows = normalized_df[normalized_df['type'] == 'INFLOW']['amount'].sum()
    total_outflows = normalized_df[normalized_df['type'] == 'OUTFLOW']['amount'].sum()
    net_flow = total_inflows - total_outflows
    
    with col1:
        st.metric("Total Inflows", f"${total_inflows:,.0f}")
    with col2:
        st.metric("Total Outflows", f"${total_outflows:,.0f}")
    with col3:
        st.metric("Net Cash Flow", f"${net_flow:,.0f}")
    with col4:
        st.metric("Transactions", f"{len(df):,}")
    
    st.markdown("---")
    
    # Daily cash flow chart
    st.subheader("Daily Cash Flow Trends")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_cf['date'],
        y=daily_cf['net_cash_flow'],
        mode='lines',
        name='Net Cash Flow',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_cf['date'],
        y=daily_cf['cumulative_position'],
        mode='lines',
        name='Cumulative Position',
        line=dict(color='green', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Daily Cash Flow Analysis',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cash Flow by Category")
        category_summary = normalized_df.groupby('category')['net_amount'].sum().reset_index()
        fig_cat = px.pie(
            category_summary,
            values='net_amount',
            names='category',
            title='Net Cash Flow Distribution'
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Volume by Type")
        type_counts = normalized_df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        fig_type = px.bar(
            type_counts,
            x='Type',
            y='Count',
            title='Transaction Volume',
            color='Type',
            color_discrete_map={'INFLOW': 'green', 'OUTFLOW': 'red'}
        )
        st.plotly_chart(fig_type, use_container_width=True)


# ML Forecasting Page
def show_ml_forecasting():
    """Display ML forecasting page."""
    st.title("🔮 ML-Driven Cash Flow Forecasting")
    
    # Load data
    df = load_transaction_data('data/sample/transactions.csv')
    normalized_df = normalize_cash_flows(df)
    daily_cf = aggregate_daily_cash_flows(normalized_df)
    
    # Forecasting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    with col2:
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
    with col3:
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
    
    if st.button("🚀 Train Model & Forecast", type="primary"):
        with st.spinner("Training model..."):
            # Initialize forecaster
            forecaster = CashFlowForecaster(n_estimators=n_estimators)
            
            # Train model
            metrics = forecaster.train(daily_cf, test_size=test_size)
            
            # Generate forecast
            forecast_df = forecaster.forecast(daily_cf, forecast_days=forecast_days)
            
            # Display metrics
            st.success("✅ Model trained successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"${metrics['mae']:,.0f}")
            with col2:
                st.metric("RMSE", f"${metrics['rmse']:,.0f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2_score']:.3f}")
            with col4:
                st.metric("CV MAE", f"${metrics['cv_mae_mean']:,.0f}")
            
            st.markdown("---")
            
            # Forecast chart
            st.subheader("Cash Flow Forecast")
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=daily_cf['date'][-90:],
                y=daily_cf['net_cash_flow'][-90:],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                mode='lines',
                name='Confidence Interval',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Cash Flow Forecast with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Net Cash Flow ($)',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Feature Importance")
                top_features = forecaster.get_feature_importance(top_n=10)
                fig_features = px.bar(
                    top_features,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Most Important Features for Prediction'
                )
                st.plotly_chart(fig_features, use_container_width=True)
            
            with col2:
                st.subheader("Forecast Statistics")
                st.dataframe(forecast_df[['date', 'forecast']].head(10), use_container_width=True)


# Basel III LCR Page
def show_basel_lcr():
    """Display Basel III LCR calculator page."""
    st.title("🏦 Basel III Liquidity Coverage Ratio")
    
    # Generate sample data
    hqla_df = generate_sample_hqla_data()
    outflows_df, inflows_df = generate_sample_cashflows()
    
    # Initialize calculator
    calculator = BaselIIILCRCalculator()
    
    # Perform calculations
    hqla_classified = calculator.classify_hqla(hqla_df)
    outflows_calc = calculator.calculate_outflows(outflows_df)
    inflows_calc = calculator.calculate_inflows(inflows_df)
    lcr_result = calculator.calculate_lcr()
    
    # Display LCR Result
    st.markdown("### LCR Calculation Result")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "LCR Ratio",
            f"{lcr_result['lcr_ratio']:.1f}%",
            delta="Above 100%" if lcr_result['meets_requirement'] else "Below 100%"
        )
    with col2:
        st.metric("Total HQLA", f"${lcr_result['total_hqla']/1e9:.2f}B")
    with col3:
        st.metric("Net Cash Outflows", f"${lcr_result['net_cash_outflows']/1e9:.2f}B")
    with col4:
        surplus_color = "normal" if lcr_result['surplus_deficit'] >= 0 else "inverse"
        st.metric(
            "Surplus/Deficit",
            f"${lcr_result['surplus_deficit']/1e9:.2f}B",
            delta_color=surplus_color
        )
    
    if lcr_result['meets_requirement']:
        st.success("✅ LCR meets Basel III minimum requirement of 100%")
    else:
        st.error("❌ LCR below Basel III minimum requirement of 100%")
    
    st.markdown("---")
    
    # HQLA Composition
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("HQLA Composition")
        hqla_comp = calculator.get_hqla_composition()
        fig_hqla = px.pie(
            hqla_comp,
            values='hqla_value',
            names='hqla_level',
            title='HQLA by Level',
            color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c']
        )
        st.plotly_chart(fig_hqla, use_container_width=True)
    
    with col2:
        st.subheader("Cash Flow Breakdown")
        outflow_breakdown = calculator.get_outflow_breakdown()
        fig_outflow = px.bar(
            outflow_breakdown,
            x='stress_outflow',
            y='category',
            orientation='h',
            title='Stress Outflows by Category'
        )
        st.plotly_chart(fig_outflow, use_container_width=True)
    
    # Sensitivity Analysis
    st.markdown("---")
    st.subheader("Sensitivity Analysis")
    
    scenarios = {
        'Mild Stress': {'hqla_shock': -0.05, 'outflow_increase': 0.10},
        'Moderate Stress': {'hqla_shock': -0.10, 'outflow_increase': 0.20},
        'Severe Stress': {'hqla_shock': -0.15, 'outflow_increase': 0.30},
        'Extreme Stress': {'hqla_shock': -0.20, 'outflow_increase': 0.40}
    }
    
    sensitivity_df = calculator.sensitivity_analysis(scenarios)
    
    fig_sens = px.bar(
        sensitivity_df,
        x='scenario',
        y='lcr_ratio',
        title='LCR Under Different Stress Scenarios',
        color='lcr_ratio',
        color_continuous_scale='RdYlGn'
    )
    fig_sens.add_hline(y=100, line_dash="dash", line_color="red", 
                       annotation_text="Minimum Requirement (100%)")
    st.plotly_chart(fig_sens, use_container_width=True)
    
    st.dataframe(sensitivity_df, use_container_width=True)


# ALM GAP Analysis Page
def show_alm_gap():
    """Display ALM GAP analysis page."""
    st.title("⚖️ Asset Liability Management (ALM) GAP Analysis")
    
    # Generate sample data
    assets_df, liabilities_df = generate_sample_alm_data()
    
    # Initialize analyzer
    analyzer = ALMGAPAnalyzer()
    
    # Perform analysis
    rsa = analyzer.classify_assets(assets_df)
    rsl = analyzer.classify_liabilities(liabilities_df)
    gap_df = analyzer.calculate_gap()
    
    # GAP Summary
    summary = analyzer.get_gap_position_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total RSA", f"${summary['total_rsa']/1e9:.2f}B")
    with col2:
        st.metric("Total RSL", f"${summary['total_rsl']/1e9:.2f}B")
    with col3:
        st.metric("Total GAP", f"${summary['total_gap']/1e9:.2f}B")
    with col4:
        st.metric("RSA/RSL Ratio", f"{summary['rsa_rsl_ratio']:.2f}")
    
    st.info(f"**Position:** {summary['position_type']} - {summary['rate_scenario_implication']}")
    
    st.markdown("---")
    
    # GAP Chart
    st.subheader("GAP Analysis by Time Bucket")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gap_df['time_bucket'],
        y=gap_df['rsa']/1e9,
        name='Rate Sensitive Assets',
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        x=gap_df['time_bucket'],
        y=gap_df['rsl']/1e9,
        name='Rate Sensitive Liabilities',
        marker_color='red'
    ))
    fig.update_layout(
        title='RSA vs RSL by Time Bucket',
        xaxis_title='Time Bucket',
        yaxis_title='Amount ($ Billions)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cumulative GAP
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cumulative GAP")
        fig_cumgap = px.line(
            gap_df,
            x='time_bucket',
            y='cumulative_gap',
            title='Cumulative GAP Position',
            markers=True
        )
        fig_cumgap.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_cumgap, use_container_width=True)
    
    with col2:
        st.subheader("GAP Ratio by Bucket")
        fig_ratio = px.bar(
            gap_df,
            x='time_bucket',
            y='gap_ratio',
            title='GAP Ratio (GAP/RSL)',
            color='gap_ratio',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    # NII Sensitivity
    st.markdown("---")
    st.subheader("Net Interest Income (NII) Sensitivity Analysis")
    
    rate_shock = st.slider("Interest Rate Shock (basis points)", -200, 200, 100, 25)
    
    sensitivity_df = analyzer.calculate_nii_sensitivity(rate_shock_bps=rate_shock)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            f"Total NII Impact ({rate_shock} bps)",
            f"${sensitivity_df['nii_impact'].sum()/1e6:.2f}M"
        )
        st.dataframe(sensitivity_df[['time_bucket', 'gap', 'nii_impact']], 
                    use_container_width=True)
    
    with col2:
        fig_nii = px.bar(
            sensitivity_df,
            x='time_bucket',
            y='nii_impact',
            title=f'NII Impact by Time Bucket ({rate_shock} bps shock)',
            color='nii_impact',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_nii, use_container_width=True)


# Market Risk Page
def show_market_risk():
    """Display market risk analytics page."""
    st.title("📈 Market Risk Analytics")
    
    # Generate sample data
    returns_df = generate_sample_returns_data(num_days=252, num_assets=3)
    portfolio = generate_sample_portfolio()
    shock_scenarios = define_shock_scenarios()
    
    # Initialize analyzer
    analyzer = MarketRiskAnalyzer(confidence_level=0.95)
    
    # Calculate returns for first asset
    returns = returns_df.iloc[:, 0]
    analyzer.calculate_returns(returns)
    
    # VaR Calculations
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=100000,
        max_value=10000000,
        value=1000000,
        step=100000
    )
    
    st.markdown("### Value at Risk (VaR) Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Historical VaR
    hist_var = analyzer.historical_var(returns, portfolio_value)
    with col1:
        st.markdown("#### Historical VaR")
        st.metric("VaR (95%)", f"${hist_var['var_dollar']:,.0f}")
        st.metric("CVaR", f"${hist_var['cvar_dollar']:,.0f}")
        st.metric("Volatility", f"{hist_var['volatility']*100:.2f}%")
    
    # Parametric VaR
    param_var = analyzer.parametric_var(returns, portfolio_value)
    with col2:
        st.markdown("#### Parametric VaR")
        st.metric("VaR (95%)", f"${param_var['var_dollar']:,.0f}")
        st.metric("CVaR", f"${param_var['cvar_dollar']:,.0f}")
        st.metric("Mean Return", f"{param_var['mean_return']*100:.3f}%")
    
    # Monte Carlo VaR
    mc_var = analyzer.monte_carlo_var(returns, portfolio_value, num_simulations=10000, random_seed=42)
    with col3:
        st.markdown("#### Monte Carlo VaR")
        st.metric("VaR (95%)", f"${mc_var['var_dollar']:,.0f}")
        st.metric("CVaR", f"${mc_var['cvar_dollar']:,.0f}")
        st.metric("Simulations", f"{mc_var['num_simulations']:,}")
    
    st.markdown("---")
    
    # Returns Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Returns Distribution")
        fig_hist = px.histogram(
            returns,
            nbins=50,
            title='Historical Returns Distribution',
            labels={'value': 'Daily Returns', 'count': 'Frequency'}
        )
        fig_hist.add_vline(
            x=hist_var['var_return'],
            line_dash="dash",
            line_color="red",
            annotation_text="VaR (95%)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Monte Carlo Simulation")
        simulated_pnl = portfolio_value - mc_var['simulated_values']
        fig_mc = px.histogram(
            simulated_pnl,
            nbins=50,
            title='Monte Carlo P&L Distribution',
            labels={'value': 'P&L', 'count': 'Frequency'}
        )
        fig_mc.add_vline(
            x=mc_var['var_dollar'],
            line_dash="dash",
            line_color="red",
            annotation_text="VaR"
        )
        st.plotly_chart(fig_mc, use_container_width=True)
    
    # Shock Scenarios
    st.markdown("---")
    st.subheader("Shock Scenario Analysis")
    
    scenario_results = analyzer.shock_scenario_analysis(portfolio, shock_scenarios)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_shock = px.bar(
            scenario_results,
            x='scenario',
            y='pnl',
            title='Portfolio Impact Under Shock Scenarios',
            color='pnl',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_shock, use_container_width=True)
    
    with col2:
        st.markdown("#### Scenario Results")
        st.dataframe(
            scenario_results[['scenario', 'pnl', 'pnl_percentage']],
            use_container_width=True
        )


# Intraday Liquidity Page
def show_intraday_liquidity():
    """Display intraday liquidity monitoring page."""
    st.title("⏰ Intraday Liquidity Monitoring")
    
    # Date selection
    selected_date = st.date_input(
        "Select Date",
        value=datetime.now().date()
    )
    
    opening_balance = st.number_input(
        "Opening Balance ($)",
        min_value=100_000_000,
        max_value=10_000_000_000,
        value=1_000_000_000,
        step=100_000_000
    )
    
    # Generate intraday data
    payments_df = generate_sample_intraday_payments(date=selected_date, opening_balance=opening_balance)
    
    # Initialize monitor
    monitor = IntradayLiquidityMonitor(opening_balance=opening_balance)
    intraday_df = monitor.process_payment_data(payments_df)
    metrics = monitor.calculate_liquidity_metrics()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Opening Balance", f"${metrics['opening_balance']/1e9:.2f}B")
    with col2:
        st.metric("Closing Balance", f"${metrics['closing_balance']/1e9:.2f}B")
    with col3:
        st.metric("Minimum Balance", f"${metrics['minimum_balance']/1e9:.2f}B")
    with col4:
        st.metric(
            "Utilization Rate",
            f"{metrics['liquidity_utilization_rate']:.1f}%"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Inflows", f"${metrics['total_inflows']/1e9:.2f}B")
    with col2:
        st.metric("Total Outflows", f"${metrics['total_outflows']/1e9:.2f}B")
    with col3:
        st.metric("Net Flow", f"${metrics['net_flow']/1e9:.2f}B")
    
    # Liquidity alerts
    alerts = monitor.generate_liquidity_alerts()
    if alerts:
        for alert in alerts:
            if alert['level'] == 'CRITICAL':
                st.error(f"🚨 {alert['message']}")
            else:
                st.warning(f"⚠️ {alert['message']}")
    
    st.markdown("---")
    
    # Intraday balance chart
    st.subheader("Intraday Balance Movements")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=intraday_df['timestamp'],
        y=intraday_df['balance']/1e6,
        mode='lines',
        name='Balance',
        line=dict(color='blue', width=2),
        fill='tozeroy'
    ))
    fig.add_hline(
        y=opening_balance/1e6,
        line_dash="dash",
        line_color="green",
        annotation_text="Opening Balance"
    )
    fig.add_hline(
        y=metrics['minimum_balance']/1e6,
        line_dash="dash",
        line_color="red",
        annotation_text="Minimum Balance"
    )
    fig.update_layout(
        title='Intraday Liquidity Position',
        xaxis_title='Time',
        yaxis_title='Balance ($ Millions)',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Time bucket analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Activity Pattern")
        hourly_df = monitor.calculate_hourly_patterns()
        fig_hourly = px.bar(
            hourly_df,
            x='hour',
            y='transaction_count',
            title='Transaction Volume by Hour',
            color='total_net_flow',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.subheader("Payment Type Distribution")
        type_summary = intraday_df.groupby('payment_type')['amount'].agg(['sum', 'count']).reset_index()
        fig_type = px.pie(
            type_summary,
            values='sum',
            names='payment_type',
            title='Payment Volume by Type',
            color_discrete_map={'INFLOW': 'green', 'OUTFLOW': 'red'}
        )
        st.plotly_chart(fig_type, use_container_width=True)


# Portfolio Optimizer Page
def show_portfolio_optimizer():
    """Display portfolio optimization page."""
    st.title("💼 Investment Portfolio Optimizer")
    
    # Portfolio parameters
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input(
            "Total Portfolio Value ($)",
            min_value=1_000_000,
            max_value=1_000_000_000,
            value=10_000_000,
            step=1_000_000
        )
    
    with col2:
        optimization_goal = st.selectbox(
            "Optimization Goal",
            ["Maximize Return", "Minimize Risk", "Efficient Frontier"]
        )
    
    # Load assets
    assets_df = generate_sample_assets()
    
    # Display available assets
    with st.expander("📋 View Asset Universe"):
        st.dataframe(assets_df, use_container_width=True)
    
    # Constraints
    st.markdown("### Portfolio Constraints")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_single_asset = st.slider("Max Single Asset (%)", 0, 50, 30) / 100
    with col2:
        target_return = st.slider("Target Return (%)", 0, 20, 8) / 100
    with col3:
        max_risk = st.slider("Max Portfolio Risk (%)", 0, 30, 15) / 100
    
    constraints = {
        'max_single_asset': max_single_asset,
        'max_total_risk': max_risk
    }
    
    # Optimize
    if st.button("🎯 Optimize Portfolio", type="primary"):
        with st.spinner("Running optimization..."):
            optimizer = PortfolioOptimizer()
            
            if optimization_goal == "Maximize Return":
                result = optimizer.optimize_max_return(
                    assets_df,
                    portfolio_value,
                    constraints=constraints
                )
            elif optimization_goal == "Minimize Risk":
                result = optimizer.optimize_min_risk(
                    assets_df,
                    portfolio_value,
                    target_return=target_return,
                    constraints=constraints
                )
            else:  # Efficient Frontier
                result = None
                frontier_df = optimizer.efficient_frontier(
                    assets_df,
                    portfolio_value,
                    num_points=15
                )
            
            if optimization_goal != "Efficient Frontier" and result['status'] == 'Optimal':
                st.success("✅ Optimization completed successfully!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
                with col2:
                    st.metric("Portfolio Risk", f"{result['portfolio_risk']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                
                st.markdown("---")
                
                # Allocation charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Optimal Allocation")
                    alloc_data = pd.DataFrame([
                        {'Asset': k, 'Percentage': v['percentage']*100, 'Amount': v['dollar_amount']}
                        for k, v in result['allocations'].items()
                        if v['percentage'] > 0.001
                    ])
                    # Sort once for reuse
                    alloc_data_sorted = alloc_data.sort_values('Percentage', ascending=False)
                    
                    fig_alloc = px.pie(
                        alloc_data_sorted,
                        values='Percentage',
                        names='Asset',
                        title='Portfolio Allocation'
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True)
                
                with col2:
                    st.subheader("Allocation Details")
                    st.dataframe(
                        alloc_data_sorted,
                        use_container_width=True
                    )
            
            elif optimization_goal == "Efficient Frontier":
                st.subheader("Efficient Frontier")
                
                fig_frontier = px.scatter(
                    frontier_df,
                    x='portfolio_risk',
                    y='portfolio_return',
                    size='sharpe_ratio',
                    color='sharpe_ratio',
                    title='Efficient Frontier',
                    labels={
                        'portfolio_risk': 'Portfolio Risk',
                        'portfolio_return': 'Expected Return',
                        'sharpe_ratio': 'Sharpe Ratio'
                    },
                    color_continuous_scale='Viridis'
                )
                fig_frontier.update_traces(marker=dict(size=15))
                st.plotly_chart(fig_frontier, use_container_width=True)
                
                st.dataframe(frontier_df, use_container_width=True)
            else:
                st.error("❌ Optimization failed. Please adjust constraints.")


# Main app logic
def main():
    """Main application logic."""
    page = sidebar_navigation()
    
    if page == "home":
        show_home()
    elif page == "cashflow":
        show_cashflow_analysis()
    elif page == "forecasting":
        show_ml_forecasting()
    elif page == "lcr":
        show_basel_lcr()
    elif page == "alm":
        show_alm_gap()
    elif page == "market_risk":
        show_market_risk()
    elif page == "intraday":
        show_intraday_liquidity()
    elif page == "portfolio":
        show_portfolio_optimizer()


if __name__ == "__main__":
    main()

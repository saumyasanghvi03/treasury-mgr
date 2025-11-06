import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from modules import (
    cashflow_forecast,
    basel_lcr,
    alm_gap,
    market_risk_var,
    intraday_liquidity,
    investment_optimizer
)
from utils import data_generator

# Page configuration
st.set_page_config(
    page_title="Treasury Management System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title
    st.markdown('<p class="main-header">💰 Treasury Management System</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        [
            "Dashboard",
            "Cash Flow Forecasting",
            "Basel III LCR Analytics",
            "ALM Gap Assessment",
            "Market Risk (VaR)",
            "Intraday Liquidity",
            "Investment Optimizer"
        ]
    )
    
    # Generate sample data
    if 'data' not in st.session_state:
        st.session_state.data = data_generator.generate_sample_data()
    
    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Cash Flow Forecasting":
        cashflow_forecast.show()
    elif page == "Basel III LCR Analytics":
        basel_lcr.show()
    elif page == "ALM Gap Assessment":
        alm_gap.show()
    elif page == "Market Risk (VaR)":
        market_risk_var.show()
    elif page == "Intraday Liquidity":
        intraday_liquidity.show()
    elif page == "Investment Optimizer":
        investment_optimizer.show()

def show_dashboard():
    st.header("Treasury Management Dashboard")
    st.write("Overview of key treasury metrics and indicators")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    data = st.session_state.data
    
    with col1:
        st.metric(
            label="Total Liquidity",
            value=f"${data['liquidity_total']/1e6:.1f}M",
            delta=f"{data['liquidity_change']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="LCR Ratio",
            value=f"{data['lcr_ratio']:.1f}%",
            delta="Compliant" if data['lcr_ratio'] >= 100 else "Below Target"
        )
    
    with col3:
        st.metric(
            label="Portfolio VaR (95%)",
            value=f"${data['var_95']/1e6:.2f}M",
            delta=f"{data['var_change']:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Net Interest Margin",
            value=f"{data['nim']:.2f}%",
            delta=f"{data['nim_change']:.2f}%"
        )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cash Flow Forecast (30 Days)")
        forecast_df = data['cashflow_forecast']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecasted',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='lightblue', width=1, dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='lightblue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Cash Flow ($)",
            hovermode='x unified',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ALM Gap Analysis")
        alm_df = data['alm_gap']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=alm_df['bucket'],
            y=alm_df['assets'],
            name='Assets',
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            x=alm_df['bucket'],
            y=-alm_df['liabilities'],
            name='Liabilities',
            marker_color='red'
        ))
        fig.add_trace(go.Scatter(
            x=alm_df['bucket'],
            y=alm_df['gap'],
            name='Gap',
            mode='lines+markers',
            line=dict(color='blue', width=2),
            yaxis='y2'
        ))
        fig.update_layout(
            barmode='relative',
            xaxis_title="Time Bucket",
            yaxis_title="Amount ($M)",
            yaxis2=dict(
                title="Gap ($M)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional info
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 **Basel III LCR**: Monitor regulatory liquidity requirements")
    
    with col2:
        st.info("📈 **Market Risk**: VaR analysis for portfolio risk management")
    
    with col3:
        st.info("⚡ **Real-time**: Intraday liquidity monitoring capabilities")

if __name__ == "__main__":
    main()

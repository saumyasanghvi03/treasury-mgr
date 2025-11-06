# Treasury Management Solution

## Overview

This repository contains an AI-enabled Treasury Management Solution built with Streamlit. The application provides comprehensive treasury analytics and risk management capabilities for financial institutions.

## Main Application

The core application is implemented in **`app.py`**, which provides the following features:

### 1. Executive Dashboard
- Holistic view of treasury cash flows
- Real-time liquidity trend analysis
- 30-day moving averages for inflows, outflows, and net positions
- Interactive visualizations of daily cash flow movements

### 2. Cash Flow Forecast
- Machine Learning-based forecasting using Random Forest regression
- Configurable forecast horizons (7-60 days)
- Historical pattern analysis with lag features
- Predictive analytics for future liquidity positions

### 3. Basel III LCR (Liquidity Coverage Ratio)
- Regulatory compliance assessment
- HQLA (High-Quality Liquid Assets) calculation with Level 1, 2A, and 2B assets
- Net 30-day cash outflow projections
- Stress scenario simulations

### 4. Market Risk (VaR)
- Monte Carlo simulation for Value at Risk calculation
- Configurable confidence levels and volatility parameters
- Downside risk exposure measurement
- Portfolio risk quantification

### 5. Investment Optimizer
- Portfolio optimization with linear programming
- Yield maximization subject to liquidity constraints
- Duration and HQLA coverage constraints
- Multi-instrument allocation (Cash, T-Bills, G-Secs, CP/CD)

## Technology Stack

- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Plotly**: Interactive visualizations
- **scikit-learn**: Machine Learning (Random Forest)
- **SciPy**: Optimization algorithms

## Getting Started

Refer to `app.py` for the complete implementation and `requirements.txt` for dependencies.

## License

Apache-2.0 License

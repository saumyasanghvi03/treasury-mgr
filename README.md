# Treasury Management Solution 💰

A comprehensive, enterprise-grade Streamlit-based Treasury Management Solution providing advanced analytics, forecasting, and risk management capabilities for financial institutions.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)

## 🎯 Overview

This Treasury Management Solution is a production-ready platform designed for treasury operations, liquidity management, and regulatory compliance. It combines machine learning, optimization algorithms, and financial analytics to provide actionable insights for treasury professionals.

## ✨ Key Features

### 📊 Cash Flow Management
- **Transaction Ingestion**: Load and normalize cash flow data from various sources
- **Historical Analysis**: Categorize and aggregate transactions by period
- **Validation Framework**: Comprehensive data quality checks
- **Multi-category Support**: Operations, Investments, Debt Service, Treasury Operations, Wholesale

### 🔮 ML-Driven Forecasting
- **Random Forest Models**: State-of-the-art machine learning for cash flow prediction
- **Feature Engineering**: Automated creation of lag features, rolling statistics, and temporal patterns
- **Confidence Intervals**: Prediction bounds using ensemble tree distributions
- **Performance Metrics**: MAE, RMSE, R², and cross-validation scores
- **Feature Importance**: Identify key drivers of cash flow patterns

### 🏦 Basel III LCR Calculator
- **HQLA Classification**: Automatic categorization into Level 1, 2A, and 2B assets
- **Haircut Application**: Regulatory-compliant haircut calculations
- **Liquidity Stress Testing**: Stress outflows and inflows under Basel III scenarios
- **Sensitivity Analysis**: Multi-scenario impact assessment
- **Regulatory Compliance**: 100% minimum requirement tracking
- **Composition Analysis**: HQLA and cash flow breakdowns

### ⚖️ ALM GAP Analysis
- **RSA/RSL Buckets**: Rate-sensitive asset and liability classification
- **Time Bucket Allocation**: 7 standard time buckets (0-30 days to 5+ years)
- **GAP Calculations**: Periodic and cumulative GAP analysis
- **NII Sensitivity**: Net Interest Income impact from rate changes
- **Duration Analysis**: Weighted average duration and duration GAP
- **Scenario Analysis**: Parallel and non-parallel rate shift scenarios
- **Position Identification**: Asset/liability sensitive position determination

### 📈 Market Risk Analytics
- **Historical VaR**: Value at Risk using historical distribution
- **Parametric VaR**: Variance-Covariance method with normal distribution
- **Monte Carlo VaR**: Simulation-based risk measurement (10,000+ simulations)
- **Conditional VaR (CVaR)**: Expected shortfall calculations
- **Shock Scenarios**: Predefined stress scenarios (equity crash, rate spike, etc.)
- **Volatility Metrics**: Rolling and EWMA volatility calculations
- **Back-testing Framework**: Model validation and accuracy testing

### ⏰ Intraday Liquidity Monitoring
- **Real-time Tracking**: Monitor cash positions throughout the trading day
- **Payment Flow Analysis**: Timestamped inflow and outflow processing
- **Liquidity Alerts**: Warning and critical threshold monitoring
- **Time Bucket Aggregation**: 30-minute interval analysis
- **Hourly Patterns**: Identify payment timing patterns
- **Utilization Metrics**: Peak usage and minimum balance tracking
- **Forecasting**: Historical pattern-based requirement forecasting

### 💼 Portfolio Optimization
- **Linear Programming**: PuLP-based optimization engine
- **Multiple Objectives**: Maximize return or minimize risk
- **Constraint Management**: Individual asset and sector-level constraints
- **Efficient Frontier**: Risk-return trade-off visualization
- **Sharpe Ratio**: Risk-adjusted return calculations
- **Rebalancing**: Calculate trades needed to achieve target allocation
- **Multi-asset Support**: Equities, fixed income, alternatives, cash

## 🏗️ Architecture

```
treasury-mgr/
├── app.py                          # Main Streamlit application
├── styles.css                      # Enterprise UI theme
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── modules/                        # Core functionality modules
│   ├── cash_flow_ingestion.py     # Data loading and normalization
│   ├── ml_forecasting.py          # Machine learning forecasting
│   ├── basel_lcr.py                # Basel III LCR calculator
│   ├── alm_gap.py                  # ALM GAP analysis
│   ├── market_risk.py              # Market risk analytics
│   ├── intraday_liquidity.py      # Intraday monitoring
│   └── portfolio_optimizer.py     # Portfolio optimization
└── data/
    └── sample/                     # Sample datasets
        ├── transactions.csv        # Historical transactions
        └── intraday_payments.csv   # Intraday payment data
```

### Design Principles

1. **Modular Architecture**: Each module is self-contained with clear interfaces
2. **Separation of Concerns**: UI logic (app.py) separate from business logic (modules/)
3. **Extensibility**: Easy to add new features or modify existing ones
4. **Production-Ready**: Comprehensive error handling and validation
5. **Professional Standards**: Clean code, documentation, and naming conventions

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saumyasanghvi03/treasury-mgr.git
   cd treasury-mgr
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import streamlit; import pandas; import sklearn; import pulp; print('All dependencies installed successfully!')"
   ```

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

3. **Navigate the application**
   - Use the sidebar menu to switch between modules
   - Start with the Home page for an overview
   - Each module has sample data pre-loaded for demonstration

## 📖 Usage Guide

### Cash Flow Analysis

1. Navigate to "Cash Flow Analysis" from the sidebar
2. Use sample data or upload your own CSV file with columns: `date`, `category`, `type`, `amount`, `counterparty`, `currency`
3. View daily trends, category breakdowns, and cumulative positions
4. Export analysis results for reporting

### ML Forecasting

1. Select "ML Forecasting" from the menu
2. Adjust forecast horizon (7-90 days) and model parameters
3. Click "Train Model & Forecast" to generate predictions
4. Review accuracy metrics (MAE, RMSE, R²)
5. Analyze confidence intervals and feature importance
6. Use forecasts for budgeting and planning

### Basel III LCR

1. Go to "Basel III LCR" module
2. Review HQLA composition and classification
3. Check LCR ratio against 100% minimum requirement
4. Run sensitivity analysis across stress scenarios
5. Examine cash flow breakdowns
6. Export results for regulatory reporting

### ALM GAP Analysis

1. Access "ALM GAP Analysis"
2. Review RSA and RSL allocations across time buckets
3. Analyze cumulative GAP position
4. Determine if position is asset or liability sensitive
5. Run NII sensitivity analysis with rate shocks
6. Use insights for interest rate risk management

### Market Risk

1. Select "Market Risk" from navigation
2. Enter portfolio value
3. Review VaR calculations (Historical, Parametric, Monte Carlo)
4. Analyze returns distributions
5. Run shock scenario analysis
6. Use results for risk limits and reporting

### Intraday Liquidity

1. Navigate to "Intraday Liquidity Monitoring"
2. Select date and opening balance
3. Monitor real-time balance movements
4. Review liquidity alerts and utilization metrics
5. Analyze hourly payment patterns
6. Use for operational liquidity management

### Portfolio Optimization

1. Go to "Portfolio Optimizer"
2. Enter portfolio value and select optimization goal
3. Set constraints (max single asset, target return, max risk)
4. Click "Optimize Portfolio"
5. Review optimal allocation and metrics
6. Generate rebalancing trades if needed

## 📊 Sample Data

The repository includes pre-generated sample datasets:

### transactions.csv
- **Records**: ~3,650 transactions (365 days)
- **Categories**: Operations, Investments, Debt Service, Treasury Operations, Wholesale
- **Format**: CSV with columns: date, category, type, amount, counterparty, currency

### intraday_payments.csv
- **Records**: ~1,500 payments (5 trading days)
- **Types**: INFLOW, OUTFLOW
- **Payment Systems**: FEDWIRE, CHIPS, ACH
- **Format**: CSV with columns: timestamp, payment_type, amount, counterparty, payment_system, date

## 🔧 Configuration

### Customizing Parameters

Edit module files to adjust default parameters:

- **ML Forecasting**: Modify `n_estimators`, `max_depth` in `ml_forecasting.py`
- **LCR Calculator**: Adjust haircuts and run-off rates in `basel_lcr.py`
- **ALM Analysis**: Customize time buckets in `alm_gap.py`
- **Market Risk**: Change confidence levels in `market_risk.py`
- **Portfolio Optimizer**: Modify default constraints in `portfolio_optimizer.py`

### Styling

Customize the UI by editing `styles.css`:
- Color scheme (primary, secondary colors)
- Font families and sizes
- Card and button styles
- Layout spacing

## 🧪 Testing

### Manual Testing

Each module includes sample data for testing:

```python
# Test cash flow ingestion
from modules.cash_flow_ingestion import generate_sample_transaction_data
df = generate_sample_transaction_data('test.csv', num_days=30)

# Test forecasting
from modules.ml_forecasting import CashFlowForecaster
forecaster = CashFlowForecaster()
# ... (see module documentation)
```

### Validation

Run validation checks:
```python
from modules.cash_flow_ingestion import validate_cash_flow_data
results = validate_cash_flow_data(df)
print(results)
```

## 📈 Performance Considerations

- **Data Volume**: Optimized for datasets with up to 1M transactions
- **Forecasting**: Training time scales with number of features and estimators
- **Monte Carlo**: 10,000 simulations take ~1-2 seconds
- **Optimization**: Linear programming solves in milliseconds for typical portfolios

## 🔐 Security Notes

- **Data Privacy**: All processing is local; no data sent to external services
- **File Uploads**: Validate and sanitize uploaded files before processing
- **Access Control**: Implement authentication for production deployments
- **Audit Trail**: Consider logging user actions for compliance

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write clear commit messages
- Add unit tests for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Treasury Management Team** - Initial work and ongoing maintenance

## 🙏 Acknowledgments

- Basel Committee on Banking Supervision for LCR framework
- Financial industry best practices and standards
- Open-source community for excellent libraries (scikit-learn, PuLP, Plotly)

## 📞 Support

For issues, questions, or suggestions:

1. Check the documentation in each module
2. Review the sample code and demos
3. Open an issue on GitHub
4. Contact the development team

## 🗺️ Roadmap

Future enhancements planned:

- [ ] Real-time data integration (APIs, databases)
- [ ] Additional ML models (LSTM, Prophet)
- [ ] Enhanced reporting and export capabilities
- [ ] Multi-currency support
- [ ] User authentication and role-based access
- [ ] Automated regulatory reporting
- [ ] Integration with treasury management systems
- [ ] Mobile-responsive design improvements
- [ ] Collaborative features (comments, sharing)
- [ ] Advanced visualization options

## 📚 Additional Resources

- [Basel III Framework](https://www.bis.org/bcbs/basel3.htm)
- [ALM Best Practices](https://www.bis.org/publ/bcbs189.htm)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [PuLP Documentation](https://coin-or.github.io/pulp/)

---

**Built with ❤️ for Treasury Professionals**
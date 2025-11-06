# 💰 Treasury Management System

A comprehensive full-stack Treasury Management Solution built with Streamlit for banks and fintech teams. This system provides end-to-end treasury decision support including ML-based forecasting, regulatory compliance analytics, risk modeling, and portfolio optimization.

## 🚀 Features

### 1. **Cash Flow Forecasting with ML**
- Advanced machine learning models (Random Forest, Gradient Boosting, Linear Regression)
- Feature engineering with lag variables, rolling statistics, and trend analysis
- 95% confidence intervals for predictions
- Model performance metrics (R², RMSE, MAE, MAPE)
- Feature importance analysis
- Customizable forecast horizons (7-90 days)

### 2. **Basel III LCR Analytics**
- Liquidity Coverage Ratio (LCR) calculation and monitoring
- High Quality Liquid Assets (HQLA) composition analysis
- Level 1, 2A, and 2B asset categorization with appropriate haircuts
- 30-day stress scenario modeling
- Cash flow analysis (inflows/outflows)
- Funding concentration risk assessment
- Interactive scenario testing with adjustable parameters
- Regulatory compliance monitoring

### 3. **ALM Gap Assessment**
- Asset-Liability Management maturity gap analysis
- Interest rate sensitivity analysis
- Duration and modified duration calculations
- Net Interest Income (NII) impact modeling
- Cumulative gap tracking across time buckets
- Price sensitivity to rate changes
- Risk metrics and hedging strategy recommendations

### 4. **Market Risk (VaR) Modeling**
- Multiple VaR methodologies:
  - Historical simulation
  - Parametric (variance-covariance)
  - Monte Carlo simulation
- Conditional VaR (Expected Shortfall/CVaR)
- Portfolio return distribution analysis
- Component VaR by asset class
- Stress testing and scenario analysis
- Risk contribution analysis
- Confidence levels: 90%, 95%, 99%

### 5. **Intraday Liquidity Monitoring**
- Real-time liquidity position tracking
- Payment flow analysis (inflows/outflows)
- Large payment monitoring
- Payment channel analysis (RTGS, SWIFT, ACH, Internal)
- Liquidity buffer tracking
- Coverage ratio monitoring
- Active alerts and threshold monitoring
- Intraday liquidity forecasting

### 6. **Investment Portfolio Optimizer**
- Linear programming-based optimization using PuLP
- Multiple optimization objectives:
  - Maximize return
  - Minimize risk
  - Risk-adjusted return (Sharpe ratio)
- Configurable constraints:
  - Budget allocation
  - Position limits
  - Minimum diversification
  - Target return requirements
  - Liquidity constraints
- Efficient frontier visualization
- Risk and return contribution analysis
- Portfolio concentration metrics (HHI)

### 7. **Interactive Dashboard**
- Comprehensive overview of key treasury metrics
- Real-time KPI monitoring
- Visual analytics with Plotly charts
- Drill-down capabilities
- Responsive design

## 📋 Requirements

- Python 3.8+
- Streamlit
- pandas
- numpy
- scikit-learn
- scipy
- plotly
- matplotlib
- PuLP (for linear programming)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/saumyasanghvi03/treasury-mgr.git
cd treasury-mgr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Navigation

Use the sidebar to navigate between different modules:
- **Dashboard**: Overview of all treasury metrics
- **Cash Flow Forecasting**: ML-based predictions
- **Basel III LCR Analytics**: Regulatory compliance
- **ALM Gap Assessment**: Interest rate risk
- **Market Risk (VaR)**: Portfolio risk modeling
- **Intraday Liquidity**: Real-time monitoring
- **Investment Optimizer**: Portfolio optimization

## 🎨 Architecture

```
treasury-mgr/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── modules/                        # Feature modules
│   ├── cashflow_forecast.py       # ML forecasting
│   ├── basel_lcr.py               # LCR analytics
│   ├── alm_gap.py                 # ALM assessment
│   ├── market_risk_var.py         # VaR modeling
│   ├── intraday_liquidity.py      # Liquidity monitoring
│   └── investment_optimizer.py    # Portfolio optimization
└── utils/                          # Utility functions
    └── data_generator.py          # Sample data generation
```

## 🔬 Technical Details

### Machine Learning Models
- **Random Forest**: Ensemble of decision trees for robust predictions
- **Gradient Boosting**: Sequential tree building for high accuracy
- **Linear Regression**: Baseline model for comparison

### Optimization Algorithm
- **Linear Programming**: Uses PuLP with CBC solver
- Handles complex constraints and multiple objectives
- Generates efficient frontier for risk-return tradeoffs

### Risk Metrics
- **VaR**: Value at Risk at multiple confidence levels
- **CVaR**: Conditional Value at Risk (Expected Shortfall)
- **Duration**: Macaulay and Modified Duration
- **HHI**: Herfindahl-Hirschman Index for concentration

## 📈 Sample Data

The application includes a comprehensive sample data generator that creates realistic:
- Historical cash flows with trends and seasonality
- Portfolio holdings across multiple asset classes
- Investment universe with risk-return characteristics
- Intraday payment flows
- LCR components and stress scenarios

## 🎓 Use Cases

1. **Treasury Management**: Daily liquidity and risk monitoring
2. **Regulatory Compliance**: Basel III LCR reporting
3. **Risk Analytics**: Portfolio VaR and stress testing
4. **Investment Planning**: Optimal portfolio allocation
5. **ALM**: Interest rate risk management
6. **Training & Education**: Treasury operations simulation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author

Saumya Sanghvi

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Optimization powered by [PuLP](https://coin-or.github.io/pulp/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/)

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.
# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/saumyasanghvi03/treasury-mgr.git
cd treasury-mgr

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Start Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Running the Demo

To see all features in action:

```bash
python demo.py
```

## Quick Feature Tour

### 1. Cash Flow Analysis
- Upload CSV or use sample data
- View daily trends and cumulative positions
- Analyze by category

### 2. ML Forecasting
- Adjust forecast horizon (7-90 days)
- Train Random Forest model
- View predictions with confidence intervals

### 3. Basel III LCR
- Review HQLA composition
- Check regulatory compliance
- Run stress scenarios

### 4. ALM GAP Analysis
- Analyze interest rate risk
- View RSA/RSL buckets
- Calculate NII sensitivity

### 5. Market Risk
- Calculate VaR (Historical, Parametric, Monte Carlo)
- Run shock scenarios
- Analyze portfolio impact

### 6. Intraday Liquidity
- Monitor real-time positions
- Track payment flows
- View liquidity alerts

### 7. Portfolio Optimization
- Optimize allocation
- Set constraints
- Generate efficient frontier

## Data Format

### Transaction Data (transactions.csv)
```csv
date,category,type,amount,counterparty,currency
2024-01-01,OPERATIONS,INFLOW,150000.00,CP_1234,USD
```

### Intraday Payments (intraday_payments.csv)
```csv
timestamp,payment_type,amount,counterparty,payment_system
2024-01-01 09:15:30,OUTFLOW,50000.00,CP_5678,FEDWIRE
```

## Common Tasks

### Custom Data Upload
1. Navigate to desired module
2. Uncheck "Use sample data"
3. Upload your CSV file

### Adjusting Parameters
- Forecast days: Use slider in ML Forecasting
- Rate shocks: Use slider in ALM GAP
- Portfolio constraints: Use inputs in Portfolio Optimizer

### Exporting Results
- Take screenshots of charts
- Copy dataframes from UI
- Generate reports from analysis

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Memory Issues
Reduce data size or forecast horizon

## Next Steps

1. Explore each module using sample data
2. Upload your own data
3. Customize parameters for your use case
4. Review comprehensive README.md for detailed documentation

## Support

For issues or questions:
- Check module documentation
- Review demo.py examples
- Consult README.md

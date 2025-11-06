# Treasury Management System - Usage Guide

This guide provides detailed instructions for using the Treasury Management System.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard](#dashboard)
3. [Cash Flow Forecasting](#cash-flow-forecasting)
4. [Basel III LCR Analytics](#basel-iii-lcr-analytics)
5. [ALM Gap Assessment](#alm-gap-assessment)
6. [Market Risk (VaR)](#market-risk-var)
7. [Intraday Liquidity](#intraday-liquidity)
8. [Investment Optimizer](#investment-optimizer)

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Dashboard

The dashboard provides a high-level overview of all treasury operations:

- **Total Liquidity**: Current liquidity position with trend
- **LCR Ratio**: Basel III compliance status
- **Portfolio VaR**: Risk exposure at 95% confidence
- **Net Interest Margin**: Profitability metric

The dashboard includes:
- 30-day cash flow forecast visualization
- ALM gap analysis chart
- Quick access to all modules

## Cash Flow Forecasting

### Purpose
Predict future cash flows using machine learning to optimize liquidity management.

### How to Use

1. **Select ML Model**:
   - Random Forest: Best for non-linear patterns
   - Gradient Boosting: High accuracy for complex data
   - Linear Regression: Simple baseline model

2. **Set Forecast Horizon**: Choose between 7-90 days

3. **Review Model Performance**:
   - R² Score: Model fit (closer to 1 is better)
   - RMSE: Prediction error in dollars
   - MAE: Average absolute error
   - MAPE: Percentage error

4. **Analyze Forecast**:
   - View predicted cash flows with confidence intervals
   - Check volatility and coefficient of variation
   - Export forecast data for planning

### Best Practices
- Use longer forecast horizons for strategic planning
- Monitor forecast accuracy against actuals
- Update model regularly with new data
- Consider seasonal patterns in your business

## Basel III LCR Analytics

### Purpose
Monitor Liquidity Coverage Ratio compliance with Basel III regulations.

### Key Metrics

**LCR Formula**: HQLA / Net Cash Outflows (30 days) ≥ 100%

**HQLA Categories**:
- Level 1 (0% haircut): Cash, central bank reserves, government securities
- Level 2A (15% haircut): High-quality corporate bonds
- Level 2B (50% haircut): Lower-rated securities

### How to Use

1. **Monitor LCR Ratio**:
   - Green (>150%): Excellent liquidity position
   - Yellow (100-120%): Adequate but monitor closely
   - Red (<100%): Regulatory breach - action required

2. **Analyze HQLA Composition**:
   - Review distribution across Level 1, 2A, 2B
   - Optimize for highest quality assets
   - Balance yield vs. liquidity

3. **Stress Testing**:
   - Adjust HQLA haircut assumptions
   - Modify outflow scenarios
   - Test deposit run-off rates
   - Analyze impact on compliance

4. **Concentration Analysis**:
   - Review funding source diversity
   - Monitor counterparty exposure
   - Track maturity profile

### Regulatory Guidelines
- Maintain LCR ≥ 100% at all times
- Report daily to supervisors
- Document stress testing assumptions
- Maintain contingency funding plans

## ALM Gap Assessment

### Purpose
Manage interest rate risk by analyzing asset-liability maturity mismatches.

### Key Concepts

**Gap = Assets - Liabilities** for each time bucket

- Positive Gap: Asset-sensitive (benefits from rising rates)
- Negative Gap: Liability-sensitive (benefits from falling rates)

### How to Use

1. **Review Maturity Profile**:
   - Identify gaps in each time bucket
   - Focus on 0-1M and 1-3M for immediate risks
   - Monitor cumulative gap trend

2. **Assess Rate Sensitivity**:
   - Use rate shock scenario (e.g., +100 bps)
   - Calculate Net Interest Income (NII) impact
   - Evaluate on different time horizons

3. **Duration Analysis**:
   - Compare asset vs. liability duration
   - Calculate duration gap
   - Understand price sensitivity to rate changes

4. **Hedging Strategies**:
   - For positive gaps: Consider pay-fixed swaps
   - For negative gaps: Consider receive-fixed swaps
   - Use futures or options for tactical hedging

### Risk Limits
- Maximum gap as % of assets: 15%
- Duration gap limit: ±2 years
- Monitor cumulative gap closely

## Market Risk (VaR)

### Purpose
Quantify potential portfolio losses using statistical methods.

### VaR Methods

1. **Historical VaR**: Uses actual historical returns distribution
2. **Parametric VaR**: Assumes normal distribution
3. **Monte Carlo VaR**: Simulation-based approach

### How to Use

1. **Configure Parameters**:
   - Confidence Level: 90%, 95%, or 99%
   - Time Horizon: 1, 5, or 10 days
   - Higher confidence = higher VaR

2. **Interpret Results**:
   - VaR = Maximum expected loss (with X% confidence)
   - CVaR = Average loss in worst cases
   - Compare across different methods

3. **Component VaR**:
   - Identify which assets contribute most to risk
   - Use for position sizing and limits
   - Rebalance to reduce concentration

4. **Stress Testing**:
   - Run market crash scenarios
   - Test interest rate shocks
   - Evaluate volatility spikes

### Risk Management
- Set VaR limits as % of portfolio (typically 1-3%)
- Backtest VaR model regularly
- Report VaR breaches immediately
- Use CVaR for tail risk assessment

## Intraday Liquidity

### Purpose
Monitor real-time liquidity position throughout the trading day.

### Key Features

1. **Position Tracking**:
   - Available liquidity vs. requirements
   - Buffer monitoring
   - Coverage ratio

2. **Payment Flows**:
   - Incoming payments by channel
   - Outgoing payments by counterparty
   - Large payment alerts (>$5M)

3. **Channel Analysis**:
   - RTGS: Real-time gross settlement
   - SWIFT: International wire transfers
   - ACH: Automated clearing house
   - Internal: Between accounts

### How to Use

1. **Monitor Dashboard**:
   - Check current buffer status
   - Review hourly liquidity profile
   - Track coverage ratio

2. **Manage Large Payments**:
   - Review pending payments >$5M
   - Prioritize time-critical payments
   - Coordinate with counterparties

3. **Respond to Alerts**:
   - Red alerts: Immediate action required
   - Yellow alerts: Monitor closely
   - Blue alerts: Informational

4. **Funding Actions**:
   - Access intraday repo facilities
   - Use central bank standing facilities
   - Arrange interbank borrowing
   - Postpone non-urgent payments

### Best Practices
- Monitor continuously during business hours
- Maintain adequate intraday buffer
- Test contingency funding sources
- Document all funding decisions

## Investment Optimizer

### Purpose
Optimize portfolio allocation using mathematical programming.

### Optimization Objectives

1. **Maximize Return**: Best for risk-tolerant portfolios
2. **Minimize Risk**: Best for conservative mandates
3. **Risk-Adjusted Return**: Balanced approach (Sharpe ratio)

### How to Use

1. **Set Parameters**:
   - Total Budget: Amount to invest
   - Objective: Return vs. risk preference
   - Min Liquidity Score: Ensure liquidity needs

2. **Configure Constraints**:
   - Max Single Position: Concentration limit (e.g., 30%)
   - Min Instruments: Diversification requirement
   - Target Return: Minimum acceptable return

3. **Run Optimization**:
   - Click "Optimize Portfolio" button
   - Review allocation results
   - Check constraint satisfaction

4. **Analyze Results**:
   - Review efficient frontier
   - Compare risk-return profile
   - Check portfolio characteristics
   - Evaluate concentration (HHI)

### Interpreting Results

**Efficient Frontier**: Shows optimal risk-return combinations
- Points on the curve are efficient
- Your portfolio shown as red star
- Individual assets shown for comparison

**Sharpe Ratio**: Risk-adjusted return metric
- >1.0: Good risk-adjusted returns
- >2.0: Excellent performance
- <0.5: Poor risk-adjusted returns

**HHI (Concentration)**: Lower is better
- <1000: Highly diversified
- 1000-1800: Moderate concentration
- >1800: Concentrated portfolio

### Best Practices
- Re-optimize regularly (monthly/quarterly)
- Compare with benchmark allocations
- Consider transaction costs
- Account for tax implications
- Document optimization assumptions

## Tips and Tricks

### General
- Use the sidebar to navigate between modules
- Export data using dataframe download buttons
- Adjust parameters to run what-if scenarios
- Compare results across different assumptions

### Performance
- Close unused browser tabs to improve performance
- Refresh page if charts don't render properly
- Use smaller date ranges for faster processing

### Data
- Default data is for demonstration purposes
- Replace with your actual data in production
- Ensure data quality for accurate results
- Validate assumptions regularly

## Support

For issues or questions:
1. Check documentation in README.md
2. Review code comments in source files
3. Run demo.py for working examples
4. Open an issue on GitHub

## Glossary

- **HQLA**: High Quality Liquid Assets
- **LCR**: Liquidity Coverage Ratio
- **VaR**: Value at Risk
- **CVaR**: Conditional VaR (Expected Shortfall)
- **ALM**: Asset-Liability Management
- **NII**: Net Interest Income
- **HHI**: Herfindahl-Hirschman Index
- **Duration**: Interest rate sensitivity measure
- **RTGS**: Real-Time Gross Settlement

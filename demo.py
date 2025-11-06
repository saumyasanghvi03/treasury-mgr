"""
Demo Script - Treasury Management Solution

This script demonstrates the key functionalities of each module
and validates that all components work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Import all modules
from modules.cash_flow_ingestion import (
    load_transaction_data, normalize_cash_flows, aggregate_daily_cash_flows,
    validate_cash_flow_data
)
from modules.ml_forecasting import CashFlowForecaster
from modules.basel_lcr import (
    BaselIIILCRCalculator, generate_sample_hqla_data, generate_sample_cashflows
)
from modules.alm_gap import ALMGAPAnalyzer, generate_sample_alm_data
from modules.market_risk import (
    MarketRiskAnalyzer, generate_sample_returns_data, generate_sample_portfolio,
    define_shock_scenarios
)
from modules.intraday_liquidity import IntradayLiquidityMonitor, generate_sample_intraday_payments
from modules.portfolio_optimizer import PortfolioOptimizer, generate_sample_assets


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_cash_flow_analysis():
    """Demo cash flow analysis functionality."""
    print_section("1. CASH FLOW ANALYSIS DEMO")
    
    # Load data
    print("Loading transaction data...")
    df = load_transaction_data('data/sample/transactions.csv')
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Normalize
    print("\nNormalizing cash flows...")
    normalized_df = normalize_cash_flows(df)
    print(f"✓ Normalized {len(normalized_df):,} records")
    
    # Aggregate
    print("\nAggregating daily cash flows...")
    daily_cf = aggregate_daily_cash_flows(normalized_df)
    print(f"✓ Created {len(daily_cf):,} daily records")
    
    # Validate
    print("\nValidating data...")
    validation = validate_cash_flow_data(df)
    print(f"✓ Validation status: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    
    # Summary statistics
    total_inflows = normalized_df[normalized_df['type'] == 'INFLOW']['amount'].sum()
    total_outflows = normalized_df[normalized_df['type'] == 'OUTFLOW']['amount'].sum()
    print(f"\n📊 Summary:")
    print(f"   - Total Inflows:  ${total_inflows:,.0f}")
    print(f"   - Total Outflows: ${total_outflows:,.0f}")
    print(f"   - Net Cash Flow:  ${total_inflows - total_outflows:,.0f}")
    
    return daily_cf


def demo_ml_forecasting(daily_cf):
    """Demo ML forecasting functionality."""
    print_section("2. ML-DRIVEN FORECASTING DEMO")
    
    # Initialize forecaster
    print("Initializing Random Forest forecaster...")
    forecaster = CashFlowForecaster(n_estimators=50)
    print("✓ Forecaster initialized")
    
    # Train model
    print("\nTraining model...")
    metrics = forecaster.train(daily_cf, test_size=0.2)
    print("✓ Model trained successfully")
    
    # Display metrics
    print(f"\n📈 Model Performance:")
    print(f"   - MAE:       ${metrics['mae']:,.0f}")
    print(f"   - RMSE:      ${metrics['rmse']:,.0f}")
    print(f"   - R² Score:  {metrics['r2_score']:.4f}")
    print(f"   - CV MAE:    ${metrics['cv_mae_mean']:,.0f}")
    
    # Generate forecast
    print("\nGenerating 30-day forecast...")
    forecast_df = forecaster.forecast(daily_cf, forecast_days=30)
    print(f"✓ Generated {len(forecast_df)} forecast points")
    
    # Top features
    print("\n🔍 Top 5 Important Features:")
    top_features = forecaster.get_feature_importance(top_n=5)
    for idx, row in top_features.iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")


def demo_basel_lcr():
    """Demo Basel III LCR calculator."""
    print_section("3. BASEL III LCR DEMO")
    
    # Generate sample data
    print("Generating sample HQLA and cash flow data...")
    hqla_df = generate_sample_hqla_data()
    outflows_df, inflows_df = generate_sample_cashflows()
    print("✓ Sample data generated")
    
    # Initialize calculator
    print("\nInitializing LCR calculator...")
    calculator = BaselIIILCRCalculator()
    print("✓ Calculator initialized")
    
    # Perform calculations
    print("\nCalculating LCR...")
    hqla_classified = calculator.classify_hqla(hqla_df)
    outflows_calc = calculator.calculate_outflows(outflows_df)
    inflows_calc = calculator.calculate_inflows(inflows_df)
    lcr_result = calculator.calculate_lcr()
    print("✓ LCR calculated")
    
    # Display results
    print(f"\n🏦 LCR Results:")
    print(f"   - LCR Ratio:           {lcr_result['lcr_ratio']:.2f}%")
    print(f"   - Total HQLA:          ${lcr_result['total_hqla']/1e9:.2f}B")
    print(f"   - Net Cash Outflows:   ${lcr_result['net_cash_outflows']/1e9:.2f}B")
    print(f"   - Surplus/Deficit:     ${lcr_result['surplus_deficit']/1e9:.2f}B")
    print(f"   - Meets Requirement:   {'✓ YES' if lcr_result['meets_requirement'] else '✗ NO'}")
    
    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    scenarios = {
        'Mild Stress': {'hqla_shock': -0.05, 'outflow_increase': 0.10},
        'Severe Stress': {'hqla_shock': -0.15, 'outflow_increase': 0.30}
    }
    sensitivity_df = calculator.sensitivity_analysis(scenarios)
    print("\n📊 Stress Scenarios:")
    for _, row in sensitivity_df.iterrows():
        print(f"   - {row['scenario']}: {row['lcr_ratio']:.2f}%")


def demo_alm_gap():
    """Demo ALM GAP analysis."""
    print_section("4. ALM GAP ANALYSIS DEMO")
    
    # Generate sample data
    print("Generating sample assets and liabilities...")
    assets_df, liabilities_df = generate_sample_alm_data()
    print("✓ Sample data generated")
    
    # Initialize analyzer
    print("\nInitializing ALM GAP analyzer...")
    analyzer = ALMGAPAnalyzer()
    print("✓ Analyzer initialized")
    
    # Perform analysis
    print("\nPerforming GAP analysis...")
    rsa = analyzer.classify_assets(assets_df)
    rsl = analyzer.classify_liabilities(liabilities_df)
    gap_df = analyzer.calculate_gap()
    print("✓ GAP analysis complete")
    
    # Get summary
    summary = analyzer.get_gap_position_summary()
    print(f"\n⚖️ GAP Analysis Summary:")
    print(f"   - Total RSA:         ${summary['total_rsa']/1e9:.2f}B")
    print(f"   - Total RSL:         ${summary['total_rsl']/1e9:.2f}B")
    print(f"   - Total GAP:         ${summary['total_gap']/1e9:.2f}B")
    print(f"   - RSA/RSL Ratio:     {summary['rsa_rsl_ratio']:.3f}")
    print(f"   - Position Type:     {summary['position_type']}")
    
    # NII sensitivity
    print("\nCalculating NII sensitivity (100 bps shock)...")
    sensitivity_df = analyzer.calculate_nii_sensitivity(rate_shock_bps=100)
    total_impact = sensitivity_df['nii_impact'].sum()
    print(f"   - Total NII Impact:  ${total_impact/1e6:.2f}M")


def demo_market_risk():
    """Demo market risk analytics."""
    print_section("5. MARKET RISK ANALYTICS DEMO")
    
    # Generate sample data
    print("Generating sample returns data...")
    returns_df = generate_sample_returns_data(num_days=252, num_assets=3)
    portfolio = generate_sample_portfolio()
    print("✓ Sample data generated")
    
    # Initialize analyzer
    print("\nInitializing market risk analyzer...")
    analyzer = MarketRiskAnalyzer(confidence_level=0.95)
    returns = returns_df.iloc[:, 0]
    analyzer.calculate_returns(returns)
    print("✓ Analyzer initialized")
    
    # Calculate VaR
    portfolio_value = 1_000_000
    print(f"\nCalculating VaR (Portfolio Value: ${portfolio_value:,})...")
    
    hist_var = analyzer.historical_var(returns, portfolio_value)
    param_var = analyzer.parametric_var(returns, portfolio_value)
    mc_var = analyzer.monte_carlo_var(returns, portfolio_value, num_simulations=10000)
    
    print(f"\n📈 VaR Results (95% Confidence):")
    print(f"   - Historical VaR:    ${hist_var['var_dollar']:,.0f}")
    print(f"   - Parametric VaR:    ${param_var['var_dollar']:,.0f}")
    print(f"   - Monte Carlo VaR:   ${mc_var['var_dollar']:,.0f}")
    
    # Shock scenarios
    print("\nRunning shock scenario analysis...")
    shock_scenarios = define_shock_scenarios()
    scenario_results = analyzer.shock_scenario_analysis(portfolio, shock_scenarios)
    print("\n💥 Shock Scenario Impacts:")
    for _, row in scenario_results.head(3).iterrows():
        print(f"   - {row['scenario']}: {row['pnl_percentage']:+.2f}%")


def demo_intraday_liquidity():
    """Demo intraday liquidity monitoring."""
    print_section("6. INTRADAY LIQUIDITY MONITORING DEMO")
    
    # Generate sample data
    opening_balance = 1_000_000_000
    print(f"Generating intraday payment data (Opening Balance: ${opening_balance/1e9:.2f}B)...")
    payments_df = generate_sample_intraday_payments(opening_balance=opening_balance)
    print(f"✓ Generated {len(payments_df):,} payments")
    
    # Initialize monitor
    print("\nInitializing intraday monitor...")
    monitor = IntradayLiquidityMonitor(opening_balance=opening_balance)
    intraday_df = monitor.process_payment_data(payments_df)
    print("✓ Monitor initialized and data processed")
    
    # Calculate metrics
    print("\nCalculating liquidity metrics...")
    metrics = monitor.calculate_liquidity_metrics()
    print("✓ Metrics calculated")
    
    print(f"\n⏰ Intraday Liquidity Summary:")
    print(f"   - Opening Balance:      ${metrics['opening_balance']/1e9:.2f}B")
    print(f"   - Closing Balance:      ${metrics['closing_balance']/1e9:.2f}B")
    print(f"   - Minimum Balance:      ${metrics['minimum_balance']/1e9:.2f}B")
    print(f"   - Utilization Rate:     {metrics['liquidity_utilization_rate']:.2f}%")
    print(f"   - Total Transactions:   {metrics['total_transactions']:,}")
    
    # Alerts
    alerts = monitor.generate_liquidity_alerts()
    if alerts:
        print(f"\n⚠️  Liquidity Alerts: {len(alerts)} alert(s) detected")
    else:
        print("\n✓ No liquidity alerts")


def demo_portfolio_optimization():
    """Demo portfolio optimization."""
    print_section("7. PORTFOLIO OPTIMIZATION DEMO")
    
    # Generate sample assets
    print("Generating sample asset universe...")
    assets_df = generate_sample_assets()
    print(f"✓ Generated {len(assets_df)} assets")
    
    # Initialize optimizer
    print("\nInitializing portfolio optimizer...")
    optimizer = PortfolioOptimizer()
    print("✓ Optimizer initialized")
    
    # Optimize for max return
    portfolio_value = 10_000_000
    constraints = {
        'max_single_asset': 0.30,
        'max_total_risk': 0.15
    }
    
    print(f"\nOptimizing portfolio (Value: ${portfolio_value/1e6:.1f}M)...")
    result = optimizer.optimize_max_return(assets_df, portfolio_value, constraints=constraints)
    
    if result['status'] == 'Optimal':
        print("✓ Optimization successful")
        
        print(f"\n💼 Portfolio Results:")
        print(f"   - Expected Return:   {result['expected_return']*100:.2f}%")
        print(f"   - Portfolio Risk:    {result['portfolio_risk']*100:.2f}%")
        print(f"   - Sharpe Ratio:      {result['sharpe_ratio']:.3f}")
        
        print("\n📊 Top 5 Allocations:")
        sorted_allocs = sorted(result['allocations'].items(), 
                              key=lambda x: x[1]['percentage'], reverse=True)
        for idx, (asset, alloc) in enumerate(sorted_allocs[:5], 1):
            if alloc['percentage'] > 0.001:
                print(f"   {idx}. {asset}: {alloc['percentage']*100:.2f}%")
    else:
        print(f"✗ Optimization failed: {result['status']}")


def main():
    """Run all demos."""
    print("\n" + "🎯"*40)
    print("   TREASURY MANAGEMENT SOLUTION - COMPREHENSIVE DEMO")
    print("🎯"*40)
    
    try:
        # Run all demos
        daily_cf = demo_cash_flow_analysis()
        demo_ml_forecasting(daily_cf)
        demo_basel_lcr()
        demo_alm_gap()
        demo_market_risk()
        demo_intraday_liquidity()
        demo_portfolio_optimization()
        
        # Final summary
        print_section("✅ DEMO COMPLETE")
        print("All modules tested successfully!")
        print("\nTo run the full application:")
        print("  streamlit run app.py")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

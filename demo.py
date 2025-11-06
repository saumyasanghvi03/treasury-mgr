#!/usr/bin/env python3
"""
Demo script to showcase Treasury Management System functionality
"""

import pandas as pd
import numpy as np
from utils import data_generator
from modules import market_risk_var, investment_optimizer

def demo_data_generation():
    """Demonstrate data generation"""
    print("\n" + "="*60)
    print("DEMO: Sample Data Generation")
    print("="*60)
    
    data = data_generator.generate_sample_data()
    
    print(f"\n✓ Generated sample data with {len(data)} components")
    print("\nKey Metrics:")
    print(f"  - Total Liquidity: ${data['liquidity_total']/1e6:.1f}M")
    print(f"  - LCR Ratio: {data['lcr_ratio']:.1f}%")
    print(f"  - VaR (95%): ${data['var_95']/1e6:.2f}M")
    print(f"  - Net Interest Margin: {data['nim']:.2f}%")
    
    print("\nPortfolio Composition:")
    print(data['portfolio'][['asset', 'value', 'weight']].to_string(index=False))
    
    return data

def demo_var_calculation():
    """Demonstrate VaR calculation"""
    print("\n" + "="*60)
    print("DEMO: Value at Risk (VaR) Calculation")
    print("="*60)
    
    data = data_generator.generate_sample_data()
    
    # Calculate VaR
    var_results = market_risk_var.calculate_var(
        data['historical_returns'],
        data['portfolio'],
        95,  # 95% confidence
        1    # 1-day horizon
    )
    
    print("\nVaR at 95% confidence (1-day):")
    print(f"  - Historical VaR: ${var_results['historical']/1e6:.2f}M")
    print(f"  - Parametric VaR: ${var_results['parametric']/1e6:.2f}M")
    print(f"  - Monte Carlo VaR: ${var_results['monte_carlo']/1e6:.2f}M")
    
    # Calculate CVaR
    portfolio_returns = market_risk_var.calculate_portfolio_returns(
        data['historical_returns'],
        data['portfolio']
    )
    portfolio_value = data['portfolio']['value'].sum()
    
    cvar = market_risk_var.calculate_cvar(portfolio_returns, portfolio_value, 95)
    print(f"\n  - Conditional VaR (CVaR): ${cvar/1e6:.2f}M")
    
    return var_results

def demo_optimization():
    """Demonstrate portfolio optimization"""
    print("\n" + "="*60)
    print("DEMO: Portfolio Optimization")
    print("="*60)
    
    data = data_generator.generate_sample_data()
    universe_df = data['investment_universe']
    
    print("\nOptimizing portfolio with:")
    print("  - Budget: $100M")
    print("  - Objective: Maximize Return")
    print("  - Max single position: 30%")
    print("  - Min instruments: 5")
    print("  - Target return: 5%")
    print("  - Min liquidity: 70")
    
    result = investment_optimizer.optimize_portfolio(
        universe_df,
        100_000_000,  # $100M budget
        "Maximize Return",
        0.30,  # 30% max position
        5,     # min 5 instruments
        0.05,  # 5% target return
        70     # liquidity score 70
    )
    
    if result['status'] == 'Optimal':
        print("\n✓ Optimization successful!")
        print(f"\nOptimal Portfolio:")
        print(f"  - Expected Return: {result['expected_return']*100:.2f}%")
        print(f"  - Portfolio Risk: {result['portfolio_risk']*100:.2f}%")
        print(f"  - Sharpe Ratio: {(result['expected_return']/result['portfolio_risk']):.2f}")
        print(f"  - Avg Liquidity: {result['avg_liquidity']:.1f}")
        
        print("\nAllocation:")
        allocation_df = result['allocation'].copy()
        allocation_df['percent'] = (allocation_df['allocation'] / 100_000_000) * 100
        print(allocation_df[['instrument', 'percent']].to_string(index=False, 
              formatters={'percent': '{:.1f}%'.format}))
    else:
        print(f"\n✗ Optimization failed: {result['status']}")
    
    return result

def demo_alm_analysis():
    """Demonstrate ALM gap analysis"""
    print("\n" + "="*60)
    print("DEMO: ALM Gap Analysis")
    print("="*60)
    
    data = data_generator.generate_sample_data()
    alm_df = data['alm_gap']
    
    print("\nMaturity Gap by Time Bucket:")
    print(alm_df.to_string(index=False))
    
    total_gap = alm_df['gap'].sum()
    total_assets = alm_df['assets'].sum()
    
    print(f"\n  - Total Assets: ${total_assets:.0f}M")
    print(f"  - Total Liabilities: ${alm_df['liabilities'].sum():.0f}M")
    print(f"  - Net Gap: ${total_gap:.0f}M")
    print(f"  - Gap Ratio: {(total_gap/total_assets)*100:.1f}%")
    
    # Calculate NII impact
    rate_shock = 100  # 100 bps
    nii_impact = (alm_df['gap'] * (rate_shock/10000)).sum()
    
    print(f"\nImpact of {rate_shock} bps rate shock:")
    print(f"  - NII Impact: ${nii_impact:.2f}M")
    
    return alm_df

def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*15 + "TREASURY MANAGEMENT SYSTEM DEMO")
    print("="*70)
    
    try:
        # Run demos
        data = demo_data_generation()
        var_results = demo_var_calculation()
        opt_result = demo_optimization()
        alm_df = demo_alm_analysis()
        
        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("="*70)
        
        print("\nTo run the full interactive application:")
        print("  streamlit run app.py")
        print("\nThen navigate to http://localhost:8501 in your browser")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

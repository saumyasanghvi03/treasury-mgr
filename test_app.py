#!/usr/bin/env python3
"""
Test script to verify all modules can be imported and basic functionality works
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported")
    except Exception as e:
        print(f"✗ Failed to import streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✓ Data libraries imported")
    except Exception as e:
        print(f"✗ Failed to import data libraries: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✓ Scikit-learn imported")
    except Exception as e:
        print(f"✗ Failed to import scikit-learn: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✓ Plotly imported")
    except Exception as e:
        print(f"✗ Failed to import plotly: {e}")
        return False
    
    try:
        import pulp
        print("✓ PuLP imported")
    except Exception as e:
        print(f"✗ Failed to import pulp: {e}")
        return False
    
    return True

def test_modules():
    """Test that custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from utils import data_generator
        print("✓ Data generator module imported")
        
        # Test data generation
        data = data_generator.generate_sample_data()
        print(f"✓ Sample data generated with {len(data)} keys")
        
    except Exception as e:
        print(f"✗ Failed with data generator: {e}")
        traceback.print_exc()
        return False
    
    try:
        from modules import cashflow_forecast
        print("✓ Cash flow forecast module imported")
    except Exception as e:
        print(f"✗ Failed to import cashflow_forecast: {e}")
        traceback.print_exc()
        return False
    
    try:
        from modules import basel_lcr
        print("✓ Basel LCR module imported")
    except Exception as e:
        print(f"✗ Failed to import basel_lcr: {e}")
        traceback.print_exc()
        return False
    
    try:
        from modules import alm_gap
        print("✓ ALM gap module imported")
    except Exception as e:
        print(f"✗ Failed to import alm_gap: {e}")
        traceback.print_exc()
        return False
    
    try:
        from modules import market_risk_var
        print("✓ Market risk VaR module imported")
    except Exception as e:
        print(f"✗ Failed to import market_risk_var: {e}")
        traceback.print_exc()
        return False
    
    try:
        from modules import intraday_liquidity
        print("✓ Intraday liquidity module imported")
    except Exception as e:
        print(f"✗ Failed to import intraday_liquidity: {e}")
        traceback.print_exc()
        return False
    
    try:
        from modules import investment_optimizer
        print("✓ Investment optimizer module imported")
    except Exception as e:
        print(f"✗ Failed to import investment_optimizer: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_data_generation():
    """Test data generation functionality"""
    print("\nTesting data generation...")
    
    try:
        from utils import data_generator
        data = data_generator.generate_sample_data()
        
        # Check expected keys
        expected_keys = [
            'liquidity_total', 'lcr_ratio', 'var_95', 'nim',
            'cashflow_forecast', 'alm_gap', 'historical_cashflow',
            'portfolio', 'historical_returns', 'lcr_components',
            'intraday_liquidity', 'investment_universe'
        ]
        
        for key in expected_keys:
            if key in data:
                print(f"✓ Data contains '{key}'")
            else:
                print(f"✗ Missing key: '{key}'")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Treasury Management System - Component Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_modules():
        tests_passed += 1
    
    if test_data_generation():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n✓ All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("  streamlit run app.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

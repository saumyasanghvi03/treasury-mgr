"""
Basel III Liquidity Coverage Ratio (LCR) Calculator Module

This module implements the Basel III LCR calculation framework with
sensitivity analysis for regulatory compliance and risk management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class BaselIIILCRCalculator:
    """
    Calculator for Basel III Liquidity Coverage Ratio (LCR).
    
    LCR = High-Quality Liquid Assets (HQLA) / Total Net Cash Outflows over 30 days
    Minimum regulatory requirement: 100%
    """
    
    def __init__(self):
        """Initialize the LCR calculator with default parameters."""
        self.hqla_data = None
        self.cash_outflows = None
        self.cash_inflows = None
        self.lcr_result = None
        
    def classify_hqla(self, assets_df):
        """
        Classify assets into HQLA categories (Level 1, 2A, 2B).
        
        Level 1: Cash, central bank reserves, sovereign bonds (0% haircut)
        Level 2A: High-quality corporate bonds, covered bonds (15% haircut)
        Level 2B: Lower-quality corporate bonds, equities (50% haircut)
        
        Args:
            assets_df (pd.DataFrame): Asset holdings with columns
                ['asset_type', 'market_value', 'hqla_level']
                
        Returns:
            pd.DataFrame: Classified HQLA with haircuts applied
        """
        hqla_df = assets_df.copy()
        
        # Define haircuts by HQLA level
        haircut_map = {
            'LEVEL_1': 0.0,
            'LEVEL_2A': 0.15,
            'LEVEL_2B': 0.50
        }
        
        # Apply haircuts
        hqla_df['haircut_rate'] = hqla_df['hqla_level'].map(haircut_map)
        hqla_df['hqla_value'] = hqla_df['market_value'] * (1 - hqla_df['haircut_rate'])
        
        # Level 2 assets cannot exceed 40% of total HQLA
        level_1_total = hqla_df[hqla_df['hqla_level'] == 'LEVEL_1']['hqla_value'].sum()
        level_2_total = hqla_df[hqla_df['hqla_level'].isin(['LEVEL_2A', 'LEVEL_2B'])]['hqla_value'].sum()
        
        if level_2_total > 0.4 * (level_1_total + level_2_total):
            # Apply cap on Level 2 assets
            adjustment_factor = (0.4 * level_1_total / 0.6) / level_2_total
            mask = hqla_df['hqla_level'].isin(['LEVEL_2A', 'LEVEL_2B'])
            hqla_df.loc[mask, 'hqla_value'] = hqla_df.loc[mask, 'hqla_value'] * adjustment_factor
        
        self.hqla_data = hqla_df
        return hqla_df
    
    def calculate_outflows(self, outflow_df):
        """
        Calculate total net cash outflows over 30-day stress period.
        
        Different outflow categories have different run-off rates:
        - Retail deposits: 3-10% (stable) to 5-40% (less stable)
        - Unsecured wholesale funding: 25-100%
        - Secured funding: 0-100% depending on collateral
        - Committed facilities: 10-100%
        - Derivatives: 100%
        
        Args:
            outflow_df (pd.DataFrame): Expected outflows with columns
                ['category', 'amount', 'run_off_rate']
                
        Returns:
            pd.DataFrame: Calculated outflows with run-off rates applied
        """
        outflow_calc = outflow_df.copy()
        
        # Apply run-off rates
        outflow_calc['stress_outflow'] = outflow_calc['amount'] * outflow_calc['run_off_rate']
        
        self.cash_outflows = outflow_calc
        return outflow_calc
    
    def calculate_inflows(self, inflow_df, cap_at_75_pct_outflows=True):
        """
        Calculate total cash inflows over 30-day period.
        
        Inflows are typically capped at 75% of total outflows to ensure
        a minimum amount of HQLA is maintained.
        
        Args:
            inflow_df (pd.DataFrame): Expected inflows with columns
                ['category', 'amount', 'inflow_rate']
            cap_at_75_pct_outflows (bool): Whether to cap inflows at 75% of outflows
                
        Returns:
            pd.DataFrame: Calculated inflows
        """
        inflow_calc = inflow_df.copy()
        
        # Apply inflow rates
        inflow_calc['stress_inflow'] = inflow_calc['amount'] * inflow_calc['inflow_rate']
        
        # Cap inflows at 75% of outflows if required
        if cap_at_75_pct_outflows and self.cash_outflows is not None:
            total_outflows = self.cash_outflows['stress_outflow'].sum()
            total_inflows = inflow_calc['stress_inflow'].sum()
            
            if total_inflows > 0.75 * total_outflows:
                adjustment_factor = (0.75 * total_outflows) / total_inflows
                inflow_calc['stress_inflow'] = inflow_calc['stress_inflow'] * adjustment_factor
                inflow_calc['capped'] = True
            else:
                inflow_calc['capped'] = False
        
        self.cash_inflows = inflow_calc
        return inflow_calc
    
    def calculate_lcr(self):
        """
        Calculate the Liquidity Coverage Ratio.
        
        LCR = HQLA / Net Cash Outflows
        where Net Cash Outflows = Total Outflows - Min(Total Inflows, 75% of Total Outflows)
        
        Returns:
            dict: LCR calculation results
        """
        if self.hqla_data is None:
            raise ValueError("HQLA data not provided. Call classify_hqla() first.")
        if self.cash_outflows is None:
            raise ValueError("Outflow data not provided. Call calculate_outflows() first.")
        if self.cash_inflows is None:
            raise ValueError("Inflow data not provided. Call calculate_inflows() first.")
        
        # Calculate totals
        total_hqla = self.hqla_data['hqla_value'].sum()
        total_outflows = self.cash_outflows['stress_outflow'].sum()
        total_inflows = self.cash_inflows['stress_inflow'].sum()
        
        # Net cash outflows (minimum 25% of gross outflows per Basel III)
        net_cash_outflows = max(total_outflows - total_inflows, 0.25 * total_outflows)
        
        # Calculate LCR
        lcr_ratio = (total_hqla / net_cash_outflows * 100) if net_cash_outflows > 0 else float('inf')
        
        self.lcr_result = {
            'lcr_ratio': lcr_ratio,
            'total_hqla': total_hqla,
            'total_outflows': total_outflows,
            'total_inflows': total_inflows,
            'net_cash_outflows': net_cash_outflows,
            'surplus_deficit': total_hqla - net_cash_outflows,
            'meets_requirement': lcr_ratio >= 100,
            'calculation_date': datetime.now()
        }
        
        return self.lcr_result
    
    def sensitivity_analysis(self, scenarios):
        """
        Perform sensitivity analysis on LCR under different scenarios.
        
        Args:
            scenarios (dict): Dictionary of scenarios with parameter adjustments
                Example: {
                    'mild_stress': {'hqla_shock': -0.05, 'outflow_increase': 0.10},
                    'severe_stress': {'hqla_shock': -0.15, 'outflow_increase': 0.30}
                }
                
        Returns:
            pd.DataFrame: LCR values under different scenarios
        """
        if self.lcr_result is None:
            raise ValueError("Base LCR not calculated. Call calculate_lcr() first.")
        
        results = []
        base_lcr = self.lcr_result['lcr_ratio']
        
        # Add base case
        results.append({
            'scenario': 'Base Case',
            'lcr_ratio': base_lcr,
            'hqla': self.lcr_result['total_hqla'],
            'net_outflows': self.lcr_result['net_cash_outflows'],
            'change_from_base': 0
        })
        
        # Analyze each scenario
        for scenario_name, adjustments in scenarios.items():
            # Adjust HQLA
            hqla_shock = adjustments.get('hqla_shock', 0)
            adjusted_hqla = self.lcr_result['total_hqla'] * (1 + hqla_shock)
            
            # Adjust outflows
            outflow_increase = adjustments.get('outflow_increase', 0)
            inflow_decrease = adjustments.get('inflow_decrease', 0)
            
            adjusted_outflows = self.lcr_result['total_outflows'] * (1 + outflow_increase)
            adjusted_inflows = self.lcr_result['total_inflows'] * (1 - inflow_decrease)
            
            # Recalculate net outflows
            adjusted_net_outflows = max(
                adjusted_outflows - adjusted_inflows,
                0.25 * adjusted_outflows
            )
            
            # Calculate stressed LCR
            stressed_lcr = (adjusted_hqla / adjusted_net_outflows * 100) if adjusted_net_outflows > 0 else float('inf')
            
            results.append({
                'scenario': scenario_name,
                'lcr_ratio': stressed_lcr,
                'hqla': adjusted_hqla,
                'net_outflows': adjusted_net_outflows,
                'change_from_base': stressed_lcr - base_lcr
            })
        
        return pd.DataFrame(results)
    
    def get_hqla_composition(self):
        """
        Get the composition breakdown of HQLA by level.
        
        Returns:
            pd.DataFrame: HQLA composition summary
        """
        if self.hqla_data is None:
            raise ValueError("HQLA data not available.")
        
        composition = self.hqla_data.groupby('hqla_level').agg({
            'market_value': 'sum',
            'hqla_value': 'sum'
        }).reset_index()
        
        total_hqla = composition['hqla_value'].sum()
        composition['percentage'] = (composition['hqla_value'] / total_hqla * 100)
        
        return composition
    
    def get_outflow_breakdown(self):
        """
        Get the breakdown of cash outflows by category.
        
        Returns:
            pd.DataFrame: Outflow breakdown summary
        """
        if self.cash_outflows is None:
            raise ValueError("Outflow data not available.")
        
        breakdown = self.cash_outflows.groupby('category').agg({
            'amount': 'sum',
            'stress_outflow': 'sum'
        }).reset_index()
        
        total_outflows = breakdown['stress_outflow'].sum()
        breakdown['percentage'] = (breakdown['stress_outflow'] / total_outflows * 100)
        
        return breakdown


def generate_sample_hqla_data():
    """
    Generate sample HQLA data for demonstration.
    
    Returns:
        pd.DataFrame: Sample HQLA holdings
    """
    np.random.seed(42)
    
    assets = [
        {'asset_type': 'Cash', 'market_value': 500_000_000, 'hqla_level': 'LEVEL_1'},
        {'asset_type': 'Central Bank Reserves', 'market_value': 1_000_000_000, 'hqla_level': 'LEVEL_1'},
        {'asset_type': 'Government Bonds', 'market_value': 2_500_000_000, 'hqla_level': 'LEVEL_1'},
        {'asset_type': 'Corporate Bonds (High Quality)', 'market_value': 800_000_000, 'hqla_level': 'LEVEL_2A'},
        {'asset_type': 'Covered Bonds', 'market_value': 600_000_000, 'hqla_level': 'LEVEL_2A'},
        {'asset_type': 'Corporate Bonds (Lower Quality)', 'market_value': 300_000_000, 'hqla_level': 'LEVEL_2B'},
        {'asset_type': 'Equities', 'market_value': 200_000_000, 'hqla_level': 'LEVEL_2B'}
    ]
    
    return pd.DataFrame(assets)


def generate_sample_cashflows():
    """
    Generate sample cash flow data for LCR calculation.
    
    Returns:
        tuple: (outflows_df, inflows_df)
    """
    outflows = [
        {'category': 'Retail Deposits (Stable)', 'amount': 5_000_000_000, 'run_off_rate': 0.03},
        {'category': 'Retail Deposits (Less Stable)', 'amount': 3_000_000_000, 'run_off_rate': 0.10},
        {'category': 'Unsecured Wholesale Funding', 'amount': 2_000_000_000, 'run_off_rate': 0.40},
        {'category': 'Secured Funding', 'amount': 1_500_000_000, 'run_off_rate': 0.25},
        {'category': 'Committed Facilities', 'amount': 1_000_000_000, 'run_off_rate': 0.10},
        {'category': 'Derivatives', 'amount': 500_000_000, 'run_off_rate': 1.0}
    ]
    
    inflows = [
        {'category': 'Retail Inflows', 'amount': 500_000_000, 'inflow_rate': 0.50},
        {'category': 'Wholesale Inflows', 'amount': 800_000_000, 'inflow_rate': 0.50},
        {'category': 'Maturing Assets', 'amount': 600_000_000, 'inflow_rate': 1.0}
    ]
    
    return pd.DataFrame(outflows), pd.DataFrame(inflows)

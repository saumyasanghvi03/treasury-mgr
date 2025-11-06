"""
Asset Liability Management (ALM) GAP Analysis Module

This module implements ALM GAP modeling with Rate Sensitive Assets (RSA)
and Rate Sensitive Liabilities (RSL) analysis across maturity buckets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ALMGAPAnalyzer:
    """
    ALM GAP analyzer for interest rate risk management.
    
    Analyzes the mismatch between rate-sensitive assets and liabilities
    across different time buckets to assess interest rate risk exposure.
    """
    
    # Time bucket thresholds in days
    BUCKET_30_DAYS = 30
    BUCKET_90_DAYS = 90
    BUCKET_180_DAYS = 180
    BUCKET_365_DAYS = 365
    BUCKET_3_YEARS = 1095
    BUCKET_5_YEARS = 1825
    
    def __init__(self):
        """Initialize the ALM GAP analyzer."""
        self.rsa_data = None
        self.rsl_data = None
        self.gap_analysis = None
        self.time_buckets = [
            '0-30 days',
            '31-90 days',
            '91-180 days',
            '181-365 days',
            '1-3 years',
            '3-5 years',
            '5+ years'
        ]
        
    def classify_assets(self, assets_df):
        """
        Classify assets as rate-sensitive and allocate to time buckets.
        
        Args:
            assets_df (pd.DataFrame): Asset holdings with columns
                ['asset_type', 'amount', 'rate_type', 'maturity_days']
                
        Returns:
            pd.DataFrame: Classified rate-sensitive assets
        """
        rsa_df = assets_df[assets_df['rate_type'].isin(['FLOATING', 'REPRICING'])].copy()
        
        # Allocate to time buckets based on maturity/repricing date
        rsa_df['time_bucket'] = rsa_df['maturity_days'].apply(self._assign_time_bucket)
        
        self.rsa_data = rsa_df
        return rsa_df
    
    def classify_liabilities(self, liabilities_df):
        """
        Classify liabilities as rate-sensitive and allocate to time buckets.
        
        Args:
            liabilities_df (pd.DataFrame): Liability holdings with columns
                ['liability_type', 'amount', 'rate_type', 'maturity_days']
                
        Returns:
            pd.DataFrame: Classified rate-sensitive liabilities
        """
        rsl_df = liabilities_df[liabilities_df['rate_type'].isin(['FLOATING', 'REPRICING'])].copy()
        
        # Allocate to time buckets
        rsl_df['time_bucket'] = rsl_df['maturity_days'].apply(self._assign_time_bucket)
        
        self.rsl_data = rsl_df
        return rsl_df
    
    def _assign_time_bucket(self, days):
        """
        Assign a time bucket based on number of days.
        
        Args:
            days (int): Number of days to maturity/repricing
            
        Returns:
            str: Time bucket label
        """
        if days <= self.BUCKET_30_DAYS:
            return '0-30 days'
        elif days <= self.BUCKET_90_DAYS:
            return '31-90 days'
        elif days <= self.BUCKET_180_DAYS:
            return '91-180 days'
        elif days <= self.BUCKET_365_DAYS:
            return '181-365 days'
        elif days <= self.BUCKET_3_YEARS:
            return '1-3 years'
        elif days <= self.BUCKET_5_YEARS:
            return '3-5 years'
        else:
            return '5+ years'
    
    def calculate_gap(self):
        """
        Calculate GAP (RSA - RSL) for each time bucket.
        
        Returns:
            pd.DataFrame: GAP analysis by time bucket
        """
        if self.rsa_data is None or self.rsl_data is None:
            raise ValueError("RSA and RSL data must be provided first.")
        
        # Aggregate RSA by time bucket
        rsa_by_bucket = self.rsa_data.groupby('time_bucket')['amount'].sum()
        
        # Aggregate RSL by time bucket
        rsl_by_bucket = self.rsl_data.groupby('time_bucket')['amount'].sum()
        
        # Create comprehensive dataframe with all time buckets
        gap_df = pd.DataFrame({'time_bucket': self.time_buckets})
        gap_df['rsa'] = gap_df['time_bucket'].map(rsa_by_bucket).fillna(0)
        gap_df['rsl'] = gap_df['time_bucket'].map(rsl_by_bucket).fillna(0)
        
        # Calculate GAP
        gap_df['gap'] = gap_df['rsa'] - gap_df['rsl']
        
        # Calculate cumulative GAP
        gap_df['cumulative_gap'] = gap_df['gap'].cumsum()
        
        # Calculate GAP ratios
        gap_df['gap_ratio'] = np.where(
            gap_df['rsl'] != 0,
            gap_df['gap'] / gap_df['rsl'],
            np.nan
        )
        
        # Calculate RSA/RSL ratio
        gap_df['rsa_rsl_ratio'] = np.where(
            gap_df['rsl'] != 0,
            gap_df['rsa'] / gap_df['rsl'],
            np.nan
        )
        
        self.gap_analysis = gap_df
        return gap_df
    
    def calculate_nii_sensitivity(self, rate_shock_bps=100):
        """
        Calculate Net Interest Income (NII) sensitivity to interest rate changes.
        
        Args:
            rate_shock_bps (int): Interest rate shock in basis points (default 100 bps = 1%)
            
        Returns:
            pd.DataFrame: NII sensitivity analysis
        """
        if self.gap_analysis is None:
            raise ValueError("GAP analysis must be calculated first.")
        
        rate_shock = rate_shock_bps / 10000  # Convert basis points to decimal
        
        sensitivity_df = self.gap_analysis.copy()
        
        # Calculate NII impact for each bucket
        # NII Impact = GAP × Rate Change
        sensitivity_df['nii_impact'] = sensitivity_df['gap'] * rate_shock
        
        # Calculate cumulative NII impact
        sensitivity_df['cumulative_nii_impact'] = sensitivity_df['nii_impact'].cumsum()
        
        # Calculate as percentage of assets
        total_rsa = sensitivity_df['rsa'].sum()
        if total_rsa > 0:
            sensitivity_df['nii_impact_pct'] = (sensitivity_df['cumulative_nii_impact'] / total_rsa * 100)
        else:
            sensitivity_df['nii_impact_pct'] = 0
        
        return sensitivity_df
    
    def duration_gap_analysis(self, assets_duration_df, liabilities_duration_df):
        """
        Perform duration GAP analysis for longer-term interest rate risk.
        
        Args:
            assets_duration_df (pd.DataFrame): Assets with duration
            liabilities_duration_df (pd.DataFrame): Liabilities with duration
            
        Returns:
            dict: Duration GAP analysis results
        """
        # Calculate weighted average duration for assets
        total_assets = assets_duration_df['amount'].sum()
        if total_assets > 0:
            weighted_asset_duration = (
                assets_duration_df['amount'] * assets_duration_df['duration']
            ).sum() / total_assets
        else:
            weighted_asset_duration = 0
        
        # Calculate weighted average duration for liabilities
        total_liabilities = liabilities_duration_df['amount'].sum()
        if total_liabilities > 0:
            weighted_liability_duration = (
                liabilities_duration_df['amount'] * liabilities_duration_df['duration']
            ).sum() / total_liabilities
        else:
            weighted_liability_duration = 0
        
        # Calculate duration GAP
        duration_gap = weighted_asset_duration - weighted_liability_duration
        
        # Calculate equity/asset ratio (simplified)
        equity = total_assets - total_liabilities
        leverage_ratio = total_assets / equity if equity != 0 else float('inf')
        
        # Adjusted duration GAP
        adjusted_duration_gap = duration_gap - (weighted_liability_duration * total_liabilities / total_assets)
        
        results = {
            'asset_duration': weighted_asset_duration,
            'liability_duration': weighted_liability_duration,
            'duration_gap': duration_gap,
            'adjusted_duration_gap': adjusted_duration_gap,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'equity': equity,
            'leverage_ratio': leverage_ratio
        }
        
        return results
    
    def scenario_analysis(self, rate_scenarios):
        """
        Perform scenario analysis under different rate environments.
        
        Args:
            rate_scenarios (dict): Dictionary of rate scenarios
                Example: {
                    'parallel_up': 100,  # +100 bps
                    'parallel_down': -50,  # -50 bps
                    'steepening': {'short': -25, 'long': 50}
                }
                
        Returns:
            pd.DataFrame: Scenario analysis results
        """
        if self.gap_analysis is None:
            raise ValueError("GAP analysis must be calculated first.")
        
        results = []
        
        for scenario_name, rate_change in rate_scenarios.items():
            if isinstance(rate_change, dict):
                # Non-parallel shift (steepening/flattening)
                # Simplified: apply different rates to short vs long buckets
                short_buckets = self.time_buckets[:3]
                long_buckets = self.time_buckets[3:]
                
                short_rate = rate_change.get('short', 0) / 10000
                long_rate = rate_change.get('long', 0) / 10000
                
                # Calculate weighted impact
                short_gap = self.gap_analysis[
                    self.gap_analysis['time_bucket'].isin(short_buckets)
                ]['gap'].sum()
                long_gap = self.gap_analysis[
                    self.gap_analysis['time_bucket'].isin(long_buckets)
                ]['gap'].sum()
                
                nii_impact = (short_gap * short_rate) + (long_gap * long_rate)
                
            else:
                # Parallel shift
                rate_change_decimal = rate_change / 10000
                total_gap = self.gap_analysis['gap'].sum()
                nii_impact = total_gap * rate_change_decimal
            
            results.append({
                'scenario': scenario_name,
                'nii_impact': nii_impact,
                'rate_change_bps': rate_change if isinstance(rate_change, (int, float)) else 'Non-parallel'
            })
        
        return pd.DataFrame(results)
    
    def get_gap_position_summary(self):
        """
        Get a summary of the GAP position.
        
        Returns:
            dict: Summary statistics of GAP position
        """
        if self.gap_analysis is None:
            raise ValueError("GAP analysis must be calculated first.")
        
        total_rsa = self.gap_analysis['rsa'].sum()
        total_rsl = self.gap_analysis['rsl'].sum()
        total_gap = self.gap_analysis['gap'].sum()
        
        # Determine position type
        if total_gap > 0:
            position = 'Asset Sensitive (Positive GAP)'
            rate_scenario = 'Benefits from rising rates'
        elif total_gap < 0:
            position = 'Liability Sensitive (Negative GAP)'
            rate_scenario = 'Benefits from falling rates'
        else:
            position = 'Balanced'
            rate_scenario = 'Neutral to rate changes'
        
        summary = {
            'total_rsa': total_rsa,
            'total_rsl': total_rsl,
            'total_gap': total_gap,
            'rsa_rsl_ratio': total_rsa / total_rsl if total_rsl != 0 else float('inf'),
            'position_type': position,
            'rate_scenario_implication': rate_scenario,
            'largest_gap_bucket': self.gap_analysis.loc[
                self.gap_analysis['gap'].abs().idxmax(), 'time_bucket'
            ],
            'cumulative_gap_1y': self.gap_analysis[
                self.gap_analysis['time_bucket'].isin(self.time_buckets[:4])
            ]['gap'].sum()
        }
        
        return summary


def generate_sample_alm_data():
    """
    Generate sample assets and liabilities data for ALM GAP analysis.
    
    Returns:
        tuple: (assets_df, liabilities_df)
    """
    np.random.seed(42)
    
    # Generate sample assets
    assets = [
        {'asset_type': 'Floating Rate Loans', 'amount': 2_000_000_000, 'rate_type': 'FLOATING', 'maturity_days': 90},
        {'asset_type': 'Variable Mortgages', 'amount': 3_500_000_000, 'rate_type': 'FLOATING', 'maturity_days': 180},
        {'asset_type': 'Corporate Loans (Repricing)', 'amount': 1_500_000_000, 'rate_type': 'REPRICING', 'maturity_days': 365},
        {'asset_type': 'Investment Securities (Floating)', 'amount': 1_000_000_000, 'rate_type': 'FLOATING', 'maturity_days': 730},
        {'asset_type': 'Long-term Floating Loans', 'amount': 2_500_000_000, 'rate_type': 'FLOATING', 'maturity_days': 1825},
        {'asset_type': 'Fixed Rate Loans', 'amount': 5_000_000_000, 'rate_type': 'FIXED', 'maturity_days': 1095}
    ]
    
    # Generate sample liabilities
    liabilities = [
        {'liability_type': 'Floating Rate Deposits', 'amount': 3_000_000_000, 'rate_type': 'FLOATING', 'maturity_days': 30},
        {'liability_type': 'Money Market Accounts', 'amount': 2_500_000_000, 'rate_type': 'FLOATING', 'maturity_days': 90},
        {'liability_type': 'Short-term Borrowings', 'amount': 1_500_000_000, 'rate_type': 'FLOATING', 'maturity_days': 180},
        {'liability_type': 'Medium-term Debt (Repricing)', 'amount': 2_000_000_000, 'rate_type': 'REPRICING', 'maturity_days': 730},
        {'liability_type': 'Long-term Debt (Floating)', 'amount': 1_000_000_000, 'rate_type': 'FLOATING', 'maturity_days': 1825},
        {'liability_type': 'Fixed Rate Deposits', 'amount': 4_000_000_000, 'rate_type': 'FIXED', 'maturity_days': 365}
    ]
    
    return pd.DataFrame(assets), pd.DataFrame(liabilities)


def generate_duration_data():
    """
    Generate sample duration data for assets and liabilities.
    
    Returns:
        tuple: (assets_duration_df, liabilities_duration_df)
    """
    assets_duration = [
        {'asset_type': 'Short-term Securities', 'amount': 1_000_000_000, 'duration': 0.5},
        {'asset_type': 'Medium-term Loans', 'amount': 5_000_000_000, 'duration': 2.5},
        {'asset_type': 'Long-term Mortgages', 'amount': 8_000_000_000, 'duration': 6.0}
    ]
    
    liabilities_duration = [
        {'liability_type': 'Demand Deposits', 'amount': 6_000_000_000, 'duration': 0.1},
        {'liability_type': 'Time Deposits', 'amount': 4_000_000_000, 'duration': 1.5},
        {'liability_type': 'Long-term Debt', 'amount': 2_000_000_000, 'duration': 4.0}
    ]
    
    return pd.DataFrame(assets_duration), pd.DataFrame(liabilities_duration)

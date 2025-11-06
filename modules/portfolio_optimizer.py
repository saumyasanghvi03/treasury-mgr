"""
Investment Portfolio Optimizer Module

This module implements portfolio optimization using linear programming
to maximize returns while respecting risk and regulatory constraints.
"""

import pandas as pd
import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value


class PortfolioOptimizer:
    """
    Portfolio optimizer using linear programming.
    
    Optimizes asset allocation to maximize expected returns subject to
    risk constraints, diversification requirements, and regulatory limits.
    """
    
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.problem = None
        self.allocation_vars = None
        self.optimal_allocation = None
        
    def optimize_max_return(self, assets_df, total_portfolio_value, constraints=None):
        """
        Optimize portfolio to maximize expected returns subject to constraints.
        
        Args:
            assets_df (pd.DataFrame): Asset data with columns
                ['asset', 'expected_return', 'risk', 'min_allocation', 'max_allocation']
            total_portfolio_value (float): Total portfolio value to allocate
            constraints (dict): Additional constraints
                Example: {
                    'max_single_asset': 0.30,
                    'min_diversification': 5,
                    'max_total_risk': 0.15
                }
                
        Returns:
            dict: Optimization results with allocations and metrics
        """
        # Create the LP problem
        prob = LpProblem("Portfolio_Optimization", LpMaximize)
        
        # Create allocation variables for each asset (as percentage of portfolio)
        asset_vars = {}
        for asset in assets_df['asset']:
            asset_vars[asset] = LpVariable(f"allocation_{asset}", lowBound=0, upBound=1)
        
        # Objective: Maximize expected return
        prob += lpSum([
            asset_vars[row['asset']] * row['expected_return'] 
            for _, row in assets_df.iterrows()
        ]), "Total_Expected_Return"
        
        # Constraint: Sum of allocations = 1 (100%)
        prob += lpSum([asset_vars[asset] for asset in asset_vars]) == 1, "Total_Allocation"
        
        # Individual asset allocation constraints
        for _, row in assets_df.iterrows():
            asset = row['asset']
            
            # Minimum allocation
            if 'min_allocation' in row and pd.notna(row['min_allocation']):
                prob += asset_vars[asset] >= row['min_allocation'], f"Min_{asset}"
            
            # Maximum allocation
            if 'max_allocation' in row and pd.notna(row['max_allocation']):
                prob += asset_vars[asset] <= row['max_allocation'], f"Max_{asset}"
        
        # Additional constraints
        if constraints:
            # Maximum single asset allocation
            if 'max_single_asset' in constraints:
                for asset in asset_vars:
                    prob += asset_vars[asset] <= constraints['max_single_asset'], f"MaxSingle_{asset}"
            
            # Risk constraint (simplified - uses individual asset risk as proxy)
            if 'max_total_risk' in constraints:
                prob += lpSum([
                    asset_vars[row['asset']] * row['risk'] 
                    for _, row in assets_df.iterrows()
                ]) <= constraints['max_total_risk'], "Max_Portfolio_Risk"
        
        # Solve the problem
        prob.solve()
        
        # Extract results
        if LpStatus[prob.status] == 'Optimal':
            allocations = {}
            for asset in asset_vars:
                allocation_pct = value(asset_vars[asset])
                allocations[asset] = {
                    'percentage': allocation_pct,
                    'dollar_amount': allocation_pct * total_portfolio_value
                }
            
            # Calculate portfolio metrics
            expected_return = sum([
                allocations[row['asset']]['percentage'] * row['expected_return']
                for _, row in assets_df.iterrows()
            ])
            
            portfolio_risk = sum([
                allocations[row['asset']]['percentage'] * row['risk']
                for _, row in assets_df.iterrows()
            ])
            
            sharpe_ratio = (expected_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            results = {
                'status': 'Optimal',
                'allocations': allocations,
                'expected_return': expected_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'total_value': total_portfolio_value,
                'objective_value': value(prob.objective)
            }
        else:
            results = {
                'status': LpStatus[prob.status],
                'message': 'Optimization did not converge to optimal solution'
            }
        
        self.problem = prob
        self.allocation_vars = asset_vars
        self.optimal_allocation = results if results['status'] == 'Optimal' else None
        
        return results
    
    def optimize_min_risk(self, assets_df, total_portfolio_value, target_return=None, constraints=None):
        """
        Optimize portfolio to minimize risk for a given target return.
        
        Args:
            assets_df (pd.DataFrame): Asset data
            total_portfolio_value (float): Total portfolio value
            target_return (float): Minimum target return (optional)
            constraints (dict): Additional constraints
            
        Returns:
            dict: Optimization results
        """
        # Create the LP problem
        prob = LpProblem("Portfolio_Risk_Minimization", LpMinimize)
        
        # Create allocation variables
        asset_vars = {}
        for asset in assets_df['asset']:
            asset_vars[asset] = LpVariable(f"allocation_{asset}", lowBound=0, upBound=1)
        
        # Objective: Minimize risk
        prob += lpSum([
            asset_vars[row['asset']] * row['risk'] 
            for _, row in assets_df.iterrows()
        ]), "Total_Portfolio_Risk"
        
        # Constraint: Sum of allocations = 1
        prob += lpSum([asset_vars[asset] for asset in asset_vars]) == 1, "Total_Allocation"
        
        # Target return constraint
        if target_return is not None:
            prob += lpSum([
                asset_vars[row['asset']] * row['expected_return'] 
                for _, row in assets_df.iterrows()
            ]) >= target_return, "Target_Return"
        
        # Individual asset constraints
        for _, row in assets_df.iterrows():
            asset = row['asset']
            
            if 'min_allocation' in row and pd.notna(row['min_allocation']):
                prob += asset_vars[asset] >= row['min_allocation'], f"Min_{asset}"
            
            if 'max_allocation' in row and pd.notna(row['max_allocation']):
                prob += asset_vars[asset] <= row['max_allocation'], f"Max_{asset}"
        
        # Additional constraints
        if constraints and 'max_single_asset' in constraints:
            for asset in asset_vars:
                prob += asset_vars[asset] <= constraints['max_single_asset'], f"MaxSingle_{asset}"
        
        # Solve
        prob.solve()
        
        # Extract results
        if LpStatus[prob.status] == 'Optimal':
            allocations = {}
            for asset in asset_vars:
                allocation_pct = value(asset_vars[asset])
                allocations[asset] = {
                    'percentage': allocation_pct,
                    'dollar_amount': allocation_pct * total_portfolio_value
                }
            
            expected_return = sum([
                allocations[row['asset']]['percentage'] * row['expected_return']
                for _, row in assets_df.iterrows()
            ])
            
            portfolio_risk = sum([
                allocations[row['asset']]['percentage'] * row['risk']
                for _, row in assets_df.iterrows()
            ])
            
            sharpe_ratio = (expected_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            results = {
                'status': 'Optimal',
                'allocations': allocations,
                'expected_return': expected_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'total_value': total_portfolio_value
            }
        else:
            results = {
                'status': LpStatus[prob.status],
                'message': 'Optimization did not converge'
            }
        
        return results
    
    def efficient_frontier(self, assets_df, total_portfolio_value, num_points=10):
        """
        Calculate the efficient frontier by optimizing across different return levels.
        
        Args:
            assets_df (pd.DataFrame): Asset data
            total_portfolio_value (float): Portfolio value
            num_points (int): Number of points on the frontier
            
        Returns:
            pd.DataFrame: Efficient frontier points
        """
        # Calculate return range
        min_return = assets_df['expected_return'].min()
        max_return = assets_df['expected_return'].max()
        
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_points = []
        
        for target_return in target_returns:
            result = self.optimize_min_risk(
                assets_df, 
                total_portfolio_value, 
                target_return=target_return
            )
            
            if result['status'] == 'Optimal':
                frontier_points.append({
                    'target_return': target_return,
                    'portfolio_return': result['expected_return'],
                    'portfolio_risk': result['portfolio_risk'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
        
        return pd.DataFrame(frontier_points)
    
    def rebalance_portfolio(self, current_holdings, target_allocations, transaction_cost=0.001):
        """
        Calculate rebalancing trades needed to achieve target allocations.
        
        Args:
            current_holdings (dict): Current holdings {asset: dollar_amount}
            target_allocations (dict): Target allocations from optimization
            transaction_cost (float): Transaction cost as percentage
            
        Returns:
            pd.DataFrame: Rebalancing trades
        """
        total_value = sum(current_holdings.values())
        
        trades = []
        
        for asset in target_allocations:
            current_value = current_holdings.get(asset, 0)
            target_value = target_allocations[asset]['dollar_amount']
            
            trade_amount = target_value - current_value
            
            if abs(trade_amount) > 0.01:  # Ignore negligible trades
                trade_cost = abs(trade_amount) * transaction_cost
                
                trades.append({
                    'asset': asset,
                    'current_value': current_value,
                    'target_value': target_value,
                    'trade_amount': trade_amount,
                    'trade_type': 'BUY' if trade_amount > 0 else 'SELL',
                    'transaction_cost': trade_cost
                })
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            total_transaction_cost = trades_df['transaction_cost'].sum()
            trades_df['total_transaction_cost'] = total_transaction_cost
        
        return trades_df
    
    def constrained_optimization_with_sectors(self, assets_df, sector_constraints, 
                                             total_portfolio_value):
        """
        Optimize with sector-level constraints.
        
        Args:
            assets_df (pd.DataFrame): Assets with 'sector' column
            sector_constraints (dict): Sector limits
                Example: {'Technology': {'min': 0.10, 'max': 0.30}}
            total_portfolio_value (float): Portfolio value
            
        Returns:
            dict: Optimization results
        """
        prob = LpProblem("Portfolio_Sector_Optimization", LpMaximize)
        
        asset_vars = {}
        for asset in assets_df['asset']:
            asset_vars[asset] = LpVariable(f"allocation_{asset}", lowBound=0, upBound=1)
        
        # Objective: Maximize expected return
        prob += lpSum([
            asset_vars[row['asset']] * row['expected_return'] 
            for _, row in assets_df.iterrows()
        ]), "Total_Expected_Return"
        
        # Total allocation constraint
        prob += lpSum([asset_vars[asset] for asset in asset_vars]) == 1, "Total_Allocation"
        
        # Sector constraints
        for sector, limits in sector_constraints.items():
            sector_assets = assets_df[assets_df['sector'] == sector]['asset'].tolist()
            
            if sector_assets:
                sector_allocation = lpSum([asset_vars[asset] for asset in sector_assets])
                
                if 'min' in limits:
                    prob += sector_allocation >= limits['min'], f"Min_Sector_{sector}"
                
                if 'max' in limits:
                    prob += sector_allocation <= limits['max'], f"Max_Sector_{sector}"
        
        # Solve
        prob.solve()
        
        if LpStatus[prob.status] == 'Optimal':
            allocations = {}
            for asset in asset_vars:
                allocation_pct = value(asset_vars[asset])
                allocations[asset] = {
                    'percentage': allocation_pct,
                    'dollar_amount': allocation_pct * total_portfolio_value
                }
            
            # Calculate sector allocations
            sector_allocations = {}
            for sector in sector_constraints.keys():
                sector_assets = assets_df[assets_df['sector'] == sector]['asset'].tolist()
                sector_total = sum([allocations[asset]['percentage'] for asset in sector_assets if asset in allocations])
                sector_allocations[sector] = sector_total
            
            expected_return = sum([
                allocations[row['asset']]['percentage'] * row['expected_return']
                for _, row in assets_df.iterrows() if row['asset'] in allocations
            ])
            
            results = {
                'status': 'Optimal',
                'allocations': allocations,
                'sector_allocations': sector_allocations,
                'expected_return': expected_return,
                'total_value': total_portfolio_value
            }
        else:
            results = {
                'status': LpStatus[prob.status],
                'message': 'Optimization failed'
            }
        
        return results


def generate_sample_assets():
    """
    Generate sample asset data for portfolio optimization.
    
    Returns:
        pd.DataFrame: Sample asset universe
    """
    assets = [
        {'asset': 'US_Large_Cap_Equity', 'expected_return': 0.10, 'risk': 0.18, 
         'sector': 'Equity', 'min_allocation': 0.05, 'max_allocation': 0.40},
        {'asset': 'US_Small_Cap_Equity', 'expected_return': 0.12, 'risk': 0.25, 
         'sector': 'Equity', 'min_allocation': 0.00, 'max_allocation': 0.20},
        {'asset': 'International_Equity', 'expected_return': 0.09, 'risk': 0.22, 
         'sector': 'Equity', 'min_allocation': 0.00, 'max_allocation': 0.30},
        {'asset': 'Government_Bonds', 'expected_return': 0.04, 'risk': 0.05, 
         'sector': 'Fixed_Income', 'min_allocation': 0.10, 'max_allocation': 0.50},
        {'asset': 'Corporate_Bonds', 'expected_return': 0.06, 'risk': 0.08, 
         'sector': 'Fixed_Income', 'min_allocation': 0.05, 'max_allocation': 0.40},
        {'asset': 'High_Yield_Bonds', 'expected_return': 0.08, 'risk': 0.15, 
         'sector': 'Fixed_Income', 'min_allocation': 0.00, 'max_allocation': 0.15},
        {'asset': 'Real_Estate', 'expected_return': 0.09, 'risk': 0.20, 
         'sector': 'Alternatives', 'min_allocation': 0.00, 'max_allocation': 0.15},
        {'asset': 'Commodities', 'expected_return': 0.07, 'risk': 0.25, 
         'sector': 'Alternatives', 'min_allocation': 0.00, 'max_allocation': 0.10},
        {'asset': 'Cash', 'expected_return': 0.02, 'risk': 0.01, 
         'sector': 'Cash', 'min_allocation': 0.02, 'max_allocation': 0.20}
    ]
    
    return pd.DataFrame(assets)


def define_sector_constraints():
    """
    Define standard sector allocation constraints.
    
    Returns:
        dict: Sector constraints
    """
    constraints = {
        'Equity': {'min': 0.20, 'max': 0.70},
        'Fixed_Income': {'min': 0.20, 'max': 0.60},
        'Alternatives': {'min': 0.00, 'max': 0.25},
        'Cash': {'min': 0.02, 'max': 0.20}
    }
    
    return constraints

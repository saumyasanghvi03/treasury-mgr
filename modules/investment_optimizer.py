import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

def show():
    st.header("💎 Investment Portfolio Optimizer")
    st.write("Optimize portfolio allocation using linear programming")
    
    # Get data
    data = st.session_state.data
    universe_df = data['investment_universe'].copy()
    
    # Optimization parameters
    st.subheader("Optimization Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_budget = st.number_input(
            "Total Investment Budget ($M)",
            min_value=10.0,
            max_value=200.0,
            value=100.0,
            step=5.0
        ) * 1_000_000
    
    with col2:
        objective = st.selectbox(
            "Optimization Objective",
            ["Maximize Return", "Minimize Risk", "Risk-Adjusted Return"]
        )
    
    with col3:
        min_liquidity = st.slider(
            "Min Liquidity Score",
            0, 100, 70,
            help="Minimum acceptable liquidity score"
        )
    
    # Constraints
    st.subheader("Portfolio Constraints")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_single_position = st.slider(
            "Max Single Position (%)",
            10, 50, 30,
            help="Maximum % in any single instrument"
        )
    
    with col2:
        min_diversification = st.slider(
            "Min # of Instruments",
            3, 9, 5,
            help="Minimum number of different instruments"
        )
    
    with col3:
        target_return = st.number_input(
            "Target Return (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        ) / 100
    
    # Run optimization
    if st.button("🚀 Optimize Portfolio", type="primary"):
        with st.spinner("Running optimization..."):
            result = optimize_portfolio(
                universe_df,
                total_budget,
                objective,
                max_single_position/100,
                min_diversification,
                target_return,
                min_liquidity
            )
        
        if result['status'] == 'Optimal':
            st.success("✓ Optimization completed successfully!")
            
            # Display results
            st.divider()
            st.subheader("Optimal Portfolio Allocation")
            
            allocation_df = result['allocation']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Return",
                    f"{result['expected_return']*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Portfolio Risk",
                    f"{result['portfolio_risk']*100:.2f}%"
                )
            
            with col3:
                sharpe = result['expected_return'] / result['portfolio_risk'] if result['portfolio_risk'] > 0 else 0
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}"
                )
            
            with col4:
                st.metric(
                    "Avg Liquidity Score",
                    f"{result['avg_liquidity']:.1f}"
                )
            
            # Allocation visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=allocation_df['instrument'],
                    values=allocation_df['allocation'],
                    hole=0.4,
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<br>Percent: %{percent}<extra></extra>'
                )])
                
                fig.update_layout(
                    title="Portfolio Allocation",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Allocation Details**")
                
                display_df = allocation_df.copy()
                display_df['percent'] = (display_df['allocation'] / total_budget) * 100
                
                st.dataframe(
                    display_df[['instrument', 'allocation', 'percent']].style.format({
                        'allocation': '${:,.0f}',
                        'percent': '{:.1f}%'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                
                st.metric("Number of Instruments", len(allocation_df))
                st.metric("Total Allocated", f"${allocation_df['allocation'].sum()/1e6:.2f}M")
            
            # Risk-Return Profile
            st.divider()
            st.subheader("Risk-Return Profile")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk contribution
                risk_contrib = allocation_df.copy()
                risk_contrib['risk_contribution'] = (
                    risk_contrib['allocation'] * universe_df.set_index('instrument').loc[risk_contrib['instrument'], 'risk'].values
                )
                
                fig = go.Figure(go.Bar(
                    x=risk_contrib['instrument'],
                    y=risk_contrib['risk_contribution'],
                    marker_color='coral',
                    text=risk_contrib['risk_contribution'].apply(lambda x: f"${x/1e6:.2f}M"),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Risk Contribution by Instrument",
                    xaxis_title="Instrument",
                    yaxis_title="Risk Contribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Return contribution
                return_contrib = allocation_df.copy()
                return_contrib['return_contribution'] = (
                    return_contrib['allocation'] * universe_df.set_index('instrument').loc[return_contrib['instrument'], 'expected_return'].values
                )
                
                fig = go.Figure(go.Bar(
                    x=return_contrib['instrument'],
                    y=return_contrib['return_contribution'],
                    marker_color='lightgreen',
                    text=return_contrib['return_contribution'].apply(lambda x: f"${x/1e6:.2f}M"),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Return Contribution by Instrument",
                    xaxis_title="Instrument",
                    yaxis_title="Return Contribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Efficient Frontier
            st.divider()
            st.subheader("Efficient Frontier Analysis")
            
            # Generate efficient frontier
            frontier_points = generate_efficient_frontier(universe_df, total_budget, max_single_position/100)
            
            fig = go.Figure()
            
            # Efficient frontier
            fig.add_trace(go.Scatter(
                x=frontier_points['risk'],
                y=frontier_points['return'],
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Current portfolio
            fig.add_trace(go.Scatter(
                x=[result['portfolio_risk']],
                y=[result['expected_return']],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            # Individual assets
            fig.add_trace(go.Scatter(
                x=universe_df['risk'],
                y=universe_df['expected_return'],
                mode='markers',
                name='Individual Assets',
                marker=dict(size=10, color='lightblue'),
                text=universe_df['instrument'],
                hovertemplate='<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Risk (Volatility)",
                yaxis_title="Expected Return",
                height=500,
                hovermode='closest',
                xaxis=dict(tickformat='.1%'),
                yaxis=dict(tickformat='.1%')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Efficient Frontier**: Shows the set of optimal portfolios that offer the highest 
            expected return for a given level of risk. The red star indicates your optimized portfolio.
            """)
            
            # Constraint Analysis
            st.divider()
            st.subheader("Constraint Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Constraint Status**")
                
                constraints = pd.DataFrame({
                    'Constraint': [
                        'Budget',
                        'Max Single Position',
                        'Min Instruments',
                        'Target Return',
                        'Min Liquidity'
                    ],
                    'Required': [
                        f"${total_budget/1e6:.1f}M",
                        f"{max_single_position}%",
                        f"{min_diversification}",
                        f"{target_return*100:.1f}%",
                        f"{min_liquidity}"
                    ],
                    'Actual': [
                        f"${allocation_df['allocation'].sum()/1e6:.1f}M",
                        f"{(allocation_df['allocation'].max()/total_budget)*100:.1f}%",
                        f"{len(allocation_df)}",
                        f"{result['expected_return']*100:.1f}%",
                        f"{result['avg_liquidity']:.1f}"
                    ],
                    'Status': [
                        '✓',
                        '✓' if allocation_df['allocation'].max()/total_budget <= max_single_position/100 else '✗',
                        '✓' if len(allocation_df) >= min_diversification else '✗',
                        '✓' if result['expected_return'] >= target_return else '✗',
                        '✓' if result['avg_liquidity'] >= min_liquidity else '✗'
                    ]
                })
                
                st.dataframe(constraints, hide_index=True, use_container_width=True)
            
            with col2:
                st.write("**Portfolio Characteristics**")
                
                chars = pd.DataFrame({
                    'Metric': [
                        'Return/Risk Ratio',
                        'Diversification Score',
                        'Avg Instrument Size',
                        'Liquidity Score',
                        'Concentration (HHI)'
                    ],
                    'Value': [
                        f"{sharpe:.2f}",
                        f"{(len(allocation_df)/len(universe_df))*100:.1f}%",
                        f"${(allocation_df['allocation'].mean()/1e6):.2f}M",
                        f"{result['avg_liquidity']:.1f}",
                        f"{calculate_hhi(allocation_df['allocation']):.0f}"
                    ]
                })
                
                st.dataframe(chars, hide_index=True, use_container_width=True)
        
        else:
            st.error(f"❌ Optimization failed: {result['status']}")
            st.write("Try relaxing some constraints or adjusting parameters.")
    
    # Investment Universe
    st.divider()
    st.subheader("Available Investment Universe")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scatter plot of risk vs return
        fig = px.scatter(
            universe_df,
            x='risk',
            y='expected_return',
            size='max_investment',
            color='liquidity_score',
            hover_name='instrument',
            labels={
                'risk': 'Risk (Volatility)',
                'expected_return': 'Expected Return',
                'liquidity_score': 'Liquidity Score'
            },
            title='Investment Universe: Risk vs Return'
        )
        
        fig.update_layout(
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Universe Statistics**")
        
        st.metric("Number of Instruments", len(universe_df))
        st.metric("Avg Expected Return", f"{universe_df['expected_return'].mean()*100:.2f}%")
        st.metric("Avg Risk", f"{universe_df['risk'].mean()*100:.2f}%")
        st.metric("Total Capacity", f"${universe_df['max_investment'].sum()/1e6:.0f}M")
    
    # Detailed universe table
    with st.expander("View Detailed Investment Universe"):
        display_df = universe_df.copy()
        st.dataframe(
            display_df.style.format({
                'expected_return': '{:.2%}',
                'risk': '{:.2%}',
                'min_investment': '${:,.0f}',
                'max_investment': '${:,.0f}'
            }),
            hide_index=True,
            use_container_width=True
        )

def optimize_portfolio(universe_df, budget, objective, max_position, min_instruments, 
                       target_return, min_liquidity):
    """Optimize portfolio using linear programming"""
    
    # Create LP problem
    if objective == "Maximize Return":
        prob = LpProblem("Portfolio_Optimization", LpMaximize)
    else:
        prob = LpProblem("Portfolio_Optimization", LpMinimize)
    
    # Decision variables: amount to invest in each instrument
    instruments = universe_df['instrument'].tolist()
    invest_vars = LpVariable.dicts("invest", instruments, lowBound=0)
    
    # Binary variables for instrument selection
    select_vars = LpVariable.dicts("select", instruments, cat='Binary')
    
    # Objective function
    if objective == "Maximize Return":
        prob += lpSum([invest_vars[i] * universe_df[universe_df['instrument']==i]['expected_return'].values[0] 
                      for i in instruments])
    elif objective == "Minimize Risk":
        prob += lpSum([invest_vars[i] * universe_df[universe_df['instrument']==i]['risk'].values[0] 
                      for i in instruments])
    else:  # Risk-Adjusted Return
        prob += lpSum([invest_vars[i] * (universe_df[universe_df['instrument']==i]['expected_return'].values[0] / 
                                        (universe_df[universe_df['instrument']==i]['risk'].values[0] + 0.001))
                      for i in instruments])
    
    # Constraints
    # 1. Budget constraint
    prob += lpSum([invest_vars[i] for i in instruments]) == budget
    
    # 2. Individual position limits
    for i in instruments:
        inst_data = universe_df[universe_df['instrument']==i].iloc[0]
        prob += invest_vars[i] <= min(inst_data['max_investment'], budget * max_position)
        prob += invest_vars[i] >= select_vars[i] * inst_data['min_investment']
        prob += invest_vars[i] <= select_vars[i] * inst_data['max_investment']
    
    # 3. Minimum number of instruments
    prob += lpSum([select_vars[i] for i in instruments]) >= min_instruments
    
    # 4. Target return constraint (for risk minimization)
    if objective == "Minimize Risk":
        prob += lpSum([invest_vars[i] * universe_df[universe_df['instrument']==i]['expected_return'].values[0] 
                      for i in instruments]) >= budget * target_return
    
    # 5. Liquidity constraint
    prob += (lpSum([invest_vars[i] * universe_df[universe_df['instrument']==i]['liquidity_score'].values[0] 
                   for i in instruments]) / budget) >= min_liquidity
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract results
    if LpStatus[prob.status] == 'Optimal':
        allocation = []
        total_return = 0
        total_risk = 0
        total_liquidity = 0
        total_allocated = 0
        
        for i in instruments:
            amount = invest_vars[i].varValue
            if amount and amount > 0.01:  # Filter out very small allocations
                inst_data = universe_df[universe_df['instrument']==i].iloc[0]
                allocation.append({
                    'instrument': i,
                    'allocation': amount
                })
                total_return += amount * inst_data['expected_return']
                total_risk += amount * inst_data['risk']
                total_liquidity += amount * inst_data['liquidity_score']
                total_allocated += amount
        
        allocation_df = pd.DataFrame(allocation)
        
        return {
            'status': 'Optimal',
            'allocation': allocation_df,
            'expected_return': total_return / total_allocated if total_allocated > 0 else 0,
            'portfolio_risk': total_risk / total_allocated if total_allocated > 0 else 0,
            'avg_liquidity': total_liquidity / total_allocated if total_allocated > 0 else 0
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'allocation': pd.DataFrame(),
            'expected_return': 0,
            'portfolio_risk': 0,
            'avg_liquidity': 0
        }

def generate_efficient_frontier(universe_df, budget, max_position, n_points=15):
    """Generate points on the efficient frontier"""
    
    min_return = universe_df['expected_return'].min()
    max_return = universe_df['expected_return'].max()
    
    target_returns = np.linspace(min_return, max_return * 0.8, n_points)
    
    frontier_points = {'return': [], 'risk': []}
    
    for target_ret in target_returns:
        result = optimize_portfolio(
            universe_df,
            budget,
            "Minimize Risk",
            max_position,
            3,  # min instruments
            target_ret,
            0  # no liquidity constraint for frontier
        )
        
        if result['status'] == 'Optimal':
            frontier_points['return'].append(result['expected_return'])
            frontier_points['risk'].append(result['portfolio_risk'])
    
    return pd.DataFrame(frontier_points)

def calculate_hhi(allocations):
    """Calculate Herfindahl-Hirschman Index for concentration"""
    total = allocations.sum()
    shares = (allocations / total) ** 2
    return shares.sum() * 10000

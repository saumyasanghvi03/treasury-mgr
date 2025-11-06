import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def show():
    st.header("🏦 Basel III Liquidity Coverage Ratio (LCR) Analytics")
    st.write("Monitor and analyze regulatory liquidity requirements under Basel III framework")
    
    # Get data
    data = st.session_state.data
    lcr_data = data['lcr_components']
    
    # Overview metrics
    st.subheader("LCR Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        lcr_ratio = lcr_data['lcr_ratio']
        st.metric(
            "LCR Ratio",
            f"{lcr_ratio:.1f}%",
            delta="Compliant ✓" if lcr_ratio >= 100 else "Below Requirement ✗",
            delta_color="normal" if lcr_ratio >= 100 else "inverse"
        )
    
    with col2:
        st.metric(
            "Total HQLA",
            f"${lcr_data['total_hqla']/1e6:.1f}M"
        )
    
    with col3:
        st.metric(
            "Net Cash Outflows",
            f"${lcr_data['total_net_outflows']/1e6:.1f}M"
        )
    
    with col4:
        buffer = lcr_data['total_hqla'] - lcr_data['total_net_outflows']
        st.metric(
            "Liquidity Buffer",
            f"${buffer/1e6:.1f}M",
            delta=f"{(buffer/lcr_data['total_net_outflows']*100):.1f}% above minimum"
        )
    
    # LCR Formula
    st.divider()
    st.subheader("LCR Calculation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Basel III LCR Formula
        
        $$
        LCR = \\frac{\\text{High Quality Liquid Assets (HQLA)}}{\\text{Total Net Cash Outflows over 30 days}} \\times 100
        $$
        
        **Regulatory Requirement:** LCR ≥ 100%
        
        The LCR ensures banks have sufficient high-quality liquid assets to survive a 30-day 
        stressed funding scenario.
        """)
    
    with col2:
        # LCR Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=lcr_ratio,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 100, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 100], 'color': "lightcoral"},
                    {'range': [100, 150], 'color': "lightgreen"},
                    {'range': [150, 200], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        
        fig.update_layout(
            title="LCR Compliance Gauge",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # HQLA Breakdown
    st.divider()
    st.subheader("High Quality Liquid Assets (HQLA) Composition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # HQLA breakdown
        hqla_breakdown = pd.DataFrame({
            'Category': ['Level 1', 'Level 2A', 'Level 2B'],
            'Amount': [
                lcr_data['hqla_level1'],
                lcr_data['hqla_level2a'],
                lcr_data['hqla_level2b']
            ],
            'Haircut': [0, 15, 50],
            'Description': [
                'Cash, central bank reserves, sovereign debt',
                'High-quality corporate bonds, covered bonds',
                'Lower-rated corporate bonds, equities'
            ]
        })
        
        hqla_breakdown['Weight'] = hqla_breakdown['Amount'] / hqla_breakdown['Amount'].sum() * 100
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=hqla_breakdown['Category'],
            values=hqla_breakdown['Amount'],
            hole=0.4,
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
        )])
        
        fig.update_layout(
            title="HQLA Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**HQLA Details**")
        st.dataframe(
            hqla_breakdown[['Category', 'Amount', 'Haircut', 'Weight']].style.format({
                'Amount': '${:,.0f}',
                'Haircut': '{:.0f}%',
                'Weight': '{:.1f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        st.info("""
        **HQLA Categories:**
        
        - **Level 1**: No haircut, includes cash and high-quality government securities
        - **Level 2A**: 15% haircut, high-quality corporate bonds
        - **Level 2B**: 50% haircut, lower-rated securities
        """)
    
    # Cash Flow Analysis
    st.divider()
    st.subheader("30-Day Stress Scenario Analysis")
    
    # Generate sample stress scenario
    stress_scenario = generate_stress_scenario()
    
    tab1, tab2, tab3 = st.tabs(["Cash Flows", "Concentration Analysis", "Scenario Testing"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cash Inflows (30-day stress)**")
            inflows_df = pd.DataFrame({
                'Source': [
                    'Maturing Assets',
                    'Secured Lending',
                    'Committed Facilities',
                    'Other Inflows'
                ],
                'Contractual': [15_000_000, 8_000_000, 5_000_000, 3_000_000],
                'Stressed': [12_000_000, 6_000_000, 2_500_000, 2_000_000]
            })
            
            inflows_df['Run-off Rate'] = ((inflows_df['Contractual'] - inflows_df['Stressed']) / 
                                          inflows_df['Contractual'] * 100)
            
            st.dataframe(
                inflows_df.style.format({
                    'Contractual': '${:,.0f}',
                    'Stressed': '${:,.0f}',
                    'Run-off Rate': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
            
            st.metric("Total Stressed Inflows", f"${inflows_df['Stressed'].sum()/1e6:.1f}M")
        
        with col2:
            st.write("**Cash Outflows (30-day stress)**")
            outflows_df = pd.DataFrame({
                'Source': [
                    'Retail Deposits',
                    'Wholesale Funding',
                    'Secured Funding',
                    'Derivatives',
                    'Other Outflows'
                ],
                'Contractual': [50_000_000, 30_000_000, 15_000_000, 8_000_000, 7_000_000],
                'Stressed': [10_000_000, 25_000_000, 13_000_000, 7_500_000, 6_500_000]
            })
            
            outflows_df['Run-off Rate'] = (outflows_df['Stressed'] / outflows_df['Contractual'] * 100)
            
            st.dataframe(
                outflows_df.style.format({
                    'Contractual': '${:,.0f}',
                    'Stressed': '${:,.0f}',
                    'Run-off Rate': '{:.1f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
            
            st.metric("Total Stressed Outflows", f"${outflows_df['Stressed'].sum()/1e6:.1f}M")
        
        # Net outflows
        net_outflows = outflows_df['Stressed'].sum() - inflows_df['Stressed'].sum()
        st.metric(
            "Net Cash Outflows (30-day)",
            f"${net_outflows/1e6:.1f}M",
            help="Total stressed outflows minus capped inflows"
        )
    
    with tab2:
        st.write("**Funding Concentration Risk**")
        
        # Sample concentration data
        funding_sources = pd.DataFrame({
            'Counterparty Type': [
                'Top 10 Depositors',
                'Money Market Funds',
                'Corporate Treasuries',
                'Retail Deposits',
                'Interbank Market',
                'Central Bank'
            ],
            'Funding Amount': [25_000_000, 18_000_000, 15_000_000, 40_000_000, 12_000_000, 20_000_000],
            'Maturity < 30 days': [8_000_000, 15_000_000, 10_000_000, 5_000_000, 10_000_000, 0]
        })
        
        funding_sources['Concentration %'] = (funding_sources['Funding Amount'] / 
                                              funding_sources['Funding Amount'].sum() * 100)
        
        # Bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total Funding',
            y=funding_sources['Counterparty Type'],
            x=funding_sources['Funding Amount'],
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Maturing < 30 days',
            y=funding_sources['Counterparty Type'],
            x=funding_sources['Maturity < 30 days'],
            orientation='h',
            marker_color='coral'
        ))
        
        fig.update_layout(
            title="Funding Source Concentration",
            xaxis_title="Amount ($)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            funding_sources.style.format({
                'Funding Amount': '${:,.0f}',
                'Maturity < 30 days': '${:,.0f}',
                'Concentration %': '{:.1f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with tab3:
        st.write("**LCR Scenario Testing**")
        
        st.write("Adjust parameters to test different stress scenarios:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hqla_shock = st.slider("HQLA Haircut (%)", -50, 0, -10)
            outflow_shock = st.slider("Outflow Multiplier (%)", 100, 200, 120)
        
        with col2:
            inflow_shock = st.slider("Inflow Reduction (%)", 0, 100, 25)
            deposit_runoff = st.slider("Retail Deposit Run-off (%)", 0, 50, 10)
        
        # Calculate stressed LCR
        stressed_hqla = lcr_data['total_hqla'] * (1 + hqla_shock/100)
        stressed_outflows = lcr_data['total_net_outflows'] * (outflow_shock/100)
        stressed_lcr = (stressed_hqla / stressed_outflows) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Base LCR", f"{lcr_data['lcr_ratio']:.1f}%")
        with col2:
            st.metric("Stressed LCR", f"{stressed_lcr:.1f}%")
        with col3:
            impact = stressed_lcr - lcr_data['lcr_ratio']
            st.metric("Impact", f"{impact:.1f}%", delta=f"{impact:.1f}%", delta_color="inverse")
        
        # Scenario comparison chart
        scenarios = pd.DataFrame({
            'Scenario': ['Base Case', 'Mild Stress', 'Severe Stress', 'Current Scenario'],
            'LCR': [lcr_data['lcr_ratio'], 115, 95, stressed_lcr]
        })
        
        fig = go.Figure()
        
        colors = ['blue' if x >= 100 else 'red' for x in scenarios['LCR']]
        
        fig.add_trace(go.Bar(
            x=scenarios['Scenario'],
            y=scenarios['LCR'],
            marker_color=colors,
            text=scenarios['LCR'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="Minimum Requirement (100%)")
        
        fig.update_layout(
            title="LCR Under Different Scenarios",
            yaxis_title="LCR (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.divider()
    st.subheader("Recommendations")
    
    if lcr_ratio >= 150:
        st.success("✓ LCR is well above regulatory minimum. Strong liquidity position.")
    elif lcr_ratio >= 120:
        st.success("✓ LCR is above regulatory minimum with good buffer.")
    elif lcr_ratio >= 100:
        st.warning("⚠ LCR meets minimum but consider building additional buffer.")
    else:
        st.error("✗ LCR is below regulatory minimum. Immediate action required!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Improve HQLA:**
        - Increase Level 1 assets
        - Convert lower-quality assets
        - Build cash reserves
        """)
    
    with col2:
        st.info("""
        **Reduce Outflows:**
        - Diversify funding sources
        - Extend liability maturity
        - Strengthen deposit base
        """)
    
    with col3:
        st.info("""
        **Monitor Risks:**
        - Concentration limits
        - Stress testing
        - Early warning indicators
        """)

def generate_stress_scenario():
    """Generate sample stress scenario data"""
    # This is a placeholder for more complex stress scenario generation
    return {
        'mild_stress': {'hqla_haircut': -5, 'outflow_mult': 1.1},
        'severe_stress': {'hqla_haircut': -20, 'outflow_mult': 1.5}
    }

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def show():
    st.header("📊 Asset-Liability Management (ALM) Gap Assessment")
    st.write("Analyze interest rate risk and maturity mismatches across time buckets")
    
    # Get data
    data = st.session_state.data
    alm_df = data['alm_gap'].copy()
    
    # Overview metrics
    st.subheader("ALM Gap Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assets = alm_df['assets'].sum()
        st.metric("Total Assets", f"${total_assets:.0f}M")
    
    with col2:
        total_liabilities = alm_df['liabilities'].sum()
        st.metric("Total Liabilities", f"${total_liabilities:.0f}M")
    
    with col3:
        total_gap = alm_df['gap'].sum()
        st.metric("Net Gap", f"${total_gap:.0f}M")
    
    with col4:
        gap_ratio = (total_gap / total_assets) * 100
        st.metric("Gap Ratio", f"{gap_ratio:.1f}%")
    
    # Gap Analysis Chart
    st.divider()
    st.subheader("Maturity Gap Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create gap chart
        fig = go.Figure()
        
        # Assets
        fig.add_trace(go.Bar(
            name='Assets',
            x=alm_df['bucket'],
            y=alm_df['assets'],
            marker_color='#2ecc71',
            hovertemplate='<b>%{x}</b><br>Assets: $%{y}M<extra></extra>'
        ))
        
        # Liabilities (negative)
        fig.add_trace(go.Bar(
            name='Liabilities',
            x=alm_df['bucket'],
            y=-alm_df['liabilities'],
            marker_color='#e74c3c',
            hovertemplate='<b>%{x}</b><br>Liabilities: $%{y}M<extra></extra>'
        ))
        
        # Gap line
        fig.add_trace(go.Scatter(
            name='Gap',
            x=alm_df['bucket'],
            y=alm_df['gap'],
            mode='lines+markers',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Gap: $%{y}M<extra></extra>'
        ))
        
        # Zero line for gap
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Asset-Liability Maturity Profile",
            xaxis_title="Time Bucket",
            yaxis_title="Amount ($M)",
            yaxis2=dict(
                title="Gap ($M)",
                overlaying='y',
                side='right'
            ),
            barmode='relative',
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Gap Interpretation**")
        
        for idx, row in alm_df.iterrows():
            gap = row['gap']
            bucket = row['bucket']
            
            if gap > 0:
                st.success(f"**{bucket}**: +${gap:.0f}M\nAsset-sensitive")
            elif gap < 0:
                st.error(f"**{bucket}**: ${gap:.0f}M\nLiability-sensitive")
            else:
                st.info(f"**{bucket}**: Balanced")
    
    # Cumulative Gap
    st.divider()
    st.subheader("Cumulative Gap Analysis")
    
    alm_df['cumulative_gap'] = alm_df['gap'].cumsum()
    alm_df['cumulative_gap_ratio'] = (alm_df['cumulative_gap'] / total_assets) * 100
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Cumulative gap chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=alm_df['bucket'],
            y=alm_df['cumulative_gap'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=10),
            name='Cumulative Gap'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Cumulative Gap Position",
            xaxis_title="Time Bucket",
            yaxis_title="Cumulative Gap ($M)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Cumulative Gap Table**")
        display_df = alm_df[['bucket', 'gap', 'cumulative_gap', 'cumulative_gap_ratio']].copy()
        st.dataframe(
            display_df.style.format({
                'gap': '${:.1f}M',
                'cumulative_gap': '${:.1f}M',
                'cumulative_gap_ratio': '{:.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    # Interest Rate Sensitivity
    st.divider()
    st.subheader("Interest Rate Sensitivity Analysis")
    
    st.write("Estimate the impact of interest rate changes on Net Interest Income (NII)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        rate_shock = st.slider(
            "Interest Rate Shock (basis points)",
            -200, 200, 100, 25,
            help="Parallel shift in interest rates"
        )
        
        time_horizon = st.selectbox(
            "Time Horizon",
            ["1 Year", "2 Years", "3 Years"],
            index=0
        )
    
    with col2:
        # Calculate NII impact
        nii_impact_df = calculate_nii_impact(alm_df, rate_shock)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=nii_impact_df['bucket'],
            y=nii_impact_df['nii_impact'],
            marker_color=['green' if x > 0 else 'red' for x in nii_impact_df['nii_impact']],
            text=nii_impact_df['nii_impact'].apply(lambda x: f"${x:.2f}M"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"NII Impact from {rate_shock} bps Rate Shock",
            xaxis_title="Time Bucket",
            yaxis_title="NII Impact ($M)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    total_nii_impact = nii_impact_df['nii_impact'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total NII Impact", f"${total_nii_impact:.2f}M")
    
    with col2:
        nii_impact_pct = (total_nii_impact / total_assets) * 100
        st.metric("Impact as % of Assets", f"{nii_impact_pct:.3f}%")
    
    with col3:
        if total_nii_impact > 0:
            st.success("Benefit from rate increase")
        else:
            st.error("Loss from rate increase")
    
    # Duration Analysis
    st.divider()
    st.subheader("Duration and Convexity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset duration
        asset_duration = calculate_duration(alm_df['bucket'], alm_df['assets'])
        liability_duration = calculate_duration(alm_df['bucket'], alm_df['liabilities'])
        
        duration_df = pd.DataFrame({
            'Category': ['Assets', 'Liabilities', 'Gap'],
            'Duration (years)': [
                asset_duration,
                liability_duration,
                asset_duration - liability_duration
            ],
            'Modified Duration': [
                asset_duration / 1.05,
                liability_duration / 1.05,
                (asset_duration - liability_duration) / 1.05
            ]
        })
        
        st.dataframe(
            duration_df.style.format({
                'Duration (years)': '{:.2f}',
                'Modified Duration': '{:.2f}'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        st.write("**Duration Gap:**")
        duration_gap = asset_duration - liability_duration
        
        if duration_gap > 0:
            st.info(f"Assets have longer duration (+{duration_gap:.2f} years). Portfolio value increases when rates fall.")
        else:
            st.warning(f"Liabilities have longer duration ({duration_gap:.2f} years). Portfolio value decreases when rates fall.")
    
    with col2:
        # Price sensitivity
        st.write("**Price Sensitivity to Rate Changes**")
        
        rate_changes = np.linspace(-2, 2, 9)  # -2% to +2%
        asset_values = []
        liability_values = []
        
        for rate in rate_changes:
            asset_val = 100 * (1 - (asset_duration / 1.05) * rate)
            liability_val = 100 * (1 - (liability_duration / 1.05) * rate)
            asset_values.append(asset_val)
            liability_values.append(liability_val)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rate_changes,
            y=asset_values,
            mode='lines+markers',
            name='Assets',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=rate_changes,
            y=liability_values,
            mode='lines+markers',
            name='Liabilities',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Value Sensitivity to Interest Rate Changes",
            xaxis_title="Interest Rate Change (%)",
            yaxis_title="Indexed Value",
            height=350,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Metrics
    st.divider()
    st.subheader("Risk Metrics & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gap limits
        max_gap = alm_df['gap'].abs().max()
        gap_limit = total_assets * 0.15  # 15% limit
        
        st.metric("Maximum Gap", f"${max_gap:.1f}M")
        st.metric("Gap Limit (15% assets)", f"${gap_limit:.1f}M")
        
        if max_gap <= gap_limit:
            st.success("✓ Within gap limits")
        else:
            st.error("✗ Exceeds gap limits")
    
    with col2:
        # Recommendations
        st.write("**Key Observations:**")
        
        # Find largest gaps
        max_positive = alm_df.loc[alm_df['gap'].idxmax()]
        max_negative = alm_df.loc[alm_df['gap'].idxmin()]
        
        if max_positive['gap'] > 5:
            st.warning(f"⚠ Large positive gap in {max_positive['bucket']}")
        
        if max_negative['gap'] < -5:
            st.warning(f"⚠ Large negative gap in {max_negative['bucket']}")
        
        if abs(duration_gap) > 1:
            st.warning(f"⚠ Significant duration gap: {duration_gap:.2f} years")
    
    with col3:
        st.write("**Hedging Strategies:**")
        
        if total_gap > 0:
            st.info("""
            **Asset-sensitive position:**
            - Consider interest rate swaps (pay fixed)
            - Issue long-term debt
            - Reduce short-term assets
            """)
        else:
            st.info("""
            **Liability-sensitive position:**
            - Consider interest rate swaps (receive fixed)
            - Extend asset maturity
            - Reduce short-term liabilities
            """)

def calculate_nii_impact(alm_df, rate_shock_bps):
    """Calculate Net Interest Income impact from rate shock"""
    rate_shock = rate_shock_bps / 10000  # Convert bps to decimal
    
    impact_df = alm_df.copy()
    impact_df['nii_impact'] = impact_df['gap'] * rate_shock
    
    return impact_df[['bucket', 'nii_impact']]

def calculate_duration(buckets, amounts):
    """Calculate weighted average duration"""
    # Map buckets to approximate durations (in years)
    duration_map = {
        '0-1M': 0.04,
        '1-3M': 0.17,
        '3-6M': 0.38,
        '6-12M': 0.75,
        '1-2Y': 1.5,
        '2-5Y': 3.5,
        '5Y+': 7.0
    }
    
    total_amount = amounts.sum()
    weighted_duration = 0
    
    for bucket, amount in zip(buckets, amounts):
        if bucket in duration_map:
            weighted_duration += duration_map[bucket] * amount / total_amount
    
    return weighted_duration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show():
    st.header("⚡ Intraday Liquidity Monitoring")
    st.write("Real-time monitoring of intraday liquidity positions and payment flows")
    
    # Get data
    data = st.session_state.data
    liquidity_df = data['intraday_liquidity'].copy()
    
    # Current status
    st.subheader("Current Intraday Position")
    
    current_time = datetime.now()
    current_available = liquidity_df.iloc[-1]['available']
    current_required = liquidity_df.iloc[-1]['required']
    current_buffer = liquidity_df.iloc[-1]['buffer']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Available Liquidity",
            f"${current_available}M",
            delta=f"{liquidity_df['available'].pct_change().iloc[-1]*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Required Liquidity",
            f"${current_required}M"
        )
    
    with col3:
        st.metric(
            "Liquidity Buffer",
            f"${current_buffer}M",
            delta="Adequate" if current_buffer > 0 else "Deficit",
            delta_color="normal" if current_buffer > 0 else "inverse"
        )
    
    with col4:
        coverage_ratio = (current_available / current_required) * 100
        st.metric(
            "Coverage Ratio",
            f"{coverage_ratio:.1f}%",
            delta="✓" if coverage_ratio >= 100 else "✗"
        )
    
    # Intraday liquidity chart
    st.divider()
    st.subheader("Intraday Liquidity Profile")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = go.Figure()
        
        # Available liquidity
        fig.add_trace(go.Scatter(
            x=liquidity_df['time'],
            y=liquidity_df['available'],
            mode='lines+markers',
            name='Available',
            line=dict(color='#2ecc71', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)'
        ))
        
        # Required liquidity
        fig.add_trace(go.Scatter(
            x=liquidity_df['time'],
            y=liquidity_df['required'],
            mode='lines+markers',
            name='Required',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        # Buffer
        fig.add_trace(go.Scatter(
            x=liquidity_df['time'],
            y=liquidity_df['buffer'],
            mode='lines+markers',
            name='Buffer',
            line=dict(color='#3498db', width=2),
            yaxis='y2'
        ))
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Intraday Liquidity Position",
            xaxis_title="Time",
            yaxis_title="Liquidity ($M)",
            yaxis2=dict(
                title="Buffer ($M)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Time-based Analysis**")
        
        min_buffer_time = liquidity_df.loc[liquidity_df['buffer'].idxmin(), 'time']
        min_buffer = liquidity_df['buffer'].min()
        
        st.metric("Lowest Buffer", f"${min_buffer:.0f}M")
        st.write(f"Time: {min_buffer_time.strftime('%H:%M')}")
        
        if min_buffer < 0:
            st.error(f"⚠ Deficit at {min_buffer_time.strftime('%H:%M')}")
        else:
            st.success("✓ Positive buffer all day")
        
        st.divider()
        
        st.write("**Coverage by Hour**")
        for idx, row in liquidity_df.iterrows():
            coverage = (row['available'] / row['required']) * 100
            time_str = row['time'].strftime('%H:%M')
            
            if coverage >= 110:
                st.success(f"{time_str}: {coverage:.0f}%")
            elif coverage >= 100:
                st.info(f"{time_str}: {coverage:.0f}%")
            else:
                st.error(f"{time_str}: {coverage:.0f}%")
    
    # Payment flows
    st.divider()
    st.subheader("Payment Flow Analysis")
    
    # Generate sample payment flows
    payment_flows = generate_payment_flows()
    
    tab1, tab2, tab3 = st.tabs(["Flow Summary", "Large Payments", "Payment Channels"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Incoming Payments**")
            
            inflows = payment_flows[payment_flows['type'] == 'Inflow']
            
            st.metric("Total Inflows", f"${inflows['amount'].sum():.1f}M")
            st.metric("Number of Payments", len(inflows))
            st.metric("Average Size", f"${inflows['amount'].mean():.2f}M")
            
            # Inflow by time
            inflow_by_hour = inflows.groupby(inflows['time'].dt.hour)['amount'].sum()
            
            fig = go.Figure(go.Bar(
                x=[f"{h:02d}:00" for h in inflow_by_hour.index],
                y=inflow_by_hour.values,
                marker_color='green',
                name='Inflows'
            ))
            
            fig.update_layout(
                title="Inflows by Hour",
                xaxis_title="Hour",
                yaxis_title="Amount ($M)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Outgoing Payments**")
            
            outflows = payment_flows[payment_flows['type'] == 'Outflow']
            
            st.metric("Total Outflows", f"${outflows['amount'].sum():.1f}M")
            st.metric("Number of Payments", len(outflows))
            st.metric("Average Size", f"${outflows['amount'].mean():.2f}M")
            
            # Outflow by time
            outflow_by_hour = outflows.groupby(outflows['time'].dt.hour)['amount'].sum()
            
            fig = go.Figure(go.Bar(
                x=[f"{h:02d}:00" for h in outflow_by_hour.index],
                y=outflow_by_hour.values,
                marker_color='red',
                name='Outflows'
            ))
            
            fig.update_layout(
                title="Outflows by Hour",
                xaxis_title="Hour",
                yaxis_title="Amount ($M)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Net flows
        st.write("**Net Payment Flows**")
        net_flow = inflows['amount'].sum() - outflows['amount'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Net Flow", f"${net_flow:.1f}M")
        with col2:
            st.metric("Inflow/Outflow Ratio", f"{(inflows['amount'].sum()/outflows['amount'].sum()):.2f}")
        with col3:
            if net_flow > 0:
                st.success("Net inflow position")
            else:
                st.warning("Net outflow position")
    
    with tab2:
        st.write("**Large Payments (> $5M)**")
        
        large_payments = payment_flows[payment_flows['amount'] > 5].copy()
        large_payments = large_payments.sort_values('amount', ascending=False)
        
        st.dataframe(
            large_payments[['time', 'type', 'amount', 'counterparty', 'channel']].style.format({
                'time': lambda x: x.strftime('%H:%M:%S'),
                'amount': '${:.2f}M'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Large Payments", len(large_payments))
        with col2:
            st.metric("Total Large Payment Value", f"${large_payments['amount'].sum():.1f}M")
    
    with tab3:
        st.write("**Payment by Channel**")
        
        channel_summary = payment_flows.groupby(['channel', 'type'])['amount'].sum().reset_index()
        
        fig = go.Figure()
        
        for payment_type in ['Inflow', 'Outflow']:
            data = channel_summary[channel_summary['type'] == payment_type]
            fig.add_trace(go.Bar(
                name=payment_type,
                x=data['channel'],
                y=data['amount'],
                marker_color='green' if payment_type == 'Inflow' else 'red'
            ))
        
        fig.update_layout(
            title="Payment Volume by Channel",
            xaxis_title="Channel",
            yaxis_title="Amount ($M)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Channel details
        channel_details = payment_flows.groupby('channel').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        channel_details.columns = ['Total ($M)', 'Count', 'Avg ($M)']
        st.dataframe(channel_details, use_container_width=True)
    
    # Real-time alerts
    st.divider()
    st.subheader("Liquidity Alerts & Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Active Alerts**")
        
        alerts = []
        
        # Check for low buffer
        if current_buffer < 5:
            alerts.append({
                'severity': 'High' if current_buffer < 0 else 'Medium',
                'message': f"Low liquidity buffer: ${current_buffer:.1f}M",
                'time': current_time.strftime('%H:%M:%S')
            })
        
        # Check for large outflows
        recent_outflows = outflows[outflows['time'] > current_time - timedelta(hours=1)]
        if recent_outflows['amount'].sum() > 20:
            alerts.append({
                'severity': 'Medium',
                'message': f"High outflow activity: ${recent_outflows['amount'].sum():.1f}M in last hour",
                'time': current_time.strftime('%H:%M:%S')
            })
        
        # Check for concentration
        if len(large_payments) > 10:
            alerts.append({
                'severity': 'Low',
                'message': f"High number of large payments: {len(large_payments)}",
                'time': current_time.strftime('%H:%M:%S')
            })
        
        if not alerts:
            st.success("✓ No active alerts - All metrics within normal range")
        else:
            for alert in alerts:
                if alert['severity'] == 'High':
                    st.error(f"🔴 **HIGH**: {alert['message']} at {alert['time']}")
                elif alert['severity'] == 'Medium':
                    st.warning(f"🟡 **MEDIUM**: {alert['message']} at {alert['time']}")
                else:
                    st.info(f"🔵 **LOW**: {alert['message']} at {alert['time']}")
    
    with col2:
        st.write("**Monitoring Thresholds**")
        
        thresholds = pd.DataFrame({
            'Metric': [
                'Min Buffer',
                'Coverage Ratio',
                'Large Payment',
                'Hourly Outflow'
            ],
            'Threshold': [
                '> $5M',
                '> 100%',
                '> $5M',
                '< $30M'
            ],
            'Current': [
                f"${current_buffer:.1f}M",
                f"{coverage_ratio:.1f}%",
                f"${large_payments['amount'].max():.1f}M",
                f"${outflows['amount'].sum():.1f}M"
            ],
            'Status': [
                '✓' if current_buffer > 5 else '✗',
                '✓' if coverage_ratio > 100 else '✗',
                '✗' if large_payments['amount'].max() > 5 else '✓',
                '✓' if outflows['amount'].sum() < 30 else '✗'
            ]
        })
        
        st.dataframe(thresholds, hide_index=True, use_container_width=True)
    
    # Forecasting
    st.divider()
    st.subheader("Intraday Liquidity Forecast")
    
    st.write("Expected liquidity position for remainder of day")
    
    # Generate simple forecast
    forecast_hours = pd.date_range(
        start=current_time.replace(minute=0, second=0),
        periods=4,
        freq='H'
    )
    
    # Simple forecast based on historical pattern
    forecast_available = [current_available, current_available + 2, current_available + 5, current_available + 3]
    forecast_required = [current_required] * 4
    
    forecast_df = pd.DataFrame({
        'time': forecast_hours,
        'available': forecast_available,
        'required': forecast_required,
        'buffer': [a - r for a, r in zip(forecast_available, forecast_required)]
    })
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=liquidity_df['time'],
        y=liquidity_df['available'],
        mode='lines+markers',
        name='Historical Available',
        line=dict(color='#2ecc71', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['available'],
        mode='lines+markers',
        name='Forecast Available',
        line=dict(color='#3498db', width=2, dash='dash')
    ))
    
    # Required
    fig.add_trace(go.Scatter(
        x=forecast_df['time'],
        y=forecast_df['required'],
        mode='lines',
        name='Required',
        line=dict(color='#e74c3c', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title="Liquidity Forecast (Next 4 Hours)",
        xaxis_title="Time",
        yaxis_title="Liquidity ($M)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Action items
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Monitoring Actions:**
        - Real-time position tracking
        - Payment queue monitoring
        - Settlement system status
        - Counterparty availability
        """)
    
    with col2:
        st.info("""
        **Funding Options:**
        - Intraday repo
        - Central bank facilities
        - Interbank borrowing
        - Securities lending
        """)
    
    with col3:
        st.info("""
        **Risk Mitigation:**
        - Payment prioritization
        - Netting arrangements
        - Contingency funding
        - Communication protocols
        """)

def generate_payment_flows():
    """Generate sample payment flow data"""
    np.random.seed(42)
    
    current_date = datetime.now().replace(hour=9, minute=0, second=0)
    
    payments = []
    
    # Generate inflows
    for i in range(50):
        time = current_date + timedelta(hours=np.random.uniform(0, 8), 
                                       minutes=np.random.uniform(0, 60))
        payments.append({
            'time': time,
            'type': 'Inflow',
            'amount': np.random.lognormal(0, 1) * 2,
            'counterparty': f"Bank-{np.random.randint(1, 20):02d}",
            'channel': np.random.choice(['RTGS', 'SWIFT', 'ACH', 'Internal'])
        })
    
    # Generate outflows
    for i in range(55):
        time = current_date + timedelta(hours=np.random.uniform(0, 8), 
                                       minutes=np.random.uniform(0, 60))
        payments.append({
            'time': time,
            'type': 'Outflow',
            'amount': np.random.lognormal(0, 1) * 2,
            'counterparty': f"Bank-{np.random.randint(1, 20):02d}",
            'channel': np.random.choice(['RTGS', 'SWIFT', 'ACH', 'Internal'])
        })
    
    df = pd.DataFrame(payments)
    df = df.sort_values('time')
    
    return df

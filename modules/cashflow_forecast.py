import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

def show():
    st.header("💸 Cash Flow Forecasting with Machine Learning")
    st.write("Predict future cash flows using advanced ML algorithms")
    
    # Get data from session state
    data = st.session_state.data
    historical_df = data['historical_cashflow'].copy()
    
    # Sidebar controls
    st.sidebar.subheader("Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
    model_type = st.sidebar.selectbox(
        "ML Model",
        ["Random Forest", "Gradient Boosting", "Linear Regression"]
    )
    
    # Feature engineering
    st.subheader("1. Historical Data & Feature Engineering")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Historical Data Points", len(historical_df))
    with col2:
        st.metric("Average Daily Cash Flow", f"${historical_df['amount'].mean()/1e6:.2f}M")
    
    # Prepare features
    df_features = prepare_features(historical_df)
    
    with st.expander("View Feature Engineering Details"):
        st.write("**Created Features:**")
        st.write("- Day of week, Day of month, Month")
        st.write("- 7-day, 14-day, and 30-day moving averages")
        st.write("- Lag features (1, 7, 30 days)")
        st.write("- Trend component")
        st.dataframe(df_features.head(10))
    
    # Train model
    st.subheader("2. Model Training & Evaluation")
    
    X = df_features.drop(['date', 'amount'], axis=1)
    y = df_features['amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Select and train model
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    else:
        model = LinearRegression()
    
    with st.spinner(f"Training {model_type} model..."):
        model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
    with col2:
        st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred))/1e6:.2f}M")
    with col3:
        st.metric("MAE", f"${mean_absolute_error(y_test, y_pred)/1e6:.2f}M")
    with col4:
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        st.metric("MAPE", f"{mape:.2f}%")
    
    # Plot actual vs predicted
    fig = go.Figure()
    test_dates = df_features.iloc[X_test.index]['date']
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Model Performance: Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Cash Flow ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate forecast
    st.subheader("3. Future Cash Flow Forecast")
    
    forecast_df = generate_forecast(model, df_features, forecast_days)
    
    # Display forecast
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot forecast
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=df_features['date'].tail(90),
            y=df_features['amount'].tail(90),
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='lightgreen', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='lightgreen', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig.update_layout(
            title=f"{forecast_days}-Day Cash Flow Forecast",
            xaxis_title="Date",
            yaxis_title="Cash Flow ($)",
            hovermode='x unified',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Forecast Summary**")
        st.metric("Average Forecast", f"${forecast_df['forecast'].mean()/1e6:.2f}M")
        st.metric("Min Forecast", f"${forecast_df['forecast'].min()/1e6:.2f}M")
        st.metric("Max Forecast", f"${forecast_df['forecast'].max()/1e6:.2f}M")
        st.metric("Total Period", f"${forecast_df['forecast'].sum()/1e6:.2f}M")
        
        # Risk indicators
        st.divider()
        st.write("**Risk Indicators**")
        volatility = forecast_df['forecast'].std()
        avg = forecast_df['forecast'].mean()
        cv = (volatility / avg) * 100
        st.metric("Volatility", f"${volatility/1e6:.2f}M")
        st.metric("Coeff. of Variation", f"{cv:.1f}%")
    
    # Detailed forecast table
    with st.expander("View Detailed Forecast Table"):
        display_df = forecast_df.copy()
        display_df['forecast'] = display_df['forecast'].apply(lambda x: f"${x/1e6:.2f}M")
        display_df['lower_bound'] = display_df['lower_bound'].apply(lambda x: f"${x/1e6:.2f}M")
        display_df['upper_bound'] = display_df['upper_bound'].apply(lambda x: f"${x/1e6:.2f}M")
        st.dataframe(display_df, use_container_width=True)
    
    # Feature importance (for tree-based models)
    if model_type in ["Random Forest", "Gradient Boosting"]:
        st.subheader("4. Feature Importance Analysis")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=feature_importance['importance'].head(10),
            y=feature_importance['feature'].head(10),
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def prepare_features(df):
    """Prepare features for ML model"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Lag features
    df['lag_1'] = df['amount'].shift(1)
    df['lag_7'] = df['amount'].shift(7)
    df['lag_30'] = df['amount'].shift(30)
    
    # Rolling statistics
    df['rolling_mean_7'] = df['amount'].rolling(window=7).mean()
    df['rolling_mean_14'] = df['amount'].rolling(window=14).mean()
    df['rolling_mean_30'] = df['amount'].rolling(window=30).mean()
    df['rolling_std_7'] = df['amount'].rolling(window=7).std()
    df['rolling_std_30'] = df['amount'].rolling(window=30).std()
    
    # Trend
    df['trend'] = range(len(df))
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def generate_forecast(model, historical_df, forecast_days):
    """Generate future forecast"""
    last_date = historical_df['date'].max()
    last_row = historical_df.iloc[-1]
    
    forecasts = []
    
    for i in range(forecast_days):
        # Create features for next day
        next_date = last_date + timedelta(days=i+1)
        
        features = {
            'day_of_week': next_date.dayofweek,
            'day_of_month': next_date.day,
            'month': next_date.month,
            'quarter': (next_date.month - 1) // 3 + 1,
            'lag_1': last_row['amount'] if i == 0 else forecasts[-1],
            'lag_7': historical_df.iloc[-7]['amount'] if i < 7 else forecasts[i-7] if i >= 7 else last_row['amount'],
            'lag_30': historical_df.iloc[-30]['amount'] if i < 30 else forecasts[i-30] if i >= 30 else last_row['amount'],
            'rolling_mean_7': historical_df['amount'].tail(7).mean(),
            'rolling_mean_14': historical_df['amount'].tail(14).mean(),
            'rolling_mean_30': historical_df['amount'].tail(30).mean(),
            'rolling_std_7': historical_df['amount'].tail(7).std(),
            'rolling_std_30': historical_df['amount'].tail(30).std(),
            'trend': last_row['trend'] + i + 1
        }
        
        X_next = pd.DataFrame([features])
        pred = model.predict(X_next)[0]
        forecasts.append(pred)
    
    # Create forecast dataframe with confidence intervals
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    # Estimate prediction intervals (simple approach)
    std_error = np.std(forecasts) * 0.5  # Simplified standard error
    
    forecast_df = pd.DataFrame({
        'date': dates,
        'forecast': forecasts,
        'lower_bound': [f - 1.96*std_error for f in forecasts],
        'upper_bound': [f + 1.96*std_error for f in forecasts]
    })
    
    return forecast_df

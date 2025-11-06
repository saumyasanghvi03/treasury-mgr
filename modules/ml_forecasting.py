"""
ML-Driven Cash Flow Forecasting Module

This module implements machine learning models for forecasting future cash flows
using historical transaction data and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta


class CashFlowForecaster:
    """
    Random Forest-based cash flow forecasting model.
    
    This class handles feature engineering, model training, and prediction
    for cash flow forecasting tasks.
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the forecaster with a Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random state for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_importance = None
        
    def engineer_features(self, df):
        """
        Create features from historical cash flow data.
        
        Features include:
        - Time-based features (day of week, month, quarter)
        - Lag features (previous periods)
        - Rolling statistics (moving averages, volatility)
        - Trend components
        
        Args:
            df (pd.DataFrame): Daily cash flow dataframe with 'date' and 'net_cash_flow'
            
        Returns:
            pd.DataFrame: Feature-engineered dataframe
        """
        feature_df = df.copy()
        feature_df = feature_df.sort_values('date').reset_index(drop=True)
        
        # Time-based features
        feature_df['day_of_week'] = feature_df['date'].dt.dayofweek
        feature_df['day_of_month'] = feature_df['date'].dt.day
        feature_df['month'] = feature_df['date'].dt.month
        feature_df['quarter'] = feature_df['date'].dt.quarter
        feature_df['is_month_start'] = feature_df['date'].dt.is_month_start.astype(int)
        feature_df['is_month_end'] = feature_df['date'].dt.is_month_end.astype(int)
        feature_df['is_quarter_start'] = feature_df['date'].dt.is_quarter_start.astype(int)
        feature_df['is_quarter_end'] = feature_df['date'].dt.is_quarter_end.astype(int)
        
        # Lag features (previous periods)
        for lag in [1, 2, 3, 7, 14, 30]:
            feature_df[f'lag_{lag}d'] = feature_df['net_cash_flow'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30, 90]:
            feature_df[f'rolling_mean_{window}d'] = feature_df['net_cash_flow'].rolling(window=window).mean()
            feature_df[f'rolling_std_{window}d'] = feature_df['net_cash_flow'].rolling(window=window).std()
        
        # Exponential moving averages
        feature_df['ema_7d'] = feature_df['net_cash_flow'].ewm(span=7, adjust=False).mean()
        feature_df['ema_30d'] = feature_df['net_cash_flow'].ewm(span=30, adjust=False).mean()
        
        # Trend indicator
        feature_df['trend'] = np.arange(len(feature_df))
        
        return feature_df
    
    def prepare_training_data(self, feature_df, target_col='net_cash_flow'):
        """
        Prepare data for model training by separating features and target.
        
        Args:
            feature_df (pd.DataFrame): Feature-engineered dataframe
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X, y) features and target arrays
        """
        # Drop rows with NaN values (due to lag/rolling features)
        clean_df = feature_df.dropna()
        
        # Define feature columns (exclude date and target)
        feature_cols = [col for col in clean_df.columns 
                       if col not in ['date', target_col, 'cumulative_position', 'gross_amount']]
        
        X = clean_df[feature_cols].values
        y = clean_df[target_col].values
        
        return X, y, feature_cols
    
    def train(self, df, test_size=0.2):
        """
        Train the Random Forest model on historical data.
        
        Args:
            df (pd.DataFrame): Historical cash flow data
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics and performance statistics
        """
        # Engineer features
        feature_df = self.engineer_features(df)
        
        # Prepare training data
        X, y, feature_cols = self.prepare_training_data(feature_df)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_cols = feature_cols
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, 
                                    scoring='neg_mean_absolute_error')
        metrics['cv_mae_mean'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        return metrics
    
    def forecast(self, df, forecast_days=30):
        """
        Generate cash flow forecasts for future periods.
        
        Args:
            df (pd.DataFrame): Historical cash flow data
            forecast_days (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: Forecast dataframe with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting. Call train() first.")
        
        # Engineer features on historical data
        feature_df = self.engineer_features(df)
        
        # Get the last date in the dataset
        last_date = feature_df['date'].max()
        
        forecasts = []
        current_df = feature_df.copy()
        
        for day in range(1, forecast_days + 1):
            # Create next date
            next_date = last_date + timedelta(days=day)
            
            # Create a new row with time features
            next_row = pd.DataFrame({
                'date': [next_date],
                'day_of_week': [next_date.dayofweek],
                'day_of_month': [next_date.day],
                'month': [next_date.month],
                'quarter': [next_date.quarter],
                'is_month_start': [int(next_date.day == 1)],
                'is_month_end': [int(next_date == (next_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1))],
                'is_quarter_start': [int((next_date.month - 1) % 3 == 0 and next_date.day == 1)],
                'is_quarter_end': [int(next_date in [pd.Timestamp(next_date.year, 3, 31), 
                                                      pd.Timestamp(next_date.year, 6, 30),
                                                      pd.Timestamp(next_date.year, 9, 30),
                                                      pd.Timestamp(next_date.year, 12, 31)])],
                'trend': [len(current_df)]
            })
            
            # Add lag features from most recent data
            for lag in [1, 2, 3, 7, 14, 30]:
                if len(current_df) >= lag:
                    next_row[f'lag_{lag}d'] = current_df.iloc[-lag]['net_cash_flow']
                else:
                    next_row[f'lag_{lag}d'] = 0
            
            # Add rolling statistics
            for window in [7, 14, 30, 90]:
                if len(current_df) >= window:
                    next_row[f'rolling_mean_{window}d'] = current_df.tail(window)['net_cash_flow'].mean()
                    next_row[f'rolling_std_{window}d'] = current_df.tail(window)['net_cash_flow'].std()
                else:
                    next_row[f'rolling_mean_{window}d'] = current_df['net_cash_flow'].mean()
                    next_row[f'rolling_std_{window}d'] = current_df['net_cash_flow'].std()
            
            # Add EMA features
            next_row['ema_7d'] = current_df.tail(7)['net_cash_flow'].ewm(span=7, adjust=False).mean().iloc[-1]
            next_row['ema_30d'] = current_df.tail(30)['net_cash_flow'].ewm(span=30, adjust=False).mean().iloc[-1]
            
            # Prepare features for prediction
            X_pred = next_row[self.feature_cols].values
            
            # Make prediction
            prediction = self.model.predict(X_pred)[0]
            
            # Estimate prediction intervals using tree predictions
            tree_predictions = np.array([tree.predict(X_pred)[0] for tree in self.model.estimators_])
            lower_bound = np.percentile(tree_predictions, 10)
            upper_bound = np.percentile(tree_predictions, 90)
            
            forecasts.append({
                'date': next_date,
                'forecast': prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_width': upper_bound - lower_bound
            })
            
            # Add prediction to current_df for next iteration
            next_row['net_cash_flow'] = prediction
            current_df = pd.concat([current_df, next_row], ignore_index=True)
        
        forecast_df = pd.DataFrame(forecasts)
        return forecast_df
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features for the model.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Top N most important features
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first.")
        
        return self.feature_importance.head(top_n)


def analyze_forecast_accuracy(actuals, forecasts):
    """
    Analyze the accuracy of forecasts against actual values.
    
    Args:
        actuals (pd.Series or np.array): Actual cash flow values
        forecasts (pd.Series or np.array): Forecasted cash flow values
        
    Returns:
        dict: Accuracy metrics
    """
    actuals = np.array(actuals)
    forecasts = np.array(forecasts)
    
    metrics = {
        'mae': mean_absolute_error(actuals, forecasts),
        'rmse': np.sqrt(mean_squared_error(actuals, forecasts)),
        'mape': np.mean(np.abs((actuals - forecasts) / actuals)) * 100,
        'r2': r2_score(actuals, forecasts)
    }
    
    return metrics

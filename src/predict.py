import numpy as np
import pandas as pd

# The model is evaluated on the held-out test set (Aug–Dec 2024). All predictions use only information available at 10:00 D-1.¶

def forecasting_DminusOne(model, test, features, target):
    """
    Generates forecasts for the test set using the provided model.
    
    Parameters:
    - model: The trained model to use for forecasting.
    - test: The test DataFrame containing features and target variable.
    
    Returns:
    - A DataFrame with timestamps, actual prices, and predicted prices.
    """
    X_test = test[features]
    y_test = test[target]
    
    # Generate predictions
    y_pred = model.predict(X_test)
    residual_error = y_test - y_pred
    abs_error = np.abs(residual_error)
    
    # Create a DataFrame to hold results
    results_df = pd.DataFrame({
        'timestamp': test['timestamp'],
        'actual_price': y_test,
        'predicted_price': y_pred,  
        'residual_error': residual_error,
        'absolute_error': abs_error
    })
    
    return results_df


def forecast_day_ahead(model, df, features, forecast_date):
    """
    Simulates what happens at 10:00 on D-1.
    """
    forecast_date = pd.Timestamp(forecast_date)
    day_mask = df['timestamp'].dt.date == forecast_date.date()
    day_df   = df[day_mask].copy()
    
    if len(day_df) == 0:
        raise ValueError(f"No data found for {forecast_date.date()}")
    if len(day_df) < 48:
        print(f"Warning: only {len(day_df)} SPs found for {forecast_date.date()}, expected 48")
    missing = day_df[features].isna().sum()
    if missing.sum() > 0:
        print("Warning: missing features:")
        print(missing[missing > 0])
    
    # Produce the forecast
    day_df['predicted_price'] = model.predict(day_df[features])
    
    return day_df[['timestamp', 'predicted_price', 'price',
                   'net_demand', 'wind_forecast', 'demand']].reset_index(drop=True)


def forecast_multiple_days(model, df, features, start_date, end_date):
    """
    Rolls the forecast forward day by day.
    """
    dates   = pd.date_range(start=start_date, end=end_date, freq='D')
    results = []
    
    for date in dates:
        try:
            day_forecast = forecast_day_ahead(model, df, features, date)
            results.append(day_forecast)
        except ValueError:
            continue
    
    if not results:
        raise ValueError("No forecasts generated — check date range")
        
    return pd.concat(results, ignore_index=True)
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
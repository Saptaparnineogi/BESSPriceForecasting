import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_lag_baseline(train, test, features):
    # Naive lag baseline — predict D-7 same SP
    lag_mae = mean_absolute_error(
        test['price'].values, 
        test['price_lag_336'].values
    )
    print(f"Naive lag-336 MAE: {lag_mae:.3f}")

    
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"XGB MAE: {mae:.3f}")
    print(f"XGB RMSE: {rmse:.3f}")
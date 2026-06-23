import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_train_test_split(df, features, test_size=0.2):
    """
    Splits the dataset into training and testing sets based on a specified test size.
    """
    df = df.dropna(subset=features).reset_index(drop=True)
    train_size = int(len(df)*(1-test_size))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    print(f"Train date range: {train['timestamp'].min()} → {train['timestamp'].max()}")
    print(f"Test date range:  {test['timestamp'].min()} → {test['timestamp'].max()}")
    print(f"\nNaNs in train features: {train[features].isna().sum().sum()}")
    print(f"NaNs in test features:  {test[features].isna().sum().sum()}")
    print(f"\nTrain price stats: mean={train['price'].mean():.1f}, std={train['price'].std():.1f}")
    print(f"Test price stats:  mean={test['price'].mean():.1f}, std={test['price'].std():.1f}")
    return train, test

def naive_baseline(train, test, features):
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(train[features], train['price'])
    dummy_pred = dummy.predict(test[features])
    print(f"Dummy mean MAE: {mean_absolute_error(test['price'], dummy_pred):.3f}")


def lag_baseline(test):
    # Naive lag baseline — predict D-7 same SP
    lag_mae = mean_absolute_error(
        test['price'].values, 
        test['price_lag_336'].values
    )
    print(f"Naive lag-336 MAE: {lag_mae:.3f}")
    


def train_xgb_model(train, test, features):
    model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=1,
    early_stopping_rounds=50,
    eval_metric='mae',
    random_state=42
    )
    val_size = int(len(train) * 0.15)
    X_tr  = train[features].iloc[:-val_size]
    y_tr  = train['price'].iloc[:-val_size]
    X_val = train[features].iloc[-val_size:]
    y_val = train['price'].iloc[-val_size:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    print(f"Best iteration: {model.best_iteration}")
    return model

def evaluate_model(model, test, features, target='price'):
    X_test = test[features]
    y_test = test[target]
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"XGB MAE: {mae:.3f}")
    print(f"XGB RMSE: {rmse:.3f}")



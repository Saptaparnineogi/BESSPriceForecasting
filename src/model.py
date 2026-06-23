import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor



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
    return dummy


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





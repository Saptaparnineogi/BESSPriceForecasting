
import numpy as np
import pandas as pd


def add_seasonality(df):
    """
    Captures systematic patterns (e.g., weekday evening peaks, lower weekend prices).
    Decomposed hour, day of week, month feature using sine and cosine transformations to preserve the cyclical nature of the 24-hour clock
    and the periodic structure of weekly trends.
    """
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"]= df["timestamp"].dt.dayofweek
    df["month"]      = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_winter"]  = df["month"].isin([11,12,1,2,3]).astype(int)

    df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]    = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]    = np.cos(2*np.pi*df["day_of_week"]/7)
    df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
    return df

def add_lagged_features(df):
    """
    Prices and system conditions have strong daily patterns. Lagged features help the model learn “similar days” behavior.
    """
    df["price_lag_336"]      = df["price"].shift(336)   # D-7 
    df["price_lag_672"]      = df["price"].shift(672)   # D-14 

    df["price_roll_mean_48"] = df["price"].shift(48).rolling(48).mean() # D-2
    df["price_roll_std_48"]  = df["price"].shift(48).rolling(48).std() # D-2
    df["demand_roll_mean_48"]= df["demand"].shift(1).rolling(48).mean() # D-2
    return df

def add_diff_and_netdemand(df):
    """
    Supply demand features
    """
    df["net_demand"]          = df["demand"] - df["wind_forecast"]
    df["wind_share"]          = df["wind_forecast"] / (df["demand"] + 1e-6)
    df["demand_change"]       = df["demand"].diff()
    df["wind_change"]         = df["wind_forecast"].diff()
    df["net_demand_change"]   = df["net_demand"].diff()
    df["net_demand_zscore"]   = (
        (df["net_demand"] - df["net_demand"].rolling(336).mean()) /
        (df["net_demand"].rolling(336).std() + 1e-6)
    )
    df["wind_zscore"]         = (
        (df["wind_forecast"] - df["wind_forecast"].rolling(336).mean()) /
        (df["wind_forecast"].rolling(336).std() + 1e-6)
    )
    df["tightness_x_demand"]  = df["net_demand_zscore"] * df["net_demand"]
    return df
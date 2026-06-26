import pandas as pd
from data_processing import get_price_data, process_mpi, get_demand_data, process_demand_data, process_wind_forecast
from feature_engineering import add_diff_and_netdemand, add_lagged_features, add_seasonality
from model import get_train_test_split, naive_baseline, train_xgb_model
from evaluate import evaluate_lag_baseline, evaluate_model
from predict import forecasting_DminusOne
from visualization import plot_forecast_results
from bessoptimizer import greedy_bess_dispatch, select_representative_day,plot_bess_dispatch


def get_time_alignment(price_df):
    full_range = pd.date_range(
    start=price_df.timestamp.min(),
    end=price_df.timestamp.max(),
    freq="30min"
    )
    base_df = pd.DataFrame({'timestamp':full_range})
    return base_df

def load_data(price_data_url, demand_data_path, wind_data_path):
    price_df = get_price_data("2023-01-01", "2024-12-31", source_url=price_data_url)
    processed_price_df = process_mpi(price_df)
    demand_df = get_demand_data(demand_data_path)
    processed_demand_df = process_demand_data(demand_df)
    wind_df = pd.read_csv(wind_data_path)
    processed_wind_df = process_wind_forecast(wind_df)
    base_df = get_time_alignment(processed_price_df)
    df = base_df.merge(processed_price_df, on='timestamp', how='left')
    df = df.merge(processed_demand_df, on="timestamp", how="left")
    df = df.merge(processed_wind_df, on="timestamp", how="left")
    return df

def main():
    price_data_src = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
    demand_data_path = r"\Users\olive\Documents\workspace\BESSPriceForecasting\data"
    wind_data_path = r"\Users\olive\Documents\workspace\BESSPriceForecasting\data\archive_1dayaheadwind.csv"
    df = load_data(price_data_src, demand_data_path, wind_data_path)
    print("Data processing complete.....")
    print("Dataframe shape:", df.shape)
    df = add_seasonality(df)
    df = add_lagged_features(df)
    df = add_diff_and_netdemand(df)
    features = [
        'demand',
        'wind_gen',
        'solar_gen',
        'net_demand',
        'interconnector_flow',
        'wind_forecast',
        'hour',
        'day_of_week',
        'month',
        'is_weekend',
        'is_winter',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'month_sin',
        'month_cos',
        'price_lag_336',
        'price_lag_672',
        'price_roll_mean_48',
        'price_roll_std_48',
        'demand_roll_mean_48',
        'wind_share',
        'demand_change',
        'wind_change',
        'net_demand_change',
        'net_demand_zscore',
        'wind_zscore',
        'tightness_x_demand'
        ]

    target = "price"
    train, test = get_train_test_split(df, features)
    dummyRegressor = naive_baseline(train, test, features)
    naive_baseline(train, test, features)
    xbgRegressor = train_xgb_model(train, test, features)
    print("Model training complete.....")
    print("Evaluating models.....")
    print("Naive Regressor")
    print("-----------------------------------")
    evaluate_model(dummyRegressor, test[features], test[target])
    print("Lag Baseline")
    print("-----------------------------------")
    evaluate_lag_baseline(train, test, features)
    print("XGB Regressor")
    print("-----------------------------------")
    evaluate_model(xbgRegressor, test[features], test[target])
    forecast_results = forecasting_DminusOne(xbgRegressor, test, features, target)
    print("Forecasting complete.....")
    plot_forecast_results(forecast_results)
    day = select_representative_day(test, position=0.5)
    dispatch_day = greedy_bess_dispatch(
        day,
        price_col="predicted_price",
        cap_mwh=100,
        power_mw=50,
        eta=0.90
        )
    plot_bess_dispatch(
    dispatch_day,
    actual_price_col="price",
    forecast_price_col="predicted_price",
    save_path="figures/bess_dispatch_example.png",
)
    

if __name__=="__main__":
    main()
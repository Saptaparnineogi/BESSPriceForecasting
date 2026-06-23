import pandas as pd
import requests
from datetime import timedelta
import glob
import os

### We are downloading data for the given period between start_date and end_date

def get_price_data(start_date, end_date, source_url, cache_file="price_data_cache.csv"):
    # Check if cached file exists
    if os.path.exists(cache_file):
        print(f"Loading price data from cache: {cache_file}")
        price_df = pd.read_csv(cache_file)
        return price_df
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    date_ranges = []
    
    current = start_date
    
    while current <= end_date:
        week_end = current + timedelta(days=6)
        if week_end > end_date:
            week_end = end_date
        date_ranges.append((current, week_end))
        current = week_end + timedelta(days=1)
        
    all_price_data = []
    
    url = source_url
    
    for start, end in date_ranges:
        params = {
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "format": "json"
        }
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            if "data" in data:
                df = pd.json_normalize(data["data"]) #Normalize semi-structured JSON data into a flat table
                all_price_data.append(df)
            print(f"Downloaded price data: {start.date()} to {end.date()}")
        except Exception as e:
            print(f"Error for {start} to {end}: {e}")
    price_df = pd.concat(all_price_data, ignore_index=True)
    price_df.to_csv(cache_file, index=False)
    print(f"Cached price data to: {cache_file}")
    return price_df

def process_mpi(price_df):
    price_df = price_df[price_df['dataProvider'] == 'APXMIDP'].copy()
    price_df["settlementDate"] = pd.to_datetime(price_df["settlementDate"])
    price_df["timestamp"] = (
        price_df["settlementDate"]
        + pd.to_timedelta((price_df["settlementPeriod"] - 1) * 30, unit="m")
    )
    # timestamp construction formula assumes 48 SPs, which is correct for the vast majority of days.
    # SP numbering starts at 1 but the clock starts at 0, you subtract 1 to align them before multiplying.
    price_df = price_df.drop_duplicates(subset="timestamp")
    price_df = price_df.sort_values("timestamp")
    price_df = price_df.reset_index(drop=True)
    full_index = pd.date_range(
        start=price_df["timestamp"].min(),
        end=price_df["timestamp"].max(),
        freq="30min"
    )
    price_df = (
        price_df
        .set_index("timestamp")
        .reindex(full_index)
    )
    price_df.index.name = "timestamp"
    price_df = price_df.reset_index()
    price_df["price"] = price_df["price"].ffill()
    
    price_df["date"] = price_df["timestamp"].dt.date
    
    price_df["settlementPeriod"] = (
        price_df["timestamp"].dt.hour * 2
        + price_df["timestamp"].dt.minute // 30
        + 1
    )
    price_df = price_df[["timestamp", "price"]]
    return price_df

def get_demand_data(demand_folder_path):
    print("Demand folder path:", demand_folder_path)
    pattern = os.path.join(demand_folder_path, "demanddata_*.csv")  # adjust pattern as needed
    file_list = glob.glob(pattern)
    demand_df = pd.concat((pd.read_csv(f) for f in file_list), ignore_index=True)
    return demand_df

def process_demand_data(demand_df):
    demand_df["SETTLEMENT_DATE"] = pd.to_datetime(demand_df["SETTLEMENT_DATE"])
    demand_df["timestamp"] = (
        demand_df["SETTLEMENT_DATE"]
        + pd.to_timedelta((demand_df["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")
    )
    demand_df = demand_df.rename(columns={
    "ND": "demand",
    "EMBEDDED_WIND_GENERATION": "wind_gen",
    "EMBEDDED_SOLAR_GENERATION": "solar_gen"
    })
    interconnectors = [
    "IFA_FLOW",
    "IFA2_FLOW",
    "BRITNED_FLOW",
    "MOYLE_FLOW",
    "EAST_WEST_FLOW",
    "NEMO_FLOW",
    "NSL_FLOW",
    "ELECLINK_FLOW",
    "VIKING_FLOW",
    "GREENLINK_FLOW"
    ]

    demand_df["interconnector_flow"] = demand_df[interconnectors].sum(axis=1)
    demand_df["net_demand"] = (
        demand_df["demand"]
        - demand_df["wind_gen"]
        - demand_df["solar_gen"]
    )
    demand_df = demand_df[[
        "timestamp",
        "demand",
        "wind_gen",
        "solar_gen",
        "net_demand",
        "interconnector_flow"
    ]]
    return demand_df


def process_wind_forecast(wind_df):
    wind_df["timestamp"] = pd.to_datetime(wind_df["Datetime_GMT"]).dt.tz_localize(None)
    wind_df = wind_df.drop_duplicates(subset="timestamp")
    wind_df = wind_df.rename(columns={"Incentive_forecast": "wind_forecast"})
    print(wind_df.columns)
    wind_df = wind_df[["timestamp", "wind_forecast"]]
    return wind_df


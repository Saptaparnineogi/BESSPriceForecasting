import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_representative_day(test: pd.DataFrame, position: float = 0.5) -> pd.DataFrame:
    """
    Select one full day from the test set.

    Parameters
    ----------
    test : pd.DataFrame
        Test dataframe containing a timestamp column.
    position : float
        Relative position in the test set. 0.5 selects a day around the middle.

    Returns
    -------
    pd.DataFrame
        One-day dataframe.
    """
    test = test.copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    idx = int(len(test) * position)
    day_start = test["timestamp"].iloc[idx].normalize()

    day = test[test["timestamp"].dt.date == day_start.date()].copy()
    day = day.reset_index(drop=True)

    return day


def greedy_bess_dispatch(
    day: pd.DataFrame,
    price_col: str = "predicted_price",
    cap_mwh: float = 100,
    power_mw: float = 50,
    eta: float = 0.90,
    soc_initial_pct: float = 0.50,
    soc_min_pct: float = 0.10,
    soc_max_pct: float = 0.95,
    charge_percentile: float = 40,
    discharge_percentile: float = 60,
    settlement_period_hours: float = 0.5,
) -> pd.DataFrame:
    """
    Simulate a simple greedy BESS dispatch strategy.

    Charges during low forecast-price periods and discharges during high
    forecast-price periods, subject to battery power and SoC constraints.
    """
    day = day.copy()

    prices = day[price_col].values
    n = len(prices)

    cap_mwh = float(cap_mwh)
    ep = power_mw * settlement_period_hours

    soc_min = cap_mwh * soc_min_pct
    soc_max = cap_mwh * soc_max_pct
    soc_initial = cap_mwh * soc_initial_pct

    action = np.zeros(n)
    energy = np.zeros(n)

    charge_threshold = np.percentile(prices, charge_percentile)
    discharge_threshold = np.percentile(prices, discharge_percentile)

    # Charge at low forecast-price periods
    soc = soc_initial
    for idx in np.argsort(prices):
        if soc >= soc_max:
            break
        if prices[idx] > charge_threshold:
            break

        charge = min(ep, (soc_max - soc) / eta)
        soc += charge * eta

        action[idx] = 1
        energy[idx] = charge

    # Discharge at high forecast-price periods
    soc = soc_initial
    for idx in np.argsort(-prices):
        if soc <= soc_min:
            break
        if prices[idx] < discharge_threshold:
            break

        discharge = min(ep, soc - soc_min)
        soc -= discharge

        action[idx] = -1
        energy[idx] = -discharge

    # Build SoC profile in chronological order
    soc_profile = np.zeros(n)
    soc = soc_initial

    for i in range(n):
        if action[i] == 1:
            soc += abs(energy[i]) * eta
        elif action[i] == -1:
            soc -= abs(energy[i])

        soc_profile[i] = soc

    revenue = np.where(
        action == -1,
        abs(energy) * prices,
        np.where(action == 1, -abs(energy) * prices, 0),
    )

    day["action"] = action
    day["energy_mwh"] = energy
    day["soc_mwh"] = soc_profile
    day["revenue_gbp"] = revenue

    day.attrs["dispatch_params"] = {
        "cap_mwh": cap_mwh,
        "power_mw": power_mw,
        "eta": eta,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "soc_initial": soc_initial,
        "charge_threshold": charge_threshold,
        "discharge_threshold": discharge_threshold,
        "total_revenue": revenue.sum(),
    }

    return day


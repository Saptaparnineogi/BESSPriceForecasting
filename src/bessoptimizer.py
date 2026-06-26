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

def plot_bess_dispatch(
    day: pd.DataFrame,
    actual_price_col: str = "price",
    forecast_price_col: str = "predicted_price",
    save_path: str | None = None,
):
    """
    Plot BESS dispatch decisions, battery SoC, and cumulative revenue.
    """
    params = day.attrs.get("dispatch_params", {})

    soc_min = params.get("soc_min", day["soc_mwh"].min())
    soc_max = params.get("soc_max", day["soc_mwh"].max())
    cap_mwh = params.get("cap_mwh", day["soc_mwh"].max())
    total_revenue = day["revenue_gbp"].sum()

    charge_mask = day["action"] == 1
    discharge_mask = day["action"] == -1

    avg_buy = day.loc[charge_mask, forecast_price_col].mean()
    avg_sell = day.loc[discharge_mask, forecast_price_col].mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Price and dispatch
    axes[0].plot(
        day["timestamp"],
        day[actual_price_col],
        lw=2,
        label="Actual price",
    )

    axes[0].plot(
        day["timestamp"],
        day[forecast_price_col],
        lw=2,
        linestyle="--",
        label="Forecast price",
    )

    axes[0].scatter(
        day.loc[charge_mask, "timestamp"],
        day.loc[charge_mask, forecast_price_col],
        s=100,
        marker="v",
        zorder=5,
        label="Charge",
    )

    axes[0].scatter(
        day.loc[discharge_mask, "timestamp"],
        day.loc[discharge_mask, forecast_price_col],
        s=100,
        marker="^",
        zorder=5,
        label="Discharge",
    )

    axes[0].set_ylabel("Price (£/MWh)")
    axes[0].set_title(
        f"BESS Dispatch — {day['timestamp'].dt.date.iloc[0]} | "
        f"Revenue: £{total_revenue:,.0f} | "
        f"Avg buy: £{avg_buy:.0f} | Avg sell: £{avg_sell:.0f}",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # SoC profile
    axes[1].fill_between(day["timestamp"], day["soc_mwh"], alpha=0.4)
    axes[1].plot(day["timestamp"], day["soc_mwh"], lw=2)

    axes[1].axhline(
        soc_max,
        lw=1,
        linestyle=":",
        label=f"Max SoC ({soc_max:.0f} MWh)",
    )

    axes[1].axhline(
        soc_min,
        lw=1,
        linestyle=":",
        label=f"Min SoC ({soc_min:.0f} MWh)",
    )

    axes[1].set_ylabel("SoC (MWh)")
    axes[1].set_title("Battery State of Charge")
    axes[1].set_ylim(0, cap_mwh * 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Cumulative revenue
    cumulative_revenue = day["revenue_gbp"].cumsum()

    axes[2].fill_between(
        day["timestamp"],
        cumulative_revenue,
        alpha=0.4,
    )

    axes[2].plot(
        day["timestamp"],
        cumulative_revenue,
        lw=2,
    )

    axes[2].axhline(0, lw=0.8, alpha=0.5)
    axes[2].set_ylabel("Cumulative Revenue (£)")
    axes[2].set_title(f"Cumulative Revenue — Total: £{total_revenue:,.0f}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
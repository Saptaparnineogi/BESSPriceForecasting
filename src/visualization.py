import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_forecast_results(results_df):
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(results_df['timestamp'], results_df['actual_price'],
            label='Actual Price', alpha=0.7, color='blue')
    ax.plot(results_df['timestamp'], results_df['predicted_price'],
            label='Forecast', alpha=0.9, 
            color='orange', linestyle='--')

    ax.set_title('Day-Ahead Price Forecast — Full Test Period (Aug–Dec 2024)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (£/MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


def add_horizontal_line(fig, df, y, row, col, name, dash="dash"):
    fig.add_trace(
        go.Scatter(
            x=[df["timestamp"].min(), df["timestamp"].max()],
            y=[y, y],
            mode="lines",
            name=name,
            line=dict(dash=dash, width=1),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

def plot_bess_dispatch_dashboard(
    day: pd.DataFrame,
    actual_price_col: str = "actual_price",
    forecast_price_col: str = "predicted_price",
    action_col: str = "action",
    soc_col: str = "soc_mwh",
    revenue_col: str = "revenue_gbp",
    energy_col: str = "energy_mwh",
    save_html: str | None = None,
    save_image: str | None = None,
):
    """
    Create an interactive Plotly dashboard for BESS dispatch simulation.

    Expected columns:
    - timestamp
    - price
    - predicted_price
    - action: 1 = charge, -1 = discharge, 0 = idle
    - soc_mwh
    - revenue_gbp
    - energy_mwh
    """

    df = day.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    params = df.attrs.get("dispatch_params", {})

    soc_min = params.get("soc_min", df[soc_col].min())
    soc_max = params.get("soc_max", df[soc_col].max())
    cap_mwh = params.get("cap_mwh", max(df[soc_col].max(), soc_max))

    charge_mask = df[action_col] == 1
    discharge_mask = df[action_col] == -1

    total_revenue = df[revenue_col].sum()
    cumulative_revenue = df[revenue_col].cumsum()

    avg_buy = df.loc[charge_mask, forecast_price_col].mean()
    avg_sell = df.loc[discharge_mask, forecast_price_col].mean()

    n_charge = int(charge_mask.sum())
    n_discharge = int(discharge_mask.sum())

    if np.isnan(avg_buy):
        avg_buy = 0

    if np.isnan(avg_sell):
        avg_sell = 0

    # Positive = discharge / sell, Negative = charge / buy
    dispatch_power = np.where(
        df[action_col] == 1,
        -abs(df[energy_col]),
        np.where(df[action_col] == -1, abs(df[energy_col]), 0),
    )

    fig = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.17,
        horizontal_spacing=0.08,
        row_heights=[0.12, 0.38, 0.25, 0.25],
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        subplot_titles=[
            "",
            "",
            "Electricity Price Forecast and Dispatch Decisions",
            "",
            "Battery State of Charge",
            "Charge / Discharge Energy",
            "Cumulative Revenue",
            "",
        ],
    )

    # KPI 1: Total revenue
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_revenue,
            number={"prefix": "£", "valueformat": ",.0f"},
            title={"text": "Estimated Arbitrage Revenue"},
        ),
        row=1,
        col=1,
    )

    # KPI 2: Spread
    spread = avg_sell - avg_buy

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=spread,
            number={"prefix": "£", "suffix": "/MWh", "valueformat": ".1f"},
            title={"text": f"Avg Sell-Buy Spread<br><span style='font-size:12px'>Buy £{avg_buy:.1f} | Sell £{avg_sell:.1f}</span>"},
        ),
        row=1,
        col=2,
    )

    # Price traces
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[actual_price_col],
            mode="lines",
            name="Actual price",
            line=dict(width=2),
            hovertemplate="Time: %{x}<br>Actual: £%{y:.2f}/MWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[forecast_price_col],
            mode="lines",
            name="Forecast price",
            line=dict(width=2, dash="dash"),
            hovertemplate="Time: %{x}<br>Forecast: £%{y:.2f}/MWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Charge markers
    fig.add_trace(
        go.Scatter(
            x=df.loc[charge_mask, "timestamp"],
            y=df.loc[charge_mask, forecast_price_col],
            mode="markers",
            name="Charge",
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="green",
                line=dict(width=1, color="darkgreen"),
            ),
            hovertemplate="Charge<br>Time: %{x}<br>Forecast price: £%{y:.2f}/MWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Discharge markers
    fig.add_trace(
        go.Scatter(
            x=df.loc[discharge_mask, "timestamp"],
            y=df.loc[discharge_mask, forecast_price_col],
            mode="markers",
            name="Discharge",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="red",
                line=dict(width=1, color="darkred"),
            ),
            hovertemplate="Discharge<br>Time: %{x}<br>Forecast price: £%{y:.2f}/MWh<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # SoC
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[soc_col],
            mode="lines",
            name="State of charge",
            fill="tozeroy",
            line=dict(width=2),
            hovertemplate="Time: %{x}<br>SoC: %{y:.1f} MWh<extra></extra>",
        ),
        row=3,
        col=1,
    )

    add_horizontal_line(
        fig,
        df,
        soc_max,
        row=3,
        col=1,
        name=f"Max SoC {soc_max:.0f} MWh",
    )

    add_horizontal_line(
        fig,
        df,
        soc_min,
        row=3,
        col=1,
        name=f"Min SoC {soc_min:.0f} MWh",
    )

    # Dispatch energy bars
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=dispatch_power,
            name="Dispatch energy",
            hovertemplate="Time: %{x}<br>Energy: %{y:.1f} MWh<extra></extra>",
        ),
        row=3,
        col=2,
    )

    add_horizontal_line(
        fig,
        df,
        0,
        row=3,
        col=2,
        name="Zero dispatch",
)

    # Cumulative revenue
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=cumulative_revenue,
            mode="lines",
            name="Cumulative revenue",
            fill="tozeroy",
            line=dict(width=3),
            hovertemplate="Time: %{x}<br>Cumulative revenue: £%{y:,.0f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    add_horizontal_line(
        fig,
        df,
        0,
        row=4,
        col=1,
        name="Zero revenue",
    )

    date_label = df["timestamp"].dt.date.iloc[0]

    fig.update_layout(
        title=dict(
            text=(
                f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
                f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
                f"Capacity {cap_mwh:.0f} MWh</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
        ),
        template="plotly_white",
        height=1150,
        width=1300,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.91,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)
    fig.update_layout(
    title=dict(
        text=(
            f"<b>BESS Dispatch Dashboard — {date_label}</b><br>"
            f"<sup>{n_charge} charge periods | {n_discharge} discharge periods | "
            f"Capacity {cap_mwh:.0f} MWh</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.98,
    ),
    template="plotly_white",
    height=1150,
    width=1300,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.91,
        xanchor="right",
        x=1,
    ),
    margin=dict(t=180, l=70, r=40, b=60),
)

    fig.update_yaxes(title_text="Price (£/MWh)", row=2, col=1)
    fig.update_yaxes(title_text="SoC (MWh)", row=3, col=1, range=[0, cap_mwh * 1.05])
    fig.update_yaxes(title_text="Energy (MWh)", row=3, col=2)
    fig.update_yaxes(title_text="Revenue (£)", row=4, col=1)

    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_xaxes(tickformat="%H:%M")

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")

    if save_image:
        fig.write_image(save_image, scale=2)
    return fig
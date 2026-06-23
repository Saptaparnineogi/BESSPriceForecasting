import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px


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
    plt.tight_layout()
    plt.show()
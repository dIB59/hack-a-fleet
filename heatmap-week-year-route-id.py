from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_traffic_heatmap_by_route(route_id, traffic_cols: List[str] = None, legend_max=None,
                                  data_file="ferry_tips_data.csv"):
    """
    Generates a heatmap of total outbound traffic for a specific route.

    Parameters:
    - route_id: The route ID to filter the data by.
    - data_file: The path to the CSV file containing the ferry data. Default is "ferry_tips_data.csv".
    - traffic_cols: Columns that will be summed
    """
    # Load data
    if traffic_cols is None:
        traffic_cols = ['cars_outbound']
    df = pd.read_csv(data_file)

    # Filter the data by the specified route_id
    df = df[df['route_id'] == route_id]

    if df.empty:
        print(f"No data found for route_id {route_id}.")
        return

    # Convert time_departure to datetime
    df['time_departure'] = pd.to_datetime(df['time_departure'])

    # Extract week, weekday
    df['week'] = df['time_departure'].dt.isocalendar().week  # Week number in the year
    df['weekday'] = df['time_departure'].dt.day_name()  # Day of the week

    # Calculate total outbound traffic

    df['total_outbound_traffic'] = df[traffic_cols].sum(axis=1)

    # Group data by week and weekday, summing across all data
    grouped_data = df.groupby(['week', 'weekday'])['total_outbound_traffic'].sum().reset_index()

    # Find global maximum value for the color scale
    if legend_max is None:
        global_max = grouped_data['total_outbound_traffic'].max()
    else:
        global_max = legend_max

    # Reorder days of the week
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create a heatmap with weeks on the y-axis and days of the week on the x-axis
    heatmap_data = grouped_data.pivot_table(
        values='total_outbound_traffic',
        index='week',  # Week number on the y-axis
        columns='weekday',  # Day of the week on the x-axis
        aggfunc='sum',
        fill_value=0
    ).reindex(columns=ordered_days)

    median_value = np.median(heatmap_data.values)
    plt.figure(figsize=(14, 20))
    sns.heatmap(
        heatmap_data,
        cmap='coolwarm',
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        vmin=0,  # Set color scale minimum
        vmax=global_max,  # Set color scale maximum
        cbar_kws={'label': 'Total Outbound Traffic'},  # Color bar label
        annot_kws={"size": 10},
        center=median_value
    )

    # Improve title and axis labels
    plt.title(f'Total Outbound Traffic Heatmap for Route ID {route_id}', fontsize=18, fontweight='bold')
    plt.xlabel('Day of the Week', fontsize=14)
    plt.ylabel('Week of the Year', fontsize=14)

    # Improve tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)  # Keep week labels horizontal for better readability

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Adjust layout and make it tight
    plt.tight_layout()

    # Show the plot
    plt.show()


# Example usage:

if __name__ == "__main__":
    traffic_col = ['cars_inbound', 'trucks_inbound', 'trucks_with_trailer_inbound', 'motorcycles_inbound',
                   'exemption_vehicles_inbound', 'pedestrians_inbound', 'buses_inbound', 'cars_outbound',
                   'trucks_outbound', 'trucks_with_trailer_outbound', 'motorcycles_outbound',
                   'exemption_vehicles_outbound', 'pedestrians_outbound', 'buses_outbound']

    plot_traffic_heatmap_by_route(17, traffic_cols=traffic_col, legend_max=3000)
    plot_traffic_heatmap_by_route(16, traffic_cols=traffic_col, legend_max=3000)
    plot_traffic_heatmap_by_route(12, traffic_cols=traffic_col, legend_max=3000)
    plot_traffic_heatmap_by_route(21, traffic_cols=traffic_col, legend_max=3000)
    plot_traffic_heatmap_by_route(38, traffic_cols=traffic_col, legend_max=3000)

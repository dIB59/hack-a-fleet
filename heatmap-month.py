import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("ferry_tips_data.csv")


# Convert time_departure to datetime
df['time_departure'] = pd.to_datetime(df['time_departure'])

# Extract month, weekday, and hour
df['month'] = df['time_departure'].dt.month_name()  # Get month name
df['weekday'] = df['time_departure'].dt.day_name()  # Get day name
df['hour'] = df['time_departure'].dt.hour           # Get hour of the day

# Calculate total outbound traffic
traffic_cols = ['cars_outbound', 'trucks_outbound', 'motorcycles_outbound', 'pedestrians_outbound']
df['total_outbound_traffic'] = df[traffic_cols].sum(axis=1)

# Group data for each month
grouped_data = df.groupby(['month', 'weekday', 'hour'])['total_outbound_traffic'].sum().reset_index()

# Find global maximum value for the color scale
global_max = grouped_data['total_outbound_traffic'].max()

# Reorder days of the week
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a heatmap for each month
months = df['month'].unique()

for month in months:
    plt.figure(figsize=(12, 8))

    # Filter data for the current month
    monthly_data = grouped_data[grouped_data['month'] == month]
    monthly_heatmap = monthly_data.pivot_table(
        values='total_outbound_traffic',
        index='weekday',
        columns='hour',
        aggfunc='sum',
        fill_value=0
    ).reindex(index=ordered_days)

    # Plot the heatmap with a fixed color scale
    sns.heatmap(
        monthly_heatmap,
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        linewidths=.5,
        vmin=0,  # Set color scale minimum
        vmax=global_max  # Set color scale maximum
    )

    # Add titles and labels
    plt.title(f'Traffic Heatmap for {month}', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of the Week', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.show()

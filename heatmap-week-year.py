import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("ferry_tips_data.csv")

# Convert time_departure to datetime
df['time_departure'] = pd.to_datetime(df['time_departure'])

# Extract week, weekday
df['week'] = df['time_departure'].dt.isocalendar().week  # Week number in the year
df['weekday'] = df['time_departure'].dt.day_name()      # Day of the week

# Calculate total outbound traffic
traffic_cols = ['cars_outbound', 'trucks_outbound', 'motorcycles_outbound', 'pedestrians_outbound']
df['total_outbound_traffic'] = df[traffic_cols].sum(axis=1)

# Group data by week and weekday, summing across all data
grouped_data = df.groupby(['week', 'weekday'])['total_outbound_traffic'].sum().reset_index()

# Find global maximum value for the color scale
global_max = grouped_data['total_outbound_traffic'].max()

# Reorder days of the week
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a heatmap with weeks on the y-axis and days of the week on the x-axis
heatmap_data = grouped_data.pivot_table(
    values='total_outbound_traffic',
    index='week',    # Week number on the y-axis
    columns='weekday',  # Day of the week on the x-axis
    aggfunc='sum',
    fill_value=0
).reindex(columns=ordered_days)

# Plot the heatmap with a fixed color scale
plt.figure(figsize=(14, 20))
sns.heatmap(
    heatmap_data,
    cmap='coolwarm',  # Color map with better contrast
    annot=True,
    fmt='.0f',
    linewidths=0.5,
    vmin=0,  # Set color scale minimum
    vmax=global_max,  # Set color scale maximum
    cbar_kws={'label': 'Total Outbound Traffic'},  # Color bar label
    annot_kws={"size": 10}  # Annotate with smaller text
)

# Improve title and axis labels
plt.title('Total Outbound Traffic Heatmap for the Year', fontsize=18, fontweight='bold')
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

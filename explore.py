import pandas as pd

# Load data
df = pd.read_csv("ferry_tips_data.csv")

# Explore data
print(df.head())
print(df.info())
print(df.describe())

# Fill or drop missing values
df.fillna(0, inplace=True)

# Convert date columns to datetime
df['time_departure'] = pd.to_datetime(df['time_departure'])
df['start_time_outbound'] = pd.to_datetime(df['start_time_outbound'], errors='coerce')
df['end_time_outbound'] = pd.to_datetime(df['end_time_outbound'], errors='coerce')
df['start_time_inbound'] = pd.to_datetime(df['start_time_inbound'], errors='coerce')
df['end_time_inbound'] = pd.to_datetime(df['end_time_inbound'], errors='coerce')

traffic_cols = ['cars_outbound', 'trucks_outbound', 'motorcycles_outbound', 'pedestrians_outbound']
df['total_outbound_traffic'] = df[traffic_cols].sum(axis=1)

df['date'] = df['time_departure'].dt.date
traffic_trends = df.groupby('date')['total_outbound_traffic'].sum()
traffic_trends.plot(title='Daily Outbound Traffic')

df['fuel_efficiency_outbound'] = df['passenger_car_equivalent_outbound'] / df['fuelcons_outbound_l']
print(df[['fuel_efficiency_outbound']].describe())

df['hour'] = df['time_departure'].dt.hour
peak_hours = df.groupby('hour')['total_outbound_traffic'].sum()
peak_hours.plot(kind='bar', title='Traffic by Hour')

import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap for traffic patterns by hour and day
df['weekday'] = df['time_departure'].dt.day_name()
heatmap_data = df.pivot_table(values='total_outbound_traffic', index='weekday', columns='hour', aggfunc='sum')
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    cmap='YlGnBu',
    annot=True,
    fmt='.0f',
    linewidths=.5,
    annot_kws={"size": 8}
)

plt.title('Traffic Heatmap', fontsize=16)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Day of the Week', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Load your dataset
df = pd.read_csv("ferry_tips_data.csv")

# Sample terminal coordinates - replace with real coordinates
terminal_coords = {
    "Rindö": (59.24126, 18.21526),
    "Värmdö": (59.24006, 18.20597)
}

aggregated_data = df.groupby(
    ['route_name', 'terminal_departure', 'terminal_arrival']
).agg({
    'distance_outbound_nm': 'mean',
    'distance_inbound_nm': 'mean',
    'route_id': 'count'  # Frequency of trips
}).reset_index()

# Map terminal names to coordinates
aggregated_data['departure_coords'] = aggregated_data['terminal_departure'].map(terminal_coords)
aggregated_data['arrival_coords'] = aggregated_data['terminal_arrival'].map(terminal_coords)

# Filter out rows with missing coordinates
aggregated_data = aggregated_data.dropna(subset=['departure_coords', 'arrival_coords'])

# Plot the routes
fig, ax = plt.subplots(figsize=(10, 8))
for _, row in aggregated_data.iterrows():
    dep_coords = row['departure_coords']
    arr_coords = row['arrival_coords']

    # Ensure coordinates are valid
    if isinstance(dep_coords, tuple) and isinstance(arr_coords, tuple):
        ax.plot(
            [dep_coords[1], arr_coords[1]],  # Longitude
            [dep_coords[0], arr_coords[0]],  # Latitude
            label=f"{row['route_name']}: {row['route_id']} trips",
            linewidth=row['route_id'] / 10,  # Adjust line thickness based on trips
        )

# Beautify the plot
ax.set_title("Route Map")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.show()

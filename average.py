import csv
from typing import List

import pandas as pd

ROUTE_ID = 0
ROUTE_NAME = 1
FERRY_NAME = 2
FERRY_ID = 5
terminal_departure = 6
terminal_arrival = 7
time_departure = 8
cars_outbound = 9
trucks_outbound = 10
trucks_with_trailer_outbound = 11
motorcycles_outbound = 12
exemption_vehicles_outbound = 13
pedestrians_outbound = 14
buses_outbound = 15
vehicles_left_at_terminal_outbound = 16
cars_inbound = 17
trucks_inbound = 18
trucks_with_trailer_inbound = 19
motorcycles_inbound = 20
exemption_vehicles_inbound = 21
pedestrians_inbound = 22
buses_inbound = 23
vehicles_left_at_terminal_inbound = 24
trip_type = 25
passenger_car_equivalent_outbound_and_inbound = 26
tailored_trip = 27
full_ferry_outbound = 28
full_ferry_inbound = 29
passenger_car_equivalent_outbound = 30
passenger_car_equivalent_inbound = 31
fuelcons_outbound_l = 32
distance_outbound_nm = 33
start_time_outbound = 34
end_time_outbound = 35
fuelcons_inbound_l = 36
distance_inbound_nm = 37
start_time_inbound = 38
end_time_inbound = 39

chunk_size = 1000


def get_average(file_path: str, trip_types: List[str]):
    ferry_sums = {}
    ferry_counts = {}
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk['trip_type'] = chunk['trip_type'].str.strip()
        filtered_trips = chunk[chunk['trip_type'].isin(trip_types)]
        grouped = filtered_trips.groupby('ferry_name')['passenger_car_equivalent_outbound_and_inbound'].agg(['sum', 'count'])
        # Update the accumulated sums and counts for each ferry
        for ferry_name, values in grouped.iterrows():

            if ferry_name in ferry_sums:
                ferry_sums[ferry_name] += values['sum']
                ferry_counts[ferry_name] += values['count']
            else:
                ferry_sums[ferry_name] = values['sum']
                ferry_counts[ferry_name] = values['count']
    return ferry_sums, ferry_counts


sums, counts = get_average("ferry_tips_data.csv", ['ordinary', 'extra', 'doubtful', 'proactive', 'doubling', 'extra'])

# Print the header
print(f"{'Ferry Name':<15} {'Average Passenger Car Equivalent':>30}")

# Print the averages for each ferry in a well-formatted way
for name in sums:
    avg = round(sums[name] / counts[name], 2)
    print(f"{name:<15} {avg:>30}")

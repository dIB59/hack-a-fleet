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


def get_average(file_path: str, aggregate_header: str,
                filter_header: str = 'trip_type',
                filter_allow: List[str] = ('ordinary', 'extra')
                ):
    ferry_sums = {}
    ferry_counts = {}
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk[('%s' % filter_header)] = chunk[filter_header].str.strip()
        filtered_trips = chunk[chunk[filter_header].isin(filter_allow)]
        grouped = filtered_trips.groupby('ferry_name')[('%s' % aggregate_header)].agg(
            ['sum', 'count'])
        # Update the accumulated sums and counts for each ferry
        for ferry_name, values in grouped.iterrows():

            if ferry_name in ferry_sums:
                ferry_sums[ferry_name] += values['sum']
                ferry_counts[ferry_name] += values['count']
            else:
                ferry_sums[ferry_name] = values['sum']
                ferry_counts[ferry_name] = values['count']

    ans = []
    for n in ferry_sums:
        average = round(ferry_sums[n] / ferry_counts[n], 2)
        ans.append((n, average))

    return ans


if __name__ == "__main__":
    answer = get_average("ferry_tips_data.csv",
                         'passenger_car_equivalent_outbound_and_inbound',
                         'trip_type',
                         ['ordinary', 'extra', 'doubtful', 'proactive', 'doubling', 'extra'],
                         )
    print(answer)

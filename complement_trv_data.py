import json
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from visualisation import *
from utils import *

with open("ferries.json", "r") as file:
    ferries = json.load(file)
print(ferries)

# Path to Excel file provided by Färjerederiet
file_path = "./ferry_trips_hackaton.xlsx"

# Load the Excel file into a Dataframe
trv_data = pd.read_excel(file_path)

# Add extra columns to the trv_data DataFrame
trv_data["fuelcons_outbound_l"] = None
trv_data["distance_outbound_nm"] = None
trv_data["start_time_outbound"] = None
trv_data["end_time_outbound"] = None

trv_data["fuelcons_inbound_l"] = None
trv_data["distance_inbound_nm"] = None
trv_data["start_time_inbound"] = None
trv_data["end_time_inbound"] = None

# Time bucket
time_bucket = "30 seconds"  # Used to fetch data from PONTOS-HUB
time_bucket_s = 30  # Used to calculate fuel consumption

# Loop through all the ferries
for ferry in ferries.keys():

    print(f"Ferry: {ferry}")

    # Get a list of all the days in the time_departure column for the ferry
    days = trv_data[trv_data["ferry_name"].str.lower() == ferry][
        "time_departure"
    ].dt.date.unique()

    # Loop through all the days with a progress bar
    for day in tqdm(days, desc="Processing days"):

        # Get departure times for the day
        departure_times = trv_data[
            (trv_data["ferry_name"].str.lower() == ferry)
            & (trv_data["time_departure"].dt.date == day)
        ]["time_departure"]

        # Check if day is after PONTOS-HUB launch date
        if pd.Timestamp(day) <= pd.Timestamp("2023-04-30"):
            break

        # Fetch vessel data from PONTOS-HUB
        start_of_day = pd.Timestamp(day)
        end_of_day = start_of_day + pd.Timedelta("1 day")
        vessel_data = fetch_vessel_data(
            ferries[ferry]["pontos_id"],
            str(start_of_day),
            str(end_of_day),
            time_bucket=time_bucket,
        )

        # Get trips from vessel data
        trips = get_trips_from_vessel_data(vessel_data)

        for departure_time in departure_times:

            # Identify outbound and inbound trips
            outbound_trip = None
            inbound_trip = None
            departure_time = pd.Timestamp(departure_time)
            closest_outbound_trip_index = min(
                range(len(trips)),
                key=lambda i: abs(pd.Timestamp(trips[i]["time"][0]) - departure_time),
            )
            closest_inbound_trip_index = (
                closest_outbound_trip_index + 1
                if closest_outbound_trip_index + 1 < len(trips)
                else closest_outbound_trip_index
            )
            closest_outbound_trip = trips[closest_outbound_trip_index]
            closest_inbound_trip = trips[closest_inbound_trip_index]
            if (
                abs(
                    (
                        pd.Timestamp(closest_outbound_trip["time"][0]) - departure_time
                    ).total_seconds()
                )
                / 60
                <= 5
            ):  # 5 minutes diff is acceptable
                outbound_trip = closest_outbound_trip
                if (
                    pd.Timestamp(closest_outbound_trip["time"][0])
                    - pd.Timestamp(closest_outbound_trip["time"][-1])
                ).total_seconds() / 60 <= 10:  # 10 minutes between first and last point is acceptable
                    inbound_trip = closest_inbound_trip

            distance_outbound_nm = None
            distance_inbound_nm = None
            fuelcons_outbound_l = None
            fuelcons_inbound_l = None
            start_time_outbound = None
            end_time_outbound = None
            start_time_inbound = None
            end_time_inbound = None

            if outbound_trip is not None:

                # Determine the start and end time for the outbound trip
                start_time_outbound = pd.Timestamp(outbound_trip["time"][0])
                end_time_outbound = pd.Timestamp(outbound_trip["time"][-1])

                # Calculate the total distance for the outbound trip
                distance_outbound_nm = sum(
                    [
                        haversine(
                            outbound_trip["path"][i], outbound_trip["path"][i - 1]
                        )
                        / 1_852
                        for i in range(1, len(outbound_trip["path"]))
                    ]
                )

                # Calculate the total fuel consumption for the outbound trip
                for key in outbound_trip.keys():
                    if "fuelcons_lph" in key:
                        if any([value == None for value in outbound_trip[key]]):
                            break
                        if fuelcons_outbound_l is None:
                            fuelcons_outbound_l = 0
                        fuelcons_outbound_l += sum(
                            [
                                fuelcons_lph * time_bucket_s / 3_600
                                for fuelcons_lph in outbound_trip[key]
                            ]
                        )

            if inbound_trip is not None:

                # Determine the start and end time for the inbound trip
                start_time_inbound = pd.Timestamp(inbound_trip["time"][0])
                end_time_inbound = pd.Timestamp(inbound_trip["time"][-1])

                # Calculate the total distance for the inbound trip
                distance_inbound_nm = sum(
                    [
                        haversine(inbound_trip["path"][i], inbound_trip["path"][i - 1])
                        / 1_852
                        for i in range(1, len(inbound_trip["path"]))
                    ]
                )

                # Calculate the total fuel consumption for the inbound trip
                for key in inbound_trip.keys():
                    if "fuelcons_lph" in key:
                        if any([value == None for value in inbound_trip[key]]):
                            break
                        if fuelcons_inbound_l is None:
                            fuelcons_inbound_l = 0
                        fuelcons_inbound_l += sum(
                            [
                                fuelcons_lph * time_bucket_s / 3_600
                                for fuelcons_lph in inbound_trip[key]
                            ]
                        )

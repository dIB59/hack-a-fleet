import os
import math
import pandas as pd
import numpy as np
import requests
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PONTOS_TOKEN = os.getenv("PONTOS_TOKEN")

if not PONTOS_TOKEN:
    raise Exception("PONTOS_TOKEN not found in environment variables")


def fetch_vessel_data(
    vessel_id, start_time, end_time, parameter_ids=["*"], time_bucket=None
):
    """
    Fetches vessel data from the PONTOS-hub API within a specified time range and for specified parameters.

    Args:
        vessel_id (str): The unique identifier of the vessel.
        start_time (str): The start time for data fetching in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).
        end_time (str): The end time for data fetching in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).
        parameter_ids (list of str, optional): A list of parameter IDs to filter the data. Defaults to ['*'].
        time_bucket (str, optional): The time bucket for averaging the data. Valid options are:
            "5 seconds", "30 seconds", "1 minute", "5 minutes", "10 minutes". Defaults to None.

    Returns:
        dict: The response from the PONTOS-hub API containing the vessel data.
    """
    # Convert string dates to datetime objects
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)

    if start >= end:
        raise ValueError("'start_time' must be before 'end_time'")

    if start < datetime.fromisoformat("2023-04-30T22:00:00"):
        raise ValueError(
            "'start_time' must be before 2023-04-30T22:00:00. PONTOS-hub does not contain data before this time."
        )

    # Choose appropriate view of vessel data view
    averaged_vessel_data_views = {
        "5 seconds": "vessel_data_5_seconds_average",
        "30 seconds": "vessel_data_30_seconds_average",
        "1 minute": "vessel_data_1_minute_average",
        "5 minutes": "vessel_data_5_minutes_average",
        "10 minutes": "vessel_data_10_minutes_average",
    }
    api_view = (
        "vessel_data"
        if time_bucket is None
        else averaged_vessel_data_views.get(time_bucket, None)
    )
    if api_view is None:
        valid_keys = ", ".join([key for key in averaged_vessel_data_views.keys()])
        raise ValueError(
            f"Invalid time_bucket '{time_bucket}'. Use one of the following: {valid_keys}"
        )

    # Construct the parameter_id filter
    parameter_id_filter = "".join(
        [f"parameter_id.ilike.*{param}*," for param in parameter_ids]
    )
    parameter_id_filter = parameter_id_filter[:-1]  # Remove the trailing comma

    # Format query string with the current time bounds
    query = f"or=({parameter_id_filter})" f"&vessel_id=eq.{vessel_id}"
    if api_view != "vessel_data":
        query += f"&bucket=gte.{start.isoformat()}&bucket=lt.{end.isoformat()}"
    else:
        query += f"&time=gte.{start.isoformat()}&time=lt.{end.isoformat()}&select=time,parameter_id,value::float"

    # Make the API request
    url = f"https://pontos.ri.se/api/{api_view}?{query}"
    response = requests.get(url, headers={"Authorization": f"Bearer {PONTOS_TOKEN}"})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            "Failed to retrieve data:", response.status_code, response.text, url
        )


def transform_vessel_data_to_dataframe(vessel_data):
    """
    Transforms vessel data into a Pandas DataFrame.

    Args:
        vessel_data (list of dict): A list of dictionaries containing vessel data returned by the PONTOS REST-API.

    Returns:
        pandas.DataFrame: A DataFrame where the index is the time, columns are parameter IDs, and values are the
                          corresponding data values. The DataFrame is pivoted to have 'parameter_id' as columns
                          and 'time' as rows.
    """

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(vessel_data)

    # Convert the time related columns to datetime format
    if "avg_time" in df.columns:
        df.rename(columns={"avg_time": "time", "avg_value": "value"}, inplace=True)
        df["bucket"] = pd.to_datetime(df["bucket"])
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")

    # Pivot the DataFrame to have parameter_ids as columns, time as rows
    pivot_df = df.pivot_table(
        index="time", columns="parameter_id", values="value", aggfunc="first"
    ).reset_index()

    # Rename index
    pivot_df.index.name = "id"

    return pivot_df


def get_trips_from_vessel_data(
    vessel_data,
    speed_threshold_kn=1.0,
    stop_time_threshold_min=1.0,
    lat="positioningsystem_latitude_deg_1",
    lon="positioningsystem_longitude_deg_1",
    sog="positioningsystem_sog_kn_1",
    time_zone="CET",
):
    """
    Processes vessel data to extract trips based on speed and stop time thresholds.

    Args:
        vessel_data (list): A list of dictionaries containing vessel data points.
        speed_threshold_kn (float, optional): The speed threshold in knots below which data points are considered stops. Defaults to 1.0 kn.
        stop_time_threshold_min (float, optional): The time threshold in minutes to consider a stop between trips. Defaults to 1.0 miinute.
        lat (str, optional): The key for latitude in the vessel data. Defaults to "positioningsystem_latitude_deg_1".
        lon (str, optional): The key for longitude in the vessel data. Defaults to "positioningsystem_longitude_deg_1".
        sog (str, optional): The key for speed over ground in the vessel data. Defaults to "positioningsystem_sog_kn_1".
        time_zone (str, optional): The time zone to which the 'time' column should be converted. Defaults to 'CET'.

    Returns:
        list: A list of dictionaries, each representing a trip. Each dictionary contains:
            - "path": A list of tuples with latitude and longitude points.
            - "time": A list of ISO8601 formatted timestamps.
            - Other attributes from the vessel data excluding latitude, longitude, and time.
    """

    # Transform vessel data to a Dataframe
    df = transform_vessel_data_to_dataframe(vessel_data)

    # Return empty list if the DataFrame is missing the required columns
    if lat not in df.columns or lon not in df.columns or sog not in df.columns:
        return []

    # Drop data points where latitude, longitude, or speed over ground is NaN
    df = df.dropna(subset=[lat, lon, sog])

    # Drop data points where the speed is below 0.5 kn
    df = df.drop(df[df[sog] < speed_threshold_kn].index)

    # Add column with time between messages (dt)
    df["dt"] = df["time"].diff().dt.total_seconds()

    # Transform time to timezone and ISO8601 format strings
    df["time"] = df["time"].dt.tz_convert(time_zone).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Split data into trips at time gaps ( dt > stop_time_threshold_min)
    trips = []
    for group in np.split(df, np.where(df.dt > stop_time_threshold_min * 60)[0]):
        path = [(p[0], p[1]) for p in group[[lat, lon]].to_records(index=False)]
        attributes = group[group.columns.difference([lat, lon, "dt"])].to_dict(
            orient="list"
        )
        trips.append({"path": path, **attributes})

    # Remove trips with less than 2 points
    trips = [trip for trip in trips if len(trip["path"]) > 1]

    return trips


def haversine(point_1, point_2):
    """
    Calculate the great-circle distance between two points on the Earth using the Haversine formula.

    Args:
        point_1 (tuple): A tuple containing the latitude and longitude of the first point (in decimal degrees)
        point_2 (tuple): A tuple containing the latitude and longitude of the second point (in decimal degrees)

    Returns:
        float: The great-circle distance between the two points in meters
    """

    R = 6371000  # Earth's radius in meters

    lat1, lon1 = point_1
    lat2, lon2 = point_2

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2) + math.cos(lat1_rad) * math.cos(lat2_rad) * (
        math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

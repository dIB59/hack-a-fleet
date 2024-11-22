# Hack-A-Fleet v2.0

This is the repository containing Python tools and datasets for _Hack-A-Fleet v2.0_.

## TLDR;

The file `ferry_trips_data.csv` contains data of multiple trips made by 5 ferries owned by Färjerederiet. The file `utils.py` contains functions that you can use to fetch and work with data from PONTOS-HUB; the file `examples.ipynb` shows how.

## The challenge

The challenge is to develop new tools or create new information/knowledge/insights that can help Färjerederiet to achieve its Vision 45, becoming climate neutral by 2045 at the latest, using a real-world dataset from a subset of Färjerederiets fleet.

The real-world dataset is `ferry_trips_data.csv`. Use this data as a start point, but feel free to complement and/or expand it by using historical and real-time data in PONTOS-HUB or from any other sources.

## File descriptions

- `ferries.json`: A JSON file containing information of the ferries owned by Färjerederiet that share their data to PONTOS-HUB. The `pontos_vessel_id` key-value pair can be use for querying the REST-API of PONTOS-HUB.
- `ferry_trips_data.csv`: A CSV file containing records of trips made by 5 ferries owned by Färjerederiet. See the section "Ferry trips" below for more information about the contents.
- `ferry_route_descriptions.md`: Descriptions of the ferry routes in the file `ferry_trips_data.csv`.
- `schedures/*.pdf`: Schedules for the ferry routes.
- `utils.py`: A Python module containing utility functions to fetch vessel data from PONTOS-HUB, manipulate it, and visualize it.
- `examples.ipynb`: A Jupyter notebook containing examples of the useage of the functions in `utils.py`.
- `excel/ferry_trips.xlsx`: An Excel file containing records of trips made by 5 ferries owned by Färjerederiet. Used as an input for the script `exel/extend_ferrytrips.py`.
- `excel/extend_ferry_trips.py`: A Python script that extends the trip information of `excel/ferry_trips.xlsx` using the data in PONTOS-HUB (e.g. distance and fuel consumption). Generates the file `ferry_trips_data.csv`. Uses data averaged within a 5 seconds time bucket.

## Get Started with the Python tools

1. Clone the repository or download all the files (see Releases) to your computer.

2. If you do not have a TOKEN to access the data in PONTOS-HUB, get one [here](https://pontos.ri.se/get_started).

3. Create a file called `.env` and save it the root directory (i.e. the same directory that contains `utils.py`). The content of the file must be:

```
PONTOS_TOKEN=<Your PONTOS access token>
```

This file is read by `utils.py` when loaded so that the requests to PONTOS-HUB are authorized.

5. Create a virtual environment (e.g. using `venv`or `conda`).

6. Install all the dependencies `pip install -r requirements.txt`.

7. Take a look at `examples.ipynb`.

## Ferry trips

The file `ferry_trips_data.csv` contains data from 5 ferries owned and operated by Färjerederiet:

- Fragancia
- Jupiter
- Merkurius
- Nina
- Yxlan

The data corresponds to the time period between 2023-03-01 and 2024-02-29. PONTOS-HUB launched on 2023-04-30, so the data contains datapoints corresponding to a period for which PONTOS-HUB has no data.

Each row in the `ferry_trips_data.csv` files contain fields describing 2 trips between two terminals:

    - An `outbound` trip from the departure terminal to the arrival terminal.
    - An `inbound` trip from the arrival terminal to the departure terminal.

The suffixes `outbound` and `inbound` in the field names indicate to which trip does the field corresponds to.

Here follows a description of _some_ of the fields as most of them are self-explanatory:

Original fields:

- `time_departure`: Time of departure for the outbound trip as recoreded by Färjerederiet. Given in Central European Time (CET).
- `vehicles_left_at_terminal_outbound/inbound`: Number of vehicles left at the terminal on departure for the outbound/inbound trip. _Estimated by the crew._
- `trip_type`: One of the following types of trip:
  - `ordinary`- Ordinary trip.
  - `doubtful` – An "ordinary" trip that does not match the timetable.
  - `extra` – An extra trip by an additional ferry that does not follow the timetable.
  - `proactive` – A trip made before the ordinary trip to stay ahead, comparable to an extra trip.
  - `doubling` – An extra trip between two regular trips to take car of vehicles left behind in the termial.
- `tailored_trip`: A special trip for vehicles with dangerous cargo. (1: True, 0: False).
- `passenger_car_equivalent_outbound/inbound`: The total number of vehicles in a outbound/inbound trip as Passenger Car Equivalent (PCE) according to the following conversion rules:
  - Length 0-6 meters (e.g. car): 1 PCE
  - Length 6-12 meters (e.g. lorry or car with trailer): 2.5 PCE
  - Length 15-24 meters (e.g. lorry with trailer): 4.5 PCE
  - Bus: 9 PCE
  - Other large vehicles (e.g. cranes, haversters): 9 PCE

Additional fields calculated with data from PONTOS-HUB (might not always contain values depending on data availability):

- `distance_outbound/inbound_nm`: Approximate distance travelled in the outbound/inbound trip. Calculated from PONTOS-HUB data and given in nautical miles.
- `fuelcons_outbound/inbound_l`: Approximate fuel consumption in the outbound/inbound trip. Calculated from PONTOS-HUB data and given in liters.
- `start_time_outbound/inbound`: Approximate start time of the outbound/inbound trip. Calculated from PONTOS-HUB data and given in CET.
- `end_time_outbound/inbound`: Approximate end time of the outbound/inbound trip. Calculated from PONTOS-HUB data and given in CET.



import pandas as pd
import solarpy
import pvlib
from urllib.error import HTTPError

# Constants 
FORECAST_STATION = "06019"
FORECAST_START = "2026-11-09"
FORECAST_END = "2026-11-11"
FORECAST_LATITUDE = 56.93
FORECAST_LONGITUDE = 8.6412

FORECAST_MODELS = [
    "dmi_harmonie_arome_europe",
    "metno_nordic",
    "ecmwf_ifs025",
    "ecmwf_aifs025_single",
    "gfs_global",
    "ecmwf_ifs",
    "dmi_seamless",
    "gfs_seamless",
]




# Get dmi irradiance data

def get_dmi_ghi(start, end, station="06019"):
    """
    Retrieve raw DMI measurements of global horizontal irradiance (GHI).

    Returns the original 10-minute data in UTC. The data is not resampled
    here, so the raw measurements remain available for quality control
    and plotting.

    Parameters
    ----------
    start, end : str or datetime-like
        Start and end of the retrieval period.
    station : str, default "06019"
        DMI station code.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a UTC DatetimeIndex and a 'ghi' column.
    """
    data, _ = solarpy.iotools.get_dmi_metobs(
        station=station,
        start=start,
        end=end,
        parameters=["radia_glob"],
        map_variables=True,
        timeout=30,
    )

    if data.empty:
        raise ValueError(
            f"No DMI data found for station {station} "
            f"between {start} and {end}."
        )

    data = data.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    if "ghi" not in data.columns and "radia_glob" in data.columns:
        data = data.rename(columns={"radia_glob": "ghi"})

    if "ghi" not in data.columns:
        raise KeyError(
            "Could not find the GHI column. "
            f"Received columns: {list(data.columns)}"
        )

    data["ghi"] = pd.to_numeric(data["ghi"], errors="coerce")
    data = data[["ghi"]]

    return data


# convert to hourly data 
def resample_dmi_ghi_hourly(data, min_observations=6):
    """
    Resample 10-minute DMI GHI measurements to hourly values.

    Timestamps are right-labeled and intervals are right-closed. For
    example, the hourly value at 10:00 includes measurements after 09:00
    up to and including 10:00.

    Hours with fewer than `min_observations` valid measurements are set
    to NaN so that incomplete hours are not included in the evaluation.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a 'ghi' column and a DatetimeIndex.
    min_observations : int, default 6
        Minimum number of valid 10-minute measurements required per hour.

    Returns
    -------
    pandas.DataFrame
        Hourly GHI values and the number of observations in each hour.
    """
    if "ghi" not in data.columns:
        raise KeyError("Input data must contain a 'ghi' column.")

    data = data.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    hourly = data[["ghi"]].resample(
        "1h",
        label="right",
        closed="right",
    ).agg(
        ghi_measured=("ghi", "mean"),
        n_observations=("ghi", "count"),
    )

    hourly.loc[
        hourly["n_observations"] < min_observations,
        "ghi_measured",
    ] = float("nan")

    return hourly


# Get forecast data from GitHub repository

URL = (
    "https://raw.githubusercontent.com/AdamRJensen/"
    "solar-resource-course-2026/refs/heads/main/forecast-data/"
)

FORECAST_STATION = "06019"
FORECAST_START = "2026-11-09"
FORECAST_END = None  # Uses the current UTC time when the function is called

FORECAST_MODELS = [
    "dmi_harmonie_arome_europe",
    "metno_nordic",
    "ecmwf_ifs025",
    "ecmwf_aifs025_single",
    "gfs_global",
    "ecmwf_ifs",
    "dmi_seamless",
    "gfs_seamless",
]


def get_forecast_data(
    hourly_measurements,
    station=FORECAST_STATION,
    start=FORECAST_START,
    end=FORECAST_END,
    models=FORECAST_MODELS,
    url=URL,
):
    """
    Retrieve forecast files, match them with measurements,
    and return one DataFrame per forecast model.
    """

    # Set the issue-time interval
    start = pd.Timestamp(start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    if end is None:
        end = pd.Timestamp.now(tz="UTC").floor("h")
    else:
        end = pd.Timestamp(end)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

    issue_times = pd.date_range(start, end, freq="1h")

    # Download each forecast file
    forecast_frames = []

    for issue_time in issue_times:
        filename = (
            f"{station}_{issue_time.strftime('%Y%m%dT%H%M')}.csv"
        )

        try:
            forecast = pd.read_csv(
                url + filename,
                index_col=0,
                parse_dates=[0],
            )
        except HTTPError:
            continue

        forecast.index = pd.to_datetime(forecast.index, utc=True)
        forecast.index.name = "valid_time"

        # Keep forecast values at or after the issue time,
        # following the course parsing notebook.
        forecast = forecast.loc[forecast.index >= issue_time]

        lead_time_h = (
            (forecast.index - issue_time).total_seconds() / 3600
        )

        forecast.index = pd.MultiIndex.from_arrays(
            [forecast.index, lead_time_h],
            names=["valid_time", "lead_time_h"],
        )

        forecast_frames.append(forecast)

    forecasts = pd.concat(forecast_frames).sort_index()

    # Prepare the hourly measurements for matching
    measurements = hourly_measurements[["ghi_measured"]].copy()
    measurements.index = pd.to_datetime(measurements.index, utc=True)
    measurements.index.name = "valid_time"

    # Create one matched DataFrame per model
    forecast_data = {}

    for model in models:
        model_forecast = (
            forecasts[model]
            .unstack()
            .stack()
            .dropna()
            .rename("ghi_forecast")
            .rename_axis(["valid_time", "lead_time_h"])
            .reset_index()
        )

        model_forecast["issue_time"] = (
            model_forecast["valid_time"]
            - pd.to_timedelta(model_forecast["lead_time_h"], unit="h")
        )

        model_forecast = model_forecast.merge(
            measurements.reset_index(),
            on="valid_time",
            how="left",
        )

        model_forecast["error"] = (
            model_forecast["ghi_forecast"]
            - model_forecast["ghi_measured"]
        )

        forecast_data[model] = model_forecast[
            [
                "issue_time",
                "valid_time",
                "lead_time_h",
                "ghi_forecast",
                "ghi_measured",
                "error",
            ]
        ]

    return forecast_data
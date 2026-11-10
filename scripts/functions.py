

import pandas as pd
import solarpy
import pvlib

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

def prep_dmi_irradiance(start, end, station="06019"):
    """
    Retrieve and prepare DMI global irradiance data.

    The 10-minute measurements are resampled to hourly averages.
    Each hourly value is right-labeled and uses a right-closed interval,
    matching the forecast timestamps.

    Parameters
    ----------
    start : str or datetime-like
        First date to retrieve, for example "2024-01-01".
    end : str or datetime-like
        Last date to retrieve, for example "2024-12-31".
    station : str, default "06019"
        DMI station code.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a UTC DatetimeIndex and a 'ghi' column,
        resampled to hourly averages.
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

    # Use a consistent name for global horizontal irradiance.
    if "ghi" not in data.columns and "radia_glob" in data.columns:
        data = data.rename(columns={"radia_glob": "ghi"})

    if "ghi" not in data.columns:
        raise KeyError(
            "The data does not contain an irradiance column. "
            f"Received columns: {list(data.columns)}"
        )

    data["ghi"] = pd.to_numeric(data["ghi"], errors="coerce")

    return data


# Get forecast data from GitHub repository

URL = "https://raw.githubusercontent.com/AdamRJensen/solar-resource-course-2026/refs/heads/main/forecast-data/"
from urllib.error import HTTPError

def get_forecast_data(station, start, end, url=URL):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize('UTC')
    else:
        start = start.tz_convert('UTC')
    if end.tzinfo is None:
        end = end.tz_localize('UTC')
    else:
        end = end.tz_convert('UTC')

    times = pd.date_range(start, end, freq='1h')

    dfs = []
    for t in times:
        filename = f"{station}_{t.strftime('%Y%m%dT%H%M')}.csv"
        try:
            dfi = pd.read_csv(url + filename, index_col=0, parse_dates=[0])
            dfi = dfi[dfi.index >= t]  # drop rows with timestamps before the forecast time
            lead_time = (dfi.index - t).total_seconds() / (60*60)
            dfi.index = [dfi.index, lead_time]
            dfs.append(dfi)
        except HTTPError:
            print(f"File not found: {filename}")
    if dfs:
        return pd.concat(dfs)
    else:
        raise FileNotFoundError("No forecast data files found.")




def prepare_forecast_comparison(hourly_data, models):
    """
    Retrieve daytime forecasts for selected models and match them with
    hourly measured GHI.

    Parameters
    ----------
    hourly_data : pandas.DataFrame
        Measurement data with a DatetimeIndex and a 'ghi' column.
        The input may contain 10-minute or hourly measurements.
    models : str or list of str
        Forecast model name or names to include.

    Returns
    -------
    pandas.DataFrame
        Long-form data with valid_time, lead_time, model,
        ghi_forecast, and ghi_measured columns.
    """
    if isinstance(models, str):
        models = [models]

    forecasts = get_forecast_data(
        station=FORECAST_STATION,
        start=FORECAST_START,
        end=FORECAST_END,
    )

    forecasts.index = forecasts.index.set_names(["valid_time", "lead_time"])

    timestamps = pd.DatetimeIndex(forecasts.index.get_level_values(0))
    solpos = pvlib.solarposition.get_solarposition(
        timestamps,
        latitude=FORECAST_LATITUDE,
        longitude=FORECAST_LONGITUDE,
        method="nrel_numpy",
    )
    solar_elevation = solpos["apparent_elevation"]

    #filter forecasts to include only daytime hours (solar elevation > 0)
    forecasts_day = forecasts.loc[solar_elevation.to_numpy() > 0]

    # unstack the forecasts for each model to have valid_time as index and lead_time as columns
    model_forecasts = {
        model: forecasts_day[model].unstack(level=1)
        for model in models
    }

    forecast_long = pd.concat(
        [
            model_data.stack()
            .rename("ghi_forecast")
            .rename_axis(["valid_time", "lead_time"])
            .reset_index()
            .assign(model=model)
            for model, model_data in model_forecasts.items()
        ],
        ignore_index=True,
    )

    measurements = hourly_data[["ghi"]].copy()
    measurements.index = pd.to_datetime(measurements.index, utc=True)
    measurements = measurements.sort_index()

    measurements = measurements.resample(
        "1h",
        label="right",
        closed="right",
    ).mean()

    measurements = (
        measurements.rename_axis("valid_time")
        .rename(columns={"ghi": "ghi_measured"})
        .reset_index()
    )

    comparison = forecast_long.merge(
        measurements,
        on="valid_time",
        how="left",
        validate="many_to_one",
    )

    return comparison





def calculate_forecast_metrics(comparison):
    """
    Calculate bias and RMSE for each model and forecast lead time.

    Bias is defined as forecast minus measurement.
    """
    valid_data = comparison.dropna(
        subset=["ghi_forecast", "ghi_measured"]
    ).copy()

    valid_data["error"] = (
        valid_data["ghi_forecast"] - valid_data["ghi_measured"]
    )

    metrics = (
        valid_data.groupby(["model", "lead_time"])
        .agg(
            bias=("error", "mean"),
            rmse=("error", lambda errors: (errors.pow(2).mean()) ** 0.5),
            n=("error", "count"),
        )
        .reset_index()
    )

    return metrics
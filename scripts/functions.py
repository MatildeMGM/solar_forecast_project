

import pandas as pd
import solarpy


def prep_dmi_irradiance(start, end, station="06019"):
    """
    Retrieve and prepare DMI irradiance data for a forecasting model.

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
        DataFrame with a UTC DatetimeIndex and a 'ghi' column.
        Measurements are resampled to hourly averages.
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

    # Ensure the time index is sorted and in UTC.
    data = data.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    # Use a consistent name for global irradiance.
    if "ghi" not in data.columns and "radia_glob" in data.columns:
        data = data.rename(columns={"radia_glob": "ghi"})

    if "ghi" not in data.columns:
        raise KeyError(
            "The data does not contain an irradiance column. "
            f"Received columns: {list(data.columns)}"
        )

    # Convert values to numeric and calculate hourly averages.
    data["ghi"] = pd.to_numeric(data["ghi"], errors="coerce")
    data = data[["ghi"]].resample("1h").mean()

    # Remove hours where the target variable is missing.
    data = data.dropna(subset=["ghi"])
    data.index.name = "timestamp"

    return data
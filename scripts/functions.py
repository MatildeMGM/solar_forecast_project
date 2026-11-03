

import pandas as pd
import solarpy


import pandas as pd
import solarpy


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
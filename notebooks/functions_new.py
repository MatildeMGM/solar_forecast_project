

import pandas as pd
import numpy as np
import solarpy
import pvlib
from urllib.error import HTTPError

# Constants 
FORECAST_STATION = "06019"
FORECAST_START = "2026-11-09"
FORECAST_END = None
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
    if not isinstance(min_observations, int) or not 1 <= min_observations <= 6:
        raise ValueError("min_observations must be an integer between 1 and 6.")

    data = data.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Duplicate measurement timestamps must be resolved before resampling.")
    if not (data.index == data.index.floor("10min")).all():
        raise ValueError("Measurements must lie on the 10-minute timestamp grid.")
    data["ghi"] = pd.to_numeric(data["ghi"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

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
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise

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

    if not forecast_frames:
        raise ValueError(f"No forecast files found for station {station} between {start} and {end}.")
    forecasts = pd.concat(forecast_frames).sort_index()

    # Prepare the hourly measurements for matching
    measurements = hourly_measurements[["ghi_measured"]].copy()
    measurements.index = pd.to_datetime(measurements.index, utc=True)
    measurements.index.name = "valid_time"
    if measurements.index.has_duplicates:
        raise ValueError("Hourly measurements must have unique timestamps.")

    # Create one matched DataFrame per model
    forecast_data = {}

    for model in models:
        model_forecast = (
            forecasts[model]
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


def calculate_forecast_metrics(forecast_unstacked, hourly_measurements):
    """Beregn RMSE, bias, MAE og antal målinger for hver model og prognoselængde.

    Bias beregnes som målt GHI minus prognosticeret GHI.
    """
    measured = hourly_measurements["ghi_measured"]
    results = {}

    for model, forecast in forecast_unstacked.items():
        forecast = forecast.replace([np.inf, -np.inf], np.nan)

        observations = measured.reindex(forecast.index).replace(
            [np.inf, -np.inf], np.nan
        )

        # Positiv fejl betyder, at prognosen ligger under målingen.
        errors = forecast.rsub(observations, axis=0)

        scores = pd.DataFrame({
            "rmse": errors.pow(2).mean().pow(0.5),
            "bias": errors.mean(),
            "mae": errors.abs().mean(),
            "n": errors.count(),
        })

        scores.index.name = "lead_time_h"

        # Udelad lead time 0, da den timebaserede måling allerede er afsluttet.
        results[model] = scores.loc[scores.index >= 1].sort_index()

    return pd.concat(results, names=["model"]).reset_index()

def calculate_overall_metrics(scores):
    rows = []

    for model, group in scores.groupby("model"):
        group = group.loc[group["n"] > 0]
        n = group["n"].sum()

        if n == 0:
            continue

        rows.append({
            "model": model,
            "rmse": np.sqrt(
                (group["rmse"] ** 2 * group["n"]).sum() / n
            ),
            "bias": (group["bias"] * group["n"]).sum() / n,
            "mae": (group["mae"] * group["n"]).sum() / n,
            "n": n,
        })

    return pd.DataFrame(rows)



def _interval_values(forecast, first_hour, last_hour):
    """Return forecast values in one lead-time interval as a stacked Series."""
    columns = [
        lead for lead in forecast.columns
        if first_hour <= lead <= last_hour
    ]

    if not columns:
        return pd.Series(dtype=float, name=forecast.columns.name)

    subset = forecast.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    subset = subset.copy()
    subset.index.name = "valid_time"
    subset.columns.name = "lead_time_h"
    values = subset.stack(dropna=True)
    values.index.names = ["valid_time", "lead_time_h"]
    return values


def _matched_interval_values(forecast, measurements, first_hour, last_hour):
    """Return forecast and measurement pairs for an interval."""
    values = _interval_values(forecast, first_hour, last_hour)

    if values.empty:
        return pd.DataFrame(columns=["forecast", "measured"])

    measured = measurements["ghi_measured"].copy()
    measured.index = pd.to_datetime(measured.index, utc=True)
    valid_times = values.index.get_level_values("valid_time")
    observations = measured.reindex(valid_times)
    observations.index = values.index

    paired = pd.concat(
        [values.rename("forecast"), observations.rename("measured")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    return paired


def fit_blend_weights_by_lead_time(
    forecast_train,
    forecast_validation,
    hourly_measurements,
    intervals,
    min_train_samples=30,
    min_validation_samples=10,
    max_iter=500,
):
    """Optimize weights for all eligible models, then validate the blend.

    For each lead-time interval, nonnegative weights summing to one are
    optimized across all models with sufficient individual train and
    validation data. Optimization uses only complete common forecast pairs.
    The candidate blend is kept only if its validation RMSE is lower than the
    training-selected best single model on those same validation pairs.
    Otherwise, the returned schedule falls back to that single model.

    Returns
    -------
    weight_schedule : list
        Tuples of (first_hour, last_hour, weights), ready for
        ``our_blend_by_lead_time``.
    diagnostics : pandas.DataFrame
        One decision row per interval, including candidate and applied weights.
    optimization_trace : pandas.DataFrame
        Training RMSE and model weights at each optimizer iteration.
    """
    from scipy.optimize import minimize

    if min_train_samples < 1 or min_validation_samples < 1:
        raise ValueError("Minimum sample counts must be positive.")

    weight_schedule = []
    diagnostic_rows = []
    trace_rows = []

    for first_hour, last_hour in intervals:
        if first_hour > last_hour:
            raise ValueError("Invalid lead-time interval.")

        label = f"{first_hour}–{last_hour} h"
        train_pairs = {
            model: _matched_interval_values(
                forecast, hourly_measurements, first_hour, last_hour
            )
            for model, forecast in forecast_train.items()
        }
        validation_pairs = {
            model: _matched_interval_values(
                forecast_validation[model],
                hourly_measurements,
                first_hour,
                last_hour,
            )
            for model in forecast_train
        }

        train_rmse = {
            model: np.sqrt(
                pairs["forecast"].sub(pairs["measured"]).pow(2).mean()
            )
            for model, pairs in train_pairs.items()
            if len(pairs) > 0
        }
        if not train_rmse:
            raise ValueError(f"No matched training forecasts in interval {label}.")

        ranked_models = sorted(train_rmse, key=train_rmse.get)
        eligible_models = [
            model for model in ranked_models
            if len(train_pairs[model]) >= min_train_samples
            and len(validation_pairs[model]) >= min_validation_samples
        ]
        candidate_models = eligible_models.copy()
        excluded_models = []

        def make_complete_pairs(models, pairs_by_model):
            if not models:
                return pd.DataFrame()
            forecasts = pd.concat(
                {
                    model: pairs_by_model[model]["forecast"]
                    for model in models
                },
                axis=1,
            )
            measurements = pairs_by_model[models[0]]["measured"].rename("measured")
            return pd.concat([forecasts, measurements], axis=1).dropna()

        train = make_complete_pairs(candidate_models, train_pairs)
        validation = make_complete_pairs(candidate_models, validation_pairs)

        # If some models have patchier coverage, remove the least available
        # models until the common comparison has enough samples in both splits.
        while candidate_models and (
            len(train) < min_train_samples
            or len(validation) < min_validation_samples
        ):
            if len(candidate_models) == 1:
                break
            least_available = min(
                candidate_models,
                key=lambda model: len(train_pairs[model])
                + len(validation_pairs[model]),
            )
            candidate_models.remove(least_available)
            excluded_models.append(least_available)
            train = make_complete_pairs(candidate_models, train_pairs)
            validation = make_complete_pairs(candidate_models, validation_pairs)

        fallback_model = ranked_models[0]
        if not train.empty and candidate_models:
            fallback_model = min(
                candidate_models,
                key=lambda model: np.sqrt(
                    train[model].sub(train["measured"]).pow(2).mean()
                ),
            )

        fallback_weights = {fallback_model: 1.0}
        candidate_weights = {}
        candidate_train_rmse = np.nan
        candidate_validation_rmse = np.nan
        baseline_train_rmse = np.nan
        baseline_validation_rmse = np.nan
        n_train = len(train)
        n_validation = len(validation)
        accepted = False
        status = "Fallback: insufficient common data"

        if not train.empty and candidate_models:
            baseline_train_rmse = np.sqrt(
                train[fallback_model].sub(train["measured"]).pow(2).mean()
            )
            if not validation.empty:
                baseline_validation_rmse = np.sqrt(
                    validation[fallback_model]
                    .sub(validation["measured"])
                    .pow(2)
                    .mean()
                )

        if (
            len(candidate_models) >= 2
            and n_train >= min_train_samples
            and n_validation >= min_validation_samples
        ):
            model_count = len(candidate_models)
            x_train = train[candidate_models].to_numpy(dtype=float)
            y_train = train["measured"].to_numpy(dtype=float)
            x_validation = validation[candidate_models].to_numpy(dtype=float)
            y_validation = validation["measured"].to_numpy(dtype=float)
            initial_weights = np.full(model_count, 1 / model_count)

            def rmse(weights, x_values, y_values):
                return np.sqrt(np.mean((x_values @ weights - y_values) ** 2))

            def record_iteration(weights):
                row = {
                    "interval": label,
                    "iteration": len([
                        item for item in trace_rows
                        if item["interval"] == label
                    ]),
                    "train_rmse": rmse(weights, x_train, y_train),
                }
                row.update({
                    f"weight__{model}": float(weight)
                    for model, weight in zip(candidate_models, weights)
                })
                trace_rows.append(row)

            record_iteration(initial_weights)
            result = minimize(
                lambda weights: np.mean((x_train @ weights - y_train) ** 2),
                initial_weights,
                method="SLSQP",
                bounds=[(0, 1)] * model_count,
                constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
                callback=record_iteration,
                options={"maxiter": max_iter, "ftol": 1e-10},
            )

            if result.success and np.isfinite(result.x).all():
                optimized = np.clip(result.x, 0, 1)
                optimized /= optimized.sum()
                candidate_weights = {
                    model: float(weight)
                    for model, weight in zip(candidate_models, optimized)
                }
                candidate_train_rmse = rmse(optimized, x_train, y_train)
                candidate_validation_rmse = rmse(
                    optimized, x_validation, y_validation
                )
                if (
                    candidate_validation_rmse < baseline_validation_rmse
                ):
                    accepted = True
                    status = "Accepted: lower validation RMSE"
                else:
                    status = "Fallback: no validation improvement"
            else:
                status = "Fallback: optimizer did not converge"

            # Include the final solution if the optimizer callback did not.
            if not trace_rows or trace_rows[-1]["interval"] != label or not np.allclose(
                [trace_rows[-1].get(f"weight__{model}", np.nan) for model in candidate_models],
                optimized if result.success else initial_weights,
                equal_nan=False,
            ):
                final_weights = optimized if result.success else initial_weights
                record_iteration(final_weights)

        elif len(candidate_models) < 2:
            status = "Fallback: fewer than two eligible models"

        applied_weights = candidate_weights if accepted else fallback_weights
        weight_schedule.append((first_hour, last_hour, applied_weights))
        diagnostic_rows.append({
            "interval": label,
            "candidate_models": candidate_models,
            "excluded_models": excluded_models,
            "candidate_weights": candidate_weights,
            "applied_weights": applied_weights,
            "training_selected_single_model": fallback_model,
            "train_rmse_blend": candidate_train_rmse,
            "train_rmse_best_single": baseline_train_rmse,
            "validation_rmse_blend": candidate_validation_rmse,
            "validation_rmse_best_single": baseline_validation_rmse,
            "n_train": n_train,
            "n_validation": n_validation,
            "accepted": accepted,
            "status": status,
        })

    return (
        weight_schedule,
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(trace_rows),
    )


def our_blended_forecasts(forecast_unstacked, weights):
    """Blend models with fixed, nonnegative weights summing to one.

    Returns NaN wherever any positively weighted model is missing.
    """
    if not weights:
        raise ValueError("Provide at least one model weight.")

    weight_values = np.asarray(list(weights.values()), dtype=float)

    if (
        not np.isfinite(weight_values).all()
        or (weight_values < 0).any()
        or not np.isclose(weight_values.sum(), 1.0)
    ):
        raise ValueError("Weights must be finite, nonnegative and sum to 1.")

    blended = None

    for model, weight in weights.items():
        if weight == 0:
            continue

        forecast = forecast_unstacked[model].replace(
            [np.inf, -np.inf], np.nan
        )

        contribution = forecast * weight

        blended = (
            contribution.copy()
            if blended is None
            else blended + contribution
        )

    return blended



def our_blend_by_lead_time(forecast_unstacked, weight_schedule):
    """Apply a separate blend to each inclusive lead-time interval."""
    parts = []
    used_leads = set()

    for first_hour, last_hour, weights in weight_schedule:
        if first_hour > last_hour:
            raise ValueError("Invalid lead-time interval.")

        blended = our_blended_forecasts(forecast_unstacked, weights)

        leads = [
            lead for lead in blended.columns
            if first_hour <= lead <= last_hour
        ]

        if used_leads.intersection(leads):
            raise ValueError("Lead-time intervals must not overlap.")

        used_leads.update(leads)
        parts.append(blended.loc[:, leads])

    if not parts:
        raise ValueError("Provide at least one lead-time interval.")

    return pd.concat(parts, axis=1).sort_index(axis=1)
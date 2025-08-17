import xarray as xr
import torch


def flatten_target_dataset(ds: xr.Dataset) -> torch.Tensor:
    """Flatten a multi-variable xarray.Dataset into a 2D torch.Tensor (time, hazard * variable)."""
    # Exclude any auxiliary variables like train_mean/std
    vars_to_use = [var for var in ds.data_vars if set(ds[var].dims) >= {"hazard", "time"}]

    # Concatenate along new variable dimension
    y_flat = xr.concat([ds[var] for var in vars_to_use], dim="variable")
    y_flat["variable"] = vars_to_use  # assign variable names
    y_flat = y_flat.transpose("time", "hazard", "variable")

    # Flatten to (time, hazard * variable)
    y_np = y_flat.values.reshape(y_flat.sizes["time"], -1)

    return torch.tensor(y_np, dtype=torch.float32)


def standardize_with_stats(ds, stats):
    standardized = {}
    for var in ds.data_vars:
        da = ds[var]
        mean = stats[f"{var}_mean"]
        std = stats[f"{var}_std"]
        standardized[var] = (da - mean) / std
    return xr.Dataset(standardized)


def compute_overall_from_daily_stats(daily_stats_ds):

    overall_stats = {}

    for var in daily_stats_ds.data_vars:
        if var.endswith("_mean"):
            base_name = var[:-5]
            mean_name = f"{base_name}_mean"
            std_name = f"{base_name}_std"

            mean_da = daily_stats_ds[mean_name]
            std_da = daily_stats_ds[std_name]

            # Mean of daily means
            overall_mean = mean_da.mean(dim="day", skipna=True)

            # Pooled std calculation:
            # std_total = sqrt( mean(std_i^2 + (mu_i - mu_total)^2) ) (law of total variance)
            variance_component = std_da**2 + (mean_da - overall_mean)**2
            overall_std = (variance_component.mean(dim="day", skipna=True))**0.5

            overall_stats[f"{base_name}_mean"] = overall_mean
            overall_stats[f"{base_name}_std"] = overall_std

    return xr.Dataset(overall_stats)
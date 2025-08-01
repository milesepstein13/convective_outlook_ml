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

def standardize(x_train, x_val): 
    return x_train, x_val
    
# TODO: bring function in from prepare_input
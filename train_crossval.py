import xarray as xr
import numpy as np
from src.crossval import run_crossval

input_dir = "/glade/work/milesep/convective_outlook_ml"
target_dir = "data/processed_data"
stats_dir = "data/processed_data"

target_str = '_new'  # '' or '_new'

# compare all models, save training and validation curves along with corresponding model name, data level, model/training hyperparameters
model_names = [
    "cnn3d_huge_kernal",
    "cnn3d_3_layer",
    "cnn3d_3_layer",
    "cnn3d_4_layer",
    "cnn3d_gelu_0_5",
    "cnn3d_gelu_0_5",
    "predict_mean",
    "predict_zero",
    "predict_zero",
    "predict_mean",
    "cnn3d_3_layer_big",
    "cnn3d_3_layer",
    "cnn3d_dropout_5_5",
    "cnn3d_huge_kernal",
    "cnn3d_3_layer",
]

levels = [
    "full_glade",
    "slgt_small_glade",
    "slgt_small_glade",
    "slgt_small_glade",
    "slgt_small_glade",
    "small_glade",
    "small_glade",
    "small_glade",
    "slgt_small_glade",
    "slgt_small_glade",
    "slgt_small_glade",
    "slgt_small_glade",
    "small_glade",
    "small_glade",
    "small_glade",
]

lrs = [
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    1e-3,
    4e-4,
    5e-3,
    5e-3,
    5e-3,
    5e-3,
    5e-3,
    1e-2,
    1e-3,
    1e-3,
    1e-3,
]

batch_sizes = [
    8,
    8,
    32,
    8,
    8,
    32,
    32,
    32,
    32,
    32,
    32,
    64,
    64,
    8,
    8,
]

epochs = [50] * len(model_names)
# restarts = [True, False, False, False, False, False, False, False, False, False, False]
# Don't do linear regression with full dataset--too many parameters
for name, level, lr, batch_size, epoch in zip(model_names, levels, lrs, batch_sizes, epochs):
    if level[:4] == 'slgt':
        slgt_mod_str = '_slgt'
    else:
        slgt_mod_str = ''
    print(name, level, lr, batch_size)
    inputs = xr.open_zarr(f"{input_dir}/train_inputs_{level}.zarr")
    targets = xr.open_dataset(f"{target_dir}/train_targets{slgt_mod_str}{target_str}.nc")
    stats = xr.open_dataset(f"{stats_dir}/daily_input_stats_{level}.nc")
    scores, train_counts, val_counts = run_crossval(inputs, targets, stats, name, batch_size = batch_size, lr = lr, epochs = epoch, level = level, restart = True, target_str = target_str)
    print(f"{name}: {np.average(scores, weights = val_counts):.3f} ± {np.std(scores):.3f}")

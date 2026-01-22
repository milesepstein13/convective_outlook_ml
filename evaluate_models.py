import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import os
import pandas as pd

from src.dataset import LazyWeatherDataset
from src.preprocessing import flatten_target_dataset, standardize_with_stats, compute_overall_from_daily_stats
from src.train_loop import evaluate
from src.models import get_model, get_model_input_dims


def evaluate_model(X, y, stats, model_name, n_splits=5, batch_size=64, optimizer_class=torch.optim.Adam, lr=1e-3, criterion=nn.MSELoss(), level=None, latest = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    days = X.day.values
    kf = KFold(n_splits=n_splits, shuffle=False)
    overall_stats = compute_overall_from_daily_stats(stats)
    val_counts = []

    losses = []
    predictions = []
    targets = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(days)):

        val_counts.append(val_idx.shape[0])

        model_spec = f"{model_name}/level={level}/opt={optimizer_class.__name__}_lr={lr}_batch={batch_size}_crit={criterion.__class__.__name__}/fold={fold}"

        model_dict = f"models/{model_spec}/"

        print(f"Fold {fold}")

        print("Selecting data...")
        train_days = days[train_idx]
        val_days = days[val_idx]

        X_train = X.sel(day=train_days)
        X_val = X.sel(day=val_days)

        y_train = y.sel(time=train_days)
        y_val = y.sel(time=val_days)

        print("Computing Stats data...")
        fold_stats = compute_overall_from_daily_stats(stats.sel(day=train_days))

        # Dummy means and std to "restandardize" the data with.
        # Essentially, the subsequent standardize() will, for both the current training and validation sets
        # 1) unstandardize (since we start with data that has been standardized across the entire non-test dataset), then
        # 2) standardize the current fold's training and validation data according to the mean and std of the current fold's training set
        print("Standardizing data...")
        conversion_stats = xr.Dataset({
            v: ((fold_stats[v] - overall_stats[v]) / overall_stats[v.replace('_mean', '_std')]) if v.endswith('_mean') else (fold_stats[v] / overall_stats[v])
            for v in fold_stats.data_vars
        })

        X_train_standardized = standardize_with_stats(X_train, conversion_stats)
        X_val_standardized = standardize_with_stats(X_val, conversion_stats)

        print("Setting up datasets...")
        input_dimensions = get_model_input_dims(model_name)

        train_ds = LazyWeatherDataset(X_train_standardized, y=flatten_target_dataset(y_train), input_dimensions=input_dimensions)
        val_ds = LazyWeatherDataset(X_val_standardized, y=flatten_target_dataset(y_val), input_dimensions=input_dimensions)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # ==== Model setup ====
        x_example, y_example = next(iter(train_loader))
        input_dim = x_example.shape[1:] if x_example.ndim > 2 else x_example.shape[1]
        output_dim = y_example.shape[1] if y_example.ndim > 1 else 1

        t = None

        model = get_model(model_name, input_dim, output_dim, t).to(device)

        if latest:
            latest_path = os.path.join(model_dict, "latest.pt")
        else:
            latest_path = os.path.join(model_dict, "best.pt")
        print('loading model')
        model.load_state_dict(torch.load(latest_path, map_location=torch.device('cpu'))['model_state_dict'])
        print("evaluating model")
        val_loss, all_preds, all_targets = evaluate(model, val_loader, criterion, device, None, None)
        losses.append(val_loss)
        predictions.append(all_preds)
        targets.append(all_targets)
    return val_counts, losses, predictions, targets, model_spec.split('/fold')[0]


latest = False

if latest:
    latest_str = 'latest_'
else:
    latest_str = 'best_'

input_dir = "/glade/work/milesep/convective_outlook_ml"
target_dir = "data/processed_data"
stats_dir = "data/processed_data"

mse_path = "results/" + latest_str + "all_mse.csv"
rmse_path = "results/" + latest_str + "all_rmses.csv"
rmse_units_path = "results/" + latest_str + "all_rmses_units.csv"

# model_names = ["cnn3d_3_layer", "cnn3d_dropout_5_0", "cnn3d_dropout_5_0", "cnn3d_dropout_5_5", "cnn3d_dropout_5_5"]
# levels = ["small", "small", "small", "small", "small"]
# lrs = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
# batch_sizes = [8, 32, 64, 32, 64]

# for name, level, lr, batch_size in zip(model_names, levels, lrs, batch_sizes):

results_df = pd.read_csv('results/results.csv')
model_names = results_df['model']

if os.path.exists(mse_path):
    mse_df = pd.read_csv(mse_path)
    done_model_specs = mse_df['model']
else:
    done_model_specs = []

for model in model_names:
    if 'batch' in model and not (model.split('/')[0] in ['predict_mean', 'predict_zero']) and not (model in list(done_model_specs)):
        name = model.split('/')[0]
        level = model.split('level=')[1].split('/')[0]
        lr = float(model.split('lr=')[1].split('_')[0])
        batch_size = int(model.split('batch=')[1].split('_')[0])
        if level[:4] == 'slgt':
            slgt_mod_str = '_slgt'
        else:
            slgt_mod_str = ''
        print(name, level, lr, batch_size)
        inputs = xr.open_zarr(f"{input_dir}/train_inputs_{level}.zarr")
        tars = xr.open_dataset(f"{target_dir}/train_targets{slgt_mod_str}.nc")
        stats = xr.open_dataset(f"{stats_dir}/daily_input_stats_{level}.nc")
        sizes, losses, predictions, targets, model_spec = evaluate_model(inputs, tars, stats, name, batch_size = batch_size, lr = lr, level = level, latest = latest)

        all_preds = np.array(predictions[0])
        all_targets = np.array(targets[0])

        for i in range(1, len(predictions)):
            all_preds = np.append(all_preds, np.array(predictions[i]), axis = 0)
            all_targets = np.append(all_targets, np.array(targets[i]), axis = 0)

        mses = ((all_preds - all_targets)**2).mean(axis=0)
        rmses = np.sqrt(mses)
        stds = tars['train_std'].values.flatten()
        rmse_units = rmses * stds

        mse_row = {
            "model": model_spec,
            "0": mses[0],
            "1": mses[1],
            "2": mses[2],
            "3": mses[3],
            "4": mses[4],
            "5": mses[5],
            "6": mses[6],
            "7": mses[7],
            "8": mses[8],
            "9": mses[9],
            "10": mses[10],
            "11": mses[11]
        }

        rmse_row = {
            "model": model_spec,
            "0": rmses[0],
            "1": rmses[1],
            "2": rmses[2],
            "3": rmses[3],
            "4": rmses[4],
            "5": rmses[5],
            "6": rmses[6],
            "7": rmses[7],
            "8": rmses[8],
            "9": rmses[9],
            "10": rmses[10],
            "11": rmses[11]
        }

        rmse_units_row = {
            "model": model_spec,
            "0": rmse_units[0],
            "1": rmse_units[1],
            "2": rmse_units[2],
            "3": rmse_units[3],
            "4": rmse_units[4],
            "5": rmse_units[5],
            "6": rmse_units[6],
            "7": rmse_units[7],
            "8": rmse_units[8],
            "9": rmse_units[9],
            "10": rmse_units[10],
            "11": rmse_units[11]
        }

        os.makedirs(os.path.dirname(mse_path), exist_ok=True)
        os.makedirs(os.path.dirname(rmse_path), exist_ok=True)
        os.makedirs(os.path.dirname(rmse_units_path), exist_ok=True)

        if os.path.exists(mse_path):
            df = pd.read_csv(mse_path)
            df = df[df["model"] != mse_row["model"]]  # overwrite if it exists
            df = pd.concat([df, pd.DataFrame([mse_row])], ignore_index=True)
        else:
            df = pd.DataFrame([mse_row])

        df.to_csv(mse_path, index=False)

        if os.path.exists(rmse_path):
            df = pd.read_csv(rmse_path)
            df = df[df["model"] != rmse_row["model"]]  # overwrite if it exists
            df = pd.concat([df, pd.DataFrame([rmse_row])], ignore_index=True)
        else:
            df = pd.DataFrame([rmse_row])

        df.to_csv(rmse_path, index=False)

        if os.path.exists(rmse_units_path):
            df = pd.read_csv(rmse_units_path)
            df = df[df["model"] != rmse_units_row["model"]]  # overwrite if it exists
            df = pd.concat([df, pd.DataFrame([rmse_units_row])], ignore_index=True)
        else:
            df = pd.DataFrame([rmse_units_row])

        df.to_csv(rmse_units_path, index=False)
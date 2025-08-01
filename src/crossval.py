from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.dataset import LazyWeatherDataset
from src.preprocessing import standardize, flatten_target_dataset
from src.train_loop import train_one_epoch, evaluate
from src.models import get_model, get_model_input_dims
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import numpy as np


def run_crossval(X, y, model_name, n_splits=5, batch_size=64, epochs=5, optimizer_class=torch.optim.Adam, lr=1e-3, criterion=nn.MSELoss(), level=None, restart = False):
    days = X.day.values
    kf = KFold(n_splits=n_splits, shuffle=False)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(days)):

        model_spec = f"{model_name}/level={level}/opt={optimizer_class.__name__}_lr={lr}_crit={criterion.__class__.__name__}/fold={fold}"

        log_dir = f"runs/{model_spec}"
        writer = SummaryWriter(log_dir=log_dir)

        model_dict = f"models/{model_spec}/"
        os.makedirs(model_dict, exist_ok=True)

        print(f"\nFold {fold}:")

        # ==== Load data ====
        train_days = days[train_idx]
        val_days = days[val_idx]

        X_train = X.sel(day=train_days)
        X_val = X.sel(day=val_days)

        y_train = y.sel(time=train_days)
        y_val = y.sel(time=val_days)

        X_train_standardized, X_val_standardized = standardize(X_train, X_val)

        input_dimensions = get_model_input_dims(model_name)

        train_ds = LazyWeatherDataset(X_train_standardized, y=flatten_target_dataset(y_train), input_dimensions=input_dimensions)
        val_ds = LazyWeatherDataset(X_val_standardized, y=flatten_target_dataset(y_val), input_dimensions=input_dimensions)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # ==== Model setup ====
        x_example, y_example = next(iter(train_loader))
        input_dim = x_example.shape[1:] if x_example.ndim > 2 else x_example.shape[1]
        output_dim = y_example.shape[1] if y_example.ndim > 1 else 1

        model = get_model(model_name, input_dim, output_dim)
        optimizer = optimizer_class(model.parameters(), lr=lr)

        latest_path = os.path.join(model_dict, "latest.pt")
        best_path = os.path.join(model_dict, "best.pt")

        # ==== Resume logic ====
        start_epoch = 0
        if os.path.exists(best_path) and not restart:
            best_checkpoint = torch.load(best_path, weights_only=False)
            best_val_loss = best_checkpoint.get('val_loss', float('inf'))
        else:
            best_val_loss = float('inf')

        # Resume model from latest
        if os.path.exists(latest_path) and not restart:
            checkpoint = torch.load(latest_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            if start_epoch >= epochs:
                print(f"Fold already trained (latest epoch = {start_epoch-1} ≥ target = {epochs-1}) — skipping.")
                acc = evaluate(model, val_loader, criterion)
                results.append(acc)
                continue
            else:
                print(f"Resuming training from epoch {start_epoch}/{epochs}")

        else:
            print("Starting training from scratch")

        # ==== Training loop ====
        for epoch in range(start_epoch, epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, writer)
            val_loss = evaluate(model, val_loader, criterion, epoch, writer)

            print(f"  Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save latest
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, latest_path)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_path)

    latest_train_scores = []
    latest_val_scores = []
    best_train_scores = []
    best_val_scores = []
    best_epochs = []

    for fold in range(n_splits):
        model_spec = f"{model_name}/level={level}/opt={optimizer_class.__name__}_lr={lr}_crit={criterion.__class__.__name__}/fold={fold}"
        model_dir = os.path.join("models", model_spec)

        latest_ckpt = torch.load(os.path.join(model_dir, "latest.pt"), map_location="cpu", weights_only=False)
        best_ckpt = torch.load(os.path.join(model_dir, "best.pt"), map_location="cpu", weights_only=False)

        latest_train_scores.append(latest_ckpt["train_loss"])
        latest_val_scores.append(latest_ckpt["val_loss"])
        best_train_scores.append(best_ckpt["train_loss"])
        best_val_scores.append(best_ckpt["val_loss"])
        best_epochs.append(best_ckpt["epoch"])

    # Save to global CSV
    row = {
        "model": model_spec.split("/fold=")[0],
        "mean_latest_train_loss": np.mean(latest_train_scores),
        "std_latest_train_loss": np.std(latest_train_scores),
        "mean_latest_val_loss": np.mean(latest_val_scores),
        "std_latest_val_loss": np.std(latest_val_scores),
        "mean_best_train_loss": np.mean(best_train_scores),
        "std_best_train_loss": np.std(best_train_scores),
        "mean_best_val_loss": np.mean(best_val_scores),
        "std_best_val_loss": np.std(best_val_scores),
        "avg_best_epoch": int(round(np.mean(best_epochs))),
    }

    results_path = "results/results.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df = df[df["model"] != row["model"]]  # overwrite if it exists
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(results_path, index=False)

    return best_val_scores
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.dataset import LazyWeatherDataset
from src.preprocessing import normalize
from src.train_loop import train_one_epoch, evaluate
from src.models import get_model
import torch
import torch.nn as nn

def run_crossval(X, y, model_name, input_dim, output_dim, n_splits=5, batch_size=64, epochs=5, optimizer=torch.optim.Adam, lr = 1e-3, criterion = nn.MSELoss()):
    kf = KFold(n_splits=n_splits, shuffle=False)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx] # TODO: may need to split manually because higher dimensionality
        X_val, y_val = X[val_idx], y[val_idx]
        X_train_scaled, X_val_scaled, _ = normalize(X_train, X_val) # TODO: normalize myself based on existing stats

        train_ds = LazyWeatherDataset(torch.tensor(X_train_scaled, dtype=torch.float32), # TODO: handle dimensions properly
                             torch.tensor(y_train, dtype=torch.long))
        val_ds = LazyWeatherDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = get_model(model_name, input_dim, output_dim)
        optimizer = optimizer(model.parameters(), lr=lr)

        for _ in range(epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)

        acc = evaluate(model, val_loader, criterion)
        results.append(acc)

    return results
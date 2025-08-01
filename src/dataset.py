from torch.utils.data import Dataset
import numpy as np
import torch


class LazyWeatherDataset(Dataset):
    def __init__(self, xr_dataset, y, input_dimensions = 2):
        self.ds = xr_dataset  # full xarray dataset
        self.y = y            # (332, target_dim) torch tensor
        self._input_dimensions = input_dimensions  # number of dimensions to flatten into: 2: day, all others (fully flattened), 4: day, lat, lon, channel (includes tod and level), 5: day, lat, lon, tod, channel (includes level)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        day_data = self.ds.sel(day=self.ds.day[idx]).load()

        if self._input_dimensions == 2:
            features = []
            for var in day_data.data_vars:
                da = day_data[var]
                if "level" in da.dims:
                    stacked = da.stack(features=("level", "latitude", "longitude", "tod")).values
                else:
                    stacked = da.stack(features=("latitude", "longitude", "tod")).values
                features.append(stacked)
            x = np.concatenate(features)
            x = torch.tensor(x, dtype=torch.float32)

        elif self._input_dimensions == 4:
            channels = []
            for var in day_data.data_vars:
                da = day_data[var]
                if "level" in da.dims:
                    da = da.transpose("level", "latitude", "longitude", "tod")
                    reshaped = da.values.reshape(
                        da.sizes["level"] * da.sizes["tod"], da.sizes["latitude"], da.sizes["longitude"]
                    )
                else:
                    da = da.transpose("latitude", "longitude", "tod")
                    reshaped = da.values.transpose(2, 0, 1)  # tod, lat, lon → tod, H, W
                channels.append(reshaped)

            x = np.concatenate(channels, axis=0)  # concatenate over channels
            x = torch.tensor(x, dtype=torch.float32)  # shape: (C, H, W)
            assert x.ndim == self._input_dimensions - 1

        elif self._input_dimensions == 5:
            tod = day_data.sizes["tod"]
            channels_per_tod = []

            for var in day_data.data_vars:
                da = day_data[var]
                if "level" in da.dims:
                    da = da.transpose("level", "latitude", "longitude", "tod")
                    data = da.values  # shape: (L, H, W, T)
                    data = data.reshape(da.sizes["level"], da.sizes["latitude"], da.sizes["longitude"], tod)
                else:
                    da = da.transpose("latitude", "longitude", "tod")
                    data = da.values[np.newaxis, ...]  # add a channel axis: (1, H, W, T)
                channels_per_tod.append(data)

            x = np.concatenate(channels_per_tod, axis=0)  # (C, H, W, T)
            x = torch.tensor(x, dtype=torch.float32)
            assert x.ndim == self._input_dimensions - 1


        else:
            raise ValueError(f"Unsupported input_dimensions: {self._input_dimensions}")

        y = self.y[idx].clone()
        return x, y

    
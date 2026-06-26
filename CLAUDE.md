# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains ML models to predict biases in NOAA/SPC Convective Outlooks using ERA5 reanalysis data. Target variables are daily bias metrics (count bias and spatial displacement) for 3 hazard types (wind, hail, tornado), with an older 4-hazard version (including all-hazard) also present. Models are evaluated on their ability to predict these biases, with probabilistic output via the SHASH (sinh-arcsinh normal) distribution.

## Common Commands

**Training (on NCAR GLADE HPC cluster):**
```bash
qsub train.pbs        # runs train_crossval.py; ~24h for slgt_full
tail -f logs/cnn3d_cv.XXXXXXX.desched1.out
```

**Evaluation (on HPC):**
```bash
qsub evaluate.pbs     # runs evaluate_models.py; ~1h for slgt_full
```

**TensorBoard monitoring:**
```bash
tensorboard --logdir runs/
```

**Local notebook workflow:** All development and analysis is done via Jupyter notebooks in the repo root.

## Data Pipeline

The pipeline runs in this order:

1. **`download_inputs_glade.ipynb`** — Transfers ERA5 data from GLADE to `/glade/work/milesep/convective_outlook_ml/inputs_raw_{detail}_glade.zarr`. This is the canonical download script; all other `download_inputs_*.ipynb` variants are deprecated (incorrect 0Z-0Z time window or incomplete).

2. **`prepare_inputs.ipynb`** — Standardizes ERA5 data and splits into train (≤2019) / test (2020+) sets. Writes standardized zarr files and daily input stats to `data/processed_data/daily_input_stats_{detail}.nc`. Outputs go to `/glade/work/milesep/convective_outlook_ml/{train,test}_inputs_{detail}.zarr`.

3. **`prepare_targets.ipynb`** — Prepares target variables from `data/raw_data/contingency_regions.nc` and `data/raw_data/labelled_pph.nc`. Detrends training targets so the trend line is flat at the 2019-05-20 endpoint. Saves to `data/processed_data/train_targets{_slgt}{_new}.nc`.

4. **`train_models.ipynb`** (or `train_crossval.py` via PBS) — Runs 5-fold cross-validation training. Saves model checkpoints to `models/{model_spec}/fold={n}/{best,latest}.pt`, TensorBoard logs to `runs/`, and summary metrics to `results/results.csv`.

5. **`compare_models.ipynb`** or **`compare_models_shash.ipynb`** — Evaluates trained models. The `_shash` variant evaluates models trained with `ShashNLL` loss, saves per-sample SHASH parameters to `results/predictions{_losses}/{model_spec}/predictions.nc`, and aggregates NLL scores to `results/{best,latest}_all_shash_nll.csv`.

6. **`clustering/` analysis notebooks** (see below).

## Detail Levels

The `{detail}` string controls the spatial/temporal domain and variable set:

| Suffix | Meaning |
|---|---|
| `small_glade` | Small spatial domain, correct 12Z–12Z window |
| `full_glade` | Full domain, correct 12Z–12Z window |
| `slgt_small_glade` | Small domain, filtered to SLGT+ risk days only |
| `slgt_full_glade` | Full domain, SLGT+ days (~24h train time) |
| `_new` (target suffix) | Uses 3-hazard targets (wind/hail/tornado); omits all-hazard |

The `_glade` suffix marks scripts that access the correct 12Z–12Z valid-day window. All non-`_glade` variants have a 12-hour timing bug but were kept for comparison.

## Model Spec Naming Convention

Checkpoints and logs are organized under:
```
{model_name}/level={level}/opt={optimizer}_lr={lr}_batch={batch}_crit={criterion}/fold={fold}/
```

Example: `cnn3d_gelu_0_5/level=slgt_small_glade_new/opt=Adam_lr=0.001_batch=8_crit=ShashNLL/fold=0/`

## Architecture

### `src/` module

- **`models.py`** — `get_model(name, input_dim, output_dim)` factory for all model variants. `CNN3D` is the main architecture: configurable convolutional layers with BatchNorm, Dropout, MaxPool, then a global average pool and FC head. `ConstantPredictor` implements the `predict_mean`/`predict_zero` baselines. `get_model_input_dims(name)` returns 2 (flat) or 5 (volumetric) for `LazyWeatherDataset`.

- **`dataset.py`** — `LazyWeatherDataset` wraps an xarray Dataset of ERA5 inputs and a torch Tensor of targets. `input_dimensions` controls how data is shaped per sample: `2` = fully flattened (for linear baselines), `5` = `(C, H, W, T)` shape where C = variable×level, H/W = lat/lon, T = time-of-day (for CNN3D; the model permutes to `(batch, C, T, H, W)` in `forward()`).

- **`crossval.py`** — `run_crossval()` runs 5-fold KFold training. Per-fold: re-standardizes data to the fold's training split mean/std (using a conversion from the pre-computed overall stats), creates DataLoaders, trains with checkpoint resume logic, saves `best.pt` and `latest.pt`. After all folds: aggregates losses, logs averaged training curves to `runs/.../fold=avg/`.

- **`preprocessing.py`** — `flatten_target_dataset()` converts xarray target Dataset to a flat `(time, hazard*variable)` tensor. `standardize_with_stats()` and `compute_overall_from_daily_stats()` handle the pooled standardization (law of total variance).

- **`shash_torch.py`** — `Shash` class implementing the sinh-arcsinh normal distribution in PyTorch: `prob()`, `log_prob()`, `cdf()`, `mean()`, `median()`, `quantile()`, `std()`, `var()`. Uses a 6th-order polynomial approximation for the Jones-Pewsey P function (needed for `mean()`/`var()`) since `scipy.special.kv` is not differentiable.

- **`loss.py`** — `ShashNLL`: negative log-likelihood loss for SHASH. Expects `outputs` of shape `(B, 4*K)` where K = number of targets; internally applies `exp()` to the sigma and tau channels for positivity.

### Key data files

- `data/raw_data/contingency_regions.nc` — gridded contingency table values
- `data/raw_data/labelled_pph.nc` — SPC outlook probabilities with bias metrics
- `data/raw_data/clustering_fields.nc` — storm-centered ERA5 fields for clustering (97×97 grid, 0.25° resolution)
- `data/raw_data/heights.nc` — full CONUS 500 hPa geopotential heights for clustering
- `data/raw_data/grid_outlooks.nc`, `grid_reports.nc`, `binary_contingency_regions.nc`, `all_reports.csv` — used in cluster analysis
- `data/processed_data/train_targets{_slgt}{_new}.nc` — training targets; includes `train_mean` and `train_std` auxiliary variables
- `baseline_shash_params.pt` — saved SHASH parameters from the climatological baseline fit

## Clustering Pipeline

The clustering workflow is separate from model training and uses SOM (Self-Organizing Maps) on synoptic weather patterns:

1. **`download_clustering_inputs_glade.ipynb`** — Downloads storm-centered ERA5 fields to `data/raw_data/clustering_fields.nc`.
2. **`narrow_clustering_inputs.ipynb`** — Subsets clustering data as needed.
3. **`clustering.ipynb`** — Initial exploration of SOM clustering.
4. **`advanced_clustering.ipynb`** — Full production pipeline:
   - Computes derived fields: demeaned Z500 (`z500_dm`), 500 hPa vorticity (`vort500`), 850 hPa moisture flux (`qflux_u850/v850`)
   - Constructs day-of-year-smoothed, detrended 500 hPa height anomalies over CONUS
   - Combines storm-centered fields and CONUS heights into a combined feature matrix (with relative weighting)
   - Reduces to 30 PCs via PCA, then fits a `MiniSom` (default 3×3 = 9 clusters)
   - For each cluster: generates composite maps, seasonal roses, directional bias arrows, performance diagrams, and summary statistics
   - Outputs to `figs/advanced_clustering/`
5. **`cluster_analysis.ipynb`** and **`distribution_analysis.ipynb`** — Downstream analysis of model performance (NLL, LLR, KL divergence, SHASH parameters) broken down by cluster, using predictions from `results/predictions_losses/{model_spec}/predictions.nc`.

## Analysis Notebooks (Other)

- **`investigate_days.ipynb`** — Composite ERA5 fields stratified by target variable terciles; outputs to `figs/composites*/`.
- **`shash_baseline.ipynb`** — Fits constant SHASH distributions (no ERA5 input) via gradient descent as a climatological baseline; 5-fold CV to get comparable NLL.
- **`shash_performance_plot.ipynb`**, **`shash_daily_skills.ipynb`** — Visualization of SHASH model performance and per-day skill metrics.
- **`integrated_gradients.ipynb`**, **`visualize_integrated_gradients.ipynb`** — Saliency analysis using integrated gradients on the CNN models.
- **`infographic.ipynb`** — Summary figure generation for the paper.
- **`distribution_analysis.ipynb`** — Distribution of predicted SHASH parameters across clusters and dates.

## Important Implementation Notes

- **Cross-validation data standardization**: Each fold re-standardizes using that fold's training split mean/std. The data on disk is pre-standardized over the full training set, so `crossval.py` computes `conversion_stats` that effectively un-standardizes then re-standardizes to the fold split.
- **ShashNLL output size**: When using `ShashNLL` as criterion, `output_dim = num_targets * 4` (one set of 4 SHASH parameters per target). The model outputs mu, log_sigma, gamma, log_tau; `ShashNLL.forward()` applies `exp()` to enforce positivity for sigma and tau.
- **CNN3D input permutation**: `LazyWeatherDataset` with `input_dimensions=5` returns shape `(C, H, W, T)`; `CNN3D.forward()` permutes to `(batch, C, T, H, W)` before the first conv layer.
- **SLGT-only filtering**: The `slgt_*` detail levels filter training data to SLGT (Slight Risk) or higher outlook days. Target files for these use the `_slgt` suffix; the `_new` suffix additionally drops the all-hazard target column (going from 12 to 9 targets).
- **Checkpoint resume**: Training automatically resumes from the latest checkpoint if one exists. Pass `restart=True` to `run_crossval()` to start from scratch.

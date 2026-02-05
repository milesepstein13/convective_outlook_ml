This project trains an ML model to predict biases (from [https://github.com/milesepstein13/severe-thunderstorm-analysis](https://github.com/milesepstein13/severe-thunderstorm-analysis)) in SPC Convective Outlooks from ERA5 reanalysis.

Target variables: 12 distinct scalars: 4 hazard types (wind, hail, tornado, and all-hazard) times 3 bias types (count bias (forecast minus observed number of gripoints for which storms were observed within 25 mi; different from the quotient of those two values in previous work), and the two components of spatial displacement found with optical flow)

Input data: ERA5 reanalysis fields

1) `dowload_inputs_glade.ipynb` transfers a specified subset of ERA5 data from within the GLADE file system to the specified path (`/glade/work/milesep/convective_outlook_ml/inputs_raw_{detail}_glade.zarr`). Use `{detail}` to specify level of detail (spatial/temporal domain, and which variables).
   1) All other versions contain a bug where they access weather data from 0Z-0Z of the valid day (12 hours too early) rather than 12Z-12Z. Of note, model performance was similar with those versions.
   2) `{detail}` ends in "_glade" for data accessed with this script
   3) Geopotential at surface, tisr, and pv are only accessable in deprecated versions.
   4) This is somewhat faster than deprecated versions.
   5) Other deprecated versions of this script are as follows:
      1) `download_inputs.ipynb` downloads a specified subset of ERA5 data from [arco-era5 on Google Cloud](https://github.com/google-research/arco-era5).  Saves file as `/glade/work/milesep/convective_outlook_ml/inputs_raw_[detail].zarr`. `detail = 'full'` results in file size of 42 GB. `detail = 'small'` results in file size of <1GB--preferred for testing. Runtime: < 1 minute until saving, which takes a long time.
      2) `downlaod_inputs_glade_2.ipynb,` `downlaod_inputs_glade_new.ipynb`: incomplete/abandoned scripts
      3) `downlaod_inputs_cds.ipynb` downloads ERA5 data from the CDS API

2. `prepare_inputs.ipynb` opens raw ERA5 data from `/glade/work/milesep/convective_outlook_ml/inputs_raw_[detail].zarr` and splits into training (-2019) and test (2020-). Computes daily mean and std values for each variable/pressure level (across all of time and space) across training data and saves these means and stds to `data/processed_data/daily_input_stats_[detail].nc`. Then standardizes test and training sets by the overall training mean/std to `/glade/work/milesep/convective_outlook_ml/[train/test]_inputs_[detail].zarr`.
3. `prepare_targets.ipynb` opens contingency table values from `data/raw_data/contingency_regions.nc` and labelled PPH from `data/raw_data/labelled_pph.nc` (see [previous work](https://github.com/milesepstein13/severe-thunderstorm-analysis)) to prepare target variable dataset. Target variables are daily `(b-c)`, northward shift, and eastward shift for each hazard type. Each variable/hazard is split into training (-2019) and test (2020-) sets, and training sets are detrended such that the line of best fit is constant at the final value of the raw training line of best fit (2019-05-20)--enforcing that we cannot assume a change of SPC forecast practices into the future. Trends are plotted in `figs/`. Final dataset is saved as `data/processed_data/train_targets.nc`
4. `train_models.ipynb` The user specifies an arbitrary number of model names (model architectures specified in `src/models.py`), detail levels, learning rate, batch sizes, and number of epochs, and a model is trained for each respective set of specifications. Most code is in `src/train_loop.py`. Models are trained with 5-fold cross validation. Training runs each fold to completion of the specified number of epochs, picking up where training left off if necessary. Throughout training, the best performing model (lowest validation error at given epoch) and most recent model are saved in `models/`, training curves are saved in `runs/`, and training and validation losses and their standard deviations are saved in `results/results.csv`.
   1. `qsub train.pbs` calls `train_crossval.py`, which is a python file version of this script. This is strongly recommended for slgt_full, as runtime per model is ~24h. When running, check logs with tail -f logs/cnn3d_cv.XXXXXXX.desched1.out
5. `compare_models.ipynb` For all unevaluated models (that slgt or not, as specified by user):
   1. The best or latest version of the model (as specified by the user) is loaded and validated for all 5 folds. For all 12 target variables, the MSE, RMSE, and RMSE (un-standardized back to original units) are calculated and saved in `results/[best/latest]_all_[r]mses[_units].csv`. This stage goes up to "Run the following 3 cells with both slgt = True and slgt = False" and can be run in `evaluate_models.py` via `qsub evaluate.pbs`, which is recommended for slgt_full because of long runtime
   2. The remainder of the script always runs quickly and can be done after the ipynb or py approach to running the first part. It plots the rmse (with units) for each of the 12 target variables against baselines in `figs/error/`
6. `investigate_days.ipynb` is an independent script that plots composite ERA5 fields, either centered or nationwide, for either full or small resolution ERA5, divided by when each target variable is in negative, positive, or zero (technically, in the bottom, middle, or top third, assuming a normal distribution). Each composite is saved in `figs/[centered_]compsosites[_full]/[variable]/[bias_type]_[neg/pos/zero]_composite.png`.

This pipeline can be run with any of the following detail levels

| detail              | small | full | slgt_small | slgt_full |
| ------------------- | ----- | ---- | ---------- | --------- |
| _ (incorrect)       | y     | y    | y          | n         |
| _cds (incorrect)    |       |      |            | n         |
| _glade              | y     | y    | y          | y         |
| dataset size        | GB    | GB   | GB         | GB        |
| download time       |       |      |            | ~3 days   |
| train time          |       |      |            | ~24 hours |
| compare_models time |       |      |            | ~1 hour   |
| best MSE            | .91   | .94  | .87        | .87       |

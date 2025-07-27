This project trains an ML model to predict biases (from [https://github.com/milesepstein13/severe-thunderstorm-analysis](https://github.com/milesepstein13/severe-thunderstorm-analysis)) in SPC Convective Outlooks from ERA5 reanalysis.

1) download_inputs.ipynb downloads a specified subset of era5 data. Use detail to specify level of detail, saves file as inputs_raw_\[detail\]. detail = 'full' results in file size of 42 GB. detail = 'small' results in file size of GB--preferred for testing.

3) prepare_inputs

4) prepare_targets

5) model training


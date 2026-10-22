# CoFreq: Complementary Frequency Modeling for Long-Term Multivariate Forecasting

This repository provides the code implementation of **CoFreq**, a lightweight framework for long-term multivariate time series forecasting. The model is registered as `FreSL` in the code, so use `--model FreSL` when running experiments.

## Overview

CoFreq models temporal patterns and cross-channel structure through two complementary modules:

- **Frequency-aware TimeMoE**: Applies rFFT to the normalized input, divides the spectrum into overlapping frequency bands, and converts each band back into a band-limited time series. A temporal expert forecasts from each band. A learnable global top-$k$ gate combines the band experts, while a full-input global expert provides a residual correction.
- **Leader Sparse Mixing (LSM)**: Ranks channels by their post-normalization temporal standard deviation, selects the top-$K$ channels, and uses a complex-valued network to mix their spectra and map them to all channels.

The two branches are combined and refined to produce the final forecast. The independent band experts are evaluated with batched tensor operations, including `einsum`, to reduce Python-level per-expert execution overhead.

## Prepare the Dataset

The experiment scripts expect the following directory layout:

```text
./dataset/
├── ETT/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather.csv
├── traffic.csv
├── electricity.csv
└── solar.csv
```

Download the public datasets and place the CSV files at these paths. The ETT scripts read from `./dataset/ETT/`; the Weather, Traffic, Electricity, and Solar scripts read from `./dataset/`. If your files are stored elsewhere, update `root_path_name` and `data_path_name` in the corresponding script.

For custom datasets, use a CSV with a `date` column and numeric time-series columns. Set `--data custom`, specify the CSV with `--data_path`, and set `--enc_in` to the number of input variables. With `--features M`, all numeric columns are used as multivariate inputs and targets.

## Dependencies

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The requirements file specifies PyTorch 2.2.2, pandas, NumPy, matplotlib, and SciPy. A CUDA-enabled PyTorch installation and a compatible NVIDIA GPU are recommended for reproducing the GPU experiments.

## Run the Experiment

Run a dataset script from the repository root. For example:

```bash
bash scripts/ETTh1.sh
```

Other scripts are available in `scripts/`:

```bash
bash scripts/ETTh2.sh
bash scripts/ETTm1.sh
bash scripts/ETTm2.sh
bash scripts/weather.sh
bash scripts/traffic.sh
bash scripts/electricity.sh
bash scripts/solar.sh
```

The scripts train and evaluate the model at prediction horizons 96, 192, 336, and 720. To run a single configuration, invoke `run.py` directly and set the dataset, prediction horizon, channel count, and model parameters. 

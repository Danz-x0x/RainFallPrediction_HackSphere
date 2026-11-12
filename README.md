# Next-Day Rainfall Prediction for Indian Weather Stations

A machine learning pipeline designed to predict next-day rainfall (`rainfall_next_day_mm`) across Indian weather stations using solely same-day meteorological observations, spatial neighbor aggregations, and geostrophic upwind proxies.

---

## 📋 Overview

Because test data rows are evaluated independently without guaranteed access to historical lags or subsequent days, this pipeline strictly avoids temporal lag features. Instead, it extracts rich spatial and physical context from concurrent weather station readings on the same calendar date, combining multiple model families via Non-Negative Least Squares (NNLS) blending.

---

## 📁 Repository Structure

* **`features.py`**: Spatial, climatological, and meteorological feature engineering module.
* **`rainfall_pipeline.py`**: End-to-end training, expanding-window cross-validation, ensembling, and inference script.
* **`out/`** *(generated upon execution)*:
  * **`submission.csv`**: Final test predictions formatted to match `sample_submission.csv`.
  * **`predictions_with_data.csv`**: Full test dataset appended with predicted rainfall (`rainfall_next_day_mm_predicted`).
  * **`cache/`**: Cached out-of-fold and test predictions (`.npy`) to accelerate iterative runs.

---

## 🧠 Feature Engineering (`features.py`)

All features are computed strictly using same-day observations (`build_features`):

1. **Temporal & Cyclical Features**:
   * Month (`mon`), seasonal monsoon indicator (`is_monsoon` for June–September), and cyclical day-of-year encodings (`doy_sin`, `doy_cos`).
   * Daily temperature range (`trange = max_temp - min_temp`) and log-transformed precipitation (`log_rain_today`).
2. **Station-Month Climatologies & Anomalies**:
   * Historical station-specific monthly means (`_clim`) for air pressure, average temperature, rainfall, wind speed, and temperature range.
   * Daily anomalies (`p_anom`, `t_anom`, `w_anom`, `rain_anom`, `trange_anom`) measuring deviations from local monthly norms.
3. **Spatial Neighbor Aggregations**:
   * Computes pairwise Haversine distances across stations within each calendar date at **50 km, 100 km, and 250 km** radii.
   * Extracts neighbor station counts, mean/max rainfall, fraction of wet stations ($\ge 1\text{ mm}$), and pressure/temperature anomaly gradients.
   * Computes directional mean rainfall across 4 compass sectors (**N, E, S, W**) within 250 km.
4. **Geostrophic Upwind Proxy**:
   * Estimates local pressure-anomaly gradients ($g_x, g_y$) to infer Northern Hemisphere geostrophic wind direction (low pressure to the left).
   * Aggregates rainfall, pressure differences, and moisture advection (`up_adv`) within a $90^\circ$ upwind cone at 100 km and 250 km radii.

---

## ⚙️ Modeling & Validation (`rainfall_pipeline.py`)

### Validation Strategy
* **Expanding-Window Cross-Validation**: Evaluates across 4 yearly folds (`2020`, `2021`, `2022`, `2023`), training strictly on years `< Y` and validating on year `Y`.
* **Metrics**: Primary evaluation via **RMSE** (overall and heavy rainfall $>50\text{ mm}$), alongside **MAE**, **$R^2$**, and binary rain/heavy-rain classification accuracy.

### Ensemble Architecture
Four distinct model components are trained with customizable sample weighting (recency half-life decay and monsoon season multipliers):
1. **Histogram Gradient Boosting (`gbm`)**: 5-seed average predicting raw rainfall (`mm`).
2. **Neural Network (`nn`)**: 5-seed average of a `(64, 32)` `MLPRegressor` trained on quantile-normalized features.
3. **Log-Target GBM (`log`)**: Gradient boosted trees predicting $\log(1 + y)$ with residual variance smearing correction upon back-transformation.
4. **Heavy Rain Classifier (`p50`)**: `HistGradientBoostingClassifier` predicting the probability of extreme rainfall ($P(\text{rain} > 50\text{ mm})$).

### Blending
* Predictions are combined using **Non-Negative Least Squares (`nnls`)** with a free-sign intercept, calibrated on dense validation years (`2022` and `2023`) to mirror test set station density.

---

## 🚀 Getting Started

### Prerequisites
Ensure Python 3.8+ is installed along with the required dependencies:
```bash
pip install numpy pandas scipy scikit-learn
```

### Running the Pipeline
Execute the pipeline from the command line:
```bash
python rainfall_pipeline.py \
  --train train_cleaned.csv \
  --test test_cleaned.csv \
  --sample sample_submission.csv \
  --outdir out
```

### Command-Line Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--train` | `train_cleaned.csv` | Path to the labeled training dataset. |
| `--test` | `test_cleaned.csv` | Path to the unlabeled test dataset. |
| `--sample` | `sample_submission.csv` | Path to the sample submission template. |
| `--outdir` | `out` | Directory where outputs and prediction caches are saved. |
| `--half_life` | `None` *(or `best_weights.json`)* | Exponential recency sample weight half-life in years (`0` to disable). |
| `--monsoon_w` | `None` *(or `best_weights.json`)* | Sample weight multiplier applied to monsoon months (Jun–Sep). |
| `--seeds` | `5` | Number of random seeds averaged for the GBM and Neural Network models. |

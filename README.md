# Next-Day Rainfall Prediction for Indian Weather Stations

A machine learning pipeline designed to predict next-day rainfall (`rainfall_next_day_mm`) across Indian weather stations using solely same-day meteorological observations, spatial neighbor aggregations, and geostrophic upwind proxies.

---

## 📋 Overview

Because test data rows are evaluated independently without guaranteed access to historical lags or subsequent days, this pipeline strictly avoids temporal lag features. Instead, it extracts rich spatial and physical context from concurrent weather station readings on the same calendar date, combining multiple model families via Non-Negative Least Squares (NNLS) blending.

Before running the "rainfall_pipeline.py" file be sure to extract the "test and train datasets.zip" file and place the "train_cleaned.csv" and "test_cleaned" files in the same directory as the python file

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


# 🌧️ Rainfall Prediction – Model Training & Validation Results

## 1. Overview

This project uses machine learning models to predict rainfall (in mm) based on historical weather data from **2015–2023**.

Multiple models were trained and evaluated using an expanding-window validation strategy. Their predictions were combined using an out-of-fold weighted ensemble.

## 2. Dataset & Feature Engineering

| Parameter | Value |
|---|---:|
| Historical data | 2015–2023 |
| Labelled training rows | 546,703 |
| Test rows | 81,803 |
| Total features | 54 |
| Recency half-life | 0 years |
| Monsoon multiplier | 1.0 |

### Feature Engineering

- Neighbour-based features at 50 km, 100 km, and 250 km.
- Sector-based geographical features.
- Upwind proxy features.
- Spatial relationships between neighbouring locations.

## 3. Models Used

Three regression approaches were evaluated:

1. **Gradient Boosting Machine (GBM)** – 5-seed averaged predictions.
2. **Neural Network (NN)** – 5-seed averaged predictions.
3. **Log-Transformed GBM** – trained using `log1p(rainfall)`.

An additional classifier estimated the probability of heavy rainfall (>50 mm), evaluated using ROC-AUC.

## 4. Validation Strategy

An expanding-window validation strategy was implemented.

| Fold | Training Rows | Validation Rows |
|---|---:|---:|
| 2020 | 90,579 | 22,361 |
| 2021 | 112,940 | 137,123 |
| 2022 | 250,063 | 147,271 |
| 2023 | 397,334 | 149,369 |

## 5. Model Performance

### RMSE Comparison

*Lower RMSE indicates better performance.*

| Model | 2020 | 2021 | 2022 | 2023 | Average RMSE |
|---|---:|---:|---:|---:|---:|
| Mean-only Baseline | 23.142 | 14.158 | 12.222 | 11.452 | 15.243 |
| GBM (5 Seeds) | 19.092 | 11.623 | 9.123 | 8.679 | **12.129** |
| Neural Network (5 Seeds) | 19.165 | 12.983 | 9.156 | 8.766 | 12.517 |
| GBM (Log1p) | 19.812 | 12.107 | 9.203 | 8.905 | 12.507 |
| **Blended Model** | **18.991** | 11.919 | **9.108** | 8.716 | 12.183 |

### Overall Validation Metrics

| Metric | Result |
|---|---:|
| Overall RMSE | **10.585 mm** |
| Overall MAE | 5.151 mm |
| Overall R² | 0.314 |
| 2023 RMSE | 8.716 mm |
| 2023 R² | 0.389 |
| Rain / No-Rain Accuracy (≥1 mm) | 64.3% |
| Rain / No-Rain Accuracy (2023) | 79.3% |
| Heavy Rain Classification Accuracy (>50 mm) | 98.5% |
| RMSE on Heavy Rain Days | 63.24 mm |

### Key Observations

- GBM achieved the best average RMSE among individual models.
- The blended model performed competitively across all validation folds.
- Prediction performance improved in later validation years.
- The 2023 validation fold achieved an RMSE of 8.716 mm.

> **Note:** Rainfall prediction is a regression task. RMSE, MAE, and R² are the primary evaluation metrics. Classification accuracy applies only to the specified rainfall thresholds.

## 6. Ensemble Blending

An out-of-fold weighted blending strategy was used to combine model predictions.

### Learned Blend Weights

| Component | Weight |
|---|---:|
| GBM | 0.494 |
| Neural Network | 0.364 |
| Log-GBM | 0.164 |
| Heavy Rain Probability Contribution | 2.208 |
| Intercept | -0.299 |

The ensemble combines predictions from multiple models along with the heavy-rain probability estimate.

## 7. Final Training & Test Predictions

After validation, the final models were trained using the complete labelled dataset (2015–2023).

### Final Pipeline

1. Train the final GBM model.
2. Train the final Neural Network model.
3. Generate test predictions.
4. Apply ensemble blending.
5. Export predictions in the required submission format.

### Test Prediction Summary

| Metric | Result |
|---|---:|
| Total Test Rows | 81,803 |
| Mean Predicted Rainfall | 3.73 mm |
| Maximum Predicted Rainfall | 201.2 mm |
| Predictions ≥1 mm | 45.3% |

## 8. Output Files

| File | Description |
|---|---|
| `out/submission.csv` | Final predictions in sample submission format. |
| `predictions_with_data.csv` | Predictions combined with corresponding dataset information. |

## 9. Conclusion

The final ensemble achieved an overall validation RMSE of **10.585 mm** and an MAE of **5.151 mm**.

GBM delivered the strongest average performance among individual models, while ensemble blending combined multiple prediction approaches.

The final pipeline used **546,703 labelled observations** to generate rainfall predictions for **81,803 test observations**.

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

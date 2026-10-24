# India AQI: Next-Day Air Quality Category Prediction

A machine learning project that predicts a location's **next calendar day's air-quality category** using today's AQI and historical observations. The notebook combines data auditing, exploratory analysis, time-aware feature engineering, Random Forest tuning, and evaluation against a persistence baseline.

**Main result:** Random Forest achieved **74.24% accuracy** on **19,294 held-out next-day predictions**, compared with **72.98%** for persistence. However, persistence performed better on macro F1, the primary selection metric. This project documents that trade-off rather than claiming an overall improvement.

## Prediction task

- **Prediction time:** end of day, after the current day's AQI and station count are available.
- **Target:** the following calendar day's category: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.
- **Location identifier:** state + area.
- **Forecast mode:** daily rolling, one day ahead. Earlier observed test-day AQI can inform later forecasts; the entire test period is not forecast in advance.

This is a classification project, not numeric AQI regression.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, joblib, and Jupyter Notebook / VS Code.

## Dataset

The notebook reads a local file named `aqi.csv`. The following statistics come from its saved execution:

| Item | Value |
|---|---:|
| Raw records | 235,785 |
| Clean records | 235,784 |
| Locations (state + area) | 292 |
| States | 32 |
| Date range | 2022-04-01 to 2025-04-30 |
| Exact next-day observation pairs | 220,419 |
| Records without an observed next day after cleaning | 15,365 |

Required columns are `date`, `state`, `area`, `aqi_value`, `air_quality_status`, `number_of_monitoring_stations`, and `prominent_pollutants`. Dates are parsed as `DD-MM-YYYY`. Pollutant names are required by the input audit but are not used as model features. The supplied CSV also contains `unit` and `note` columns.

The notebook does not establish the dataset's original source or redistribution license. Verify attribution and permission before publishing the CSV.

## Workflow

### 1. Data validation and cleaning

- Remove exact duplicate records.
- Exclude invalid dates, missing location names, and AQI outside 0–500.
- Exclude duplicate location-date groups rather than arbitrarily choosing a row.
- Compare supplied categories with the AQI bands implemented in the notebook and remove inconsistent labels.
- Treat negative monitoring-station counts as missing.

The saved run removed one inconsistent label and found no exact duplicates, invalid rows, or conflicting location-date records. Missing model inputs are filled with training-set medians through a pipeline.

### 2. Exploratory analysis

The notebook displays AQI distribution, category counts, daily and monthly trends, and locations with the highest mean AQI among those with at least 180 observations.

Daily averages are unweighted means across reporting locations, not population-weighted national exposure estimates. Location comparisons may cover different dates and reporting coverage.

### 3. Feature engineering

| Feature | Meaning |
|---|---|
| `aqi_value` | Today's AQI |
| `number_of_monitoring_stations` | Today's station count |
| `aqi_lag1`, `aqi_lag3`, `aqi_lag7` | AQI exactly 1, 3, and 7 calendar days earlier |
| `aqi_mean7`, `aqi_mean14` | Mean AQI over the preceding 7 and 14 calendar days, excluding today |
| `aqi_std7` | AQI standard deviation over the preceding 7 calendar days, excluding today |
| `month_sin`, `month_cos` | Cyclical month features for the forecast date |
| `weekday` | Weekday of the forecast date |

Rolling features require at least two historical observations. Missing dates remain missing; no future observations are used to fill gaps. The target is joined at exactly the next calendar day. State and area are used for grouping and joins, not as model inputs.

### 4. Chronological splitting

Every location with the same target date belongs to the same partition.

| Partition | Target dates | Pairs |
|---|---|---:|
| Training | 2022-04-02 to 2024-12-01 | 188,011 |
| Validation | 2024-12-02 to 2025-01-30 | 13,114 |
| Test | 2025-01-31 to 2025-04-30 | 19,294 |

The last 90 calendar days form the test period; the preceding 60 calendar days form validation. After selecting parameters, the forest is refitted on training + validation data before test evaluation.

### 5. Model tuning

Three manually specified Random Forest configurations are evaluated using validation macro F1:

| Trees | Maximum depth | Minimum leaf size | Validation macro F1 |
|---:|---:|---:|---:|
| 60 | 12 | 10 | 0.5142 |
| 80 | 16 | 5 | **0.5347** |
| 80 | 20 | 10 | 0.5111 |

The best forest uses `n_estimators=80`, `max_depth=16`, and `min_samples_leaf=5`. Shared settings include `max_features='sqrt'`, `bootstrap=True`, `max_samples=0.7`, `n_jobs=2`, and `random_state=42`.

Each tree draws a bootstrap sample sized at 70% of the fitting data. This is **not a 70/30 train-test split**; all eligible pairs remain in the chronological partitions.

This is **chronological holdout tuning**, not cross-validation, GridSearchCV, or RandomizedSearchCV.

## Results

### Baseline comparison

Persistence predicts that tomorrow's category will equal today's category. Its validation macro F1 was **0.5844**, exceeding the best forest's **0.5347**. Therefore, the validation-selected approach was **Persistence**.

| Test metric | Persistence | Random Forest |
|---|---:|---:|
| Accuracy | 72.98% | **74.24%** |
| Macro F1 | **0.4887** | 0.4843 |
| Weighted F1 | 0.7300 | **0.7373** |
| Balanced accuracy | **49.21%** | 45.01% |

The forest improved accuracy by approximately **1.26 percentage points**, but its macro F1 was approximately **0.45 percentage points lower** using the unrounded notebook scores. Higher accuracy does not establish better performance across all six categories.

### Random Forest performance by category

| Category | Precision | Recall | F1 | Test support |
|---|---:|---:|---:|---:|
| Good | 0.7782 | 0.6133 | 0.6860 | 1,854 |
| Satisfactory | 0.7483 | 0.7541 | 0.7512 | 8,121 |
| Moderate | 0.7440 | 0.8117 | 0.7764 | 8,289 |
| Poor | 0.5523 | 0.3340 | 0.4163 | 964 |
| Very Poor | 0.5217 | 0.1875 | 0.2759 | 64 |
| Severe | 0.0000 | 0.0000 | 0.0000 | 2 |

Performance is strongest for common categories. Recall is limited for Poor and Very Poor, and the model detected neither of the two Severe examples. Two examples are insufficient to establish reliable Severe-category performance. Macro F1 gives each category equal weight, while weighted F1 is dominated by more common categories.

## Run locally

1. Place `AQI_Next_Day_Prediction(3).ipynb` and `aqi.csv` in the same project folder.
2. Open that folder in VS Code and select a Python notebook kernel, or use Jupyter.
3. Install the dependencies in that kernel with this notebook command:

   ```python
   %pip install pandas numpy matplotlib scikit-learn joblib threadpoolctl ipykernel
   ```

4. Restart the kernel if packages were installed, then choose **Run All**.

The current loader uses `Path.cwd() / 'aqi.csv'`. Ensure the kernel's working directory is the project folder. Although the notebook introduction mentions a `data/` subfolder, the current code does not search it automatically. The existing `DATA_PATH is None` condition also does not check file existence; a missing file will fail at `pd.read_csv`.

For Google Colab, upload both the notebook and `aqi.csv` and run the cells in order. No API key is required. Training time depends on hardware and package versions. Results above are recorded notebook outputs, not a guarantee that every environment will reproduce identical values.

## Forecast example

After running the training cells:

```python
forecast_next_day('Andhra Pradesh', 'Amaravati')
```

The function uses the latest available observation for that location unless `as_of` is supplied. It requires an observation on the chosen date and rejects dates before the fitted model's training-target cutoff.

The saved example forecasts **2025-05-01**, returning Satisfactory for both the forest and persistence. This is a historical demonstration using the end of the supplied file, not a live forecast. Its outcome is not present in the dataset.

The function returns the forest category, an **uncalibrated model score**, and the persistence category. That score is not a calibrated probability of being correct. The function displays both methods rather than automatically applying the validation-selected approach.

## Generated files

Running the final cell creates these files in `outputs/`:

| File | Contents |
|---|---|
| `test_metrics.csv` | Aggregate test metrics for both approaches |
| `validation_search.csv` | Three configurations, validation scores, and fitting times |
| `test_predictions.csv` | Location/date identifiers, actual categories, predictions, and category-distance errors |
| `aqi_model.joblib` | Fitted Random Forest pipeline, feature names, category names, and training-target cutoff |
| `run_config.json` | Seed, selected parameters, chosen approach, record counts, split dates, library versions, and dataset SHA-256 |
| `resume_bullet.txt` | A summary reporting measured results and the baseline comparison |

The saved estimator is the **Random Forest**, even though validation selects Persistence. Its inputs must be engineered using the notebook's feature-building logic; it does not accept raw CSV rows directly. Only load trusted joblib files. Plots are displayed in the notebook and are not separately exported by the current code.

## Limitations and future improvements

- The test window covers only 90 calendar days; year-round reliability has not been established.
- Rare high-pollution categories have limited support and weak recall.
- The model lacks weather inputs and explicit location features.
- Sudden category changes can produce large forecasting errors.
- Saved outputs contain date-operation deprecation warnings; these did not prevent completion but warrant compatibility checks.
- There is no live data integration, deployed API, monitoring system, or measured financial or public-health impact.

Possible next steps include chronological backtesting across seasons, evaluating class weighting and additional features on development data, and making inference explicitly honor the validation-selected approach. Future improvements should be evaluated on a fresh held-out period rather than repeatedly tuning against the reported test set.

## Project takeaway

The project demonstrates reproducible data preparation, calendar-aware feature engineering, chronological model selection, and baseline-aware evaluation. The measured outcome shows why a higher accuracy score alone is insufficient to justify replacing a simple forecasting baseline.

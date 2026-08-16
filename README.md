# Robust Regression Engine

A notebook-centered housing-price regression project using `houseprice_dataset.csv`. The notebook loads and inspects property data, prepares model inputs, validates Ridge Regression, compares five regression models, and analyzes Random Forest feature importance.

## Workflow

The notebook follows this sequence:

1. Load and inspect `houseprice_dataset.csv`.
2. Prepare training and testing inputs.
3. Validate Ridge Regression using five-fold cross-validation and parameter search.
4. Evaluate five regression models on a held-out test set:

   * Ridge Regression
   * Lasso Regression
   * Decision Tree
   * Random Forest
   * Support Vector Regression
5. Compare model performance.
6. Inspect Random Forest feature importance.

## Dataset

The notebook expects the following file in the same directory:

```text
houseprice_dataset.csv
```

### Dataset Fields

| Field              | Role                                                        |
| ------------------ | ----------------------------------------------------------- |
| `property_id`      | Supplied identifier; removed before model input preparation |
| `sale_date`        | Converted into `sale_year` and `sale_month`                 |
| `area_sqft`        | Model feature                                               |
| `bedrooms`         | Model feature                                               |
| `bathrooms`        | Model feature                                               |
| `location_score`   | Model feature                                               |
| `property_age`     | Model feature                                               |
| `distance_city_km` | Model feature                                               |
| `near_school`      | Model feature                                               |
| `near_metro`       | Model feature                                               |
| `crime_rate_index` | Model feature                                               |
| `house_price_inr`  | Regression target                                           |

The recorded dataset inspection contains:

* **3,800 rows**
* **12 columns**
* **No null values reported**

## Data Preparation

The effective input preparation process is:

1. Set `house_price_inr` as the target variable.
2. Remove `property_id`.
3. Convert `sale_date` into:

   * `sale_year`
   * `sale_month`
4. Remove the original `sale_date` column.
5. Split the dataset into training and testing sets.
6. Scale numerical features using parameters fitted only on the training data.
7. Apply the same scaler to the test data.

### Train/Test Split

```text
test_size = 0.20
random_state = 42
```

This produces:

```text
Training rows: 3,040
Testing rows:    760
```

### Scaled Features

```text
area_sqft
bedrooms
bathrooms
location_score
property_age
distance_city_km
crime_rate_index
sale_year
sale_month
```

`near_school` and `near_metro` are not included in the named scaling list.

## Running the Notebook

Make sure `houseprice_dataset.csv` is available using the exact relative filename expected by the notebook.

The initial inspection uses:

```python
df = pd.read_csv("houseprice_dataset.csv")

print(df.columns)
print(df.info())
print(df.head())
```

Run the notebook cells **in order**, since later cells depend on variables and models created by earlier cells.

### Workflow

```text
Dataset Loading
      ↓
Data Inspection
      ↓
Feature Preparation
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Ridge Validation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Random Forest Feature Importance
```

## Ridge Regression Validation

Ridge Regression is validated using the training data with five-fold cross-validation and parameter search.

The captured validation result selected:

```text
alpha = 1
```

The fixed-alpha and parameter-search validation paths are treated as separate validation flows in the notebook.

## Model Evaluation

The models are evaluated on the held-out test set using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination

### Results

| Model                     |              MAE |             RMSE |         R² |
| ------------------------- | ---------------: | ---------------: | ---------: |
| Ridge Regression          |     1,945,385.07 |     2,539,951.81 |     0.9199 |
| Lasso Regression          |     1,945,519.92 |     2,539,709.14 |     0.9199 |
| Decision Tree             |     2,402,688.05 |     3,303,192.45 |     0.8645 |
| **Random Forest**         | **1,753,015.22** | **2,400,226.59** | **0.9285** |
| Support Vector Regression |     6,994,027.95 |     8,987,330.83 |    -0.0029 |

## Best Performing Model

Based on the recorded test-set results, **Random Forest** is the best-performing model among the five evaluated models.

### Random Forest Performance

```text
MAE  = 1,753,015.22
RMSE = 2,400,226.59
R²   = 0.9285
```

Random Forest achieves:

* The **lowest MAE**
* The **lowest RMSE**
* The **highest R²**

Therefore, Random Forest is the captured selection for the housing-price regression task.

## Project Structure

```text
Robust_Regression_Engine/
│
├── Robust_Regression_Engine.ipynb
├── houseprice_dataset.csv
└── README.md
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* Regression & Machine Learning
* Feature Scaling
* Cross-Validation
* Hyperparameter Search
* Feature Importance Analysis

## Models Used

### Ridge Regression

Linear regression with L2 regularization. Used for validation and model comparison.

### Lasso Regression

Linear regression with L1 regularization.

### Decision Tree Regression

A tree-based regression model capable of capturing nonlinear relationships.

### Random Forest Regression

An ensemble of decision trees used to improve generalization and prediction performance.

### Support Vector Regression

SVR-based regression evaluated as part of the model comparison.

## Key Takeaways

* The dataset contains **3,800 property records**.
* `house_price_inr` is the prediction target.
* `sale_date` is transformed into year and month features.
* The data uses an **80/20 train-test split**.
* Ridge validation selected `alpha = 1`.
* Random Forest achieved the best recorded test performance.
* Support Vector Regression performed poorly on the captured evaluation, producing a negative R².
* Random Forest is the preferred model based on the recorded comparison.

## Author

**Neel Patel**

GitHub: [patelneel9080](https://github.com/patelneel9080)

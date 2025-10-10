# Airline Price Predictor

The primary objective is to accurately predict the total cost of an airline ticket (`totalFare`) based on various flight characteristics, booking details, and route information.

## About the Dataset

The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices), which was originally referenced in the research paper titled *“Fare Game: A Mean Field Model of Stochastic Intensity Control in Dynamic Ticket Pricing”* by **Burak Aydın**, **Emre Parmaksız**, and **Ronnie Sircar** ([Paper Link](https://sircar.princeton.edu/sites/g/files/toruqf5981/files/documents/ticket_MFG.pdf)).

The original dataset contained over **one million records**, making it computationally challenging to process on a personal computer. Therefore, the dataset was **truncated to 600,000 entries** to maintain a balance between computational efficiency and data representativeness, ensuring that no significant information was lost.

## Data Preparation and Feature Engineering

The initial dataset contained a mix of numerical, categorical, and time-related features. The following key data preparation and feature engineering steps were performed:

1.  **Duration Conversion:** The `travelDuration` column (e.g., `PT2H29M`) was converted into a new numerical feature, **`duration_minutes`**.
2.  **Number of Stops:** A new feature, **`Num_of_stops`**, was created by counting the number of flight segments and subtracting one.
3.  **Route Encoding:** The categorical airport codes (`startingAirport` and `destinationAirport`) were converted into numerical features using **One-Hot Encoding**.
4.  **Binary Feature Conversion:** Boolean columns (`isBasicEconomy`, `isRefundable`, `isNonStop`) were converted to integers (`0` or `1`).
5.  **Missing Value Imputation:** Missing values in the **`totalTravelDistance`** column were imputed using a computed ratio of average travel distance to total fare from non-missing samples ($\frac{\text{totalTravelDistance}}{\text{totalFare}}$).
6.  **Feature Selection:** The `baseFare` column, which is highly correlated with the target `totalFare` and would make the prediction trivial, was intentionally dropped from the feature set. The `totalFare` column itself was also dropped to form the feature matrix $X$.
7.  **Target Transformation:** Both the feature matrix ($X$) and the target variable ($y$) were log-transformed using `np.log1p` to stabilize variance and make the distribution more Gaussian-like, which often improves regression model performance.

## Model Training and Hyperparameter Tuning

A **Random Forest Regressor** was chosen for its robustness and performance on structured data.

### Hyperparameter Tuning

The optimal hyperparameters for the Random Forest model were determined using the **Optuna** library for automated hyperparameter optimization. The objective was to **minimize the Mean Absolute Error (MAE)** (based on the first experiment's `objective` function), and the tuning process explored:

* **`n_estimators`**: The number of trees in the forest (suggested range: 100 to 500).
* **`max_depth`**: The maximum depth of the tree (suggested range: 10 to 50).

The best hyperparameters found by Optuna were:
* `n_estimators`: **115**
* `max_depth`: **46**
* *Note: Although the Optuna study was prematurely interrupted (as shown in the notebook output), the best parameters from the completed trials were used for the final model.*

## Model Evaluation Metrics

This document provides a brief explanation of key regression evaluation metrics used to assess model performance.


### 📊 R² Score (Coefficient of Determination)
- Measures how well the model explains the variance in the target variable.  
- **Interpretation:**  
    - **Higher R² → Better model fit**  
     - **R² = 1:** Perfect prediction  
     - **R² = 0:** Model performs no better than the mean


### 📈 MAE (Mean Absolute Error)
- Represents the average of the absolute differences between the predicted and actual values.  
- Indicates how far, on average, the predictions are from the true values (in the same units as the target).
- **Interpretation:**  
    - Measures average prediction error in the same units as the target.  
    - **Lower MAE → More accurate predictions**  



### 📉 RMSE (Root Mean Squared Error)
- Measures the square root of the average of squared differences between predicted and actual values.  
- Penalizes larger errors more than MAE, making it sensitive to outliers.
- **Interpretation:**  
    -  Penalizes larger errors more than MAE.  
    - **Lower RMSE → Better model performance**

### Model Performance

The final model was trained using the best hyperparameters on a log-transformed training set. Predictions were inverse-transformed (`np.expm1`) to evaluate the performance against the original scale test set.

| Metric | Experiment|
| :--- | :--- |
| **$R^2$ Score** | 0.7433790853735714 |
| **MAE** | 65.596297276914 |
| **RMSE** | 121.61509644787276 |

The model demonstrates **strong and reliable performance**.  
An **R² score of 0.74** indicates that it successfully explains about **74% of the variance** in the target variable, reflecting a solid fit to the data.  
The **MAE of ~65.6** shows a **relatively small average prediction error**, while the **RMSE of ~121.6** - lightly higher due to its sensitivity to large deviations - suggests that the model performs well overall, with only a few larger errors.  

Overall, these metrics highlight a **well-trained and effective model** that captures the underlying patterns accurately, with room for minor improvements to further enhance performance.
## Feature Importance

A breakdown of the most important feature groups in predicting the `totalFare` shows:

| Feature Group | Importance |
| :--- | :--- |
| `totalTravelDistance` | 0.430247 |
| `duration_minutes` | 0.147112 |
| `seatsRemaining` | 0.099897 |
| `isBasicEconomy` | 0.098417 |
| `DestinationAirport` (One-Hot Encoded) | 0.074632 |
| `StartingAirport` (One-Hot Encoded) | 0.071344 |
| `Num_of_stops` | 0.071158 |
| `elapsedDays` | 0.006902 |
| `isNonStop` | 0.000287 |
| `isRefundable` | 0.000004 |

**Conclusion on Feature Importance:** The **distance** of the flight is the single most important predictor, followed by **flight duration** and the **number of seats remaining**. The categorical airport and stop-related features also contribute significantly.

## Saved Model

The final trained model, along with the one-hot encoder object and the list of columns used for training, has been saved into a pickle file named **`airline_model.pkl`** for easy deployment and future use.  

When running the Jupyter Notebook, this file will be **automatically generated and saved** after the model training process completes.  
This allows the model to be **quickly loaded later** for making predictions without the need to retrain.

---

### Prerequisites

* Python3
* pandas
* numpy
* scikit-learn
* Optuna
* matplotlib
* seaborn

### Usage

To load and use the trained model:

```python
import pickle
import numpy as np
import pandas as pd
from utils import parse_duration_to_minutes, categorize_stops

# Load the saved objects
with open("airline_model.pkl", "rb") as f:
    to_load = pickle.load(f)

model = to_load['model']
encoder = to_load['encoder']
training_columns = to_load['columns']

# Example: Prepare new data for prediction (assuming 'new_data' is a DataFrame)
# 1. Apply feature engineering functions (`parse_duration_to_minutes`, `categorize_stops`)
# 2. Apply One-Hot Encoding using the loaded 'encoder'
# 3. Align columns to match 'training_columns'
# 4. Apply log transformation using `np.log1p()`
# 5. Predict using `model.predict()`
# 6. Inverse-transform the prediction using `np.expm1()`
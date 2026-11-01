# ✈️ Flight Price Prediction (PRCP-1025)

A machine learning project that analyses what drives airfares and predicts flight ticket prices. Customers can use it to plan journeys, and airlines can see which prices are sustainable.

The final model, a **tuned XGBoost regressor**, explains about **84% of the variation in price** (R² = 0.8419) with a typical error of roughly 1,100–1,200 per ticket.

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `flight_price_prediction_final.ipynb` | Main notebook: analysis, modelling, comparison and reports |
| `Flight_Fare.xlsx` | Dataset (not included here; add it before running, see [Setup](#-setup)) |

---

## 📊 Dataset

The dataset has **10,683 flight records** and 11 columns.

| Column | Description |
|---|---|
| `Airline` | Airline name |
| `Date_of_Journey` | Date of travel (`dd/mm/yyyy`) |
| `Source` / `Destination` | Departure and arrival cities |
| `Route` | Route taken |
| `Dep_Time` / `Arrival_Time` | Departure and arrival times |
| `Duration` | Flight duration as text (e.g. `2h 50m`) |
| `Total_Stops` | Number of stops (e.g. `non-stop`, `1 stop`) |
| `Additional_Info` | Extra information (mostly `No info`) |
| `Price` | **Target variable**: ticket price |

---

## 🔍 Project Workflow

The notebook is organised into four sections that match the project deliverables.

### 1. Data Analysis Report
- **Data quality:** removed 1 row with missing values and 220 duplicate rows, leaving **10,462 rows**.
- **Feature engineering:**
  - `Date_of_Journey` → `Journey_day`, `Journey_month`
  - `Dep_Time` / `Arrival_Time` → hour and minute columns
  - `Duration` → `Duration_mins`
  - `Total_Stops` → ordered numeric values (0–4)
- **EDA:**
  - Price by airline, source, stops, month and duration
  - Price distribution and skewness
  - Correlation matrix
  - Outlier analysis
- **Key insights:**
  - `Price` is right-skewed (skewness ≈ 1.86).
  - Median price rises with the number of stops, from about 4,667 for non-stop to about 17,686 for 4 stops.
  - Jet Airways, IndiGo and Air India operate most flights.
  - Premium fares (e.g. Jet Airways Business) are real, so the high-price outliers were **kept**.
  - `Route` and `Additional_Info` were dropped as redundant or low-signal.

### 2. Predictive Model
- One-hot encoding of `Airline`, `Source` and `Destination` (`drop_first=True`).
- 80/20 train/test split (`random_state=42`).
- Models trained: Linear Regression, Decision Tree, Random Forest, XGBoost and SVR (with `StandardScaler`).
- Hyperparameter tuning with `GridSearchCV` (3-fold CV) for Random Forest and XGBoost.
- Feature importance and actual-vs-predicted plots for the best model.

### 3. Model Comparison Report

All models were evaluated on the same hold-out test set:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.5838 | 1998.64 | 2945.80 |
| Decision Tree | 0.7146 | 1383.09 | 2439.33 |
| SVR | 0.7482 | 1363.85 | 2291.12 |
| Random Forest | 0.8134 | 1172.66 | 1972.58 |
| XGBoost | 0.8357 | 1132.33 | 1850.80 |
| Tuned Random Forest | 0.8390 | **1117.24** | 1831.99 |
| **Tuned XGBoost** | **0.8419** | 1183.94 | **1815.44** |

**Best hyperparameters**
- XGBoost: `learning_rate=0.05`, `max_depth=7`, `n_estimators=200`
- Random Forest: `max_depth=15`, `min_samples_split=5`, `n_estimators=200`

**Recommendation:** use **Tuned XGBoost** for production, as it has the highest R² and lowest RMSE. Tuned Random Forest is a very close alternative with a slightly lower MAE.

### 4. Report on Challenges Faced

The notebook documents 13 data and modelling challenges, each with the technique used and the reason. Examples:
- Dates and times stored as text
- Free-text durations
- Duplicate rows
- Skewed prices with outliers
- Feature scaling for SVR (fitted on train only, to avoid leakage)
- Overfitting in the single Decision Tree

---

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- XGBoost
- openpyxl (to read the `.xlsx` file)

---

## 🚀 Setup

1. **Clone the repository** and move into the folder.

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost openpyxl jupyter
   ```

3. **Add the dataset.** The notebook reads the data from `/content/Flight_Fare.xlsx` (the Google Colab path). If you run it locally, change this line in the notebook to point to your file:
   ```python
   df = pd.read_excel('Flight_Fare.xlsx')
   ```

4. **Launch the notebook:**
   ```bash
   jupyter notebook flight_price_prediction_final.ipynb
   ```
   Then run all cells in order. The notebook also runs directly in Google Colab.

> ⏱️ The `GridSearchCV` cells fit 24 models each, so they can take a few minutes.

---

## 🔮 Future Improvements

- Add the **booking date** (days before departure), which is likely to improve accuracy most.
- Add **holiday and festival** information to capture seasonal demand.
- Retrain regularly and monitor prediction error in production.

---

## 📌 Conclusion

Airline, number of stops, duration, route and travel date are the main drivers of flight prices. Tree-based ensembles clearly outperform linear models, and the tuned XGBoost model (R² ≈ 0.84) is recommended for production.

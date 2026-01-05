# Predicting_Housing_Price_California_Distirct_using_Regression_Models

# California Housing Price Prediction

## 📌 Overview
This project predicts median house values in California using the **California Housing dataset**.  
It demonstrates end-to-end ML workflow: preprocessing, feature engineering, scaling, model training, and evaluation.

---

## ⚙️ Workflow
1. **Data Loading & Exploration**
   - Loaded `housing.csv` dataset
   - Checked null values (`total_bedrooms` had 207 missing entries)
   - Explored categorical feature `ocean_proximity`

2. **Preprocessing**
   - Randomized dataset to avoid ordering bias
   - One-hot encoded `ocean_proximity`
   - Dropped nulls (final dataset size: 20,433 rows)
   - Feature scaling applied to 8 numerical columns using `StandardScaler`

3. **Feature Selection**
   - Selected 13 predictors + target (`median_house_value`)
   - Combined scaled numerical features with categorical one-hot features

4. **Data Splitting**
   - Train: 18,000 samples
   - Validation: 1,217 samples
   - Test: 1,218 samples

5. **Models Implemented**
   - Linear Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest
   - Gradient Boosting

---

## 📊 Results (RMSE)
| Model              | Train RMSE | Validation RMSE | Test RMSE |
|--------------------|------------|-----------------|-----------|
| Linear Regression  | 68,593     | 66,530          | 71,382    |
| KNN (k=10)         | 53,759     | 58,193          | 62,161    |
| Random Forest (d=10)| 43,559    | 52,143          | 53,390    |
| Gradient Boosting (n=250)| 47,274 | 49,263         | 51,411    |

---

## 🚀 Key Insights
- **Random Forest** and **Gradient Boosting** outperform Linear Regression and KNN.
- Gradient Boosting achieved the **best generalization** (lowest validation/test RMSE).
- Feature scaling improved model stability for distance-based algorithms (KNN).
- Categorical encoding of `ocean_proximity` added meaningful predictive power.

---

## 📂 Tech Stack
- **Python** (NumPy, Pandas, Scikit-learn)
- **Models**: Linear Regression, KNN, Random Forest, Gradient Boosting
- **Evaluation**: RMSE, Train/Validation/Test split

---

## 📌 Next Steps
- Hyperparameter tuning (GridSearchCV for Random Forest/GBR)
- Feature engineering (rooms per household, bedrooms per room, population per household)
- Cross-validation for robust performance estimates

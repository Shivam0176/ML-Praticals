# ============================================================
# LAB 10: ENSEMBLE LEARNING - RANDOM FOREST vs LINEAR REGRESSION
# Dataset: California Housing (bundled with sklearn, no internet)
#
# Big picture: one "smart guesser" can be beaten by the
# AVERAGE of many slightly different guessers.
# Random Forest = 100 decision trees, each trained on a
# random sample of the data; the final answer is their mean.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)


# ------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# STEP 2: SEPARATE FEATURES/TARGET AND SPLIT
# ------------------------------------------------------------

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)


# ------------------------------------------------------------
# STEP 3: TRAIN MODEL 1 - LINEAR REGRESSION (ONE STRAIGHT LINE)
# ------------------------------------------------------------

# Simple, fast, easy to interpret. Weakness: real house
# prices are NOT a straight-line function of the features.

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)


# ------------------------------------------------------------
# STEP 4: TRAIN MODEL 2 - RANDOM FOREST (100 TREES)
# ------------------------------------------------------------

# Each tree:
#   - sees a random BOOTSTRAP sample of rows (with replacement)
#   - at every split considers only a RANDOM SUBSET of features
# Alone, each tree overfits badly. Averaged together, the
# random errors cancel out and the forest becomes accurate.
# (This averaging idea is called BAGGING.)

rf_model = RandomForestRegressor(
    n_estimators=100,      # number of trees
    max_depth=15,          # don't let trees grow insanely deep
    random_state=42,
    n_jobs=-1              # use all CPU cores (faster)
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("\nBoth models trained!")


# ------------------------------------------------------------
# STEP 5: EVALUATE BOTH WITH THE SAME FOUR METRICS
# ------------------------------------------------------------

# MSE  - mean squared error (punishes big misses)
# RMSE - sqrt of MSE, same units as price
# MAE  - mean absolute error (robust to outliers)
# R2   - share of price variance explained (1 = perfect)

def evaluate(name, truth, prediction):
    mse = mean_squared_error(truth, prediction)
    mae = mean_absolute_error(truth, prediction)
    r2 = r2_score(truth, prediction)
    print(f"\n{name}")
    print("  MSE :", round(mse, 4))
    print("  RMSE:", round(np.sqrt(mse), 4))
    print("  MAE :", round(mae, 4))
    print("  R2  :", round(r2, 4))
    return [name, mse, np.sqrt(mse), mae, r2]

linear_row = evaluate("Linear Regression", y_test, linear_predictions)
rf_row = evaluate("Random Forest", y_test, rf_predictions)

results = pd.DataFrame(
    [linear_row, rf_row],
    columns=["Model", "MSE", "RMSE", "MAE", "R2 Score"])

print("\n=== SIDE-BY-SIDE COMPARISON ===")
print(results.round(4))


# ------------------------------------------------------------
# STEP 6: VISUALIZE WHO WON
# ------------------------------------------------------------

# Lower RMSE/MAE and higher R2 = better model.
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["RMSE"], color="steelblue")
plt.title("RMSE Comparison (lower is better)")
plt.ylabel("RMSE")
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["R2 Score"], color="seagreen")
plt.title("R2 Score Comparison (higher is better)")
plt.ylabel("R2 Score")
plt.show()


# ------------------------------------------------------------
# STEP 7: SCATTER PLOT OF RANDOM FOREST PREDICTIONS
# ------------------------------------------------------------

# Points hugging the diagonal = accurate predictions,
# exactly the improving we hoped for versus the straight line.

plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_predictions, alpha=0.3)
plt.xlabel("Actual house value")
plt.ylabel("Predicted house value")
plt.title("Random Forest: Actual vs Predicted")
plt.show()


# ------------------------------------------------------------
# STEP 8: WHAT DOES THE FOREST LOOK AT? (FEATURE IMPORTANCE)
# ------------------------------------------------------------

# Every tree split, choosing a feature, lowered the error.
# Adding up these reductions per feature across all trees
# tells us which features actually drive predictions.
# Expect median income (MedInc) to dominate.

importance_table = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_table = importance_table.sort_values(
    by="Importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance_table.round(4))

plt.figure(figsize=(9, 5))
sns.barplot(data=importance_table, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.show()


# ------------------------------------------------------------
# STEP 9: FINAL VERDICT
# ------------------------------------------------------------

if rf_row[4] > linear_row[4]:
    winner = "Random Forest"
else:
    winner = "Linear Regression"

print("\n=== FINAL RESULT ===")
print("Winner:", winner)
print("Linear Regression R2:", round(linear_row[4], 4))
print("Random Forest R2:    ", round(rf_row[4], 4))

print("\nLesson: averaging many randomized models (bagging)")
print("beats a single rigid model when the data is nonlinear -")
print("at the price of losing interpretability.")

# ============================================================
# LAB 10: ENSEMBLE LEARNING
# Random Forest vs Linear Regression
# Dataset: California Housing
#
# Goal: show that combining MANY decision trees (a Random
# Forest) beats a single linear model on data where the
# relationships are not straight lines.
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
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

data = fetch_california_housing(as_frame=True)

df = data.frame.copy()

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)


# ------------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET, THEN SPLIT
# ------------------------------------------------------------

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data: ", X_test.shape)


# ------------------------------------------------------------
# 3. BASELINE MODEL - LINEAR REGRESSION
# ------------------------------------------------------------

# A single straight-line model. Fast and simple, but it
# cannot capture curves or interactions between features.

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)


# ------------------------------------------------------------
# 4. ENSEMBLE MODEL - RANDOM FOREST (100 TREES)
# ------------------------------------------------------------

# A Random Forest trains 100 decision trees, each on a random
# sample of rows (bootstrap) and a random subset of features.
# The final prediction = the AVERAGE of all 100 trees.
# Averaging shaky trees gives one very stable model.

rf_model = RandomForestRegressor(
    n_estimators=100,   # number of trees in the forest
    max_depth=15,       # cap tree size to avoid crazy overfitting
    random_state=42,
    n_jobs=-1           # use all CPU cores (faster fitting)
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("\nBoth Models Trained Successfully!")


# ------------------------------------------------------------
# 5. EVALUATE BOTH MODELS WITH THE SAME METRICS
# ------------------------------------------------------------

def evaluate_model(name, true_values, predictions):
    """Return a one-row table of common regression metrics."""

    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    print(f"\n{name}")
    print("  MSE : ", round(mse, 4))
    print("  RMSE: ", round(rmse, 4))
    print("  MAE : ", round(mae, 4))
    print("  R2  : ", round(r2, 4))

    return {
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }


linear_row = evaluate_model(
    "Linear Regression", y_test, linear_predictions
)
rf_row = evaluate_model(
    "Random Forest", y_test, rf_predictions
)

results = pd.DataFrame([linear_row, rf_row])

print("\n========== SIDE-BY-SIDE COMPARISON ==========")
print(results.round(4))


# ------------------------------------------------------------
# 6. VISUALIZE: ERROR COMPARISON
# ------------------------------------------------------------

# Lower RMSE / MAE and higher R2 = better model.

plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["RMSE"], color="steelblue", label="RMSE")
plt.bar(results["Model"], results["MAE"], color="orange", label="MAE")
plt.title("Error Comparison (lower is better)")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(results["Model"], results["R2 Score"], color="seagreen")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison (higher is better)")
plt.show()


# ------------------------------------------------------------
# 7. ACTUAL VS PREDICTED (RANDOM FOREST)
# ------------------------------------------------------------

# Points near the diagonal = good predictions.

plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_predictions, alpha=0.3)
plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Random Forest: Actual vs Predicted")
plt.show()


# ------------------------------------------------------------
# 8. FEATURE IMPORTANCE (WHAT DOES THE FOREST LOOK AT?)
# ------------------------------------------------------------

# Each tree contributes a "how much did this feature reduce
# error" score during fitting. Averaged over all trees it
# tells us which features drive the predictions.

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance.round(4))

plt.figure(figsize=(9, 5))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.show()


# ------------------------------------------------------------
# 9. FINAL RESULT
# ------------------------------------------------------------

linear_r2 = linear_row["R2 Score"]
rf_r2 = rf_row["R2 Score"]

print("\n========== FINAL RESULT ==========")

if rf_r2 > linear_r2:
    winner = "Random Forest"
else:
    winner = "Linear Regression"

print(f"The winner is: {winner}")

print("\nLinear Regression R2:", round(linear_r2, 4))
print("Random Forest R2:    ", round(rf_r2, 4))

print("\nTakeaway: when the data has curved/interaction effects,")
print("an ensemble of shallow decision trees generalizes far")
print("better than one straight line - at the cost of being")
print("harder to interpret and slower to train.")

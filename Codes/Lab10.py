
# ============================================================
# LAB 10: ENSEMBLE LEARNING
# Random Forest vs Linear Regression
# Dataset: California Housing
# ============================================================

# 1. Import Libraries

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

# 2. Load Dataset

data = fetch_california_housing(as_frame=True)

df = data.frame.copy()

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 3. Separate Features and Target

X = df.drop("MedHouseVal", axis=1)

y = df["MedHouseVal"]

# 4. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 5. Baseline Model - Linear Regression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# 6. Ensemble Model - Random Forest

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\nBoth Models Trained Successfully!")

# 7. Evaluate Linear Regression

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(linear_mse)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

# 8. Evaluate Random Forest

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(rf_mse)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

# 9. Compare Model Performance

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MSE": [
        linear_mse,
        rf_mse
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse
    ],

    "MAE": [
        linear_mae,
        rf_mae
    ],

    "R2 Score": [
        linear_r2,
        rf_r2
    ]
})

print("\n========== MODEL COMPARISON ==========")

print(results.round(4))

# 10. Visualize Model Comparison

results.set_index("Model")[
    ["RMSE", "MAE"]
].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Performance Comparison of Models")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.show()

# 11. Compare R2 Scores

plt.figure(figsize=(7, 5))

plt.bar(
    results["Model"],
    results["R2 Score"]
)

plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")
plt.show()

# 12. Actual vs Predicted Values

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.5
)

plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")

plt.title("Random Forest: Actual vs Predicted")

plt.show()

# 13. Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.show()

# 14. Final Result

print("\n========== FINAL RESULT ==========")

if rf_r2 > linear_r2:
    print("Random Forest performed better than Linear Regression.")
else:
    print("Linear Regression performed better than Random Forest.")

print("\nLinear Regression R2:", round(linear_r2, 4))

print("Random Forest R2:", round(rf_r2, 4))

print("\nEnsemble Learning Experiment Completed!")

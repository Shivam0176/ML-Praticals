
# ============================================================
# LAB 2: MULTIPLE LINEAR REGRESSION
# Dataset: California Housing Dataset
# ============================================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 2. Load Dataset

housing = fetch_california_housing(as_frame=True)

df = housing.frame.copy()

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())


# 3. Dataset Exploration

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# 4. Visualize Dataset

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()


# 5. Separate Independent and Dependent Variables

X = df.drop("MedHouseVal", axis=1)

y = df["MedHouseVal"]

print("\nIndependent Variables:")
print(X.head())

print("\nDependent Variable:")
print(y.head())


# 6. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# 7. Create Multiple Linear Regression Model

model = LinearRegression()


# 8. Train the Model

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# 9. Make Predictions

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.head())

print("\nPredicted Values:")
print(y_pred[:5])


# 10. Evaluate Model Performance

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Mean Squared Error (MSE):", round(mse, 4))

print("Root Mean Squared Error (RMSE):", round(rmse, 4))

print("R2 Score:", round(r2, 4))


# 11. Display Regression Coefficients

print("\nRegression Coefficients:")

for feature, coefficient in zip(
    X.columns,
    model.coef_
):
    print(feature, ":", round(coefficient, 4))

print("\nIntercept:", round(model.intercept_, 4))


# 12. Actual vs Predicted Values

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual House Values")

plt.ylabel("Predicted House Values")

plt.title("Actual vs Predicted Values")

plt.show()


# 13. Residual Plot

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()


# 14. Final Output

print("\n======================================")
print("MULTIPLE LINEAR REGRESSION COMPLETED")
print("======================================")

print("Dataset: California Housing")

print("Number of Features:", X.shape[1])

print("Training Samples:", len(X_train))

print("Testing Samples:", len(X_test))

print("MSE:", round(mse, 4))

print("RMSE:", round(rmse, 4))

print("R2 Score:", round(r2, 4))

print("\nModel successfully trained and evaluated!")

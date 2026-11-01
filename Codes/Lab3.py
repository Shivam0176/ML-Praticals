# ============================================================
# LAB 3: MULTIPLE LINEAR REGRESSION
# Dataset: California Housing Dataset
#
# Goal: predict a house's median value from several
# numeric features using a single linear model.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

housing = fetch_california_housing(as_frame=True)

df = housing.frame.copy()

print("Dataset Loaded!")
print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 2. EXPLORE THE DATASET
# ------------------------------------------------------------

print("\nDataset Shape (rows, columns):")
print(df.shape)

print("\nColumn Info:")
df.info()

print("\nStatistical Description:")
print(df.describe())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. VISUALIZE RELATIONSHIPS (CORRELATION HEATMAP)
# ------------------------------------------------------------

# Each cell is the Pearson correlation (-1 to +1) between
# two columns. Look at the last row: how each feature
# relates to the target MedHouseVal.

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------
# 4. SEPARATE INPUTS (X) AND TARGET (y)
# ------------------------------------------------------------

X = df.drop("MedHouseVal", axis=1)  # all 8 features
y = df["MedHouseVal"]               # the value we predict

print("\nIndependent Variables (X):")
print(X.head())

print("\nDependent Variable (y):")
print(y.head())


# ------------------------------------------------------------
# 5. SPLIT INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

# 80% of rows are used to train the model,
# 20% are kept aside to test it on unseen data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape: ", X_test.shape)


# ------------------------------------------------------------
# 6. CREATE AND TRAIN THE MODEL
# ------------------------------------------------------------

# The model learns the straight-line equation:
#   predicted value = b0 + b1*MedInc + b2*HouseAge + ...

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ------------------------------------------------------------
# 7. MAKE PREDICTIONS ON THE TEST SET
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual Values (first 5):")
print(y_test.head().to_numpy())

print("\nPredicted Values (first 5):")
print(np.round(y_pred[:5], 2))


# ------------------------------------------------------------
# 8. EVALUATE THE MODEL
# ------------------------------------------------------------

# MSE  - average of squared errors (bigger errors punished more)
# RMSE - square root of MSE, in the same units as house value
# R2   - fraction of the target's variance explained (1 = perfect)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Mean Squared Error (MSE):        ", round(mse, 4))
print("Root Mean Squared Error (RMSE):  ", round(rmse, 4))
print("R2 Score:                        ", round(r2, 4))


# ------------------------------------------------------------
# 9. LOOK AT THE REGRESSION COEFFICIENTS
# ------------------------------------------------------------

# Each coefficient says: for a one-unit increase in that
# feature, the predicted value changes by this much,
# keeping all other features fixed.

print("\nRegression Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"  {feature:<10} : {round(coefficient, 4)}")

print("Intercept:", round(model.intercept_, 4))


# ------------------------------------------------------------
# 10. PLOT 1: ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

# Points close to the diagonal line = good predictions.

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Actual vs Predicted Values")
plt.show()


# ------------------------------------------------------------
# 11. PLOT 2: RESIDUAL PLOT
# ------------------------------------------------------------

# A residual = actual value - predicted value.
# Ideally residuals scatter randomly around 0.
# Patterns mean the straight line missed something.

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n======================================")
print("MULTIPLE LINEAR REGRESSION COMPLETED")
print("======================================")

print("Dataset:            California Housing")
print("Number of Features:", X.shape[1])
print("Training Samples:  ", len(X_train))
print("Testing Samples:   ", len(X_test))
print("MSE:    ", round(mse, 4))
print("RMSE:   ", round(rmse, 4))
print("R2:     ", round(r2, 4))

print("\nModel successfully trained and evaluated!")

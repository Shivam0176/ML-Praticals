# ============================================================
# LAB 3: MULTIPLE LINEAR REGRESSION
# Dataset: California Housing (bundled with sklearn, no internet)
#
# Big picture: predict a house value from several features
# using a straight-line equation:
#     value = b0 + b1*x1 + b2*x2 + ... + b8*x8
# The model "learns" the b values from training data.
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
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("Dataset loaded. First 5 rows:")
print(df.head())
print("\nShape (rows, columns):", df.shape)

#quick summary: range of values, no missing values?
print("\nAre there missing values?")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 2: VISUALIZE HOW FEATURES RELATE TO EACH OTHER
# ------------------------------------------------------------

# The heatmap shows the correlation between every pair of
# columns (-1 to +1). The last column tells us which features
# are most related to the house value we want to predict.

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------
# STEP 3: SEPARATE INPUT (X) AND TARGET (y)
# ------------------------------------------------------------

X = df.drop("MedHouseVal", axis=1)   # 8 features
y = df["MedHouseVal"]                # the value to predict


# ------------------------------------------------------------
# STEP 4: SPLIT INTO TRAIN (80%) AND TEST (20%)
# ------------------------------------------------------------

# The model learns from the training part only.
# The test part is kept secret until the end, to obtain a fair
# estimate of how the model performs on new, unseen houses.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)


# ------------------------------------------------------------
# STEP 5: TRAIN THE MODEL
# ------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained!")


# ------------------------------------------------------------
# STEP 6: PREDICT ON THE TEST SET
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual values (first 5):   ", y_test.head().to_numpy())
print("Predicted values (first 5):", np.round(y_pred[:5], 2))


# ------------------------------------------------------------
# STEP 7: MEASURE HOW GOOD THE PREDICTIONS ARE
# ------------------------------------------------------------

# MSE  - average squared error (small errors are "free",
#        large errors are punished a lot)
# RMSE - square root of MSE, in house-value units again
# R2   - 1.0 = perfect, 0.0 = as good as always predicting
#        the average, negative = worse than that

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL EVALUATION ===")
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))


# ------------------------------------------------------------
# STEP 8: READ THE MODEL - WHAT DID IT LEARN?
# ------------------------------------------------------------

# Each coefficient says how much the predicted value changes
# when that feature increases by 1 (other features fixed).
# Example: MedInc (median income) has the biggest effect.

print("\n=== REGRESSION COEFFICIENTS ===")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"  {feature:<10} : {round(coefficient, 4)}")
print("  intercept : ", round(model.intercept_, 4))


# ------------------------------------------------------------
# STEP 9: PLOTS TO CHECK THE MODEL
# ------------------------------------------------------------

# Plot 1: actual vs predicted. Points close to the diagonal
# line mean accurate predictions.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual house value")
plt.ylabel("Predicted house value")
plt.title("Actual vs Predicted")
plt.show()

# Plot 2: residuals (error = actual - predicted) should look
# like a random cloud around 0. Visible patterns mean a
# straight line was not the perfect choice (see Lab 4!).
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted value")
plt.ylabel("Residual (error)")
plt.title("Residual Plot")
plt.show()

print("\nLinear regression experiment complete! R2 =", round(r2, 4))

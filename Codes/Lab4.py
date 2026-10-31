
# ============================================================
# LAB 4: POLYNOMIAL REGRESSION
# Study of Model Complexity and Bias-Variance Trade-off
# Dataset: Auto MPG
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
data = fetch_openml(name="autoMpg", version=1, as_frame=True)
df = data.frame.copy()

print("First 5 Rows:")
print(df.head())

# 2. Data Cleaning
df["horsepower"] = pd.to_numeric(
    df["horsepower"], errors="coerce"
)

df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")

df = df.dropna(subset=["horsepower", "mpg"])

# Use horsepower as input and mpg as target
X = df[["horsepower"]]
y = df["mpg"]

print("\nDataset Shape:", df.shape)

# 3. Visualize Original Data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.5)
plt.xlabel("Horsepower")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Horsepower vs MPG")
plt.show()

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Polynomial Models (Degree 1 to 15)
degrees = range(1, 16)

train_errors = []
test_errors = []
cv_errors = []
models = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_errors.append(
        np.sqrt(mean_squared_error(y_train, train_pred))
    )

    test_errors.append(
        np.sqrt(mean_squared_error(y_test, test_pred))
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )

    cv_errors.append(-scores.mean())
    models.append(model)

# 6. Plot Training and Testing Errors
plt.figure(figsize=(10, 6))

plt.plot(degrees, train_errors, marker="o", label="Training RMSE")
plt.plot(degrees, test_errors, marker="o", label="Testing RMSE")
plt.plot(degrees, cv_errors, marker="s", label="5-Fold CV RMSE")

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Effect of Polynomial Degree on Prediction Error")
plt.legend()
plt.grid()
plt.show()

# 7. Find Optimal Degree using Cross-Validation
best_index = np.argmin(cv_errors)
best_degree = list(degrees)[best_index]

best_model = models[best_index]

print("\nOptimal Polynomial Degree:", best_degree)
print("Best CV RMSE:", round(cv_errors[best_index], 4))

# 8. Evaluate Best Model
y_pred = best_model.predict(X_test)

print("\nModel Evaluation")
print("Test RMSE:",
      round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

print("Test R2 Score:",
      round(r2_score(y_test, y_pred), 4))

# 9. Plot Polynomial Regression Curves
X_plot = pd.DataFrame({
    "horsepower": np.linspace(
        X["horsepower"].min(),
        X["horsepower"].max(),
        300
    )
})

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label="Actual Data")

for degree in [1, 2, 3, 5, 10, 15]:
    model = models[degree - 1]
    plt.plot(
        X_plot,
        model.predict(X_plot),
        label=f"Degree {degree}"
    )

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression of Different Degrees")
plt.legend()
plt.show()

# 10. Bootstrap Bias-Variance Estimation
B = 100
rng = np.random.default_rng(42)

predictions = np.zeros((B, len(X_test)))

for b in range(B):

    indices = rng.integers(0, len(X_train), len(X_train))

    X_boot = X_train.iloc[indices]
    y_boot = y_train.iloc[indices]

    model = models[best_index]

    # Refit a fresh model for each bootstrap sample
    from sklearn.base import clone
    boot_model = clone(model)

    boot_model.fit(X_boot, y_boot)

    predictions[b] = boot_model.predict(X_test)

mean_predictions = predictions.mean(axis=0)

bias_squared = np.mean(
    (mean_predictions - y_test.to_numpy()) ** 2
)

variance = np.mean(
    np.var(predictions, axis=0)
)

print("\nBootstrap Bias-Variance Analysis")
print("Estimated Bias Squared:", round(bias_squared, 4))
print("Estimated Variance:", round(variance, 4))
print("Approximate Total Error:", round(bias_squared + variance, 4))

print("\nPolynomial Regression Completed!")

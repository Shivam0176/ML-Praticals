
# ============================================================
# LAB 5: RIDGE, LASSO AND ELASTIC NET REGRESSION
# Dataset: Wine Quality (Red Wine)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Load Dataset
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

df = pd.read_csv(url, sep=";")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Separate Features and Target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Correlation Matrix
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()

# 4. Variance Inflation Factor (VIF)
X_vif = pd.DataFrame(
    StandardScaler().fit_transform(X),
    columns=X.columns
)

vif_data = pd.DataFrame()

vif_data["Feature"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i)
    for i in range(X_vif.shape[1])
]

print("\nVariance Inflation Factor:")
print(vif_data)

# 5. Condition Number
condition_number = np.linalg.cond(X_vif.values)

print("\nCondition Number:", round(condition_number, 4))

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Define Models
models = {
    "OLS": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=10000))
    ]),

    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000
        ))
    ])
}

# 8. Train and Evaluate Models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mse, rmse, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MSE", "RMSE", "R2 Score"]
)

print("\nModel Comparison:")
print(results_df)

# 9. Plot Model Comparison
results_df.set_index("Model")[["RMSE"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.show()

# 10. Ridge and Lasso Coefficients
ridge_model = models["Ridge"].named_steps["model"]
lasso_model = models["Lasso"].named_steps["model"]

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\nCoefficient Comparison:")
print(coefficients)

# 11. Plot Coefficients
coefficients.set_index("Feature").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Ridge vs Lasso Coefficients")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 12. Coefficient Paths
alphas = np.logspace(-3, 2, 50)

ridge_paths = []
lasso_paths = []

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for alpha in alphas:

    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=10000)

    ridge.fit(X_scaled, y)
    lasso.fit(X_scaled, y)

    ridge_paths.append(ridge.coef_)
    lasso_paths.append(lasso.coef_)

ridge_paths = np.array(ridge_paths)
lasso_paths = np.array(lasso_paths)

plt.figure(figsize=(10, 6))

for i, feature in enumerate(X.columns):
    plt.plot(alphas, ridge_paths[:, i], label=feature)

plt.xscale("log")
plt.title("Ridge Coefficient Paths")
plt.xlabel("Alpha")
plt.ylabel("Coefficient")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

plt.figure(figsize=(10, 6))

for i, feature in enumerate(X.columns):
    plt.plot(alphas, lasso_paths[:, i], label=feature)

plt.xscale("log")
plt.title("Lasso Coefficient Paths")
plt.xlabel("Alpha")
plt.ylabel("Coefficient")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

print("\nRegularization Analysis Completed!")

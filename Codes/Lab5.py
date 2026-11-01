# ============================================================
# LAB 5: RIDGE, LASSO AND ELASTIC NET REGRESSION
# Dataset: Wine Quality (Red Wine)
#
# Goal: fix the problem that occurs when features are highly
# correlated with each other (multicollinearity). Regularized
# regression shrinks the coefficients so the model is stable
# and generalizes better.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ------------------------------------------------------------
# 1. LOAD THE DATASET (needs internet - UCI wine quality)
# ------------------------------------------------------------

url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

df = pd.read_csv(url, sep=";")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)


# ------------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

# Target: quality score of the wine (0-10).
# Features: 11 physicochemical measurements.

X = df.drop("quality", axis=1)
y = df["quality"]


# ------------------------------------------------------------
# 3. CORRELATION MATRIX (WHO IS RELATED TO WHOM?)
# ------------------------------------------------------------

# High off-diagonal values show features that move together.
# Wine is famous for this: acidity features and the two
# sulfur-dioxide features are strongly correlated.

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# ------------------------------------------------------------
# 4. VARIANCE INFLATION FACTOR (VIF)
# ------------------------------------------------------------

# VIF asks: "can this feature be predicted by the OTHER features?"
#   VIF = 1   -> not correlated with others at all
#   VIF > 5   -> serious multicollinearity
#   VIF > 10  -> very serious

# Scale first so every feature is comparable.
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

vif_table = pd.DataFrame()
vif_table["Feature"] = X.columns
vif_table["VIF"] = [
    variance_inflation_factor(X_scaled_full, i)
    for i in range(X_scaled_full.shape[1])
]

print("\nVariance Inflation Factor (VIF > 5 is a warning):")
print(vif_table)


# ------------------------------------------------------------
# 5. CONDITION NUMBER
# ------------------------------------------------------------

# Another collinearity check in one number:
#   large value (hundreds or more) = unstable problem.

condition_number = np.linalg.cond(X_scaled_full)

print("\nCondition Number:", round(condition_number, 4))


# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale using statistics from the TRAINING set only,
# then apply the same scaling to the test set.
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


# ------------------------------------------------------------
# 7. DEFINE THE FOUR MODELS (NO PIPELINE - PLAIN OBJECTS)
# ------------------------------------------------------------

# Each one is linear regression PLUS a size penalty:
#   Ridge       -> penalizes the SUM OF SQUARES of coefficients (L2)
#   Lasso       -> penalizes the SUM OF |coefficients| (L1)
#   Elastic Net -> a mix of both penalties
# alpha = penalty strength (0 means "no penalty" = plain OLS).

linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.01, max_iter=10000)
elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)

models = {
    "OLS": linear_model,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "ElasticNet": elastic_model
}


# ------------------------------------------------------------
# 8. TRAIN AND EVALUATE ALL FOUR MODELS
# ------------------------------------------------------------

results = []

for name, one_model in models.items():

    one_model.fit(X_train_scaled, y_train)

    predictions = one_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results.append([name, mse, rmse, r2])

results_table = pd.DataFrame(
    results,
    columns=["Model", "MSE", "RMSE", "R2 Score"]
)

print("\n========== MODEL COMPARISON ==========")
print(results_table.round(4))


# ------------------------------------------------------------
# 9. PLOT MODEL COMPARISON
# ------------------------------------------------------------

plot_data = pd.DataFrame({
    "Model": results_table["Model"],
    "RMSE": results_table["RMSE"]
})

plt.figure(figsize=(9, 5))
plt.bar(plot_data["Model"], plot_data["RMSE"], color="steelblue")
plt.title("RMSE Comparison (lower is better)")
plt.ylabel("RMSE")
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(plot_data["Model"], results_table["R2 Score"], color="darkorange")
plt.title("R2 Score Comparison (higher is better)")
plt.ylabel("R2 Score")
plt.show()


# ------------------------------------------------------------
# 10. COMPARE RIDGE AND LASSO COEFFICIENTS
# ------------------------------------------------------------

# OLS coefficients are hard to interpret when features are
# correlated (huge, cancelling values). Ridge shrinks them all,
# and Lasso pushes the least useful ones to exactly ZERO.

coefficient_table = pd.DataFrame({
    "Feature": X.columns,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\n========== COEFFICIENT COMPARISON ==========")
print(coefficient_table.round(4))

# Positions on the x-axis for each feature
feature_positions = np.arange(len(X.columns))

bar_width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(
    feature_positions - bar_width / 2,
    ridge_model.coef_,
    width=bar_width,
    label="Ridge"
)
plt.bar(
    feature_positions + bar_width / 2,
    lasso_model.coef_,
    width=bar_width,
    label="Lasso"
)
plt.xticks(feature_positions, X.columns, rotation=45)
plt.title("Ridge vs Lasso Coefficients")
plt.ylabel("Coefficient Value")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. COEFFICIENT PATHS (HOW COEFFICIENTS CHANGE WITH ALPHA)
# ------------------------------------------------------------

# We grow alpha from small to large and record the coefficients.
#   Ridge : paths shrink smoothly toward 0 (never exactly 0)
#   Lasso : paths hit 0 one by one -> automatic feature selection

alpha_values = np.logspace(-3, 2, 50)   # 50 values from 0.001 to 100

ridge_paths = []
lasso_paths = []

for alpha in alpha_values:

    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=10000)

    # Fit both on ALL the data - this plot is an inspection tool,
    # not a scoring experiment.
    ridge.fit(X_scaled_full, y)
    lasso.fit(X_scaled_full, y)

    ridge_paths.append(ridge.coef_)
    lasso_paths.append(lasso.coef_)

ridge_paths = np.array(ridge_paths)   # rows: alpha, columns: features
lasso_paths = np.array(lasso_paths)


# --- Ridge plot: one line per feature ---

plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.plot(alpha_values, ridge_paths[:, i], label=feature)
plt.xscale("log")
plt.title("Ridge Coefficient Paths (shrink smoothly to 0)")
plt.xlabel("Alpha (penalty strength)")
plt.ylabel("Coefficient")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# --- Lasso plot: lines crash flat onto zero ---

plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.plot(alpha_values, lasso_paths[:, i], label=feature)
plt.xscale("log")
plt.title("Lasso Coefficient Paths (hit exactly 0)")
plt.xlabel("Alpha (penalty strength)")
plt.ylabel("Coefficient")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n=====================================")
print("REGULARIZATION ANALYSIS COMPLETED")
print("=====================================")

best_row = results_table["R2 Score"].idxmax()
best_name = results_table.loc[best_row, "Model"]
best_r2 = results_table.loc[best_row, "R2 Score"]

print("Best model by R2 Score:", best_name, "-", round(best_r2, 4))
print("Ridge shrinks coefficients; Lasso removes features by making")
print("their coefficients exactly zero. Elastic Net does something in")
print("between, which helps when features travel in correlated groups.")

# ============================================================
# LAB 5: RIDGE, LASSO AND ELASTIC NET REGULARIZATION
# Dataset: Red Wine Quality (UCI) - needs internet
#
# Big picture: wine features are correlated with each other,
# which makes plain linear regression unstable. We add a
# penalty on coefficient size to make it behave:
#   Ridge   - L2 penalty, shrinks coefficients, keeps all
#   Lasso   - L1 penalty, can set some to exactly ZERO
#   Elastic - mix of both
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
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------

url = ("https://archive.ics.uci.edu/ml/"
       "machine-learning-databases/wine-quality/winequality-red.csv")
df = pd.read_csv(url, sep=";")

print("Dataset loaded. First 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# ------------------------------------------------------------
# STEP 2: PROVE THE FEATURES ARE CORRELATED
# ------------------------------------------------------------

# The heatmap shows feature-to-feature correlations. The two
# sulfur-dioxide columns and the acidity group are strongly
# connected - this is exactly the problem Ridge/Lasso solve.

X = df.drop("quality", axis=1)
y = df["quality"]

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# ------------------------------------------------------------
# STEP 3: VARIANCE INFLATION FACTOR (VIF)
# ------------------------------------------------------------

# VIF answers per feature: "how predictable is this feature
# from all the others?"   1 = independent,  >5 = a problem.
# Requires scaled data, so we scale first.

feature_scaler = StandardScaler()
X_full_scaled = feature_scaler.fit_transform(X)

vif_table = pd.DataFrame()
vif_table["Feature"] = X.columns
vif_table["VIF"] = [variance_inflation_factor(X_full_scaled, i)
                    for i in range(X_full_scaled.shape[1])]

print("\n=== VARIANCE INFLATION FACTOR ===")
print(vif_table.round(2))

# One summary number for the whole matrix: if it is large
# (hundreds+), the regression problem is unstable.
print("\nCondition number:", round(np.linalg.cond(X_full_scaled), 2))


# ------------------------------------------------------------
# STEP 4: SPLIT AND SCALE (USING TRAIN STATISTICS ONLY)
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


# ------------------------------------------------------------
# STEP 5: TRAIN FOUR MODELS AND COMPARE
# ------------------------------------------------------------

# alpha = strength of the size penalty. alpha = 0 would
# turn every model back into plain linear regression.

linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.01, max_iter=10000)
elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)

model_list = [
    ["OLS", linear_model],
    ["Ridge", ridge_model],
    ["Lasso", lasso_model],
    ["ElasticNet", elastic_model],
]

results = []

for name, model in model_list:
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    results.append([name, mse, np.sqrt(mse), r2_score(y_test, predictions)])

results_table = pd.DataFrame(
    results, columns=["Model", "MSE", "RMSE", "R2 Score"])

print("\n=== MODEL COMPARISON ===")
print(results_table.round(4))

plt.figure(figsize=(9, 5))
plt.bar(results_table["Model"], results_table["R2 Score"],
        color="darkorange")
plt.title("R2 Score Comparison (higher is better)")
plt.ylabel("R2 Score")
plt.show()


# ------------------------------------------------------------
# STEP 6: LOOK AT WHAT REGULARIZATION DID TO THE COEFFICIENTS
# ------------------------------------------------------------

# Ridge compressed all coefficients; Lasso zeroed some out
# (those features got "deleted" automatically).

coefficient_table = pd.DataFrame({
    "Feature": X.columns,
    "OLS": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_,
})

print("\n=== COEFFICIENTS: OLS vs RIDGE vs LASSO ===")
print(coefficient_table.round(4))

positions = np.arange(len(X.columns))
bar_width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(positions - bar_width, coefficient_table["OLS"],
        width=bar_width, label="OLS")
plt.bar(positions, coefficient_table["Ridge"],
        width=bar_width, label="Ridge")
plt.bar(positions + bar_width, coefficient_table["Lasso"],
        width=bar_width, label="Lasso")
plt.xticks(positions, X.columns, rotation=45)
plt.title("Coefficients by Model - Lasso pushes some to zero")
plt.ylabel("Coefficient value")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 7: COEFFICIENT PATHS - EFFECT OF GROWING alpha
# ------------------------------------------------------------

# We fit Ridge (then Lasso) for 50 alpha values and record
# every coefficient. One line per feature:
#   Ridge: lines shrink smoothly and never quite reach 0
#   Lasso: lines crash flat onto 0, one after another

alpha_values = np.logspace(-3, 2, 50)     # 0.001 ... 100

ridge_paths = []
lasso_paths = []

for alpha in alpha_values:

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_full_scaled, y)
    ridge_paths.append(ridge.coef_)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_full_scaled, y)
    lasso_paths.append(lasso.coef_)

ridge_paths = np.array(ridge_paths)
lasso_paths = np.array(lasso_paths)

# Ridge plot
plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.plot(alpha_values, ridge_paths[:, i], label=feature)
plt.xscale("log")
plt.title("Ridge Paths - smooth shrinkage to almost zero")
plt.xlabel("Alpha")
plt.ylabel("Coefficient")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# Lasso plot
plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.plot(alpha_values, lasso_paths[:, i], label=feature)
plt.xscale("log")
plt.title("Lasso Paths - features eliminated one by one")
plt.xlabel("Alpha")
plt.ylabel("Coefficient")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

print("\nRegularization analysis complete - bigger alpha means")
print("a simpler model (more bias, less variance).")

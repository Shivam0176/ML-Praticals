# ============================================================
# LAB 4: POLYNOMIAL REGRESSION
# Study of Model Complexity and Bias-Variance Trade-off
# Dataset: Auto MPG
#
# Goal: see what happens when we make a model more
# and more complex (higher polynomial degree):
#   - too simple   -> underfits (high bias)
#   - too complex  -> overfits (high variance)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def fit_polynomial_model(degree, X_train, y_train):
    """Train one polynomial model and return its 3 parts.

    Step 1: PolynomialFeatures turns x into x, x^2, ..., x^degree
    Step 2: StandardScaler shrinks the huge powers to a safe range
    Step 3: LinearRegression fits the equation on the new features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    scaler = StandardScaler()
    lin_reg = LinearRegression()

    X_poly = poly.fit_transform(X_train)        # make powers of x
    X_scaled = scaler.fit_transform(X_poly)     # rescale each power
    lin_reg.fit(X_scaled, y_train)              # fit the equation

    return poly, scaler, lin_reg


def predict_with_parts(poly, scaler, lin_reg, X_data):
    """Apply the same three steps to new data."""
    X_poly = poly.transform(X_data)
    X_scaled = scaler.transform(X_poly)
    return lin_reg.predict(X_scaled)


# ------------------------------------------------------------
# 1. LOAD THE DATASET (needs internet)
# ------------------------------------------------------------

data = fetch_openml(name="autoMpg", version=1, as_frame=True)
df = data.frame.copy()

print("First 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 2. CLEAN THE DATA
# ------------------------------------------------------------

# On OpenML the MPG column is stored under the name "class".
df = df.rename(columns={"class": "mpg"})

# "horsepower" contains a few "?" values.
# to_numeric(..., errors="coerce") turns "?" into NaN,
# then we remove rows that are missing.

df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df["mpg"] = pd.to_numeric(df["mpg"], errors="coerce")
df = df.dropna(subset=["horsepower", "mpg"])

# Input: horsepower.  Target: mpg.
# We keep it as a 1-column table because PolynomialFeatures
# expects a table, not a plain list.

X = df[["horsepower"]]
y = df["mpg"]

print("\nDataset Shape after cleaning:", df.shape)


# ------------------------------------------------------------
# 3. VISUALIZE THE RAW DATA
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.5)
plt.xlabel("Horsepower")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Horsepower vs MPG - the relationship is curved, not straight")
plt.show()


# ------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ------------------------------------------------------------
# 5. TRAIN MODELS OF DEGREE 1 TO 15 AND RECORD THEIR ERRORS
# ------------------------------------------------------------

degrees = list(range(1, 16))       # 1, 2, 3, ..., 15

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

train_errors = []   # RMSE on the training data
test_errors = []    # RMSE on unseen test data
cv_errors = []      # average RMSE from 5-fold cross-validation

for degree in degrees:

    # --- A) fit on the training data ---
    poly, scaler, lin_reg = fit_polynomial_model(degree, X_train, y_train)

    train_pred = predict_with_parts(poly, scaler, lin_reg, X_train)
    test_pred = predict_with_parts(poly, scaler, lin_reg, X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # --- B) 5-fold cross-validation (all inside training data) ---
    # Split the training data into 5 parts. Each time, train on
    # 4 parts and validate on the 1 held-out part.
    cv_rmse = 0
    for subset_a, subset_b in kfold.split(X_train):

        X_fold_train = X_train.iloc[subset_a]
        y_fold_train = y_train.iloc[subset_a]
        X_fold_val = X_train.iloc[subset_b]
        y_fold_val = y_train.iloc[subset_b]

        fold_poly, fold_scaler, fold_reg = fit_polynomial_model(
            degree, X_fold_train, y_fold_train
        )
        fold_pred = predict_with_parts(fold_poly, fold_scaler, fold_reg, X_fold_val)

        cv_rmse = cv_rmse + np.sqrt(mean_squared_error(y_fold_val, fold_pred))

    cv_rmse = cv_rmse / 5  # average over the 5 folds

    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    cv_errors.append(cv_rmse)

    print(f"Degree {degree:>2}:  "
          f"train RMSE = {round(train_rmse, 3)},  "
          f"test RMSE = {round(test_rmse, 3)},  "
          f"CV RMSE = {round(cv_rmse, 3)}")


# ------------------------------------------------------------
# 6. PLOT TRAINING / TESTING / CV ERRORS vs DEGREE
# ------------------------------------------------------------

# Expected shape: errors drop at first, then the TEST/CV
# error goes back up while the TRAIN error keeps falling.
# That rise = overfitting.

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


# ------------------------------------------------------------
# 7. PICK THE BEST DEGREE USING CROSS-VALIDATION
# ------------------------------------------------------------

best_index = np.argmin(cv_errors)
best_degree = degrees[best_index]

plt.figure(figsize=(10, 4))
plt.bar([str(d) for d in degrees], cv_errors)
plt.xlabel("Polynomial Degree")
plt.ylabel("CV RMSE")
plt.title("Cross-Validation Error for Each Degree")
plt.show()

print("\nOptimal Polynomial Degree:", best_degree)
print("Best CV RMSE:", round(cv_errors[best_index], 4))


# ------------------------------------------------------------
# 8. EVALUATE THE BEST MODEL ON THE TEST SET
# ------------------------------------------------------------

best_poly, best_scaler, best_reg = fit_polynomial_model(
    best_degree, X_train, y_train
)

final_pred = predict_with_parts(best_poly, best_scaler, best_reg, X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
final_r2 = r2_score(y_test, final_pred)

print("\nBest Model on Test Data")
print("Test RMSE:", round(final_rmse, 4))
print("Test R2:  ", round(final_r2, 4))


# ------------------------------------------------------------
# 9. VISUALIZE CURVES OF DIFFERENT DEGREES
# ------------------------------------------------------------

# We draw a smooth line from low to high horsepower for a few
# degrees. Low degrees = too simple. High degrees = wild curves.

hp_grid = np.linspace(
    X["horsepower"].min(),
    X["horsepower"].max(),
    300
)
hp_grid_table = pd.DataFrame({"horsepower": hp_grid})

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.4, label="Actual Data")

for degree in [1, 2, 3, 5, 10, 15]:
    poly, scaler, lin_reg = fit_polynomial_model(degree, X_train, y_train)
    grid_pred = predict_with_parts(poly, scaler, lin_reg, hp_grid_table)
    plt.plot(hp_grid_table, grid_pred, label=f"Degree {degree}")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression Curves of Different Degrees")
plt.legend()
plt.show()


# ------------------------------------------------------------
# 10. BOOTSTRAP: ESTIMATE BIAS AND VARIANCE OF THE BEST MODEL
# ------------------------------------------------------------

# Idea: imitate "many different training sets" by sampling the
# training data WITH REPLACEMENT (a bootstrap sample), fit the
# best model on each sample, and watch how its predictions move.
#
#   bias^2  = squared distance of the AVERAGE prediction from truth
#   variance = average spread of predictions across the samples

n_bootstraps = 30
random_state = np.random.default_rng(42)

y_test_values = y_test.to_numpy()
prediction_matrix = np.zeros((n_bootstraps, len(X_test)))

for bootstrap_number in range(n_bootstraps):

    # Random indices chosen with replacement = a bootstrap sample
    sample_indices = random_state.integers(0, len(X_train), len(X_train))

    X_sample = X_train.iloc[sample_indices]
    y_sample = y_train.iloc[sample_indices]

    poly, scaler, lin_reg = fit_polynomial_model(
        best_degree, X_sample, y_sample
    )

    prediction_matrix[bootstrap_number] = predict_with_parts(
        poly, scaler, lin_reg, X_test
    )

average_predictions = prediction_matrix.mean(axis=0)

bias_squared = np.mean((average_predictions - y_test_values) ** 2)
variance = np.mean(np.var(prediction_matrix, axis=0))

print("\n========== BOOTSTRAP BIAS-VARIANCE ANALYSIS ==========")
print("Estimated Bias Squared:  ", round(bias_squared, 4))
print("Estimated Variance:      ", round(variance, 4))
print("Approximate Total Error: ", round(bias_squared + variance, 4))
print("(true test error is a bit higher; noise accounts for the rest)")


# ------------------------------------------------------------
# 11. COMPARE BIAS AND VARIANCE ACROSS DEGREES
# ------------------------------------------------------------

# Now repeat the bootstrap for different degrees to SEE the
# trade-off: low degree -> high bias; high degree -> high variance.

degrees_to_check = [1, 2, 3, 5, 10, 15]
bias_list = []
variance_list = []

for degree in degrees_to_check:

    bootstrap_predictions = []

    for bootstrap_number in range(n_bootstraps):

        sample_indices = random_state.integers(0, len(X_train), len(X_train))

        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]

        poly, scaler, lin_reg = fit_polynomial_model(
            degree, X_sample, y_sample
        )

        bootstrap_predictions.append(
            predict_with_parts(poly, scaler, lin_reg, X_test)
        )

    bootstrap_predictions = np.array(bootstrap_predictions)
    average_predictions = bootstrap_predictions.mean(axis=0)

    bias_list.append(
        np.mean((average_predictions - y_test_values) ** 2)
    )
    variance_list.append(
        np.mean(np.var(bootstrap_predictions, axis=0))
    )

plt.figure(figsize=(9, 5))
plt.plot(degrees_to_check, bias_list, marker="o", label="Bias Squared")
plt.plot(degrees_to_check, variance_list, marker="s", label="Variance")
plt.xlabel("Polynomial Degree")
plt.ylabel("Estimated Error Component")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid()
plt.show()


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n=====================================")
print("POLYNOMIAL REGRESSION EXPERIMENT DONE")
print("=====================================")

print("Data points:     ", len(df))
print("Best degree:     ", best_degree)
print("Best CV RMSE:    ", round(cv_errors[best_index], 4))
print("Test RMSE:       ", round(final_rmse, 4))
print("Test R2:         ", round(final_r2, 4))
print("Bias^2 estimate: ", round(bias_squared, 4))
print("Variance estimate:", round(variance, 4))

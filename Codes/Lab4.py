# ============================================================
# LAB 4: POLYNOMIAL REGRESSION AND BIAS-VARIANCE TRADE-OFF
# Dataset: Auto MPG (OpenML) - needs internet
#
# Big picture: a straight line cannot fit curved data.
# We give the model powers of the input (x, x^2, x^3, ...)
# and study what happens as the highest power (DEGREE) grows:
#   - low degree  -> too simple -> UNDERFITTING (high bias)
#   - high degree -> too flexible -> OVERFITTING (high variance)
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
# STEP 1: LOAD AND CLEAN THE DATA
# ------------------------------------------------------------

data = fetch_openml(name="autoMpg", version=1, as_frame=True)
df = data.frame.copy()

# On OpenML the MPG column is stored under the name "class".
df = df.rename(columns={"class": "mpg"})

# The horsepower column contains a few "?" marks.
# to_numeric(..., errors="coerce") replaces them with NaN,
# then we simply throw those rows away.
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower", "mpg"])

# Input: horsepower (a curved relationship with MPG).
# Output: MPG (miles per gallon).
X = df[["horsepower"]]
y = df["mpg"]

print("Clean data shape:", df.shape)


# ------------------------------------------------------------
# STEP 2: LOOK AT THE RAW DATA
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.5)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Horsepower vs MPG - clearly a curve, not a line")
plt.show()


# ------------------------------------------------------------
# STEP 3: SPLIT INTO TRAIN AND TEST SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# STEP 4: TRAIN A POLYNOMIAL MODEL FOR EVERY DEGREE (1..15)
# ------------------------------------------------------------

# For each degree we do three things:
#   a) PolynomialFeatures makes x, x^2, ..., x^degree columns
#   b) StandardScaler rescales them (x^15 is a huge number,
#      without scaling the math breaks down)
#   c) LinearRegression finds the best equation
#
# We store the 3 fitted parts in a list so we can reuse them
# later for plotting and the bootstrap analysis.

degrees = list(range(1, 16))
fitted_models = []    # one entry per degree = (poly, scaler, reg)

train_errors = []     # error on the data the model was trained on
test_errors = []      # error on unseen test data
cv_errors = []        # error estimated by 5-fold cross-validation

for degree in degrees:

    # a) create power columns
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    # b) rescale the new columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)

    # c) fit the equation on the rescaled columns
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)

    fitted_models.append([poly, scaler, reg])

    # ---- training error (same data as training -> optimistic) ----
    train_pred = reg.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    # ---- test error (unseen data -> honest) ----
    X_test_poly = poly.transform(X_test)
    X_test_scaled = scaler.transform(X_test_poly)
    test_pred = reg.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

    print(f"Degree {degree:>2}: train RMSE = {round(train_rmse, 3)},",
          f" test RMSE = {round(test_rmse, 3)}")


# ------------------------------------------------------------
# STEP 5: CROSS-VALIDATION - A FAIRER WAY TO COMPARE DEGREES
# ------------------------------------------------------------

# A single test set can be "lucky" or "unlucky". Cross-
# validation trains 5 times on 4/5 of the training data and
# tests on the held-out 1/5, then averages.

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for degree in degrees:

    fold_errors = []

    for train_part, val_part in kfold.split(X_train):

        X_part = X_train.iloc[train_part]   # 4/5 of training data
        y_part = y_train.iloc[train_part]
        X_val = X_train.iloc[val_part]      # held-out 1/5
        y_val = y_train.iloc[val_part]

        # fit on the 4/5 part only
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_part_poly = poly.fit_transform(X_part)

        scaler = StandardScaler()
        X_part_scaled = scaler.fit_transform(X_part_poly)

        reg = LinearRegression()
        reg.fit(X_part_scaled, y_part)

        # score on the held-out part
        X_val_poly = poly.transform(X_val)
        X_val_scaled = scaler.transform(X_val_poly)
        val_pred = reg.predict(X_val_scaled)
        fold_errors.append(np.sqrt(mean_squared_error(y_val, val_pred)))

    cv_errors.append(np.mean(fold_errors))


# ------------------------------------------------------------
# STEP 6: PLOT ALL THREE ERRORS TOGETHER
# ------------------------------------------------------------

# How to read the plot:
#   - training error keeps dropping as degree rises
#   - but test/CV error rises again after the sweet spot
#     -> that rise is OVERFITTING

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, marker="o", label="Training RMSE")
plt.plot(degrees, test_errors, marker="o", label="Testing RMSE")
plt.plot(degrees, cv_errors, marker="s", label="5-Fold CV RMSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Model Complexity vs Error (the U-shape = bias-variance)")
plt.legend()
plt.grid()
plt.show()


# ------------------------------------------------------------
# STEP 7: PICK THE BEST DEGREE AND EVALUATE IT ON THE TEST SET
# ------------------------------------------------------------

best_index = int(np.argmin(cv_errors))
best_degree = degrees[best_index]
poly, scaler, reg = fitted_models[best_index]

X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)
final_pred = reg.predict(X_test_scaled)

print("\nBest degree by cross-validation:", best_degree)
print("Test RMSE :", round(np.sqrt(mean_squared_error(y_test, final_pred)), 4))
print("Test R2   :", round(r2_score(y_test, final_pred), 4))


# ------------------------------------------------------------
# STEP 8: DRAW FITTED CURVES FOR A FEW DEGREES
# ------------------------------------------------------------

# We reuse the already-fitted models and just ask them to
# predict on a smooth grid of horsepower values.

hp_grid = pd.DataFrame({
    "horsepower": np.linspace(
        X["horsepower"].min(), X["horsepower"].max(), 300)
})

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.4, label="Actual data")

for degree in [1, 2, 3, 6, 15]:
    poly, scaler, reg = fitted_models[degree - 1]
    curve = reg.predict(scaler.transform(poly.transform(hp_grid)))
    plt.plot(hp_grid, curve, label=f"Degree {degree}")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Fitted Curves - degree 15 wiggles wildly (overfitting)")
plt.legend()
plt.show()


# ------------------------------------------------------------
# STEP 9: BOOTSTRAP - MEASURE BIAS AND VARIANCE
# ------------------------------------------------------------

# Idea: pretend we have many different training sets by
# sampling the training data WITH REPLACEMENT 30 times.
#   - BIAS squared  = (average prediction - truth)^2
#     -> systematic error of the model family
#   - VARIANCE = how much the predictions move around
#     between those samples -> sensitivity to data

n_bootstraps = 30
random_state = np.random.default_rng(42)
y_test_values = y_test.to_numpy()

degrees_to_check = [1, 2, 3, 6, 10, 15]
bias_results = []
variance_results = []

for degree in degrees_to_check:

    all_predictions = np.zeros((n_bootstraps, len(X_test)))

    for b in range(n_bootstraps):

        sample_indices = random_state.integers(0, len(X_train), len(X_train))

        # fit the SAME degree on this bootstrap sample
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        scaler = StandardScaler()
        reg = LinearRegression()

        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]

        poly.fit(X_sample)
        X_sample_scaled = scaler.fit_transform(poly.transform(X_sample))
        reg.fit(X_sample_scaled, y_sample)

        X_test_poly = poly.transform(X_test)
        X_test_scaled = scaler.transform(X_test_poly)
        all_predictions[b] = reg.predict(X_test_scaled)

    average_prediction = all_predictions.mean(axis=0)

    bias_squared = np.mean((average_prediction - y_test_values) ** 2)
    variance = np.mean(np.var(all_predictions, axis=0))

    bias_results.append(bias_squared)
    variance_results.append(variance)

    print(f"Degree {degree:>2}: bias^2 = {round(bias_squared, 3)},",
          f" variance = {round(variance, 3)}")


# ------------------------------------------------------------
# STEP 10: PLOT THE TRADE-OFF
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.plot(degrees_to_check, bias_results, marker="o", label="Bias squared")
plt.plot(degrees_to_check, variance_results, marker="s", label="Variance")
plt.xlabel("Polynomial Degree")
plt.ylabel("Error component")
plt.title("Bias falls, Variance rises as complexity grows")
plt.legend()
plt.grid()
plt.show()

print("\nTotal error = bias^2 + variance + irreducible noise.")
print("The best degree sits where their SUM is smallest.")

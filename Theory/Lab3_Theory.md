# Lab 3 Theory — Multiple Linear Regression

## 1. Simple vs Multiple Linear Regression

- **Simple:** one predictor — \( \hat y = b_0 + b_1 x \)
- **Multiple:** many predictors — the model here:

\( \hat y = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n \)

Target: predict house **median value** from 8 predictors (MedInc, HouseAge, AveRooms, AveOccup, latitude, longitude, …) in the California Housing dataset.

Each coefficient \( b_i \) is the expected change in \( y \) for a one-unit change in \( x_i \), **holding all other predictors constant**.

## 2. How the Model is Fitted — Ordinary Least Squares (OLS)

OLS chooses coefficients that **minimize the sum of squared residuals**:

\( \min_{b} \sum_{i=1}^{n} (y_i - \hat y_i)^2 \)

Closed-form solution: \( \mathbf{b} = (X^T X)^{-1} X^T \mathbf{y} \)

- Pros: exact, fast, no hyperparameters.
- Cons: requires \( X^T X \) to be invertible → suffers badly when features are collinear (addressed in Labs 4 & 5).

## 3. Assumptions of Linear Regression

1. **Linearity** — relation between predictors and target is linear.
2. **Independence** — errors are not correlated with each other.
3. **Homoscedasticity** — error variance is constant across all levels of \( x \).
4. **Normality of errors** — residuals ≈ Gaussian (needed for inference/confidence intervals).
5. **No perfect multicollinearity** — predictors must not be exact (or near-exact) linear combinations of each other.

**Residual plots** in the lab check 3–5: residuals scattered randomly around 0 (good) vs. curved (non-linearity), or a funnel shape (heteroscedasticity).

## 4. Evaluation Metrics

Let \( n \) = samples, \( y \) = actual, \( \hat y \) = predicted, \( \bar y \) = mean.

- **MSE:** \( \dfrac{1}{n}\sum (y_i - \hat y_i)^2 \) — squared units; punishes large errors.
- **RMSE:** \( \sqrt{MSE} \) — same units as target; interpretable.
- **MAE:** \( \dfrac{1}{n}\sum |y_i - \hat y_i| \) — robust to outliers.
- **R² (coefficient of determination):**

  \( R^2 = 1 - \dfrac{SS_{res}}{SS_{tot}} = 1 - \dfrac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2} \)

  Fraction of variance in \( y \) explained by the model. 1 = perfect, 0 = no better than predicting the mean, negative = worse than the mean.
- **Adjusted R²** penalizes adding useless predictors:

  \( 1 - (1-R^2)\dfrac{n-1}{n-p-1} \) (p = number of predictors).

## 5. Key Visualizations

1. **Scatter plots / pair plots** — check linearity and spot outliers before fitting.
2. **Correlation heatmap** — predictor–target correlation (e.g., MedInc vs MedHouseVal) and inter-predictor correlation (latitude/longitude cluster together geographically).
3. **Actual vs Predicted scatter** — points hugging the y = x line → accurate model.
4. **Residual plot** — residuals vs predicted values; a random band around 0 validates assumptions, patterns expose model misspecification.

## 6. Workflow Recap

Load → explore (head, info, describe) → visualize → separate X/y → train_test_split → fit OLS → predict on test set → compute MSE/R² → inspect coefficients/intercept → diagnostic plots.

**Typical outcome:** R² ≈ 0.57–0.60 on the 20% test split — a strong signal but clearly imperfect, motivating polynomial features (Lab 4) and regularized models (Lab 5) and tree ensembles (Lab 10).

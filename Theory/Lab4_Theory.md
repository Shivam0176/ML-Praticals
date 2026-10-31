# Lab 4 Theory — Polynomial Regression, Model Complexity and the Bias-Variance Trade-off

## 1. Polynomial Regression

Linear regression is limited to straight-line/planar relationships. Polynomial regression keeps the model *linear in the parameters* but **transforms the features** into powers:

Given one feature \( x \), degree-d terms become \( x, x^2, x^3, \dots, x^d \):

\( \hat y = b_0 + b_1 x + b_2 x^2 + \dots + b_d x^d \)

In sklearn this is done with `PolynomialFeatures(degree=d)` inside a pipeline (`make_pipeline`), so scaling and fitting always run in the correct order.

- **Degree 1** = ordinary linear regression.
- Higher degrees produce curves — but also explode feature count (and for multiple features, include interaction terms like \( x_1x_2 \)).

## 2. Model Complexity: Underfitting vs Overfitting

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Degree | too low | optimal | too high |
| Training error | high | low | very low |
| Test error | high | low | rises again |
| Cause | too simple / high bias | — | too flexible / high variance |

The lab plots **train vs test MSE across degrees 1–15**: both errors drop at first; after the sweet spot the test error climbs while train error keeps falling — the classic overfitting signature (e.g., degree-15 curve wiggling to pass through training points).

## 3. Bias-Variance Trade-off

Decomposition of expected test error:

\( E[(y - \hat f(x))^2] = \underbrace{\text{Bias}^2}_{\text{systematic error}} + \underbrace{\text{Variance}}_{\text{sensitivity to data}} + \underbrace{\sigma^2}_{\text{irreducible noise}} \)

- **Bias** — error from wrong simplifying assumptions (low degree → can't capture the curve).
- **Variance** — error from over-sensitivity to the particular training sample (high degree → model chases noise).
- **Increasing capacity → bias ↓, variance ↑.** The goal is minimizing total error, not minimizing either term.

## 4. K-Fold Cross-Validation

One train/test split is noisy. **K-fold CV** splits the data into K parts, trains on K−1 and validates on the held-out part, rotating K times; the K scores are averaged.

- Every point is used for validation exactly once.
- Mean CV error per degree → robust choice of the **optimal degree**.
- `KFold` / `cross_val_score(model, X, y, cv=k, scoring="neg_mean_squared_error")` in sklearn.

## 5. Bootstrap Bias-Variance Estimation

The bootstrap approximates *"different training sets"* by resampling the data **with replacement** many times (e.g., 30 samples of n points):

1. For each bootstrap sample of degree d, fit a polynomial.
2. Predict on a fixed set of test points.
3. Across samples compute:
   - **Bias²** — (mean prediction − true value)² — systematic miss.
   - **Variance** — spread of predictions across bootstrap models.
4. Plot bias² and variance vs degree: bias falls with degree, variance rises — directly visualizing the trade-off.

## 6. Practical Points

- Auto MPG needs cleaning (`horsepower` has `"?"` entries → `pd.to_numeric(errors="coerce")` then impute/drop NaN).
- **Scaling matters:** \( x^{15} \) has huge magnitude; standard scaling keeps the optimization stable.
- Use the selected degree for a final train/test evaluation and report R² and RMSE.

## 7. Key Takeaways

- Polynomial regression = linear regression on engineered features.
- Complexity is a hyperparameter: choose it with cross-validation, never by test-set peeking.
- Train vs test error curves + bootstrap decomposition are the standard tools to *see* bias and variance.

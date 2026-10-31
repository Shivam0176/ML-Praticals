# Lab 5 Theory — Ridge, Lasso and Elastic Net (Regularized Regression)

## 1. The Problem: Overfitting and Multicollinearity

On the Red Wine Quality dataset, the physicochemical features are **correlated with each other** (e.g., fixed acidity ↔ pH ↔ citric acid; free ↔ total sulfur dioxide). With correlated predictors, OLS:

- produces **unstable, high-variance coefficients** (sign flips, huge magnitudes that cancel each other);
- overfits — low train error but poor generalization.

**Regularization** fixes this by adding a penalty on coefficient size to the loss function, shrinking coefficients toward zero.

## 2. Diagnosing Multicollinearity

### Correlation matrix
Pearson r between feature pairs shows which features move together.

### Variance Inflation Factor (VIF)
Regresses each feature on all others:

\( VIF_j = \dfrac{1}{1 - R_j^2} \)

where \( R_j^2 \) is from predicting feature *j* using the remaining features.

- VIF = 1 → no collinearity; VIF > 5 (commonly) → problematic; VIF > 10 → severe.

### Condition Number
Ratio of largest to smallest eigenvalue (or singular values) of the design matrix \( X \):

\( \kappa = \dfrac{\lambda_{max}}{\lambda_{min}} \)

Large κ (e.g., > 30) → \( X^T X \) is ill-conditioned → OLS solutions sensitive to tiny data changes.

## 3. The Three Regularized Models

All minimize residual error **plus** a penalty term (λ/alpha ≥ 0 controls strength).

| Model | Penalty | Effect |
|-------|---------|--------|
| **Ridge (L2)** | \( \alpha \sum b_j^2 \) | Shrinks coefficients smoothly toward 0, **never exactly 0**; handles correlated features by distributing weight |
| **Lasso (L1)** | \( \alpha \sum \|b_j\| \) | Can push coefficients **exactly to 0 → automatic feature selection** |
| **Elastic Net** | \( \alpha\big[r\sum\|b_j\| + \frac{1-r}{2}\sum b_j^2\big] \) | Blend: `l1_ratio` r mixes L1 and L2 → grouped selection with stability |

- With **alpha = 0** both reduce to plain linear regression.
- With **alpha → ∞** all coefficients → 0.
- Too small α ≈ OLS (overfits); too large α ≈ underfits. The lab sweeps a range of alphas and compares test R² to pick a good value.

## 4. Why Scaling is Mandatory

The penalty treats all coefficients equally, so features must be on the same scale (`StandardScaler` in a pipeline). Otherwise features measured in larger units get unfairly penalized and scale dominates signal.

## 5. Interpretive Plots

1. **Model comparison bar chart** — test MSE/R² for Linear vs Ridge vs Lasso vs Elastic Net across alphas: regularized models win when multicollinearity exists.
2. **Coefficient comparison** — OLS coefficients are large and erratic; Ridge compresses them all; Lasso zeroes out the least useful ones.
3. **Coefficient paths** — coefficient value vs log(alpha) as α grows:
   - Ridge paths shrink smoothly and asymptotically toward 0.
   - Lasso paths hit exactly 0 one by one — showing the order in which features are eliminated.

## 6. Key Takeaways

- Ridge = shrinkage; Lasso = shrinkage + selection; Elastic Net = compromise for correlated groups.
- Always scale before penalized regression.
- Diagnose collinearity with correlation, VIF and condition number before trusting plain OLS.
- Tuning α via validation is what makes regularization a *bias-variance knob*: bigger α → more bias, less variance.

# Lab 10 Theory — Ensemble Learning: Random Forest vs Linear Regression

## 1. What is an Ensemble?

An **ensemble** combines several models so their combined prediction is stronger than any single one. Two families:

| Family | Idea | Example |
|--------|------|---------|
| **Bagging (Bootstrap Aggregating)** | train many models in parallel on resampled data, average/vote | Random Forest |
| **Boosting** | train models sequentially, each fixing predecessors' errors | AdaBoost, Gradient Boosting, XGBoost |

Ensemble principle: if individual models make **uncorrelated errors**, averaging cancels error → accuracy ↑, variance ↓.

## 2. Bagging

From the original dataset of n points, draw **B bootstrap samples** (n points each, with replacement). Train one weak learner per sample and aggregate:

- Regression → average of the B predictions.
- Classification → majority vote.

Because each learner sees only a noisy replica of the data, each is unstable — but averaging them averages out that instability.

## 3. Random Forest (sklearn `RandomForestRegressor`)

Random Forest = **bagging + random feature selection**, on decision trees:

1. **Bootstrap sampling** of rows for every tree (trees not sampled = *out-of-bag*, usable for free validation).
2. **Random subspace** — at each split, only a random subset of features (default √p for classification, p/3 for regression) is eligible. This **decorrelates the trees** — the crucial difference from plain bagging.
3. Each tree is grown deep (low bias, high variance); the forest average brings variance back down.

**Why it beats one tree:** a single deep tree overfits; Averaging B deep trees on B samples cuts variance roughly by the correlation factor: \( \rho\sigma^2 + \dfrac{1-\rho}{B}\sigma^2 \) — the lower the tree correlation ρ, the closer to σ²/B.

## 4. Decision Trees in 30 Seconds

A tree recursively splits the data choosing the split that most reduces impurity. For regression the impurity measure is variance (MSE criterion): the split minimizing \( \sum_i (y_i - \bar y_{leaf})^2 \). Leaves return the mean target of their sample. Trees capture nonlinearity and interactions automatically, need no scaling, but overfit alone.

## 5. Why Compare with Linear Regression?

- Linear regression: high bias, low variance — wins when the true relation is linear.
- California housing price is **nonlinear** (e.g., price saturates at high MedInc, location effects are curved), so a flexible ensemble should generalize better.
- The experiment demonstrates the **model-family vs data-structure fit**: same split, same metrics, very different scores.

## 6. Regression Metrics Compared

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | \( \frac{1}{n}\sum (y_i - \hat y_i)^2 \) | squared error, punishes big errors |
| **RMSE** | \( \sqrt{MSE} \) | target units |
| **MAE** | \( \frac{1}{n}\sum \|y_i - \hat y_i\| \) | robust to outliers |
| **R²** | \( 1 - \frac{SS_{res}}{SS_{tot}} \) | fraction of variance explained |

Typical outcome: Linear R² ≈ 0.57–0.60; Random Forest R² ≈ 0.80–0.81 — a large improvement, along with visibly tighter actual-vs-predicted points along the y = x line.

## 7. Feature Importance

Random Forests measure importance as the **total impurity (variance) reduction** a feature contributes across all trees, normalized to sum 1. The lab's bar chart typically ranks:

- `MedInc` ≫ others (income is the dominant price driver),
- followed by `AveOccup`, `Latitude`, `Longitude` (geography), etc.

Limitations: importances are biased toward high-cardinality/continuous features and only measure training-time usefulness, not causation.

## 8. Key Takeaways

- Ensembles reduce variance by combining many weakly-correlated learners.
- Random Forest = bootstrap + per-split feature randomness + tree averaging → strong, low-tuning baseline.
- Ensemble ≠ always better: it sacrifices interpretability and speed, and on truly linear data the linear model is closer to the truth with far less complexity.

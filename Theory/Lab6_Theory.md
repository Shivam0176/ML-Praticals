# Lab 6 Theory — Logistic Regression (Binary Classification)

## 1. Why Not Linear Regression for Classification?

Linear regression outputs unbounded values — probabilities must lie in [0, 1]. Pushing class labels through a line also gives nonsense predictions. **Logistic regression** models the *probability* of the positive class instead.

## 2. The Model

### Sigmoid (logistic) function

\( \sigma(z) = \dfrac{1}{1 + e^{-z}}, \quad z = b_0 + b_1x_1 + \dots + b_nx_n \)

- σ(z) → 1 as z → +∞, → 0 as z → −∞; σ(0) = 0.5.
- Output \( p = \sigma(z) \) = P(class = 1 | x).

### Odds, Logit, and the Decision Boundary

- **Odds:** \( \dfrac{p}{1-p} \) — ratio of probability of occurrence vs not.
- **Log-odds (logit):** \( \ln\frac{p}{1-p} = z \) — linear in features (why it's still "regression").
- **Decision rule:** predict class 1 if \( p \geq \text{threshold} \) (default 0.5, i.e., z = 0). The boundary is where \( \sigma(z) = \) threshold — a hyperplane in feature space.
- Coefficients: \( e^{b_j} \) = multiplicative change in odds per unit of \( x_j \).

## 3. Training — Log Loss (Binary Cross-Entropy)

Linear regression's MSE is replaced by:

\( L = -\dfrac{1}{n}\sum \big[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \big] \)

- Confidently wrong → huge loss (log of small number).
- Minimized by gradient descent on the standardized features (`StandardScaler` in a pipeline — logistic regression is a gradient-trained model and sensitive to scale).

## 4. Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN (Type II error) |
| **Actual Negative** | FP (Type I error) | TN |

Computed manually in the lab step-by-step, then verified with `confusion_matrix` from sklearn.

## 5. Performance Metrics

- **Accuracy** \( = \dfrac{TP+TN}{TP+TN+FP+FN} \) — misleading on imbalanced data.
- **Precision** \( = \dfrac{TP}{TP+FP} \) — of predicted positives, how many are real? (Costly false alarms → prioritize.)
- **Recall / Sensitivity / TPR** \( = \dfrac{TP}{TP+FN} \) — of real positives, how many caught? (Costly misses, e.g., cancer → prioritize.)
- **Specificity / TNR** \( = \dfrac{TN}{TN+FP} \)
- **F1-score** \( = \dfrac{2 \cdot Precision \cdot Recall}{Precision + Recall} \) — harmonic mean; punishes imbalance between the two.
- **Classification report** — sklearn prints all of these per class plus support.

## 6. Stratified K-Fold Cross-Validation

Plain K-fold can make a fold contain too few positives. **StratifiedKFold preserves the class ratio** in every fold — essential for imbalanced medical-style data. The lab reports accuracy per fold and the mean ± std, showing estimate stability.

## 7. Decision Threshold Analysis

The trained model gives probabilities; the threshold converts them to labels. Varying the threshold (e.g., 0.1 → 0.9):

- **Low threshold** → more positives predicted → **recall ↑, precision ↓**
- **High threshold** → stricter → **precision ↑, recall ↓**

Plotting **precision and recall vs threshold** lets you pick an operating point suited to the application (medical screening favors recall; spam filters favor precision). This mirrors the ROC/PR-curve view of classifier performance.

## 8. Workflow Recap

Load → split (stratify by class) → pipeline(scale + logistic) → fit → predict & predict_proba → manual confusion matrix → verify vs sklearn → metrics + classification report → stratified 5-fold CV → threshold sweep with precision/recall plot.

# ============================================================
# LAB 6: LOGISTIC REGRESSION
# Binary Classification using Breast Cancer Dataset
#
# Goal: predict whether a tumor is malignant (1) or benign (0),
# then measure quality honestly with a confusion matrix and
# precision / recall / F1 - not just accuracy.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

# In the raw dataset: 0 = malignant, 1 = benign.
# We flip it so 1 = malignant (the dangerous class we hunt for).
y = pd.Series((data.target == 0).astype(int), name="malignant")

print("First 5 Rows:")
print(X.head())

print("\nDataset Shape:", X.shape)

print("\nClass Distribution:")
print("Malignant:", (y == 1).sum())
print("Benign:   ", (y == 0).sum())


# ------------------------------------------------------------
# 2. TRAIN-TEST SPLIT
# ------------------------------------------------------------

# stratify=y keeps the same malignant/benign ratio in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 3. SCALE THE FEATURES
# ------------------------------------------------------------

# Features have very different ranges (0.05 up to 4000).
# Scaling puts them all on mean 0 / std 1, which logistic
# regression needs for stable fitting.
# Fit the scaler on TRAIN only, apply the same stats to TEST.

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


# ------------------------------------------------------------
# 4. TRAIN THE LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

# No pipeline needed - we simply scale, then fit by hand.
# The model outputs a probability via the sigmoid function:
#   probability = 1 / (1 + e^-(b0 + b1*x1 + ... + bn*xn))
# Predicted class = 1 (malignant) if probability >= 0.5.

model = LogisticRegression(max_iter=5000)

model.fit(X_train_scaled, y_train)

print("\nModel Training Completed!")


# ------------------------------------------------------------
# 5. PREDICT ON THE TEST SET
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)


# ------------------------------------------------------------
# 6. CONFUSION MATRIX (MANUALLY FIRST)
# ------------------------------------------------------------

#              predicted: Pos   predicted: Neg
# actual Pos     TP               FN
# actual Neg     FP               TN

tp = np.sum((y_test == 1) & (y_pred == 1))   # caught cancer
tn = np.sum((y_test == 0) & (y_pred == 0))   # correctly healthy
fp = np.sum((y_test == 0) & (y_pred == 1))   # false alarm
fn = np.sum((y_test == 1) & (y_pred == 0))   # missed cancer - worst error

total = tp + tn + fp + fn

print("\n========== MANUAL CONFUSION MATRIX ==========")
print("True Positive  (TP):", tp)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Negative  (TN):", tn)
print("Total:", total)


# ------------------------------------------------------------
# 7. CALCULATE THE METRICS BY HAND
# ------------------------------------------------------------

accuracy = (tp + tn) / total

if (tp + fp) > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if (tp + fn) > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if (tn + fp) > 0:
    specificity = tn / (tn + fp)
else:
    specificity = 0

if (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

print("\n========== PERFORMANCE METRICS (MANUAL) ==========")
print("Accuracy:   ", round(accuracy, 4))
print("Precision:  ", round(precision, 4))
print("Recall:     ", round(recall, 4))
print("Specificity:", round(specificity, 4))
print("F1 Score:   ", round(f1, 4))


# ------------------------------------------------------------
# 8. VERIFY WITH SCIKIT-LEARN
# ------------------------------------------------------------

verification_cm = confusion_matrix(y_test, y_pred)

print("\nScikit-learn Confusion Matrix:      [[TN FP] [FN TP]]")
print(verification_cm)

print("\nScikit-learn Metrics:")
print("Accuracy:  ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall:    ", recall_score(y_test, y_pred))
print("F1 Score:  ", f1_score(y_test, y_pred))


# ------------------------------------------------------------
# 9. VISUALIZE THE CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
sns.heatmap(
    verification_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ------------------------------------------------------------
# 10. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    )
)


# ------------------------------------------------------------
# 11. STRATIFIED 5-FOLD CROSS VALIDATION (BY HAND)
# ------------------------------------------------------------

# A single train/test score is a bit lucky/unlucky.
# Cross-validation averages 5 scores, so the result is stable.
# "Stratified" = each fold has the same class ratio as the data.

cv_splitter = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_scores = []

for fold_number, (train_idx, val_idx) in enumerate(
        cv_splitter.split(X, y), start=1):

    X_cv_train = X.iloc[train_idx]
    y_cv_train = y.iloc[train_idx]
    X_cv_val = X.iloc[val_idx]
    y_cv_val = y.iloc[val_idx]

    # Re-scale inside each fold (a small but honest habit)
    fold_scaler = StandardScaler()
    X_cv_train_scaled = fold_scaler.fit_transform(X_cv_train)
    X_cv_val_scaled = fold_scaler.transform(X_cv_val)

    fold_model = LogisticRegression(max_iter=5000)
    fold_model.fit(X_cv_train_scaled, y_cv_train)

    fold_predictions = fold_model.predict(X_cv_val_scaled)
    fold_accuracy = accuracy_score(y_cv_val, fold_predictions)

    fold_scores.append(fold_accuracy)

    print(f"Fold {fold_number} accuracy: {round(fold_accuracy, 4)}")

print("\nMean CV Accuracy:  ", round(np.mean(fold_scores), 4))
print("Std of CV Accuracy:", round(np.std(fold_scores), 4))


# ------------------------------------------------------------
# 12. DECISION THRESHOLD ANALYSIS
# ------------------------------------------------------------

# Normally the boundary is probability >= 0.5. But we can move it:
#   lower threshold -> predict MORE maligants -> recall goes up,
#                      some false alarms (precision down)
#   higher threshold -> predictions must be very sure -> precision
#                       goes up, but we may miss cancers (recall down)

probabilities = model.predict_proba(X_test_scaled)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

precision_values = []
recall_values = []

for threshold in thresholds:

    predictions_at_threshold = (probabilities >= threshold).astype(int)

    precision_values.append(
        precision_score(
            y_test, predictions_at_threshold, zero_division=0
        )
    )
    recall_values.append(
        recall_score(
            y_test, predictions_at_threshold, zero_division=0
        )
    )

threshold_table = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision_values,
    "Recall": recall_values
})

print("\n========== THRESHOLD ANALYSIS ==========")
print(threshold_table.round(4))


# ------------------------------------------------------------
# 13. PLOT PRECISION AND RECALL VS THRESHOLD
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision_values, marker="o", label="Precision")
plt.plot(thresholds, recall_values, marker="o", label="Recall")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall at Different Thresholds")
plt.legend()
plt.grid()
plt.show()


# ------------------------------------------------------------
# 14. FINAL SUMMARY
# ------------------------------------------------------------

print("\n====================================")
print("LOGISTIC REGRESSION COMPLETED")
print("====================================")

print("Test accuracy:      ", round(accuracy, 4))
print("Mean CV accuracy:   ", round(np.mean(fold_scores), 4))
print("Note: recall matters most here - a missed cancer (FN)")
print("is far more costly than a false alarm (FP).")

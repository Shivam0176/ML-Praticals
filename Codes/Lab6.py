# ============================================================
# LAB 6: LOGISTIC REGRESSION (BINARY CLASSIFICATION)
# Dataset: Breast Cancer (bundled with sklearn, no internet)
#
# Big picture: logistic regression predicts a PROBABILITY
# between 0 and 1 using the sigmoid function:
#     p = 1 / (1 + e^-(b0 + b1*x1 + ... + bn*xn))
# Then we judge it not just by accuracy, but by precision,
# recall and F1 - which matter a lot in medical data.
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
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)


# ------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Raw coding: 0 = malignant, 1 = benign.
# We flip it so 1 = malignant - the "positive" class we care
# most about catching (missing cancer = worst mistake).
y = pd.Series((data.target == 0).astype(int), name="malignant")

print("Data shape:", X.shape)
print("Malignant tumors:", (y == 1).sum())
print("Benign tumors:   ", (y == 0).sum())


# ------------------------------------------------------------
# STEP 2: SPLIT INTO TRAIN AND TEST
# ------------------------------------------------------------

# stratify=y keeps the same sick/healthy ratio in both parts.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ------------------------------------------------------------
# STEP 3: SCALE THE FEATURES
# ------------------------------------------------------------

# Features range from about 0.05 up to 4000. Logistic
# regression finds coefficients by gradients, which behave
# badly with huge scale differences - so we standardize.
# Scaler learns mean/std from TRAIN only, reuses on TEST.

feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)


# ------------------------------------------------------------
# STEP 4: TRAIN AND PREDICT
# ------------------------------------------------------------

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
print("\nModel trained and predictions made!")


# ------------------------------------------------------------
# STEP 5: CONFUSION MATRIX - COUNT EVERY OUTCOME BY HAND
# ------------------------------------------------------------

# Corner cases when the truth is malignant (1) and benign (0):
#   TP: said malignant, was malignant  (good catch)
#   FN: said benign,    was malignant  (MISSED cancer - worst)
#   FP: said malignant, was benign     (false alarm)
#   TN: said benign,    was benign     (correct)

tp = np.sum((y_test == 1) & (predictions == 1))
tn = np.sum((y_test == 0) & (predictions == 0))
fp = np.sum((y_test == 0) & (predictions == 1))
fn = np.sum((y_test == 1) & (predictions == 0))

total = tp + tn + fp + fn

print("\n=== MANUAL CONFUSION MATRIX ===")
print("TP (caught cancer):      ", tp)
print("FN (missed cancer):      ", fn)
print("FP (false alarm):        ", fp)
print("TN (correctly healthy):  ", tn)


# ------------------------------------------------------------
# STEP 6: COMPUTE ALL METRICS FROM SIMPLE FORMULAS
# ------------------------------------------------------------

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0)

print("\n=== METRICS BY HAND ===")
print("Accuracy   :", round(accuracy, 4),
      "- share of all correct answers")
print("Precision  :", round(precision, 4),
      "- of predictions made, how many were right?")
print("Recall     :", round(recall, 4),
      "- of all maligants, how many we caught?")
print("Specificity:", round(specificity, 4),
      "- of healthy, how many we left alone?")
print("F1 Score   :", round(f1, 4),
      "- balance of precision and recall")


# ------------------------------------------------------------
# STEP 7: VERIFY THE METRICS ARE CORRECT WITH SKLEARN
# ------------------------------------------------------------

sklearn_cm = confusion_matrix(y_test, predictions)

print("\n=== SCIKIT-LEARN CHECKS ===")
print("Confusion matrix [[TN FP] [FN TP]]:")
print(sklearn_cm)
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall   :", recall_score(y_test, predictions))
print("F1 Score :", f1_score(y_test, predictions))


# ------------------------------------------------------------
# STEP 8: VISUALIZE THE CONFUSION MATRIX
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
sns.heatmap(sklearn_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, predictions,
                            target_names=["Benign", "Malignant"]))


# ------------------------------------------------------------
# STEP 9: STRATIFIED 5-FOLD CROSS VALIDATION BY HAND
# ------------------------------------------------------------

# One train/test score could be lucky. CV gives 5 scores:
# each fold trains on 4/5 of the data and tests on 1/5.
# "Stratified" = every fold keeps the class ratio.

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=42)

fold_accuracies = []

for fold_number, (train_idx, val_idx) in enumerate(
        cv_splitter.split(X, y), start=1):

    X_fold_train = X.iloc[train_idx]
    y_fold_train = y.iloc[train_idx]
    X_fold_val = X.iloc[val_idx]
    y_fold_val = y.iloc[val_idx]

    fold_scaler = StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = fold_scaler.transform(X_fold_val)

    fold_model = LogisticRegression()
    fold_model.fit(X_fold_train_scaled, y_fold_train)
    fold_predictions = fold_model.predict(X_fold_val_scaled)

    fold_accuracy = accuracy_score(y_fold_val, fold_predictions)
    fold_accuracies.append(fold_accuracy)
    print(f"Fold {fold_number} accuracy:", round(fold_accuracy, 4))

print("\nMean CV accuracy:   ", round(np.mean(fold_accuracies), 4))
print("Std of CV accuracy: ", round(np.std(fold_accuracies), 4))


# ------------------------------------------------------------
# STEP 10: WHY THE 0.5 THRESHOLD MATTERS - MOVE IT!
# ------------------------------------------------------------

# The model gives probabilities first, classes second.
# Lower threshold -> flag more tumors as malignant:
#   recall goes UP (fewer missed cancers), precision goes DOWN.
# Higher threshold: the opposite. For cancer detection we
# usually accept more false alarms to miss fewer tumors.

probabilities = model.predict_proba(X_test_scaled)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
precision_at = []
recall_at = []

for threshold in thresholds:

    moved_predictions = (probabilities >= threshold).astype(int)

    precision_at.append(precision_score(
        y_test, moved_predictions, zero_division=0))
    recall_at.append(recall_score(
        y_test, moved_predictions, zero_division=0))

threshold_table = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision_at,
    "Recall": recall_at,
})

print("\n=== THRESHOLD ANALYSIS ===")
print(threshold_table.round(4))

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision_at, marker="o", label="Precision")
plt.plot(thresholds, recall_at, marker="o", label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall for Different Thresholds")
plt.legend()
plt.grid()
plt.show()

print("\nLogistic regression lab complete!")
print("Accuracy:", round(accuracy, 4),
      "| CV accuracy:", round(np.mean(fold_accuracies), 4))

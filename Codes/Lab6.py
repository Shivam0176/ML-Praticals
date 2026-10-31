
# ============================================================
# LAB 6: LOGISTIC REGRESSION
# Binary Classification using Breast Cancer Dataset
# ============================================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 2. Load Dataset

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

# Original target: 0 = Malignant, 1 = Benign
# Convert to: 1 = Malignant, 0 = Benign
y = (data.target == 0).astype(int)

print("First 5 Rows:")
print(X.head())

print("\nDataset Shape:", X.shape)

print("\nClass Distribution:")
print(y.value_counts())

# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Feature Scaling and Model Creation

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

# 5. Train Model

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 6. Prediction

y_pred = model.predict(X_test)

# 7. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("\nTrue Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive:", tp)

# 8. Calculate Performance Metrics Manually

accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp) if (tp + fp) else 0

recall = tp / (tp + fn) if (tp + fn) else 0

specificity = tn / (tn + fp) if (tn + fp) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) else 0
)

print("\n========== PERFORMANCE METRICS ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Specificity:", round(specificity, 4))
print("F1 Score:", round(f1, 4))

# 9. Verify using Scikit-learn

print("\nScikit-learn Metrics:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# 10. Visualize Confusion Matrix

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
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

# 11. Classification Report

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    )
)

# 12. Stratified 5-Fold Cross Validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n5-Fold Cross Validation Scores:")
print(cv_scores)

print("Mean CV Accuracy:", round(cv_scores.mean(), 4))

# 13. Decision Threshold Analysis

probabilities = model.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

precision_values = []
recall_values = []

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    precision_values.append(
        precision_score(y_test, predictions, zero_division=0)
    )

    recall_values.append(
        recall_score(y_test, predictions, zero_division=0)
    )

threshold_df = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision_values,
    "Recall": recall_values
})

print("\nThreshold Analysis:")
print(threshold_df)

# 14. Plot Precision and Recall

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    precision_values,
    marker="o",
    label="Precision"
)

plt.plot(
    thresholds,
    recall_values,
    marker="o",
    label="Recall"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision vs Recall at Different Thresholds")
plt.legend()
plt.grid()
plt.show()

print("\nLogistic Regression Completed!")

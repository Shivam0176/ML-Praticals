# ============================================================
# LAB 1: FEATURE ENGINEERING AND DATA PREPROCESSING
# Dataset: Titanic Dataset (OpenML)
#
# Goal: turn a messy dataset into a clean, numeric,
# model-ready table. We do every step by hand so that
# the logic is easy to see and explain.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

print("Loading Titanic Dataset (needs internet)...")

titanic = fetch_openml(
    name="titanic",
    version=1,
    as_frame=True
)

df = titanic.frame.copy()

print("\nDataset Loaded!")
print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 2. EXPLORE THE DATASET
# ------------------------------------------------------------

print("\nDataset Shape (rows, columns):")
print(df.shape)

print("\nColumn Info:")
df.info()

print("\nMissing Values in Each Column:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. SELECT THE COLUMNS WE WANT TO USE
# ------------------------------------------------------------

# We keep a small, meaningful set of features.
numeric_columns = ["age", "sibsp", "parch", "fare"]
category_columns = ["pclass", "sex", "embarked"]

X = df[numeric_columns + category_columns].copy()
y = df["survived"].copy()

# Some rows have no recorded survival value - drop them.
has_target = y.notnull()
X = X[has_target]
y = y[has_target].astype(int)

print("\nTarget class counts (0 = died, 1 = survived):")
print(y.value_counts())


# ------------------------------------------------------------
# 4. VISUALIZE MISSING VALUES
# ------------------------------------------------------------

# Heatmap: yellow = missing, purple = present.
plt.figure(figsize=(10, 5))
sns.heatmap(X.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Titanic Dataset")
plt.show()


# ------------------------------------------------------------
# 5. SPLIT FIRST (BEFORE FILLING MISSING VALUES)
# ------------------------------------------------------------

# We split BEFORE looking at the data, so that the test
# data never influences the statistics we compute.
# (This avoids "data leakage".)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape: ", X_test.shape)


# ------------------------------------------------------------
# 6. FILL MISSING VALUES (IMPUTATION)
# ------------------------------------------------------------

# Rule used everywhere in ML:
#   - numeric columns  -> fill with the MEDIAN (robust to outliers)
#   - category columns -> fill with the MODE (most frequent value)
# The statistic must come from the TRAINING data only,
# and the same value is used to fill the test data.

for column in numeric_columns:

    train_median = X_train[column].median()

    X_train[column] = X_train[column].fillna(train_median)
    X_test[column] = X_test[column].fillna(train_median)

    print(f"Filled {column} with training median = {train_median}")

for column in category_columns:

    train_mode = X_train[column].mode()[0]  # most frequent value

    X_train[column] = X_train[column].fillna(train_mode)
    X_test[column] = X_test[column].fillna(train_mode)

    print(f"Filled {column} with training mode = '{train_mode}'")

print("\nMissing values after filling:")
print("Train:", X_train.isnull().sum().sum(),
      " Test:", X_test.isnull().sum().sum())


# ------------------------------------------------------------
# 7. CONVERT CATEGORIES TO NUMBERS (ONE-HOT ENCODING)
# ------------------------------------------------------------

# Models cannot work with text like "male" / "female".
# One-hot encoding makes one new 0/1 column per category.

X_train = pd.get_dummies(X_train, dtype=float)
X_test = pd.get_dummies(X_test, dtype=float)

# In case a rare category appears in train but not in test
# (or the other way around), we line up the columns:

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nColumns after one-hot encoding:")
print(list(X_train.columns))


# ------------------------------------------------------------
# 8. SCALE ALL FEATURES (STANDARDIZATION)
# ------------------------------------------------------------

# Standardization: each value becomes (value - mean) / std.
# After this every column has mean 0 and std 1,
# so no feature dominates just because of its units.

scaler = StandardScaler()

X_train_processed = scaler.fit_transform(X_train)   # learn mean/std from train...
X_test_processed = scaler.transform(X_test)         # ...apply the SAME stats to test

# Turn numpy arrays back into DataFrames to keep column names.
X_train_processed = pd.DataFrame(
    X_train_processed, columns=X_train.columns, index=X_train.index
)
X_test_processed = pd.DataFrame(
    X_test_processed, columns=X_train.columns, index=X_test.index
)

print("\nProcessed Training Data:")
print(X_train_processed.head())

print("\nFinal Dataset Shapes:")
print("Original: ", X.shape)
print("Train:    ", X_train_processed.shape)
print("Test:     ", X_test_processed.shape)


# ------------------------------------------------------------
# 9. VISUALIZE THE SCALED (NUMERIC) FEATURES
# ------------------------------------------------------------

X_train[numeric_columns].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distributions After Scaling (numeric columns)")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. CORRELATION HEATMAP
# ------------------------------------------------------------

# Shows numeric relationships between the engineered features.
# Values near +1 or -1 are strong correlations; near 0 is weak.

plt.figure(figsize=(12, 8))
sns.heatmap(
    X_train_processed.corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Correlation Heatmap of Processed Features")
plt.show()


# ------------------------------------------------------------
# 11. SMALL DEMO: DOES THIS DATA WORK FOR A MODEL?
# ------------------------------------------------------------

# A quick logistic regression to show the processed data
# really is model-ready (this is a demo, not the main focus).

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

predictions = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, predictions)

print("\nDemo Model - Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("====================================")
print("FEATURE ENGINEERING COMPLETED")
print("====================================")

print("Original shape:  ", X.shape)
print("Train shape:     ", X_train_processed.shape)
print("Test shape:      ", X_test_processed.shape)
print("Missing values left:",
      X_train_processed.isnull().sum().sum())

print("\nDataset is ready for Machine Learning!")

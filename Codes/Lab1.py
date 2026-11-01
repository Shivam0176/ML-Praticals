# ============================================================
# LAB 1: FEATURE ENGINEERING AND DATA PREPROCESSING
# Dataset: Titanic (OpenML) - needs internet
#
# Big picture: raw data is messy. We clean it step by step:
#   1. look at the data
#   2. fix missing values
#   3. convert text categories into numbers
#   4. bring all columns to a similar scale
# Only then is the data ready for a machine learning model.
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------

titanic = fetch_openml(name="titanic", version=1, as_frame=True)
df = titanic.frame.copy()

print("Dataset loaded. First 5 rows:")
print(df.head())
print("\nShape (rows, columns):", df.shape)


# ------------------------------------------------------------
# STEP 2: EXPLORE THE DATA
# ------------------------------------------------------------

# How many missing values does each column have?
print("\nMissing values per column:")
print(df.isnull().sum())

# Basic statistics (only for the columns we will keep)
print("\nStatistics of numeric columns:")
print(df[["age", "sibsp", "parch", "fare"]].describe())


# ------------------------------------------------------------
# STEP 3: CHOOSE OUR COLUMNS
# ------------------------------------------------------------

# Keep a few useful columns, drop the rest (names, tickets...).
numeric_columns = ["age", "sibsp", "parch", "fare"]
category_columns = ["pclass", "sex", "embarked"]

X = df[numeric_columns + category_columns].copy()
y = df["survived"].astype(int)          # 0 = did not survive, 1 = survived

# A heat map where yellow = missing value makes gaps easy to see.
plt.figure(figsize=(10, 5))
sns.heatmap(X.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values (yellow = missing)")
plt.show()


# ------------------------------------------------------------
# STEP 4: SPLIT INTO TRAIN AND TEST (BEFORE FIXING MISSING VALUES)
# ------------------------------------------------------------

# Why split first? The statistics used for cleaning (median,
# most frequent value) must come from the TRAINING data only.
# If the test data influenced them, our accuracy would be
# fake-optimistic. This is called "data leakage".

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)


# ------------------------------------------------------------
# STEP 5: FILL MISSING VALUES (IMPUTATION)
# ------------------------------------------------------------

# Numeric columns  -> fill with the training MEDIAN
# (median is not changed much by extreme values)
# Category columns -> fill with the training MODE (most common)

for column in numeric_columns:
    train_median = X_train[column].median()
    X_train[column] = X_train[column].fillna(train_median)
    X_test[column] = X_test[column].fillna(train_median)
    print("Filled", column, "with median:", train_median)

for column in category_columns:
    train_mode = X_train[column].mode()[0]
    X_train[column] = X_train[column].fillna(train_mode)
    X_test[column] = X_test[column].fillna(train_mode)
    print("Filled", column, "with most frequent:", train_mode)

print("\nMissing values after filling:",
      "train =", X_train.isnull().sum().sum(),
      ", test =", X_test.isnull().sum().sum())


# ------------------------------------------------------------
# STEP 6: TURN TEXT CATEGORIES INTO NUMBERS (ONE-HOT ENCODING)
# ------------------------------------------------------------

# Models can't read "male"/"female". get_dummies creates one
# new 0/1 column per category, e.g. sex_male (1, 0).
# The test set may miss a category that train has, so we
# reindex the test columns to match train, filling 0.

X_train = pd.get_dummies(X_train, dtype=float)
X_test = pd.get_dummies(X_test, dtype=float)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nColumns after encoding:")
print(list(X_train.columns))


# ------------------------------------------------------------
# STEP 7: SCALE ALL COLUMNS (STANDARDIZATION)
# ------------------------------------------------------------

# Each value is replaced by (value - mean) / std_dev.
# Result: every column has mean 0 and std 1, so "fare"
# (hundreds) no longer overshadows "parch" (0-5).
# We learn the mean/std from train, then re-use them on test.

scaler = StandardScaler()
X_train_processed = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns)
X_test_processed = pd.DataFrame(
    scaler.transform(X_test), columns=X_train.columns)

print("\nProcessed train data (all values now on the same scale):")
print(X_train_processed.head())


# ------------------------------------------------------------
# STEP 8: FINAL CHECK - DOES A MODEL LEARN FROM THIS DATA?
# ------------------------------------------------------------

# A quick logistic regression to prove the pipeline works.

model = LogisticRegression()
model.fit(X_train_processed, y_train)
predictions = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, predictions)
print("\nDemo model accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))


# ------------------------------------------------------------
# STEP 9: CORRELATION HEATMAP OF THE PROCESSED DATA
# ------------------------------------------------------------

plt.figure(figsize=(12, 8))
sns.heatmap(X_train_processed.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Processed Features")
plt.show()

print("\nPreprocessing complete - the data is now model-ready!")

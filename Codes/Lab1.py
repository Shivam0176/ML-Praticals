
# ============================================================
# LAB 1: FEATURE ENGINEERING AND DATA PREPROCESSING
# Dataset: Titanic Dataset (OpenML)
# ============================================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 2. Load Dataset

print("Loading Titanic Dataset...")

titanic = fetch_openml(
    name="titanic",
    version=1,
    as_frame=True
)

df = titanic.frame.copy()

print("\nDataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())


# 3. Dataset Exploration

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nStatistical Description:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())


# 4. Data Cleaning

# Select relevant features
features = [
    "age",
    "sibsp",
    "parch",
    "fare",
    "pclass",
    "sex",
    "embarked"
]

X = df[features].copy()
y = df["survived"].copy()

# Remove rows where target is missing
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Convert target into integer values
y = y.astype(int)

# Separate numerical and categorical columns
numerical_features = [
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "pclass",
    "sex",
    "embarked"
]


# 5. Visualize Missing Values

plt.figure(figsize=(10, 5))

sns.heatmap(
    X.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Values in Titanic Dataset")
plt.show()


# 6. Handling Missing Values

# Numerical columns: Median Imputation
# Categorical columns: Mode Imputation

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])


# 7. Encoding and Feature Scaling

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# 8. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# 9. Apply Feature Engineering

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the same preprocessing
X_test_processed = preprocessor.transform(X_test)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Convert processed data into DataFrames
X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# 10. Display Processed Dataset

print("\nProcessed Training Dataset:")
print(X_train_processed.head())

print("\nProcessed Dataset Shape:")
print(X_train_processed.shape)

print("\nMissing Values After Preprocessing:")
print(X_train_processed.isnull().sum().sum())


# 11. Visualize Scaled Numerical Features

scaled_features = [
    col for col in X_train_processed.columns
    if col.startswith("num__")
]

X_train_processed[scaled_features].hist(
    figsize=(12, 8),
    bins=20
)

plt.suptitle("Distribution of Scaled Numerical Features")
plt.tight_layout()
plt.show()


# 12. Correlation Heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(
    X_train_processed.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap of Processed Features")
plt.show()


# 13. Model Training (Optional Demonstration)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

print("\nModel Evaluation:")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 14. Final Output

print("\n====================================")
print("FEATURE ENGINEERING COMPLETED")
print("====================================")

print("Original Dataset Shape:", X.shape)

print(
    "Processed Training Dataset:",
    X_train_processed.shape
)

print(
    "Processed Testing Dataset:",
    X_test_processed.shape
)

print(
    "Remaining Missing Values:",
    X_train_processed.isnull().sum().sum()
)

print("\nDataset is ready for Machine Learning!")

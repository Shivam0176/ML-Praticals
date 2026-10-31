# Lab 1 Theory — Feature Engineering and Data Preprocessing

## 1. What is Feature Engineering and Preprocessing?

Real-world data is rarely clean. Before any model can learn from it, the data must be made complete, numeric and comparable in scale.

- **Data Preprocessing:** cleaning and transforming raw data into a usable form (handling missing values, encoding, scaling).
- **Feature Engineering:** creating or transforming features so that the signal available to the model is stronger (e.g., deriving `is_alone` from `sibsp` + `parch`, scaling numeric features).

Good preprocessing often matters more than the choice of model.

## 2. Types of Missing Values

| Type | Meaning | Example |
|------|---------|---------|
| **MCAR** (Missing Completely At Random) | No relation to any variable | Random sensor failure |
| **MAR** (Missing At Random) | Related to other observed variables | Younger passengers missing `age` records |
| **MNAR** (Missing Not At Random) | Related to the missing value itself | High earners not reporting wealth |

### Handling strategies
- **Drop rows/columns** — only when missingness is small or a column is mostly empty.
- **Imputation** — fill using a statistic:
  - *Mean* — numeric, no outliers
  - *Median* — numeric, skewed or with outliers (robust)
  - *Mode* — categorical (most frequent)
- **Flag + fill** — add an indicator column *and* impute, so the model knows the value was missing.

Titanic-specific: `age` → median per group; `embarked` → mode; `deck`/`cabin` → dropped or converted to a *has-value* flag.

## 3. Categorical Encoding

Machine-learning models work with numbers, so categories must be encoded:

- **Label Encoding** — each category → an integer (0, 1, 2…). Imposes a fake order; fine for ordinal data or tree models.
- **One-Hot Encoding** — each category becomes its own 0/1 column. Avoids fake ordering; standard for nominal data like `sex`, `embarked`. `OneHotEncoder(handle_unknown="ignore")` makes pipelines robust to unseen categories.
- **Drop_first / dropping one level** — avoids perfect collinearity between dummy columns (the dummy-variable trap).

## 4. Feature Scaling

Features on different scales (e.g., `fare` up to 500+, `pclass` 1–3) distort distance-based and regularized/gradient models.

- **Standardization (Z-score):**

  \( z = \dfrac{x - \mu}{\sigma} \)

  → mean 0, std 1. Used in this lab with `StandardScaler`.

- **Normalization (Min-Max):**

  \( x' = \dfrac{x - x_{min}}{x_{max} - x_{min}} \)

  → bounded range [0, 1]; sensitive to outliers.

**Note:** scaling is essential for logistic regression/KNN/SVM, optional for trees and forests.

## 5. Train–Test Split and Data Leakage

- The dataset is split (commonly 80/20) so the model is evaluated on unseen data.
- **Rule:** fit every transformer (imputer, encoder, scaler) **on the training set only**, then apply it to the test set. Fitting on full data leaks test statistics into training → over-optimistic scores (data leakage).

## 6. Pipelines and ColumnTransformer

- **`Pipeline`** — chains steps so preprocessing and modeling run as one object, preventing leakage and making the workflow reproducible.
- **`ColumnTransformer`** — applies different transformations to different column subsets: median imputation + scaling for numeric columns, constant-fill + one-hot for categorical columns.

## 7. Correlation

Pearson correlation measures linear association between two variables:

\( r = \dfrac{\sum (x-\bar x)(y-\bar y)}{\sqrt{\sum (x-\bar x)^2 \sum (y-\bar y)^2}} \), range −1 to +1.

A **heatmap** of the processed features reveals which variables relate to the target (`survived`) and which features are inter-correlated. Correlation ≠ causation; it also only captures *linear* relationships.

## 8. Why This Lab Matters

Every later lab reuses these ideas: scaling before regularized regression (Lab 5), encoding/scaling in classification (Lab 6), and standardization before clustering (Lab 8). A leak-free pipeline is the foundation of honest evaluation.

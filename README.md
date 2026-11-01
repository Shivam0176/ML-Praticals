# Machine Learning Lab — External Practicals

A collection of hands-on Python practicals covering the core topics of the Machine Learning lab: data preprocessing, regression, classification, clustering, and ensemble learning. Each lab is a self-contained script that loads a public dataset, walks through the full ML workflow step by step (exploration → preprocessing → modeling → evaluation → visualization), and prints/plots all key results.

## Requirements

- **Python 3.8+**
- **Libraries:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

- **Internet connection** — required for Lab 1, Lab 4 and Lab 5, whose datasets are downloaded at runtime (OpenML / UCI). All other labs use datasets bundled with scikit-learn.

## Lab Index

| Lab | Topic | Dataset | PDF | Code | Theory |
|-----|-------|---------|-----|------|--------|
| 1 | Feature Engineering & Data Preprocessing | [Titanic (OpenML)](https://www.openml.org/d/40945) | [PDF](./PDFs/Lab-1.pdf) | [Lab1.py](./Codes/Lab1.py) | [Theory](./Theory/Lab1_Theory.md) |
| 3 | Multiple Linear Regression | [California Housing (sklearn)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | [PDF](./PDFs/Lab-3.pdf) | [Lab3.py](./Codes/Lab3.py) | [Theory](./Theory/Lab3_Theory.md) |
| 4 | Polynomial Regression & Bias-Variance | [Auto MPG (OpenML)](https://www.openml.org/d/42203) | [PDF](./PDFs/Lab-4.pdf) | [Lab4.py](./Codes/Lab4.py) | [Theory](./Theory/Lab4_Theory.md) |
| 5 | Ridge, Lasso & Elastic Net | [Wine Quality — Red (UCI)](https://archive.ics.uci.edu/dataset/186/wine+quality) | [PDF](./PDFs/Lab-5.pdf) | [Lab5.py](./Codes/Lab5.py) | [Theory](./Theory/Lab5_Theory.md) |
| 6 | Logistic Regression | [Breast Cancer (sklearn)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset) | [PDF](./PDFs/Lab-6.pdf) | [Lab6.py](./Codes/Lab6.py) | [Theory](./Theory/Lab6_Theory.md) |
| 8 | K-Means Clustering | [Iris (sklearn)](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) | [PDF](./PDFs/Lab-8.pdf) | [Lab8.py](./Codes/Lab8.py) | [Theory](./Theory/Lab8_Theory.md) |
| 10 | Ensemble Learning (Random Forest vs Linear Regression) | [California Housing (sklearn)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | [PDF](./PDFs/Lab-10.pdf) | [Lab10.py](./Codes/Lab10.py) | [Theory](./Theory/Lab10_Theory.md) |

## Lab Descriptions

### Lab 1 — Feature Engineering and Data Preprocessing
Highlights the classic problems that arise when working with real, messy data. Using the Titanic passenger dataset, the lab explores missing values and flag columns, visualizes missingness, then builds a `ColumnTransformer` pipeline that imputes numeric gaps, one-hot encodes categorical variables, and standard-scales the features. The result is a clean, model-ready matrix (verified by a quick Logistic Regression demo), followed by a correlation heatmap of the processed features.

### Lab 3 — Multiple Linear Regression
Extends simple regression to many predictors using the California Housing dataset. After an EDA pass (distributions, scatter/correlation plots), the data is split into train/test sets, a `LinearRegression` model is trained, and performance is measured with MSE and R². The lab inspects the regression coefficients, plots actual vs predicted values, and checks residuals for the assumptions of linear regression.

### Lab 4 — Polynomial Regression & Model Complexity
Studies how model capacity affects generalization using Auto MPG. Polynomial models of degree 1 to 15 are trained via pipelines (`PolynomialFeatures` + scaling), and train/test errors are plotted to expose underfitting and overfitting. K-fold cross-validation selects the optimal degree, and a bootstrap experiment estimates bias and variance across degrees for a concrete look at the bias-variance trade-off.

### Lab 5 — Ridge, Lasso and Elastic Net Regression
Demonstrates regularization on the Red Wine Quality dataset — deliberately chosen because its features are highly correlated. The lab diagnoses multicollinearity with a correlation matrix, Variance Inflation Factor (VIF) and the condition number, then compares plain Linear, Ridge, Lasso and Elastic Net models across multiple alpha values. Side-by-side score plots, coefficient comparisons and coefficient paths show how each regularizer shrinks coefficients and handles collinear features.

### Lab 6 — Logistic Regression
A complete binary-classification walkthrough on the Breast Cancer Wisconsin dataset. A pipeline (StandardScaler + LogisticRegression) is trained, and a confusion matrix is built — first computed manually (TP/FP/TN/FN), then verified against scikit-learn — followed by accuracy, precision, recall and F1. The lab adds stratified 5-fold cross-validation and a decision-threshold analysis, plotting how precision and recall trade off as the threshold moves.

### Lab 8 — K-Means Clustering
Unsupervised learning on the Iris dataset. Features are standardized, then the number of clusters k is chosen two ways: the Elbow method (inertia) and the Silhouette method. The final K-Means model is fit with the optimal k and evaluated with silhouette and Davies-Bouldin scores, cluster-wise means, and a confusion-style comparison against the true species, with a PCA projection visualizing the clusters and their centroids in 2D.

### Lab 10 — Ensemble Learning
Compares a single linear model against an ensemble on the California Housing dataset. A baseline `LinearRegression` and a `RandomForestRegressor` are trained on the same split and evaluated with MSE, RMSE, MAE and R². Side-by-side bar charts and actual-vs-predicted scatter plots show the ensemble's advantage, and feature importances reveal which predictors the Random Forest relies on most.

## How to Run

From this folder:

```bash
python Codes/Lab1.py
```

Each script prints its full analysis in the console and opens matplotlib windows/plots as it progresses. Close each figure window to continue to the next step.

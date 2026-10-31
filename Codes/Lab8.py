
# ============================================================
# LAB 8: K-MEANS CLUSTERING
# Dataset: Iris
# ============================================================

# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA

# 2. Load Dataset

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("First 5 Rows:")
print(X.head())

print("\nDataset Shape:", X.shape)

# 3. Feature Scaling

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# 4. Elbow Method

wcss = []

K_values = range(2, 11)

for k in K_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# 5. Plot Elbow Curve

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_values),
    wcss,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.xticks(list(K_values))
plt.grid()
plt.show()

# 6. Silhouette Method

silhouette_scores = []

for k in K_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

# 7. Plot Silhouette Scores

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.grid()
plt.show()

# 8. Find Optimal K using Silhouette Score

optimal_k = list(K_values)[
    np.argmax(silhouette_scores)
]

print("\nOptimal Number of Clusters:", optimal_k)

# 9. Apply Final K-Means Model

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

X["Cluster"] = clusters

print("\nCluster Assignments:")
print(X.head(10))

# 10. Evaluate Clustering

sil_score = silhouette_score(X_scaled, clusters)

db_score = davies_bouldin_score(X_scaled, clusters)

print("\n========== CLUSTER EVALUATION ==========")

print("Silhouette Score:", round(sil_score, 4))

print("Davies-Bouldin Score:", round(db_score, 4))

print("WCSS:", round(kmeans.inertia_, 4))

# 11. Reduce Dimensions using PCA for Visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters,
    palette="Set1",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering Visualization")
plt.legend(title="Cluster")
plt.show()

# 12. Visualize Cluster Centroids

centroids_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    marker="X",
    color="red",
    s=200,
    label="Centroids"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters and Centroids")
plt.legend()
plt.show()

print("\nK-Means Clustering Completed!")

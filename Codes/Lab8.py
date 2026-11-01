# ============================================================
# LAB 8: K-MEANS CLUSTERING
# Dataset: Iris
#
# Goal: group the flowers WITHOUT using their labels,
# and decide how many groups (k) make sense using
# the Elbow method and the Silhouette method.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

# The true species are kept aside for interpretation only
# (K-means never gets to see them).
true_species = pd.Series(iris.target).map(
    {0: "setosa", 1: "versicolor", 2: "virginica"}
)

print("First 5 Rows:")
print(X.head())

print("\nDataset Shape:", X.shape)


# ------------------------------------------------------------
# 2. SCALE THE FEATURES
# ------------------------------------------------------------

# K-means measures distances, so a wide-ranging feature would
# dominate the clustering. Scaling gives every feature the
# same influence (mean 0, std 1).

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# ------------------------------------------------------------
# 3. ELBOW METHOD - HOW MANY CLUSTERS SHOULD WE PICK?
# ------------------------------------------------------------

# For candidate k values, fit K-means and record its WCSS
# ("within-cluster sum of squares" = how tight the clusters are).
# WCSS always decreases when k rises; the "elbow" is where the
# decrease suddenly becomes small - adding clusters stops helping.

candidate_k_values = range(2, 11)

wcss_values = []

for k in candidate_k_values:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    wcss_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(candidate_k_values), wcss_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.xticks(list(candidate_k_values))
plt.grid()
plt.show()


# ------------------------------------------------------------
# 4. SILHOUETTE METHOD - ANOTHER WAY TO PICK K
# ------------------------------------------------------------

# For each point: (average distance to nearest OTHER cluster
#                  - average distance inside its OWN cluster)
#              divided by the larger distance
# Score is between -1 and +1; higher = clusters well separated.

silhouette_values = []

for k in candidate_k_values:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_values.append(score)

    print(f"K = {k}: silhouette score = {round(score, 4)}")

plt.figure(figsize=(8, 5))
plt.plot(list(candidate_k_values), silhouette_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Average Silhouette Score")
plt.title("Silhouette Method")
plt.grid()
plt.show()

best_k = list(candidate_k_values)[np.argmax(silhouette_values)]

print("\nBest number of clusters by silhouette:", best_k)
print("(If 2 and 3 are close, 3 often matches the real species.)")


# ------------------------------------------------------------
# 5. FIT THE FINAL K-MEANS MODEL
# ------------------------------------------------------------

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

cluster_labels = final_kmeans.fit_predict(X_scaled)

X_with_clusters = X.copy()
X_with_clusters["Cluster"] = cluster_labels

print("\nFirst 10 Rows with Cluster Numbers:")
print(X_with_clusters.head(10))


# ------------------------------------------------------------
# 6. EVALUATE THE CLUSTERING
# ------------------------------------------------------------

# Silhouette (higher = better) and Davies-Bouldin (lower = better)
# need no ground truth, so they work for all clustering problems.

silhouette_final = silhouette_score(X_scaled, cluster_labels)
davies_final = davies_bouldin_score(X_scaled, cluster_labels)

print("\n========== CLUSTER EVALUATION ==========")
print("Silhouette Score:    ", round(silhouette_final, 4))
print("Davies-Bouldin Score:", round(davies_final, 4))
print("WCSS (tightness):    ", round(final_kmeans.inertia_, 4))


# ------------------------------------------------------------
# 7. SHOW THE FOUND CLUSTERS AGAINST REAL SPECIES
# ------------------------------------------------------------

print("\nCluster sizes:")
print(X_with_clusters["Cluster"].value_counts().sort_index())

print("\nCluster vs true species (K-means never saw these):")
comparison_table = pd.crosstab(
    X_with_clusters["Cluster"],
    true_species.values
)
print(comparison_table)


# ------------------------------------------------------------
# 8. REDUCE TO 2 DIMENSIONS WITH PCA FOR VISUALIZATION
# ------------------------------------------------------------

# We clustered in 4-D. To draw a picture, PCA finds the 2 best
# viewing directions (the directions with the most variance)
# and every point is projected onto them.

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- Plot: points colored by CLUSTER ---
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=50
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-Means Clustering (K = {best_k})")
plt.show()


# ------------------------------------------------------------
# 9. SHOW THE FOUND CLUSTERS AGAINST REAL SPECIES
# ------------------------------------------------------------

# --- Plot: centroids marked with red X ---
# Cluster centers get pushed through the same PCA projection
# so they can be drawn on top of the points.

centroids_pca = pca.transform(final_kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
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


# --- Plot: the same points colored by TRUE species ---

plt.figure(figsize=(8, 6))
for species_name in ["setosa", "versicolor", "virginica"]:
    mask = (true_species.values == species_name)
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=50,
        label=species_name
    )
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("True Iris Species (for comparison)")
plt.legend()
plt.show()


# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n====================================")
print("K-MEANS CLUSTERING COMPLETED")
print("====================================")

print("Data points:          ", len(X))
print("Chosen K:             ", best_k)
print("Silhouette Score:     ", round(silhouette_final, 4))
print("Davies-Bouldin Score: ", round(davies_final, 4))
print("WCSS:                 ", round(final_kmeans.inertia_, 4))
print("\nK-means found structure in unlabeled data - compare the")
print("cluster/species table above to see how well it matches!")

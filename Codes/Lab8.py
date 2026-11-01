# ============================================================
# LAB 8: K-MEANS CLUSTERING (UNSUPERVISED LEARNING)
# Dataset: Iris (bundled with sklearn, no internet)
#
# Big picture: no labels are given. K-means groups similar
# flowers automatically:
#   1. place k centers randomly
#   2. assign every point to its nearest center
#   3. move each center to the average of its points
#   4. repeat until nothing moves
# We test which k works best with Elbow + Silhouette methods.
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
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Species kept aside only for the LAST comparison plot.
# The clustering algorithm itself never sees them.
true_species = pd.Series(iris.target).map(
    {0: "setosa", 1: "versicolor", 2: "virginica"})

print("Data shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())


# ------------------------------------------------------------
# STEP 2: SCALE THE FEATURES
# ------------------------------------------------------------

# K-means uses DISTANCE between points. Without scaling, a
# wide-ranging feature would dominate every distance check.
# Standardization gives all features the same voice.

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# ------------------------------------------------------------
# STEP 3: ELBOW METHOD FOR CHOOSING K
# ------------------------------------------------------------

# For k = 2..10 we fit K-means and record WCSS (sum of
# squared distances of points to their center = "tightness").
# More clusters always lower WCSS - the ELBOW point is where
# tightening suddenly stops being worth extra clusters.

candidate_ks = list(range(2, 11))
wcss_values = []

for k in candidate_ks:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss_values.append(kmeans.inertia_)   # the WCSS value

plt.figure(figsize=(8, 5))
plt.plot(candidate_ks, wcss_values, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (tightness)")
plt.title("Elbow Method - bend decides the sensible K")
plt.xticks(candidate_ks)
plt.grid()
plt.show()


# ------------------------------------------------------------
# STEP 4: SILHOUETTE METHOD - A DIFFERENT WAY TO CHOOSE K
# ------------------------------------------------------------

# For each point compare:
#   a = average distance to its OWN cluster (cohesion)
#   b = average distance to the NEAREST other cluster (separation)
# silhouette = (b - a) / max(a, b), between -1 and 1.
# Pick the k with the highest average silhouette.

silhouette_values = []

for k in candidate_ks:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_values.append(silhouette_score(X_scaled, labels))

    print(f"K = {k}: silhouette = {round(silhouette_values[-1], 4)}")

plt.figure(figsize=(8, 5))
plt.plot(candidate_ks, silhouette_values, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Average silhouette score")
plt.title("Silhouette Method - higher is better")
plt.grid()
plt.show()

best_k = candidate_ks[int(np.argmax(silhouette_values))]
print("\nBest K by silhouette:", best_k)


# ------------------------------------------------------------
# STEP 5: FIT THE FINAL MODEL WITH THE BEST K
# ------------------------------------------------------------

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

print("\nCluster sizes:")
print(X.assign(Cluster=cluster_labels)["Cluster"].value_counts())


# ------------------------------------------------------------
# STEP 6: SCORE THE CLUSTERING QUALITY (NO LABELS NEEDED)
# ------------------------------------------------------------

# Silhouette  -> higher is better (clusters well separated)
# Davies-Bouldin -> lower is better (compact, far clusters)
# WCSS        -> lower is better (tighter clusters)

final_silhouette = silhouette_score(X_scaled, cluster_labels)
final_davies = davies_bouldin_score(X_scaled, cluster_labels)

print("\n=== CLUSTER EVALUATION ===")
print("Silhouette score:     ", round(final_silhouette, 4))
print("Davies-Bouldin score: ", round(final_davies, 4))
print("WCSS:                 ", round(final_kmeans.inertia_, 4))


# ------------------------------------------------------------
# STEP 7: DOWNSCALE 4 DIMENSIONS TO 2 FOR PICTURES (PCA)
# ------------------------------------------------------------

# We cannot draw 4-D. PCA finds the two directions along
# which the data is most spread out, then we look only
# along those two - a 2-D "shadow" of the data.

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Plot 1: points colored by the cluster K-means assigned.
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c=cluster_labels, cmap="viridis", s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-Means Clusters (K = {best_k})")
plt.show()


# ------------------------------------------------------------
# STEP 8: SHOW CLUSTER CENTERS ON THE SAME PICTURE
# ------------------------------------------------------------

# The centers (red X) are pushed through the same PCA
# projectors so they appear on top of the points.

centers_2d = pca.transform(final_kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c=cluster_labels, cmap="viridis", s=50)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
            marker="X", color="red", s=200, label="Centers")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Clusters with Their Centers")
plt.legend()
plt.show()


# ------------------------------------------------------------
# STEP 9: COMPARE WHAT K-MEANS FOUND WITH THE TRUE SPECIES
# ------------------------------------------------------------

# Cluster vs true species table: K-means could and did group
# setosa perfectly; versicolor and virginica partially merge.

print("\n=== FOUND CLUSTERS vs TRUE SPECIES ===")
print(pd.crosstab(cluster_labels, true_species.values))

plt.figure(figsize=(8, 6))
for species in ["setosa", "versicolor", "virginica"]:
    mask = (true_species.values == species)
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                s=50, label=species)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("True Species (K-means never saw these)")
plt.legend()
plt.show()

print("\nK-means discovered real structure in unlabeled data!")
